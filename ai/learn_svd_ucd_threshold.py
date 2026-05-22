from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.linalg import svd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
CLASSICAL_DIR = ROOT / "classical"
for extra_path in (ROOT, CLASSICAL_DIR):
    text = str(extra_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from multiuser_simulation_environment import MultiUserSimulationEnvironment
from sic_sample_average import build_multiuser_sample_average


@dataclass(frozen=True)
class ThresholdSample:
    channel_index: int
    config_id: int
    tokens: np.ndarray
    token_mask: np.ndarray
    config_features: np.ndarray
    threshold_db: float
    crossing_type: str
    svd_rates: np.ndarray
    ucd_rates: np.ndarray
    delta_rates: np.ndarray


@dataclass(frozen=True)
class TrainArtifacts:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    config_mean: np.ndarray
    config_std: np.ndarray
    target_mean: float
    target_std: float


@dataclass(frozen=True)
class SystemConfig:
    num_users: int
    num_tx_antennas: int
    num_rx_antennas: int
    num_rf_chains: int
    num_streams_per_user: int


@dataclass(frozen=True)
class ConfigSpec:
    config_id: int
    system_config: SystemConfig
    num_channels: int


class ResidualSelfAttentionBlock(nn.Module):
    """A lightweight self-attention residual block."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.norm(hidden + attended)


class SpectralSetTransformerThresholdRegressor(nn.Module):
    """Config-conditioned self-attention encoder with masked pooling readout."""

    def __init__(
        self,
        token_dim: int,
        config_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be positive.")

        self.num_layers = int(num_layers)

        self.token_projection = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.config_projection = nn.Sequential(
            nn.Linear(config_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.encoder_blocks = nn.ModuleList(
            ResidualSelfAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(self.num_layers)
        )
        self.pool_score = nn.Linear(d_model, 1)
        self.shared_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.threshold_head = nn.Linear(d_model, 1)
        self.sign_condition_projection = nn.Sequential(
            nn.Linear(d_model + 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        config_features: torch.Tensor,
        sign_snr_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape (batch, max_tokens, token_dim).")
        if token_mask.ndim != 2 or token_mask.shape[:2] != tokens.shape[:2]:
            raise ValueError("token_mask must have shape (batch, max_tokens).")
        if config_features.ndim != 2 or config_features.shape[0] != tokens.shape[0]:
            raise ValueError("config_features must have shape (batch, config_dim).")

        valid_mask = token_mask.to(dtype=torch.bool)
        key_padding_mask = ~valid_mask
        config_hidden = self.config_projection(config_features)
        hidden = self.token_projection(tokens) + config_hidden.unsqueeze(1)
        hidden = hidden * valid_mask.unsqueeze(-1).to(hidden.dtype)
        for block in self.encoder_blocks:
            hidden = block(hidden, key_padding_mask=key_padding_mask)
            hidden = hidden * valid_mask.unsqueeze(-1).to(hidden.dtype)

        pool_scores = self.pool_score(hidden).squeeze(-1)
        pool_scores = pool_scores.masked_fill(~valid_mask, float("-inf"))
        pool_weight = torch.softmax(pool_scores, dim=-1)
        global_state = torch.sum(hidden * pool_weight.unsqueeze(-1), dim=1)
        shared_state = self.shared_head(global_state)
        threshold_pred = self.threshold_head(shared_state).squeeze(-1)

        sign_logits = None
        if sign_snr_values is not None:
            if sign_snr_values.ndim != 1:
                raise ValueError("sign_snr_values must have shape (num_sign_points,).")
            snr_values = sign_snr_values.to(device=tokens.device, dtype=tokens.dtype)
            scale = torch.clamp(torch.max(torch.abs(snr_values)), min=1.0)
            snr_norm = snr_values / scale
            sign_features = torch.stack(
                (
                    snr_norm,
                    snr_norm**2,
                    torch.ones_like(snr_norm),
                ),
                dim=-1,
            )
            sign_features = sign_features.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            sign_state = shared_state.unsqueeze(1).expand(-1, sign_features.shape[1], -1)
            sign_input = torch.cat((sign_state, sign_features), dim=-1)
            sign_logits = self.sign_condition_projection(sign_input).squeeze(-1)

        return threshold_pred, sign_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate BD-spectrum labels and train a lightweight attention regressor "
            "for the SVD/UCD switching SNR threshold."
        )
    )
    parser.add_argument("--num-users", type=int, default=2)
    parser.add_argument("--num-tx-antennas", type=int, default=16)
    parser.add_argument("--num-rx-antennas", type=int, default=4)
    parser.add_argument("--num-rf-chains", type=int, default=8)
    parser.add_argument("--num-streams-per-user", type=int, default=4)
    parser.add_argument("--digital-power-constraint", type=float, default=None)
    parser.add_argument("--bits-per-symbol", type=int, default=6)
    parser.add_argument("--channel-type", type=str, default="cdl-a")
    parser.add_argument(
        "--config-pool",
        type=str,
        default=None,
        help=(
            "Optional semicolon-separated config pool. "
            "Each item uses 'K,Nt,Nr,Nrf,d', for example "
            "'2,16,4,8,4;4,16,4,8,2'."
        ),
    )
    parser.add_argument(
        "--config-pool-file",
        type=str,
        default=None,
        help="Optional JSON file containing a list of system config objects.",
    )
    parser.add_argument("--snr-start-db", type=float, default=0.0)
    parser.add_argument("--snr-stop-db", type=float, default=30.0)
    parser.add_argument("--snr-step-db", type=float, default=5.0)
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument(
        "--channels-per-config",
        type=int,
        default=None,
        help="Optional override for the number of channels generated per config.",
    )
    parser.add_argument("--train-num-samples", type=int, default=64)
    parser.add_argument("--train-num-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--labeling", choices=("gray_standard", "nr_like"), default="nr_like")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent / "results" / "svd_ucd_threshold"))
    parser.add_argument("--output-prefix", type=str, default="spectral_threshold")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--generator-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--huber-delta-db", type=float, default=1.0)
    parser.add_argument("--sign-loss-weight", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--coarse-snr-step-db", type=float, default=4.0)
    parser.add_argument("--skip-rate-curves", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-new-channels", type=int, default=None)
    parser.add_argument("--eval-model", type=str, default=None)
    return parser.parse_args()


def build_snr_values(start_db: float, stop_db: float, step_db: float) -> np.ndarray:
    if step_db <= 0.0:
        raise ValueError(f"snr_step_db must be positive, got {step_db}.")
    return np.arange(start_db, stop_db + 0.5 * step_db, step_db, dtype=float)


def build_coarse_snr_values(
    start_db: float,
    stop_db: float,
    coarse_step_db: float,
    fine_step_db: float,
) -> np.ndarray:
    coarse_step_db = max(float(coarse_step_db), float(fine_step_db))
    values = np.arange(start_db, stop_db + 0.5 * coarse_step_db, coarse_step_db, dtype=float)
    if values[-1] < stop_db - 1e-9:
        values = np.append(values, float(stop_db))
    if values[0] > start_db + 1e-9:
        values = np.insert(values, 0, float(start_db))
    return np.asarray(np.unique(np.round(values, decimals=10)), dtype=float)


def build_default_sign_point_indices(snr_values_db: np.ndarray) -> np.ndarray:
    return np.unique(
        np.clip(
            np.round(np.linspace(0, len(snr_values_db) - 1, num=min(7, len(snr_values_db)))).astype(int),
            0,
            len(snr_values_db) - 1,
        )
    )


def extract_subgrid_rates(
    source_snr_values_db: np.ndarray,
    source_rates: np.ndarray,
    target_snr_values_db: np.ndarray,
) -> np.ndarray:
    rate_map = {
        round(float(snr_db), 10): float(rate)
        for snr_db, rate in zip(source_snr_values_db, source_rates)
    }
    return np.asarray([rate_map[round(float(snr_db), 10)] for snr_db in target_snr_values_db], dtype=float)


def validate_system_config(config: SystemConfig) -> None:
    if config.num_users <= 0 or config.num_streams_per_user <= 0:
        raise ValueError(f"Invalid users/streams config: {config}.")
    if config.num_tx_antennas < config.num_rf_chains:
        raise ValueError(
            "num_tx_antennas must satisfy Nt >= Nrf, got "
            f"{config.num_tx_antennas} < {config.num_rf_chains}."
        )
    if config.num_rx_antennas < config.num_streams_per_user:
        raise ValueError(
            "num_rx_antennas must satisfy Nr >= num_streams_per_user, got "
            f"{config.num_rx_antennas} < {config.num_streams_per_user}."
        )
    total_streams = config.num_users * config.num_streams_per_user
    if config.num_rf_chains < total_streams:
        raise ValueError(
            "num_rf_chains must satisfy Nrf >= total_streams, got "
            f"{config.num_rf_chains} < {total_streams}."
        )
    # A sufficient BD feasibility check: the stacked interference matrix for a
    # user has at most min((K-1) * Nr, Nrf) rank, so the RF-domain nullity is
    # lower-bounded by Nrf - min((K-1) * Nr, Nrf). We require that lower bound
    # to be at least the desired per-user stream count.
    interference_rank_upper_bound = min(
        (config.num_users - 1) * config.num_rx_antennas,
        config.num_rf_chains,
    )
    bd_nullity_lower_bound = max(config.num_rf_chains - interference_rank_upper_bound, 0)
    if bd_nullity_lower_bound < config.num_streams_per_user:
        raise ValueError(
            "Per-user digital BD requires enough RF-domain null-space. "
            f"For K={config.num_users}, d={config.num_streams_per_user}, "
            f"Nr={config.num_rx_antennas}, Nrf={config.num_rf_chains}, "
            f"the nullity lower bound is only {bd_nullity_lower_bound}."
        )
    if config.num_rf_chains % config.num_users != 0:
        raise ValueError(
            "Dedicated per-user RF requires Nrf divisible by K, got "
            f"{config.num_rf_chains} and {config.num_users}."
        )


def _resolve_config_channel_count(
    item: dict | None,
    args: argparse.Namespace,
    num_configs: int,
    config_index: int,
) -> int:
    if args.channels_per_config is not None:
        count = int(args.channels_per_config)
    elif item is not None and "num_channels" in item:
        count = int(item["num_channels"])
    elif num_configs == 1:
        count = int(args.num_channels)
    else:
        base = int(args.num_channels) // num_configs
        remainder = int(args.num_channels) % num_configs
        count = base + (1 if config_index < remainder else 0)
    if count <= 0:
        raise ValueError("Each config must generate at least one channel.")
    return count


def parse_config_pool(args: argparse.Namespace) -> list[ConfigSpec]:
    if args.config_pool_file is not None:
        config_path = Path(args.config_pool_file)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("config_pool_file must contain a JSON list.")
        configs = []
        for config_index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError("Each config in config_pool_file must be an object.")
            config = SystemConfig(
                num_users=int(item["num_users"]),
                num_tx_antennas=int(item["num_tx_antennas"]),
                num_rx_antennas=int(item["num_rx_antennas"]),
                num_rf_chains=int(item["num_rf_chains"]),
                num_streams_per_user=int(item["num_streams_per_user"]),
            )
            validate_system_config(config)
            configs.append(
                ConfigSpec(
                    config_id=config_index,
                    system_config=config,
                    num_channels=_resolve_config_channel_count(
                        item=item,
                        args=args,
                        num_configs=len(payload),
                        config_index=config_index,
                    ),
                )
            )
        if not configs:
            raise ValueError("config_pool_file must contain at least one valid config.")
        return configs

    if args.config_pool is None:
        config = SystemConfig(
            num_users=int(args.num_users),
            num_tx_antennas=int(args.num_tx_antennas),
            num_rx_antennas=int(args.num_rx_antennas),
            num_rf_chains=int(args.num_rf_chains),
            num_streams_per_user=int(args.num_streams_per_user),
        )
        validate_system_config(config)
        return [
            ConfigSpec(
                config_id=0,
                system_config=config,
                num_channels=int(args.channels_per_config or args.num_channels),
            )
        ]

    configs = []
    items = [item.strip() for item in str(args.config_pool).split(";") if item.strip()]
    for config_index, item in enumerate(items):
        text = item.strip()
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 5:
            raise ValueError(
                "Each config-pool item must use 'K,Nt,Nr,Nrf,d', got "
                f"{text!r}."
            )
        config = SystemConfig(
            num_users=int(parts[0]),
            num_tx_antennas=int(parts[1]),
            num_rx_antennas=int(parts[2]),
            num_rf_chains=int(parts[3]),
            num_streams_per_user=int(parts[4]),
        )
        validate_system_config(config)
        configs.append(
            ConfigSpec(
                config_id=config_index,
                system_config=config,
                num_channels=_resolve_config_channel_count(
                    item=None,
                    args=args,
                    num_configs=len(items),
                    config_index=config_index,
                ),
            )
        )
    if not configs:
        raise ValueError("config_pool must contain at least one valid config.")
    return configs


def build_env_from_config(
    config: SystemConfig,
    args: argparse.Namespace,
) -> MultiUserSimulationEnvironment:
    return MultiUserSimulationEnvironment(
        num_users=int(config.num_users),
        num_tx_antennas=int(config.num_tx_antennas),
        num_rx_antennas=int(config.num_rx_antennas),
        num_rf_chains=int(config.num_rf_chains),
        num_streams_per_user=int(config.num_streams_per_user),
        channel_type=str(args.channel_type),
        digital_power_constraint=args.digital_power_constraint,
        ucd_waterfill=False,
    )


def config_from_features(config_features: np.ndarray) -> SystemConfig:
    config_features = np.asarray(config_features, dtype=float)
    return SystemConfig(
        num_users=int(round(float(config_features[0]))),
        num_tx_antennas=int(round(float(config_features[1]))),
        num_rx_antennas=int(round(float(config_features[2]))),
        num_rf_chains=int(round(float(config_features[3]))),
        num_streams_per_user=int(round(float(config_features[4]))),
    )


def extract_bd_spectral_tokens(
    env: MultiUserSimulationEnvironment,
    user_channels: np.ndarray,
    f_rf: np.ndarray,
) -> np.ndarray:
    """Return per-stream tokens with spectral and normalized position features."""

    effective_channels = env.build_effective_channels(user_channels, f_rf)
    tokens = []
    eps = 1e-12
    for user_idx in range(env.num_users):
        bd_basis = env.build_bd_digital_basis(effective_channels, user_idx)
        reduced_channel = effective_channels[user_idx] @ bd_basis
        singular_values = svd(reduced_channel, full_matrices=False, compute_uv=False)
        singular_values = singular_values[: env.num_streams_per_user]
        singular_values = np.maximum(np.asarray(singular_values, dtype=float), eps)
        log_sigma = np.log(singular_values)
        centered_log_sigma = log_sigma - float(np.mean(log_sigma))
        power_fraction = singular_values**2 / max(float(np.sum(singular_values**2)), eps)
        for stream_idx in range(env.num_streams_per_user):
            user_position = (float(user_idx) + 0.5) / max(float(env.num_users), 1.0)
            stream_position = (float(stream_idx) + 0.5) / max(float(env.num_streams_per_user), 1.0)
            tokens.append(
                [
                    float(log_sigma[stream_idx]),
                    float(centered_log_sigma[stream_idx]),
                    float(power_fraction[stream_idx]),
                    float(user_position),
                    float(stream_position),
                ]
            )
    return np.asarray(tokens, dtype=np.float32)


def build_config_features(
    env: MultiUserSimulationEnvironment,
    args: argparse.Namespace,
) -> np.ndarray:
    return np.asarray(
        [
            float(env.num_users),
            float(env.num_tx_antennas),
            float(env.num_rx_antennas),
            float(env.num_rf_chains),
            float(env.num_streams_per_user),
            float(env.total_streams),
            float(args.bits_per_symbol),
            float(env.num_rf_chains) / max(float(env.num_tx_antennas), 1.0),
            float(env.total_streams) / max(float(env.num_rf_chains), 1.0),
            float(env.num_streams_per_user) / max(float(env.num_rx_antennas), 1.0),
        ],
        dtype=np.float32,
    )


def interpolate_threshold(
    snr_values_db: np.ndarray,
    delta_rates: np.ndarray,
) -> tuple[float, str]:
    snr_values_db = np.asarray(snr_values_db, dtype=float)
    delta_rates = np.asarray(delta_rates, dtype=float)
    eps = 1e-10
    if len(snr_values_db) > 1:
        guard_step = float(np.median(np.diff(snr_values_db)))
    else:
        guard_step = 1.0

    if np.all(delta_rates >= -eps):
        return float(snr_values_db[0] - guard_step), "always_ucd"
    if np.all(delta_rates <= eps):
        return float(snr_values_db[-1] + guard_step), "always_svd"

    crossing_indices = []
    first_neg_to_pos_idx = None
    first_exact_idx = None
    for idx in range(len(snr_values_db) - 1):
        left = float(delta_rates[idx])
        right = float(delta_rates[idx + 1])
        if abs(left) <= eps and first_exact_idx is None:
            first_exact_idx = idx
        if left * right < 0.0:
            crossing_indices.append(idx)
        if left <= 0.0 <= right and first_neg_to_pos_idx is None:
            first_neg_to_pos_idx = idx

    # The threshold is defined by the first SNR where UCD starts to beat SVD.
    # Later exact-grid ties at high SNR should never override this earliest
    # negative-to-positive crossing.
    if first_neg_to_pos_idx is not None:
        idx = first_neg_to_pos_idx
        crossing_type = "single_cross" if len(crossing_indices) == 1 else "multi_cross_first"
        if abs(float(delta_rates[idx])) <= eps:
            return float(snr_values_db[idx]), "exact_grid"
    elif first_exact_idx is not None:
        return float(snr_values_db[first_exact_idx]), "exact_grid"
    elif crossing_indices:
        idx = crossing_indices[0]
        crossing_type = "reverse_cross"
    else:
        closest_idx = int(np.argmin(np.abs(delta_rates)))
        return float(snr_values_db[closest_idx]), "closest_grid"

    left_snr = float(snr_values_db[idx])
    right_snr = float(snr_values_db[idx + 1])
    left_delta = float(delta_rates[idx])
    right_delta = float(delta_rates[idx + 1])
    denominator = right_delta - left_delta
    if abs(denominator) <= eps:
        return float(0.5 * (left_snr + right_snr)), crossing_type
    threshold = left_snr + (0.0 - left_delta) * (right_snr - left_snr) / denominator
    return float(np.clip(threshold, snr_values_db[0], snr_values_db[-1])), crossing_type


def evaluate_svd_ucd_rates_on_grid(
    env: MultiUserSimulationEnvironment,
    true_channels: np.ndarray,
    f_rf: np.ndarray,
    snr_values_db: np.ndarray,
    sample_average,
    bits_per_symbol: int,
    labeling: str,
) -> tuple[np.ndarray, np.ndarray]:
    svd_rates = []
    ucd_rates = []
    for snr_db in snr_values_db:
        snr_linear = 10 ** (float(snr_db) / 10.0)
        snr_per_stream = snr_linear / env.total_streams
        svd_chain = env.build_structured_digital_chain(
            user_channels=true_channels,
            f_rf=f_rf,
            snr_per_stream=snr_per_stream,
            strategy="svd",
        )
        ucd_chain = env.build_structured_digital_chain(
            user_channels=true_channels,
            f_rf=f_rf,
            snr_per_stream=snr_per_stream,
            strategy="ucd",
        )
        svd_eval = env.evaluate_precoder_current_receiver_average_fixed_chain(
            user_channels=true_channels,
            f_rf=f_rf,
            f_bb=svd_chain.f_bb,
            r_chains=svd_chain.r_chains,
            q_chains=svd_chain.q_chains,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=int(bits_per_symbol),
            sample_average=sample_average,
            labeling=str(labeling),
        )
        ucd_eval = env.evaluate_ucd_precoder_current_receiver_average_b_chain(
            user_channels=true_channels,
            f_rf=f_rf,
            f_bb=ucd_chain.f_bb,
            q_chains=ucd_chain.q_chains,
            r_chains=ucd_chain.r_chains,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=int(bits_per_symbol),
            sample_average=sample_average,
            labeling=str(labeling),
        )
        svd_rates.append(float(svd_eval.sum_rate))
        ucd_rates.append(float(ucd_eval.sum_rate))
    return np.asarray(svd_rates, dtype=float), np.asarray(ucd_rates, dtype=float)


def locate_refinement_interval(
    coarse_snr_values_db: np.ndarray,
    coarse_delta_rates: np.ndarray,
    fine_start_db: float,
    fine_stop_db: float,
    fine_step_db: float,
) -> np.ndarray:
    eps = 1e-10
    if np.all(coarse_delta_rates >= -eps) or np.all(coarse_delta_rates <= eps):
        return build_snr_values(fine_start_db, fine_stop_db, fine_step_db)

    first_neg_to_pos_idx = None
    first_cross_idx = None
    for idx in range(len(coarse_snr_values_db) - 1):
        left = float(coarse_delta_rates[idx])
        right = float(coarse_delta_rates[idx + 1])
        if left * right < 0.0 and first_cross_idx is None:
            first_cross_idx = idx
        if left <= 0.0 <= right and first_neg_to_pos_idx is None:
            first_neg_to_pos_idx = idx

    idx = first_neg_to_pos_idx if first_neg_to_pos_idx is not None else first_cross_idx
    if idx is None:
        return build_snr_values(fine_start_db, fine_stop_db, fine_step_db)

    left_snr = max(float(coarse_snr_values_db[idx]) - float(fine_step_db), float(fine_start_db))
    right_snr = min(float(coarse_snr_values_db[idx + 1]) + float(fine_step_db), float(fine_stop_db))
    return build_snr_values(left_snr, right_snr, fine_step_db)


def generate_one_threshold_sample(
    env: MultiUserSimulationEnvironment,
    snr_values_db: np.ndarray,
    args: argparse.Namespace,
    channel_idx: int,
    config_id: int,
) -> ThresholdSample:
    # Independent per-channel seeding makes long label generation resumable.
    np.random.seed(int(args.seed) + 1_000_003 * int(channel_idx))
    true_channels = env.generate_user_channels()
    f_rf = env.build_analog_precoder(true_channels)
    tokens = extract_bd_spectral_tokens(env=env, user_channels=true_channels, f_rf=f_rf)
    token_mask = np.ones(tokens.shape[0], dtype=np.float32)
    config_features = build_config_features(env=env, args=args)
    sample_average = build_multiuser_sample_average(
        env=env,
        bits_per_symbol=int(args.bits_per_symbol),
        num_samples=int(args.train_num_samples),
        num_repeats=int(args.train_num_repeats),
        base_seed=int(args.seed) + 1000 * int(channel_idx),
        labeling=str(args.labeling),
    )

    fine_step_db = float(args.snr_step_db)
    coarse_snr_values_db = build_coarse_snr_values(
        float(snr_values_db[0]),
        float(snr_values_db[-1]),
        float(args.coarse_snr_step_db),
        fine_step_db,
    )
    if coarse_snr_values_db.shape == snr_values_db.shape and np.allclose(coarse_snr_values_db, snr_values_db):
        eval_snr_values_db = snr_values_db
    else:
        coarse_svd_array, coarse_ucd_array = evaluate_svd_ucd_rates_on_grid(
            env=env,
            true_channels=true_channels,
            f_rf=f_rf,
            snr_values_db=coarse_snr_values_db,
            sample_average=sample_average,
            bits_per_symbol=int(args.bits_per_symbol),
            labeling=str(args.labeling),
        )
        eval_snr_values_db = locate_refinement_interval(
            coarse_snr_values_db=coarse_snr_values_db,
            coarse_delta_rates=coarse_ucd_array - coarse_svd_array,
            fine_start_db=float(snr_values_db[0]),
            fine_stop_db=float(snr_values_db[-1]),
            fine_step_db=fine_step_db,
        )

    refined_svd_array, refined_ucd_array = evaluate_svd_ucd_rates_on_grid(
        env=env,
        true_channels=true_channels,
        f_rf=f_rf,
        snr_values_db=eval_snr_values_db,
        sample_average=sample_average,
        bits_per_symbol=int(args.bits_per_symbol),
        labeling=str(args.labeling),
    )
    refined_delta_array = refined_ucd_array - refined_svd_array
    threshold_db, crossing_type = interpolate_threshold(eval_snr_values_db, refined_delta_array)

    if bool(args.skip_rate_curves):
        sign_point_indices = build_default_sign_point_indices(snr_values_db)
        sign_snr_values_db = np.asarray(snr_values_db[sign_point_indices], dtype=float)
        needed_snr_values_db = np.asarray(
            np.unique(
                np.concatenate(
                    [
                        eval_snr_values_db,
                        sign_snr_values_db,
                    ]
                )
            ),
            dtype=float,
        )
        if len(needed_snr_values_db) == len(eval_snr_values_db) and np.allclose(needed_snr_values_db, eval_snr_values_db):
            needed_svd_array = refined_svd_array
            needed_ucd_array = refined_ucd_array
        else:
            needed_svd_array, needed_ucd_array = evaluate_svd_ucd_rates_on_grid(
                env=env,
                true_channels=true_channels,
                f_rf=f_rf,
                snr_values_db=needed_snr_values_db,
                sample_average=sample_average,
                bits_per_symbol=int(args.bits_per_symbol),
                labeling=str(args.labeling),
            )
        svd_array = np.full(len(snr_values_db), np.nan, dtype=float)
        ucd_array = np.full(len(snr_values_db), np.nan, dtype=float)
        for snr_db, svd_rate, ucd_rate in zip(needed_snr_values_db, needed_svd_array, needed_ucd_array):
            match_idx = np.where(np.isclose(snr_values_db, float(snr_db)))[0]
            if len(match_idx) > 0:
                target_idx = int(match_idx[0])
                svd_array[target_idx] = float(svd_rate)
                ucd_array[target_idx] = float(ucd_rate)
    else:
        if len(eval_snr_values_db) == len(snr_values_db) and np.allclose(eval_snr_values_db, snr_values_db):
            svd_array = refined_svd_array
            ucd_array = refined_ucd_array
        else:
            svd_array, ucd_array = evaluate_svd_ucd_rates_on_grid(
                env=env,
                true_channels=true_channels,
                f_rf=f_rf,
                snr_values_db=snr_values_db,
                sample_average=sample_average,
                bits_per_symbol=int(args.bits_per_symbol),
                labeling=str(args.labeling),
            )
    delta_array = ucd_array - svd_array
    return ThresholdSample(
        channel_index=int(channel_idx),
        config_id=int(config_id),
        tokens=tokens,
        token_mask=token_mask,
        config_features=config_features,
        threshold_db=threshold_db,
        crossing_type=crossing_type,
        svd_rates=svd_array,
        ucd_rates=ucd_array,
        delta_rates=delta_array,
    )


def load_threshold_samples(dataset_path: Path) -> list[ThresholdSample]:
    if not dataset_path.exists():
        return []

    data = np.load(dataset_path, allow_pickle=True)
    tokens = np.asarray(data["tokens"], dtype=np.float32)
    if "token_masks" in data.files:
        token_masks = np.asarray(data["token_masks"], dtype=np.float32)
    else:
        token_masks = np.ones(tokens.shape[:2], dtype=np.float32)
    if "config_features" in data.files:
        config_features = np.asarray(data["config_features"], dtype=np.float32)
    else:
        config_features = np.zeros((tokens.shape[0], 10), dtype=np.float32)
    if "config_ids" in data.files:
        config_ids = np.asarray(data["config_ids"], dtype=np.int64)
    else:
        config_ids = np.zeros(tokens.shape[0], dtype=np.int64)
    thresholds = np.asarray(data["thresholds_db"], dtype=float)
    crossing_types = np.asarray(data["crossing_types"]).astype(str)
    svd_rates = np.asarray(data["svd_rates"], dtype=float)
    ucd_rates = np.asarray(data["ucd_rates"], dtype=float)
    delta_rates = np.asarray(data["delta_rates"], dtype=float)
    if "channel_indices" in data.files:
        channel_indices = np.asarray(data["channel_indices"], dtype=int)
    else:
        channel_indices = np.arange(tokens.shape[0], dtype=int)

    samples = []
    for sample_idx in range(tokens.shape[0]):
        samples.append(
            ThresholdSample(
                channel_index=int(channel_indices[sample_idx]),
                config_id=int(config_ids[sample_idx]),
                tokens=tokens[sample_idx],
                token_mask=token_masks[sample_idx],
                config_features=config_features[sample_idx],
                threshold_db=float(thresholds[sample_idx]),
                crossing_type=str(crossing_types[sample_idx]),
                svd_rates=svd_rates[sample_idx],
                ucd_rates=ucd_rates[sample_idx],
                delta_rates=delta_rates[sample_idx],
            )
        )
    return sorted(samples, key=lambda sample: sample.channel_index)


def build_generation_plan(config_specs: list[ConfigSpec]) -> list[tuple[int, SystemConfig]]:
    plan = []
    for spec in config_specs:
        plan.extend((spec.config_id, spec.system_config) for _ in range(spec.num_channels))
    return plan


def _generate_threshold_sample_worker(
    payload: tuple[int, int, SystemConfig, np.ndarray, dict[str, object]],
) -> ThresholdSample:
    channel_idx, config_id, system_config, snr_values_db, args_dict = payload
    args = argparse.Namespace(**args_dict)
    env = build_env_from_config(system_config, args)
    return generate_one_threshold_sample(
        env=env,
        snr_values_db=snr_values_db,
        args=args,
        channel_idx=channel_idx,
        config_id=config_id,
    )


def generate_threshold_samples(
    generation_plan: list[tuple[int, SystemConfig]],
    snr_values_db: np.ndarray,
    args: argparse.Namespace,
    existing_samples: list[ThresholdSample] | None = None,
) -> list[ThresholdSample]:
    samples = sorted(list(existing_samples or []), key=lambda sample: sample.channel_index)
    existing_indices = {sample.channel_index for sample in samples}
    checkpoint_every = max(int(args.checkpoint_every), 0)
    max_new_channels = args.max_new_channels
    if max_new_channels is not None:
        max_new_channels = max(int(max_new_channels), 0)

    pending_payloads = []
    for channel_idx, (config_id, system_config) in enumerate(generation_plan):
        if channel_idx in existing_indices:
            continue
        if max_new_channels is not None and len(pending_payloads) >= max_new_channels:
            break
        pending_payloads.append((channel_idx, config_id, system_config, snr_values_db, vars(args).copy()))

    if not pending_payloads:
        return samples

    new_count = 0
    if int(args.generator_workers) > 0:
        with ProcessPoolExecutor(max_workers=int(args.generator_workers)) as executor:
            future_map = {
                executor.submit(_generate_threshold_sample_worker, payload): payload[0]
                for payload in pending_payloads
            }
            for future in as_completed(future_map):
                sample = future.result()
                samples.append(sample)
                samples = sorted(samples, key=lambda item: item.channel_index)
                existing_indices.add(sample.channel_index)
                new_count += 1
                print(
                    f"channel {sample.channel_index + 1}/{len(generation_plan)}: "
                    f"threshold={sample.threshold_db:.3f} dB, type={sample.crossing_type}, "
                    f"delta_min={np.nanmin(sample.delta_rates):.4f}, delta_max={np.nanmax(sample.delta_rates):.4f}",
                    flush=True,
                )

                should_checkpoint = (
                    checkpoint_every > 0
                    and (new_count % checkpoint_every == 0 or len(samples) == len(generation_plan))
                )
                if should_checkpoint:
                    write_dataset_files(samples, snr_values_db, args)
                    print(f"checkpoint saved after {len(samples)}/{len(generation_plan)} channels", flush=True)
    else:
        for payload in pending_payloads:
            sample = _generate_threshold_sample_worker(payload)
            samples.append(sample)
            samples = sorted(samples, key=lambda item: item.channel_index)
            existing_indices.add(sample.channel_index)
            new_count += 1
            print(
                f"channel {sample.channel_index + 1}/{len(generation_plan)}: "
                f"threshold={sample.threshold_db:.3f} dB, type={sample.crossing_type}, "
                f"delta_min={np.nanmin(sample.delta_rates):.4f}, delta_max={np.nanmax(sample.delta_rates):.4f}",
                flush=True,
            )

            should_checkpoint = (
                checkpoint_every > 0
                and (new_count % checkpoint_every == 0 or len(samples) == len(generation_plan))
            )
            if should_checkpoint:
                write_dataset_files(samples, snr_values_db, args)
                print(f"checkpoint saved after {len(samples)}/{len(generation_plan)} channels", flush=True)

    return samples


def write_dataset_files(
    samples: list[ThresholdSample],
    snr_values_db: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix)
    dataset_path = out_dir / f"{prefix}_dataset.npz"
    summary_path = out_dir / f"{prefix}_thresholds.csv"
    curve_path = out_dir / f"{prefix}_curves.csv"

    max_tokens = max(sample.tokens.shape[0] for sample in samples)
    token_dim = samples[0].tokens.shape[1]
    tokens = np.zeros((len(samples), max_tokens, token_dim), dtype=np.float32)
    token_masks = np.zeros((len(samples), max_tokens), dtype=np.float32)
    for sample_idx, sample in enumerate(samples):
        token_count = sample.tokens.shape[0]
        tokens[sample_idx, :token_count] = sample.tokens
        token_masks[sample_idx, :token_count] = sample.token_mask
    config_features = np.stack([sample.config_features for sample in samples], axis=0)
    config_ids = np.asarray([sample.config_id for sample in samples], dtype=np.int64)
    thresholds = np.asarray([sample.threshold_db for sample in samples], dtype=np.float32)
    crossing_types = np.asarray([sample.crossing_type for sample in samples])
    channel_indices = np.asarray([sample.channel_index for sample in samples], dtype=np.int64)
    svd_rates = np.stack([sample.svd_rates for sample in samples], axis=0)
    ucd_rates = np.stack([sample.ucd_rates for sample in samples], axis=0)
    delta_rates = np.stack([sample.delta_rates for sample in samples], axis=0)

    np.savez(
        dataset_path,
        tokens=tokens,
        token_masks=token_masks,
        config_features=config_features,
        config_ids=config_ids,
        thresholds_db=thresholds,
        crossing_types=crossing_types,
        channel_indices=channel_indices,
        svd_rates=svd_rates,
        ucd_rates=ucd_rates,
        delta_rates=delta_rates,
        snr_values_db=np.asarray(snr_values_db, dtype=np.float32),
        args_json=json.dumps(vars(args), indent=2, sort_keys=True),
    )

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel_index", "config_id", "threshold_db", "crossing_type", "delta_min", "delta_max"])
        for sample in samples:
            writer.writerow(
                [
                    sample.channel_index,
                    sample.config_id,
                    f"{sample.threshold_db:.8g}",
                    sample.crossing_type,
                    f"{float(np.min(sample.delta_rates)):.8g}",
                    f"{float(np.max(sample.delta_rates)):.8g}",
                ]
            )

    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel_index", "config_id", "snr_db", "svd_rate", "ucd_rate", "delta_rate"])
        for sample in samples:
            for snr_db, svd_rate, ucd_rate, delta_rate in zip(
                snr_values_db,
                sample.svd_rates,
                sample.ucd_rates,
                sample.delta_rates,
            ):
                writer.writerow(
                    [
                        sample.channel_index,
                        sample.config_id,
                        f"{float(snr_db):.8g}",
                        f"{float(svd_rate):.8g}",
                        f"{float(ucd_rate):.8g}",
                        f"{float(delta_rate):.8g}",
                    ]
                )
    return dataset_path, summary_path, curve_path


def split_indices(num_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    if num_samples < 5:
        return indices, indices, indices
    train_end = max(1, int(round(0.7 * num_samples)))
    val_end = max(train_end + 1, int(round(0.85 * num_samples)))
    val_end = min(val_end, num_samples - 1)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def standardize_tokens(tokens: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_tokens = tokens[train_idx]
    feature_mean = np.mean(train_tokens, axis=(0, 1), keepdims=True)
    feature_std = np.std(train_tokens, axis=(0, 1), keepdims=True)
    feature_std = np.maximum(feature_std, 1e-6)
    return (tokens - feature_mean) / feature_std, feature_mean.squeeze(), feature_std.squeeze()


def standardize_config_features(
    config_features: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_configs = config_features[train_idx]
    feature_mean = np.mean(train_configs, axis=0, keepdims=True)
    feature_std = np.std(train_configs, axis=0, keepdims=True)
    feature_std = np.maximum(feature_std, 1e-6)
    return (
        (config_features - feature_mean) / feature_std,
        feature_mean.squeeze(),
        feature_std.squeeze(),
    )


def build_sign_targets(delta_rates: np.ndarray, sign_point_indices: np.ndarray) -> np.ndarray:
    sign_values = delta_rates[:, sign_point_indices]
    return (sign_values > 0.0).astype(np.float32)


def pack_sample_arrays(
    samples: list[ThresholdSample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_tokens = max(sample.tokens.shape[0] for sample in samples)
    token_dim = samples[0].tokens.shape[1]
    tokens = np.zeros((len(samples), max_tokens, token_dim), dtype=np.float32)
    token_masks = np.zeros((len(samples), max_tokens), dtype=np.float32)
    config_features = np.stack([sample.config_features for sample in samples], axis=0).astype(np.float32)
    thresholds = np.asarray([sample.threshold_db for sample in samples], dtype=np.float32)
    svd_rates = np.stack([sample.svd_rates for sample in samples], axis=0)
    ucd_rates = np.stack([sample.ucd_rates for sample in samples], axis=0)
    for sample_idx, sample in enumerate(samples):
        token_count = sample.tokens.shape[0]
        tokens[sample_idx, :token_count] = sample.tokens
        token_masks[sample_idx, :token_count] = sample.token_mask
    return tokens, token_masks, config_features, thresholds, svd_rates, ucd_rates


def build_loader(
    tokens: np.ndarray,
    token_masks: np.ndarray,
    config_features: np.ndarray,
    targets: np.ndarray,
    sign_targets: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(tokens[indices].astype(np.float32)),
        torch.from_numpy(token_masks[indices].astype(np.float32)),
        torch.from_numpy(config_features[indices].astype(np.float32)),
        torch.from_numpy(targets[indices].astype(np.float32)),
        torch.from_numpy(sign_targets[indices].astype(np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(indices))),
        shuffle=shuffle,
        num_workers=max(int(num_workers), 0),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    threshold_criterion: nn.Module,
    sign_criterion: nn.Module,
    sign_snr_values: torch.Tensor,
    sign_loss_weight: float,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    sign_snr_values = sign_snr_values.to(device)
    for batch_tokens, batch_token_masks, batch_config_features, batch_targets, batch_sign_targets in loader:
        batch_tokens = batch_tokens.to(device, non_blocking=True)
        batch_token_masks = batch_token_masks.to(device, non_blocking=True)
        batch_config_features = batch_config_features.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        batch_sign_targets = batch_sign_targets.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(use_amp and device.type == "cuda"),
        ):
            predictions, sign_logits = model(
                batch_tokens,
                batch_token_masks,
                batch_config_features,
                sign_snr_values=sign_snr_values,
            )
            loss = threshold_criterion(predictions, batch_targets)
            if sign_logits is not None and sign_loss_weight > 0.0:
                loss = loss + float(sign_loss_weight) * sign_criterion(sign_logits, batch_sign_targets)
        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_threshold_predictions(
    thresholds_true: np.ndarray,
    thresholds_pred: np.ndarray,
    svd_rates: np.ndarray,
    ucd_rates: np.ndarray,
    snr_values_db: np.ndarray,
) -> dict[str, float]:
    mae = float(np.mean(np.abs(thresholds_pred - thresholds_true)))
    rmse = float(np.sqrt(np.mean((thresholds_pred - thresholds_true) ** 2)))

    valid_mask = np.isfinite(svd_rates) & np.isfinite(ucd_rates)
    if not np.any(valid_mask):
        return {
            "threshold_mae_db": mae,
            "threshold_rmse_db": rmse,
            "selected_rate": float("nan"),
            "oracle_rate": float("nan"),
            "always_svd_rate": float("nan"),
            "always_ucd_rate": float("nan"),
            "regret_to_oracle": float("nan"),
            "grid_decision_accuracy": float("nan"),
        }

    snr_grid = np.asarray(snr_values_db, dtype=float)[None, :]
    choose_ucd_pred = snr_grid >= thresholds_pred[:, None]
    selected_rates = np.where(choose_ucd_pred, ucd_rates, svd_rates)
    oracle_rates = np.maximum(svd_rates, ucd_rates)
    always_svd_rate = float(np.nanmean(np.where(valid_mask, svd_rates, np.nan)))
    always_ucd_rate = float(np.nanmean(np.where(valid_mask, ucd_rates, np.nan)))
    selected_rate = float(np.nanmean(np.where(valid_mask, selected_rates, np.nan)))
    oracle_rate = float(np.nanmean(np.where(valid_mask, oracle_rates, np.nan)))
    regret = float(np.nanmean(np.where(valid_mask, oracle_rates - selected_rates, np.nan)))
    decision_accuracy = float(
        np.nanmean(
            np.where(
                valid_mask,
                choose_ucd_pred == (ucd_rates > svd_rates),
                np.nan,
            )
        )
    )
    return {
        "threshold_mae_db": mae,
        "threshold_rmse_db": rmse,
        "selected_rate": selected_rate,
        "oracle_rate": oracle_rate,
        "always_svd_rate": always_svd_rate,
        "always_ucd_rate": always_ucd_rate,
        "regret_to_oracle": regret,
        "grid_decision_accuracy": decision_accuracy,
    }


def train_threshold_model(
    samples: list[ThresholdSample],
    snr_values_db: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    tokens, token_masks, config_features, thresholds, svd_rates, ucd_rates = pack_sample_arrays(samples)
    train_idx, val_idx, test_idx = split_indices(len(samples), int(args.seed) + 17)
    tokens_norm, feature_mean, feature_std = standardize_tokens(tokens, train_idx)
    config_norm, config_mean, config_std = standardize_config_features(config_features, train_idx)

    target_mean = float(np.mean(thresholds[train_idx]))
    target_std = float(np.std(thresholds[train_idx]))
    target_std = max(target_std, 1e-6)
    targets_norm = (thresholds - target_mean) / target_std
    sign_point_indices = np.unique(
        np.clip(
            np.round(np.linspace(0, len(snr_values_db) - 1, num=min(7, len(snr_values_db)))).astype(int),
            0,
            len(snr_values_db) - 1,
        )
    )
    sign_targets = build_sign_targets(ucd_rates - svd_rates, sign_point_indices)
    sign_snr_values = torch.from_numpy(np.asarray(snr_values_db[sign_point_indices], dtype=np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = SpectralSetTransformerThresholdRegressor(
        token_dim=tokens.shape[-1],
        config_dim=config_features.shape[-1],
        d_model=int(args.d_model),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    ).to(device)
    threshold_criterion = nn.SmoothL1Loss(beta=max(float(args.huber_delta_db) / target_std, 1e-6))
    sign_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    use_amp = bool(device.type == "cuda" and not bool(args.disable_amp))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    train_loader = build_loader(
        tokens_norm,
        token_masks,
        config_norm,
        targets_norm,
        sign_targets,
        train_idx,
        int(args.batch_size),
        True,
        int(args.num_workers),
    )
    val_loader = build_loader(
        tokens_norm,
        token_masks,
        config_norm,
        targets_norm,
        sign_targets,
        val_idx,
        int(args.batch_size),
        False,
        int(args.num_workers),
    )

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            threshold_criterion,
            sign_criterion,
            sign_snr_values,
            float(args.sign_loss_weight),
            optimizer,
            device,
            use_amp,
            scaler,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            threshold_criterion,
            sign_criterion,
            sign_snr_values,
            float(args.sign_loss_weight),
            None,
            device,
            False,
            None,
        )
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 25 == 0 or epoch == int(args.epochs):
            print(f"epoch {epoch:4d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}", flush=True)
        if patience_counter >= int(args.patience):
            print(f"early stopping at epoch {epoch}, best_epoch={best_epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        all_predictions_norm, _ = model(
            torch.from_numpy(tokens_norm.astype(np.float32)).to(device, non_blocking=True),
            torch.from_numpy(token_masks.astype(np.float32)).to(device, non_blocking=True),
            torch.from_numpy(config_norm.astype(np.float32)).to(device, non_blocking=True),
            sign_snr_values=sign_snr_values.to(device),
        )
        all_predictions_norm = all_predictions_norm.cpu().numpy()
    all_predictions_db = all_predictions_norm * target_std + target_mean

    metrics = {
        "num_samples": float(len(samples)),
        "num_train": float(len(train_idx)),
        "num_val": float(len(val_idx)),
        "num_test": float(len(test_idx)),
        "best_epoch": float(best_epoch),
        "best_val_loss": float(best_val_loss),
    }
    metrics.update(
        {
            f"train_{key}": value
            for key, value in evaluate_threshold_predictions(
                thresholds[train_idx],
                all_predictions_db[train_idx],
                svd_rates[train_idx],
                ucd_rates[train_idx],
                snr_values_db,
            ).items()
        }
    )
    metrics.update(
        {
            f"test_{key}": value
            for key, value in evaluate_threshold_predictions(
                thresholds[test_idx],
                all_predictions_db[test_idx],
                svd_rates[test_idx],
                ucd_rates[test_idx],
                snr_values_db,
            ).items()
        }
    )

    artifacts = TrainArtifacts(
        feature_mean=np.asarray(feature_mean, dtype=np.float32),
        feature_std=np.asarray(feature_std, dtype=np.float32),
        config_mean=np.asarray(config_mean, dtype=np.float32),
        config_std=np.asarray(config_std, dtype=np.float32),
        target_mean=target_mean,
        target_std=target_std,
    )
    out_dir = Path(args.out_dir)
    prefix = str(args.output_prefix)
    model_path = out_dir / f"{prefix}_model.pt"
    metrics_path = out_dir / f"{prefix}_metrics.json"
    prediction_path = out_dir / f"{prefix}_predictions.csv"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "artifacts": asdict(artifacts),
            "args": vars(args),
            "metrics": metrics,
        },
        model_path,
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel_index", "threshold_true_db", "threshold_pred_db", "split"])
        split_name = np.full(len(samples), "train", dtype=object)
        split_name[val_idx] = "val"
        split_name[test_idx] = "test"
        for sample_idx, sample in enumerate(samples):
            writer.writerow(
                [
                    sample.channel_index,
                    f"{float(thresholds[sample_idx]):.8g}",
                    f"{float(all_predictions_db[sample_idx]):.8g}",
                    split_name[sample_idx],
                ]
            )

    print(f"saved model: {model_path}", flush=True)
    print(f"saved metrics: {metrics_path}", flush=True)
    print(f"saved predictions: {prediction_path}", flush=True)
    return metrics


def evaluate_saved_model(
    samples: list[ThresholdSample],
    snr_values_db: np.ndarray,
    args: argparse.Namespace,
    model_path: Path,
) -> dict[str, float]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_args = checkpoint.get("args", {})
    artifacts = checkpoint["artifacts"]

    tokens, token_masks, config_features, thresholds, svd_rates, ucd_rates = pack_sample_arrays(samples)

    feature_mean = np.asarray(artifacts["feature_mean"], dtype=np.float32).reshape(1, 1, -1)
    feature_std = np.asarray(artifacts["feature_std"], dtype=np.float32).reshape(1, 1, -1)
    config_mean = np.asarray(artifacts["config_mean"], dtype=np.float32).reshape(1, -1)
    config_std = np.asarray(artifacts["config_std"], dtype=np.float32).reshape(1, -1)
    target_mean = float(artifacts["target_mean"])
    target_std = float(artifacts["target_std"])
    tokens_norm = (tokens - feature_mean) / np.maximum(feature_std, 1e-6)
    config_norm = (config_features - config_mean) / np.maximum(config_std, 1e-6)
    sign_point_indices = np.unique(
        np.clip(
            np.round(np.linspace(0, len(snr_values_db) - 1, num=min(7, len(snr_values_db)))).astype(int),
            0,
            len(snr_values_db) - 1,
        )
    )
    sign_snr_values = torch.from_numpy(np.asarray(snr_values_db[sign_point_indices], dtype=np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralSetTransformerThresholdRegressor(
        token_dim=tokens.shape[-1],
        config_dim=config_features.shape[-1],
        d_model=int(model_args.get("d_model", getattr(args, "d_model", 32))),
        num_heads=int(model_args.get("num_heads", getattr(args, "num_heads", 2))),
        num_layers=int(model_args.get("num_layers", 1)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        predictions_norm, _ = model(
            torch.from_numpy(tokens_norm.astype(np.float32)).to(device),
            torch.from_numpy(token_masks.astype(np.float32)).to(device),
            torch.from_numpy(config_norm.astype(np.float32)).to(device),
            sign_snr_values=sign_snr_values.to(device),
        )
        predictions_norm = predictions_norm.cpu().numpy()
    predictions_db = predictions_norm * target_std + target_mean

    metrics = evaluate_threshold_predictions(
        thresholds_true=thresholds,
        thresholds_pred=predictions_db,
        svd_rates=svd_rates,
        ucd_rates=ucd_rates,
        snr_values_db=snr_values_db,
    )
    metrics["num_eval_samples"] = float(len(samples))

    out_dir = Path(args.out_dir)
    prefix = str(args.output_prefix)
    metrics_path = out_dir / f"{prefix}_eval_metrics.json"
    prediction_path = out_dir / f"{prefix}_eval_predictions.csv"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel_index", "threshold_true_db", "threshold_pred_db", "abs_error_db"])
        for sample_idx, sample in enumerate(samples):
            abs_error = abs(float(predictions_db[sample_idx]) - float(thresholds[sample_idx]))
            writer.writerow(
                [
                    sample.channel_index,
                    f"{float(thresholds[sample_idx]):.8g}",
                    f"{float(predictions_db[sample_idx]):.8g}",
                    f"{abs_error:.8g}",
                ]
            )

    print(f"loaded model: {model_path}", flush=True)
    print(f"saved eval metrics: {metrics_path}", flush=True)
    print(f"saved eval predictions: {prediction_path}", flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    if args.num_channels <= 0:
        raise ValueError("num_channels must be positive.")
    snr_values_db = build_snr_values(args.snr_start_db, args.snr_stop_db, args.snr_step_db)
    config_specs = parse_config_pool(args)
    generation_plan = build_generation_plan(config_specs)

    print(
        "generating SVD/UCD threshold dataset: "
        f"channels={len(generation_plan)}, snr={snr_values_db[0]:g}:{args.snr_step_db:g}:{snr_values_db[-1]:g} dB, "
        f"samples={args.train_num_samples}, repeats={args.train_num_repeats}, "
        f"configs={len(config_specs)}",
        flush=True,
    )
    dataset_path = Path(args.out_dir) / f"{args.output_prefix}_dataset.npz"
    existing_samples = load_threshold_samples(dataset_path) if args.resume else []
    if existing_samples:
        print(
            f"resuming from {len(existing_samples)}/{len(generation_plan)} saved channels: {dataset_path}",
            flush=True,
        )
    samples = generate_threshold_samples(
        generation_plan=generation_plan,
        snr_values_db=snr_values_db,
        args=args,
        existing_samples=existing_samples,
    )
    dataset_path, summary_path, curve_path = write_dataset_files(samples, snr_values_db, args)
    print(f"saved dataset: {dataset_path}", flush=True)
    print(f"saved thresholds: {summary_path}", flush=True)
    print(f"saved curves: {curve_path}", flush=True)

    type_counts: dict[str, int] = {}
    for sample in samples:
        type_counts[sample.crossing_type] = type_counts.get(sample.crossing_type, 0) + 1
    print(f"crossing types: {type_counts}", flush=True)

    if args.eval_model is not None:
        metrics = evaluate_saved_model(
            samples=samples,
            snr_values_db=snr_values_db,
            args=args,
            model_path=Path(args.eval_model),
        )
        print("external eval metrics:", flush=True)
        for key, value in metrics.items():
            print(f"  {key}: {value:.6g}", flush=True)
        return

    if args.skip_train:
        return

    metrics = train_threshold_model(samples=samples, snr_values_db=snr_values_db, args=args)
    print("training metrics:", flush=True)
    for key, value in metrics.items():
        print(f"  {key}: {value:.6g}", flush=True)


if __name__ == "__main__":
    main()
