from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
CLASSICAL_DIR = ROOT / "classical"
for extra_path in (ROOT, CLASSICAL_DIR):
    text = str(extra_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from learn_svd_ucd_threshold import (
    ThresholdSample,
    config_from_features,
    evaluate_threshold_predictions,
    load_threshold_samples,
)


@dataclass(frozen=True)
class PerConfigArtifacts:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float
    config_id: int


class SpectralAttentionThresholdRegressor(nn.Module):
    """Original lightweight fixed-config attention regressor."""

    def __init__(
        self,
        num_users: int,
        num_streams_per_user: int,
        token_dim: int,
        d_model: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.num_users = int(num_users)
        self.num_streams_per_user = int(num_streams_per_user)
        self.token_count = self.num_users * self.num_streams_per_user

        user_ids = []
        stream_ids = []
        for user_idx in range(self.num_users):
            for stream_idx in range(self.num_streams_per_user):
                user_ids.append(user_idx)
                stream_ids.append(stream_idx)
        self.register_buffer("user_ids", torch.tensor(user_ids, dtype=torch.long), persistent=False)
        self.register_buffer("stream_ids", torch.tensor(stream_ids, dtype=torch.long), persistent=False)

        self.token_projection = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.user_embedding = nn.Embedding(self.num_users, d_model)
        self.stream_embedding = nn.Embedding(self.num_streams_per_user, d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.pool_score = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != self.token_count:
            raise ValueError(
                "tokens must have shape (batch, num_users * num_streams_per_user, token_dim)."
            )

        user_emb = self.user_embedding(self.user_ids).unsqueeze(0)
        stream_emb = self.stream_embedding(self.stream_ids).unsqueeze(0)
        embedded = self.token_projection(tokens) + user_emb + stream_emb
        attended, _ = self.attention(embedded, embedded, embedded, need_weights=False)
        hidden = self.norm(embedded + attended)
        pool_weight = torch.softmax(self.pool_score(hidden).squeeze(-1), dim=-1)
        global_state = torch.sum(hidden * pool_weight.unsqueeze(-1), dim=1)
        return self.head(global_state).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one original-style threshold regressor per config_id.",
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--huber-delta-db", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260522)
    return parser.parse_args()


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


def build_loader(
    tokens: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(tokens[indices].astype(np.float32)),
        torch.from_numpy(targets[indices].astype(np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(indices))),
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    for batch_tokens, batch_targets in loader:
        batch_tokens = batch_tokens.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(use_amp and device.type == "cuda"),
        ):
            predictions = model(batch_tokens)
            loss = criterion(predictions, batch_targets)
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


def trim_group_tokens(samples: list[ThresholdSample]) -> np.ndarray:
    trimmed = []
    for sample in samples:
        valid_count = int(np.sum(sample.token_mask > 0.5))
        # Keep only the original spectral features:
        # [log_sigma, centered_log_sigma, power_fraction].
        trimmed.append(np.asarray(sample.tokens[:valid_count, :3], dtype=np.float32))
    token_shapes = {item.shape[0] for item in trimmed}
    if len(token_shapes) != 1:
        raise ValueError(f"Config group contains mixed token counts: {sorted(token_shapes)}")
    return np.stack(trimmed, axis=0).astype(np.float32)


def train_one_config(
    config_id: int,
    samples: list[ThresholdSample],
    snr_values_db: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = config_from_features(samples[0].config_features)
    tokens = trim_group_tokens(samples)
    thresholds = np.asarray([sample.threshold_db for sample in samples], dtype=np.float32)
    svd_rates = np.stack([sample.svd_rates for sample in samples], axis=0)
    ucd_rates = np.stack([sample.ucd_rates for sample in samples], axis=0)

    train_idx, val_idx, test_idx = split_indices(len(samples), int(args.seed) + 97 * int(config_id))
    tokens_norm, feature_mean, feature_std = standardize_tokens(tokens, train_idx)
    target_mean = float(np.mean(thresholds[train_idx]))
    target_std = max(float(np.std(thresholds[train_idx])), 1e-6)
    targets_norm = (thresholds - target_mean) / target_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = SpectralAttentionThresholdRegressor(
        num_users=int(config.num_users),
        num_streams_per_user=int(config.num_streams_per_user),
        token_dim=tokens.shape[-1],
        d_model=int(args.d_model),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
    ).to(device)
    criterion = nn.SmoothL1Loss(beta=max(float(args.huber_delta_db) / target_std, 1e-6))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    use_amp = bool(device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    train_loader = build_loader(tokens_norm, targets_norm, train_idx, int(args.batch_size), True)
    val_loader = build_loader(tokens_norm, targets_norm, val_idx, int(args.batch_size), False)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, use_amp, scaler)
        val_loss = run_epoch(model, val_loader, criterion, None, device, False, None)
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch == 1 or epoch % 25 == 0 or epoch == int(args.epochs):
            print(
                f"[config {config_id}] epoch {epoch:4d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}",
                flush=True,
            )
        if patience_counter >= int(args.patience):
            print(f"[config {config_id}] early stopping at epoch {epoch}, best_epoch={best_epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        predictions_norm = model(torch.from_numpy(tokens_norm.astype(np.float32)).to(device)).cpu().numpy()
    predictions_db = predictions_norm * target_std + target_mean

    metrics = {
        "config_id": float(config_id),
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
                predictions_db[train_idx],
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
                predictions_db[test_idx],
                svd_rates[test_idx],
                ucd_rates[test_idx],
                snr_values_db,
            ).items()
        }
    )

    artifacts = PerConfigArtifacts(
        feature_mean=np.asarray(feature_mean, dtype=np.float32),
        feature_std=np.asarray(feature_std, dtype=np.float32),
        target_mean=target_mean,
        target_std=target_std,
        config_id=int(config_id),
    )
    model_path = out_dir / f"config_{config_id}_model.pt"
    metrics_path = out_dir / f"config_{config_id}_metrics.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "artifacts": asdict(artifacts),
            "config": asdict(config),
            "args": vars(args),
            "metrics": metrics,
        },
        model_path,
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(f"[config {config_id}] saved model: {model_path}", flush=True)
    print(f"[config {config_id}] saved metrics: {metrics_path}", flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    samples = load_threshold_samples(Path(args.dataset))
    if not samples:
        raise ValueError(f"No samples loaded from dataset: {args.dataset}")

    dataset = np.load(args.dataset, allow_pickle=True)
    snr_values_db = np.asarray(dataset["snr_values_db"], dtype=float)

    grouped: dict[int, list[ThresholdSample]] = {}
    for sample in samples:
        grouped.setdefault(int(sample.config_id), []).append(sample)

    all_metrics = {}
    for config_id in sorted(grouped):
        print(f"training config_id={config_id} with {len(grouped[config_id])} samples", flush=True)
        all_metrics[f"config_{config_id}"] = train_one_config(config_id, grouped[config_id], snr_values_db, args)

    summary_path = Path(args.out_dir) / "per_config_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2, sort_keys=True)
    print(f"saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
