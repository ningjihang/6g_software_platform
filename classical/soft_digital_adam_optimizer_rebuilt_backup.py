from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from multiuser_simulation_environment import MultiUserSimulationEnvironment, ReceiverAverageEvaluation
from soft_sic_surrogate import evaluate_soft_sic_surrogate


@dataclass(frozen=True)
class SoftRateAverageEvaluation:
    sum_rate: float
    sum_rate_std: float
    bit_error_rate: float
    user_rates: np.ndarray
    user_bit_error_rates: np.ndarray
    offdiag_to_desired: float


@dataclass(frozen=True)
class FixedAnalogSoftAdamResult:
    baseline_strategy: str
    f_rf: np.ndarray
    baseline_f_bb: np.ndarray
    optimized_f_bb: np.ndarray
    baseline_soft_train_eval: SoftRateAverageEvaluation
    optimized_soft_train_eval: SoftRateAverageEvaluation
    baseline_soft_eval_eval: SoftRateAverageEvaluation | None
    optimized_soft_eval_eval: SoftRateAverageEvaluation | None
    baseline_receiver_eval: ReceiverAverageEvaluation
    optimized_receiver_eval: ReceiverAverageEvaluation
    history: list[float]


def _to_soft_eval(receiver_eval: ReceiverAverageEvaluation) -> SoftRateAverageEvaluation:
    return SoftRateAverageEvaluation(
        sum_rate=float(receiver_eval.sum_rate),
        sum_rate_std=float(receiver_eval.sum_rate_std),
        bit_error_rate=float(receiver_eval.bit_error_rate),
        user_rates=np.asarray(receiver_eval.user_rates, dtype=float),
        user_bit_error_rates=np.asarray(receiver_eval.user_bit_error_rates, dtype=float),
        offdiag_to_desired=float(receiver_eval.offdiag_to_desired),
    )


def _score_from_receiver_eval(receiver_eval: ReceiverAverageEvaluation) -> float:
    return float(receiver_eval.sum_rate)


class FixedAnalogSoftTHPObjective:
    """Legacy-compatible fixed-analog objective using exact THP Monte Carlo evaluation."""

    def __init__(
        self,
        env: MultiUserSimulationEnvironment,
        user_channels: np.ndarray,
        f_rf: np.ndarray,
        snr_per_stream: float,
        bits_per_symbol: int,
        sample_average,
        device: str = "cpu",
    ):
        del device
        self.env = env
        self.user_channels = np.asarray(user_channels, dtype=complex)
        self.f_rf = np.asarray(f_rf, dtype=complex)
        self.snr_per_stream = float(snr_per_stream)
        self.bits_per_symbol = int(bits_per_symbol)
        self.sample_average = sample_average
        self.effective_channels = env.build_effective_channels(self.user_channels, self.f_rf)
        self.null_bases = [
            env.build_bd_null_basis(self.effective_channels, user_idx)
            for user_idx in range(env.num_users)
        ]

    def build_f_bb(self, coeff_blocks: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
        if len(coeff_blocks) != self.env.num_users:
            raise ValueError(
                f"Expected {self.env.num_users} user coefficient blocks, got {len(coeff_blocks)}."
            )
        projected_blocks = []
        for user_idx, coeff_block in enumerate(coeff_blocks):
            projected_blocks.append(self.null_bases[user_idx] @ np.asarray(coeff_block, dtype=complex))
        f_bb = np.hstack(projected_blocks)
        f_bb = self.env.normalize_digital_precoder(self.f_rf, f_bb)
        return projected_blocks, np.asarray(f_bb, dtype=complex)

    def coeff_blocks_from_f_bb(self, f_bb: np.ndarray) -> list[np.ndarray]:
        user_blocks = self.env.split_user_blocks(np.asarray(f_bb, dtype=complex))
        coeff_blocks = []
        for user_idx, user_block in enumerate(user_blocks):
            coeff_blocks.append(self.null_bases[user_idx].conj().T @ user_block)
        return coeff_blocks

    def evaluate_f_bb(self, f_bb: np.ndarray) -> SoftRateAverageEvaluation:
        receiver_eval = self.env.evaluate_precoder_current_receiver_average_thp(
            user_channels=self.user_channels,
            f_rf=self.f_rf,
            f_bb=np.asarray(f_bb, dtype=complex),
            snr_per_stream=self.snr_per_stream,
            bits_per_symbol=self.bits_per_symbol,
            sample_average=self.sample_average,
        )
        return _to_soft_eval(receiver_eval)

    def evaluate_coeff_blocks(self, coeff_blocks: list[np.ndarray]) -> tuple[SoftRateAverageEvaluation, np.ndarray]:
        _, f_bb = self.build_f_bb(coeff_blocks)
        return self.evaluate_f_bb(f_bb), f_bb


def _select_best_structured_baseline(
    env: MultiUserSimulationEnvironment,
    user_channels: np.ndarray,
    f_rf: np.ndarray,
    snr_per_stream: float,
    objective: FixedAnalogSoftTHPObjective,
    strategies: tuple[str, ...] = ("svd", "gmd"),
) -> tuple[str, np.ndarray, SoftRateAverageEvaluation]:
    best_strategy = ""
    best_f_bb = None
    best_eval = None
    best_rate = -np.inf
    for strategy in strategies:
        chain = env.build_structured_digital_chain(
            user_channels=user_channels,
            f_rf=f_rf,
            snr_per_stream=snr_per_stream,
            strategy=strategy,
        )
        evaluation = objective.evaluate_f_bb(chain.f_bb)
        if evaluation.sum_rate > best_rate + 1e-12:
            best_strategy = strategy
            best_f_bb = np.asarray(chain.f_bb, dtype=complex)
            best_eval = evaluation
            best_rate = evaluation.sum_rate
    if best_f_bb is None or best_eval is None:
        raise RuntimeError("Failed to construct a fixed-analog structured baseline.")
    return best_strategy, best_f_bb, best_eval


def _random_complex_like(rng: np.random.Generator, reference: np.ndarray) -> np.ndarray:
    scale = 1.0 / np.sqrt(2.0)
    return scale * (
        rng.standard_normal(np.shape(reference)) + 1j * rng.standard_normal(np.shape(reference))
    )


def optimize_fixed_analog_soft_digital_adam(
    env: MultiUserSimulationEnvironment,
    user_channels: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    train_average,
    eval_average=None,
    num_iterations: int = 100,
    learning_rate: float = 0.03,
    grad_clip_norm: float | None = 10.0,
    weight_decay: float = 0.0,
    device: str = "cpu",
    f_rf: np.ndarray | None = None,
    baseline_strategies: tuple[str, ...] = ("svd", "gmd"),
    initial_f_bb: np.ndarray | None = None,
    initial_strategy_name: str | None = None,
) -> FixedAnalogSoftAdamResult:
    del weight_decay
    effective_eval_average = train_average if eval_average is None else eval_average
    design_user_channels = np.asarray(user_channels, dtype=complex)
    if f_rf is None:
        f_rf = env.build_analog_precoder(design_user_channels)
    f_rf = np.asarray(f_rf, dtype=complex)

    train_objective = FixedAnalogSoftTHPObjective(
        env=env,
        user_channels=design_user_channels,
        f_rf=f_rf,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=train_average,
    )
    eval_objective = FixedAnalogSoftTHPObjective(
        env=env,
        user_channels=design_user_channels,
        f_rf=f_rf,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=effective_eval_average,
    )

    # If an explicit starting digital precoder is provided, continue optimizing
    # from that point instead of reselecting a new structured baseline.
    if initial_f_bb is not None:
        baseline_strategy = (
            str(initial_strategy_name).strip().lower()
            if initial_strategy_name is not None
            else "external"
        )
        baseline_f_bb = np.asarray(initial_f_bb, dtype=complex)
        baseline_soft_train_eval = train_objective.evaluate_f_bb(baseline_f_bb)
    else:
        baseline_strategy, baseline_f_bb, baseline_soft_train_eval = _select_best_structured_baseline(
            env=env,
            user_channels=design_user_channels,
            f_rf=f_rf,
            snr_per_stream=snr_per_stream,
            objective=train_objective,
            strategies=baseline_strategies,
        )
    baseline_soft_eval_eval = eval_objective.evaluate_f_bb(baseline_f_bb)
    baseline_receiver_eval = env.evaluate_precoder_current_receiver_average_thp(
        user_channels=design_user_channels,
        f_rf=f_rf,
        f_bb=baseline_f_bb,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=effective_eval_average,
    )

    current_coeff_blocks = train_objective.coeff_blocks_from_f_bb(baseline_f_bb)
    current_f_bb = np.asarray(baseline_f_bb, dtype=complex)
    current_eval = baseline_soft_train_eval
    best_coeff_blocks = [np.asarray(block, dtype=complex).copy() for block in current_coeff_blocks]
    best_f_bb = np.asarray(current_f_bb, dtype=complex)
    best_eval = current_eval
    best_score = _score_from_receiver_eval(
        env.evaluate_precoder_current_receiver_average_thp(
            user_channels=design_user_channels,
            f_rf=f_rf,
            f_bb=current_f_bb,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            sample_average=train_average,
        )
    )
    history = [current_eval.sum_rate]
    torch_device = torch.device(device)
    real_dtype = torch.float32 if torch_device.type == "cuda" else torch.float64
    parameter_real = []
    for coeff_block in current_coeff_blocks:
        coeff_tensor = torch.tensor(
            np.stack((np.real(coeff_block), np.imag(coeff_block)), axis=-1),
            dtype=real_dtype,
            device=torch_device,
            requires_grad=True,
        )
        parameter_real.append(coeff_tensor)

    optimizer = torch.optim.Adam(parameter_real, lr=float(learning_rate))

    for _ in range(int(num_iterations)):
        optimizer.zero_grad(set_to_none=True)
        coeff_blocks_torch = [
            param[..., 0].to(torch.complex64 if torch_device.type == "cuda" else torch.complex128)
            + 1j * param[..., 1].to(torch.complex64 if torch_device.type == "cuda" else torch.complex128)
            for param in parameter_real
        ]
        loss, surrogate_stats, _, candidate_f_bb_t = evaluate_soft_sic_surrogate(
            user_channels=design_user_channels,
            f_rf=f_rf,
            c_blocks=coeff_blocks_torch,
            sample_average=train_average,
            bits_per_symbol=bits_per_symbol,
            snr_per_stream=snr_per_stream,
            num_streams_per_user=env.num_streams_per_user,
            digital_power_constraint=env.digital_power_constraint,
            interference_penalty_weight=0.0,
            temperature=1.0,
            labeling="gray_standard",
            device=device,
            bd_bases=train_objective.null_bases,
            llr_mode="exact",
        )
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameter_real, max_norm=float(grad_clip_norm))
        optimizer.step()

        candidate_f_bb = np.asarray(candidate_f_bb_t.detach().cpu(), dtype=complex)
        candidate_receiver_eval = env.evaluate_precoder_current_receiver_average_thp(
            user_channels=design_user_channels,
            f_rf=f_rf,
            f_bb=candidate_f_bb,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            sample_average=train_average,
        )
        candidate_score = _score_from_receiver_eval(candidate_receiver_eval)
        current_eval = SoftRateAverageEvaluation(
            sum_rate=float(surrogate_stats.sum_rate),
            sum_rate_std=float(surrogate_stats.sum_rate_std),
            bit_error_rate=float(surrogate_stats.bit_error_rate),
            user_rates=np.asarray(surrogate_stats.user_rates, dtype=float),
            user_bit_error_rates=np.asarray(surrogate_stats.user_bit_error_rates, dtype=float),
            offdiag_to_desired=float(surrogate_stats.offdiag_to_desired),
        )
        current_f_bb = candidate_f_bb
        current_coeff_blocks = train_objective.coeff_blocks_from_f_bb(candidate_f_bb)

        if candidate_score > best_score + 1e-12:
            best_coeff_blocks = [np.asarray(block, dtype=complex).copy() for block in current_coeff_blocks]
            best_f_bb = np.asarray(candidate_f_bb, dtype=complex)
            best_eval = current_eval
            best_score = candidate_score
        history.append(best_eval.sum_rate)

    optimized_f_bb = np.asarray(best_f_bb, dtype=complex)
    optimized_soft_train_eval = best_eval
    optimized_soft_eval_eval = eval_objective.evaluate_f_bb(optimized_f_bb)
    optimized_receiver_eval = env.evaluate_precoder_current_receiver_average_thp(
        user_channels=design_user_channels,
        f_rf=f_rf,
        f_bb=optimized_f_bb,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=effective_eval_average,
    )

    return FixedAnalogSoftAdamResult(
        baseline_strategy=baseline_strategy,
        f_rf=f_rf,
        baseline_f_bb=np.asarray(baseline_f_bb, dtype=complex),
        optimized_f_bb=optimized_f_bb,
        baseline_soft_train_eval=baseline_soft_train_eval,
        optimized_soft_train_eval=optimized_soft_train_eval,
        baseline_soft_eval_eval=baseline_soft_eval_eval,
        optimized_soft_eval_eval=optimized_soft_eval_eval,
        baseline_receiver_eval=baseline_receiver_eval,
        optimized_receiver_eval=optimized_receiver_eval,
        history=history,
    )


__all__ = [
    "FixedAnalogSoftAdamResult",
    "FixedAnalogSoftTHPObjective",
    "SoftRateAverageEvaluation",
    "optimize_fixed_analog_soft_digital_adam",
]
