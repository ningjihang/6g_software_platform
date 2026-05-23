from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from multiuser_simulation_environment import MultiUserSimulationEnvironment, ReceiverAverageEvaluation
from soft_digital_adam_optimizer import (
    FixedAnalogSoftTHPObjective,
    SoftRateAverageEvaluation,
    optimize_fixed_analog_soft_digital_adam,
)
from soft_sic_surrogate import evaluate_soft_sic_surrogate


@dataclass(frozen=True)
class BaselineInitialization:
    strategy: str
    f_rf: np.ndarray
    f_bb: np.ndarray
    soft_train_eval: SoftRateAverageEvaluation
    receiver_eval: ReceiverAverageEvaluation


@dataclass(frozen=True)
class SoftJointAOResult:
    baseline_strategy: str
    baseline_f_rf: np.ndarray
    baseline_f_bb: np.ndarray
    baseline_soft_train_eval: SoftRateAverageEvaluation
    baseline_receiver_eval: ReceiverAverageEvaluation
    optimized_f_rf: np.ndarray
    optimized_f_bb: np.ndarray
    optimized_soft_train_eval: SoftRateAverageEvaluation
    optimized_receiver_eval: ReceiverAverageEvaluation
    history: list[float]
    training_temperature_history: list[float]
    selected_start_strategy: str | None = None
    candidate_start_strategies: tuple[str, ...] = ()


def _temperature_schedule(
    outer_iterations: int,
    initial_temperature: float,
    final_temperature: float | None,
) -> list[float]:
    if outer_iterations <= 0:
        return []
    if final_temperature is None:
        final_temperature = initial_temperature
    if outer_iterations == 1:
        return [float(initial_temperature)]
    start = max(float(initial_temperature), 1e-6)
    stop = max(float(final_temperature), 1e-6)
    return np.geomspace(start, stop, int(outer_iterations)).astype(float).tolist()


def _score(
    receiver_eval: ReceiverAverageEvaluation,
    interference_penalty_weight: float,
    user_fairness_penalty_weight: float,
) -> float:
    fairness_penalty = float(np.std(np.asarray(receiver_eval.user_rates, dtype=float)))
    return float(
        receiver_eval.sum_rate
        - float(interference_penalty_weight) * float(receiver_eval.offdiag_to_desired)
        - float(user_fairness_penalty_weight) * fairness_penalty
    )


def _project_user_blocks_to_new_rf(
    env: MultiUserSimulationEnvironment,
    user_channels: np.ndarray,
    f_rf: np.ndarray,
    reference_f_bb: np.ndarray,
) -> np.ndarray:
    effective_channels = env.build_effective_channels(user_channels, f_rf)
    reference_user_blocks = env.split_user_blocks(reference_f_bb)
    projected_blocks = []
    for user_idx, user_block in enumerate(reference_user_blocks):
        null_basis = env.build_bd_null_basis(effective_channels, user_idx)
        coeff_block = null_basis.conj().T @ np.asarray(user_block, dtype=complex)
        projected_blocks.append(null_basis @ coeff_block)
    f_bb = np.hstack(projected_blocks)
    return env.normalize_digital_precoder(f_rf, f_bb)


def _build_initial_baseline(
    env: MultiUserSimulationEnvironment,
    user_channels: np.ndarray,
    receiver_user_channels: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    train_average,
    eval_average,
    device: str,
    baseline_strategies: tuple[str, ...],
) -> BaselineInitialization:
    del device
    f_rf = env.build_analog_precoder(user_channels)
    result = optimize_fixed_analog_soft_digital_adam(
        env=env,
        user_channels=user_channels,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        train_average=train_average,
        eval_average=eval_average,
        num_iterations=0,
        f_rf=f_rf,
        baseline_strategies=baseline_strategies,
    )
    baseline_receiver_eval = env.evaluate_precoder_current_receiver_average_thp(
        user_channels=receiver_user_channels,
        f_rf=result.f_rf,
        f_bb=result.baseline_f_bb,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=eval_average,
    )
    return BaselineInitialization(
        strategy=result.baseline_strategy,
        f_rf=np.asarray(result.f_rf, dtype=complex),
        f_bb=np.asarray(result.baseline_f_bb, dtype=complex),
        soft_train_eval=result.baseline_soft_train_eval,
        receiver_eval=baseline_receiver_eval,
    )


def optimize_soft_joint_ao(
    env: MultiUserSimulationEnvironment,
    user_channels: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    train_average,
    eval_average=None,
    receiver_user_channels: np.ndarray | None = None,
    outer_iterations: int = 5,
    digital_steps: int = 80,
    analog_steps: int = 40,
    digital_lr: float = 0.03,
    analog_lr: float = 0.01,
    initial_temperature: float = 1.0,
    final_temperature: float | None = None,
    selection_temperature: float = 1.0,
    grad_clip_norm: float | None = 10.0,
    interference_penalty_weight: float = 0.05,
    user_fairness_penalty_weight: float = 0.0,
    device: str = "cpu",
    baseline_strategies: tuple[str, ...] = ("svd", "gmd"),
) -> SoftJointAOResult:
    del selection_temperature
    design_user_channels = np.asarray(user_channels, dtype=complex)
    effective_eval_average = train_average if eval_average is None else eval_average
    receiver_user_channels = (
        design_user_channels
        if receiver_user_channels is None
        else np.asarray(receiver_user_channels, dtype=complex)
    )

    baseline = _build_initial_baseline(
        env=env,
        user_channels=design_user_channels,
        receiver_user_channels=receiver_user_channels,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        train_average=train_average,
        eval_average=effective_eval_average,
        device=device,
        baseline_strategies=baseline_strategies,
    )

    current_f_rf = np.asarray(baseline.f_rf, dtype=complex)
    current_f_bb = np.asarray(baseline.f_bb, dtype=complex)
    current_train_eval = baseline.soft_train_eval
    current_receiver_eval = env.evaluate_precoder_current_receiver_average_thp(
        user_channels=receiver_user_channels,
        f_rf=current_f_rf,
        f_bb=current_f_bb,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=effective_eval_average,
    )
    best_f_rf = np.asarray(current_f_rf, dtype=complex)
    best_f_bb = np.asarray(current_f_bb, dtype=complex)
    best_train_eval = current_train_eval
    best_receiver_eval = current_receiver_eval
    best_score = _score(
        current_receiver_eval,
        interference_penalty_weight=interference_penalty_weight,
        user_fairness_penalty_weight=user_fairness_penalty_weight,
    )

    history = [best_receiver_eval.sum_rate]
    temperature_history = _temperature_schedule(
        outer_iterations=outer_iterations,
        initial_temperature=initial_temperature,
        final_temperature=final_temperature,
    )
    rng = np.random.default_rng()
    sqrt_nt = np.sqrt(float(env.num_tx_antennas))

    for temperature in temperature_history:
        fixed_bd_bases = FixedAnalogSoftTHPObjective(
            env=env,
            user_channels=design_user_channels,
            f_rf=current_f_rf,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            sample_average=train_average,
        ).null_bases
        digital_result = optimize_fixed_analog_soft_digital_adam(
            env=env,
            user_channels=design_user_channels,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            train_average=train_average,
            eval_average=effective_eval_average,
            num_iterations=digital_steps,
            learning_rate=max(float(digital_lr) * float(temperature), 1e-6),
            grad_clip_norm=grad_clip_norm,
            device=device,
            f_rf=current_f_rf,
            baseline_strategies=(baseline.strategy,),
            initial_f_bb=current_f_bb,
            initial_strategy_name=baseline.strategy,
        )
        current_f_bb = np.asarray(digital_result.optimized_f_bb, dtype=complex)
        current_train_eval = digital_result.optimized_soft_train_eval
        current_receiver_eval = env.evaluate_precoder_current_receiver_average_thp(
            user_channels=receiver_user_channels,
            f_rf=current_f_rf,
            f_bb=current_f_bb,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            sample_average=effective_eval_average,
        )
        current_score = _score(
            current_receiver_eval,
            interference_penalty_weight=interference_penalty_weight,
            user_fairness_penalty_weight=user_fairness_penalty_weight,
        )
        if current_score > best_score + 1e-12:
            best_f_rf = np.asarray(current_f_rf, dtype=complex)
            best_f_bb = np.asarray(current_f_bb, dtype=complex)
            best_train_eval = current_train_eval
            best_receiver_eval = current_receiver_eval
            best_score = current_score
        history.append(best_receiver_eval.sum_rate)

        theta = np.angle(current_f_rf * sqrt_nt)
        local_theta = np.asarray(theta, dtype=float)
        local_f_rf = np.asarray(current_f_rf, dtype=complex)
        local_f_bb = np.asarray(current_f_bb, dtype=complex)
        local_receiver_eval = current_receiver_eval
        local_score = current_score
        local_train_eval = current_train_eval

        theta_param = torch.tensor(
            local_theta,
            dtype=torch.float32 if device == "cuda" else torch.float64,
            device=torch.device(device),
            requires_grad=True,
        )
        analog_optimizer = torch.optim.Adam([theta_param], lr=max(float(analog_lr) * float(temperature), 1e-6))
        current_coeff_blocks = FixedAnalogSoftTHPObjective(
            env=env,
            user_channels=design_user_channels,
            f_rf=local_f_rf,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            sample_average=train_average,
        ).coeff_blocks_from_f_bb(local_f_bb)
        coeff_blocks_torch = [
            torch.as_tensor(np.asarray(block, dtype=np.complex128), dtype=torch.complex64 if device == "cuda" else torch.complex128, device=torch.device(device))
            for block in current_coeff_blocks
        ]

        for _ in range(int(analog_steps)):
            analog_optimizer.zero_grad(set_to_none=True)
            loss, candidate_stats, f_rf_t, f_bb_t = evaluate_soft_sic_surrogate(
                user_channels=design_user_channels,
                theta=theta_param,
                c_blocks=coeff_blocks_torch,
                sample_average=train_average,
                bits_per_symbol=bits_per_symbol,
                snr_per_stream=snr_per_stream,
                num_streams_per_user=env.num_streams_per_user,
                digital_power_constraint=env.digital_power_constraint,
                interference_penalty_weight=interference_penalty_weight,
                temperature=max(float(temperature), 1e-6),
                labeling="gray_standard",
                device=device,
                bd_bases=fixed_bd_bases,
                llr_mode="exact",
            )
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_([theta_param], max_norm=float(grad_clip_norm))
            analog_optimizer.step()
            with torch.no_grad():
                theta_param.copy_((theta_param + np.pi) % (2.0 * np.pi) - np.pi)

            candidate_f_rf = np.asarray(f_rf_t.detach().cpu(), dtype=complex)
            candidate_f_bb = np.asarray(f_bb_t.detach().cpu(), dtype=complex)
            candidate_train_eval = SoftRateAverageEvaluation(
                sum_rate=float(candidate_stats.sum_rate),
                sum_rate_std=float(candidate_stats.sum_rate_std),
                bit_error_rate=float(candidate_stats.bit_error_rate),
                user_rates=np.asarray(candidate_stats.user_rates, dtype=float),
                user_bit_error_rates=np.asarray(candidate_stats.user_bit_error_rates, dtype=float),
                offdiag_to_desired=float(candidate_stats.offdiag_to_desired),
            )
            candidate_receiver_eval = env.evaluate_precoder_current_receiver_average_thp(
                user_channels=receiver_user_channels,
                f_rf=candidate_f_rf,
                f_bb=candidate_f_bb,
                snr_per_stream=snr_per_stream,
                bits_per_symbol=bits_per_symbol,
                sample_average=effective_eval_average,
            )
            candidate_score = _score(
                candidate_receiver_eval,
                interference_penalty_weight=interference_penalty_weight,
                user_fairness_penalty_weight=user_fairness_penalty_weight,
            )
            local_theta = np.asarray(theta_param.detach().cpu(), dtype=float)
            local_f_rf = np.asarray(candidate_f_rf, dtype=complex)
            local_f_bb = np.asarray(candidate_f_bb, dtype=complex)
            local_train_eval = candidate_train_eval
            local_receiver_eval = candidate_receiver_eval
            local_score = candidate_score

        current_f_rf = local_f_rf
        current_f_bb = local_f_bb
        current_train_eval = local_train_eval
        current_receiver_eval = local_receiver_eval
        if local_score > best_score + 1e-12:
            best_f_rf = np.asarray(local_f_rf, dtype=complex)
            best_f_bb = np.asarray(local_f_bb, dtype=complex)
            best_train_eval = local_train_eval
            best_receiver_eval = local_receiver_eval
            best_score = local_score
        history.append(best_receiver_eval.sum_rate)

    return SoftJointAOResult(
        baseline_strategy=baseline.strategy,
        baseline_f_rf=np.asarray(baseline.f_rf, dtype=complex),
        baseline_f_bb=np.asarray(baseline.f_bb, dtype=complex),
        baseline_soft_train_eval=baseline.soft_train_eval,
        baseline_receiver_eval=baseline.receiver_eval,
        optimized_f_rf=np.asarray(best_f_rf, dtype=complex),
        optimized_f_bb=np.asarray(best_f_bb, dtype=complex),
        optimized_soft_train_eval=best_train_eval,
        optimized_receiver_eval=best_receiver_eval,
        history=history,
        training_temperature_history=temperature_history,
        selected_start_strategy=baseline.strategy,
        candidate_start_strategies=tuple(baseline_strategies),
    )


__all__ = [
    "BaselineInitialization",
    "SoftJointAOResult",
    "optimize_soft_joint_ao",
]
