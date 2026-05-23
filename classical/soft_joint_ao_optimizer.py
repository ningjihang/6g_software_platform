from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from multiuser_simulation_environment import (
    MultiUserSimulationEnvironment,
    ReceiverAverageEvaluation,
)
from sic_sample_average import MultiUserSICSampleAverage
from soft_digital_adam_optimizer import (
    FixedAnalogSoftTHPObjective,
    _select_best_structured_baseline,
    SoftRateAverageEvaluation,
    _straight_through_centered_modulo_complex_torch,
)
from bicm_metrics import get_constellation
from thp_precoding import qam_axis_order_and_spacing


@dataclass
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


@dataclass(frozen=True)
class BaselineInitialization:
    strategy: str
    f_rf: np.ndarray
    f_bb: np.ndarray
    soft_train_eval: SoftRateAverageEvaluation
    receiver_eval: ReceiverAverageEvaluation


class JointSoftTHPObjective:
    def __init__(
        self,
        env: MultiUserSimulationEnvironment,
        user_channels: np.ndarray,
        snr_per_stream: float,
        bits_per_symbol: int,
        sample_average: MultiUserSICSampleAverage,
        device: str = "cpu",
    ):
        """????????????"""
        self.env = env
        self.user_channels = np.asarray(user_channels, dtype=complex)
        self.snr_per_stream = float(snr_per_stream)
        self.bits_per_symbol = int(bits_per_symbol)
        self.sample_average = sample_average
        self.device = torch.device(device)
        self.real_dtype = torch.float32
        self.complex_dtype = torch.complex64
        self.eye_rf = torch.eye(env.num_rf_chains, dtype=self.complex_dtype, device=self.device)
        self.channel_tensors = [
            torch.as_tensor(channel, dtype=self.complex_dtype, device=self.device)
            for channel in self.user_channels
        ]
        self.sqrt_nt = float(np.sqrt(env.num_tx_antennas))
        self.total_streams = env.total_streams
        self.sqrt_snr = float(np.sqrt(self.snr_per_stream))
        self.log2e = 1.0 / np.log(2.0)
        self.replica_radius = 1
        m_axis, spacing = qam_axis_order_and_spacing(bits_per_symbol)
        self.period = float(m_axis * spacing)

        symbols_np, bits_np = get_constellation(bits_per_symbol)
        self.constellation = torch.as_tensor(symbols_np, dtype=self.complex_dtype, device=self.device)
        self.bit_table = torch.as_tensor(bits_np, dtype=self.real_dtype, device=self.device)
        self.replica_constellation, self.replica_bit_table = self._build_replica_tables(
            symbols_np=symbols_np,
            bits_np=bits_np,
        )
        self.mask_one = []
        self.mask_zero = []
        for bit_idx in range(bits_per_symbol):
            mask = self.replica_bit_table[:, bit_idx] > 0.5
            self.mask_one.append(mask)
            self.mask_zero.append(~mask)
        self.mask_one_tensor = torch.stack(self.mask_one, dim=0).transpose(0, 1).reshape(
            1, 1, 1, -1, self.bits_per_symbol
        )
        self.mask_zero_tensor = torch.stack(self.mask_zero, dim=0).transpose(0, 1).reshape(
            1, 1, 1, -1, self.bits_per_symbol
        )
        self.samples = self._prepare_samples()

    def _build_replica_tables(
        self,
        symbols_np: np.ndarray,
        bits_np: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Enumerate modulo replicas for the THP receive metric."""

        replica_symbols = []
        replica_bits = []
        for real_shift in range(-self.replica_radius, self.replica_radius + 1):
            for imag_shift in range(-self.replica_radius, self.replica_radius + 1):
                shift = self.period * (real_shift + 1j * imag_shift)
                replica_symbols.append(symbols_np + shift)
                replica_bits.append(bits_np)
        return (
            torch.as_tensor(
                np.concatenate(replica_symbols),
                dtype=self.complex_dtype,
                device=self.device,
            ),
            torch.as_tensor(
                np.concatenate(replica_bits, axis=0),
                dtype=self.real_dtype,
                device=self.device,
            ),
        )

    def _prepare_samples(self) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], ...]:
        """?? prepare samples ???"""
        prepared = []
        for batch_group in self.sample_average.batches:
            user_batches = []
            for batch in batch_group:
                symbol_indices = torch.as_tensor(batch.symbol_indices, dtype=torch.long, device=self.device)
                noise = torch.as_tensor(batch.noise, dtype=self.complex_dtype, device=self.device)
                symbols = self.constellation[symbol_indices]
                bits = self.bit_table[symbol_indices]
                user_batches.append((symbols, bits, noise))
            prepared.append(tuple(user_batches))
        return tuple(prepared)

    def _qr_factors_with_positive_diagonal(
        self,
        channel_block: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """?? qr factors with positive diagonal ???"""
        q_factor, r_factor = torch.linalg.qr(channel_block, mode="reduced")
        q_factor = q_factor[:, : self.env.num_streams_per_user]
        r_factor = r_factor[: self.env.num_streams_per_user, : self.env.num_streams_per_user]
        diagonal = torch.diagonal(r_factor)
        phase = torch.exp(-1j * torch.angle(diagonal))
        phase_matrix = torch.diag(phase.to(dtype=self.complex_dtype))
        q_aligned = q_factor @ phase_matrix.conj().transpose(0, 1)
        r_aligned = phase_matrix @ r_factor
        return q_aligned, r_aligned

    def _thp_precoded_symbols_batch(
        self,
        triangular_channels: torch.Tensor,
        original_symbols: torch.Tensor,
    ) -> torch.Tensor:
        """Run batched per-user THP recursion with a straight-through modulo surrogate."""

        diagonal = torch.diagonal(triangular_channels, dim1=1, dim2=2)
        normalized_upper = triangular_channels / diagonal.unsqueeze(-1)
        precoded_by_stream: list[torch.Tensor | None] = [None] * self.env.num_streams_per_user

        for stream_idx in range(self.env.num_streams_per_user - 1, -1, -1):
            if stream_idx + 1 < self.env.num_streams_per_user:
                coeff = normalized_upper[:, stream_idx, stream_idx + 1 :]
                future = torch.stack(
                    [stream for stream in precoded_by_stream[stream_idx + 1 :] if stream is not None],
                    dim=2,
                )
                interference = torch.einsum("kl,ksl->ks", coeff, future)
                pre_modulo = original_symbols[:, :, stream_idx] - interference
            else:
                pre_modulo = original_symbols[:, :, stream_idx]
            precoded_by_stream[stream_idx] = _straight_through_centered_modulo_complex_torch(
                pre_modulo,
                self.period,
            )

        return torch.stack([stream for stream in precoded_by_stream if stream is not None], dim=2)

    def _equalize_received_batch(
        self,
        received: torch.Tensor,
        triangular_channels: torch.Tensor,
    ) -> torch.Tensor:
        """Apply THP receive-side equalization and modulo folding for all users."""

        diagonal = torch.diagonal(triangular_channels, dim1=1, dim2=2)
        equalized = received / max(self.sqrt_snr, 1e-12)
        equalized = equalized / diagonal[:, None, :]
        return _straight_through_centered_modulo_complex_torch(equalized, self.period)

    def _user_rates_for_repeat_batch(
        self,
        triangular_channels: torch.Tensor,
        transmitted_bits: torch.Tensor,
        equalized_received: torch.Tensor,
        soft_temperature: float = 1.0,
    ) -> torch.Tensor:
        """Estimate THP-aware bit-wise rates for all users in a repeat."""

        num_samples = equalized_received.shape[1]
        temperature = max(float(soft_temperature), 1e-4)
        diagonal = torch.diagonal(triangular_channels, dim1=1, dim2=2)
        scaled_codewords = (
            self.sqrt_snr
            * diagonal[:, :, None]
            * self.replica_constellation[None, None, :]
        )
        received_scaled = self.sqrt_snr * equalized_received * diagonal[:, None, :]
        distances = torch.abs(
            received_scaled[:, :, :, None] - scaled_codewords[:, None, :, :]
        ) ** 2
        scaled_neg_distances = -(distances / temperature).unsqueeze(-1)
        large_neg = torch.finfo(self.real_dtype).min
        llr_one = torch.logsumexp(
            scaled_neg_distances.masked_fill(~self.mask_one_tensor, large_neg),
            dim=3,
        )
        llr_zero = torch.logsumexp(
            scaled_neg_distances.masked_fill(~self.mask_zero_tensor, large_neg),
            dim=3,
        )
        llr = temperature * (llr_one - llr_zero)
        signed_llr = (2.0 * transmitted_bits - 1.0) * llr
        loss_sum = torch.sum(F.softplus(-signed_llr) * self.log2e, dim=(1, 2, 3))

        ideal_rate = float(self.env.num_streams_per_user * self.bits_per_symbol)
        return torch.clamp(
            torch.as_tensor(ideal_rate, dtype=self.real_dtype, device=self.device)
            - loss_sum / max(num_samples, 1),
            min=0.0,
        )

    def projector_blocks(
        self,
        f_rf: torch.Tensor,
        digital_blocks: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        """?? projector blocks ???"""
        effective_channels = [channel @ f_rf for channel in self.channel_tensors]
        projected_blocks = []
        for user_idx in range(self.env.num_users):
            if self.env.num_users == 1:
                projector = self.eye_rf
            else:
                interference_stack = torch.cat(
                    [effective_channels[idx] for idx in range(self.env.num_users) if idx != user_idx],
                    dim=0,
                )
                projector = self.eye_rf - torch.linalg.pinv(interference_stack) @ interference_stack
            projected_blocks.append(projector @ digital_blocks[user_idx])

        f_bb = torch.cat(projected_blocks, dim=1)
        full_precoder = f_rf @ f_bb
        power = torch.sum(torch.abs(full_precoder) ** 2).real
        alpha = torch.sqrt(
            torch.as_tensor(float(self.env.total_streams), dtype=self.real_dtype, device=self.device)
            / torch.clamp(power, min=1e-12)
        )
        # Apply the global alpha before any downstream channel construction so
        # both the desired blocks and the interference penalty use the same
        # total-power-normalized precoder.
        projected_blocks = [alpha * block for block in projected_blocks]
        f_bb = alpha * f_bb
        return projected_blocks, f_bb, effective_channels

    def build_precoder(
        self,
        theta: torch.Tensor,
        digital_blocks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """?????precoder?"""
        f_rf = torch.exp(1j * theta) / self.sqrt_nt
        projected_blocks, f_bb, _ = self.projector_blocks(f_rf, digital_blocks)
        return f_rf, projected_blocks, f_bb

    def compute_metrics(
        self,
        theta: torch.Tensor,
        digital_blocks: list[torch.Tensor],
        soft_temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """?????metrics?"""
        f_rf, projected_blocks, f_bb = self.build_precoder(theta, digital_blocks)
        effective_channels = [channel @ f_rf for channel in self.channel_tensors]
        effective_stack = torch.stack(effective_channels, dim=0)
        projected_stack = torch.stack(projected_blocks, dim=0)
        triangular_channels = []
        rotated_blocks: list[list[torch.Tensor]] = []
        desired_power = torch.zeros((), dtype=self.real_dtype, device=self.device)
        offdiag_power = torch.zeros((), dtype=self.real_dtype, device=self.device)
        for user_idx in range(self.env.num_users):
            desired_block = effective_stack[user_idx] @ projected_stack[user_idx]
            q_factor, triangular_channel = self._qr_factors_with_positive_diagonal(desired_block)
            triangular_channels.append(triangular_channel)
            rx_blocks = torch.matmul(
                effective_stack[user_idx].unsqueeze(0),
                projected_stack,
            )
            block_powers = torch.sum(torch.abs(rx_blocks) ** 2, dim=(1, 2)).real
            desired_power = desired_power + block_powers[user_idx]
            offdiag_power = offdiag_power + (torch.sum(block_powers) - block_powers[user_idx])
            rotated_user = torch.matmul(
                q_factor.conj().transpose(0, 1).unsqueeze(0),
                rx_blocks,
            )
            rotated_blocks.append([rotated_user[tx_user] for tx_user in range(self.env.num_users)])

        triangular_channels_t = torch.stack(triangular_channels, dim=0)
        rotated_blocks_t = torch.stack(
            [torch.stack(blocks, dim=0) for blocks in rotated_blocks],
            dim=0,
        )
        repeat_rates = []
        user_rate_accum = torch.zeros(self.env.num_users, dtype=self.real_dtype, device=self.device)
        for batch_group in self.samples:
            original_symbols_all = torch.stack([batch[0] for batch in batch_group], dim=0)
            transmitted_bits_all = torch.stack([batch[1] for batch in batch_group], dim=0)
            noise_all = torch.stack([batch[2] for batch in batch_group], dim=0)
            transmitted_symbols_all = self._thp_precoded_symbols_batch(
                triangular_channels_t,
                original_symbols_all,
            )
            received_all = noise_all + self.sqrt_snr * torch.einsum(
                "tsj,rtji->rsi",
                transmitted_symbols_all,
                rotated_blocks_t.transpose(-1, -2),
            )
            equalized_all = self._equalize_received_batch(
                received=received_all,
                triangular_channels=triangular_channels_t,
            )
            user_rates = self._user_rates_for_repeat_batch(
                triangular_channels=triangular_channels_t,
                transmitted_bits=transmitted_bits_all,
                equalized_received=equalized_all,
                soft_temperature=soft_temperature,
            )
            repeat_sum = torch.sum(user_rates)
            user_rate_accum = user_rate_accum + user_rates
            repeat_rates.append(repeat_sum)

        repeat_rates_t = torch.stack(repeat_rates)
        mean_rate = torch.mean(repeat_rates_t)
        std_rate = torch.std(repeat_rates_t, unbiased=False)
        user_rates = user_rate_accum / self.sample_average.num_repeats
        interference_penalty = offdiag_power / torch.clamp(desired_power, min=1e-12)
        mean_user_rate = torch.mean(user_rates)
        min_user_rate = torch.min(user_rates)
        user_fairness_penalty = (
            (mean_user_rate - min_user_rate)
            / torch.clamp(mean_user_rate + min_user_rate, min=1e-6)
        ) ** 2
        return mean_rate, std_rate, user_rates, f_rf, f_bb, interference_penalty, user_fairness_penalty

    def evaluate(
        self,
        theta: torch.Tensor,
        digital_blocks: list[torch.Tensor],
        soft_temperature: float = 1.0,
    ) -> tuple[SoftRateAverageEvaluation, np.ndarray, np.ndarray]:
        """?? evaluate ???"""
        with torch.no_grad():
            mean_rate, std_rate, user_rates, f_rf, f_bb, _, _ = self.compute_metrics(
                theta,
                digital_blocks,
                soft_temperature=soft_temperature,
            )
        return (
            SoftRateAverageEvaluation(
                sum_rate=float(mean_rate.item()),
                sum_rate_std=float(std_rate.item()),
                user_rates=user_rates.detach().cpu().numpy(),
            ),
            f_rf.detach().cpu().numpy(),
            f_bb.detach().cpu().numpy(),
        )

def _build_initial_baseline(
    env: MultiUserSimulationEnvironment,
    user_channels: np.ndarray,
    receiver_user_channels: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    train_average: MultiUserSICSampleAverage,
    eval_average: MultiUserSICSampleAverage,
    device: str,
    baseline_strategies: tuple[str, ...],
) -> BaselineInitialization:
    """?? build initial baseline ???"""
    f_rf = env.build_analog_precoder(user_channels)
    train_objective = FixedAnalogSoftTHPObjective(
        env=env,
        user_channels=user_channels,
        f_rf=f_rf,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=train_average,
        device=device,
    )
    strategy, f_bb, train_eval = _select_best_structured_baseline(
        env=env,
        user_channels=user_channels,
        f_rf=f_rf,
        snr_per_stream=snr_per_stream,
        objective=train_objective,
        strategies=baseline_strategies,
    )
    receiver_eval = env.evaluate_precoder_current_receiver_average_thp(
        user_channels=receiver_user_channels,
        f_rf=f_rf,
        f_bb=f_bb,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=eval_average,
    )
    return BaselineInitialization(
        strategy=strategy,
        f_rf=f_rf,
        f_bb=f_bb,
        soft_train_eval=train_eval,
        receiver_eval=receiver_eval,
    )


def _build_explicit_initialization(
    env: MultiUserSimulationEnvironment,
    user_channels: np.ndarray,
    receiver_user_channels: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    train_average: MultiUserSICSampleAverage,
    eval_average: MultiUserSICSampleAverage,
    device: str,
    strategy: str,
    f_rf: np.ndarray,
    f_bb: np.ndarray,
) -> BaselineInitialization:
    train_objective = FixedAnalogSoftTHPObjective(
        env=env,
        user_channels=user_channels,
        f_rf=f_rf,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=train_average,
        device=device,
    )
    train_eval = train_objective.evaluate_f_bb(f_bb)
    receiver_eval = env.evaluate_precoder_current_receiver_average_thp(
        user_channels=receiver_user_channels,
        f_rf=f_rf,
        f_bb=f_bb,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=eval_average,
    )
    return BaselineInitialization(
        strategy=str(strategy),
        f_rf=np.asarray(f_rf, dtype=complex),
        f_bb=np.asarray(f_bb, dtype=complex),
        soft_train_eval=train_eval,
        receiver_eval=receiver_eval,
    )


def _build_temperature_schedule(
    outer_iterations: int,
    initial_temperature: float,
    final_temperature: float,
) -> list[float]:
    """?? build temperature schedule ???"""
    if outer_iterations <= 0:
        return []
    if initial_temperature <= 0.0 or final_temperature <= 0.0:
        raise ValueError("Soft receiver temperatures must be positive.")
    if outer_iterations == 1:
        return [float(initial_temperature)]
    if np.isclose(initial_temperature, final_temperature):
        return [float(initial_temperature)] * outer_iterations
    return [
        float(value)
        for value in np.geomspace(initial_temperature, final_temperature, num=outer_iterations)
    ]


def _run_joint_ao_from_initialization(
    env: MultiUserSimulationEnvironment,
    objective: JointSoftTHPObjective,
    receiver_user_channels: np.ndarray,
    eval_average: MultiUserSICSampleAverage,
    snr_per_stream: float,
    bits_per_symbol: int,
    baseline_f_rf: np.ndarray,
    baseline_f_bb: np.ndarray,
    baseline_soft_train_eval: SoftRateAverageEvaluation,
    outer_iterations: int,
    digital_steps: int,
    analog_steps: int,
    digital_lr: float,
    analog_lr: float,
    initial_temperature: float,
    final_temperature: float,
    selection_temperature: float,
    grad_clip_norm: float | None,
    interference_penalty_weight: float,
    user_fairness_penalty_weight: float,
) -> tuple[
    SoftRateAverageEvaluation,
    np.ndarray,
    np.ndarray,
    ReceiverAverageEvaluation,
    list[float],
    list[float],
]:
    """?? run joint ao from initialization ???"""
    theta_param = torch.nn.Parameter(
        torch.as_tensor(
            np.angle(baseline_f_rf * np.sqrt(env.num_tx_antennas)),
            dtype=torch.float32,
            device=objective.device,
        )
    )
    digital_params = [
        torch.nn.Parameter(
            torch.as_tensor(block, dtype=torch.complex64, device=objective.device)
        )
        for block in env.split_user_blocks(baseline_f_bb)
    ]

    best_theta = theta_param.detach().clone()
    best_digital = [param.detach().clone() for param in digital_params]
    with torch.no_grad():
        baseline_mean_rate, _, _, _, _, baseline_interference, baseline_user_fairness = objective.compute_metrics(
            theta_param,
            digital_params,
            soft_temperature=selection_temperature,
        )
        if user_fairness_penalty_weight > 0.0:
            baseline_selection_score = (
                baseline_mean_rate
                - interference_penalty_weight * baseline_interference
                - user_fairness_penalty_weight * baseline_user_fairness
            )
        else:
            baseline_selection_score = baseline_mean_rate
    best_train_rate = float(baseline_mean_rate.item())
    best_selection_score = float(baseline_selection_score.item())
    history = [best_train_rate]
    temperature_history = _build_temperature_schedule(
        outer_iterations=outer_iterations,
        initial_temperature=initial_temperature,
        final_temperature=final_temperature,
    )

    for train_temperature in temperature_history:
        optimizer_d = torch.optim.Adam(digital_params, lr=digital_lr)
        for _ in range(digital_steps):
            optimizer_d.zero_grad(set_to_none=True)
            mean_rate, _, _, _, _, interference_penalty, user_fairness_penalty = objective.compute_metrics(
                theta_param,
                digital_params,
                soft_temperature=train_temperature,
            )
            loss = (
                -mean_rate
                + interference_penalty_weight * interference_penalty
                + user_fairness_penalty_weight * user_fairness_penalty
            )
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(digital_params, grad_clip_norm)
            optimizer_d.step()

        optimizer_theta = torch.optim.Adam([theta_param], lr=analog_lr)
        for _ in range(analog_steps):
            optimizer_theta.zero_grad(set_to_none=True)
            mean_rate, _, _, _, _, interference_penalty, user_fairness_penalty = objective.compute_metrics(
                theta_param,
                digital_params,
                soft_temperature=train_temperature,
            )
            loss = (
                -mean_rate
                + interference_penalty_weight * interference_penalty
                + user_fairness_penalty_weight * user_fairness_penalty
            )
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_([theta_param], grad_clip_norm)
            optimizer_theta.step()
            with torch.no_grad():
                theta_param.copy_((theta_param + np.pi) % (2.0 * np.pi) - np.pi)

        with torch.no_grad():
            current_mean_rate, _, _, _, _, current_interference, current_user_fairness = objective.compute_metrics(
                theta_param,
                digital_params,
                soft_temperature=selection_temperature,
            )
            if user_fairness_penalty_weight > 0.0:
                current_selection_score = (
                    current_mean_rate
                    - interference_penalty_weight * current_interference
                    - user_fairness_penalty_weight * current_user_fairness
                )
            else:
                current_selection_score = current_mean_rate
        current_train_rate = float(current_mean_rate.item())
        current_score = float(current_selection_score.item())
        history.append(current_train_rate)
        if current_score > best_selection_score + 1e-12:
            best_train_rate = current_train_rate
            best_selection_score = current_score
            best_theta = theta_param.detach().clone()
            best_digital = [param.detach().clone() for param in digital_params]

    optimized_soft_train_eval, optimized_f_rf, optimized_f_bb = objective.evaluate(
        best_theta,
        best_digital,
        soft_temperature=selection_temperature,
    )
    optimized_receiver_eval = env.evaluate_precoder_current_receiver_average_thp(
        user_channels=receiver_user_channels,
        f_rf=optimized_f_rf,
        f_bb=optimized_f_bb,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=eval_average,
    )
    return (
        optimized_soft_train_eval,
        optimized_f_rf,
        optimized_f_bb,
        optimized_receiver_eval,
        history,
        temperature_history,
    )


def optimize_soft_joint_ao(
    env: MultiUserSimulationEnvironment,
    user_channels: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    train_average: MultiUserSICSampleAverage,
    eval_average: MultiUserSICSampleAverage | None = None,
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
    initial_strategy: str | None = None,
    initial_f_rf: np.ndarray | None = None,
    initial_f_bb: np.ndarray | None = None,
) -> SoftJointAOResult:
    """?????soft joint ao?"""
    if final_temperature is None:
        final_temperature = initial_temperature
    effective_eval_average = train_average if eval_average is None else eval_average
    effective_receiver_channels = (
        np.asarray(user_channels, dtype=complex)
        if receiver_user_channels is None
        else np.asarray(receiver_user_channels, dtype=complex)
    )

    if initial_strategy is not None and initial_f_rf is not None and initial_f_bb is not None:
        baseline = _build_explicit_initialization(
            env=env,
            user_channels=user_channels,
            receiver_user_channels=effective_receiver_channels,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            train_average=train_average,
            eval_average=effective_eval_average,
            device=device,
            strategy=initial_strategy,
            f_rf=initial_f_rf,
            f_bb=initial_f_bb,
        )
    else:
        baseline = _build_initial_baseline(
            env=env,
            user_channels=user_channels,
            receiver_user_channels=effective_receiver_channels,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            train_average=train_average,
            eval_average=effective_eval_average,
            device=device,
            baseline_strategies=baseline_strategies,
        )
    objective = JointSoftTHPObjective(
        env=env,
        user_channels=user_channels,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=train_average,
        device=device,
    )
    baseline_strategy = baseline.strategy
    baseline_f_rf = baseline.f_rf
    baseline_f_bb = baseline.f_bb
    baseline_soft_train_eval = baseline.soft_train_eval
    baseline_receiver_eval = baseline.receiver_eval

    (
        optimized_soft_train_eval,
        optimized_f_rf,
        optimized_f_bb,
        optimized_receiver_eval,
        history,
        temperature_history,
    ) = _run_joint_ao_from_initialization(
        env=env,
        objective=objective,
        receiver_user_channels=effective_receiver_channels,
        eval_average=effective_eval_average,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        baseline_f_rf=baseline_f_rf,
        baseline_f_bb=baseline_f_bb,
        baseline_soft_train_eval=baseline_soft_train_eval,
        outer_iterations=outer_iterations,
        digital_steps=digital_steps,
        analog_steps=analog_steps,
        digital_lr=digital_lr,
        analog_lr=analog_lr,
        initial_temperature=initial_temperature,
        final_temperature=final_temperature,
        selection_temperature=selection_temperature,
        grad_clip_norm=grad_clip_norm,
        interference_penalty_weight=interference_penalty_weight,
        user_fairness_penalty_weight=user_fairness_penalty_weight,
    )

    return SoftJointAOResult(
        baseline_strategy=baseline_strategy,
        baseline_f_rf=baseline_f_rf,
        baseline_f_bb=baseline_f_bb,
        baseline_soft_train_eval=baseline_soft_train_eval,
        baseline_receiver_eval=baseline_receiver_eval,
        optimized_f_rf=optimized_f_rf,
        optimized_f_bb=optimized_f_bb,
        optimized_soft_train_eval=optimized_soft_train_eval,
        optimized_receiver_eval=optimized_receiver_eval,
        history=history,
        training_temperature_history=temperature_history,
        selected_start_strategy=baseline_strategy,
        candidate_start_strategies=(baseline_strategy,),
    )
