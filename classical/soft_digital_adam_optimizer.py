from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from bicm_metrics import get_constellation
from multiuser_simulation_environment import (
    MultiUserSimulationEnvironment,
    ReceiverAverageEvaluation,
)
from sic_sample_average import MultiUserSICSampleAverage
from thp_precoding import qam_axis_order_and_spacing


def _centered_modulo_torch(x: torch.Tensor, period: float) -> torch.Tensor:
    """Map a real tensor into [-period/2, period/2)."""

    return x - torch.floor((x + period / 2.0) / period) * period


def _centered_modulo_complex_torch(x: torch.Tensor, period: float) -> torch.Tensor:
    """Apply centered modulo independently on I/Q."""

    return _centered_modulo_torch(torch.real(x), period) + 1j * _centered_modulo_torch(
        torch.imag(x),
        period,
    )


def _straight_through_centered_modulo_complex_torch(
    x: torch.Tensor,
    period: float,
) -> torch.Tensor:
    """Use the wrapped value in forward pass and identity gradient in backward pass."""

    wrapped = _centered_modulo_complex_torch(x, period)
    return x + (wrapped - x).detach()


@dataclass
class SoftRateAverageEvaluation:
    sum_rate: float
    sum_rate_std: float
    user_rates: np.ndarray


@dataclass
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


class FixedAnalogSoftTHPObjective:
    def __init__(
        self,
        env: MultiUserSimulationEnvironment,
        user_channels: np.ndarray,
        f_rf: np.ndarray,
        snr_per_stream: float,
        bits_per_symbol: int,
        sample_average: MultiUserSICSampleAverage,
        device: str = "cpu",
    ):
        """????????????"""
        self.env = env
        self.user_channels = np.asarray(user_channels, dtype=complex)
        self.f_rf = np.asarray(f_rf, dtype=complex)
        self.snr_per_stream = float(snr_per_stream)
        self.bits_per_symbol = int(bits_per_symbol)
        self.sample_average = sample_average
        self.device = torch.device(device)
        self.real_dtype = torch.float32
        self.complex_dtype = torch.complex64
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

        effective_channels_np = env.build_effective_channels(self.user_channels, self.f_rf)
        self.effective_channels = [
            torch.as_tensor(channel, dtype=self.complex_dtype, device=self.device)
            for channel in effective_channels_np
        ]
        self.null_bases_np = [
            env.build_bd_null_basis(effective_channels_np, user_idx)
            for user_idx in range(env.num_users)
        ]
        self.null_bases = [
            torch.as_tensor(basis, dtype=self.complex_dtype, device=self.device)
            for basis in self.null_bases_np
        ]
        self.f_rf_torch = torch.as_tensor(self.f_rf, dtype=self.complex_dtype, device=self.device)
        self.samples = self._prepare_samples()

    def _build_replica_tables(
        self,
        symbols_np: np.ndarray,
        bits_np: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Enumerate the periodic replicas seen by the THP modulo receiver."""

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

    def coeff_blocks_from_f_bb(self, f_bb: np.ndarray) -> list[np.ndarray]:
        """?? coeff blocks from f bb ???"""
        user_blocks = self.env.split_user_blocks(np.asarray(f_bb, dtype=complex))
        coeff_blocks = []
        for user_idx, user_block in enumerate(user_blocks):
            coeff_blocks.append(self.null_bases_np[user_idx].conj().T @ user_block)
        return coeff_blocks

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

    def build_f_bb(self, coeff_blocks: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        """?????f bb?"""
        user_blocks = [basis @ coeff for basis, coeff in zip(self.null_bases, coeff_blocks)]

        f_bb = torch.cat(user_blocks, dim=1)
        full_precoder = self.f_rf_torch @ f_bb
        power = torch.sum(torch.abs(full_precoder) ** 2).real
        scaling = torch.sqrt(
            torch.as_tensor(float(self.total_streams), dtype=self.real_dtype, device=self.device)
            / torch.clamp(power, min=1e-12)
        )
        user_blocks = [scaling * block for block in user_blocks]
        return user_blocks, scaling * f_bb

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
    ) -> torch.Tensor:
        """Estimate THP-aware bit-wise rates for all users in a repeat."""

        num_samples = equalized_received.shape[1]
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
        neg_distances = -distances.unsqueeze(-1)
        large_neg = torch.finfo(self.real_dtype).min
        llr_one = torch.logsumexp(
            neg_distances.masked_fill(~self.mask_one_tensor, large_neg),
            dim=3,
        )
        llr_zero = torch.logsumexp(
            neg_distances.masked_fill(~self.mask_zero_tensor, large_neg),
            dim=3,
        )
        llr = llr_one - llr_zero
        signed_llr = (2.0 * transmitted_bits - 1.0) * llr
        loss_sum = torch.sum(F.softplus(-signed_llr) * self.log2e, dim=(1, 2, 3))

        ideal_rate = float(self.env.num_streams_per_user * self.bits_per_symbol)
        return torch.clamp(
            torch.as_tensor(ideal_rate, dtype=self.real_dtype, device=self.device)
            - loss_sum / max(num_samples, 1),
            min=0.0,
        )

    def compute_metrics(
        self,
        coeff_blocks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """?????metrics?"""
        user_blocks, f_bb = self.build_f_bb(coeff_blocks)
        effective_stack = torch.stack(self.effective_channels, dim=0)
        user_block_stack = torch.stack(user_blocks, dim=0)
        triangular_channels = []
        rotated_blocks: list[list[torch.Tensor]] = []
        for user_idx in range(self.env.num_users):
            desired_block = effective_stack[user_idx] @ user_block_stack[user_idx]
            q_factor, triangular_channel = self._qr_factors_with_positive_diagonal(desired_block)
            rx_blocks = torch.matmul(
                effective_stack[user_idx].unsqueeze(0),
                user_block_stack,
            )
            rotated_user = torch.matmul(
                q_factor.conj().transpose(0, 1).unsqueeze(0),
                rx_blocks,
            )
            triangular_channels.append(triangular_channel)
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
            )
            repeat_sum = torch.sum(user_rates)
            user_rate_accum = user_rate_accum + user_rates
            repeat_rates.append(repeat_sum)

        repeat_rates_t = torch.stack(repeat_rates)
        return (
            torch.mean(repeat_rates_t),
            torch.std(repeat_rates_t, unbiased=False),
            user_rate_accum / self.sample_average.num_repeats,
            f_bb,
        )

    def evaluate_coeff_blocks(self, coeff_blocks: list[torch.Tensor]) -> tuple[SoftRateAverageEvaluation, np.ndarray]:
        """?????coeff blocks?"""
        with torch.no_grad():
            mean_rate, std_rate, user_rates, f_bb = self.compute_metrics(coeff_blocks)
        return (
            SoftRateAverageEvaluation(
                sum_rate=float(mean_rate.item()),
                sum_rate_std=float(std_rate.item()),
                user_rates=user_rates.detach().cpu().numpy(),
            ),
            f_bb.detach().cpu().numpy(),
        )

    def evaluate_f_bb(self, f_bb: np.ndarray) -> SoftRateAverageEvaluation:
        """?????f bb?"""
        coeff_blocks_np = self.coeff_blocks_from_f_bb(f_bb)
        coeff_blocks = [
            torch.as_tensor(coeff, dtype=self.complex_dtype, device=self.device)
            for coeff in coeff_blocks_np
        ]
        evaluation, _ = self.evaluate_coeff_blocks(coeff_blocks)
        return evaluation

def _select_best_structured_baseline(
    env: MultiUserSimulationEnvironment,
    user_channels: np.ndarray,
    f_rf: np.ndarray,
    snr_per_stream: float,
    objective: FixedAnalogSoftTHPObjective,
    strategies: tuple[str, ...] = ("svd", "gmd"),
) -> tuple[str, np.ndarray, SoftRateAverageEvaluation]:
    """?? select best structured baseline ???"""
    best_strategy = ""
    best_f_bb = None
    best_eval = None
    best_rate = -np.inf

    for strategy in strategies:
        f_bb, _, _ = env.build_structured_digital_precoder(
            user_channels=user_channels,
            f_rf=f_rf,
            snr_per_stream=snr_per_stream,
            strategy=strategy,
        )
        evaluation = objective.evaluate_f_bb(f_bb)
        if evaluation.sum_rate > best_rate + 1e-12:
            best_strategy = strategy
            best_rate = evaluation.sum_rate
            best_f_bb = f_bb
            best_eval = evaluation

    if best_f_bb is None or best_eval is None:
        raise RuntimeError("Failed to construct a fixed-analog structured baseline.")
    return best_strategy, best_f_bb, best_eval


def optimize_fixed_analog_soft_digital_adam(
    env: MultiUserSimulationEnvironment,
    user_channels: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    train_average: MultiUserSICSampleAverage,
    eval_average: MultiUserSICSampleAverage | None = None,
    num_iterations: int = 100,
    learning_rate: float = 0.03,
    grad_clip_norm: float | None = 10.0,
    weight_decay: float = 0.0,
    device: str = "cpu",
    f_rf: np.ndarray | None = None,
    baseline_strategies: tuple[str, ...] = ("svd", "gmd"),
) -> FixedAnalogSoftAdamResult:
    """?????fixed analog soft digital adam?"""
    effective_eval_average = train_average if eval_average is None else eval_average
    if f_rf is None:
        f_rf = env.build_analog_precoder(user_channels)
    else:
        f_rf = np.asarray(f_rf, dtype=complex)
    train_objective = FixedAnalogSoftTHPObjective(
        env=env,
        user_channels=user_channels,
        f_rf=f_rf,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=train_average,
        device=device,
    )
    eval_objective = FixedAnalogSoftTHPObjective(
        env=env,
        user_channels=user_channels,
        f_rf=f_rf,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=effective_eval_average,
        device=device,
    )

    baseline_strategy, baseline_f_bb, baseline_soft_train_eval = _select_best_structured_baseline(
        env=env,
        user_channels=user_channels,
        f_rf=f_rf,
        snr_per_stream=snr_per_stream,
        objective=train_objective,
        strategies=baseline_strategies,
    )
    baseline_soft_eval_eval = eval_objective.evaluate_f_bb(baseline_f_bb)

    coeff_init = train_objective.coeff_blocks_from_f_bb(baseline_f_bb)
    coeff_params = [
        torch.nn.Parameter(
            torch.as_tensor(coeff, dtype=train_objective.complex_dtype, device=train_objective.device)
        )
        for coeff in coeff_init
    ]
    optimizer = torch.optim.Adam(coeff_params, lr=learning_rate, weight_decay=weight_decay)

    best_rate = baseline_soft_train_eval.sum_rate
    best_coeff = [param.detach().clone() for param in coeff_params]
    history = [baseline_soft_train_eval.sum_rate]

    for _ in range(num_iterations):
        optimizer.zero_grad(set_to_none=True)
        mean_rate, _, _, _ = train_objective.compute_metrics(coeff_params)
        loss = -mean_rate
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(coeff_params, grad_clip_norm)
        optimizer.step()

        current_eval, _ = train_objective.evaluate_coeff_blocks(coeff_params)
        history.append(current_eval.sum_rate)
        if current_eval.sum_rate > best_rate + 1e-12:
            best_rate = current_eval.sum_rate
            best_coeff = [param.detach().clone() for param in coeff_params]

    optimized_soft_train_eval, optimized_f_bb = train_objective.evaluate_coeff_blocks(best_coeff)
    optimized_soft_eval_eval = eval_objective.evaluate_f_bb(optimized_f_bb)

    baseline_receiver_eval = env.evaluate_precoder_current_receiver_average_thp(
        user_channels=user_channels,
        f_rf=f_rf,
        f_bb=baseline_f_bb,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=effective_eval_average,
    )
    optimized_receiver_eval = env.evaluate_precoder_current_receiver_average_thp(
        user_channels=user_channels,
        f_rf=f_rf,
        f_bb=optimized_f_bb,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=effective_eval_average,
    )

    return FixedAnalogSoftAdamResult(
        baseline_strategy=baseline_strategy,
        f_rf=f_rf,
        baseline_f_bb=baseline_f_bb,
        optimized_f_bb=optimized_f_bb,
        baseline_soft_train_eval=baseline_soft_train_eval,
        optimized_soft_train_eval=optimized_soft_train_eval,
        baseline_soft_eval_eval=baseline_soft_eval_eval,
        optimized_soft_eval_eval=optimized_soft_eval_eval,
        baseline_receiver_eval=baseline_receiver_eval,
        optimized_receiver_eval=optimized_receiver_eval,
        history=history,
    )
