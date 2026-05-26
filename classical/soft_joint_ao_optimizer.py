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
    return x - torch.floor((x + period / 2.0) / period) * period


def _centered_modulo_complex_torch(x: torch.Tensor, period: float) -> torch.Tensor:
    return _centered_modulo_torch(torch.real(x), period) + 1j * _centered_modulo_torch(
        torch.imag(x),
        period,
    )


def _straight_through_centered_modulo_complex_torch(
    x: torch.Tensor,
    period: float,
) -> torch.Tensor:
    wrapped = _centered_modulo_complex_torch(x, period)
    return x + (wrapped - x).detach()


@dataclass
class SoftRateAverageEvaluation:
    sum_rate: float
    sum_rate_std: float
    user_rates: np.ndarray


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
    debug_rows: list[dict[str, float]]


@dataclass(frozen=True)
class BaselineInitialization:
    strategy: str
    receiver_mode: str
    f_rf: np.ndarray
    f_bb: np.ndarray
    soft_train_eval: SoftRateAverageEvaluation
    receiver_eval: ReceiverAverageEvaluation


@dataclass(frozen=True)
class FrozenAnalogState:
    projectors: tuple[torch.Tensor, ...]
    q_chains: tuple[torch.Tensor, ...]
    noise_cholesky: tuple[torch.Tensor, ...]
    noise_std: tuple[torch.Tensor, ...]


class JointFixedReceiverSoftObjective:
    def __init__(
        self,
        env: MultiUserSimulationEnvironment,
        user_channels: np.ndarray,
        snr_per_stream: float,
        bits_per_symbol: int,
        sample_average: MultiUserSICSampleAverage,
        q_chains: list[np.ndarray],
        strategy: str,
        device: str = "cpu",
        svd_receiver_mode: str = "parallel",
    ):
        self.env = env
        self.user_channels = np.asarray(user_channels, dtype=complex)
        self.snr_per_stream = float(snr_per_stream)
        self.bits_per_symbol = int(bits_per_symbol)
        self.sample_average = sample_average
        self.strategy = str(strategy).lower()
        self.svd_receiver_mode = str(svd_receiver_mode).lower()
        self.device = torch.device(device)
        self.real_dtype = torch.float32
        self.complex_dtype = torch.complex64
        self.sqrt_nt = float(np.sqrt(env.num_tx_antennas))
        self.sqrt_snr = float(np.sqrt(self.snr_per_stream))
        self.log2e = 1.0 / np.log(2.0)
        self.eye_rf = torch.eye(env.num_rf_chains, dtype=self.complex_dtype, device=self.device)
        self.channel_tensors = [
            torch.as_tensor(channel, dtype=self.complex_dtype, device=self.device)
            for channel in self.user_channels
        ]

        symbols_np, bits_np = get_constellation(bits_per_symbol)
        self.constellation = torch.as_tensor(symbols_np, dtype=self.complex_dtype, device=self.device)
        self.bit_table = torch.as_tensor(bits_np, dtype=self.real_dtype, device=self.device)
        self.mask_one_tensor, self.mask_zero_tensor = self._build_bit_masks(self.bit_table)

        self.replica_radius = 1
        m_axis, spacing = qam_axis_order_and_spacing(bits_per_symbol)
        self.period = float(m_axis * spacing)
        self.replica_constellation, self.replica_bit_table = self._build_replica_tables(
            symbols_np=symbols_np,
            bits_np=bits_np,
        )
        self.replica_mask_one_tensor, self.replica_mask_zero_tensor = self._build_bit_masks(
            self.replica_bit_table
        )

        self.q_chains = [
            torch.as_tensor(chain, dtype=self.complex_dtype, device=self.device)
            for chain in q_chains
        ]
        self.samples = self._prepare_samples()

    def _build_replica_tables(
        self,
        symbols_np: np.ndarray,
        bits_np: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _build_bit_masks(self, bit_table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_one = []
        mask_zero = []
        for bit_idx in range(self.bits_per_symbol):
            current_mask = bit_table[:, bit_idx] > 0.5
            mask_one.append(current_mask)
            mask_zero.append(~current_mask)
        return (
            torch.stack(mask_one, dim=0).transpose(0, 1).reshape(1, 1, 1, -1, self.bits_per_symbol),
            torch.stack(mask_zero, dim=0).transpose(0, 1).reshape(1, 1, 1, -1, self.bits_per_symbol),
        )

    def _prepare_samples(self) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], ...]:
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
        q_factor, r_factor = torch.linalg.qr(channel_block, mode="reduced")
        q_factor = q_factor[:, : self.env.num_streams_per_user]
        r_factor = r_factor[: self.env.num_streams_per_user, : self.env.num_streams_per_user]
        diagonal = torch.diagonal(r_factor)
        phase = torch.exp(-1j * torch.angle(diagonal))
        phase_matrix = torch.diag(phase.to(dtype=self.complex_dtype))
        q_aligned = q_factor @ phase_matrix.conj().transpose(0, 1)
        r_aligned = phase_matrix @ r_factor
        return q_aligned, r_aligned

    def _noise_cholesky_from_q(self, q_chain: torch.Tensor) -> torch.Tensor:
        noise_cov = q_chain.conj().transpose(0, 1) @ q_chain
        noise_cov = (noise_cov + noise_cov.conj().transpose(0, 1)) / 2.0
        return torch.linalg.cholesky(
            noise_cov + 1e-12 * torch.eye(noise_cov.shape[0], dtype=self.complex_dtype, device=self.device)
        )

    def _noise_std_from_q(self, q_chain: torch.Tensor) -> torch.Tensor:
        noise_cov = q_chain.conj().transpose(0, 1) @ q_chain
        noise_variance = torch.clamp(torch.real(torch.diagonal(noise_cov)), min=1e-12)
        return torch.sqrt(noise_variance)

    def _current_ucd_receiver_chains(
        self,
        f_rf: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        current_chain = self.env.build_structured_digital_chain(
            user_channels=self.user_channels,
            f_rf=f_rf.detach().cpu().numpy(),
            snr_per_stream=self.snr_per_stream,
            strategy="ucd",
        )
        current_q_chains = [
            torch.as_tensor(chain, dtype=self.complex_dtype, device=self.device)
            for chain in current_chain.q_chains
        ]
        current_noise_cholesky = [self._noise_cholesky_from_q(q_chain) for q_chain in current_q_chains]
        current_noise_std = [self._noise_std_from_q(q_chain) for q_chain in current_q_chains]
        return current_q_chains, current_noise_cholesky, current_noise_std

    def projector_blocks(
        self,
        f_rf: torch.Tensor,
        digital_blocks: list[torch.Tensor],
        fixed_projectors: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        effective_channels = [channel @ f_rf for channel in self.channel_tensors]
        projected_blocks = []
        for user_idx in range(self.env.num_users):
            if fixed_projectors is not None:
                projector = fixed_projectors[user_idx]
            elif self.env.num_users == 1:
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
        projected_blocks = [alpha * block for block in projected_blocks]
        f_bb = alpha * f_bb
        return projected_blocks, f_bb, effective_channels

    def build_precoder(
        self,
        theta: torch.Tensor,
        digital_blocks: list[torch.Tensor],
        frozen_analog_state: FrozenAnalogState | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        f_rf = torch.exp(1j * theta) / self.sqrt_nt
        fixed_projectors = None if frozen_analog_state is None else frozen_analog_state.projectors
        projected_blocks, f_bb, effective_channels = self.projector_blocks(
            f_rf, digital_blocks, fixed_projectors=fixed_projectors
        )
        return f_rf, projected_blocks, f_bb, effective_channels

    def capture_frozen_analog_state(
        self,
        theta: torch.Tensor,
    ) -> FrozenAnalogState:
        with torch.no_grad():
            f_rf = torch.exp(1j * theta) / self.sqrt_nt
            effective_channels = [channel @ f_rf for channel in self.channel_tensors]
            projectors = []
            for user_idx in range(self.env.num_users):
                if self.env.num_users == 1:
                    projector = self.eye_rf
                else:
                    interference_stack = torch.cat(
                        [effective_channels[idx] for idx in range(self.env.num_users) if idx != user_idx],
                        dim=0,
                    )
                    projector = self.eye_rf - torch.linalg.pinv(interference_stack) @ interference_stack
                projectors.append(projector.detach().clone())

            if self.strategy == "ucd":
                q_chains, noise_cholesky, noise_std = self._current_ucd_receiver_chains(f_rf)
                q_snapshot = tuple(chain.detach().clone() for chain in q_chains)
                chol_snapshot = tuple(chain.detach().clone() for chain in noise_cholesky)
                std_snapshot = tuple(chain.detach().clone() for chain in noise_std)
            else:
                q_snapshot = ()
                chol_snapshot = ()
                std_snapshot = ()

        return FrozenAnalogState(
            projectors=tuple(projectors),
            q_chains=q_snapshot,
            noise_cholesky=chol_snapshot,
            noise_std=std_snapshot,
        )

    def _thp_precoded_symbols_batch(
        self,
        triangular_channels: torch.Tensor,
        original_symbols: torch.Tensor,
    ) -> torch.Tensor:
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
        diagonal_channels: torch.Tensor,
    ) -> torch.Tensor:
        equalized = received / max(self.sqrt_snr, 1e-12)
        equalized = equalized / diagonal_channels[:, None, :]
        return _straight_through_centered_modulo_complex_torch(equalized, self.period)

    def _user_rates_sic_batch(
        self,
        upper_channels: torch.Tensor,
        transmitted_bits: torch.Tensor,
        received_samples: torch.Tensor,
        soft_temperature: float,
    ) -> torch.Tensor:
        num_samples = received_samples.shape[1]
        temperature = max(float(soft_temperature), 1e-4)
        scaled_upper = self.sqrt_snr * upper_channels
        large_neg = torch.finfo(self.real_dtype).min
        loss_sum = torch.zeros(self.env.num_users, dtype=self.real_dtype, device=self.device)
        future_soft_symbols: list[torch.Tensor | None] = [None] * self.env.num_streams_per_user

        for stream_idx in range(self.env.num_streams_per_user - 1, -1, -1):
            if stream_idx + 1 < self.env.num_streams_per_user:
                coeff = scaled_upper[:, stream_idx, stream_idx + 1 :]
                future = torch.stack(
                    [stream for stream in future_soft_symbols[stream_idx + 1 :] if stream is not None],
                    dim=2,
                )
                interference = torch.einsum("kl,ksl->ks", coeff, future)
                residual = received_samples[:, :, stream_idx] - interference
            else:
                residual = received_samples[:, :, stream_idx]

            gain = scaled_upper[:, stream_idx, stream_idx]
            codewords = gain[:, None] * self.constellation[None, :]
            distances = torch.abs(residual[:, :, None] - codewords[:, None, :]) ** 2
            scaled_neg_distances = -(distances / temperature).unsqueeze(-1)
            mask_one = self.mask_one_tensor.reshape(1, 1, -1, self.bits_per_symbol)
            mask_zero = self.mask_zero_tensor.reshape(1, 1, -1, self.bits_per_symbol)
            llr_one = torch.logsumexp(
                scaled_neg_distances.masked_fill(~mask_one, large_neg),
                dim=2,
            )
            llr_zero = torch.logsumexp(
                scaled_neg_distances.masked_fill(~mask_zero, large_neg),
                dim=2,
            )
            llr = temperature * (llr_one - llr_zero)
            signed_llr = (2.0 * transmitted_bits[:, :, stream_idx, :] - 1.0) * llr
            loss_sum = loss_sum + torch.sum(F.softplus(-signed_llr) * self.log2e, dim=(1, 2))

            posterior = torch.softmax(-(distances / temperature), dim=2)
            future_soft_symbols[stream_idx] = torch.sum(
                posterior * self.constellation[None, None, :],
                dim=2,
            )

        ideal_rate = float(self.env.num_streams_per_user * self.bits_per_symbol)
        return torch.clamp(
            torch.as_tensor(ideal_rate, dtype=self.real_dtype, device=self.device)
            - loss_sum / max(num_samples, 1),
            min=0.0,
        )

    def _user_rates_parallel_batch(
        self,
        diagonal_channels: torch.Tensor,
        transmitted_bits: torch.Tensor,
        received_samples: torch.Tensor,
        soft_temperature: float,
    ) -> torch.Tensor:
        num_samples = received_samples.shape[1]
        temperature = max(float(soft_temperature), 1e-4)
        scaled_codewords = (
            self.sqrt_snr
            * diagonal_channels[:, :, None]
            * self.constellation[None, None, :]
        )
        distances = torch.abs(received_samples[:, :, :, None] - scaled_codewords[:, None, :, :]) ** 2
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

    def _user_rates_thp_batch(
        self,
        diagonal_channels: torch.Tensor,
        transmitted_bits: torch.Tensor,
        equalized_received: torch.Tensor,
        soft_temperature: float,
    ) -> torch.Tensor:
        num_samples = equalized_received.shape[1]
        temperature = max(float(soft_temperature), 1e-4)
        scaled_codewords = (
            self.sqrt_snr
            * diagonal_channels[:, :, None]
            * self.replica_constellation[None, None, :]
        )
        received_scaled = self.sqrt_snr * equalized_received * diagonal_channels[:, None, :]
        distances = torch.abs(
            received_scaled[:, :, :, None] - scaled_codewords[:, None, :, :]
        ) ** 2
        scaled_neg_distances = -(distances / temperature).unsqueeze(-1)
        large_neg = torch.finfo(self.real_dtype).min
        llr_one = torch.logsumexp(
            scaled_neg_distances.masked_fill(~self.replica_mask_one_tensor, large_neg),
            dim=3,
        )
        llr_zero = torch.logsumexp(
            scaled_neg_distances.masked_fill(~self.replica_mask_zero_tensor, large_neg),
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

    def compute_metrics(
        self,
        theta: torch.Tensor,
        digital_blocks: list[torch.Tensor],
        soft_temperature: float = 1.0,
        frozen_analog_state: FrozenAnalogState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f_rf, projected_blocks, f_bb, effective_channels = self.build_precoder(
            theta,
            digital_blocks,
            frozen_analog_state=frozen_analog_state,
        )
        effective_stack = torch.stack(effective_channels, dim=0)
        projected_stack = torch.stack(projected_blocks, dim=0)

        if self.strategy == "ucd" and frozen_analog_state is not None:
            runtime_q_chains = list(frozen_analog_state.q_chains)
            runtime_noise_cholesky = list(frozen_analog_state.noise_cholesky)
            runtime_noise_std = list(frozen_analog_state.noise_std)
        elif self.strategy == "ucd":
            runtime_q_chains, runtime_noise_cholesky, runtime_noise_std = self._current_ucd_receiver_chains(f_rf)
        else:
            runtime_q_chains = []
            runtime_noise_cholesky = []
            runtime_noise_std = []

        rotated_blocks = []
        equalizer_diagonal_channels = []
        metric_diagonal_channels = []
        current_blocks = []
        for user_idx in range(self.env.num_users):
            rx_blocks = torch.matmul(
                effective_stack[user_idx].unsqueeze(0),
                projected_stack,
            )
            if self.strategy == "svd":
                desired_block = rx_blocks[user_idx]
                q_factor, triangular_block = self._qr_factors_with_positive_diagonal(desired_block)
                current_q = q_factor
                current_block = triangular_block
                current_diag = torch.diagonal(triangular_block, dim1=0, dim2=1)
                metric_diag = current_diag
            else:
                current_q = runtime_q_chains[user_idx]
                current_block = current_q.conj().transpose(0, 1) @ rx_blocks[user_idx]
                current_diag = torch.diagonal(current_block, dim1=0, dim2=1)
                metric_diag = current_diag / runtime_noise_std[user_idx].to(dtype=self.complex_dtype)
            current_blocks.append(current_block)
            rotated_user = torch.matmul(
                current_q.conj().transpose(0, 1).unsqueeze(0),
                rx_blocks,
            )
            rotated_blocks.append(rotated_user)
            equalizer_diagonal_channels.append(current_diag)
            metric_diagonal_channels.append(metric_diag)

        rotated_blocks_t = torch.stack(rotated_blocks, dim=0)
        equalizer_diagonal_channels_t = torch.stack(equalizer_diagonal_channels, dim=0)
        metric_diagonal_channels_t = torch.stack(metric_diagonal_channels, dim=0)
        current_blocks_t = torch.stack(current_blocks, dim=0)

        repeat_rates = []
        user_rate_accum = torch.zeros(self.env.num_users, dtype=self.real_dtype, device=self.device)
        for batch_group in self.samples:
            original_symbols_all = torch.stack([batch[0] for batch in batch_group], dim=0)
            transmitted_bits_all = torch.stack([batch[1] for batch in batch_group], dim=0)
            noise_all = torch.stack([batch[2] for batch in batch_group], dim=0)
            if self.strategy == "svd":
                colored_noise_all = noise_all
            else:
                colored_noise_all = torch.matmul(
                    noise_all,
                    torch.stack(runtime_noise_cholesky, dim=0).transpose(1, 2),
                )

            if self.strategy == "svd":
                transmitted_symbols_all = original_symbols_all
            elif self.strategy == "ucd":
                desired_blocks = torch.stack(
                    [rotated_blocks_t[user_idx, user_idx] for user_idx in range(self.env.num_users)],
                    dim=0,
                )
                transmitted_symbols_all = self._thp_precoded_symbols_batch(
                    desired_blocks,
                    original_symbols_all,
                )
            else:
                raise ValueError(f"Unsupported AO strategy: {self.strategy}")

            received_all = colored_noise_all + self.sqrt_snr * torch.einsum(
                "tsj,rtji->rsi",
                transmitted_symbols_all,
                rotated_blocks_t.transpose(-1, -2),
            )

            if self.strategy == "svd":
                if self.svd_receiver_mode == "parallel":
                    user_rates = self._user_rates_parallel_batch(
                        diagonal_channels=equalizer_diagonal_channels_t,
                        transmitted_bits=transmitted_bits_all,
                        received_samples=received_all,
                        soft_temperature=soft_temperature,
                    )
                else:
                    user_rates = self._user_rates_sic_batch(
                        upper_channels=current_blocks_t,
                        transmitted_bits=transmitted_bits_all,
                        received_samples=received_all,
                        soft_temperature=soft_temperature,
                    )
            else:
                equalized_all = self._equalize_received_batch(
                    received=received_all,
                    diagonal_channels=equalizer_diagonal_channels_t,
                )
                user_rates = self._user_rates_thp_batch(
                    diagonal_channels=metric_diagonal_channels_t,
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
        return mean_rate, std_rate, user_rates, f_rf, f_bb

    def evaluate(
        self,
        theta: torch.Tensor,
        digital_blocks: list[torch.Tensor],
        soft_temperature: float = 1.0,
        frozen_analog_state: FrozenAnalogState | None = None,
    ) -> tuple[SoftRateAverageEvaluation, np.ndarray, np.ndarray]:
        with torch.no_grad():
            mean_rate, std_rate, user_rates, f_rf, f_bb = self.compute_metrics(
                theta,
                digital_blocks,
                soft_temperature=soft_temperature,
                frozen_analog_state=frozen_analog_state,
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


def _build_explicit_initialization(
    env: MultiUserSimulationEnvironment,
    user_channels: np.ndarray,
    receiver_user_channels: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    train_average: MultiUserSICSampleAverage,
    eval_average: MultiUserSICSampleAverage,
    strategy: str,
    f_rf: np.ndarray,
    f_bb: np.ndarray,
    device: str,
) -> BaselineInitialization:
    normalized_strategy = str(strategy).lower()
    if normalized_strategy == "svd":
        receiver_mode = (
            "parallel"
            if np.array_equal(np.asarray(user_channels), np.asarray(receiver_user_channels))
            else "sic"
        )
    elif normalized_strategy == "gmd":
        receiver_mode = "thp"
    elif normalized_strategy == "ucd":
        receiver_mode = "ucd_thp"
    else:
        raise ValueError(f"Unsupported explicit AO initialization strategy: {normalized_strategy}")

    objective = JointFixedReceiverSoftObjective(
        env=env,
        user_channels=user_channels,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        sample_average=train_average,
        q_chains=env.build_structured_digital_chain(
            user_channels=user_channels,
            f_rf=f_rf,
            snr_per_stream=snr_per_stream,
            strategy=normalized_strategy,
        ).q_chains,
        strategy=normalized_strategy,
        device=device,
        svd_receiver_mode=receiver_mode if normalized_strategy == "svd" else "sic",
    )
    coeff_blocks = [
        torch.nn.Parameter(torch.as_tensor(block, dtype=objective.complex_dtype, device=objective.device))
        for block in env.split_user_blocks(f_bb)
    ]
    train_eval, _, _ = objective.evaluate(
        theta=torch.nn.Parameter(
            torch.as_tensor(
                np.angle(f_rf * np.sqrt(env.num_tx_antennas)),
                dtype=torch.float32,
                device=objective.device,
            )
        ),
        digital_blocks=coeff_blocks,
        soft_temperature=1.0,
    )

    if receiver_mode == "parallel":
        receiver_eval = env.evaluate_precoder_current_receiver_average_parallel(
            user_channels=receiver_user_channels,
            f_rf=f_rf,
            f_bb=f_bb,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            sample_average=eval_average,
        )
    elif receiver_mode == "sic":
        receiver_eval = env.evaluate_precoder_current_receiver_average(
            user_channels=receiver_user_channels,
            f_rf=f_rf,
            f_bb=f_bb,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            sample_average=eval_average,
        )
    elif receiver_mode == "ucd_thp":
        ucd_chain = env.build_structured_digital_chain(
            user_channels=user_channels,
            f_rf=f_rf,
            snr_per_stream=snr_per_stream,
            strategy="ucd",
        )
        receiver_eval = env.evaluate_ucd_precoder_current_receiver_average_b_chain(
            user_channels=receiver_user_channels,
            f_rf=f_rf,
            f_bb=f_bb,
            q_chains=ucd_chain.q_chains,
            r_chains=ucd_chain.r_chains,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            sample_average=eval_average,
        )
    else:
        raise ValueError(f"Unsupported receiver mode: {receiver_mode}")

    return BaselineInitialization(
        strategy=normalized_strategy,
        receiver_mode=receiver_mode,
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
    objective: JointFixedReceiverSoftObjective,
    receiver_user_channels: np.ndarray,
    eval_average: MultiUserSICSampleAverage,
    snr_per_stream: float,
    bits_per_symbol: int,
    baseline_f_rf: np.ndarray,
    baseline_f_bb: np.ndarray,
    outer_iterations: int,
    digital_steps: int,
    analog_steps: int,
    digital_lr: float,
    analog_lr: float,
    initial_temperature: float,
    final_temperature: float,
    selection_temperature: float,
    grad_clip_norm: float | None,
    debug: bool = False,
) -> tuple[
    SoftRateAverageEvaluation,
    np.ndarray,
    np.ndarray,
    ReceiverAverageEvaluation,
    list[float],
    list[float],
    list[dict[str, float]],
]:
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
        baseline_mean_rate, _, _, _, _ = objective.compute_metrics(
            theta_param,
            digital_params,
            soft_temperature=selection_temperature,
        )
    best_selection_score = float(baseline_mean_rate.item())
    history = [best_selection_score]
    temperature_history = _build_temperature_schedule(
        outer_iterations=outer_iterations,
        initial_temperature=initial_temperature,
        final_temperature=final_temperature,
    )
    debug_rows: list[dict[str, float]] = []

    for outer_idx, train_temperature in enumerate(temperature_history):
        optimizer_d = torch.optim.Adam(digital_params, lr=digital_lr)
        for digital_idx in range(digital_steps):
            optimizer_d.zero_grad(set_to_none=True)
            mean_rate, _, _, _, _ = objective.compute_metrics(
                theta_param,
                digital_params,
                soft_temperature=train_temperature,
            )
            loss = -mean_rate
            loss.backward()
            grad_sq = 0.0
            grad_abs_max = 0.0
            for param in digital_params:
                if param.grad is None:
                    continue
                grad_np = param.grad.detach()
                grad_sq += float(torch.sum(torch.abs(grad_np) ** 2).item())
                grad_abs_max = max(grad_abs_max, float(torch.max(torch.abs(grad_np)).item()))
            pre_step_blocks = [param.detach().clone() for param in digital_params]
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(digital_params, grad_clip_norm)
            optimizer_d.step()
            step_delta = 0.0
            for before, after in zip(pre_step_blocks, digital_params):
                step_delta += float(torch.sum(torch.abs(after.detach() - before) ** 2).item())
            if debug:
                debug_rows.append(
                    {
                        "phase": 0.0,
                        "outer_iter": float(outer_idx),
                        "inner_iter": float(digital_idx),
                        "temperature": float(train_temperature),
                        "rate": float(mean_rate.item()),
                        "loss": float(loss.item()),
                        "grad_norm": float(np.sqrt(grad_sq)),
                        "grad_abs_max": grad_abs_max,
                        "step_delta_norm": float(np.sqrt(step_delta)),
                    }
                )

        frozen_analog_state = objective.capture_frozen_analog_state(theta_param)
        optimizer_theta = torch.optim.Adam([theta_param], lr=analog_lr)
        for analog_idx in range(analog_steps):
            optimizer_theta.zero_grad(set_to_none=True)
            mean_rate, _, _, _, _ = objective.compute_metrics(
                theta_param,
                digital_params,
                soft_temperature=train_temperature,
                frozen_analog_state=frozen_analog_state,
            )
            loss = -mean_rate
            loss.backward()
            theta_grad = theta_param.grad.detach() if theta_param.grad is not None else None
            theta_grad_norm = float(torch.norm(theta_grad).item()) if theta_grad is not None else 0.0
            theta_grad_abs_max = float(torch.max(torch.abs(theta_grad)).item()) if theta_grad is not None else 0.0
            theta_before = theta_param.detach().clone()
            with torch.no_grad():
                pre_rate_eval, pre_f_rf_eval, pre_f_bb_eval = objective.evaluate(
                    theta_param,
                    [param.detach() for param in digital_params],
                    soft_temperature=selection_temperature,
                    frozen_analog_state=frozen_analog_state,
                )
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_([theta_param], grad_clip_norm)
            optimizer_theta.step()
            with torch.no_grad():
                theta_param.copy_((theta_param + np.pi) % (2.0 * np.pi) - np.pi)
            theta_step_delta = float(torch.norm(theta_param.detach() - theta_before).item())
            with torch.no_grad():
                post_rate_eval, post_f_rf_eval, post_f_bb_eval = objective.evaluate(
                    theta_param,
                    [param.detach() for param in digital_params],
                    soft_temperature=selection_temperature,
                    frozen_analog_state=frozen_analog_state,
                )
            f_rf_step_delta = float(np.linalg.norm(post_f_rf_eval - pre_f_rf_eval))
            f_bb_step_delta = float(np.linalg.norm(post_f_bb_eval - pre_f_bb_eval))
            if debug:
                debug_rows.append(
                    {
                        "phase": 1.0,
                        "outer_iter": float(outer_idx),
                        "inner_iter": float(analog_idx),
                        "temperature": float(train_temperature),
                        "rate": float(mean_rate.item()),
                        "loss": float(loss.item()),
                        "grad_norm": theta_grad_norm,
                        "grad_abs_max": theta_grad_abs_max,
                        "step_delta_norm": theta_step_delta,
                        "pre_eval_rate": float(pre_rate_eval.sum_rate),
                        "post_eval_rate": float(post_rate_eval.sum_rate),
                        "f_rf_step_delta": f_rf_step_delta,
                        "f_bb_step_delta": f_bb_step_delta,
                    }
                )

        with torch.no_grad():
            current_mean_rate, _, _, _, _ = objective.compute_metrics(
                theta_param,
                digital_params,
                soft_temperature=selection_temperature,
            )
        current_score = float(current_mean_rate.item())
        history.append(current_score)
        if current_score > best_selection_score + 1e-12:
            best_selection_score = current_score
            best_theta = theta_param.detach().clone()
            best_digital = [param.detach().clone() for param in digital_params]

    optimized_soft_train_eval, optimized_f_rf, optimized_f_bb = objective.evaluate(
        best_theta,
        best_digital,
        soft_temperature=selection_temperature,
    )
    if objective.strategy == "svd":
        if objective.svd_receiver_mode == "parallel":
            optimized_receiver_eval = env.evaluate_precoder_current_receiver_average_parallel(
                user_channels=receiver_user_channels,
                f_rf=optimized_f_rf,
                f_bb=optimized_f_bb,
                snr_per_stream=snr_per_stream,
                bits_per_symbol=bits_per_symbol,
                sample_average=eval_average,
            )
        else:
            optimized_receiver_eval = env.evaluate_precoder_current_receiver_average(
                user_channels=receiver_user_channels,
                f_rf=optimized_f_rf,
                f_bb=optimized_f_bb,
                snr_per_stream=snr_per_stream,
                bits_per_symbol=bits_per_symbol,
                sample_average=eval_average,
            )
    else:
        optimized_ucd_chain = env.build_structured_digital_chain(
            user_channels=receiver_user_channels,
            f_rf=optimized_f_rf,
            snr_per_stream=snr_per_stream,
            strategy="ucd",
        )
        optimized_receiver_eval = env.evaluate_ucd_precoder_current_receiver_average_b_chain(
            user_channels=receiver_user_channels,
            f_rf=optimized_f_rf,
            f_bb=optimized_f_bb,
            q_chains=optimized_ucd_chain.q_chains,
            r_chains=optimized_ucd_chain.r_chains,
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
        debug_rows,
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
    device: str = "cpu",
    initial_strategy: str | None = None,
    initial_f_rf: np.ndarray | None = None,
    initial_f_bb: np.ndarray | None = None,
    debug: bool = False,
) -> SoftJointAOResult:
    if final_temperature is None:
        final_temperature = initial_temperature
    if initial_strategy is None or initial_f_rf is None or initial_f_bb is None:
        raise ValueError("optimize_soft_joint_ao now requires an explicit initialization.")

    effective_eval_average = train_average if eval_average is None else eval_average
    effective_receiver_channels = (
        np.asarray(user_channels, dtype=complex)
        if receiver_user_channels is None
        else np.asarray(receiver_user_channels, dtype=complex)
    )

    baseline = _build_explicit_initialization(
        env=env,
        user_channels=user_channels,
        receiver_user_channels=effective_receiver_channels,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        train_average=train_average,
        eval_average=effective_eval_average,
        strategy=initial_strategy,
        f_rf=initial_f_rf,
        f_bb=initial_f_bb,
        device=device,
    )

    if baseline.strategy == "svd":
        q_chains = env.build_structured_digital_chain(
            user_channels=user_channels,
            f_rf=baseline.f_rf,
            snr_per_stream=snr_per_stream,
            strategy="svd",
        ).q_chains
        objective = JointFixedReceiverSoftObjective(
            env=env,
            user_channels=user_channels,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            sample_average=train_average,
            q_chains=q_chains,
            strategy="svd",
            device=device,
            svd_receiver_mode=baseline.receiver_mode,
        )
    elif baseline.strategy == "ucd":
        q_chains = env.build_structured_digital_chain(
            user_channels=user_channels,
            f_rf=baseline.f_rf,
            snr_per_stream=snr_per_stream,
            strategy="ucd",
        ).q_chains
        objective = JointFixedReceiverSoftObjective(
            env=env,
            user_channels=user_channels,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            sample_average=train_average,
            q_chains=q_chains,
            strategy="ucd",
            device=device,
        )
    else:
        raise ValueError(f"Unsupported AO initialization strategy: {baseline.strategy}")

    (
        optimized_soft_train_eval,
        optimized_f_rf,
        optimized_f_bb,
        optimized_receiver_eval,
        history,
        temperature_history,
        debug_rows,
    ) = _run_joint_ao_from_initialization(
        env=env,
        objective=objective,
        receiver_user_channels=effective_receiver_channels,
        eval_average=effective_eval_average,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        baseline_f_rf=baseline.f_rf,
        baseline_f_bb=baseline.f_bb,
        outer_iterations=outer_iterations,
        digital_steps=digital_steps,
        analog_steps=analog_steps,
        digital_lr=digital_lr,
        analog_lr=analog_lr,
        initial_temperature=initial_temperature,
        final_temperature=final_temperature,
        selection_temperature=selection_temperature,
        grad_clip_norm=grad_clip_norm,
        debug=debug,
    )

    return SoftJointAOResult(
        baseline_strategy=baseline.strategy,
        baseline_f_rf=baseline.f_rf,
        baseline_f_bb=baseline.f_bb,
        baseline_soft_train_eval=baseline.soft_train_eval,
        baseline_receiver_eval=baseline.receiver_eval,
        optimized_f_rf=optimized_f_rf,
        optimized_f_bb=optimized_f_bb,
        optimized_soft_train_eval=optimized_soft_train_eval,
        optimized_receiver_eval=optimized_receiver_eval,
        history=history,
        training_temperature_history=temperature_history,
        debug_rows=debug_rows,
    )
