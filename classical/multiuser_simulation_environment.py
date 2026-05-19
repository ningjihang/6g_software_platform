from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space
from scipy.linalg import svd

from analog_precoder import AnalogPrecoder
from bicm_metrics import (
    estimate_bicm_gmi_thp_from_received,
    estimate_bit_error_rate_thp_from_received,
    get_constellation,
)
from channel_model import ChannelModel
from digital_precoder import DigitalStructuredPrecoder
from thp_precoding import centered_modulo_complex, thp_receive_equalized, thp_transmit_from_upper


@dataclass
class ReceiverAverageEvaluation:
    sum_rate: float
    sum_rate_std: float
    bit_error_rate: float
    user_rates: np.ndarray
    user_bit_error_rates: np.ndarray
    user_rho: list[np.ndarray]
    leakage_matrix: np.ndarray
    offdiag_to_desired: float


@dataclass(frozen=True)
class StructuredChain:
    f_bb: np.ndarray
    effective_channels: list[np.ndarray]
    bd_digital_bases: list[np.ndarray]
    f_blocks: list[np.ndarray]
    q_chains: list[np.ndarray]
    r_chains: list[np.ndarray]
    f_rf_blocks: list[np.ndarray]


class MultiUserSimulationEnvironment:
    """
    Exact-rate multi-user utilities under the current per-user-RF hybrid-precoding model.

    This class intentionally contains only:
    - channel generation
    - user-wise digital-BD helper
    - strict BD + THP/UCD receiver evaluation
    - structured SVD/GMD/UCD construction helpers

    Any optimizer built on top of this class should optimize the same exact
    multi-user bit-wise rate returned by the current receiver path.
    """

    def __init__(
        self,
        num_users: int,
        num_tx_antennas: int,
        num_rx_antennas: int,
        num_rf_chains: int,
        num_streams_per_user: int,
        channel_type: str = "cdl-a",
        digital_power_constraint: float | None = None,
        ucd_waterfill: bool = True,
        ucd_min_power_loading: float = 0.0,
    ):
        """????????????"""
        self.num_users = num_users
        self.num_tx_antennas = num_tx_antennas
        self.num_rx_antennas = num_rx_antennas
        self.num_rf_chains = num_rf_chains
        self.num_streams_per_user = num_streams_per_user
        self.total_streams = num_users * num_streams_per_user
        if self.num_rf_chains % self.num_users != 0:
            raise ValueError(
                "Dedicated per-user RF requires num_rf_chains divisible by num_users, got "
                f"{self.num_rf_chains} and {self.num_users}."
            )
        self.rf_chains_per_user = self.num_rf_chains // self.num_users

        if self.total_streams > num_rf_chains:
            raise ValueError(
                "Total user streams exceed available RF chains: "
                f"{self.total_streams} > {num_rf_chains}"
            )
        if self.rf_chains_per_user < self.num_streams_per_user:
            raise ValueError(
                "Dedicated per-user RF requires rf_chains_per_user >= num_streams_per_user, got "
                f"{self.rf_chains_per_user} < {self.num_streams_per_user}."
            )
        if digital_power_constraint is None:
            self.digital_power_constraint = float(self.total_streams)
        else:
            self.digital_power_constraint = float(digital_power_constraint)
            if self.digital_power_constraint <= 0.0:
                raise ValueError(
                    "digital_power_constraint must be positive, got "
                    f"{digital_power_constraint}."
                )

        self.channel_model = ChannelModel(
            num_tx_antennas=num_tx_antennas,
            num_rx_antennas=num_rx_antennas,
            channel_type=channel_type,
        )
        self.analog_precoder = AnalogPrecoder(num_tx_antennas, num_rf_chains)
        self.digital_precoder = DigitalStructuredPrecoder(
            num_rf_chains,
            num_streams_per_user,
            ucd_waterfill=ucd_waterfill,
            ucd_min_power_loading=ucd_min_power_loading,
        )

    def generate_user_channels(self) -> np.ndarray:
        """?????user channels?"""
        return np.stack([self.channel_model.generate_channel() for _ in range(self.num_users)], axis=0)

    def build_analog_precoder(self, user_channels: np.ndarray) -> np.ndarray:
        """?????analog precoder?"""
        return self.analog_precoder.build_precoder(
            user_channels=user_channels,
            num_streams_per_user=self.num_streams_per_user,
        )

    def build_fixed_baseline_precoder(self) -> tuple[np.ndarray, np.ndarray]:
        """Build a channel-agnostic baseline with no SVD/GMD/BD/AO design.

        The RF stage uses the first columns of a normalized DFT codebook, and
        the digital stage simply maps the data streams onto the first RF chains.
        This gives a full-rank, deterministic reference without any channel
        matrix decomposition.
        """

        antenna_idx = np.arange(self.num_tx_antennas, dtype=float).reshape(-1, 1)
        rf_idx = np.arange(self.num_rf_chains, dtype=float).reshape(1, -1)
        f_rf = np.exp(2j * np.pi * antenna_idx * rf_idx / self.num_tx_antennas) / np.sqrt(
            self.num_tx_antennas
        )

        f_bb = np.zeros((self.num_rf_chains, self.total_streams), dtype=complex)
        for stream_idx in range(self.total_streams):
            f_bb[stream_idx, stream_idx] = 1.0
        f_bb = self.normalize_digital_precoder(f_rf=f_rf, f_bb=f_bb)
        return f_rf, f_bb

    def build_effective_channels(self, user_channels: np.ndarray, f_rf: np.ndarray) -> list[np.ndarray]:
        """?????effective channels?"""
        return [user_channels[user_idx] @ f_rf for user_idx in range(self.num_users)]

    def split_rf_blocks(self, f_rf: np.ndarray) -> list[np.ndarray]:
        """Split the dedicated RF precoder into user-specific RF blocks."""
        f_rf = np.asarray(f_rf, dtype=complex)
        return [f_rf[:, self.user_rf_slice(user_idx)] for user_idx in range(self.num_users)]

    def user_rf_slice(self, user_index: int) -> slice:
        start = int(user_index * self.rf_chains_per_user)
        stop = int(start + self.rf_chains_per_user)
        return slice(start, stop)

    def build_bd_digital_basis(self, effective_channels: list[np.ndarray], user_index: int) -> np.ndarray:
        """Build the BD digital basis N_k used in F_BB = [N_1 F_1, ..., N_K F_K]."""
        if self.num_users == 1:
            return np.eye(self.num_rf_chains, dtype=complex)

        interference_stack = np.vstack(
            [effective_channels[idx] for idx in range(self.num_users) if idx != user_index]
        )
        basis = null_space(interference_stack)
        if basis.shape[1] < self.num_streams_per_user:
            raise ValueError(
                "Digital-BD null-space is too small for the requested per-user streams. "
                f"User {user_index}: nullity={basis.shape[1]}, "
                f"required={self.num_streams_per_user}. "
                "Try fewer users/streams or more RF chains."
            )
        return basis

    def qr_factors_with_positive_diagonal(self, channel_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """?? qr factors with positive diagonal ???"""
        q_factor, r_factor = np.linalg.qr(channel_block, mode="reduced")
        q_factor = q_factor[:, : self.num_streams_per_user]
        r_factor = r_factor[: self.num_streams_per_user, : self.num_streams_per_user]
        diagonal = np.diag(r_factor)
        phase = np.exp(-1j * np.angle(diagonal))
        phase_matrix = np.diag(phase)
        q_aligned = q_factor @ phase_matrix.conj().T
        r_aligned = phase_matrix @ r_factor
        return q_aligned, r_aligned

    def qr_with_positive_diagonal(self, channel_block: np.ndarray) -> np.ndarray:
        """?? qr with positive diagonal ???"""
        return self.qr_factors_with_positive_diagonal(channel_block)[1]

    def normalize_digital_precoder(self, f_rf: np.ndarray, f_bb: np.ndarray) -> np.ndarray:
        """??????digital precoder?"""
        full_precoder = f_rf @ f_bb
        power = float(np.linalg.norm(full_precoder, "fro") ** 2)
        if power <= 1e-12:
            return f_bb
        return f_bb * np.sqrt(self.digital_power_constraint / power)

    def _qam_hard_decision(self, samples: np.ndarray, bits_per_symbol: int) -> np.ndarray:
        """Slice complex samples to the nearest normalized square-QAM constellation points."""

        symbols, bits = get_constellation(bits_per_symbol)
        sample_array = np.asarray(samples, dtype=complex).reshape(-1)
        distances = np.abs(sample_array[:, None] - symbols[None, :]) ** 2
        hard_indices = np.argmin(distances, axis=1)
        return symbols[hard_indices].reshape(np.shape(samples))

    def qam_modulo_period(self, bits_per_symbol: int) -> float:
        """Return the square-QAM THP modulo period for the normalized constellation."""

        m_axis = int(np.sqrt(2**bits_per_symbol))
        return float(m_axis * 2.0 / np.sqrt((2.0 / 3.0) * (2**bits_per_symbol - 1)))

    def build_structured_digital_chain(
        self,
        user_channels: np.ndarray,
        f_rf: np.ndarray,
        snr_per_stream: float,
        strategy: str,
    ) -> StructuredChain:
        """Build the user-wise digital blocks in the paper form F_BB = [N_1 F_1, ..., N_K F_K]."""

        strategy = strategy.lower()
        if strategy not in {"svd", "gmd", "ucd"}:
            raise ValueError(f"Unsupported strategy: {strategy}")

        user_channels = np.asarray(user_channels, dtype=complex)
        f_rf_blocks = self.split_rf_blocks(f_rf)
        effective_channels = self.build_effective_channels(user_channels, f_rf)
        bd_digital_bases = []
        user_blocks = []
        f_blocks = []
        r_chains = []
        q_chains = []

        for user_idx in range(self.num_users):
            n_k = self.build_bd_digital_basis(effective_channels, user_idx)
            reduced_channel = effective_channels[user_idx] @ n_k
            u_eff, singular_values, vh_eff = svd(reduced_channel, full_matrices=False)
            singular_values = singular_values[: self.num_streams_per_user]
            u_eff = u_eff[:, : self.num_streams_per_user]
            v_eff = vh_eff.conj().T[:, : self.num_streams_per_user]

            if strategy == "svd":
                f_k_local = v_eff
                q_local = u_eff
                r_local = np.diag(singular_values)
            elif strategy == "gmd":
                gmd_block = self.digital_precoder.design_gmd_transceiver_from_svd(
                    u_eff=u_eff,
                    v_eff=v_eff,
                    singular_values=singular_values,
                )
                f_k_local = gmd_block.p_gmd
                q_local = gmd_block.q_gmd
                r_local = gmd_block.r_gmd
            else:
                ucd_block = self.digital_precoder.design_ucd_transceiver_from_svd(
                    u_eff=u_eff,
                    v_eff=v_eff,
                    singular_values=singular_values,
                    snr_per_stream=snr_per_stream,
                )
                f_k_local = ucd_block.p_ucd
                q_local = ucd_block.w_ucd
                r_local = ucd_block.r_aug_ucd

            user_blocks.append(n_k @ f_k_local)
            f_blocks.append(np.asarray(f_k_local, dtype=complex))
            bd_digital_bases.append(n_k)
            r_chains.append(np.asarray(r_local, dtype=complex))
            q_chains.append(np.asarray(q_local, dtype=complex))

        f_bb = np.hstack(user_blocks)
        full_precoder = f_rf @ f_bb
        power = float(np.linalg.norm(full_precoder, "fro") ** 2)
        if power > 1e-12:
            correction_scale = np.sqrt(self.digital_power_constraint / power)
            f_bb = f_bb * correction_scale
            r_chains = [correction_scale * r_local for r_local in r_chains]
        return StructuredChain(
            f_bb=f_bb,
            effective_channels=effective_channels,
            bd_digital_bases=bd_digital_bases,
            f_blocks=f_blocks,
            q_chains=q_chains,
            r_chains=r_chains,
            f_rf_blocks=f_rf_blocks,
        )

    def split_user_blocks(self, f_bb: np.ndarray) -> list[np.ndarray]:
        """?????user blocks?"""
        user_blocks = []
        start_col = 0
        for _ in range(self.num_users):
            end_col = start_col + self.num_streams_per_user
            user_blocks.append(f_bb[:, start_col:end_col])
            start_col = end_col
        return user_blocks

    def evaluate_precoder_current_receiver_average_fixed_chain(
        self,
        user_channels: np.ndarray,
        f_rf: np.ndarray,
        f_bb: np.ndarray,
        r_chains: list[np.ndarray],
        q_chains: list[np.ndarray],
        snr_per_stream: float,
        bits_per_symbol: int,
        sample_average,
        apply_modulo: bool = True,
        labeling: str = "gray_standard",
    ) -> ReceiverAverageEvaluation:
        """Evaluate a structured P/Q/R chain without rebuilding a new receive decomposition."""

        symbols, _ = get_constellation(bits_per_symbol, labeling=labeling)
        effective_channels = self.build_effective_channels(user_channels, f_rf)
        user_blocks = self.split_user_blocks(f_bb)

        leakage_matrix = np.zeros((self.num_users, self.num_users), dtype=float)
        g_chains: list[list[np.ndarray]] = []
        receiver_covariances: list[np.ndarray] = []
        effective_diagonals: list[np.ndarray] = []
        user_rho = []

        for rx_user in range(self.num_users):
            q_rx = np.asarray(q_chains[rx_user], dtype=complex)
            user_g_blocks = []
            for tx_user in range(self.num_users):
                leakage_block = effective_channels[rx_user] @ user_blocks[tx_user]
                leakage_matrix[rx_user, tx_user] = float(np.linalg.norm(leakage_block, "fro") ** 2)
                user_g_blocks.append(q_rx.conj().T @ leakage_block)
            g_chains.append(user_g_blocks)
            noise_covariance = q_rx.conj().T @ q_rx
            receiver_covariances.append(noise_covariance)
            noise_variance = np.maximum(np.real(np.diag(noise_covariance)), 1e-12)
            g_diag = np.diag(user_g_blocks[rx_user])
            effective_diagonals.append(g_diag / np.sqrt(noise_variance))
            design_diagonal = np.abs(g_diag) ** 2
            user_rho.append(
                np.maximum(
                    float(snr_per_stream) * design_diagonal / noise_variance,
                    1e-12,
                )
            )

        desired_power = float(np.trace(leakage_matrix))
        offdiag_power = float(np.sum(leakage_matrix) - desired_power)
        offdiag_to_desired = offdiag_power / max(desired_power, 1e-12)

        repeat_sum_rates = np.zeros(sample_average.num_repeats, dtype=float)
        user_rate_accum = np.zeros(self.num_users, dtype=float)
        user_ber_accum = np.zeros(self.num_users, dtype=float)
        sqrt_snr = float(np.sqrt(snr_per_stream))
        period = self.qam_modulo_period(bits_per_symbol)

        for repeat_idx, user_batches in enumerate(sample_average.batches):
            original_symbols = [symbols[batch.symbol_indices] for batch in user_batches]
            transmitted_symbols = []
            for user_idx in range(self.num_users):
                _, dp_cod_data, _ = thp_transmit_from_upper(
                    cod_data=original_symbols[user_idx].T,
                    upper_b=g_chains[user_idx][user_idx],
                    modem_bits=bits_per_symbol,
                )
                transmitted_symbols.append(dp_cod_data.T)

            for rx_user in range(self.num_users):
                noise_stream = np.asarray(user_batches[rx_user].noise, dtype=complex)
                noise_cov = receiver_covariances[rx_user]
                noise_cov = (noise_cov + noise_cov.conj().T) / 2.0
                chol_noise = np.linalg.cholesky(
                    noise_cov + 1e-12 * np.eye(noise_cov.shape[0], dtype=complex)
                )
                received = noise_stream @ chol_noise.T
                for tx_user in range(self.num_users):
                    received += sqrt_snr * (
                        transmitted_symbols[tx_user] @ g_chains[rx_user][tx_user].T
                    )

                y_equ = received.T / max(sqrt_snr, 1e-12)
                g_diag = np.diag(g_chains[rx_user][rx_user])
                y1 = y_equ / g_diag[:, None]
                dp_rx_data = centered_modulo_complex(y1, period) if apply_modulo else y1

                rate_value = estimate_bicm_gmi_thp_from_received(
                    diagonal_channel=effective_diagonals[rx_user],
                    snr_per_stream=snr_per_stream,
                    bits_per_symbol=bits_per_symbol,
                    symbol_indices=user_batches[rx_user].symbol_indices,
                    received_samples=dp_rx_data.T,
                    labeling=labeling,
                )
                ber_value = estimate_bit_error_rate_thp_from_received(
                    diagonal_channel=effective_diagonals[rx_user],
                    snr_per_stream=snr_per_stream,
                    bits_per_symbol=bits_per_symbol,
                    symbol_indices=user_batches[rx_user].symbol_indices,
                    received_samples=dp_rx_data.T,
                    labeling=labeling,
                )
                repeat_sum_rates[repeat_idx] += rate_value
                user_rate_accum[rx_user] += rate_value
                user_ber_accum[rx_user] += ber_value

        user_bit_error_rates = user_ber_accum / sample_average.num_repeats
        return ReceiverAverageEvaluation(
            sum_rate=float(np.mean(repeat_sum_rates)),
            sum_rate_std=float(np.std(repeat_sum_rates)),
            bit_error_rate=float(np.mean(user_bit_error_rates)),
            user_rates=user_rate_accum / sample_average.num_repeats,
            user_bit_error_rates=user_bit_error_rates,
            user_rho=user_rho,
            leakage_matrix=leakage_matrix,
            offdiag_to_desired=offdiag_to_desired,
        )

    def evaluate_ucd_precoder_current_receiver_average_b_chain(
        self,
        user_channels: np.ndarray,
        f_rf: np.ndarray,
        f_bb: np.ndarray,
        q_chains: list[np.ndarray],
        r_chains: list[np.ndarray],
        snr_per_stream: float,
        bits_per_symbol: int,
        sample_average,
        apply_modulo: bool = True,
        labeling: str = "gray_standard",
    ) -> ReceiverAverageEvaluation:
        """Evaluate UCD exactly on the runtime effective matrix B = W^H H P."""

        symbols, _ = get_constellation(bits_per_symbol, labeling=labeling)
        effective_channels = self.build_effective_channels(user_channels, f_rf)
        user_blocks = self.split_user_blocks(f_bb)

        leakage_matrix = np.zeros((self.num_users, self.num_users), dtype=float)
        b_chains: list[list[np.ndarray]] = []
        receiver_covariances: list[np.ndarray] = []
        effective_diagonals: list[np.ndarray] = []
        user_rho = []

        for rx_user in range(self.num_users):
            w_rx = np.asarray(q_chains[rx_user], dtype=complex)
            user_b_blocks = []
            for tx_user in range(self.num_users):
                leakage_block = effective_channels[rx_user] @ user_blocks[tx_user]
                leakage_matrix[rx_user, tx_user] = float(np.linalg.norm(leakage_block, "fro") ** 2)
                user_b_blocks.append(w_rx.conj().T @ leakage_block)
            b_chains.append(user_b_blocks)
            noise_covariance = w_rx.conj().T @ w_rx
            receiver_covariances.append(noise_covariance)
            noise_variance = np.maximum(np.real(np.diag(noise_covariance)), 1e-12)
            b_diag = np.diag(user_b_blocks[rx_user])
            effective_diagonals.append(b_diag / np.sqrt(noise_variance))
            user_rho.append(
                np.maximum(
                    float(snr_per_stream) * (np.abs(b_diag) ** 2) / noise_variance,
                    1e-12,
                )
            )

        desired_power = float(np.trace(leakage_matrix))
        offdiag_power = float(np.sum(leakage_matrix) - desired_power)
        offdiag_to_desired = offdiag_power / max(desired_power, 1e-12)

        repeat_sum_rates = np.zeros(sample_average.num_repeats, dtype=float)
        user_rate_accum = np.zeros(self.num_users, dtype=float)
        user_ber_accum = np.zeros(self.num_users, dtype=float)
        sqrt_snr = float(np.sqrt(snr_per_stream))
        period = self.qam_modulo_period(bits_per_symbol)

        for repeat_idx, user_batches in enumerate(sample_average.batches):
            original_symbols = [symbols[batch.symbol_indices] for batch in user_batches]
            transmitted_symbols = []
            for user_idx in range(self.num_users):
                _, dp_cod_data, _ = thp_transmit_from_upper(
                    cod_data=original_symbols[user_idx].T,
                    upper_b=b_chains[user_idx][user_idx],
                    modem_bits=bits_per_symbol,
                )
                transmitted_symbols.append(dp_cod_data.T)

            for rx_user in range(self.num_users):
                noise_stream = np.asarray(user_batches[rx_user].noise, dtype=complex)
                noise_cov = receiver_covariances[rx_user]
                noise_cov = (noise_cov + noise_cov.conj().T) / 2.0
                chol_noise = np.linalg.cholesky(
                    noise_cov + 1e-12 * np.eye(noise_cov.shape[0], dtype=complex)
                )
                received = noise_stream @ chol_noise.T
                for tx_user in range(self.num_users):
                    received += sqrt_snr * (
                        transmitted_symbols[tx_user] @ b_chains[rx_user][tx_user].T
                    )

                y_equ = received.T / max(sqrt_snr, 1e-12)
                b_diag = np.diag(b_chains[rx_user][rx_user])
                y1 = y_equ / b_diag[:, None]
                dp_rx_data = centered_modulo_complex(y1, period) if apply_modulo else y1

                rate_value = estimate_bicm_gmi_thp_from_received(
                    diagonal_channel=effective_diagonals[rx_user],
                    snr_per_stream=snr_per_stream,
                    bits_per_symbol=bits_per_symbol,
                    symbol_indices=user_batches[rx_user].symbol_indices,
                    received_samples=dp_rx_data.T,
                    labeling=labeling,
                )
                ber_value = estimate_bit_error_rate_thp_from_received(
                    diagonal_channel=effective_diagonals[rx_user],
                    snr_per_stream=snr_per_stream,
                    bits_per_symbol=bits_per_symbol,
                    symbol_indices=user_batches[rx_user].symbol_indices,
                    received_samples=dp_rx_data.T,
                    labeling=labeling,
                )
                repeat_sum_rates[repeat_idx] += rate_value
                user_rate_accum[rx_user] += rate_value
                user_ber_accum[rx_user] += ber_value

        user_bit_error_rates = user_ber_accum / sample_average.num_repeats
        return ReceiverAverageEvaluation(
            sum_rate=float(np.mean(repeat_sum_rates)),
            sum_rate_std=float(np.std(repeat_sum_rates)),
            bit_error_rate=float(np.mean(user_bit_error_rates)),
            user_rates=user_rate_accum / sample_average.num_repeats,
            user_bit_error_rates=user_bit_error_rates,
            user_rho=user_rho,
            leakage_matrix=leakage_matrix,
            offdiag_to_desired=offdiag_to_desired,
        )





