from __future__ import annotations

import numpy as np

try:
    from _bootstrap import ensure_classical_on_path
except ModuleNotFoundError:
    from ._bootstrap import ensure_classical_on_path

ensure_classical_on_path()

from multiuser_simulation_environment import MultiUserSimulationEnvironment


class FullyDigitalMuMimoBicmEnvironment:
    """
    Fully-digital multi-user MIMO-BICM wrapper.

    The fully-digital model is obtained from the existing hybrid environment by
    fixing the analog precoder to the identity matrix. The user-wise structured
    baselines still follow the same digital-BD-first construction path.
    """

    def __init__(
        self,
        num_users: int,
        num_tx_antennas: int,
        num_rx_antennas: int,
        num_streams_per_user: int,
        channel_type: str = "cdl-a",
        digital_power_constraint: float | None = None,
        ucd_waterfill: bool = True,
        ucd_min_power_loading: float = 0.0,
    ):
        """????????????"""
        self.core = MultiUserSimulationEnvironment(
            num_users=num_users,
            num_tx_antennas=num_tx_antennas,
            num_rx_antennas=num_rx_antennas,
            num_rf_chains=num_tx_antennas,
            num_streams_per_user=num_streams_per_user,
            channel_type=channel_type,
            digital_power_constraint=digital_power_constraint,
            ucd_waterfill=ucd_waterfill,
            ucd_min_power_loading=ucd_min_power_loading,
        )

        self.num_users = self.core.num_users
        self.num_tx_antennas = self.core.num_tx_antennas
        self.num_rx_antennas = self.core.num_rx_antennas
        self.num_rf_chains = self.core.num_rf_chains
        self.num_streams_per_user = self.core.num_streams_per_user
        self.total_streams = self.core.total_streams
        self.digital_power_constraint = self.core.digital_power_constraint
        self.channel_type = channel_type

    def identity_precoder(self) -> np.ndarray:
        """?? identity precoder ???"""
        return np.eye(self.num_tx_antennas, dtype=complex)

    def generate_user_channels(self) -> np.ndarray:
        """?????user channels?"""
        return self.core.generate_user_channels()

    def split_user_blocks(self, f: np.ndarray) -> list[np.ndarray]:
        """?????user blocks?"""
        return self.core.split_user_blocks(np.asarray(f, dtype=complex))

    def normalize_precoder(self, f: np.ndarray) -> np.ndarray:
        """??????precoder?"""
        return self.core.normalize_digital_precoder(
            f_rf=self.identity_precoder(),
            f_bb=np.asarray(f, dtype=complex),
        )

    def build_bd_digital_basis(
        self,
        user_channels: np.ndarray,
        user_index: int,
    ) -> np.ndarray:
        """Build the first digital precoding layer T_k for user k."""
        return self.core.build_bd_digital_basis(
            effective_channels=np.asarray(user_channels, dtype=complex),
            user_index=user_index,
        )

    def build_structured_chain(
        self,
        user_channels: np.ndarray,
        snr_per_stream: float,
        strategy: str,
    ):
        """Build a full-digital structured P/Q/R chain."""
        return self.core.build_structured_digital_chain(
            user_channels=np.asarray(user_channels, dtype=complex),
            f_rf=self.identity_precoder(),
            snr_per_stream=snr_per_stream,
            strategy=strategy,
        )

    def evaluate_precoder_current_receiver_average_fixed_chain(
        self,
        user_channels: np.ndarray,
        f: np.ndarray,
        r_chains: list[np.ndarray],
        q_chains: list[np.ndarray],
        snr_per_stream: float,
        bits_per_symbol: int,
        sample_average,
        labeling: str = "gray_standard",
    ):
        """Evaluate a full-digital structured chain using design-stage P/Q/R and receive-side true diagonals."""
        return self.core.evaluate_precoder_current_receiver_average_fixed_chain(
            user_channels=np.asarray(user_channels, dtype=complex),
            f_rf=self.identity_precoder(),
            f_bb=np.asarray(f, dtype=complex),
            r_chains=r_chains,
            q_chains=q_chains,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            sample_average=sample_average,
            labeling=labeling,
        )

    def evaluate_precoder_current_receiver_average_parallel(
        self,
        user_channels: np.ndarray,
        f: np.ndarray,
        snr_per_stream: float,
        bits_per_symbol: int,
        sample_average,
        labeling: str = "gray_standard",
    ):
        """Evaluate the full-digital SVD branch with parallel per-stream detection."""
        return self.core.evaluate_precoder_current_receiver_average_parallel(
            user_channels=np.asarray(user_channels, dtype=complex),
            f_rf=self.identity_precoder(),
            f_bb=np.asarray(f, dtype=complex),
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            sample_average=sample_average,
            labeling=labeling,
        )

    def evaluate_precoder_current_receiver_average_thp(
        self,
        user_channels: np.ndarray,
        f: np.ndarray,
        snr_per_stream: float,
        bits_per_symbol: int,
        sample_average,
        labeling: str = "gray_standard",
    ):
        """Evaluate the full-digital GMD branch with the same THP receiver path as hybrid."""
        return self.core.evaluate_precoder_current_receiver_average_thp(
            user_channels=np.asarray(user_channels, dtype=complex),
            f_rf=self.identity_precoder(),
            f_bb=np.asarray(f, dtype=complex),
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            sample_average=sample_average,
            labeling=labeling,
        )

    def evaluate_ucd_precoder_current_receiver_average_b_chain(
        self,
        user_channels: np.ndarray,
        f: np.ndarray,
        q_chains: list[np.ndarray],
        r_chains: list[np.ndarray],
        snr_per_stream: float,
        bits_per_symbol: int,
        sample_average,
        labeling: str = "gray_standard",
    ):
        """Evaluate the full-digital UCD runtime chain using B = W^H H P."""
        return self.core.evaluate_ucd_precoder_current_receiver_average_b_chain(
            user_channels=np.asarray(user_channels, dtype=complex),
            f_rf=self.identity_precoder(),
            f_bb=np.asarray(f, dtype=complex),
            q_chains=q_chains,
            r_chains=r_chains,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            sample_average=sample_average,
            labeling=labeling,
        )

    def build_projected_bd_precoder(
        self,
        user_channels: np.ndarray,
        digital_blocks: np.ndarray,
    ) -> np.ndarray:
        """?????projected bd precoder?"""
        user_channels = np.asarray(user_channels, dtype=complex)
        digital_blocks = np.asarray(digital_blocks, dtype=complex)
        expected_shape = (
            self.num_users,
            self.num_tx_antennas,
            self.num_streams_per_user,
        )
        if digital_blocks.shape != expected_shape:
            raise ValueError(
                "digital_blocks must have shape "
                f"{expected_shape}, got {digital_blocks.shape}."
            )

        projected_blocks = []
        for user_idx in range(self.num_users):
            digital_bd_basis = self.build_bd_digital_basis(user_channels, user_idx)
            projector = digital_bd_basis @ digital_bd_basis.conj().T
            projected_blocks.append(projector @ digital_blocks[user_idx])

        f = np.hstack(projected_blocks)
        return self.normalize_precoder(f)

    def coeff_blocks_from_precoder(
        self,
        user_channels: np.ndarray,
        f: np.ndarray,
    ) -> list[np.ndarray]:
        """?? coeff blocks from precoder ???"""
        user_channels = np.asarray(user_channels, dtype=complex)
        coeff_blocks = []
        for user_idx, user_block in enumerate(self.split_user_blocks(f)):
            digital_bd_basis = self.build_bd_digital_basis(user_channels, user_idx)
            coeff_blocks.append(digital_bd_basis.conj().T @ user_block)
        return coeff_blocks
