from __future__ import annotations

import numpy as np
from scipy.linalg import svd


class AnalogPrecoder:
    """Shared-RF analog precoder builder under a constant-modulus constraint."""

    def __init__(self, num_tx_antennas: int, num_rf_chains: int):
        """????????????"""
        self.num_tx_antennas = int(num_tx_antennas)
        self.num_rf_chains = int(num_rf_chains)

    def build_precoder(
        self,
        user_channels: np.ndarray,
        num_streams_per_user: int,
    ) -> np.ndarray:
        """Build one shared RF precoder from user channel phase directions."""
        user_channels = np.asarray(user_channels, dtype=complex)
        if user_channels.ndim != 3:
            raise ValueError(
                "user_channels must have shape (num_users, num_rx_antennas, num_tx_antennas)."
            )
        if user_channels.shape[2] != self.num_tx_antennas:
            raise ValueError(
                "user_channels has incompatible transmit dimension: "
                f"expected {self.num_tx_antennas}, got {user_channels.shape[2]}."
            )

        num_users = int(user_channels.shape[0])
        if self.num_rf_chains < int(num_streams_per_user):
            raise ValueError(
                "Shared RF design requires num_rf_chains >= num_streams_per_user, got "
                f"{self.num_rf_chains} < {num_streams_per_user}."
            )

        candidate_columns = []
        for user_idx in range(num_users):
            _, _, vh_user = svd(user_channels[user_idx], full_matrices=False)
            user_basis = vh_user.conj().T
            candidate_columns.append(user_basis[:, : min(self.num_rf_chains, user_basis.shape[1])])

        phase_source = np.column_stack(candidate_columns)
        if phase_source.shape[1] < self.num_rf_chains:
            antenna_idx = np.arange(self.num_tx_antennas, dtype=float).reshape(-1, 1)
            extra = self.num_rf_chains - phase_source.shape[1]
            dft_idx = np.arange(extra, dtype=float).reshape(1, -1)
            filler_block = np.exp(2j * np.pi * antenna_idx * dft_idx / self.num_tx_antennas)
            phase_source = np.column_stack([phase_source, filler_block])

        phase_source = phase_source[:, : self.num_rf_chains]
        return np.exp(1j * np.angle(phase_source)) / np.sqrt(self.num_tx_antennas)
