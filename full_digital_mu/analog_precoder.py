from scipy.linalg import svd
import numpy as np


class AnalogPrecoder:
    """Shared-RF analog precoder builder under a constant-modulus constraint."""

    def __init__(self, num_tx_antennas: int, num_rf_chains: int):
        self.num_tx_antennas = num_tx_antennas
        self.num_rf_chains = num_rf_chains

    def build_precoder(
        self,
        user_channels: np.ndarray,
        num_streams_per_user: int,
    ) -> np.ndarray:
        """Build one shared RF precoder from user channel phase directions."""
        if self.num_rf_chains % len(user_channels) != 0:
            raise ValueError(
                "Dedicated per-user RF requires num_rf_chains divisible by the number of users."
            )

        rf_per_user = self.num_rf_chains // len(user_channels)
        if rf_per_user < num_streams_per_user:
            raise ValueError(
                "Each user must own at least num_streams_per_user RF chains."
            )

        f_rf_blocks = []
        for user_idx in range(len(user_channels)):
            _, _, vh_user = svd(user_channels[user_idx], full_matrices=False)
            v_user = vh_user.conj().T[:, :rf_per_user]
            phase_only = np.exp(1j * np.angle(v_user)) / np.sqrt(self.num_tx_antennas)
            f_rf_blocks.append(np.asarray(phase_only, dtype=complex))

        return np.hstack(f_rf_blocks)
