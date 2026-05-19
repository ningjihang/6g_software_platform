from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianCsiEstimate:
    estimated_channels: np.ndarray
    error_channels: np.ndarray


def estimate_user_channels_with_gaussian_error(
    true_user_channels: np.ndarray,
    nmse_db: float,
    seed: int | None = None,
    per_user_normalization: bool = True,
) -> GaussianCsiEstimate:
    """?????user channels with gaussian error?"""
    true_user_channels = np.asarray(true_user_channels, dtype=complex)
    if true_user_channels.ndim != 3:
        raise ValueError(
            "true_user_channels must have shape (num_users, num_rx_antennas, num_tx_antennas)."
        )

    nmse_linear = float(10 ** (float(nmse_db) / 10.0))
    if nmse_linear < 0.0:
        raise ValueError(f"NMSE must be non-negative, got {nmse_linear}.")

    rng = np.random.default_rng(seed)
    raw_error = (
        rng.standard_normal(true_user_channels.shape)
        + 1j * rng.standard_normal(true_user_channels.shape)
    ) / np.sqrt(2.0)
    error_channels = np.zeros_like(true_user_channels, dtype=complex)

    if per_user_normalization:
        for user_idx in range(true_user_channels.shape[0]):
            signal_power = float(np.linalg.norm(true_user_channels[user_idx], "fro") ** 2)
            raw_error_power = float(np.linalg.norm(raw_error[user_idx], "fro") ** 2)
            target_error_power = nmse_linear * signal_power
            if raw_error_power <= 1e-12 or target_error_power <= 0.0:
                continue
            error_channels[user_idx] = raw_error[user_idx] * np.sqrt(
                target_error_power / raw_error_power
            )
    else:
        signal_power = float(np.linalg.norm(true_user_channels, "fro") ** 2)
        raw_error_power = float(np.linalg.norm(raw_error, "fro") ** 2)
        target_error_power = nmse_linear * signal_power
        if raw_error_power > 1e-12 and target_error_power > 0.0:
            error_channels = raw_error * np.sqrt(target_error_power / raw_error_power)

    estimated_channels = true_user_channels + error_channels
    return GaussianCsiEstimate(
        estimated_channels=estimated_channels,
        error_channels=error_channels,
    )
