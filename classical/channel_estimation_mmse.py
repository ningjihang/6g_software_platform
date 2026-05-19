from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MmsePilotCsiEstimate:
    estimated_channels: np.ndarray
    pilot_observations: np.ndarray
    pilot_noise: np.ndarray
    pilot_matrix: np.ndarray
    prior_variances: np.ndarray


@dataclass(frozen=True)
class SampleChannelCovariance:
    covariance: np.ndarray
    mean: np.ndarray
    diagonal_loading: float
    num_samples: int


def build_repeated_identity_pilot(
    num_tx_antennas: int,
    pilot_length: int,
) -> np.ndarray:
    """?????repeated identity pilot?"""
    if pilot_length < num_tx_antennas:
        raise ValueError(
            "pilot_length must be at least num_tx_antennas for orthogonal pilot estimation: "
            f"{pilot_length} < {num_tx_antennas}"
        )
    if pilot_length % num_tx_antennas != 0:
        raise ValueError(
            "pilot_length must be a multiple of num_tx_antennas for the repeated-identity pilot used here: "
            f"{pilot_length} % {num_tx_antennas} != 0"
        )

    repeats = pilot_length // num_tx_antennas
    return np.tile(np.eye(num_tx_antennas, dtype=complex), repeats)


def estimate_channel_covariance_from_model(
    channel_model,
    num_samples: int,
    seed: int | None = None,
    diagonal_loading: float = 1e-6,
) -> SampleChannelCovariance:
    """?????channel covariance from model?"""
    if num_samples <= 1:
        raise ValueError(f"num_samples must be greater than 1, got {num_samples}.")
    if diagonal_loading < 0.0:
        raise ValueError(f"diagonal_loading must be non-negative, got {diagonal_loading}.")

    rng_state = np.random.get_state()
    try:
        if seed is not None:
            np.random.seed(seed)
        samples = []
        for _ in range(num_samples):
            channel = np.asarray(channel_model.generate_channel(), dtype=complex)
            samples.append(channel.reshape(-1, order="F"))
    finally:
        np.random.set_state(rng_state)

    sample_matrix = np.asarray(samples, dtype=complex)
    sample_mean = np.mean(sample_matrix, axis=0)
    centered = sample_matrix - sample_mean[None, :]
    # Each row stores one vec(H)^T sample, so we need E[h h^H] with the
    # conjugation applied to the second factor rather than the first.
    covariance = (centered.T @ centered.conj()) / float(num_samples)
    if diagonal_loading > 0.0:
        scale = float(np.trace(covariance).real / max(covariance.shape[0], 1))
        covariance = covariance + (diagonal_loading * max(scale, 1e-12)) * np.eye(
            covariance.shape[0],
            dtype=complex,
        )

    return SampleChannelCovariance(
        covariance=covariance,
        mean=sample_mean,
        diagonal_loading=float(diagonal_loading),
        num_samples=int(num_samples),
    )


def estimate_user_channels_with_mmse_pilots_full_covariance(
    true_user_channels: np.ndarray,
    channel_covariance: np.ndarray,
    channel_mean: np.ndarray,
    pilot_length: int,
    pilot_snr_db: float,
    seed: int | None = None,
) -> MmsePilotCsiEstimate:
    """?????user channels with mmse pilots full covariance?"""
    true_user_channels = np.asarray(true_user_channels, dtype=complex)
    if true_user_channels.ndim != 3:
        raise ValueError(
            "true_user_channels must have shape (num_users, num_rx_antennas, num_tx_antennas)."
        )

    num_users, num_rx_antennas, num_tx_antennas = true_user_channels.shape
    dim = num_rx_antennas * num_tx_antennas
    channel_covariance = np.asarray(channel_covariance, dtype=complex)
    if channel_covariance.shape != (dim, dim):
        raise ValueError(
            "channel_covariance must have shape "
            f"({dim}, {dim}), got {channel_covariance.shape}."
        )
    channel_mean = np.asarray(channel_mean, dtype=complex).reshape(-1)
    if channel_mean.shape != (dim,):
        raise ValueError(
            f"channel_mean must have shape ({dim},), got {channel_mean.shape}."
        )

    pilot_matrix = build_repeated_identity_pilot(
        num_tx_antennas=num_tx_antennas,
        pilot_length=pilot_length,
    )
    pilot_snr_linear = float(10 ** (float(pilot_snr_db) / 10.0))
    noise_variance = 1.0 / max(pilot_snr_linear, 1e-12)
    rng = np.random.default_rng(seed)

    pilot_noise = (
        rng.standard_normal((num_users, num_rx_antennas, pilot_length))
        + 1j * rng.standard_normal((num_users, num_rx_antennas, pilot_length))
    ) / np.sqrt(2.0)
    pilot_noise = np.sqrt(noise_variance) * pilot_noise
    pilot_observations = true_user_channels @ pilot_matrix + pilot_noise

    observation_matrix = np.kron(pilot_matrix.T, np.eye(num_rx_antennas, dtype=complex))
    innovation = (
        observation_matrix @ channel_covariance @ observation_matrix.conj().T
        + noise_variance * np.eye(num_rx_antennas * pilot_length, dtype=complex)
    )
    mmse_gain = channel_covariance @ observation_matrix.conj().T @ np.linalg.inv(innovation)

    estimated_channels = np.zeros_like(true_user_channels, dtype=complex)
    prior_mean_observation = observation_matrix @ channel_mean
    for user_idx in range(num_users):
        observation_vector = pilot_observations[user_idx].reshape(-1, order="F")
        channel_vector = channel_mean + mmse_gain @ (observation_vector - prior_mean_observation)
        estimated_channels[user_idx] = channel_vector.reshape(
            num_rx_antennas,
            num_tx_antennas,
            order="F",
        )

    return MmsePilotCsiEstimate(
        estimated_channels=estimated_channels,
        pilot_observations=pilot_observations,
        pilot_noise=pilot_noise,
        pilot_matrix=pilot_matrix,
        prior_variances=np.full(num_users, np.nan, dtype=float),
    )
