from functools import lru_cache

import numpy as np

try:
    from cdl_a_channel import (
        list_supported_sionna_channel_types,
        resolve_channel_profile,
        sample_cdl_a_channels_numpy,
        sample_sionna_channels_numpy,
    )
except ModuleNotFoundError:
    from .cdl_a_channel import (
        list_supported_sionna_channel_types,
        resolve_channel_profile,
        sample_cdl_a_channels_numpy,
        sample_sionna_channels_numpy,
    )


@lru_cache(maxsize=2048)
def _cached_sample_cdl_a(
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
    seed: int,
    carrier_frequency: float,
    delay_spread: float,
    sampling_frequency: float,
) -> np.ndarray:
    """Cache immutable CDL-A samples to reduce repeated setup overhead."""
    channels = sample_cdl_a_channels_numpy(
        batch_size=batch_size,
        num_rx_antennas=num_rx_antennas,
        num_tx_antennas=num_tx_antennas,
        seed=seed,
        carrier_frequency=carrier_frequency,
        delay_spread=delay_spread,
        sampling_frequency=sampling_frequency,
    )
    channels = np.array(channels, copy=True)
    channels.setflags(write=False)
    return channels


class ChannelModel:
    """Full-digital channel generator. Defaults to 3GPP CDL-A."""

    def __init__(
        self,
        num_tx_antennas: int,
        num_rx_antennas: int,
        channel_type: str = "cdl-a",
        *,
        frequency_band: str | float | int | None = None,
        carrier_frequency: float | None = None,
        delay_spread: float | None = None,
        sampling_frequency: float | None = None,
    ):
        self.num_tx_antennas = num_tx_antennas
        self.num_rx_antennas = num_rx_antennas
        self.channel_type = str(channel_type).strip().lower().replace("_", "-")
        self.channel_profile = resolve_channel_profile(
            frequency_band=frequency_band,
            carrier_frequency=carrier_frequency,
            delay_spread=delay_spread,
            sampling_frequency=sampling_frequency,
        )

    def generate_channel(self) -> np.ndarray:
        if self.channel_type == "rayleigh":
            return (
                np.random.randn(self.num_rx_antennas, self.num_tx_antennas)
                + 1j * np.random.randn(self.num_rx_antennas, self.num_tx_antennas)
            ) / np.sqrt(2.0)

        if self.channel_type == "cdl-a":
            seed = int(np.random.randint(0, 2**31 - 1))
            return _cached_sample_cdl_a(
                batch_size=1,
                num_rx_antennas=self.num_rx_antennas,
                num_tx_antennas=self.num_tx_antennas,
                seed=seed,
                carrier_frequency=float(self.channel_profile["carrier_frequency_hz"]),
                delay_spread=float(self.channel_profile["delay_spread_s"]),
                sampling_frequency=float(self.channel_profile["sampling_frequency_hz"]),
            )[0].copy()

        normalized = self.channel_type
        if normalized.startswith("cdl-") and normalized != "cdl-a":
            normalized = f"sionna-{normalized}"
        if normalized.startswith("tdl-"):
            normalized = f"sionna-{normalized}"
        if normalized == "uma":
            normalized = "sionna-uma"

        if normalized.startswith("sionna-"):
            seed = int(np.random.randint(0, 2**31 - 1))
            return sample_sionna_channels_numpy(
                channel_type=normalized,
                batch_size=1,
                num_rx_antennas=self.num_rx_antennas,
                num_tx_antennas=self.num_tx_antennas,
                seed=seed,
                carrier_frequency=float(self.channel_profile["carrier_frequency_hz"]),
                delay_spread=float(self.channel_profile["delay_spread_s"]),
                sampling_frequency=float(self.channel_profile["sampling_frequency_hz"]),
            )[0].copy()

        supported = ["cdl-a", "rayleigh", *list_supported_sionna_channel_types()]
        raise ValueError(
            f"Unsupported channel_type: {self.channel_type}. "
            f"Supported examples: {supported[:12]}"
        )
