from pathlib import Path
import sys
from functools import lru_cache

import numpy as np

try:
    from cdl_a_channel import (
        list_supported_sionna_channel_types,
        sample_cdl_a_channels_numpy,
        sample_sionna_channels_numpy,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from cdl_a_channel import (
        list_supported_sionna_channel_types,
        sample_cdl_a_channels_numpy,
        sample_sionna_channels_numpy,
    )


@lru_cache(maxsize=2048)
def _cached_sample_cdl_a(
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
    seed: int,
) -> np.ndarray:
    """?? cached sample cdl a ???"""
    channels = sample_cdl_a_channels_numpy(
        batch_size=batch_size,
        num_rx_antennas=num_rx_antennas,
        num_tx_antennas=num_tx_antennas,
        seed=seed,
    )
    # Freeze the cached value so callers cannot accidentally mutate shared state.
    channels = np.array(channels, copy=True)
    channels.setflags(write=False)
    return channels


class ChannelModel:
    """Project-wide channel generator. Defaults to 3GPP CDL-A."""

    def __init__(self, num_tx_antennas: int, num_rx_antennas: int, channel_type: str = "cdl-a"):
        """????????????"""
        self.num_tx_antennas = num_tx_antennas
        self.num_rx_antennas = num_rx_antennas
        self.channel_type = str(channel_type).strip().lower().replace("_", "-")

    def generate_channel(self) -> np.ndarray:
        """?????channel?"""
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
            )[0].copy()

        supported = ["cdl-a", "rayleigh", *list_supported_sionna_channel_types()]
        raise ValueError(
            f"Unsupported channel_type: {self.channel_type}. "
            f"Supported examples: {supported[:12]}"
        )
