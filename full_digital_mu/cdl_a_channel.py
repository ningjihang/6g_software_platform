from __future__ import annotations

import math
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import torch

try:
    from modular_ui.config_6g import FrequencyBand, resolve_6g_channel_profile
except Exception:  # pragma: no cover - optional dependency
    FrequencyBand = None
    resolve_6g_channel_profile = None

_MPL_CONFIG_DIR = Path(__file__).resolve().parent / ".mplconfig"
_MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

from sionna.phy.channel import cir_to_ofdm_channel, gen_single_sector_topology
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy import config as sionna_config
try:
    from sionna.phy.channel.tr38901 import TDL
except Exception:  # pragma: no cover
    TDL = None
try:
    from sionna.phy.channel.tr38901 import PanelArray, UMa
except Exception:  # pragma: no cover
    PanelArray = None
    UMa = None


DEFAULT_CDL_A_CARRIER_FREQUENCY = 3.5e9
DEFAULT_CDL_A_DELAY_SPREAD = 30e-9
DEFAULT_CDL_A_SAMPLING_FREQUENCY = 30.72e6
DEFAULT_TDL_DELAY_SPREAD = 30e-9
DEFAULT_TDL_SAMPLING_FREQUENCY = 30.72e6
DEFAULT_UMA_SAMPLING_FREQUENCY = 30.72e6
XL_MIMO_MIN_TX_ANTENNAS = 256


def is_xl_mimo_array(num_tx_antennas: int) -> bool:
    return int(num_tx_antennas) >= XL_MIMO_MIN_TX_ANTENNAS


def resolve_channel_profile(
    *,
    frequency_band: str | float | int | None = None,
    carrier_frequency: float | None = None,
    delay_spread: float | None = None,
    sampling_frequency: float | None = None,
) -> dict[str, float | bool | str]:
    """Resolve a frequency-dependent channel profile."""
    if carrier_frequency is not None:
        band_source = carrier_frequency
    else:
        band_source = frequency_band

    if resolve_6g_channel_profile is not None:
        profile = dict(resolve_6g_channel_profile(band_source))
    else:
        numeric = float(band_source) if band_source is not None else DEFAULT_CDL_A_CARRIER_FREQUENCY
        if numeric >= 250e9:
            profile = {
                "band_label": "300 GHz (Sub-THz)",
                "carrier_frequency_hz": 300e9,
                "delay_spread_s": 2e-9,
                "sampling_frequency_hz": 245.76e6,
                "is_sub_thz": True,
            }
        elif numeric >= 100e9:
            profile = {
                "band_label": "140 GHz (Sub-THz)",
                "carrier_frequency_hz": 140e9,
                "delay_spread_s": 5e-9,
                "sampling_frequency_hz": 122.88e6,
                "is_sub_thz": True,
            }
        elif numeric >= 20e9:
            profile = {
                "band_label": "28 GHz (mmWave)",
                "carrier_frequency_hz": 28e9,
                "delay_spread_s": 15e-9,
                "sampling_frequency_hz": 61.44e6,
                "is_sub_thz": False,
            }
        else:
            profile = {
                "band_label": "3.5 GHz (Sub-6G)",
                "carrier_frequency_hz": DEFAULT_CDL_A_CARRIER_FREQUENCY,
                "delay_spread_s": DEFAULT_CDL_A_DELAY_SPREAD,
                "sampling_frequency_hz": DEFAULT_CDL_A_SAMPLING_FREQUENCY,
                "is_sub_thz": False,
            }

    if carrier_frequency is not None:
        profile["carrier_frequency_hz"] = float(carrier_frequency)
    if delay_spread is not None:
        profile["delay_spread_s"] = float(delay_spread)
    if sampling_frequency is not None:
        profile["sampling_frequency_hz"] = float(sampling_frequency)
    return profile


def _factor_array_shape(num_antennas: int) -> tuple[int, int]:
    if num_antennas <= 0:
        raise ValueError(f"num_antennas must be positive, got {num_antennas}.")

    num_rows = int(math.sqrt(num_antennas))
    while num_rows > 1 and num_antennas % num_rows != 0:
        num_rows -= 1
    num_cols = num_antennas // num_rows
    return num_rows, num_cols


@lru_cache(maxsize=64)
def _build_cdl_model(
    model: str,
    num_tx_antennas: int,
    num_rx_antennas: int,
    carrier_frequency: float,
    delay_spread: float,
    min_speed: float,
    max_speed: float,
    device: str,
) -> CDL:
    tx_rows, tx_cols = _factor_array_shape(num_tx_antennas)
    rx_rows, rx_cols = _factor_array_shape(num_rx_antennas)

    tx_array = AntennaArray(
        num_rows=tx_rows,
        num_cols=tx_cols,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
        device=device,
    )
    rx_array = AntennaArray(
        num_rows=rx_rows,
        num_cols=rx_cols,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
        device=device,
    )

    return CDL(
        model=model.upper(),
        delay_spread=delay_spread,
        carrier_frequency=carrier_frequency,
        ut_array=tx_array,
        bs_array=rx_array,
        direction="uplink",
        min_speed=min_speed,
        max_speed=max_speed,
        device=device,
    )


@lru_cache(maxsize=64)
def _build_tdl_model(
    model: str,
    num_tx_antennas: int,
    num_rx_antennas: int,
    carrier_frequency: float,
    delay_spread: float,
    min_speed: float,
    max_speed: float,
    device: str,
):
    if TDL is None:
        raise RuntimeError("Sionna TDL is unavailable in the current environment.")
    return TDL(
        model=model.upper(),
        delay_spread=delay_spread,
        carrier_frequency=carrier_frequency,
        min_speed=min_speed,
        max_speed=max_speed,
        num_rx_ant=num_rx_antennas,
        num_tx_ant=num_tx_antennas,
        device=device,
    )


@lru_cache(maxsize=64)
def _build_uma_model(
    num_tx_antennas: int,
    num_rx_antennas: int,
    carrier_frequency: float,
    device: str,
):
    if UMa is None or PanelArray is None:
        raise RuntimeError("Sionna UMa is unavailable in the current environment.")

    tx_rows, tx_cols = _factor_array_shape(num_tx_antennas)
    rx_rows, rx_cols = _factor_array_shape(num_rx_antennas)

    bs_array = PanelArray(
        num_rows_per_panel=tx_rows,
        num_cols_per_panel=tx_cols,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
        device=device,
    )
    ut_array = PanelArray(
        num_rows_per_panel=rx_rows,
        num_cols_per_panel=rx_cols,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
        device=device,
    )

    return UMa(
        carrier_frequency=carrier_frequency,
        o2i_model="low",
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
        enable_pathloss=False,
        enable_shadow_fading=False,
        device=device,
    )


def _sample_with_optional_seed(
    sample_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    seed: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seed is None:
        return sample_fn()

    previous_seed = sionna_config.seed
    previous_py_state = (
        None if getattr(sionna_config, "_py_rng", None) is None else sionna_config._py_rng.getstate()
    )
    previous_np_state = (
        None
        if getattr(sionna_config, "_np_rng", None) is None
        else deepcopy(sionna_config._np_rng.bit_generator.state)
    )
    previous_torch_states = {
        device_name: generator.get_state()
        for device_name, generator in getattr(sionna_config, "_torch_rngs", {}).items()
    }
    with torch.random.fork_rng():
        try:
            sionna_config.seed = int(seed)
            return sample_fn()
        finally:
            sionna_config.seed = previous_seed
            if previous_py_state is not None and getattr(sionna_config, "_py_rng", None) is not None:
                sionna_config._py_rng.setstate(previous_py_state)
            if previous_np_state is not None and getattr(sionna_config, "_np_rng", None) is not None:
                sionna_config._np_rng.bit_generator.state = previous_np_state
            if getattr(sionna_config, "_torch_rngs", None) is not None:
                for device_name, state in previous_torch_states.items():
                    sionna_config._torch_rngs[device_name].set_state(state)


def _torch_channel_samples_to_numpy(
    channel_tensor: torch.Tensor,
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
) -> np.ndarray:
    channel_np = channel_tensor.detach().cpu().numpy()
    channel_np = np.asarray(channel_np)
    channel_np = np.squeeze(channel_np)

    if channel_np.ndim >= 4:
        axes = channel_np.shape
        tx_axis = next((idx for idx, size in enumerate(axes) if size == num_tx_antennas), None)
        rx_axis = next((idx for idx, size in enumerate(axes) if size == num_rx_antennas and idx != tx_axis), None)
        batch_axis = next((idx for idx, size in enumerate(axes) if size == batch_size and idx not in {tx_axis, rx_axis}), 0)
        if tx_axis is None or rx_axis is None:
            raise ValueError(
                f"Could not identify Tx/Rx axes in sampled channel with shape {channel_np.shape}."
            )
        channel_np = np.moveaxis(channel_np, (batch_axis, rx_axis, tx_axis), (0, 1, 2))
        while channel_np.ndim > 3:
            channel_np = channel_np[..., 0]
    elif channel_np.ndim == 3:
        if channel_np.shape != (batch_size, num_rx_antennas, num_tx_antennas):
            raise ValueError(
                f"Unexpected 3-D channel shape {channel_np.shape}; "
                f"expected {(batch_size, num_rx_antennas, num_tx_antennas)}."
            )
    elif channel_np.ndim == 2 and batch_size == 1:
        channel_np = channel_np[None, ...]
    else:
        raise ValueError(
            f"Unexpected channel tensor shape {channel_np.shape}; cannot form "
            f"(batch, Nr, Nt)=({batch_size}, {num_rx_antennas}, {num_tx_antennas})."
        )

    return np.asarray(channel_np, dtype=np.complex64)


def sample_cdl_a_channels_numpy(
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
    seed: int | None = None,
    carrier_frequency: float = DEFAULT_CDL_A_CARRIER_FREQUENCY,
    delay_spread: float = DEFAULT_CDL_A_DELAY_SPREAD,
    sampling_frequency: float = DEFAULT_CDL_A_SAMPLING_FREQUENCY,
) -> np.ndarray:
    return sample_sionna_channels_numpy(
        channel_type="cdl-a",
        batch_size=batch_size,
        num_rx_antennas=num_rx_antennas,
        num_tx_antennas=num_tx_antennas,
        seed=seed,
        carrier_frequency=carrier_frequency,
        delay_spread=delay_spread,
        sampling_frequency=sampling_frequency,
    )


def list_supported_sionna_channel_types() -> list[str]:
    base = ["sionna-cdl-a", "sionna-cdl-b", "sionna-cdl-c", "sionna-cdl-d", "sionna-cdl-e"]
    if TDL is not None:
        base.extend(["sionna-tdl-a", "sionna-tdl-b", "sionna-tdl-c", "sionna-tdl-d", "sionna-tdl-e"])
    if UMa is not None and PanelArray is not None:
        base.append("sionna-uma")
    return base


def sample_sionna_channels_numpy(
    channel_type: str,
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
    seed: int | None = None,
    carrier_frequency: float = DEFAULT_CDL_A_CARRIER_FREQUENCY,
    delay_spread: float = DEFAULT_CDL_A_DELAY_SPREAD,
    sampling_frequency: float = DEFAULT_CDL_A_SAMPLING_FREQUENCY,
) -> np.ndarray:
    normalized = str(channel_type).strip().lower().replace("_", "-")
    if normalized == "cdl-a":
        normalized = "sionna-cdl-a"

    device = "cpu"

    if normalized.startswith("sionna-cdl-"):
        model_name = normalized.split("sionna-cdl-", 1)[1]
        channel_model = _build_cdl_model(
            model=model_name,
            num_tx_antennas=num_tx_antennas,
            num_rx_antennas=num_rx_antennas,
            carrier_frequency=float(carrier_frequency),
            delay_spread=float(delay_spread),
            min_speed=0.0,
            max_speed=0.0,
            device=device,
        )

        def _sample() -> tuple[torch.Tensor, torch.Tensor]:
            a, tau = channel_model(batch_size=int(batch_size), num_time_steps=1, sampling_frequency=float(sampling_frequency))
            return a, tau

        a, tau = _sample_with_optional_seed(_sample, seed)
        h = cir_to_ofdm_channel(
            frequencies=torch.zeros(1, dtype=torch.float32),
            a=a,
            tau=tau,
            normalize=True,
        )
        return _torch_channel_samples_to_numpy(h[..., 0], batch_size, num_rx_antennas, num_tx_antennas)

    if normalized.startswith("sionna-tdl-"):
        model_name = normalized.split("sionna-tdl-", 1)[1]
        channel_model = _build_tdl_model(
            model=model_name,
            num_tx_antennas=num_tx_antennas,
            num_rx_antennas=num_rx_antennas,
            carrier_frequency=float(carrier_frequency),
            delay_spread=float(delay_spread),
            min_speed=0.0,
            max_speed=0.0,
            device=device,
        )

        def _sample() -> tuple[torch.Tensor, torch.Tensor]:
            a, tau = channel_model(batch_size=int(batch_size), num_time_steps=1, sampling_frequency=float(sampling_frequency))
            return a, tau

        a, tau = _sample_with_optional_seed(_sample, seed)
        h = cir_to_ofdm_channel(
            frequencies=torch.zeros(1, dtype=torch.float32),
            a=a,
            tau=tau,
            normalize=True,
        )
        return _torch_channel_samples_to_numpy(h[..., 0], batch_size, num_rx_antennas, num_tx_antennas)

    if normalized == "sionna-uma":
        channel_model = _build_uma_model(
            num_tx_antennas=num_tx_antennas,
            num_rx_antennas=num_rx_antennas,
            carrier_frequency=float(carrier_frequency),
            device=device,
        )

        def _sample() -> tuple[torch.Tensor, torch.Tensor]:
            topology = gen_single_sector_topology(
                batch_size=int(batch_size),
                num_ut=1,
                scenario="uma",
                min_ut_velocity=0.0,
                max_ut_velocity=0.0,
            )
            channel_model.set_topology(*topology)
            a, tau = channel_model(batch_size=int(batch_size), num_time_steps=1, sampling_frequency=float(sampling_frequency))
            return a, tau

        a, tau = _sample_with_optional_seed(_sample, seed)
        h = cir_to_ofdm_channel(
            frequencies=torch.zeros(1, dtype=torch.float32),
            a=a,
            tau=tau,
            normalize=True,
        )
        return _torch_channel_samples_to_numpy(h[..., 0], batch_size, num_rx_antennas, num_tx_antennas)

    raise ValueError(
        f"Unsupported Sionna channel type {channel_type!r}. "
        f"Supported types: {list_supported_sionna_channel_types()}"
    )
