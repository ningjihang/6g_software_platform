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
except Exception:  # pragma: no cover - modular UI is optional in some contexts
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
except Exception:  # pragma: no cover - optional in some environments
    TDL = None
try:
    from sionna.phy.channel.tr38901 import PanelArray, UMa
except Exception:  # pragma: no cover - optional in some environments
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
    """
    Resolve a frequency-dependent channel profile for Sub-6G/mmWave/Sub-THz runs.
    """
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
    """?? factor array shape ???"""
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
    """Build and cache a Sionna CDL channel model."""
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
    """Build and cache a Sionna TDL channel model."""
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
    """Build and cache a Sionna UMa channel model."""
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
            if previous_py_state is None:
                sionna_config._py_rng = None
            else:
                _ = sionna_config.py_rng
                sionna_config._py_rng.setstate(previous_py_state)
            if previous_np_state is None:
                sionna_config._np_rng = None
            else:
                _ = sionna_config.np_rng
                sionna_config._np_rng.bit_generator.state = previous_np_state
            for device_name, state in previous_torch_states.items():
                sionna_config._torch_rngs[device_name].set_state(state)


def _channel_matrix_from_cir(
    path_coefficients: torch.Tensor,
    path_delays: torch.Tensor,
    out_device: torch.device,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    frequencies = torch.zeros(1, dtype=torch.float32, device=path_coefficients.device)
    channel_frequency = cir_to_ofdm_channel(
        frequencies=frequencies,
        a=path_coefficients,
        tau=path_delays,
        normalize=True,
    )
    channel_matrix = channel_frequency[:, 0, :, 0, :, 0, 0]
    return channel_matrix.to(device=out_device, dtype=out_dtype)


def sample_cdl_channels_torch(
    model: str,
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.complex64,
    carrier_frequency: float = DEFAULT_CDL_A_CARRIER_FREQUENCY,
    delay_spread: float = DEFAULT_CDL_A_DELAY_SPREAD,
    sampling_frequency: float = DEFAULT_CDL_A_SAMPLING_FREQUENCY,
    min_speed: float = 0.0,
    max_speed: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample Sionna CDL channels for a given CDL model name."""
    torch_device = torch.device(device)
    sionna_device = "cpu"
    cdl_model = _build_cdl_model(
        model=model,
        num_tx_antennas=num_tx_antennas,
        num_rx_antennas=num_rx_antennas,
        carrier_frequency=float(carrier_frequency),
        delay_spread=float(delay_spread),
        min_speed=float(min_speed),
        max_speed=float(max_speed),
        device=sionna_device,
    )

    path_coefficients, path_delays = _sample_with_optional_seed(
        sample_fn=lambda: cdl_model(
            batch_size=batch_size,
            num_time_steps=1,
            sampling_frequency=float(sampling_frequency),
        ),
        seed=seed,
    )
    return _channel_matrix_from_cir(path_coefficients, path_delays, out_device=torch_device, out_dtype=dtype)


def sample_cdl_a_channels_torch(
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.complex64,
    carrier_frequency: float = DEFAULT_CDL_A_CARRIER_FREQUENCY,
    delay_spread: float = DEFAULT_CDL_A_DELAY_SPREAD,
    sampling_frequency: float = DEFAULT_CDL_A_SAMPLING_FREQUENCY,
    min_speed: float = 0.0,
    max_speed: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Backward-compatible wrapper for CDL-A sampling."""
    return sample_cdl_channels_torch(
        model="A",
        batch_size=batch_size,
        num_rx_antennas=num_rx_antennas,
        num_tx_antennas=num_tx_antennas,
        device=device,
        dtype=dtype,
        carrier_frequency=carrier_frequency,
        delay_spread=delay_spread,
        sampling_frequency=sampling_frequency,
        min_speed=min_speed,
        max_speed=max_speed,
        seed=seed,
    )


def sample_tdl_channels_torch(
    model: str,
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.complex64,
    carrier_frequency: float = DEFAULT_CDL_A_CARRIER_FREQUENCY,
    delay_spread: float = DEFAULT_TDL_DELAY_SPREAD,
    sampling_frequency: float = DEFAULT_TDL_SAMPLING_FREQUENCY,
    min_speed: float = 0.0,
    max_speed: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample Sionna TDL channels for a given TDL model name."""
    torch_device = torch.device(device)
    sionna_device = "cpu"
    tdl_model = _build_tdl_model(
        model=model,
        num_tx_antennas=num_tx_antennas,
        num_rx_antennas=num_rx_antennas,
        carrier_frequency=float(carrier_frequency),
        delay_spread=float(delay_spread),
        min_speed=float(min_speed),
        max_speed=float(max_speed),
        device=sionna_device,
    )

    path_coefficients, path_delays = _sample_with_optional_seed(
        sample_fn=lambda: tdl_model(
            batch_size=batch_size,
            num_time_steps=1,
            sampling_frequency=float(sampling_frequency),
        ),
        seed=seed,
    )
    return _channel_matrix_from_cir(path_coefficients, path_delays, out_device=torch_device, out_dtype=dtype)


def sample_uma_channels_torch(
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.complex64,
    carrier_frequency: float = DEFAULT_CDL_A_CARRIER_FREQUENCY,
    sampling_frequency: float = DEFAULT_UMA_SAMPLING_FREQUENCY,
    min_speed: float = 0.0,
    max_speed: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample Sionna Urban Macro channels."""
    torch_device = torch.device(device)
    sionna_device = "cpu"
    uma_model = _build_uma_model(
        num_tx_antennas=num_tx_antennas,
        num_rx_antennas=num_rx_antennas,
        carrier_frequency=float(carrier_frequency),
        device=sionna_device,
    )

    def sample_fn() -> tuple[torch.Tensor, torch.Tensor]:
        topology = gen_single_sector_topology(
            batch_size=batch_size,
            num_ut=1,
            scenario="uma",
            min_ut_velocity=float(min_speed),
            max_ut_velocity=float(max_speed),
            device=sionna_device,
        )
        uma_model.set_topology(*topology)
        return uma_model(
            num_time_samples=1,
            sampling_frequency=float(sampling_frequency),
        )

    path_coefficients, path_delays = _sample_with_optional_seed(
        sample_fn=sample_fn,
        seed=seed,
    )
    return _channel_matrix_from_cir(path_coefficients, path_delays, out_device=torch_device, out_dtype=dtype)


def sample_cdl_channels_numpy(
    model: str,
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
    seed: int,
    carrier_frequency: float = DEFAULT_CDL_A_CARRIER_FREQUENCY,
    delay_spread: float = DEFAULT_CDL_A_DELAY_SPREAD,
    sampling_frequency: float = DEFAULT_CDL_A_SAMPLING_FREQUENCY,
    min_speed: float = 0.0,
    max_speed: float = 0.0,
) -> np.ndarray:
    """NumPy wrapper for `sample_cdl_channels_torch`."""
    channel_matrix = sample_cdl_channels_torch(
        model=model,
        batch_size=batch_size,
        num_rx_antennas=num_rx_antennas,
        num_tx_antennas=num_tx_antennas,
        device=torch.device("cpu"),
        dtype=torch.complex64,
        carrier_frequency=carrier_frequency,
        delay_spread=delay_spread,
        sampling_frequency=sampling_frequency,
        min_speed=min_speed,
        max_speed=max_speed,
        seed=seed,
    )
    return channel_matrix.cpu().numpy()


def sample_cdl_a_channels_numpy(
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
    seed: int,
    carrier_frequency: float = DEFAULT_CDL_A_CARRIER_FREQUENCY,
    delay_spread: float = DEFAULT_CDL_A_DELAY_SPREAD,
    sampling_frequency: float = DEFAULT_CDL_A_SAMPLING_FREQUENCY,
    min_speed: float = 0.0,
    max_speed: float = 0.0,
) -> np.ndarray:
    """Backward-compatible NumPy wrapper for CDL-A sampling."""
    return sample_cdl_channels_numpy(
        model="A",
        batch_size=batch_size,
        num_rx_antennas=num_rx_antennas,
        num_tx_antennas=num_tx_antennas,
        seed=seed,
        carrier_frequency=carrier_frequency,
        delay_spread=delay_spread,
        sampling_frequency=sampling_frequency,
        min_speed=min_speed,
        max_speed=max_speed,
    )


def sample_tdl_channels_numpy(
    model: str,
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
    seed: int,
    carrier_frequency: float = DEFAULT_CDL_A_CARRIER_FREQUENCY,
    delay_spread: float = DEFAULT_TDL_DELAY_SPREAD,
    sampling_frequency: float = DEFAULT_TDL_SAMPLING_FREQUENCY,
    min_speed: float = 0.0,
    max_speed: float = 0.0,
) -> np.ndarray:
    """NumPy wrapper for `sample_tdl_channels_torch`."""
    channel_matrix = sample_tdl_channels_torch(
        model=model,
        batch_size=batch_size,
        num_rx_antennas=num_rx_antennas,
        num_tx_antennas=num_tx_antennas,
        device=torch.device("cpu"),
        dtype=torch.complex64,
        carrier_frequency=carrier_frequency,
        delay_spread=delay_spread,
        sampling_frequency=sampling_frequency,
        min_speed=min_speed,
        max_speed=max_speed,
        seed=seed,
    )
    return channel_matrix.cpu().numpy()


def sample_uma_channels_numpy(
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
    seed: int,
    carrier_frequency: float = DEFAULT_CDL_A_CARRIER_FREQUENCY,
    sampling_frequency: float = DEFAULT_UMA_SAMPLING_FREQUENCY,
    min_speed: float = 0.0,
    max_speed: float = 0.0,
) -> np.ndarray:
    """NumPy wrapper for `sample_uma_channels_torch`."""
    channel_matrix = sample_uma_channels_torch(
        batch_size=batch_size,
        num_rx_antennas=num_rx_antennas,
        num_tx_antennas=num_tx_antennas,
        device=torch.device("cpu"),
        dtype=torch.complex64,
        carrier_frequency=carrier_frequency,
        sampling_frequency=sampling_frequency,
        min_speed=min_speed,
        max_speed=max_speed,
        seed=seed,
    )
    return channel_matrix.cpu().numpy()


@lru_cache(maxsize=1)
def _probe_supported_cdl_models() -> tuple[str, ...]:
    supported: list[str] = []
    for candidate in [chr(code) for code in range(ord("A"), ord("Z") + 1)]:
        try:
            _ = _build_cdl_model(
                model=candidate,
                num_tx_antennas=1,
                num_rx_antennas=1,
                carrier_frequency=DEFAULT_CDL_A_CARRIER_FREQUENCY,
                delay_spread=DEFAULT_CDL_A_DELAY_SPREAD,
                min_speed=0.0,
                max_speed=0.0,
                device="cpu",
            )
            supported.append(candidate)
        except Exception:
            continue
    return tuple(supported)


@lru_cache(maxsize=1)
def _probe_supported_tdl_models() -> tuple[str, ...]:
    if TDL is None:
        return tuple()

    supported: list[str] = []
    candidates = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    candidates.extend([f"{head}{tail}" for head, tail in [("A", "30"), ("B", "100"), ("C", "300")]])
    for candidate in candidates:
        try:
            _ = _build_tdl_model(
                model=candidate,
                num_tx_antennas=1,
                num_rx_antennas=1,
                carrier_frequency=DEFAULT_CDL_A_CARRIER_FREQUENCY,
                delay_spread=DEFAULT_TDL_DELAY_SPREAD,
                min_speed=0.0,
                max_speed=0.0,
                device="cpu",
            )
            supported.append(candidate)
        except Exception:
            continue
    return tuple(supported)


def list_supported_sionna_channel_types() -> list[str]:
    """List channel-type strings that can be sampled via Sionna in this env."""
    channel_types: list[str] = []
    channel_types.extend([f"sionna-cdl-{model.lower()}" for model in _probe_supported_cdl_models()])
    channel_types.extend([f"sionna-tdl-{model.lower()}" for model in _probe_supported_tdl_models()])
    if UMa is not None and PanelArray is not None:
        channel_types.append("sionna-uma")
    return channel_types


def _parse_sionna_channel_type(channel_type: str) -> tuple[str, str]:
    text = str(channel_type).strip().lower().replace("_", "-")
    if text.startswith("sionna-"):
        text = text[len("sionna-") :]
    if text.startswith("cdl-"):
        return "cdl", text.split("-", 1)[1].upper()
    if text.startswith("tdl-"):
        return "tdl", text.split("-", 1)[1].upper()
    if text == "uma":
        return "uma", "UMa"
    raise ValueError(
        f"Unsupported Sionna channel_type: {channel_type}. "
        "Expected one of 'sionna-cdl-*', 'sionna-tdl-*', or 'sionna-uma'."
    )


def sample_sionna_channels_torch(
    channel_type: str,
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.complex64,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample channels for a Sionna channel-type string."""
    family, model = _parse_sionna_channel_type(channel_type)
    if family == "cdl":
        return sample_cdl_channels_torch(
            model=model,
            batch_size=batch_size,
            num_rx_antennas=num_rx_antennas,
            num_tx_antennas=num_tx_antennas,
            device=device,
            dtype=dtype,
            seed=seed,
        )
    if family == "tdl":
        return sample_tdl_channels_torch(
            model=model,
            batch_size=batch_size,
            num_rx_antennas=num_rx_antennas,
            num_tx_antennas=num_tx_antennas,
            device=device,
            dtype=dtype,
            seed=seed,
        )
    if family == "uma":
        return sample_uma_channels_torch(
            batch_size=batch_size,
            num_rx_antennas=num_rx_antennas,
            num_tx_antennas=num_tx_antennas,
            device=device,
            dtype=dtype,
            seed=seed,
        )
    raise ValueError(f"Unsupported Sionna channel family: {family}")


def sample_sionna_channels_numpy(
    channel_type: str,
    batch_size: int,
    num_rx_antennas: int,
    num_tx_antennas: int,
    seed: int,
) -> np.ndarray:
    """NumPy wrapper for `sample_sionna_channels_torch`."""
    channel_matrix = sample_sionna_channels_torch(
        channel_type=channel_type,
        batch_size=batch_size,
        num_rx_antennas=num_rx_antennas,
        num_tx_antennas=num_tx_antennas,
        device=torch.device("cpu"),
        dtype=torch.complex64,
        seed=seed,
    )
    return channel_matrix.cpu().numpy()
