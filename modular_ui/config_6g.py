from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FrequencyBand(str, Enum):
    SUB6G_3P5 = "3.5 GHz (Sub-6G)"
    MMWAVE_28 = "28 GHz (mmWave)"
    SUBTHZ_140 = "140 GHz (Sub-THz)"
    SUBTHZ_300 = "300 GHz (Sub-THz)"
    SUB6G = "3.5 GHz (Sub-6G)"
    MMWAVE = "28 GHz (mmWave)"

    @property
    def frequency_hz(self) -> float:
        return {
            FrequencyBand.SUB6G_3P5: 3.5e9,
            FrequencyBand.SUB6G: 3.5e9,
            FrequencyBand.MMWAVE_28: 28e9,
            FrequencyBand.MMWAVE: 28e9,
            FrequencyBand.SUBTHZ_140: 140e9,
            FrequencyBand.SUBTHZ_300: 300e9,
        }[self]

    @property
    def delay_spread_s(self) -> float:
        return {
            FrequencyBand.SUB6G_3P5: 30e-9,
            FrequencyBand.SUB6G: 30e-9,
            FrequencyBand.MMWAVE_28: 15e-9,
            FrequencyBand.MMWAVE: 15e-9,
            FrequencyBand.SUBTHZ_140: 5e-9,
            FrequencyBand.SUBTHZ_300: 2e-9,
        }[self]

    @property
    def sampling_frequency_hz(self) -> float:
        return {
            FrequencyBand.SUB6G_3P5: 30.72e6,
            FrequencyBand.SUB6G: 30.72e6,
            FrequencyBand.MMWAVE_28: 61.44e6,
            FrequencyBand.MMWAVE: 61.44e6,
            FrequencyBand.SUBTHZ_140: 122.88e6,
            FrequencyBand.SUBTHZ_300: 245.76e6,
        }[self]

    @property
    def is_sub_thz(self) -> bool:
        return self in {FrequencyBand.SUBTHZ_140, FrequencyBand.SUBTHZ_300}

    @classmethod
    def from_value(cls, value: str | FrequencyBand | float | int | None) -> FrequencyBand:
        if isinstance(value, FrequencyBand):
            return value
        if value is None:
            return cls.SUB6G_3P5

        text = str(value).strip()
        for band in cls:
            if text == band.value:
                return band

        compact = text.lower().replace("_", "").replace("-", "").replace(" ", "")
        aliases = {
            "3.5ghz(sub6g)": cls.SUB6G_3P5,
            "3.5ghz": cls.SUB6G_3P5,
            "sub6g": cls.SUB6G_3P5,
            "28ghz(mmwave)": cls.MMWAVE_28,
            "28ghz": cls.MMWAVE_28,
            "mmwave": cls.MMWAVE_28,
            "140ghz(subthz)": cls.SUBTHZ_140,
            "140ghz": cls.SUBTHZ_140,
            "subthz140": cls.SUBTHZ_140,
            "300ghz(subthz)": cls.SUBTHZ_300,
            "300ghz": cls.SUBTHZ_300,
            "subthz300": cls.SUBTHZ_300,
        }
        if compact in aliases:
            return aliases[compact]

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return cls.SUB6G_3P5
        if numeric >= 250e9:
            return cls.SUBTHZ_300
        if numeric >= 100e9:
            return cls.SUBTHZ_140
        if numeric >= 20e9:
            return cls.MMWAVE_28
        return cls.SUB6G_3P5


@dataclass(frozen=True, slots=True)
class XlMimoProfile:
    num_tx_antennas: int
    label: str
    is_xl_mimo: bool


XL_MIMO_PROFILES: tuple[XlMimoProfile, ...] = (
    XlMimoProfile(16, "16", False),
    XlMimoProfile(64, "64", False),
    XlMimoProfile(256, "256 (XL-MIMO)", True),
    XlMimoProfile(512, "512 (XL-MIMO)", True),
    XlMimoProfile(1024, "1024 (XL-MIMO)", True),
)


XL_MIMO_CONFIG: dict[str, dict[str, object]] = {
    str(profile.num_tx_antennas): {
        "label": profile.label,
        "xl": profile.is_xl_mimo,
    }
    for profile in XL_MIMO_PROFILES
}


def display_frequency_bands() -> list[str]:
    return [
        FrequencyBand.SUB6G_3P5.value,
        FrequencyBand.MMWAVE_28.value,
        FrequencyBand.SUBTHZ_140.value,
        FrequencyBand.SUBTHZ_300.value,
    ]


def is_xl_mimo(num_tx_antennas: int | str) -> bool:
    try:
        return int(num_tx_antennas) >= 256
    except (TypeError, ValueError):
        return False


def band_from_carrier_frequency(carrier_frequency_hz: float | int | None) -> FrequencyBand:
    return FrequencyBand.from_value(carrier_frequency_hz)


def resolve_6g_channel_profile(
    frequency_band: str | FrequencyBand | float | int | None,
) -> dict[str, float | bool | str]:
    band = FrequencyBand.from_value(frequency_band)
    return {
        "band_label": band.value,
        "carrier_frequency_hz": band.frequency_hz,
        "delay_spread_s": band.delay_spread_s,
        "sampling_frequency_hz": band.sampling_frequency_hz,
        "is_sub_thz": band.is_sub_thz,
    }
