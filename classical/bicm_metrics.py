import itertools
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio

from thp_precoding import qam_axis_order_and_spacing


def stable_logsumexp(x: np.ndarray) -> float:
    """?? stable logsumexp ???"""
    x = np.asarray(x, dtype=float)
    x_max = np.max(x)
    return float(x_max + np.log(np.sum(np.exp(x - x_max))))


def stable_binary_logloss_from_signed_llr(signed_llr: float) -> float:
    """?? stable binary logloss from signed llr ???"""
    return float(np.logaddexp(0.0, -float(signed_llr)) / np.log(2.0))


@dataclass(frozen=True)
class SampleBatch:
    symbol_indices: np.ndarray
    noise: np.ndarray
    bit_matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        """?????????????????"""
        if self.symbol_indices.ndim != 2:
            raise ValueError("symbol_indices must have shape (num_samples, num_streams).")
        if self.noise.shape != self.symbol_indices.shape:
            raise ValueError("noise must match symbol_indices shape.")
        if self.bit_matrix is not None:
            if self.bit_matrix.ndim != 3:
                raise ValueError(
                    "bit_matrix must have shape (num_samples, num_streams, bits_per_symbol)."
                )
            if self.bit_matrix.shape[:2] != self.symbol_indices.shape:
                raise ValueError(
                    "bit_matrix leading dimensions must match symbol_indices shape."
                )


def _normalize_labeling(labeling: str) -> str:
    normalized = str(labeling).strip().lower()
    if normalized not in {"gray_standard", "nr_like"}:
        raise ValueError(f"Unsupported labeling={labeling!r}.")
    return normalized


@lru_cache(maxsize=4)
def _load_nr_like_mappings() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    mapping_path = Path(__file__).resolve().parents[1] / "tools" / "nr_modulation_mappings_from_matlab.mat"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"NR-like mapping file not found: {mapping_path}. "
            "Run tools/export_nr_modulation_mappings_matlab.m first."
        )

    mat = sio.loadmat(mapping_path)
    output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for bits_per_symbol, bits_key, symbols_key in (
        (4, "bits_16qam", "symbols_16qam"),
        (6, "bits_64qam", "symbols_64qam"),
        (8, "bits_256qam", "symbols_256qam"),
    ):
        if bits_key not in mat or symbols_key not in mat:
            continue
        bits = np.asarray(mat[bits_key], dtype=int)
        symbols = np.asarray(mat[symbols_key], dtype=complex).reshape(-1)
        if bits.shape[0] != symbols.shape[0]:
            if bits_per_symbol == 8 and symbols.size != bits.shape[0]:
                continue
            raise ValueError(
                f"Loaded mapping has inconsistent shapes for {bits_per_symbol}-bit modulation: "
                f"bits={bits.shape}, symbols={symbols.shape}."
            )
        output[bits_per_symbol] = (symbols, bits)
    return output


def modulate_bits_with_labeling(
    bits_per_symbol: int,
    bits: np.ndarray,
    labeling: str = "gray_standard",
) -> np.ndarray:
    """Modulate bits using the selected labeling."""

    bits_array = np.asarray(bits, dtype=int)
    original_shape = bits_array.shape
    if bits_array.ndim == 1:
        if bits_array.size % bits_per_symbol != 0:
            raise ValueError("Flat bit vector length must be divisible by bits_per_symbol.")
        bits_rows = bits_array.reshape(-1, bits_per_symbol)
    else:
        if bits_array.shape[-1] != bits_per_symbol:
            raise ValueError("Last bit dimension must equal bits_per_symbol.")
        bits_rows = bits_array.reshape(-1, bits_per_symbol)

    symbols, bit_table = get_constellation(bits_per_symbol, labeling=labeling)
    symbol_lookup = {
        tuple(int(v) for v in bit_row): symbol
        for bit_row, symbol in zip(bit_table, symbols)
    }
    symbols = np.asarray(
        [symbol_lookup[tuple(int(v) for v in row)] for row in bits_rows],
        dtype=complex,
    )

    if bits_array.ndim == 1:
        return symbols
    return symbols.reshape(original_shape[:-1])


def nr_modulation_fudan(bits_per_symbol: int, bits: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for the MATLAB/NR-like labeling branch."""

    return modulate_bits_with_labeling(bits_per_symbol, bits, labeling="nr_like")


def generate_sample_batch(
    bits_per_symbol: int,
    num_streams: int,
    num_samples: int = 256,
    seed: int | None = None,
    labeling: str = "gray_standard",
) -> SampleBatch:
    """Generate a shared Monte Carlo sample batch."""
    labeling = _normalize_labeling(labeling)
    symbols, _ = get_constellation(bits_per_symbol, labeling=labeling)
    rng = np.random.default_rng(seed)
    bit_matrix = rng.integers(
        0,
        2,
        size=(num_samples, num_streams, bits_per_symbol),
        dtype=int,
    )
    transmitted_symbols = modulate_bits_with_labeling(
        bits_per_symbol=bits_per_symbol,
        bits=bit_matrix,
        labeling=labeling,
    )
    distance = np.abs(
        transmitted_symbols.reshape(-1, 1) - symbols.reshape(1, -1)
    ) ** 2
    symbol_indices = np.argmin(distance, axis=1).reshape(num_samples, num_streams)
    noise = (
        rng.standard_normal((num_samples, num_streams))
        + 1j * rng.standard_normal((num_samples, num_streams))
    ) / np.sqrt(2.0)
    return SampleBatch(
        symbol_indices=np.asarray(symbol_indices, dtype=int),
        noise=np.asarray(noise, dtype=complex),
        bit_matrix=np.asarray(bit_matrix, dtype=int),
    )


def get_constellation(
    bits_per_symbol: int,
    labeling: str = "gray_standard",
) -> tuple[np.ndarray, np.ndarray]:
    """?? get constellation ???"""
    labeling = _normalize_labeling(labeling)

    if labeling == "nr_like":
        mappings = _load_nr_like_mappings()
        if bits_per_symbol in mappings:
            symbols, bits = mappings[bits_per_symbol]
            return np.asarray(symbols, dtype=complex), np.asarray(bits, dtype=int)
        if bits_per_symbol == 8:
            # MATLAB NR_modulation uses qammod(..., 256, 'InputType','bit'),
            # which is effectively the toolbox default Gray labeling.
            pass

    if bits_per_symbol == 2:
        symbols = np.array(
            [
                -1.0 - 1.0j,
                -1.0 + 1.0j,
                1.0 + 1.0j,
                1.0 - 1.0j,
            ],
            dtype=complex,
        ) / np.sqrt(2.0)
        bits = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 1],
                [1, 0],
            ],
            dtype=int,
        )
        return symbols, bits

    if bits_per_symbol == 4:
        pam_gray = [
            (-3.0, (0, 0)),
            (-1.0, (0, 1)),
            (1.0, (1, 1)),
            (3.0, (1, 0)),
        ]
        symbols = []
        bits = []
        for real_amp, real_bits in pam_gray:
            for imag_amp, imag_bits in pam_gray:
                symbols.append((real_amp + 1j * imag_amp) / np.sqrt(10.0))
                bits.append(real_bits + imag_bits)
        return np.asarray(symbols, dtype=complex), np.asarray(bits, dtype=int)

    if bits_per_symbol == 6:
        pam_gray = [
            (-7.0, (0, 0, 0)),
            (-5.0, (0, 0, 1)),
            (-3.0, (0, 1, 1)),
            (-1.0, (0, 1, 0)),
            (1.0, (1, 1, 0)),
            (3.0, (1, 1, 1)),
            (5.0, (1, 0, 1)),
            (7.0, (1, 0, 0)),
        ]
        symbols = []
        bits = []
        for real_amp, real_bits in pam_gray:
            for imag_amp, imag_bits in pam_gray:
                symbols.append((real_amp + 1j * imag_amp) / np.sqrt(42.0))
                bits.append(real_bits + imag_bits)
        return np.asarray(symbols, dtype=complex), np.asarray(bits, dtype=int)

    if bits_per_symbol == 8:
        # 256-QAM via Gray-coded 16-PAM on I/Q axes.
        pam_gray = []
        bits_per_axis = bits_per_symbol // 2
        for idx, amplitude in enumerate(np.arange(-15.0, 16.0, 2.0)):
            gray_value = idx ^ (idx >> 1)
            gray_bits = tuple(
                (gray_value >> bit_idx) & 1
                for bit_idx in range(bits_per_axis - 1, -1, -1)
            )
            pam_gray.append((float(amplitude), gray_bits))

        symbols = []
        bits = []
        for real_amp, real_bits in pam_gray:
            for imag_amp, imag_bits in pam_gray:
                symbols.append((real_amp + 1j * imag_amp) / np.sqrt(170.0))
                bits.append(real_bits + imag_bits)
        return np.asarray(symbols, dtype=complex), np.asarray(bits, dtype=int)

    raise ValueError(
        f"Unsupported bits_per_symbol={bits_per_symbol}. "
        "Only QPSK/16-QAM/64-QAM/256-QAM are supported."
    )


def get_constellation_fudan_64qam() -> tuple[np.ndarray, np.ndarray]:
    """Return the 64-QAM constellation aligned with Fudan `NR_modulation.m`."""

    axis_map = [
        (-7.0, (1, 1, 1)),
        (-5.0, (1, 1, 0)),
        (-3.0, (1, 0, 0)),
        (-1.0, (1, 0, 1)),
        (1.0, (0, 0, 1)),
        (3.0, (0, 0, 0)),
        (5.0, (0, 1, 0)),
        (7.0, (0, 1, 1)),
    ]
    symbols = []
    for real_amp, real_bits in axis_map:
        for imag_amp, imag_bits in axis_map:
            symbols.append((real_amp + 1j * imag_amp) / np.sqrt(42.0))
    symbols = np.asarray(symbols, dtype=complex)
    bits = np.asarray([real_bits + imag_bits for real_amp, real_bits in axis_map for imag_amp, imag_bits in axis_map], dtype=int)
    return symbols, bits


def soft_demod_fudan_dp_v2(bits_per_symbol: int, qam_data: np.ndarray) -> np.ndarray:
    """
    1:1 replica of the MATLAB `softDemod_dp_v2` slicer logic.

    Parameters
    ----------
    bits_per_symbol:
        Modulation order in bits per complex symbol. The 64-QAM branch
        (`bits_per_symbol == 6`) is implemented to match the MATLAB code.
    qam_data:
        Complex samples with shape `(num_samples, num_streams)` or flattened.

    Returns
    -------
    np.ndarray
        Soft bits with shape `(bits_per_symbol, num_total_samples)`.
    """

    y = np.asarray(qam_data, dtype=complex).flatten()
    length = y.size
    rx = np.real(y)
    ix = np.imag(y)
    soft_bits = np.zeros((bits_per_symbol, length), dtype=float)

    if bits_per_symbol != 6:
        raise ValueError(
            "soft_demod_fudan_dp_v2 currently supports only 64-QAM "
            f"(bits_per_symbol=6), got {bits_per_symbol}."
        )

    a = np.sqrt(3.0 / 2.0 / 63.0)

    def _soft_metric(samples: np.ndarray, bit_id: int) -> np.ndarray:
        if bit_id in (0, 1):
            t = np.floor(samples / (2.0 * a)).astype(int)
            rem = np.mod(t, 8)
            base = 2.0 * a * t.astype(float)
            b0 = np.empty_like(samples, dtype=float)
            b1 = np.empty_like(samples, dtype=float)

            if bit_id == 0:
                mask = rem == 1
                b0[mask] = a + base[mask]
                b1[mask] = -3.0 * a + base[mask]
                mask = rem == 2
                b0[mask] = a + base[mask]
                b1[mask] = 5.0 * a + base[mask]
                mask = rem == 3
                b0[mask] = a + base[mask]
                b1[mask] = 3.0 * a + base[mask]
                mask = rem == 4
                b0[mask] = -a + base[mask]
                b1[mask] = a + base[mask]
                mask = rem == 5
                b0[mask] = -3.0 * a + base[mask]
                b1[mask] = a + base[mask]
                mask = rem == 6
                b0[mask] = 5.0 * a + base[mask]
                b1[mask] = a + base[mask]
                mask = rem == 7
                b0[mask] = 3.0 * a + base[mask]
                b1[mask] = a + base[mask]
                mask = rem == 0
                b0[mask] = a + base[mask]
                b1[mask] = -a + base[mask]
            else:
                mask = rem == 1
                b0[mask] = a + base[mask]
                b1[mask] = 3.0 * a + base[mask]
                mask = rem == 2
                b0[mask] = -a + base[mask]
                b1[mask] = a + base[mask]
                mask = rem == 3
                b0[mask] = -3.0 * a + base[mask]
                b1[mask] = a + base[mask]
                mask = rem == 4
                b0[mask] = 5.0 * a + base[mask]
                b1[mask] = a + base[mask]
                mask = rem == 5
                b0[mask] = 3.0 * a + base[mask]
                b1[mask] = a + base[mask]
                mask = rem == 6
                b0[mask] = a + base[mask]
                b1[mask] = -a + base[mask]
                mask = rem == 7
                b0[mask] = a + base[mask]
                b1[mask] = -3.0 * a + base[mask]
                mask = rem == 0
                b0[mask] = a + base[mask]
                b1[mask] = 5.0 * a + base[mask]

        elif bit_id == 2:
            t = np.floor(samples / (4.0 * a)).astype(int)
            rem = np.mod(t, 2)
            base = 4.0 * a * t.astype(float)
            b0 = np.where(rem == 1, a + base, 3.0 * a + base)
            b1 = np.where(rem == 1, 3.0 * a + base, a + base)
        else:
            raise ValueError(f"Unsupported 64-QAM axis bit_id={bit_id}.")

        return -(samples - b0) ** 2 + (samples - b1) ** 2

    soft_bits[0, :] = _soft_metric(rx, 0)
    soft_bits[1, :] = _soft_metric(ix, 0)
    soft_bits[2, :] = _soft_metric(rx, 1)
    soft_bits[3, :] = _soft_metric(ix, 1)
    soft_bits[4, :] = _soft_metric(rx, 2)
    soft_bits[5, :] = _soft_metric(ix, 2)
    return soft_bits


@lru_cache(maxsize=32)
def enumerate_symbol_vectors(
    bits_per_symbol: int,
    num_streams: int,
    labeling: str = "gray_standard",
) -> tuple[np.ndarray, np.ndarray]:
    """?? enumerate symbol vectors ???"""
    symbols, bits = get_constellation(bits_per_symbol, labeling=labeling)
    constellation_size = len(symbols)
    total_candidates = constellation_size ** num_streams
    if total_candidates > 20000:
        raise ValueError(
            "Exact vector enumeration is too large for max-log LLR evaluation: "
            f"{constellation_size}^{num_streams} = {total_candidates}. "
            "Reduce per-user streams or modulation order."
        )

    index_grid = np.asarray(list(itertools.product(range(constellation_size), repeat=num_streams)), dtype=int)
    vector_symbols = symbols[index_grid]
    vector_bits = bits[index_grid]
    return vector_symbols, vector_bits


def estimate_bicm_gmi(
    upper_triangular_channel: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    num_samples: int = 256,
    seed: int | None = None,
) -> float:
    """?????bicm gmi?"""
    num_streams = upper_triangular_channel.shape[1]
    vector_symbols, vector_bits = enumerate_symbol_vectors(bits_per_symbol, num_streams)
    num_candidates = vector_symbols.shape[0]

    scaled_channel = np.sqrt(snr_per_stream) * upper_triangular_channel
    codewords = (scaled_channel @ vector_symbols.T).T

    bit_masks = {}
    for stream_idx in range(num_streams):
        for bit_idx in range(bits_per_symbol):
            bit_values = vector_bits[:, stream_idx, bit_idx]
            mask_one = bit_values == 1
            mask_zero = ~mask_one
            bit_masks[(stream_idx, bit_idx)] = (mask_zero, mask_one)

    rng = np.random.default_rng(seed)
    ideal_rate = float(num_streams * bits_per_symbol)
    loss_sum = 0.0

    for _ in range(num_samples):
        candidate_index = int(rng.integers(0, num_candidates))
        transmitted_vector = vector_symbols[candidate_index]
        transmitted_bits = vector_bits[candidate_index]
        noise = (
            rng.standard_normal(num_streams) + 1j * rng.standard_normal(num_streams)
        ) / np.sqrt(2.0)
        received = scaled_channel @ transmitted_vector + noise

        distances = np.sum(np.abs(received[None, :] - codewords) ** 2, axis=1)

        for stream_idx in range(num_streams):
            for bit_idx in range(bits_per_symbol):
                mask_zero, mask_one = bit_masks[(stream_idx, bit_idx)]
                min_zero = np.min(distances[mask_zero])
                min_one = np.min(distances[mask_one])
                llr = min_zero - min_one
                bit_value = transmitted_bits[stream_idx, bit_idx]
                signed_llr = (2 * bit_value - 1) * llr
                loss_sum += stable_binary_logloss_from_signed_llr(signed_llr)

    return max(0.0, ideal_rate - loss_sum / max(num_samples, 1))


def estimate_symbol_mi(
    upper_triangular_channel: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    num_samples: int = 256,
    seed: int | None = None,
) -> float:
    """?????symbol mi?"""
    num_streams = upper_triangular_channel.shape[1]
    vector_symbols, _ = enumerate_symbol_vectors(bits_per_symbol, num_streams)
    num_candidates = vector_symbols.shape[0]

    scaled_channel = np.sqrt(snr_per_stream) * upper_triangular_channel
    codewords = (scaled_channel @ vector_symbols.T).T

    rng = np.random.default_rng(seed)
    log_cardinality = np.log2(num_candidates)
    penalty_sum = 0.0

    for _ in range(num_samples):
        candidate_index = int(rng.integers(0, num_candidates))
        transmitted_vector = vector_symbols[candidate_index]
        noise = (
            rng.standard_normal(num_streams) + 1j * rng.standard_normal(num_streams)
        ) / np.sqrt(2.0)
        received = scaled_channel @ transmitted_vector + noise

        distances = np.sum(np.abs(received[None, :] - codewords) ** 2, axis=1)
        reference = distances[candidate_index]
        penalty_sum += stable_logsumexp(-(distances - reference)) / np.log(2.0)

    return max(0.0, log_cardinality - penalty_sum / max(num_samples, 1))


def estimate_scalar_bicm_gmi(
    channel_gain: complex,
    snr_per_stream: float,
    bits_per_symbol: int,
    num_samples: int = 256,
    seed: int | None = None,
) -> float:
    """?????scalar bicm gmi?"""
    symbols, bits = get_constellation(bits_per_symbol)
    codewords = np.sqrt(snr_per_stream) * channel_gain * symbols
    bit_masks = {}
    for bit_idx in range(bits_per_symbol):
        bit_values = bits[:, bit_idx]
        mask_one = bit_values == 1
        mask_zero = ~mask_one
        bit_masks[bit_idx] = (mask_zero, mask_one)

    rng = np.random.default_rng(seed)
    ideal_rate = float(bits_per_symbol)
    loss_sum = 0.0

    for _ in range(num_samples):
        candidate_index = int(rng.integers(0, len(symbols)))
        transmitted_symbol = symbols[candidate_index]
        transmitted_bits = bits[candidate_index]
        noise = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2.0)
        received = np.sqrt(snr_per_stream) * channel_gain * transmitted_symbol + noise
        distances = np.abs(received - codewords) ** 2

        for bit_idx in range(bits_per_symbol):
            mask_zero, mask_one = bit_masks[bit_idx]
            llr = stable_logsumexp(-distances[mask_one]) - stable_logsumexp(-distances[mask_zero])
            bit_value = transmitted_bits[bit_idx]
            signed_llr = (2 * bit_value - 1) * llr
            loss_sum += stable_binary_logloss_from_signed_llr(signed_llr)

    return max(0.0, ideal_rate - loss_sum / max(num_samples, 1))


def estimate_scalar_symbol_mi(
    channel_gain: complex,
    snr_per_stream: float,
    bits_per_symbol: int,
    num_samples: int = 256,
    seed: int | None = None,
) -> float:
    """?????scalar symbol mi?"""
    symbols, _ = get_constellation(bits_per_symbol)
    codewords = np.sqrt(snr_per_stream) * channel_gain * symbols
    rng = np.random.default_rng(seed)
    log_cardinality = np.log2(len(symbols))
    penalty_sum = 0.0

    for _ in range(num_samples):
        candidate_index = int(rng.integers(0, len(symbols)))
        transmitted_symbol = symbols[candidate_index]
        noise = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2.0)
        received = np.sqrt(snr_per_stream) * channel_gain * transmitted_symbol + noise
        distances = np.abs(received - codewords) ** 2
        reference = distances[candidate_index]
        penalty_sum += stable_logsumexp(-(distances - reference)) / np.log(2.0)

    return max(0.0, log_cardinality - penalty_sum / max(num_samples, 1))


def estimate_bicm_gmi_streamwise(
    upper_triangular_channel: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    num_samples: int = 256,
    seed: int | None = None,
) -> float:
    """?????bicm gmi streamwise?"""
    diag_entries = np.diag(upper_triangular_channel)
    total_gmi = 0.0
    for stream_idx, gain in enumerate(diag_entries):
        total_gmi += estimate_scalar_bicm_gmi(
            channel_gain=gain,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            num_samples=num_samples,
            seed=None if seed is None else seed + stream_idx,
    )
    return float(total_gmi)


def estimate_bicm_gmi_sic_from_received(
    upper_triangular_channel: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    symbol_indices: np.ndarray,
    received_samples: np.ndarray,
    labeling: str = "gray_standard",
) -> float:
    """Estimate BICM-GMI for SIC detection from projected received samples."""

    num_streams = upper_triangular_channel.shape[1]
    symbols, bits = get_constellation(bits_per_symbol, labeling=labeling)
    symbol_indices = np.asarray(symbol_indices, dtype=int)
    received_samples = np.asarray(received_samples, dtype=complex)

    if symbol_indices.ndim != 2:
        raise ValueError("symbol_indices must have shape (num_samples, num_streams).")
    if symbol_indices.shape[1] != num_streams:
        raise ValueError(
            "symbol_indices do not match the upper-triangular channel width: "
            f"{symbol_indices.shape[1]} != {num_streams}."
        )
    if received_samples.shape != symbol_indices.shape:
        raise ValueError(
            "received_samples must match symbol_indices shape after receiver projection: "
            f"{received_samples.shape} != {symbol_indices.shape}."
        )

    bit_masks = {}
    for bit_idx in range(bits_per_symbol):
        bit_values = bits[:, bit_idx]
        mask_one = bit_values == 1
        mask_zero = ~mask_one
        bit_masks[bit_idx] = (mask_zero, mask_one)

    transmitted_bits = bits[symbol_indices]
    scaled_channel = np.sqrt(snr_per_stream) * upper_triangular_channel
    ideal_rate = float(num_streams * bits_per_symbol)
    loss_sum = 0.0
    total_samples = symbol_indices.shape[0]

    for sample_idx in range(total_samples):
        received = received_samples[sample_idx]
        sample_bits = transmitted_bits[sample_idx]
        detected_symbols = np.zeros(num_streams, dtype=complex)

        for stream_idx in range(num_streams - 1, -1, -1):
            interference = 0.0j
            if stream_idx + 1 < num_streams:
                interference = np.dot(
                    scaled_channel[stream_idx, stream_idx + 1 :],
                    detected_symbols[stream_idx + 1 :],
                )
            residual = received[stream_idx] - interference
            gain = scaled_channel[stream_idx, stream_idx]
            distances = np.abs(residual - gain * symbols) ** 2
            hard_index = int(np.argmin(distances))
            detected_symbols[stream_idx] = symbols[hard_index]

            for bit_idx in range(bits_per_symbol):
                mask_zero, mask_one = bit_masks[bit_idx]
                llr = stable_logsumexp(-distances[mask_one]) - stable_logsumexp(-distances[mask_zero])
                bit_value = sample_bits[stream_idx, bit_idx]
                signed_llr = (2 * bit_value - 1) * llr
                loss_sum += stable_binary_logloss_from_signed_llr(signed_llr)

    return max(0.0, ideal_rate - loss_sum / max(total_samples, 1))


def estimate_bit_error_rate_sic_from_received(
    upper_triangular_channel: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    symbol_indices: np.ndarray,
    received_samples: np.ndarray,
    labeling: str = "gray_standard",
) -> float:
    """Estimate BER for SIC detection from projected received samples."""

    num_streams = upper_triangular_channel.shape[1]
    symbols, bits = get_constellation(bits_per_symbol, labeling=labeling)
    symbol_indices = np.asarray(symbol_indices, dtype=int)
    received_samples = np.asarray(received_samples, dtype=complex)

    if symbol_indices.ndim != 2:
        raise ValueError("symbol_indices must have shape (num_samples, num_streams).")
    if symbol_indices.shape[1] != num_streams:
        raise ValueError(
            "symbol_indices do not match the upper-triangular channel width: "
            f"{symbol_indices.shape[1]} != {num_streams}."
        )
    if received_samples.shape != symbol_indices.shape:
        raise ValueError(
            "received_samples must match symbol_indices shape after receiver projection: "
            f"{received_samples.shape} != {symbol_indices.shape}."
        )

    transmitted_bits = bits[symbol_indices]
    scaled_channel = np.sqrt(snr_per_stream) * upper_triangular_channel
    total_samples = symbol_indices.shape[0]
    total_bits = int(total_samples * num_streams * bits_per_symbol)
    bit_errors = 0

    for sample_idx in range(total_samples):
        received = received_samples[sample_idx]
        detected_indices = np.zeros(num_streams, dtype=int)

        for stream_idx in range(num_streams - 1, -1, -1):
            interference = 0.0j
            if stream_idx + 1 < num_streams:
                interference = np.dot(
                    scaled_channel[stream_idx, stream_idx + 1 :],
                    symbols[detected_indices[stream_idx + 1 :]],
                )
            residual = received[stream_idx] - interference
            gain = scaled_channel[stream_idx, stream_idx]
            distances = np.abs(residual - gain * symbols) ** 2
            detected_indices[stream_idx] = int(np.argmin(distances))

        detected_bits = bits[detected_indices]
        bit_errors += int(np.count_nonzero(detected_bits != transmitted_bits[sample_idx]))

    return float(bit_errors / max(total_bits, 1))


def estimate_bicm_gmi_parallel_from_received(
    diagonal_channel: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    symbol_indices: np.ndarray,
    received_samples: np.ndarray,
    labeling: str = "gray_standard",
) -> float:
    """Estimate BICM-GMI for parallel streams without SIC."""

    diagonal_channel = np.asarray(diagonal_channel, dtype=complex)
    if diagonal_channel.ndim == 2:
        diag_entries = np.diag(diagonal_channel)
    elif diagonal_channel.ndim == 1:
        diag_entries = diagonal_channel
    else:
        raise ValueError("diagonal_channel must be a diagonal matrix or a 1-D diagonal vector.")

    symbols, bits = get_constellation(bits_per_symbol, labeling=labeling)
    symbol_indices = np.asarray(symbol_indices, dtype=int)
    received_samples = np.asarray(received_samples, dtype=complex)

    if symbol_indices.ndim != 2:
        raise ValueError("symbol_indices must have shape (num_samples, num_streams).")
    if symbol_indices.shape[1] != len(diag_entries):
        raise ValueError(
            "symbol_indices do not match the diagonal channel width: "
            f"{symbol_indices.shape[1]} != {len(diag_entries)}."
        )
    if received_samples.shape != symbol_indices.shape:
        raise ValueError(
            "received_samples must match symbol_indices shape after receiver projection: "
            f"{received_samples.shape} != {symbol_indices.shape}."
        )

    bit_masks = {}
    for bit_idx in range(bits_per_symbol):
        bit_values = bits[:, bit_idx]
        mask_one = bit_values == 1
        mask_zero = ~mask_one
        bit_masks[bit_idx] = (mask_zero, mask_one)

    transmitted_bits = bits[symbol_indices]
    sqrt_snr = float(np.sqrt(snr_per_stream))
    ideal_rate = float(symbol_indices.shape[1] * bits_per_symbol)
    loss_sum = 0.0
    total_samples = symbol_indices.shape[0]

    for sample_idx in range(total_samples):
        received = received_samples[sample_idx]
        sample_bits = transmitted_bits[sample_idx]
        for stream_idx, gain in enumerate(diag_entries):
            distances = np.abs(received[stream_idx] - sqrt_snr * gain * symbols) ** 2
            for bit_idx in range(bits_per_symbol):
                mask_zero, mask_one = bit_masks[bit_idx]
                llr = stable_logsumexp(-distances[mask_one]) - stable_logsumexp(-distances[mask_zero])
                bit_value = sample_bits[stream_idx, bit_idx]
                signed_llr = (2 * bit_value - 1) * llr
                loss_sum += stable_binary_logloss_from_signed_llr(signed_llr)

    return max(0.0, ideal_rate - loss_sum / max(total_samples, 1))


def estimate_bit_error_rate_parallel_from_received(
    diagonal_channel: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    symbol_indices: np.ndarray,
    received_samples: np.ndarray,
    labeling: str = "gray_standard",
) -> float:
    """Estimate BER for parallel per-stream detection from projected samples."""

    diagonal_channel = np.asarray(diagonal_channel, dtype=complex)
    if diagonal_channel.ndim == 2:
        diag_entries = np.diag(diagonal_channel)
    elif diagonal_channel.ndim == 1:
        diag_entries = diagonal_channel
    else:
        raise ValueError("diagonal_channel must be a diagonal matrix or a 1-D diagonal vector.")

    symbols, bits = get_constellation(bits_per_symbol, labeling=labeling)
    symbol_indices = np.asarray(symbol_indices, dtype=int)
    received_samples = np.asarray(received_samples, dtype=complex)

    if symbol_indices.ndim != 2:
        raise ValueError("symbol_indices must have shape (num_samples, num_streams).")
    if symbol_indices.shape[1] != len(diag_entries):
        raise ValueError(
            "symbol_indices do not match the diagonal channel width: "
            f"{symbol_indices.shape[1]} != {len(diag_entries)}."
        )
    if received_samples.shape != symbol_indices.shape:
        raise ValueError(
            "received_samples must match symbol_indices shape after receiver projection: "
            f"{received_samples.shape} != {symbol_indices.shape}."
        )

    transmitted_bits = bits[symbol_indices]
    sqrt_snr = float(np.sqrt(snr_per_stream))
    total_samples = symbol_indices.shape[0]
    total_bits = int(total_samples * len(diag_entries) * bits_per_symbol)
    bit_errors = 0

    for sample_idx in range(total_samples):
        received = received_samples[sample_idx]
        sample_bits = transmitted_bits[sample_idx]
        for stream_idx, gain in enumerate(diag_entries):
            distances = np.abs(received[stream_idx] - sqrt_snr * gain * symbols) ** 2
            hard_index = int(np.argmin(distances))
            bit_errors += int(np.count_nonzero(bits[hard_index] != sample_bits[stream_idx]))

    return float(bit_errors / max(total_bits, 1))


@lru_cache(maxsize=32)
def enumerate_thp_symbol_replicas(
    bits_per_symbol: int,
    replica_radius: int,
    labeling: str = "gray_standard",
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate periodic constellation replicas used by modulo receivers."""

    if replica_radius < 0:
        raise ValueError(f"replica_radius must be non-negative, got {replica_radius}.")

    symbols, bits = get_constellation(bits_per_symbol, labeling=labeling)
    m_axis, spacing = qam_axis_order_and_spacing(bits_per_symbol)
    period = m_axis * spacing

    replica_symbols = []
    replica_bits = []
    for real_shift in range(-replica_radius, replica_radius + 1):
        for imag_shift in range(-replica_radius, replica_radius + 1):
            shift = period * (real_shift + 1j * imag_shift)
            replica_symbols.append(symbols + shift)
            replica_bits.append(bits)
    return np.concatenate(replica_symbols), np.concatenate(replica_bits, axis=0)


def estimate_bicm_gmi_thp_from_received(
    diagonal_channel: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    symbol_indices: np.ndarray,
    received_samples: np.ndarray,
    replica_radius: int = 1,
    labeling: str = "gray_standard",
) -> float:
    """Estimate BICM-GMI for THP after modulo equalization using periodic replicas."""

    diagonal_channel = np.asarray(diagonal_channel, dtype=complex)
    if diagonal_channel.ndim == 2:
        diag_entries = np.diag(diagonal_channel)
    elif diagonal_channel.ndim == 1:
        diag_entries = diagonal_channel
    else:
        raise ValueError("diagonal_channel must be a diagonal matrix or a 1-D diagonal vector.")

    symbol_indices = np.asarray(symbol_indices, dtype=int)
    received_samples = np.asarray(received_samples, dtype=complex)
    if symbol_indices.ndim != 2:
        raise ValueError("symbol_indices must have shape (num_samples, num_streams).")
    if received_samples.shape != symbol_indices.shape:
        raise ValueError(
            "received_samples must match symbol_indices shape after THP equalization: "
            f"{received_samples.shape} != {symbol_indices.shape}."
        )
    if symbol_indices.shape[1] != len(diag_entries):
        raise ValueError(
            "symbol_indices do not match the THP diagonal width: "
            f"{symbol_indices.shape[1]} != {len(diag_entries)}."
        )

    symbols, bits = get_constellation(bits_per_symbol, labeling=labeling)
    replica_symbols, replica_bits = enumerate_thp_symbol_replicas(
        bits_per_symbol=bits_per_symbol,
        replica_radius=replica_radius,
        labeling=labeling,
    )
    bit_masks = {}
    for bit_idx in range(bits_per_symbol):
        bit_values = replica_bits[:, bit_idx]
        mask_one = bit_values == 1
        mask_zero = ~mask_one
        bit_masks[bit_idx] = (mask_zero, mask_one)

    transmitted_bits = bits[symbol_indices]
    sqrt_snr = float(np.sqrt(snr_per_stream))
    ideal_rate = float(symbol_indices.shape[1] * bits_per_symbol)
    loss_sum = 0.0
    total_samples = symbol_indices.shape[0]

    for stream_idx, gain in enumerate(diag_entries):
        scaled_codewords = sqrt_snr * gain * replica_symbols
        for sample_idx in range(total_samples):
            received = sqrt_snr * gain * received_samples[sample_idx, stream_idx]
            distances = np.abs(received - scaled_codewords) ** 2
            sample_bits = transmitted_bits[sample_idx, stream_idx]

            for bit_idx in range(bits_per_symbol):
                mask_zero, mask_one = bit_masks[bit_idx]
                llr = stable_logsumexp(-distances[mask_one]) - stable_logsumexp(-distances[mask_zero])
                bit_value = sample_bits[bit_idx]
                signed_llr = (2 * bit_value - 1) * llr
                loss_sum += stable_binary_logloss_from_signed_llr(signed_llr)

    return max(0.0, ideal_rate - loss_sum / max(total_samples, 1))


def estimate_bit_error_rate_thp_from_received(
    diagonal_channel: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    symbol_indices: np.ndarray,
    received_samples: np.ndarray,
    replica_radius: int = 1,
    labeling: str = "gray_standard",
) -> float:
    """Estimate BER for THP after modulo equalization using replica hard decisions."""

    diagonal_channel = np.asarray(diagonal_channel, dtype=complex)
    if diagonal_channel.ndim == 2:
        diag_entries = np.diag(diagonal_channel)
    elif diagonal_channel.ndim == 1:
        diag_entries = diagonal_channel
    else:
        raise ValueError("diagonal_channel must be a diagonal matrix or a 1-D diagonal vector.")

    symbol_indices = np.asarray(symbol_indices, dtype=int)
    received_samples = np.asarray(received_samples, dtype=complex)
    if symbol_indices.ndim != 2:
        raise ValueError("symbol_indices must have shape (num_samples, num_streams).")
    if received_samples.shape != symbol_indices.shape:
        raise ValueError(
            "received_samples must match symbol_indices shape after THP equalization: "
            f"{received_samples.shape} != {symbol_indices.shape}."
        )
    if symbol_indices.shape[1] != len(diag_entries):
        raise ValueError(
            "symbol_indices do not match the THP diagonal width: "
            f"{symbol_indices.shape[1]} != {len(diag_entries)}."
        )

    symbols, bits = get_constellation(bits_per_symbol, labeling=labeling)
    replica_symbols, replica_bits = enumerate_thp_symbol_replicas(
        bits_per_symbol=bits_per_symbol,
        replica_radius=replica_radius,
        labeling=labeling,
    )
    transmitted_bits = bits[symbol_indices]
    sqrt_snr = float(np.sqrt(snr_per_stream))
    total_samples = symbol_indices.shape[0]
    total_bits = int(total_samples * len(diag_entries) * bits_per_symbol)
    bit_errors = 0

    for stream_idx, gain in enumerate(diag_entries):
        scaled_codewords = sqrt_snr * gain * replica_symbols
        for sample_idx in range(total_samples):
            received = sqrt_snr * gain * received_samples[sample_idx, stream_idx]
            distances = np.abs(received - scaled_codewords) ** 2
            hard_index = int(np.argmin(distances))
            bit_errors += int(
                np.count_nonzero(replica_bits[hard_index] != transmitted_bits[sample_idx, stream_idx])
            )

    return float(bit_errors / max(total_bits, 1))


def estimate_symbol_mi_streamwise(
    upper_triangular_channel: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    num_samples: int = 256,
    seed: int | None = None,
) -> float:
    """?????symbol mi streamwise?"""
    diag_entries = np.diag(upper_triangular_channel)
    total_mi = 0.0
    for stream_idx, gain in enumerate(diag_entries):
        total_mi += estimate_scalar_symbol_mi(
            channel_gain=gain,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            num_samples=num_samples,
            seed=None if seed is None else seed + stream_idx,
        )
    return float(total_mi)


def estimate_bicm_gmi_auto(
    upper_triangular_channel: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    num_samples: int = 256,
    seed: int | None = None,
    max_joint_candidates: int = 20000,
) -> float:
    """?????bicm gmi auto?"""
    constellation_size = 2 ** bits_per_symbol
    num_streams = upper_triangular_channel.shape[1]
    total_candidates = constellation_size ** num_streams
    if total_candidates <= max_joint_candidates:
        return estimate_bicm_gmi(
            upper_triangular_channel=upper_triangular_channel,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            num_samples=num_samples,
            seed=seed,
        )
    return estimate_bicm_gmi_streamwise(
        upper_triangular_channel=upper_triangular_channel,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        num_samples=num_samples,
        seed=seed,
    )


def estimate_symbol_mi_auto(
    upper_triangular_channel: np.ndarray,
    snr_per_stream: float,
    bits_per_symbol: int,
    num_samples: int = 256,
    seed: int | None = None,
    max_joint_candidates: int = 20000,
) -> float:
    """?????symbol mi auto?"""
    constellation_size = 2 ** bits_per_symbol
    num_streams = upper_triangular_channel.shape[1]
    total_candidates = constellation_size ** num_streams
    if total_candidates <= max_joint_candidates:
        return estimate_symbol_mi(
            upper_triangular_channel=upper_triangular_channel,
            snr_per_stream=snr_per_stream,
            bits_per_symbol=bits_per_symbol,
            num_samples=num_samples,
            seed=seed,
        )
    return estimate_symbol_mi_streamwise(
        upper_triangular_channel=upper_triangular_channel,
        snr_per_stream=snr_per_stream,
        bits_per_symbol=bits_per_symbol,
        num_samples=num_samples,
        seed=seed,
    )
