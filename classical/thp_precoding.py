from __future__ import annotations

import numpy as np


def qam_axis_order_and_spacing(modem_bits: int) -> tuple[int, float]:
    """Return the PAM-axis order and normalized spacing for square QAM."""

    spacing_map = {
        2: 2.0 / np.sqrt(2.0),
        4: 2.0 / np.sqrt(10.0),
        6: 2.0 / np.sqrt(42.0),
        8: 2.0 / np.sqrt(170.0),
    }
    if modem_bits not in spacing_map:
        raise ValueError(
            f"Unsupported modem_bits={modem_bits}, expected one of {sorted(spacing_map)}."
        )
    m_axis = 2 ** (modem_bits // 2)
    return m_axis, float(spacing_map[modem_bits])


def centered_modulo(x: np.ndarray, period: float) -> np.ndarray:
    """Map a real-valued array into [-period/2, period/2)."""

    x = np.asarray(x, dtype=float)
    return x - np.floor((x + period / 2.0) / period) * period


def centered_modulo_complex(x: np.ndarray, period: float) -> np.ndarray:
    """Apply the centered modulo independently on the real and imaginary parts."""

    x = np.asarray(x, dtype=np.complex128)
    return centered_modulo(np.real(x), period) + 1j * centered_modulo(np.imag(x), period)


def compute_effective_channel(
    w: np.ndarray,
    h: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute B = W^H H P and its row-normalized version."""

    b = np.asarray(w, dtype=np.complex128).conj().T @ np.asarray(h, dtype=np.complex128) @ np.asarray(
        p,
        dtype=np.complex128,
    )
    diagonal = np.diag(b)
    if np.any(np.isclose(diagonal, 0.0)):
        raise ValueError("Effective channel has zero diagonal entries.")
    b1 = b / diagonal[:, None]
    return b, b1


def rq_decompose(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Economy-size RQ decomposition using NumPy QR."""

    h = np.asarray(h, dtype=np.complex128)
    if h.ndim != 2:
        raise ValueError("h must be a 2-D matrix.")
    num_rows, num_cols = h.shape
    if num_rows > num_cols:
        raise ValueError(
            "rq_decompose expects Nr <= Nt for THP precoding, "
            f"got h.shape={h.shape}."
        )

    flipped = np.flipud(h).T
    q_factor, r_factor = np.linalg.qr(flipped)
    r = np.flipud(r_factor.T)
    r = np.fliplr(r)
    q = np.flipud(q_factor.T)
    return r, q


def build_rq_thp_factors(
    h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the RQ-based THP factors for a full-row-rank channel."""

    h = np.asarray(h, dtype=np.complex128)
    num_rows, _ = h.shape
    r_factor, q_factor = rq_decompose(h)
    p = q_factor.conj().T
    w = np.eye(num_rows, dtype=np.complex128)
    b, b1 = compute_effective_channel(w, h, p)
    return w, p, b, b1


def thp_precoder_from_b1(
    cod_data: np.ndarray,
    b1: np.ndarray,
    modem_bits: int,
) -> np.ndarray:
    """Run the recursive THP modulo precoder on a normalized upper channel."""

    cod_data = np.asarray(cod_data, dtype=np.complex128)
    b1 = np.asarray(b1, dtype=np.complex128)
    if cod_data.ndim != 2:
        raise ValueError("cod_data must have shape [num_streams, num_symbols].")
    if b1.ndim != 2 or b1.shape[0] != b1.shape[1]:
        raise ValueError("b1 must be a square matrix.")
    if b1.shape[0] != cod_data.shape[0]:
        raise ValueError(
            "Stream count mismatch between b1 and cod_data: "
            f"{b1.shape} vs {cod_data.shape}."
        )

    m_axis, spacing = qam_axis_order_and_spacing(modem_bits)
    period = m_axis * spacing
    dp_cod_data = cod_data.copy()

    num_streams = dp_cod_data.shape[0]
    for stream_idx in range(num_streams - 1, -1, -1):
        if stream_idx + 1 < num_streams:
            interference = b1[stream_idx, stream_idx + 1 :] @ dp_cod_data[stream_idx + 1 :, :]
            pre_modulo = dp_cod_data[stream_idx, :] - interference
        else:
            pre_modulo = dp_cod_data[stream_idx, :]
        dp_cod_data[stream_idx, :] = centered_modulo_complex(pre_modulo, period)

    return dp_cod_data


def apply_transmit_precoder(p: np.ndarray, dp_cod_data: np.ndarray) -> np.ndarray:
    """Map layer-domain THP symbols to the transmit domain."""

    return np.asarray(p, dtype=np.complex128) @ np.asarray(dp_cod_data, dtype=np.complex128)


def thp_transmit_from_upper(
    cod_data: np.ndarray,
    upper_b: np.ndarray,
    modem_bits: int,
    p: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run THP when the effective upper-triangular channel is already known."""

    upper_b = np.asarray(upper_b, dtype=np.complex128)
    diagonal = np.diag(upper_b)
    if np.any(np.isclose(diagonal, 0.0)):
        raise ValueError("upper_b has zero diagonal entries.")
    b1 = upper_b / diagonal[:, None]
    dp_cod_data = thp_precoder_from_b1(cod_data=cod_data, b1=b1, modem_bits=modem_bits)
    x_tx = dp_cod_data if p is None else apply_transmit_precoder(p, dp_cod_data)
    return x_tx, dp_cod_data, b1


def thp_precoder(
    cod_data: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    p: np.ndarray,
    modem_bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end THP recursion from explicit W/H/P factors."""

    cod_data = np.asarray(cod_data, dtype=np.complex128)
    if cod_data.ndim != 2:
        raise ValueError("cod_data must have shape [num_streams, num_symbols].")

    b, b1 = compute_effective_channel(w=w, h=h, p=p)
    if b1.shape[0] != cod_data.shape[0]:
        raise ValueError(
            "Effective channel dimension does not match cod_data: "
            f"b1={b1.shape}, cod_data={cod_data.shape}."
        )

    dp_cod_data = thp_precoder_from_b1(cod_data=cod_data, b1=b1, modem_bits=modem_bits)
    return dp_cod_data, b, b1


def thp_transmit_from_channel(
    cod_data: np.ndarray,
    h: np.ndarray,
    modem_bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end THP transmit path using an RQ triangularization."""

    _, p, b, b1 = build_rq_thp_factors(h)
    dp_cod_data = thp_precoder_from_b1(cod_data=cod_data, b1=b1, modem_bits=modem_bits)
    x_tx = apply_transmit_precoder(p, dp_cod_data)
    return x_tx, dp_cod_data, p, b, b1


def thp_receive_equalized(
    y_rx: np.ndarray,
    modem_bits: int,
    w: np.ndarray | None = None,
    diag_b: np.ndarray | None = None,
    apply_modulo: bool = True,
) -> np.ndarray:
    """Equalize a THP receive signal and optionally apply the receive-side modulo."""

    z = np.asarray(y_rx, dtype=np.complex128)
    if w is not None:
        z = np.asarray(w, dtype=np.complex128).conj().T @ z
    if diag_b is not None:
        diag_b = np.asarray(diag_b, dtype=np.complex128)
        z = z / diag_b[:, None]
    if apply_modulo:
        m_axis, spacing = qam_axis_order_and_spacing(modem_bits)
        z = centered_modulo_complex(z, m_axis * spacing)
    return z
