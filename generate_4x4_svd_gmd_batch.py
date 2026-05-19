from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import svd

try:
    from full_digital_mu._bootstrap import ensure_classical_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_classical_on_path

ensure_classical_on_path()

from digital_precoder import DigitalStructuredPrecoder

DEFAULT_TOLERANCE = 1e-8


def parse_args() -> argparse.Namespace:
    """?????args?"""
    parser = argparse.ArgumentParser(
        description="Generate a batch of 4x4 SVD/GMD decompositions using the repo logic.",
    )
    parser.add_argument("--count", type=int, default=10, help="Number of 4x4 matrices to generate.")
    parser.add_argument("--seed", type=int, default=20260407, help="Random seed.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("classical") / "results"),
        help="Directory used to store the generated JSON/CSV outputs.",
    )
    value_group = parser.add_mutually_exclusive_group()
    value_group.add_argument(
        "--real-only",
        dest="real_only",
        action="store_true",
        help="Generate real-valued 4x4 matrices. This is the default mode.",
    )
    value_group.add_argument(
        "--complex",
        dest="real_only",
        action="store_false",
        help="Generate complex-valued 4x4 matrices.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Number of decimals used in the human-readable report.",
    )
    parser.set_defaults(real_only=True)
    return parser.parse_args()


def make_output_prefix(count: int, seed: int, real_only: bool) -> str:
    """?????output prefix?"""
    value_type = "real" if real_only else "complex"
    return f"batch_4x4_svd_gmd_{value_type}_n{count}_seed{seed}"


def generate_matrix(rng: np.random.Generator, real_only: bool) -> np.ndarray:
    """?????matrix?"""
    if real_only:
        candidate_values = np.array([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0], dtype=float)
        while True:
            matrix = rng.choice(candidate_values, size=(4, 4), replace=True)
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            if singular_values[-1] > 0.25:
                return matrix
    return (
        rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    ) / np.sqrt(2.0)


def qr_factors_with_positive_diagonal(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """?? qr factors with positive diagonal ???"""
    q_factor, r_factor = np.linalg.qr(matrix, mode="reduced")
    diagonal = np.diag(r_factor)
    phase = np.exp(-1j * np.angle(diagonal))
    phase_matrix = np.diag(phase)
    q_aligned = q_factor @ phase_matrix.conj().T
    r_aligned = phase_matrix @ r_factor
    return q_aligned, r_aligned


def serialize_matrix(matrix: np.ndarray) -> dict[str, list[list[float]]]:
    """??????matrix?"""
    return {
        "real": np.real(matrix).tolist(),
        "imag": np.imag(matrix).tolist(),
    }


def format_scalar(value: complex, decimals: int) -> str:
    """??????scalar?"""
    real_part = float(np.real(value))
    imag_part = float(np.imag(value))
    if abs(real_part) < 0.5 * 10 ** (-decimals):
        real_part = 0.0
    if abs(imag_part) < 0.5 * 10 ** (-decimals):
        imag_part = 0.0
    if imag_part == 0.0:
        return f"{real_part: .{decimals}f}"
    sign = "+" if imag_part >= 0 else "-"
    return f"{real_part: .{decimals}f}{sign}{abs(imag_part):.{decimals}f}j"


def format_matrix(matrix: np.ndarray, decimals: int) -> str:
    """??????matrix?"""
    lines: list[str] = []
    for row in np.asarray(matrix):
        formatted_row = ", ".join(format_scalar(entry, decimals) for entry in row)
        lines.append(f"[{formatted_row}]")
    return "\n".join(lines)


def format_vector(vector: np.ndarray, decimals: int) -> str:
    """??????vector?"""
    entries = ", ".join(format_scalar(entry, decimals) for entry in np.asarray(vector))
    return f"[{entries}]"


def build_sample_report(
    sample_index: int,
    matrix: np.ndarray,
    singular_values: np.ndarray,
    r_svd: np.ndarray,
    svd_reconstruction_error: float,
    target_diagonal: float,
    r_gmd: np.ndarray,
    gmd_reconstruction_error: float,
    decimals: int,
    tolerance: float,
) -> str:
    """?????sample report?"""
    svd_diag_abs = np.abs(np.diag(r_svd))
    gmd_diag_abs = np.abs(np.diag(r_gmd))
    gmd_target_diag = np.full_like(gmd_diag_abs, target_diagonal, dtype=float)
    svd_diag_max_deviation = float(np.max(np.abs(svd_diag_abs - singular_values)))
    gmd_diag_max_deviation = float(np.max(np.abs(gmd_diag_abs - gmd_target_diag)))
    svd_pass = svd_diag_max_deviation <= tolerance and svd_reconstruction_error <= tolerance
    gmd_pass = gmd_diag_max_deviation <= tolerance and gmd_reconstruction_error <= tolerance

    return "\n".join(
        [
            "=" * 88,
            f"Sample {sample_index}: {'PASS' if svd_pass and gmd_pass else 'FAIL'}",
            "H =",
            format_matrix(matrix, decimals),
            "",
            f"SVD singular values       = {format_vector(singular_values, decimals)}",
            f"SVD diag(|R|)            = {format_vector(svd_diag_abs, decimals)}",
            f"SVD diag max deviation   = {svd_diag_max_deviation:.6e}",
            f"SVD reconstruction error = {svd_reconstruction_error:.6e}",
            "R_svd =",
            format_matrix(r_svd, decimals),
            "",
            f"GMD target diag          = {format_vector(gmd_target_diag, decimals)}",
            f"GMD diag(|R|)            = {format_vector(gmd_diag_abs, decimals)}",
            f"GMD diag max deviation   = {gmd_diag_max_deviation:.6e}",
            f"GMD reconstruction error = {gmd_reconstruction_error:.6e}",
            "R_gmd =",
            format_matrix(r_gmd, decimals),
            "",
            f"Verdict: SVD {'PASS' if svd_pass else 'FAIL'}, GMD {'PASS' if gmd_pass else 'FAIL'}",
        ]
    )


def build_text_report(
    meta: dict[str, object],
    sample_reports: list[str],
    summary_rows: list[dict[str, float]],
) -> str:
    """?????text report?"""
    all_pass = all(bool(row["svd_pass"]) and bool(row["gmd_pass"]) for row in summary_rows)
    header_lines = [
        "4x4 SVD/GMD decomposition report",
        f"value_type: {'real' if bool(meta['real_only']) else 'complex'}",
        f"count: {meta['count']}",
        f"seed: {meta['seed']}",
        f"matrix_shape: {meta['matrix_shape']}",
        f"logic_source: {meta['logic_source']}",
        f"mode: {meta['mode']}",
        f"overall_verdict: {'PASS' if all_pass else 'FAIL'}",
    ]
    return "\n".join(header_lines + [""] + sample_reports + ["", "=" * 88])


def build_sample_result(
    sample_index: int,
    matrix: np.ndarray,
    decimals: int,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[dict[str, object], dict[str, float], str]:
    """?????sample result?"""
    num_streams = 4
    snr_per_stream = 1.0
    precoder = DigitalStructuredPrecoder(num_rf_chains=num_streams, num_streams=num_streams)

    u_factor, singular_values, vh_factor = svd(matrix, full_matrices=False)
    v_eff = vh_factor.conj().T[:, :num_streams]
    mu = singular_values[:num_streams] ** 2

    v_svd = v_eff.copy()
    q_svd, r_svd = qr_factors_with_positive_diagonal(matrix @ v_svd)
    svd_reconstruction_error = float(
        np.linalg.norm(matrix - q_svd @ r_svd @ v_svd.conj().T, ord="fro")
    )
    rho_svd = precoder.compute_stream_snr(matrix, v_svd, snr_per_stream)

    rho_target_gmd = precoder.build_target_rho(mu, "gmd")
    v_gmd = precoder.apply_gtd(
        v_eff=v_eff,
        singular_values=singular_values[:num_streams],
        rho_target=rho_target_gmd,
        snr_per_stream=snr_per_stream,
    )
    q_gmd, r_gmd = qr_factors_with_positive_diagonal(matrix @ v_gmd)
    gmd_reconstruction_error = float(
        np.linalg.norm(matrix - q_gmd @ r_gmd @ v_gmd.conj().T, ord="fro")
    )
    rho_gmd = precoder.compute_stream_snr(matrix, v_gmd, snr_per_stream)
    omega_gmd = v_eff.conj().T @ v_gmd

    svd_diag_abs = np.abs(np.diag(r_svd))
    gmd_diag_abs = np.abs(np.diag(r_gmd))
    target_diagonal = np.sqrt(rho_target_gmd[0])
    gmd_target_diag = np.full_like(gmd_diag_abs, target_diagonal, dtype=float)
    svd_diag_max_deviation = float(np.max(np.abs(svd_diag_abs - singular_values)))
    gmd_diag_max_deviation = float(np.max(np.abs(gmd_diag_abs - gmd_target_diag)))
    svd_pass = bool(
        svd_diag_max_deviation <= tolerance and svd_reconstruction_error <= tolerance
    )
    gmd_pass = bool(
        gmd_diag_max_deviation <= tolerance and gmd_reconstruction_error <= tolerance
    )

    sample_result: dict[str, object] = {
        "matrix": serialize_matrix(matrix),
        "svd": {
            "u": serialize_matrix(u_factor),
            "singular_values": singular_values.tolist(),
            "vh": serialize_matrix(vh_factor),
            "v_precoder": serialize_matrix(v_svd),
            "q": serialize_matrix(q_svd),
            "r": serialize_matrix(r_svd),
            "diag_abs_r": svd_diag_abs.tolist(),
            "diag_max_deviation": svd_diag_max_deviation,
            "target_rho": mu.tolist(),
            "realized_rho": rho_svd.tolist(),
            "reconstruction_error_fro": svd_reconstruction_error,
            "pass": svd_pass,
        },
        "gmd": {
            "omega": serialize_matrix(omega_gmd),
            "v_precoder": serialize_matrix(v_gmd),
            "q": serialize_matrix(q_gmd),
            "r": serialize_matrix(r_gmd),
            "diag_abs_r": gmd_diag_abs.tolist(),
            "diag_max_deviation": gmd_diag_max_deviation,
            "target_rho": rho_target_gmd.tolist(),
            "realized_rho": rho_gmd.tolist(),
            "target_diagonal": float(target_diagonal),
            "reconstruction_error_fro": gmd_reconstruction_error,
            "pass": gmd_pass,
        },
        "verdict": "PASS" if svd_pass and gmd_pass else "FAIL",
    }

    summary_row = {
        "sample_index": float(sample_index),
        "svd_sigma_1": float(singular_values[0]),
        "svd_sigma_2": float(singular_values[1]),
        "svd_sigma_3": float(singular_values[2]),
        "svd_sigma_4": float(singular_values[3]),
        "svd_diag_1": float(svd_diag_abs[0]),
        "svd_diag_2": float(svd_diag_abs[1]),
        "svd_diag_3": float(svd_diag_abs[2]),
        "svd_diag_4": float(svd_diag_abs[3]),
        "svd_diag_max_deviation": svd_diag_max_deviation,
        "gmd_target_diag": float(target_diagonal),
        "gmd_diag_1": float(gmd_diag_abs[0]),
        "gmd_diag_2": float(gmd_diag_abs[1]),
        "gmd_diag_3": float(gmd_diag_abs[2]),
        "gmd_diag_4": float(gmd_diag_abs[3]),
        "gmd_diag_max_deviation": gmd_diag_max_deviation,
        "svd_reconstruction_error_fro": svd_reconstruction_error,
        "gmd_reconstruction_error_fro": gmd_reconstruction_error,
        "svd_pass": float(svd_pass),
        "gmd_pass": float(gmd_pass),
    }
    sample_report = build_sample_report(
        sample_index=sample_index,
        matrix=matrix,
        singular_values=singular_values,
        r_svd=r_svd,
        svd_reconstruction_error=svd_reconstruction_error,
        target_diagonal=float(target_diagonal),
        r_gmd=r_gmd,
        gmd_reconstruction_error=gmd_reconstruction_error,
        decimals=decimals,
        tolerance=tolerance,
    )
    return sample_result, summary_row, sample_report


def save_json(path: Path, payload: dict[str, object]) -> None:
    """?json????????"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, float]]) -> None:
    """?csv????????"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_text(path: Path, content: str) -> None:
    """?text????????"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    """????????"""
    args = parse_args()
    if args.count <= 0:
        raise ValueError(f"--count must be positive, got {args.count}.")
    if args.decimals < 0:
        raise ValueError(f"--decimals must be non-negative, got {args.decimals}.")

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    prefix = make_output_prefix(count=args.count, seed=args.seed, real_only=args.real_only)
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"
    txt_path = out_dir / f"{prefix}.txt"

    samples: list[dict[str, object]] = []
    summary_rows: list[dict[str, float]] = []
    sample_reports: list[str] = []

    for sample_index in range(args.count):
        matrix = generate_matrix(rng=rng, real_only=args.real_only)
        sample_result, summary_row, sample_report = build_sample_result(
            sample_index=sample_index,
            matrix=matrix,
            decimals=args.decimals,
        )
        sample_result["sample_index"] = sample_index
        samples.append(sample_result)
        summary_rows.append(summary_row)
        sample_reports.append(sample_report)

    payload = {
        "meta": {
            "count": args.count,
            "seed": args.seed,
            "matrix_shape": [4, 4],
            "real_only": bool(args.real_only),
            "decimals": args.decimals,
            "tolerance": DEFAULT_TOLERANCE,
            "logic_source": "classical/digital_precoder.py",
            "mode": "single-user fully-digital with F_RF = I_4 and snr_per_stream = 1.0",
        },
        "samples": samples,
    }
    save_json(json_path, payload)
    save_csv(csv_path, summary_rows)
    report_text = build_text_report(meta=payload["meta"], sample_reports=sample_reports, summary_rows=summary_rows)
    save_text(txt_path, report_text)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved TXT : {txt_path}")
    print("")
    print(" sample | svd diag dev | gmd diag dev | svd err | gmd err | verdict")
    print("--------------------------------------------------------------------")
    for row in summary_rows:
        verdict = "PASS" if row["svd_pass"] and row["gmd_pass"] else "FAIL"
        print(
            f"{int(row['sample_index']):7d} | "
            f"{row['svd_diag_max_deviation']:12.4e} | "
            f"{row['gmd_diag_max_deviation']:12.4e} | "
            f"{row['svd_reconstruction_error_fro']:7.2e} | "
            f"{row['gmd_reconstruction_error_fro']:7.2e} | "
            f"{verdict}"
        )
    print("")
    print(report_text)


if __name__ == "__main__":
    main()
