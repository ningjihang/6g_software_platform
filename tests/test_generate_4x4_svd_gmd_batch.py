from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_4x4_svd_gmd_batch.py"
ROOT_DIR = str(MODULE_PATH.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
MODULE_SPEC = importlib.util.spec_from_file_location("generate_4x4_svd_gmd_batch", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
batch = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(batch)


def test_real_matrix_generator_returns_readable_full_rank_matrix() -> None:
    """?? test real matrix generator returns readable full rank matrix ???"""
    rng = np.random.default_rng(7)
    matrix = batch.generate_matrix(rng=rng, real_only=True)

    expected_values = np.array([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0], dtype=float)
    singular_values = np.linalg.svd(matrix, compute_uv=False)

    assert np.isrealobj(matrix)
    assert np.isin(matrix, expected_values).all()
    assert singular_values[-1] > 0.25


def test_real_sample_report_marks_svd_and_gmd_as_pass() -> None:
    """?? test real sample report marks svd and gmd as pass ???"""
    matrix = np.array(
        [
            [2.0, -1.5, 0.5, 1.0],
            [-0.5, 1.5, 2.0, -1.0],
            [1.0, 0.5, -1.5, 2.0],
            [1.5, -0.5, 1.0, -2.0],
        ],
        dtype=float,
    )

    sample_result, summary_row, report = batch.build_sample_result(
        sample_index=0,
        matrix=matrix,
        decimals=4,
    )

    svd_diag = np.asarray(sample_result["svd"]["diag_abs_r"], dtype=float)
    singular_values = np.asarray(sample_result["svd"]["singular_values"], dtype=float)
    gmd_diag = np.asarray(sample_result["gmd"]["diag_abs_r"], dtype=float)
    gmd_target = float(sample_result["gmd"]["target_diagonal"])

    np.testing.assert_allclose(
        sample_result["matrix"]["imag"],
        np.zeros((4, 4)),
        atol=batch.DEFAULT_TOLERANCE,
    )
    np.testing.assert_allclose(svd_diag, singular_values, atol=batch.DEFAULT_TOLERANCE)
    np.testing.assert_allclose(
        gmd_diag,
        np.full(4, gmd_target, dtype=float),
        atol=batch.DEFAULT_TOLERANCE,
    )
    assert sample_result["svd"]["pass"] is True
    assert sample_result["gmd"]["pass"] is True
    assert summary_row["svd_pass"] == 1.0
    assert summary_row["gmd_pass"] == 1.0
    assert sample_result["verdict"] == "PASS"
    assert "Sample 0: PASS" in report
    assert "Verdict: SVD PASS, GMD PASS" in report
