# Classical Workflow

This directory now keeps only the current hybrid structured-chain workflow.

## Main entry points

- `compare_hybrid_svd_gmd_ucd.py`
  Main hybrid comparison entry point for `svd`, `gmd`, and `ucd`.
  The active path uses the unified design-time transmit/receive chain logic.

## Core shared modules

- `multiuser_simulation_environment.py`
  Per-user-RF environment and evaluation logic for THP/UCD-style receiver
  processing.
- `analog_precoder.py`
  Analog precoder design from per-user channel SVD phases.
- `digital_precoder.py`
  Structured digital precoder construction, including UCD.
- `thp_precoding.py`
  THP / modulo helpers reused by the matched UCD receive path.
- `bicm_metrics.py`
  BICM-GMI and BER estimators used by the current receiver chain.
- `sic_sample_average.py`
  Shared Monte Carlo batches for fair comparison.
- `channel_model.py`
  Channel sampling wrapper.
- `channel_estimation.py`
  Imperfect-CSI helper using an additive complex Gaussian error model.
- `channel_estimation_mmse.py`
  Pilot-aided MMSE channel estimation with a full covariance estimated from
  channel samples.

## Modes inside `compare_hybrid_svd_gmd_ucd.py`

- `perfect`
  Perfect transmitter CSI.
- `gaussian`
  Additive complex Gaussian CSI error model.
- `mmse_fullcov`
  Pilot-aided MMSE with a full covariance estimated from channel samples.

## Outputs

- `results/`
  Saved hybrid `svd/gmd/ucd` CSV and figure outputs.
