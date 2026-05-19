# Fully-Digital MU MIMO-BICM

This folder now keeps only the full-digital structured-chain workflow.

Scope:

- fully-digital multi-user precoding with `F_RF = I`
- shared Monte Carlo sample batches for fair comparison
- `svd`, `gmd`, and `ucd` comparison on the unified structured-chain path

Files:

- `fd_mu_environment.py`: fully-digital environment wrapper with `F_RF = I`.
- `compare_full_digital_svd_gmd_ucd_fair.py`: runs the full-digital
  `svd/gmd/ucd` comparison on shared fixed Monte Carlo batches.

Notes:

- Old UCD-only and multi-baseline comparison code paths are no longer part of
  the active workflow.
- The implementation reuses the existing channel model, BD helpers, and
  receiver-side Monte Carlo GMI evaluation from `classical/`.
