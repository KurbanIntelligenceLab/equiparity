# Verification

Two checkers, neither of which needs a GPU, a trained model, or any data beyond this repository.

```bash
uv sync --extra verify
uv run python verification/verify_theory.py    # the theorems and corollaries, numerically
uv run python verification/verify_claims.py    # every reported quantity, against results/
```

`verify_theory.py` builds explicit representation matrices and checks the parity-gap statements by
construction: that an all-even relabelling admits maps a parity-typed model forbids, that the
forbidden subspace is exactly the odd-parity one, and that the corollary identities hold
symbolically. It reports no failures.

`verify_claims.py` re-derives every quantity the study reports from the records in `results/` and
asserts it against the published value: the accuracy table, the false-flag fractions and their
threshold sweep, the parameter-capacity ratios, the augmentation and loss-reweighting arms, the
readout-pooling comparison, the named-material predictions, and the symmetry decomposition of the
evaluation population. It reports `PASSED 184  FAILED 0`.

## The parity gate

A Go/No-Go, not a diagnostic: no configuration may train unless both arms of every core in the
profile pass it.

```bash
uv run --extra nequip pytest tests/verification -q
```

## Reproducing the underlying measurements

[`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) maps every reported quantity to the script that
produced it and the configs it used.

The measurements on released models need no training, but they do need those models. Each runs
under its own isolated interpreter rather than `uv run`, because the released code pins dependency
sets that conflict with each other and with this project's profiles; one checkpoint is behind an
access gate. [`../docs/EXTERNAL_MODELS.md`](../docs/EXTERNAL_MODELS.md) covers both.

Their per-structure outputs are released here in full — `results/tensor_predictors/`,
`results/random_init/` and `results/frozen_backbone/`, each a 2,000-element vector over the
evaluation population — so any reported fraction can be recomputed without obtaining a single
external model.
