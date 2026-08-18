# Verification

```bash
uv run python verification/verify_theory.py    # the theorems and corollaries, numerically
uv run python verification/verify_claims.py    # every reported quantity, against results/
```

Neither needs a GPU, a trained model, or any data beyond this repository.

`verify_theory.py` builds explicit representation matrices and checks the parity-gap statements
by construction: that an all-even relabelling admits maps a parity-typed model forbids, that the
forbidden subspace is exactly the odd-parity one, and that the corollary identities hold
symbolically.

`verify_claims.py` re-derives every quantity the study reports from the records in `results/` and
asserts it against the published value: the accuracy table, the false-flag fractions and their
threshold sweep, the parameter-capacity ratios, the augmentation and loss-reweighting arms, the
readout-pooling comparison, the named-material predictions, and the symmetry decomposition of the
evaluation population.

## Reproducing the measurements

The parity verification gate is a Go/No-Go: no configuration may train unless both arms of every
core in the profile pass it.

```bash
uv run --extra nequip pytest tests/verification -q
```

The measurements on released models need no training and read only public checkpoints:

```bash
uv run python scripts/t1_sota_eval.py         # dedicated crystal-tensor predictors
uv run python scripts/t2_backbone_probe.py    # rotation-only potentials at random init
uv run python scripts/t4_frozen_backbone.py   # frozen backbones with a fitted head
```

Their per-structure outputs are in `results/t1/`, `results/t2/` and `results/t4/` as
2,000-element vectors over the evaluation population.
