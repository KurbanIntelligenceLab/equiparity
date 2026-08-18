# Verification

Two scripts check the study's claims against the released records. Neither needs a GPU,
a trained model, or any data beyond what is in this repository.

```bash
uv run python verification/verify_theory.py    # the theorems and corollaries, numerically
uv run python verification/verify_claims.py    # every reported quantity, against results/
```

`verify_theory.py` builds explicit representation matrices and checks the parity-gap
statements by construction: that an all-even relabelling admits maps a parity-typed model
forbids, that the forbidden subspace is exactly the odd-parity one, and that the displayed
corollary identities hold symbolically.

`verify_claims.py` re-derives every quantity the study reports from the JSON records in
`results/` and asserts it against the value published for it. It covers the accuracy table,
the false-flag fractions and their threshold sweep, the parameter-capacity ratios, the
augmentation and loss-reweighting arms, the readout-pooling comparison, the named-material
predictions, the symmetry decomposition of the evaluation population, and the arithmetic
that connects them. It reports 184 passing checks and no failures against this tree.

Both are pure consistency checks on the released records. Reproducing the records themselves
from model outputs requires training, which is what `scripts/` and `configs/` are for.

## Reproducing the measurements

The parity verification gate is a Go/No-Go: no configuration may train unless both arms of
every core in the profile pass it.

```bash
uv run --extra nequip pytest tests/verification -q
```

The measurements on released models need no training of ours and read only public
checkpoints:

```bash
uv run python scripts/t1_sota_eval.py         # dedicated crystal-tensor predictors
uv run python scripts/t2_backbone_probe.py    # rotation-only potentials at random init
uv run python scripts/t4_frozen_backbone.py   # frozen backbones with a fitted head
```

Their per-structure outputs are in `results/t1/`, `results/t2/` and `results/t4/` as
2,000-element vectors over the evaluation population, so a reviewer can check the reported
fractions without rerunning anything.
