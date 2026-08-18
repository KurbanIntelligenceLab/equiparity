# equiparity — When Parity Matters

Can SO(3)-equivariant models predict physically impossible material properties?

O(3)-equivariant models, whose features carry parity labels, produce symmetry-forced zeros
for the piezoelectric tensor of centrosymmetric crystals by construction. SO(3)-equivariant
models, which have no parity labels, fail: they predict nonzero tensors for cases that are
zero by Neumann's principle. This repository holds the study that quantifies when parity
matters — never for energies, marginally for dipoles, categorically for odd-parity tensors —
and turns it into a practitioner rule.

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Dependencies are declared in
`pyproject.toml` and pinned exactly in `uv.lock`; there is no `requirements.txt` and no
`pip install` path.

```bash
uv sync                 # base environment
uv sync --extra nequip  # + NequIP and Allegro
uv sync --extra mace    # + MACE (conflicts with nequip; install one at a time)
uv sync --extra data    # + pymatgen, mp-api, spglib for dataset preparation
uv run <command>        # run anything inside the locked environment
```

The `nequip` and `mace` extras cannot co-install, which is why CI runs them as a matrix.

## Verifying the claims

Two scripts check every reported quantity against the released records. Neither needs a GPU,
a trained model, or any data beyond this repository.

```bash
uv run python verification/verify_theory.py    # the theorems and corollaries, numerically
uv run python verification/verify_claims.py    # every reported quantity, against results/
```

`verify_claims.py` reports 184 passing checks and no failures against this tree. See
[`verification/README.md`](verification/README.md) for what each covers and how to reproduce
the underlying measurements.

## Using the parity toggle

The matched-pair builders are the reusable part. Each core is constructed twice from one
typed config, differing only in whether the features carry parity labels:

```python
from equiparity.domain.parity import ParityMode
from equiparity.models.nequip import NequIPConfig, build_nequip_matched

config = NequIPConfig(r_max=5.0, type_names=("H", "C", "N", "O"))
o3 = build_nequip_matched(config, ParityMode.O3)  # parity-typed irreps
so3 = build_nequip_matched(config, ParityMode.SO3)  # all-even relabelling
```

One config instance builds both arms, so the parity labelling is the only difference between
them. The SO(3) arm is not NequIP's `parity=False` preset, which keeps natural-parity irreps
and stays O(3)-equivariant; it relabels the edge spherical harmonics and hidden irreps as
all-even, removing parity as an e3nn selection rule. `build_allegro_matched` and
`build_mace_matched` take the same form.

The verification gate is a Go/No-Go: no configuration may train unless both arms of every
core in the profile pass it.

```bash
uv run --extra nequip pytest tests/verification -q
```

## Data

Every dataset is public. Manifests (`data/manifests/`, each carrying SHA-256 digests of the
processed archives) and split definitions (`data/splits/`) are versioned. The processed
Materials Project archives (`data/raw/mp/*.npz`, 6.7 MB across six files) are tracked: they
are the exact arrays the evaluation reads, they carry the `mp-*` identifiers of the
2,000-crystal centrosymmetric population in both coordinate variants, and no public endpoint
returns them. QM9 is not tracked — its source tarball is pinned by content hash in
`data/manifests/qm9.yaml` and `scripts/prepare_qm9.py` rebuilds the 133,885 `.xyz` files.

```bash
uv run --extra data python scripts/prepare_qm9.py
uv run --extra data python scripts/prepare_mp.py
```

`data/figure_series/` holds the machine-readable series behind the published figures: the
threshold sweep, the rutile distortion sweep, the Jacobian points, the raw-coordinate
threshold curves, and the epoch curves.

## Results

`results/` holds the frozen measurement records the study cites, at two levels of
granularity. Per-run summary records cover the matched-pair grid: test error with its seed
spread and paired test per core and target, false-flag fractions and violation medians per
arm and coordinate variant, per-seed values for the augmentation, loss-reweighting and
readout-pooling experiments, the threshold sweep, and the symmetry metadata of all 2,000
evaluation crystals.

For the measurements on released models the per-structure vectors are given in full:
`results/t1/` carries the predicted rank-3 tensor and its Frobenius norm for both dedicated
tensor predictors in each of four mask and coordinate conditions; `results/t2/` the violation
magnitude for three rotation-only potentials at random initialization; `results/t4/` the same
for every seed of two frozen-backbone heads. Each is a 2,000-element vector over the
evaluation population.

## Layout

```text
src/equiparity/     package code: domain, io, data, features, models, training,
                    evaluation, verification, workflows, reproducibility, cli
tests/              mirrors src/equiparity/, plus the parity verification gate
verification/       claim and theorem checkers that read only results/
configs/            experiment configs (YAML), including the 84-run grid
scripts/            orchestration and experiment drivers, no scientific logic
results/            frozen measurement records
data/               manifests, splits, processed archives, figure series
docs/results/       hand-written appendices with no generator
vendor/hotpp/       vendored HotPP (MIT, arXiv:2402.15286), no PyPI release
outputs/            per-run results and provenance manifests (not committed)
```

Appendices produced by a script are not committed; they regenerate:

```bash
uv run python scripts/analyze_results.py                        # a1-a4
uv run --extra data python scripts/analyze_ood_symmetry.py      # a5
```

`vendor/hotpp/` is load-bearing: `scripts/f4_noneN3_control.py` imports it, and the claim
that it carries no e3nn dependency is checkable against that tree.

## License

MIT — see [LICENSE](LICENSE).

## Known limitations

Exact numerical reproducibility is not guaranteed across torch, CUDA or cuDNN releases, or
across devices. Two GPU classes were used and are not interchangeable, so per-core
wall-clock is not a like-for-like architecture comparison. The CliffordSTF core is present
but was withdrawn from the study for numerical conditioning.
