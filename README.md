# equiparity

Code, configurations and measurement records for:

**A single design choice determines whether machine learning models of materials make physically
impossible predictions**

Can Polat<sup>1</sup> ([0000-0002-1458-302X](https://orcid.org/0000-0002-1458-302X)),
Mustafa Kurban<sup>2,3</sup> ([0000-0002-7263-0234](https://orcid.org/0000-0002-7263-0234)),
Erchin Serpedin<sup>1</sup> ([0000-0001-9069-770X](https://orcid.org/0000-0001-9069-770X)),
Hasan Kurban<sup>4</sup> ([0000-0003-3142-2866](https://orcid.org/0000-0003-3142-2866))

1. Department of Electrical and Computer Engineering, Texas A&M University, College Station, Texas, USA
2. Department of Electrical and Computer Engineering, Texas A&M University at Qatar, Doha, Qatar
3. Department of Prosthetics and Orthotics, Ankara University, Ankara, Turkey
4. College of Science and Engineering, Hamad Bin Khalifa University, Doha, Qatar

Corresponding authors: Mustafa Kurban ([kurbanm@ankara.edu.tr](mailto:kurbanm@ankara.edu.tr)),
Hasan Kurban ([hkurban@hbku.edu.qa](mailto:hkurban@hbku.edu.qa))

## Abstract

Machine-learned models are replacing first-principles calculations across materials discovery, and
physical symmetry is the central guarantee built into them. The debate over how much symmetry to
hard-wire rather than learn has run on rotations, where a symmetry error is an approximation
error. Some constraints are exact: symmetry forces certain property tensors to exactly zero, so a
nonzero prediction is physically impossible rather than inaccurate. Here we show that whether a
model can make such predictions is decided before training by one rarely reported design bit,
whether its features carry parity labels, and derive a criterion, the parity gap, that computes
from group theory alone which properties and crystals are exposed. Across matched architecture
pairs differing only in that bit, evaluated on two thousand centrosymmetric crystals whose
piezoelectric tensor must vanish, parity-labelled arms sit at the floating-point floor while
rotation-only arms predict forbidden responses on 90–96% of crystals, six orders of magnitude
apart, at no accuracy cost. Training on explicit zeros does not recover exactness, and a head on a
frozen universal potential inherits its backbone's symmetry group. One reflection at random
initialization verifies the label in seconds.
## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Dependencies are declared in
`pyproject.toml` and pinned exactly in `uv.lock`; there is no `requirements.txt` and no
`pip install` path.

Checking the claims needs no GPU and no deep-learning stack:

```bash
uv sync --extra verify
```

Retraining or preparing data needs one of the model profiles. MACE pins `e3nn==0.4.4` while
NequIP and Allegro require `e3nn>=0.6`, so the two cannot co-install; CI runs them as a matrix.

```bash
uv sync --extra nequip  # + NequIP and Allegro
uv sync --extra mace    # + MACE (conflicts with nequip; install one at a time)
uv sync --extra data    # + pymatgen, mp-api, spglib for dataset preparation
uv run <command>        # run anything inside the locked environment
```

## Verifying the claims

Two scripts check every reported quantity against the released records. Neither needs a GPU, a
trained model, or any data beyond this repository.

```bash
uv run python verification/verify_theory.py    # the theorems and corollaries, numerically
uv run python verification/verify_claims.py    # every reported quantity, against results/
```

`verify_claims.py` reports `PASSED 184  FAILED 0` against this tree, and `verify_theory.py`
reports no failures. See [`verification/README.md`](verification/README.md) for what each covers
and how to reproduce the underlying measurements.

## Using the parity toggle

The matched-pair builders are the reusable part. Each core is constructed twice from one typed
config, differing only in whether the features carry parity labels:

```python
from equiparity.domain.parity import ParityMode
from equiparity.models.nequip import NequIPConfig, build_nequip_matched

config = NequIPConfig(r_max=5.0, type_names=("H", "C", "N", "O"))
o3 = build_nequip_matched(config, ParityMode.O3)  # parity-typed irreps
so3 = build_nequip_matched(config, ParityMode.SO3)  # all-even relabelling
```

One config instance builds both arms, so the parity labelling is the only difference between them.
The SO(3) arm is not NequIP's `parity=False` preset, which keeps natural-parity irreps and stays
O(3)-equivariant; it relabels the edge spherical harmonics and hidden irreps as all-even, removing
parity as an e3nn selection rule. `build_allegro_matched` and `build_mace_matched` take the same
form.

The verification gate is a Go/No-Go: no configuration may train unless both arms of every core in
the profile pass it.

```bash
uv run --extra nequip pytest tests/verification -q
```

## Experiments

[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) is the index: one row per experiment, naming the
question it answers, the script that runs it, the configs it uses and the record it writes.
Measurements on externally released models additionally need checkpoints this repository does not
redistribute — [`docs/EXTERNAL_MODELS.md`](docs/EXTERNAL_MODELS.md) says how to obtain each.
[`docs/COMPUTE.md`](docs/COMPUTE.md) reports the hardware, precision policy and cost.

## Data

Every dataset is public. Manifests (`data/manifests/`, each carrying SHA-256 digests of the
processed archives) and split definitions (`data/splits/`) are versioned. The processed Materials
Project archives (`data/raw/mp/*.npz`, 6.7 MB across six files) are tracked: they are the exact
arrays the evaluation reads, they carry the `mp-*` identifiers of the 2,000-crystal centrosymmetric
population in both coordinate variants, and no public endpoint returns them. QM9 is not tracked —
its source tarball is pinned by content hash in `data/manifests/qm9.yaml` and
`scripts/data/prepare_qm9.py` rebuilds the 133,885 `.xyz` files.

```bash
uv run --extra data python scripts/data/prepare_qm9.py
uv run --extra data python scripts/data/prepare_mp.py
```

`data/figure_series/` holds the machine-readable series behind the published figures, one file per
panel, documented in [`data/figure_series/README.txt`](data/figure_series/README.txt).

## Results

`results/` holds the frozen measurement records the study cites, at two levels of granularity.
Per-run summary records cover the matched-pair grid: test error with its seed spread and paired
test per core and target, false-flag fractions and violation medians per arm and coordinate
variant, per-seed values for the augmentation, loss-reweighting and readout-pooling experiments,
the threshold sweep, and the symmetry metadata of all 2,000 evaluation crystals. Every record names
the script that produced it in [`results/README.md`](results/README.md), so a claim traces to a
record and a record traces to code.

For the measurements on released models the per-structure vectors are given in full:
`results/tensor_predictors/` carries the predicted rank-3 tensor and its Frobenius norm for both
dedicated tensor predictors in each of four mask and coordinate conditions; `results/random_init/`
the violation magnitude for three rotation-only potentials at random initialization;
`results/frozen_backbone/` the same for every seed of two frozen-backbone heads. Each is a
2,000-element vector over the evaluation population.

## Layout

```text
src/equiparity/     package code: domain, io, data, features, models, training,
                    evaluation, verification, workflows, reproducibility, cli
tests/              mirrors src/equiparity/, plus the parity verification gate
verification/       claim and theorem checkers that read only results/
configs/            experiment configs (YAML), including the 84-run grid
scripts/            drivers, grouped as data/, grids/, experiments/, analysis/
results/            frozen measurement records, with README.md naming each producer
data/               manifests, splits, processed archives, figure series
docs/               experiment index, external-model sources, compute
vendor/hotpp/       vendored HotPP (MIT, arXiv:2402.15286), no PyPI release
outputs/            per-run results and provenance manifests (not committed)
```

Re-running a driver end to end reads the raw training-run tree, which is not part of this release.
It defaults to `runs/` at the repository root; set `$PARITY_RUNS` to point elsewhere. The analysis
scripts under `scripts/analysis/` and the experiment drivers that aggregate over seeds are the ones
that need it; nothing in `verification/` does.

## Citing this work

The archived release is deposited at Zenodo under DOI
[10.5281/zenodo.22003285](https://doi.org/10.5281/zenodo.22003285). Machine-readable metadata is in
[CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).

## Known limitations

Exact numerical reproducibility is not guaranteed across torch, CUDA or cuDNN releases, or across
devices. Two GPU classes were used and are not interchangeable, so per-core wall-clock is not a
like-for-like architecture comparison; see [`docs/COMPUTE.md`](docs/COMPUTE.md). The CliffordSTF
core is present but was withdrawn from the study for numerical conditioning.
