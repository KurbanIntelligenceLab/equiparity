# equiparity — When Parity Matters

Can SO(3)-equivariant models predict physically impossible material properties?

O(3)-equivariant models, whose features carry parity labels, produce symmetry-forced zeros
for the piezoelectric tensor of centrosymmetric crystals by construction. SO(3)-equivariant
models, which have no parity labels, fail: they predict nonzero tensors for cases that are
zero by Neumann's principle. This repository holds the study that quantifies when parity
matters — never for energies, marginally for dipoles, categorically for odd-parity tensors —
and turns it into a practitioner rule.

## The manuscript

The submission is [`docs/draft/`](docs/draft/). Its figure generator, figure-data inputs and
validators sit beside it, so the paper and the code that produces it move together.

```bash
cd docs/draft
pdflatex main && bibtex main && pdflatex main && pdflatex main
python build_figures.py     # writes figures/, byte-identical on rerun
python verify_figures.py    # every hard-coded figure number against its source
python audit_numbers.py     # every quantity the manuscript reports twice
```

[`docs/draft/REPRODUCE.md`](docs/draft/REPRODUCE.md) maps each figure to its generator and its
data source, lists the validators, and states which claims were executed rather than read.
[`docs/draft/TODO_SUBMISSION.md`](docs/draft/TODO_SUBMISSION.md) lists what is still open.

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

`scratch_hotpp/` vendors HotPP, which has no PyPI release under that name;
`scripts/f4_noneN3_control.py` imports it from there, and the Supplementary Information's
claim that it carries no e3nn dependency is checkable against that tree.

## Layout

```text
src/equiparity/   package code: domain, io, data, features, models, training,
                  evaluation, verification, workflows, reproducibility, cli
tests/            mirrors src/equiparity/, plus the parity verification gate
configs/          experiment configs (YAML), including the 84-run grid
scripts/          orchestration and experiment drivers, no scientific logic
results/          frozen measurement records the manuscript cites
docs/draft/       the submission: sources, figures, generator, figdata, validators
docs/results/     hand-written appendices with no generator
outputs/          per-run results and provenance manifests (not committed)
```

Appendices produced by a script are not committed; they regenerate:

```bash
uv run python scripts/analyze_results.py                        # a1-a4
uv run --extra data python scripts/analyze_ood_symmetry.py      # a5
```

## Verification

The parity verification gate is a Go/No-Go: no config may train unless both arms of every
core in the profile pass it.

```bash
uv run --extra nequip pytest tests -q
uv run --extra nequip pytest tests/verification -q
```

## License

MIT — see [LICENSE](LICENSE). Citation metadata (`CITATION.cff`) is added at release.

## Known limitations

Exact numerical reproducibility is not guaranteed across torch, CUDA or cuDNN releases, or
across devices. Two GPU classes were used and are not interchangeable, so per-core
wall-clock is not a like-for-like architecture comparison. The CliffordSTF core is present
but was withdrawn from the study for numerical conditioning, as the Supplementary
Information records.
