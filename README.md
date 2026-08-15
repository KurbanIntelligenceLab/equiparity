# equiparity — When Parity Matters

Can SO(3)-equivariant models predict physically impossible material properties? This
repository holds one scientific study: O(3)-equivariant models (features carry parity
labels) produce symmetry-forced zeros for the piezoelectric tensor of centrosymmetric
crystals by construction, while SO(3)-equivariant models (no parity labels) fail —
predicting nonzero tensors for cases that are zero by Neumann's principle. The paper
quantifies when parity matters (never for energies, somewhat for dipoles, categorically
for odd-parity tensors) and turns it into a practitioner rule.

## The manuscript

The submission is [`docs/draft/`](docs/draft/). Its figure generator, figure-data inputs and
validators sit beside it, so the paper and the code that produces it move together.
[`docs/draft/REPRODUCE.md`](docs/draft/REPRODUCE.md) maps each figure to its generator and
its data source, gives the exact commands, and states what has not been verified.
[`docs/draft/TODO_SUBMISSION.md`](docs/draft/TODO_SUBMISSION.md) lists the open items.

```bash
cd docs/draft
python build_figures.py     # writes figures/, byte-identical on rerun
python verify_figures.py    # every hard-coded figure number against its source
python audit_numbers.py     # every quantity the manuscript reports twice
```

## The study

| Document | Contents |
|---|---|
| [`INTRO.md`](INTRO.md) | the physical constraint, the mechanism under test, the design |
| [`METHODS.md`](METHODS.md) | data, models, parity toggle, metrics, training, verification gate |
| [`RESULTS.md`](RESULTS.md) | all measurements as tables, with limitations |
| [`docs/results/`](docs/results/) | 20 per-experiment appendices: per-run values, threshold curves, distributions, compute, symmetry audit |

Regenerate the study tables and appendix curves:

```bash
python3 scripts/analyze_results.py                            # tables, curves, appendices A1–A4
uv run --extra nequip python scripts/analyze_ood_symmetry.py  # appendix A5 (needs spglib)
```

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The core ML stack (nequip, nequip-allegro, mace-torch, e3nn, torch) and materials tooling
(pymatgen, mp-api, spglib, jarvis-tools) are added on top of the base environment; exact
versions are frozen in `uv.lock`. Training targets NVIDIA A100 GPUs; local verification
runs on an RTX 5090 (CUDA 12.8 torch build). See [`.claude/CLAUDE.md`](.claude/CLAUDE.md)
for pinned versions and the correction to the work plan's stale pins.

## Data access

Manifests (`data/manifests/`, with SHA-256 digests) and split definitions
(`data/splits/`) are versioned. The processed Materials Project archives
(`data/raw/mp/*.npz`, 6.7 MB) are also tracked: they are the exact arrays the evaluation
reads and no public endpoint returns them. QM9 is not tracked — its source tarball is
pinned by content hash in `data/manifests/qm9.yaml` and `scripts/prepare_qm9.py` rebuilds
the 133,885 `.xyz` files from it. Datasets: QM9, MP Elastic and MP Piezoelectric
(Materials Project API), a centrosymmetric OOD evaluation set derived from MP, and
JARVIS-DFT for the SOTA-predictor comparison.

`scratch_hotpp/` vendors HotPP, which has no PyPI release under that name;
`scripts/f4_noneN3_control.py` imports it from there.

## Reproduce main results

Each experiment writes a provenance `manifest.json` (see CODING_RULES.md §E) into its
`outputs/<experiment_id>/` directory. Reproduction commands are added as the experiment
workflows land; run order is fixed as QM9 → MP Elastic → Piezoelectric OOD → parameter-matched.

## Repository layout

```text
src/equiparity/   importable package code (domain, io, data, features, models,
                  training, evaluation, workflows, reproducibility, cli)
tests/            mirrors src/equiparity/
configs/          versioned experiment configs (YAML)
data/manifests/   dataset manifests (hashes, sources, licenses)
data/splits/      split definitions (seed, method, counts)
scripts/          orchestration only, no scientific logic
results/          frozen measurement records the manuscript cites
outputs/          per-experiment results and manifests (not committed)
docs/draft/       the submission: sources, figures, figure generator, figdata, validators
docs/results/     per-experiment appendices
to_be_deleted/    material retired by consolidation (gitignored; see its MANIFEST.md)
```

[`CONSOLIDATION_REPORT.md`](CONSOLIDATION_REPORT.md) records what was retired and why, what
was verified before anything moved, and the before/after accounting.

## Citation

Citation metadata (`CITATION.cff`) is added at release.

## License

MIT — see [LICENSE](LICENSE).

## Known limitations

Exact numerical reproducibility is not guaranteed across torch/CUDA/cuDNN releases or
devices. The study scope is fixed to the architectures and datasets in the work plan;
additional architectures are revision-scope, not first-submission.
