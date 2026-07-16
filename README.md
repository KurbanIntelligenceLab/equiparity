# equiparity — When Parity Matters

Can SO(3)-equivariant models predict physically impossible material properties? This
repository holds one scientific study: O(3)-equivariant models (features carry parity
labels) produce symmetry-forced zeros for the piezoelectric tensor of centrosymmetric
crystals by construction, while SO(3)-equivariant models (no parity labels) fail —
predicting nonzero tensors for cases that are zero by Neumann's principle. The paper
quantifies when parity matters (never for energies, somewhat for dipoles, categorically
for odd-parity tensors) and turns it into a practitioner rule.

## The study

| Document | Contents |
|---|---|
| [`INTRO.md`](INTRO.md) | the physical constraint, the mechanism under test, the design |
| [`METHODS.md`](METHODS.md) | data, models, parity toggle, metrics, training, verification gate |
| [`RESULTS.md`](RESULTS.md) | all measurements as tables, with limitations |
| [`results/`](results/) | appendices as machine-readable data + generated tables (`tables.md`, `tables_extra.md`): per-run values, threshold curves, distributions, symmetry audit |

Regenerate every table and figure:

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
runs on an RTX 5090 (CUDA 12.8 torch build).

## Data access

Raw datasets are external and never committed; only manifests (`data/manifests/`) and
split definitions (`data/splits/`) are versioned. Datasets: QM9 (MoleculeNet), MP Elastic
and MP Piezoelectric (Materials Project API), a centrosymmetric OOD evaluation set derived
from MP, and JARVIS-DFT as the piezoelectric fallback.

## Reproduce main results

Each experiment writes a provenance `manifest.json` into its
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
results/          machine-readable appendices, generated tables and figures
outputs/          per-experiment results and manifests (not committed)
```

## Citation

SOON

## License

MIT — see [LICENSE](LICENSE).

## Known limitations

Exact numerical reproducibility is not guaranteed across torch/CUDA/cuDNN releases or
devices. The study scope is fixed to the architectures and datasets in the work plan;
additional architectures are revision-scope, not first-submission.
