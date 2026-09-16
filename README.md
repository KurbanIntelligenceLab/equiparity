# equiparity

Code, configurations and measurement records for:

**The parity gap in crystal tensor prediction**

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

Crystal symmetry can determine whether a tensor response must vanish, providing a direct test for
learned predictions from symmetry alone. We derive the parity gap, which measures the
piezoelectric tensor freedom permitted by a crystal's proper rotations and removed by inversion.
Across matched models, forbidden responses follow the gap assigned to each centrosymmetric
crystal class. A polar distortion path links the predicted response to the loss of inversion
symmetry. Regression controls reveal no consistent accuracy cost from parity enforcement in the
tested configurations. Explicit zero labels reduce violations but leave residual forbidden
responses, whereas model construction and output antisymmetrization enforce the zero under the
required invariances.

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Dependencies are declared in
`pyproject.toml` and pinned exactly in `uv.lock`.

```bash
uv sync --extra verify   # CLI, release checks and records; no GPU needed
uv run equiparity --help
```

Training, data preparation and aggregation each need an extra. MACE pins `e3nn==0.4.4` while
NequIP and Allegro require `e3nn>=0.6`, so the two model profiles cannot co-install; CI tests them
as a matrix.

| Extra | Adds | Needed for |
|---|---|---|
| `verify` | sympy | `equiparity verify` |
| `nequip` | NequIP, Allegro, torch-geometric | `equiparity run` with NequIP, Allegro or EquiformerV2 |
| `mace` | MACE (conflicts with `nequip`) | `equiparity run` with MACE |
| `data` | pymatgen, mp-api, spglib | `equiparity data prepare` |
| `analysis` | scipy, matplotlib | `equiparity aggregate` |

## Command-line interface

Every workflow in the study is a subcommand. Commands that print data accept `--json`, and every
command exits non-zero on failure, so the CLI can be scripted or driven by an agent.

| Command | What it does |
|---|---|
| `equiparity verify [--theory] [--claims] [--proofs]` | Reconcile the manuscript's claims with the released records |
| `equiparity records list [--item "Figure 2"]` | Catalogue of result records and the manuscript items they back |
| `equiparity records show NAME` | One record's contents, manuscript items, producer and path |
| `equiparity grid generate {main,meanpool,sumpool,augmentation,loss-weight,zero-injection,all}` | Write an experiment matrix to `configs/` |
| `equiparity run CONFIG` | Train and evaluate one run, writing provenance to `outputs/` |
| `equiparity data prepare {qm9,mp,idealize}` | Build processed archives, dataset manifests and splits |
| `equiparity aggregate --runs PATH` | Rebuild the summary records from a training-run tree |
| `equiparity manifest` | Rebuild `results/run_manifest.json` from the committed configs |

## Verifying the claims

The release checks need no GPU, trained model, or data beyond this repository.

```bash
uv run equiparity verify            # theory + claims
uv run equiparity verify --proofs   # Lean formalization; needs elan (see proofs/README.md)
```

`--theory` checks the parity-gap identities numerically and recomputes the parity-gap table
(Table 1). `--claims` checks every reported quantity against the committed records and confirms
that `results/index.json` catalogues every record. `--proofs` builds the Lean 4 formalization in
[`proofs/`](proofs/README.md) and audits it for unfinished proofs and nonstandard axioms.

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

## Reproducing the experiments

```bash
uv run equiparity grid generate main                                   # 84 configs + run lists
uv run --extra nequip equiparity run configs/grid/nequip_piezoelectric_o3_seed0.yaml
uv run --extra analysis equiparity aggregate --runs /path/to/runs      # summary records
```

Run lists (`configs/<grid>/nequip_runs.txt`, `mace_runs.txt`) group runs by install profile. Each
run writes a provenance manifest, config snapshot and metrics to `outputs/<experiment_id>/`.
`equiparity aggregate` expects the run tree layout used for the study (`metrics/*.json` and
`raw/box*/<run>/`); the raw training-run tree itself is not part of this release.

## Data

Every dataset is public. Manifests (`data/manifests/`, each carrying SHA-256 digests of the
processed archives) and split definitions (`data/splits/`) are versioned. The processed Materials
Project archives (`data/raw/mp/*.npz`, 6.7 MB across six files) are tracked: they are the exact
arrays the evaluation reads, they carry the `mp-*` identifiers of the 2,000-crystal centrosymmetric
population in both coordinate variants, and no public endpoint returns them. QM9 is not tracked; its
source archive is pinned by content hash in `data/manifests/qm9.yaml`.

```bash
uv run --extra data equiparity data prepare qm9   # from the extracted dsgdb9nsd .xyz files
uv run --extra data equiparity data prepare mp    # needs MP_TOKEN in the environment or .env
```

## Results

[`results/`](results/README.md) holds the frozen measurement records behind every number, figure
panel and table in the manuscript, together with `run_manifest.json`, the execution manifest of all
195 training runs. [`results/index.json`](results/index.json) maps each record to the manuscript
items it backs and to what produced it.

## Layout

```text
src/equiparity/   package: domain, io, data, features, models, training, evaluation,
                  verification, workflows, cli
tests/            mirrors src/equiparity/, plus the parity verification gate and release checks
configs/          per-run experiment configs, one directory per grid
results/          frozen measurement records, their index, and the execution manifest
data/             manifests, splits, and processed public-data archives
proofs/           Lean 4 formalization and its audit script
```

## Citing this work

If you use this code or the released records, please cite the article and the software archive:

```bibtex
@article{polat2026paritygap,
  title   = {The parity gap in crystal tensor prediction},
  author  = {Polat, Can and Kurban, Mustafa and Serpedin, Erchin and Kurban, Hasan},
  year    = {2026},
  note    = {Manuscript under review}
}

@software{polat2026equiparity,
  title     = {equiparity: code and measurement records for "The parity gap in crystal tensor prediction"},
  author    = {Polat, Can and Kurban, Mustafa and Serpedin, Erchin and Kurban, Hasan},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22003285},
  url       = {https://github.com/KurbanIntelligenceLab/equiparity}
}
```

## License

MIT; see [LICENSE](LICENSE).

## Known limitations

Exact numerical reproducibility is not guaranteed across torch, CUDA or cuDNN releases, or across
devices. Two GPU classes were used and are not interchangeable, so per-core wall-clock is not a
like-for-like architecture comparison; `results/run_manifest.json` records which GPU ran each core.
