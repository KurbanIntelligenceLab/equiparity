# Methods

This document specifies the experimental design, data, models, metrics, and training protocol in
sufficient detail to reproduce the study. Results are reported separately in [`RESULTS.md`](RESULTS.md).

---

## 1. Physical setting

Neumann's principle states that any macroscopic physical property tensor of a crystal must be
invariant under every symmetry operation of the crystal's point group. The piezoelectric tensor
`e_{ijk}` relates polarisation to strain and is a **parity-odd, rank-3** Cartesian tensor: under
spatial inversion `I: r -> -r` it transforms as `e -> (-1)^3 e = -e`.

If a crystal is **centrosymmetric** — its space group contains the inversion operation — then Neumann's
principle requires `e = -e`, hence

> **e = 0 exactly, for every centrosymmetric crystal.**

This is not an empirical regularity but an algebraic identity. It provides a falsifiable, label-free
test: any model that predicts a nonzero piezoelectric tensor for a centrosymmetric crystal has made a
physically impossible prediction, and no ground-truth measurement is needed to detect it.

There are 230 crystallographic space groups, of which **92 are centrosymmetric** (the 11 centrosymmetric
Laue classes; enumerated in `src/equiparity/domain/spacegroup.py`).

### The experimental variable

An **O(3)-equivariant** network carries a parity label on every internal feature (irreducible
representations of O(3), written `l e` for even and `l o` for odd). Composition rules then force the
odd-parity output to vanish for a centrosymmetric input: the zero is *structural*, holding at any
parameter values, including at initialisation.

An **SO(3)-equivariant** network is identical except that parity labels are removed (all irreps
declared even). It remains exactly rotation-equivariant, but it is no longer reflection-equivariant,
so nothing forces the odd-parity output to zero.

The study holds architecture, hyperparameters, data, and seeds fixed and varies **only** this labelling.

---

## 2. Models

Four cores. Three are matched O(3)/SO(3) pairs that share one architecture and differ only in the
parity labelling. The fourth, EquiformerV2, is a fixed, widely-deployed model that is SO(3)-only by
construction and has no O(3) arm.

| Core | Family | Parity arms |
|---|---|---|
| NequIP | e3nn convolutional message-passing | O(3), SO(3) |
| Allegro | e3nn strictly-local descriptors | O(3), SO(3) |
| MACE | e3nn body-ordered message-passing | O(3), SO(3) |
| EquiformerV2 | spherical-harmonic transformer (eSCN) | SO(3) only |

Full grid: **7 arms × 4 targets × 3 seeds = 84 runs.**

### 2.1 The parity toggle

All arms are built through one helper (`src/equiparity/models/irreps.py:14-27`):

```python
def degree_irreps(l_max, mult, mode):
    terms = []
    for degree in range(l_max + 1):
        even = not mode.has_parity or degree % 2 == 0
        parity = "e" if even else "o"
        terms.append(f"{mult}x{degree}{parity}")
    return " + ".join(terms)
```

- **O(3)** — natural spherical-harmonic parity `(-1)^l`: `Nx0e + Nx1o + Nx2e + ...`
- **SO(3)** — every degree labelled even: `Nx0e + Nx1e + Nx2e + ...`

The geometric content is identical; only the parity bookkeeping differs, so e3nn stops enforcing
reflection equivariance. Both edge spherical harmonics and hidden irreps are relabelled.

Per-core implementation:

- **NequIP / Allegro.** The framework's preset `parity` boolean is **not** an SO(3) toggle: with
  `parity=False` the model remains fully O(3)-equivariant, because the source builds natural-parity
  spherical harmonics unconditionally. A genuine SO(3) arm therefore requires the raw-irreps route,
  `FullNequIPGNNModel` / `FullAllegroModel`, with all-even edge SH. Allegro additionally relabels
  `tensor_track_allowed_irreps`. (`models/nequip.py:61-101`, `models/allegro.py:51-80`)
- **MACE.** Exposes a correct native toggle, `use_so3=not mode.has_parity`, which builds all-even
  spherical harmonics (`p=1`). No patch required. (`models/mace.py:75`)
- **EquiformerV2.** Its `SO3_Embedding` features carry no parity labels. It has no O(3) mode; that
  absence is the point of including it.

### 2.2 Output heads

Targets are expressed as O(3) irreps (`src/equiparity/domain/target.py`), ordered here by parity
character rather than tensor rank:

| Target | Irreps | Components | Parity | Unit |
|---|---|---|---|---|
| U₀ | `1x0e` | 1 | even | eV |
| dipole μ | `1x1o` | 3 | **odd** | Debye |
| elastic C | `2x0e + 2x2e + 1x4e` | 21 | even | GPa |
| piezoelectric e | `2x1o + 1x2o + 1x3o` | 18 | **odd** | C/m² |

`output_irreps()` (`models/irreps.py:30-40`) applies the toggle to the head: the SO(3) arm rewrites the
target irreps with `.replace("o", "e")`. This is precisely the step that permits an SO(3) model to emit
a symmetry-forbidden nonzero odd tensor.

Cartesian↔irreps change of basis uses `e3nn.o3.ReducedTensorProducts` to encode the index symmetries
(`features/tensor_irreps.py:30-41`):

```python
o3.ReducedTensorProducts("ijk=ikj",  i="1o", j="1o", k="1o")            # piezoelectric
o3.ReducedTensorProducts("ijkl=jikl=klij", i="1o", j="1o", k="1o", l="1o")  # elastic
```

The change of basis is **orthonormal**, so the Frobenius norm is preserved: `‖T‖_irreps = ‖T‖_Cartesian`.
Voigt order is `[xx, yy, zz, yz, xz, xy]`; Materials Project stores direct tensor components with no
engineering factor of two.

**The dipole head is a direct equivariant L=1 readout**, never a charge × position construction. The
charge-weighted sum is used only to build the *label* (§3.1); the model predicts the vector directly
from equivariant features via `o3.Linear`. The O(3) arm reads out `1o`; the SO(3) arm reads out `1e`.

Each core reads out from the deepest layer whose irreps still carry `l > 0`, then sums the per-atom (or
per-edge) contributions per structure, which preserves equivariance:

| Core | Readout site |
|---|---|
| NequIP | penultimate convnet layer (`layer{L-2}_convnet`), per-atom |
| Allegro | deepest `tps` layer with `l > 0`, per-edge (the final tps collapses to `1x0e`) |
| MACE | deepest `interactions.{i}.linear` with `l > 0`, selected by `_deepest_tensor_probe()` |
| EquiformerV2 | `backbone.norm` output; `SO3_Embedding` `(l,m)`-major coefficients remapped to e3nn's mul-major layout |

MACE's probe was originally hardcoded to `interactions.0.linear`, the shallowest block, whose irreps are
effectively scalar. This under-specified the L=1 dipole head; `_deepest_tensor_probe()` corrects it. The
O(3) piezoelectric zero is unaffected by the choice of probe, since it is structural.

---

## 3. Data

Raw datasets are external and not committed; `data/manifests/` records provenance and hashes, and
`data/splits/` records split definitions. All splits use **seed 42**.

### 3.1 QM9

Source: `dsgdb9nsd`, figshare 978904 (2014 release, CC0). The 3,054 uncharacterised molecules
(figshare 3195404) are excluded, leaving **130,831 of 133,885**.

- **U₀** — internal energy at 0 K, converted Hartree→eV (`HARTREE_TO_EV = 27.211386245988`). This is
  the **total** energy, with no atomic-reference (atomref) subtraction. The processed distribution has
  mean −11,178.97 eV and standard deviation **1,085.57 eV**.
- **dipole μ** — the label is constructed as `Σ_i q_i r_i` from the reference Mulliken charges and
  converted to Debye (`E_ANGSTROM_TO_DEBYE = 4.803204544`). A Mulliken-charge dipole underestimates the
  DFT magnitude |μ| by roughly 10–25 %; its **direction and parity are exact**, which is what the parity
  comparison requires. The magnitude bias applies identically to both arms.

Split: random, seed 42 — **110,000 train / 10,000 validation / 10,831 test**.

### 3.2 Materials Project

Fetched 2026-07-04 via `mp_api`; all data CC-BY-4.0.

| Dataset | Query / field | Filter | Kept | Dropped | Split (train/val/test) |
|---|---|---|---|---|---|
| Piezoelectric | `materials.piezoelectric.search`, `d.total` (DFPT, 3×6 Voigt → 18) | drop \|e\| > 50 C/m² | **3,312** | 4 | 2,649 / 331 / 332 |
| Elastic | `materials.elasticity.search`, `elastic_tensor.ieee_format` (6×6 → 36) | drop \|C\| > 2000 GPa (failed DFT) | **13,080** | 147 | 10,464 / 1,308 / 1,308 |

Split fractions are 80/10/10 (`n_train = int(0.8N)`, `n_val = int(0.1N)`, remainder test).

### 3.3 Out-of-distribution set: centrosymmetric crystals

Selection (`scripts/prepare_mp.py:186-226`):

1. Query `materials.summary.search` restricted to the **92 centrosymmetric space groups**, with
   `band_gap > 0.1 eV` (insulators — piezoelectricity is defined for insulators).
2. Shuffle (seed 42), then **re-verify each candidate's space group with spglib** at
   `symprec = 1e-3` via pymatgen's `SpacegroupAnalyzer`, keeping only structures confirmed
   centrosymmetric.
3. Cap at 2,000.

Result: **2,000 verified structures, 103 rejected** as spglib/MP mismatches. No labels are stored — the
true tensor is exactly zero by symmetry.

**Two coordinate variants are evaluated for every model.** DFT-relaxed coordinates satisfy inversion
symmetry only to within the relaxation tolerance, so a model that is exactly O(3)-equivariant can still
return a small nonzero tensor from the residual asymmetry of its input. To separate the *structural*
guarantee from *robustness on real data*, both are kept:

| Variant | Construction | Atoms |
|---|---|---|
| **idealized** | `spglib.standardize_cell(to_primitive=True, no_idealize=False, symprec=1e-3)` — coordinates snapped onto the exact space group | 72,280 |
| **raw** | the unmodified DFT-relaxed coordinates | 72,642 |

Both files contain the same 2,000 materials in the same order, so per-structure predictions are
directly comparable.

The selection tolerance interacts with the raw variant, and this is quantified in
[`docs/results/a5_ood_symmetry.md`](docs/results/a5_ood_symmetry.md): all 2,000 raw structures are
centrosymmetric at `symprec = 1e-3`, but only 1,956 (97.8 %) at `1e-4` and 1,869 (93.5 %) at `1e-5`.
For structures below tolerance, a nonzero piezoelectric tensor is not, strictly, symmetry-forbidden.
This is a property of the data, not of any model, and it must be accounted for when interpreting the
raw variant.

---

## 4. Metrics

### 4.1 Accuracy

Mean absolute error and root-mean-square error, computed per component on the flattened tensor and
reported in the target's physical unit (`evaluation/metrics.py`). Predictions are un-normalised before
metrics are computed.

### 4.2 Violation magnitude

For each OOD structure, the **violation magnitude** is the Frobenius norm of the predicted tensor
(`training/nequip_tensor.py:77-83`):

```python
def violation_magnitudes(predictions):
    return np.sqrt((predictions**2).sum(axis=1))
```

Because the irreps basis is orthonormal this equals the Cartesian Frobenius norm. The true value is
exactly zero for every structure in the OOD set, so this quantity *is* the error.

**False-flag fraction** at threshold τ is the fraction of OOD structures whose violation exceeds τ:

```python
false_flag(τ) = (magnitudes > τ).mean()
```

The headline operating point is **τ = 0.01 C/m²**. Rather than assert that this is "materials-relevant",
we report the full curve over **25 log-spaced thresholds from 1e-4 to 1** (`ood_eval.py:18`) and
calibrate τ against the empirical distribution of real piezoelectric tensors: 98.53 % of the
piezoelectric crystals in the training split have `‖e‖_F > 0.01 C/m²` (median 0.5086 C/m²). A model
whose violation exceeds τ is therefore predicting a response that would be read as a genuine
piezoelectric.

Per-structure violation vectors are saved (`ood_violations_{idealized,raw}.npy`, 2,000 floats each) so
that distributions, threshold curves, and paired structure-level tests can be recomputed offline.

**The metric is extensive.** The readout sums per-atom (or per-edge) contributions, so ‖T‖ grows with
system size: Spearman ρ(violation, n_atoms) ranges 0.49–0.69 across arms, for *both* parity modes. A
fixed absolute threshold is therefore mildly size-dependent. This does not affect the O(3) result,
whose zero is structural and size-independent, but it should be kept in mind when reading absolute
SO(3) violation magnitudes.

---

## 5. Training protocol

Identical for both arms of every matched pair; the parity labelling is the only difference.

| Setting | Value |
|---|---|
| Optimiser | Adam, `lr = 2e-3`, `weight_decay = 0` |
| LR schedule | **none** (constant) |
| Loss | MSE on normalised targets |
| Layers / `r_max` / features | 3 / 5.0 Å / 64 |
| Precision | float32 weights, **float64 geometry** |
| Early stopping | none; every run trains the full epoch budget |

Per-target:

| Target | Dataset | `l_max` | Epochs | Batch | Train samples used |
|---|---|---|---|---|---|
| U₀ | QM9 | 2 | 100 | 32 | 25,000 (capped) |
| dipole | QM9 | 2 | 100 | 32 | 25,000 (capped) |
| elastic | MP elastic | 2 | 120 | 16 | 10,464 (full) |
| piezoelectric | MP piezoelectric | 3 | 150 | 16 | 2,649 (full) |

**Target normalisation.** Scalars are standardised by the training mean and standard deviation.
Vector and tensor targets are scaled by the training standard deviation **only** — subtracting a mean
would add a constant, non-equivariant offset and destroy the very property under test. Predictions are
multiplied back by `scale` before metrics and OOD evaluation.

**Checkpoints.** Validation MAE is evaluated every `max(1, epochs // 10)` epochs; the lowest-val
checkpoint is retained (`checkpoint_best.pt`), alongside a resumable `checkpoint_latest.pt`
(model + optimiser + epoch). Reported metrics are taken at the **final** epoch, not at the best
checkpoint.

### 5.1 A caveat on the U₀ experiment

U₀ is the parity-even scalar control: it establishes that removing parity labels costs nothing when no
symmetry constraint is active. It is *not* a claim of competitive QM9 accuracy, and should not be read
as one. Three choices make the absolute error large:

1. the target is **total** internal energy with no atomic-reference subtraction, so the model must
   learn a quantity spanning ~18,000 eV with standard deviation 1,086 eV;
2. training uses a **25,000-molecule subset** of the 110,000 available;
3. there is no learning-rate schedule and only 100 epochs.

The resulting MAEs (16.8–53.7 eV, i.e. 1.5–4.9 % of the target standard deviation) are far from the ~5 meV
achieved by tuned models on atomisation energies. Because **both arms share this configuration
exactly**, the O(3)-vs-SO(3) comparison remains valid; only the absolute scale is uninformative. U₀
also has by far the largest seed-to-seed variance of the four targets, which limits the resolution of
that comparison (see `RESULTS.md` §8).

---

## 6. Parity verification gate

The parity toggle was verified numerically **before** any training, because a mislabelled arm would
invalidate the entire study (`src/equiparity/verification/equivariance.py`, `scripts/parity_audit.py`).

For each core and arm, an internal equivariant feature layer is probed with a random proper rotation
(det = +1) and a random improper reflection (det = −1), and the error

```
max | feat(g·x) − D(g)·feat(x) |
```

is measured against e3nn's parity-aware representation `D(g)`. An O(3) arm must satisfy both; an SO(3)
arm must satisfy rotation and **fail** reflection.

| Core | Arm | irreps | rotation err | reflection err | params | verdict |
|---|---|---|---|---|---|---|
| NequIP | O(3) | `16x0e+16x1o+16x2e` | 4.7e-16 | 4.7e-16 | 42,304 | O3 |
| NequIP | SO(3) | `16x0e+16x1e+16x2e` | 4.7e-16 | 9.7e-03 | 45,376 | SO3 |
| Allegro | O(3) | `1x0e+1x1o+1x2e` | 5.4e-15 | 3.3e-15 | 12,608 | O3 |
| Allegro | SO(3) | `1x0e+1x1e+1x2e` | 9.3e-15 | 4.9e+00 | 12,640 | SO3 |
| MACE | O(3) | `16x0e+16x1o+16x2e` | 2.4e-07 | 9.4e-08 | 43,280 | O3 |
| MACE | SO(3) | `16x0e+16x1e+16x2e` | 2.4e-07 | 1.3e+00 | 50,512 | SO3 |

Rotation equivariance holds in every arm; only the SO(3) arms break reflection. A positive control
confirms the probe discriminates: a hand-built all-even construction yields reflection error ~1e-2,
while the natural-parity build yields ~1e-16.

Two caveats are recorded rather than hidden. **MACE's** symmetric-contraction tensors remain float32
even under a float64 cast, so its equivariance error floors near 1e-7; the gate applies float32
thresholds to MACE alone. This gate ran at **16 features** (not the production 64) and `l_max = 2`; the
parameter counts above are gate-scale. **EquiformerV2** is excluded from this internal-feature probe —
its eSCN features are not parity-labelled irreps — and is verified instead at the output level, by the
OOD test itself.

Note that in every arm the SO(3) parameter count **exceeds** the O(3) count, because all-even labelling
opens additional tensor-product paths. This holds at production scale as well (`RESULTS.md` §6), so
capacity cannot explain any SO(3) deficit.

---

## 7. Models considered and excluded

**Equiformer v1** — demoted, not silently omitted. Its 2022 dependency stack (Python 3.8 / torch 1.10 /
CUDA 11.3) does not build for the RTX 5090 (sm_120), and it is e3nn-based, so as a fourth toggleable
core it would exercise no equivariance mechanism not already covered by NequIP, Allegro, and MACE.
EquiformerV2 supersedes it as the transformer representative.

**CliffordSTF** — a geometric-algebra (non-e3nn) O(3) core, intended to show that the finding is a
property of O(3)-equivariance rather than of the e3nn implementation. It was dropped as a
**conditioning failure, not a scientific counterexample**, and its runs are archived rather than
deleted. In exact arithmetic it is O(3)-exact: on machine-perfect centrosymmetric input it cancels to
~1e-15. But Cl(3,0) grades reach only `l = 1`, so an `l = 3` output must be assembled through a
**cubic** tensor product, which is ill-conditioned. Once trained, it amplifies the ~1e-6 residual
coordinate asymmetry of real crystals by 3,000–25,000×, producing a false-flag fraction of 0.42. That
number measures the readout's condition number, not the parity of the algebra.

The observation is reportable in its own right: *O(3)-equivariance guarantees an exact zero in theory,
but realising that guarantee numerically on real data requires a well-conditioned (linear) readout of
native high-order features.* ICTP was surveyed as a replacement and rejected — its rank-3 features are
embedded in internal product-basis tensors with no clean linear readout.

Consequently **all three O(3) arms in this study share the e3nn irrep implementation**, which is the
study's principal limitation (`RESULTS.md` §13).

Details: [`docs/checkpoint7_report.md`](docs/checkpoint7_report.md).

---

## 8. Reproducibility

Every run writes `outputs/<experiment_id>/` with:

- `manifest.json` — `git_sha`, `git_dirty`, `config_hash`, `config_path`, `dataset_manifest`,
  `split_manifest`, `seed`, `python_version`, `package_version`, `timestamp_utc`, `hostname`,
  `gpu_model`, `cuda_version`, `driver_version`
- `config_snapshot.yaml` — the exact hashed configuration
- `metrics.json` — accuracy, `n_params`, `epochs_run`, `ood_variants`, `timing`
- `ood_violations_{idealized,raw}.npy`, `checkpoint_best.pt`, `checkpoint_latest.pt`

`experiment_id = <git_sha[:8]>_<config_hash[:8]>_<utc_timestamp>`. `write_manifest` **refuses to write
when the git tree is dirty** unless explicitly overridden, which protects final-result runs.
`seed_everything` fixes `PYTHONHASHSEED`, `random`, NumPy, and torch (including CUDA), enables
deterministic algorithms, and disables cuDNN benchmarking.

### Environment

MACE pins `e3nn == 0.4.4`; NequIP and Allegro require `e3nn >= 0.6`. They cannot share an environment,
so `pyproject.toml` declares them as **conflicting uv extras** over a shared base — one lockfile, two
mutually exclusive install profiles. Each run's manifest records which profile it used.

```bash
uv sync --extra nequip   # nequip, nequip-allegro, equiformer_v2  (e3nn 0.6)
uv sync --extra mace     # mace-torch                             (e3nn 0.4.4)
```

Resolved pins: Python 3.12; `torch 2.11.0+cu128`; `nequip 0.18.0`; `nequip-allegro 0.8.3`;
`mace-torch 0.3.16`; `e3nn 0.6.0` (nequip profile) / `0.4.4` (mace profile); `pymatgen 2026.5.4`;
`spglib 2.7.0`; `mp-api 0.46.4`; `ase 3.29.0`; `numpy 2.5.1`; `scipy 1.18.0`.

Exact numerical reproducibility is not guaranteed across torch/CUDA/cuDNN releases or devices.

### Regenerating the analysis

```bash
python3 scripts/analyze_results.py                              # tables, curves, appendices A1–A4
uv run --extra nequip python scripts/analyze_ood_symmetry.py    # appendix A5 (needs spglib)
```
