# Mean-pooling control arm: readiness report

This is the code-and-verification half of the reviewer's pooling critique (piezoelectric and
elastic tensors are intensive properties, but every readout sums with an unnormalized
`index_add_`). No GPU was rented and no training grid was launched, per the task's rules; a CPU
dry run proves the path trains end to end.

## 1. What changed (file:line)

### New shared helper

- `src/equiparity/models/pooling.py` (new file) — `pool_per_structure(per_unit, unit_to_graph,
  n_graphs, pooling)` accumulates via the same `index_add_` call the four cores already used,
  then either returns the sum unchanged (`pooling="sum"`) or divides by the per-structure unit
  count (`pooling="mean"`). `validate_pooling` rejects anything else. This is the single place
  the sum/mean algebra lives; every core calls it instead of constructing its own zeros tensor.

### Per-core config + readout (all four cores)

- `src/equiparity/models/nequip.py`: `NequIPConfig.pooling: str = "sum"` field + `__post_init__`
  validation; `NequIPTensorModel.forward` (originally line 217, the reviewer-cited
  `out.index_add_(0, batch_index, per_atom)`) now calls `pool_per_structure(per_atom,
  batch_index, n_graphs, self.pooling)`. Pooling unit: atoms (`AtomicDataDict.BATCH_KEY`).
- `src/equiparity/models/allegro.py`: `AllegroConfig.pooling` field, same validation.
  `AllegroTensorModel.forward` (originally line 205, the reviewer-cited per-**edge**
  `index_add_`) now computes `edge_struct = batch_index[edge_index[0]]` (edge → structure) and
  calls `pool_per_structure(per_edge, edge_struct, n_graphs, self.pooling)`. **The Allegro
  subtlety**: because Allegro's readout has no per-atom message (edge-centric core), "mean"
  here divides by the structure's **edge count**, not atom count — documented in both the
  `pooling.py` module docstring and an inline comment at the call site.
- `src/equiparity/models/mace.py`: `MACEConfig.pooling` field, same validation.
  `MACETensorModel.forward` (originally line 211) now pools atoms via `pool_per_structure`.
- `src/equiparity/models/equiformer.py`: `EquiformerV2Config.pooling` field, same validation.
  `EquiformerV2TensorModel.forward` (originally line 137) now pools atoms via
  `pool_per_structure`.
- In every core, `pooling: str = "sum"` is the dataclass **default** — an existing config with
  no `pooling` key parses to `"sum"` and the forward pass takes the unchanged `sum` branch,
  which is algebraically identical to the pre-patch `index_add_` call (verified bit-identical
  by test, §2).

### Config threading (no parallel mechanism — the existing `ModelHyperparams(**model_raw)` path)

- `src/equiparity/domain/experiment.py`: added `ModelHyperparams.pooling: str = "sum"` +
  `__post_init__` validation (`ConfigError` on anything else). Because `io/config.py` already
  does `ModelHyperparams(**model_raw)` from the YAML `model:` block, a config author can write
  `model: {pooling: mean, ...}` with **zero** changes to the YAML loader — confirmed by
  `tests/io/test_config.py::test_pooling_mean_loads_from_yaml`.
- `src/equiparity/training/nequip_tensor.py`, `nequip_vector.py`, `mace_tensor.py`,
  `equiformer_tensor.py`, `src/equiparity/inference/reload.py`: every call site that constructs
  a per-core `*Config` from `ExperimentConfig` now passes `pooling=config.model.pooling`
  through (six construction sites total: NequIP/Allegro tensor builder, NequIP/Allegro dipole
  builder, MACE tensor-head builder, MACE dipole builder, MACE reload, EquiformerV2 `_config`
  helper — the EquiformerV2 reload path already flowed through `_config` and needed no separate
  edit).
- `src/equiparity/workflows/run_experiment.py`: added `pooling: config.model.pooling` to the
  provenance config snapshot (same rationale as the existing `target_scale` field there —
  omitting it would silently drop pooling mode from the run's config hash and manifest, so a
  mean-pooled run's `outputs/` directory would be indistinguishable from a sum-pooled one on
  disk).

## 2. Test results

New tests (all passing, both environments):

- `tests/models/test_pooling.py` (7 tests, no core dependency) — `pool_per_structure` unit tests:
  sum mode bit-identical to a hand-rolled `index_add_`, mean mode divides by the correct
  per-structure count, mean-of-K-identical-copies recovers the unpooled value (the same algebra
  `scripts/f3_size_consistency.py` measures end to end, restated as a pure-tensor check),
  rejection of unknown modes.
- `tests/io/test_config.py` (+3 tests) — `pooling` defaults to `"sum"` on an existing config
  with no `pooling` key, `model: {pooling: mean}` loads correctly from YAML, an invalid value
  raises `ConfigError`.
- `tests/verification/test_nequip_pooling.py` (10 tests, parametrized over nequip + allegro) —
  per core: (a) `pooling: sum` bit-identical to the pre-patch `index_add_` readout (regression
  guard), (b) `pooling: mean` preserves rotation equivariance (float64, err < 1e-10), (c)
  `pooling: mean` preserves the mirror law (reflection, float64, err < 1e-10), (d) the O(3) arm
  gives an exact zero (< 1e-10) on a centrosymmetric rutile TiO2 cell (P4_2/mnm, subgroup 422 —
  the same E7-safe control the repo already uses in `test_physics_claims.py`/
  `inference/structures.py`) under mean pooling.
- `tests/verification/test_mace_pooling.py` (5 tests) — same four properties for MACE, with
  MACE's known float32-internal precision floor (module docstring in `models/mace.py`; the
  existing `test_mace_gate.py` uses the same 1e-5/1e-6-class bounds rather than NequIP/
  Allegro's 1e-10/1e-12).
- `tests/verification/test_equiformer_pooling.py` (3 tests) — sum-mode regression guard (within
  a single forward call, since EquiformerV2 draws a fresh random per-edge frame on every call —
  `inference/reload.py`'s own `seeded_predict` docstring notes this) and mean-mode rotation
  equivariance. No structural-zero test: EquiformerV2 is "a fixed SO(3) representative" (module
  docstring) with all-even backbone features, so a `ParityMode.O3`-labelled instance has no
  valid odd-parity path from an all-even source and reads out identically zero everywhere — a
  degenerate check, not the Theorem-1 property, so it is correctly omitted rather than reported
  as a positive result.

Baseline regression (existing suite, run twice — before and after all edits, in both
model-family environments used elsewhere in this repo):

| environment | before | after |
|---|---|---|
| `parityinms-nequip` (nequip, allegro, equiformer_v2, clifford tests run) | 76 passed, 5 skipped | 99 passed, 6 skipped |
| `parityinms-mace` (mace tests run) | 72 passed, 7 skipped | 87 passed, 9 skipped |

Zero failures in either environment, before or after. The one added skip in each environment is
the corresponding pooling test file for the core absent from that environment (nequip env skips
`test_mace_pooling.py`; mace env skips `test_nequip_pooling.py` and `test_equiformer_pooling.py`,
the latter for lack of `torch_geometric`). Both runs used `PYTHONPATH=src` rather than a
site-packages install, so edits are picked up live (`pip install -e` is unavailable through the
package-management tool's flag allowlist; a non-editable copy install was tried first and found
stale on edit, then abandoned for this reason).

## 3. Fair-comparison decision

**Target-scale normalization is automatic and requires no code change**, but the sum-pooled
frozen overrides in the existing grid configs (`configs/{e1,h3,t3}/*.yaml`, `target_scale:
0.749134`) must NOT be reused for a mean-pooled run. Every tensor trainer computes
`scale = config.training.target_scale or float(train_targets.std()) or 1.0`
(`nequip_tensor.py:200`, mirrored in `mace_tensor.py` and `equiformer_tensor.py`) — leaving
`target_scale` unset (the default) makes the trainer refit `scale` from `train_targets.std()`
on whatever pooling produced those targets... except the **targets themselves never change**
with pooling (they're ground-truth DFT tensors, not model output); it is the **model's raw
output magnitude** that shrinks by roughly the mean atom count under mean pooling. The refit
protects the loss landscape and the reported MAE/RMSE (both computed in the rescaled, physical
units, via `preds = preds * scale`) — it has no bearing on the OOD violation metric, which is
the number that matters for the false-flag comparison. `scripts/generate_grid_meanpool.py`
leaves `target_scale` unset in every generated config for exactly this reason, so each run
refits independently rather than inheriting a sum-pooled scale.

**The violation threshold is the real fair-comparison problem, and it needs a rescaled
threshold, not the repo's existing per-atom metric.** The headline's OOD violation is `‖T‖_F`
computed directly on the model's (rescaled) tensor output, compared against a fixed **absolute**
threshold of 0.01 C/m² (`snote:threshold`, `supplementary.tex`). Under mean pooling, that same
physical crystal produces a `‖T‖_F` divided by roughly its atom count relative to the sum-pooled
arm — so 0.01 C/m² is no longer the same operating point on the mean-pooled scale, and a naive
run-both-and-compare would be comparing different y-intercepts, not different models.

The repo's existing `f2_per_atom.py` metric (`‖T‖_F / n_atoms` against a threshold rescaled by
the OOD set's median atom count) is **not** the right tool for this comparison, and the
distinction matters: `f2_per_atom.py` answers "is the *sum-pooled* headline's false-flag rate an
artifact of atom-count confound within the sum-pooled arm's own extensive metric?" — it
divides an already sum-pooled output by `n_atoms` post hoc, as a robustness check on one arm.
The mean-pooled control arm's raw model output is *already* a per-structure average — dividing
it by `n_atoms` again would double-normalize, most crystals in the OOD set do not have the
median atom count, so a mean-pooled model's output at 0.01 C/m² is not "the same" event as a
sum-pooled model's output divided down to that scale by `n_atoms`.

**The fair comparison is therefore: compare the two arms' full false-flag-vs-threshold curves on
their own natural output scale, not a single frozen 0.01 C/m² absolute number.** Both training
runs already write exactly this: `metrics.json["ood_variants"]["idealized"]["thresholds"]`
sweeps 25 log-spaced thresholds from 1e-4 to 1 (confirmed by inspection of the dry run's own
`metrics.json`, §4), and the manuscript's supplementary note already states the intent to report
`snote:threshold` this way for the sum-pooled arms — "Machine-readable curves for all 25
thresholds, both variants and every arm accompany the code release." The mean-pooled arm's
curve at its own natural scale is the correct like-for-like comparison to the sum-pooled arm's
curve at its scale: both ask "does an SO(3) model produce a spuriously large output relative to
its own training distribution of outputs," which is threshold-choice-invariant, rather than
"does it exceed one hardcoded absolute number calibrated for the other pooling's scale." A
single-number headline for the mean-pooled control arm (if one is wanted for the main table)
should therefore use a threshold refit the same way `target_scale` is refit — from the
mean-pooled arm's own train-target or train-prediction distribution — never the sum-pooled
0.01 C/m² constant, and never `f2_per_atom.py`'s per-atom-of-a-sum rescaling.

## 4. Dry run (CPU, no GPU rented)

The committed processed npz (`data/raw/mp/mp_piezoelectric_processed.npz`,
`data/raw/mp/mp_ood_centrosymmetric_processed.npz`) was absent from this checkout, and
`scripts/prepare_mp.py`'s network fetch (`mp_api.client.MPRester`) was reachable using the
`MP_API_KEY` credential, so a small **real** subset was fetched instead of a synthetic one: 40
piezoelectric structures (`mp_api.client.MPRester.materials.piezoelectric.search` +
`.materials.summary.search`, same fields `scripts/prepare_mp.py` uses) and 15 spglib-verified
centrosymmetric structures for the OOD set (`equiparity.io.materials_project.space_group_number`
+ `equiparity.domain.spacegroup.is_centrosymmetric`, same functions `prepare_mp.py` calls),
converted through the repo's own `pymatgen_to_structure`/`tensor_sample` boundary functions —
not a parallel conversion path. These live under `data/raw/mp_dryrun/` (isolated from the
canonical `data/raw/mp/` path) plus a matching split file at
`data/splits/mp_piezoelectric_dryrun_split.npz`. An attempt to also stage the OOD file at the
canonical `data/raw/mp/mp_ood_centrosymmetric_processed.npz` path was reverted (deleted via the
sandbox's Trash-safe delete) once it was noticed that this would leave a 15-structure fake file
sitting exactly where a real GPU run's OOD evaluation reads a ~2000-structure real one; a repeat
of `tests/training/test_piezo_ood.py` afterward confirmed it correctly reverted to `SKIPPED...
MP OOD data not present`.

Two configs were run end to end on CPU (`configs/mp_piezoelectric_meanpool_dryrun.yaml`, O3;
`configs/mp_piezoelectric_meanpool_dryrun_so3.yaml`, SO3 — both NequIP, `l_max=3`, 3 layers, 16
features, `pooling: mean`, 3 epochs, `device: cpu`, via `python -m equiparity.cli run <config>
--allow-dirty`):

- O3 arm: completed 3 epochs, `test.mae=0.0174`, `ood_false_flag_fraction=0.0`,
  `ood_violation_median=2.33e-12` — the exact-zero structural property held even on a
  vanishingly small dry-run training set, as expected.
- SO3 arm: completed 3 epochs, `ood_violation_median=6.56e-06` — six orders of magnitude larger
  than the O3 arm's, confirming the parity mechanism the headline depends on is visible under
  mean pooling too, though `ood_false_flag_fraction=0.0` at the frozen 0.01 C/m² threshold on
  this untrained/tiny dry run (expected: an undertrained mean-pooled model's raw output scale on
  15 OOD structures is not representative of a converged 150-epoch run; the dry run's job is to
  prove the pipeline executes, not to estimate the real false-flag rate).
- Both runs produced `config_snapshot.yaml` with `model.pooling: mean` correctly recorded,
  `checkpoint_best.pt`/`checkpoint_latest.pt`, `metrics.json` with the full 25-threshold sweep,
  and `manifest.json` provenance — confirming the whole path (config loading → training loop →
  checkpointing → evaluation → OOD violation computation) executes under `pooling: mean`.

### Exact GPU commands for the full grid

`scripts/generate_grid_meanpool.py` (new; reuses `generate_grid.py`'s `CORE_PARITY`/`PROFILE`/
`SEEDS`/`TARGET` tables so the mean-pooled configs are field-for-field identical to the existing
sum-pooled grid except for one added `pooling: mean` line — verified by diffing a generated
config against its sum-pooled counterpart in `configs/grid/`, which differs by exactly that one
line) writes 42 configs to `configs/grid_meanpool/` — restricted to `elastic` and
`piezoelectric` (the two intensive targets the reviewer's critique concerns; U0/dipole are
molecular QM9 targets with no supercell/pooling-scale question) across all four cores, both
parity arms where valid (NequIP/Allegro/MACE: O3+SO3; EquiformerV2: SO3-only), 3 seeds:

```bash
python scripts/generate_grid_meanpool.py   # writes configs/grid_meanpool/*.yaml (42 files)

# nequip-profile runs (NequIP, Allegro, EquiformerV2 — 30 configs)
for cfg in $(cat configs/grid_meanpool/nequip_runs.txt); do
    python -m equiparity.cli run "$cfg"
done

# mace-profile runs (MACE — 12 configs)
for cfg in $(cat configs/grid_meanpool/mace_runs.txt); do
    python -m equiparity.cli run "$cfg"
done
```

Each config sets `device: cuda`; run on a machine with the corresponding extras installed
(`uv sync --extra nequip` / `uv sync --extra mace`), matching the existing grid's docker-profile
split.

### Estimated GPU-hours

Computed from `docs/results/a1_per_seed.md`'s per-run `train_seconds` column (the sum-pooled
headline's own measured wall-clock times), filtered to the 42 rows whose `target` is
`piezoelectric` or `elastic` — the identical `{core, parity, target, seed}` combination set the
mean-pooled grid reproduces:

| core | total train time (existing sum-pooled runs) |
|---|---|
| NequIP (12 runs) | 11,909 s = 3.31 h |
| Allegro (12 runs) | 8,193 s = 2.28 h |
| MACE (12 runs) | 60,291 s = 16.75 h |
| EquiformerV2 (6 runs) | 39,219 s = 10.89 h |
| **Total (42 runs)** | **119,612 s = 33.23 GPU-h** |

Mean pooling replaces one `index_add_` with the same `index_add_` plus one division by a
per-structure scalar count — a negligible per-step cost difference — so **33.2 GPU-hours** is
the estimate for the full mean-pooled grid, assuming the same hardware and batch sizes as the
sum-pooled headline run. This does not include EqV2-on-U0/dipole (excluded — molecular targets,
no pooling question) or any additional seeds/ablations beyond the existing 3-seed grid.

## Files written

- `src/equiparity/models/pooling.py` (new)
- `src/equiparity/models/{nequip,allegro,mace,equiformer}.py` (modified)
- `src/equiparity/domain/experiment.py` (modified)
- `src/equiparity/training/{nequip_tensor,nequip_vector,mace_tensor,equiformer_tensor}.py`
  (modified)
- `src/equiparity/inference/reload.py` (modified)
- `src/equiparity/workflows/run_experiment.py` (modified)
- `tests/models/test_pooling.py` (new)
- `tests/io/test_config.py` (modified — 3 new tests)
- `tests/verification/{test_nequip_pooling,test_mace_pooling,test_equiformer_pooling}.py` (new)
- `scripts/generate_grid_meanpool.py` (new)
- `configs/grid_meanpool/*.yaml` (42 new, generated)
- `configs/mp_piezoelectric_meanpool_dryrun{,_so3}.yaml` (new, dry-run only)
- `data/raw/mp_dryrun/*.npz`, `data/splits/mp_piezoelectric_dryrun_split.npz` (new, dry-run only
  — real MP data fetched via `MP_API_KEY`, isolated from the canonical `data/raw/mp/` path)
- `docs/results/f5_pooling_arms.md` (this file)

`docs/draft/sections/*.tex` was not touched, per the task's rules.
