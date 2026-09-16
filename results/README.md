# Result records

Every number, figure panel and table in the manuscript is backed by a file in this directory.
[`index.json`](index.json) is the machine-readable catalogue: for each record it lists its contents,
the manuscript items it backs, and the command that rebuilds or checks it where one exists. Do not
edit records by hand.

```bash
uv run equiparity records list                    # catalogue
uv run equiparity records list --item "Figure 2"   # records behind one display item
uv run equiparity verify                          # reconcile reported values with these records
```

## Manuscript display items

| Item | Records |
|---|---|
| Table 1 | `parity_gap_table.csv` |
| Figure 1c | `distribution_percentiles.json`, `stats.json` |
| Figure 2a | `stats.json`, `rotation_subgroup.json` |
| Figure 2b | `symmetry_breaking.csv`, `symmetry_breaking.json` |
| Figure 3a | `stats.json` |
| Figure 3b | `zero_injection_curve.json`, `zero_injection_sets.json` |
| Figure 3c | `loss_weight_sweep.json` |
| Supplementary Figure 1 (threshold curves) | `threshold_curves.csv` |
| Supplementary Table 1 (regression controls) | `stats.json`, `tables.md` |
| Supplementary Table 2 (pooling) | `pooling_arms.json` |
| Supplementary Table 3 (augmentation) | `augmentation.json`, `augmentation_eval_split.json` |
| Supplementary Table 4 (output antisymmetrization) | `inversion_averaging.json` |

## Execution manifest

[`run_manifest.json`](run_manifest.json) lists all 195 training runs: the 84-run matched-pair grid and the
pooling, augmentation, loss-weight and zero-injection studies. Each entry names its config file with
SHA-256, the dataset and split manifests it read, and its install profile. The manifest also records
the locked package versions, the GPU used per core, and compute totals. Rebuild it with
`uv run equiparity manifest`; a test fails if the committed copy is stale.

## All records

| Record | Contents | Manuscript items | Rebuilt by |
|---|---|---|---|
| `parity_gap_table.csv` | Parity gap per centrosymmetric class at rank 1 and rank 3 (strain-symmetric piezoelectric space) | Table 1 | `equiparity verify --theory` |
| `distribution_percentiles.json` | Percentiles, maxima and false-flag fraction of predicted piezoelectric norms on idealized inputs, per arm | Figure 1c |  |
| `stats.json` | Test error of both arms per core and target with seed spread and paired test; false-flag fraction, violation median, structure-level Wilcoxon and bootstrap CIs per arm and coordinate variant | Figure 1c; Figure 2a; Figure 3a; Supplementary Table 1 (regression controls); Supplementary Note: Population, thresholds, and readout robustness | `equiparity aggregate` |
| `rotation_subgroup.json` | False-flag fraction resolved by point-group family | Figure 2a; Supplementary Note: Population, thresholds, and readout robustness (point-group validation) |  |
| `symmetry_breaking.csv` | Predicted tensor norm against polar distortion amplitude, per material, core, arm and seed | Figure 2b |  |
| `symmetry_breaking.json` | Distortion-path measurements with symmetry verification of each path point | Figure 2b |  |
| `zero_injection_curve.json` | False-flag fraction against the number of injected zero-labelled crystals | Figure 3b; Supplementary Note: Training interventions and output enforcement |  |
| `zero_injection_sets.json` | Zero-labelled set definitions for the injection curve | Figure 3b |  |
| `loss_weight_sweep.json` | False-flag fraction against zero-row loss weight on trained, seen and unseen crystals | Figure 3c; Supplementary Note: Training interventions and output enforcement |  |
| `threshold_curves.csv` | False-flag fraction at 25 log-spaced thresholds per arm and coordinate variant | Supplementary Figure 1 (threshold curves) | `equiparity aggregate` |
| `pooling_arms.json` | Per-seed summed and mean-pooled readout arms | Supplementary Table 2 (pooling) |  |
| `augmentation.json` | Per-seed seen and unseen false-flag fractions for the augmentation arms, with the trained-zero control | Supplementary Table 3 (augmentation) |  |
| `augmentation_eval_split.json` | Seen/unseen index partition of the evaluation population and its space groups | Supplementary Table 3 (augmentation) |  |
| `inversion_averaging.json` | False-flag fraction and violation before and after output inversion antisymmetrization, with a non-centrosymmetric control | Supplementary Table 4 (output antisymmetrization) |  |
| `appendix_stats.json` | Parameter counts and capacity ratios, target calibration, O(3) floor, size dependence, compute | Supplementary Note: Model construction, regression controls, and output audit (compute and environment); Methods (reproducibility) | `equiparity aggregate` |
| `run_manifest.json` | Execution manifest: every training run with config and SHA-256, data manifests, install profile, locked package versions, hardware and compute | Methods (reproducibility) | `equiparity manifest` |
| `tables.md` | Rendered accuracy, violation and timing tables | Supplementary Table 1 (regression controls) | `equiparity aggregate` |
| `tables_extra.md` | Rendered capacity and full accuracy tables | Supplementary Note: Model construction, regression controls, and output audit | `equiparity aggregate` |
| `ood_spacegroups.json` | Space group, symbol, crystal family and atom count of each of the 2,000 evaluation crystals | Methods (data and splits) |  |
| `ood_symmetry.json` | Centrosymmetry counts and selection tolerance of the evaluation population | Methods (data and splits); Supplementary Note: Population, thresholds, and readout robustness (coordinate tolerance) |  |
| `split_contamination.json` | Train/evaluation overlap check | Methods (data and splits) |  |
| `size_consistency.json` | Supercell scaling of the summed readout | Supplementary Note: Population, thresholds, and readout robustness (readout scaling) |  |
| `equiformer_v2_upstream.json` | Vendored-against-upstream reconstruction of the EquiformerV2 source | Supplementary Note: Model construction, regression controls, and output audit (EquiformerV2 provenance) |  |
