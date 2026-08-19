# Result records

Every quantity the study reports is derived from a file in this directory, and each record below
names the script that produced it.

```bash
uv run python verification/verify_claims.py
```

## Matched-pair grid

Aggregated over seeds unless the record says otherwise.

| Record | Produced by | Contents |
|---|---|---|
| `stats.json` | `scripts/analysis/aggregate_grid.py` | test error of both arms per core and target with seed spread and paired test; false-flag fraction and violation median per core, arm and coordinate variant |
| `appendix_stats.json` | `scripts/analysis/aggregate_grid.py` | parameter counts and SO(3)/O(3) capacity ratios, target calibration, O(3) floor, size dependence, compute |
| `threshold_curves.csv` | `scripts/analysis/aggregate_grid.py` | false-flag fraction at 25 log-spaced thresholds, per arm and coordinate variant |
| `tables.md`, `tables_extra.md` | `scripts/analysis/aggregate_grid.py` | rendered tables of the above |
| `augmentation.json` | `scripts/experiments/augmentation.py` | per-seed seen and unseen false-flag fractions for the augmentation arms |
| `augmentation_eval_split.json` | `scripts/data/prepare_augmented_piezoelectric.py` | the seen/unseen index partition of the evaluation population and its space groups |
| `loss_weight_sweep.json` | `scripts/experiments/loss_weight_sweep.py` | false-flag fraction against zero-target loss weight |
| `pooling_arms.json` | `scripts/grids/generate_grid_meanpool.py` | per-seed summed and mean-pooled readout arms |
| `zero_injection_sets.json` | `scripts/data/prepare_zero_injection_sets.py` | augmentation set definitions for the learning curve |
| `zero_injection_curve.json` | `scripts/experiments/zero_injection_curve.py` | false-flag fraction against training-set size |
| `epoch_curve.json` | `scripts/experiments/epoch_curve.py` | false-flag fraction against epoch |
| `best_vs_final.json` | `scripts/experiments/best_vs_final.py` | best-checkpoint against final-epoch comparison |
| `fnorm_ewt.json` | `scripts/analysis/fnorm_ewt.py` | Frobenius-norm and element-wise error metrics |
| `dipole_bias.json` | `scripts/analysis/dipole_bias.py` | dipole-target bias measurement |

## Evaluation population

| Record | Produced by | Contents |
|---|---|---|
| `ood_spacegroups.json` | `scripts/data/ood_spacegroups.py` | space group, international symbol, crystal family and atom count for each of the 2,000 evaluation crystals |
| `ood_symmetry.json` | `scripts/analysis/ood_symmetry.py` | centrosymmetry counts and selection tolerance |
| `rotation_subgroup.json` | `scripts/experiments/rotation_subgroup.py` | false-flag fraction resolved by point-group family |
| `qm9_pointgroups.json` | `scripts/data/qm9_pointgroups.py` | point-group assignment across QM9 |
| `split_contamination.json` | `scripts/experiments/split_contamination.py` | train/evaluation overlap check |
| `prevalence_audit.json` | `scripts/experiments/prevalence_audit.py` | the released-architecture audit: parity class per model, with the evidence for each |

## Measurements on released models

Per-structure vectors over the full 2,000-crystal evaluation population, so a reported fraction
can be recomputed without retraining.

| Record | Produced by | Contents |
|---|---|---|
| `tensor_predictors.json`, `tensor_predictors/` | `scripts/experiments/tensor_predictors.py` | the two dedicated crystal-tensor predictors in each of four mask and coordinate conditions: predicted rank-3 tensor (`*_tensors.npy`, 2000x3x6 Voigt) and its Frobenius norm (`*.npy`, 2000) |
| `random_init.json`, `random_init/` | `scripts/experiments/random_init_probe.py` | three rotation-only potentials at random initialization: violation magnitude (2000) each |
| `frozen_backbone.json`, `frozen_backbone/` | `scripts/experiments/frozen_backbone.py` | two frozen backbones with a fitted head, three seeds each: violation magnitude (2000) per seed |
| `frozen_backbone_distortion.json`, `frozen_backbone_distortion.csv` | `scripts/experiments/frozen_backbone_distortion.py` | tensor norm against distortion amplitude for the frozen-backbone heads |

## Mechanism and controls

| Record | Produced by | Contents |
|---|---|---|
| `symmetry_breaking.json`, `symmetry_breaking.csv` | `scripts/experiments/symmetry_breaking.py` | tensor norm against distortion amplitude, per material, core, arm and seed |
| `jacobian.json` | `scripts/experiments/jacobian.py` | even-subspace fraction of the input-output Jacobian |
| `inversion_averaging.json` | `scripts/experiments/inversion_averaging.py` | effect of averaging a prediction with its inverted image |
| `output_parity.json` | `scripts/experiments/output_parity.py` | parity type of each core's output irreps |
| `named_materials.json`, `named_structures.json` | `scripts/experiments/named_materials.py` | predicted tensor norm for ten named centrosymmetric compounds |
| `per_atom_readout.json` | `scripts/experiments/per_atom_readout.py` | per-atom against pooled readout |
| `size_consistency.json` | `scripts/experiments/size_consistency.py` | supercell scaling of the summed readout |
| `non_e3nn_control.json` | `scripts/experiments/non_e3nn_control.py` | the non-e3nn O(3) control (HotPP): mirror law and structural zero on periodic crystals |
| `equiformer_v2_upstream.json` | `scripts/experiments/equiformer_v2_upstream_repro.py` | vendored-against-upstream reconstruction of the EquiformerV2 source |
| `inheritance_probes.json` | `scripts/experiments/inheritance_probes.py` | probe models for the inheritance argument |
| `theory_bounds.json` | `scripts/experiments/theory_bounds.py` | numerical bounds accompanying the theorems |
