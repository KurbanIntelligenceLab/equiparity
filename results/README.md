# Result records

Every quantity the study reports is derived from a file in this directory. Each record names
the script that produced it, so a claim traces to a record and a record traces to code.

`verification/verify_claims.py` re-derives every reported quantity from these files and asserts
it against the published value; `python verification/verify_claims.py` reports 184 passing
checks against this tree and needs no GPU and no data beyond what is here.

## Matched-pair grid

Aggregated over seeds unless the record says otherwise.

| Record | Produced by | Contents |
|---|---|---|
| `stats.json` | `analyze_results.py` | test error of both arms per core and target with seed spread and paired test; false-flag fraction and violation median per core, arm and coordinate variant |
| `appendix_stats.json` | `analyze_results.py` | parameter counts and SO(3)/O(3) capacity ratios, target calibration, O(3) floor, size dependence, compute |
| `threshold_curves.csv` | `analyze_results.py` | false-flag fraction at 25 log-spaced thresholds, per arm and coordinate variant |
| `tables.md`, `tables_extra.md` | `analyze_results.py` | rendered tables of the above |
| `e1_augmentation.json` | `e1_augmentation.py` | per-seed seen and unseen false-flag fractions for the augmentation arms |
| `e1_eval_split.json` | `prepare_e1_augmented.py` | the seen/unseen index partition of the evaluation population and its space groups |
| `h3_loss_weight.json` | `h3_loss_weight.py` | false-flag fraction against zero-target loss weight |
| `f5_pooling_arms.json` | `generate_grid_meanpool.py` | per-seed summed and mean-pooled readout arms |
| `t3_augmentation_sets.json` | `prepare_t3_augmented.py` | augmentation set definitions for the learning curve |
| `t3_learning_curve.json` | `t3_learning_curve.py` | false-flag fraction against training-set size |
| `h1_epoch_curve.json` | `h1_epoch_curve.py` | false-flag fraction against epoch |
| `h1_best_vs_final.json` | `h1_best_vs_final.py` | best-checkpoint against final-epoch comparison |
| `fnorm_ewt.json` | `eval_fnorm.py` | Frobenius-norm and element-wise error metrics |
| `dipole_bias.json` | `measure_dipole_bias.py` | dipole-target bias measurement |

## Evaluation population

| Record | Produced by | Contents |
|---|---|---|
| `ood_spacegroups.json` | `ood_spacegroups.py` | space group, international symbol, crystal family and atom count for each of the 2,000 evaluation crystals |
| `ood_symmetry.json` | `analyze_ood_symmetry.py` | centrosymmetry counts and selection tolerance |
| `e7_rotation_subgroup.json` | `e7_rotation_subgroup.py` | false-flag fraction resolved by point-group family |
| `qm9_pointgroups.json` | `qm9_pointgroups.py` | point-group assignment across QM9 |
| `t0_contamination.json` | `t0_contamination.py` | train/evaluation overlap check |
| `prevalence_audit.json` | `prevalence_audit.py` | the released-architecture audit: parity class per model, with the evidence for each |

## Measurements on released models

Per-structure vectors over the full 2,000-crystal evaluation population, so a reported fraction
can be recomputed without retraining.

| Record | Produced by | Contents |
|---|---|---|
| `t1_sota_predictors.json`, `t1/` | `t1_sota_eval.py` | the two dedicated crystal-tensor predictors in each of four mask and coordinate conditions: predicted rank-3 tensor (`*_tensors.npy`, 2000x3x6 Voigt) and its Frobenius norm (`*.npy`, 2000) |
| `t2_random_init.json`, `t2/` | `t2_backbone_probe.py` | three rotation-only potentials at random initialization: violation magnitude (2000) each |
| `t4_frozen_backbone.json`, `t4/` | `t4_frozen_backbone.py` | two frozen backbones with a fitted head, three seeds each: violation magnitude (2000) per seed |
| `t4_distortion_materials.json`, `t4_distortion_materials.csv` | `t4_distortion_materials.py` | tensor norm against distortion amplitude for the frozen-backbone heads |

## Mechanism and controls

| Record | Produced by | Contents |
|---|---|---|
| `e2_symmetry_breaking.json`, `e2_symmetry_breaking.csv` | `e2_symmetry_breaking.py` | tensor norm against distortion amplitude, per material, core, arm and seed |
| `e3_jacobian.json` | `e3_jacobian.py` | even-subspace fraction of the input-output Jacobian |
| `e4_inversion_averaging.json` | `e4_inversion_averaging.py` | effect of averaging a prediction with its inverted image |
| `e5_output_parity.json` | `e5_output_parity.py` | parity type of each core's output irreps |
| `e6_named_materials.json`, `e6_named_structures.json` | `e6_named_materials.py` | predicted tensor norm for ten named centrosymmetric compounds |
| `f2_per_atom.json` | `f2_per_atom.py` | per-atom against pooled readout |
| `f3_size_consistency.json` | `f3_size_consistency.py` | supercell scaling of the summed readout |
| `f4_noneN3_control.json` | `f4_noneN3_control.py` | the non-e3nn O(3) control (HotPP): mirror law and structural zero on periodic crystals |
| `h1_upstream.json` | `h1_upstream_repro.py` | vendored-against-upstream reconstruction of the EquiformerV2 source |
| `h2_probes.json` | `h2_probe_models.py` | probe models for the inheritance argument |
| `theory_bounds.json` | `h4_theory_bounds.py` | numerical bounds accompanying the theorems |
