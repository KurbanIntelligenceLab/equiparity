# Experiments

One row per experiment: the question it answers, the script that runs it, and the record it
writes under `results/`. Every record also names its producer in
[`results/README.md`](../results/README.md), so a claim traces to a record and a record traces to
code.

Scripts marked **external** need model checkpoints this repository does not redistribute; see
[EXTERNAL_MODELS.md](EXTERNAL_MODELS.md). Everything else runs from what is committed here, given
the datasets rebuilt by `scripts/data/`.

## The matched-pair grid

The core ablation: four architectures x two parity modes x four targets x three seeds, 84 runs.

| Question | Script | Configs | Record |
|---|---|---|---|
| How do the two arms compare on accuracy and on forbidden predictions? | `analysis/aggregate_grid.py` | `configs/grid/` | `stats.json`, `appendix_stats.json`, `threshold_curves.csv`, `tables.md`, `tables_extra.md` |
| Frobenius-norm and element-wise error on the piezoelectric test set | `analysis/fnorm_ewt.py` | `configs/grid/` | `fnorm_ewt.json` |
| Does the point-charge dipole target understate the DFT dipole? | `analysis/dipole_bias.py` | — | `dipole_bias.json` |

## Can the gap be trained away?

| Question | Script | Configs | Record |
|---|---|---|---|
| Does training SO(3) on zero-labelled crystals fix the false flags? | `experiments/augmentation.py` | `configs/augmentation/` | `augmentation.json`, `augmentation_eval_split.json` |
| How many exact zeros must SO(3) see before it complies? | `experiments/zero_injection_curve.py` | `configs/zero_injection/` | `zero_injection_curve.json`, `zero_injection_sets.json` |
| Does up-weighting the zero rows a hundredfold force compliance? | `experiments/loss_weight_sweep.py` | `configs/loss_weight/` | `loss_weight_sweep.json` |
| Does the violation shrink with training epochs? | `experiments/epoch_curve.py` | `configs/epoch_curve/` | `epoch_curve.json` |
| Would early stopping have hidden it? | `experiments/best_vs_final.py` | `configs/epoch_curve/` | `best_vs_final.json` |
| Does symmetrising the output at test time help? | `experiments/inversion_averaging.py` | — | `inversion_averaging.json` |

## Mechanism

| Question | Script | Configs | Record |
|---|---|---|---|
| Does the guarantee switch off exactly when the symmetry does? | `experiments/symmetry_breaking.py` | `configs/grid/` | `symmetry_breaking.json`, `symmetry_breaking.csv` |
| Is the guarantee visible in the input-output Jacobian? | `experiments/jacobian.py` | `configs/grid/` | `jacobian.json` |
| What parity do the trained models' output irreps actually carry? | `experiments/output_parity.py` | `configs/grid/` | `output_parity.json` |
| What does each model predict for familiar centrosymmetric compounds? | `experiments/named_materials.py` | `configs/grid/` | `named_materials.json`, `named_structures.json` |
| Are SO(3)'s correct zeros the ones rotations alone already forbid? | `experiments/rotation_subgroup.py` | `configs/grid/` | `rotation_subgroup.json` |
| Numerical bounds accompanying the theorems | `experiments/theory_bounds.py` | `configs/grid/` | `theory_bounds.json` |

## Controls

Each rules out an alternative explanation for the separation.

| What it rules out | Script | Configs | Record |
|---|---|---|---|
| Extensive pooling, not parity, drives the result | `experiments/per_atom_readout.py` | `configs/grid/` | `per_atom_readout.json` |
| Summed readouts disagree between a cell and its supercell | `experiments/size_consistency.py`, `experiments/size_consistency_mace.py` | — | `size_consistency.json` |
| Mean pooling would change the conclusion | `grids/generate_grid_meanpool.py` | `configs/grid_meanpool/`, `configs/grid_sumpool/` | `pooling_arms.json` |
| The effect is an e3nn artifact | `experiments/non_e3nn_control.py` | — | `non_e3nn_control.json` |
| Evaluation crystals leaked into training | `experiments/split_contamination.py` | `configs/grid/` | `split_contamination.json` |
| A QM9 molecule's point group forces its dipole to vanish | `data/qm9_pointgroups.py` | — | `qm9_pointgroups.json` |
| The evaluation set is not really centrosymmetric | `analysis/ood_symmetry.py`, `data/ood_spacegroups.py` | — | `ood_symmetry.json`, `ood_spacegroups.json` |

## Released models

Whether the design bit is present in models other people ship, and whether it is inherited.

| Question | Script | Record |
|---|---|---|
| Which released architectures carry parity labels on their features? | `experiments/prevalence_audit.py` | `prevalence_audit.json` |
| Do the undetermined audit rows hold up under measurement? | `experiments/inheritance_probes.py`, `experiments/vector_only_probes.py` | `inheritance_probes.json` |
| Do dedicated crystal-tensor predictors satisfy the zeros? (**external**) | `experiments/tensor_predictors.py` | `tensor_predictors.json`, `tensor_predictors/` |
| Do rotation-only potentials violate at random initialization? (**external**) | `experiments/random_init_probe.py` | `random_init.json`, `random_init/` |
| Does a head on a frozen universal potential inherit the bit? (**external**) | `experiments/frozen_backbone.py`, `experiments/cache_esen_features.py`, `experiments/cache_mace_features.py` | `frozen_backbone.json`, `frozen_backbone/` |
| Does the inherited behaviour hold along other distortion paths? (**external**) | `experiments/frozen_backbone_distortion.py` | `frozen_backbone_distortion.json`, `frozen_backbone_distortion.csv` |
| Is the vendored EquiformerV2 faithful to upstream? | `experiments/equiformer_v2_upstream_{build,repro,report}.py` | `equiformer_v2_upstream.json` |

## Gates

Not experiments: checks that must pass before anything trains or is believed.

| Check | Command |
|---|---|
| Parity verification gate, both arms of every core | `uv run --extra nequip pytest tests/verification -q` |
| The same gate as a standalone report | `uv run --extra nequip python scripts/experiments/parity_audit.py` |
| Does a given NequIP construction really break parity? | `uv run --extra nequip python scripts/experiments/parity_toggle_probe.py` |
| Theorems and corollaries, numerically | `uv run python verification/verify_theory.py` |
| Every reported quantity, against `results/` | `uv run python verification/verify_claims.py` |

## Re-running a driver

Drivers that aggregate over training runs read the raw run tree, which is not part of this
release. It defaults to `runs/` at the repository root; set `$PARITY_RUNS` to point elsewhere.
Nothing under `verification/` needs it, and neither do the released-model measurements above.
