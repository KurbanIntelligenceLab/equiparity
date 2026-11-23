# The parity gap in crystal tensor prediction

Can Polat<sup>1</sup>, Mustafa Kurban<sup>2,3</sup>, Erchin Serpedin<sup>1</sup>, and Hasan Kurban<sup>4</sup>

1. Department of Electrical and Computer Engineering, Texas A&M University, College Station, Texas, USA
2. Department of Electrical and Computer Engineering, Texas A&M University at Qatar, Doha, Qatar
3. Department of Prosthetics and Orthotics, Ankara University, Ankara, Turkey
4. College of Science and Engineering, Hamad Bin Khalifa University, Doha, Qatar

Correspondence to [Mustafa Kurban](mailto:kurbanm@ankara.edu.tr) or [Hasan Kurban](mailto:hkurban@hbku.edu.qa).

## Abstract

A centrosymmetric crystal has no linear piezoelectric response to strain. Can a model that respects rotations recover this zero without enforcing inversion? We define the *parity gap* as the number of independent tensor components that a crystal's proper rotations allow and inversion removes. For piezoelectricity, the gap is positive in ten of the eleven centrosymmetric crystal classes. We tested matched NequIP, Allegro and MACE predictors. Both symmetry variants outperformed constant predictors on held-out piezoelectric and elastic tensors. On a separate set of 2,000 idealized centrosymmetric crystals, none of the nine primary inversion-equivariant models predicted a piezoelectric norm above 0.01 C/m². The nine rotation-only models exceeded this threshold on 58.05–82.45% of crystals, yet all stayed below threshold in cubic class m3̅m, where rotations suffice. Giving zero-labelled examples greater weight during training reduced violations on unseen crystals. Where the gap is positive, internal parity, inversion antisymmetrization or projection onto the known crystal group's invariant tensor space can enforce the physical zero by construction.

## Reproduce the reported measurements

Start in the repository root with Python 3.12 and `uv`. The CPU checks recompute the parity-gap dimensions and compare the released measurements with the underlying predictions. The [numerical source guide](results/README.md) maps each figure and table to its data.

```sh
uv sync --locked --extra verify
uv run --no-sync equiparity verify --json
uv run --no-sync equiparity measure-masks --json
```

The [formal proofs](proofs/README.md) can also be checked with `uv run --no-sync equiparity verify --proofs` after installing Lean.

## Train a crystal model

The example trains a NequIP piezoelectric model with inversion symmetry and tensor Huber loss on a CUDA GPU. Run from a clean Git checkout. NequIP and Allegro share the `nequip` profile. For MACE, use the `mace` extra in a separate environment with a MACE configuration.

```sh
uv sync --locked --extra nequip --extra data --extra verify
config=configs/crystal/tensor_huber_nequip_piezoelectric_o3_seed0.yaml
uv run --no-sync equiparity run "$config"
```

The training log prints the run directory, which contains the configuration, learning history and checkpoint selected by validation MAE. The [run manifest](results/run_manifest.json) identifies the configurations and data splits for all 108 training runs.

## Evaluate a trained checkpoint

Set `run_dir` to the directory printed by training. Use the same environment and configuration as the training run. The first command fixes the configuration and dataset hashes before the test set is evaluated.

```sh
run_dir='outputs/crystal/EXPERIMENT_ID'
uv run --no-sync equiparity lock-evaluation "$config" --output "$run_dir/protocol.json"
uv run --no-sync equiparity evaluate-checkpoint "$run_dir" --partition test \
  --protocol-lock "$run_dir/protocol.json" --output-dir "$run_dir/heldout"
uv run --no-sync equiparity audit-crystal-symmetry "$run_dir" --output-dir "$run_dir/symmetry"
```

The held-out evaluation writes predicted and reference tensor coefficients to `heldout/test_predictions.npz` and reports errors against the constant predictors in `heldout/test_diagnostics.json`. For piezoelectric checkpoints, the symmetry audit writes per-crystal norms for random and trained weights to `symmetry/parity_vectors.npz` and class-level flag rates to `symmetry/parity_diagnostics.json`. Both evaluation commands accept `--device cpu`. Use a fresh output directory for each evaluation.

## Data and controls

The crystal runs use the [processed structures and tensors](data/raw/mp/), [property-associated source documents](data/property_sources/) and [fixed splits](data/splits/). The [mean-pooled crystal models](configs/crystal/) and [zero-label controls](configs/zero_labels/) use validation-selected checkpoints. The [sum-pooled piezoelectric](configs/sum_pooled/) and [QM9 molecular](configs/molecular/) controls use final-epoch weights.

For QM9, extract the molecular structures under `data/raw/qm9/xyz/` and place `uncharacterized.txt` in `data/raw/qm9/`, then run `uv run --no-sync equiparity data prepare qm9`. The [QM9 manifest](data/manifests/qm9.yaml) identifies the source files and units.
