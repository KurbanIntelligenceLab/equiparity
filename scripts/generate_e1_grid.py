"""Expand the E1 augmentation-rebuttal grid: SO(3) arms only, 4 cores x 3 seeds = 12 runs.

Identical hyperparameters to the headline piezoelectric runs. Two deliberate differences:

1. ``dataset`` / ``processed_npz`` / ``split_npz`` point at the augmented set (real piezoelectric
   tensors + 1,000 centrosymmetric crystals labelled exactly zero).
2. ``training.target_scale`` is **frozen** to the un-augmented value. Adding zero rows shrinks the
   recomputed std from 0.749134 to 0.638289; freezing it keeps augmented violation magnitudes
   directly comparable to the main table.

O(3) arms are deliberately NOT retrained: their zeros are structural and hold for any weights, so
the existing runs already answer the question. Say exactly that in the paper.
"""

from __future__ import annotations

from pathlib import Path

CORES = ["nequip", "allegro", "equiformer_v2", "mace"]
PROFILE = {"nequip": "nequip", "allegro": "nequip", "equiformer_v2": "nequip", "mace": "mace"}
SEEDS = [0, 1, 2]
FEATURES = 64
FROZEN_TARGET_SCALE = 0.749134


def _config_yaml(core: str, seed: int) -> str:
    return (
        "\n".join(
            [
                f"seed: {seed}",
                f"core: {core}",
                "parity: so3",
                "target: piezoelectric",
                "dataset: mp_piezoelectric_augmented",
                "processed_npz: data/raw/mp/mp_piezoelectric_augmented_processed.npz",
                "split_npz: data/splits/mp_piezoelectric_augmented_split.npz",
                "output_dir: outputs",
                "model:",
                "  num_layers: 3",
                "  l_max: 3",
                f"  num_features: {FEATURES}",
                "  r_max: 5.0",
                "training:",
                "  batch_size: 16",
                "  epochs: 150",
                "  lr: 0.002",
                "  device: cuda",
                "  precision: float32",
                f"  target_scale: {FROZEN_TARGET_SCALE}",
            ]
        )
        + "\n"
    )


def main() -> None:
    out = Path("configs/e1")
    out.mkdir(parents=True, exist_ok=True)
    runs: dict[str, list[str]] = {"nequip": [], "mace": []}
    for core in CORES:
        for seed in SEEDS:
            name = f"{core}_piezoelectric_so3_aug_seed{seed}.yaml"
            (out / name).write_text(_config_yaml(core, seed))
            runs[PROFILE[core]].append(f"configs/e1/{name}")
    for profile, paths in runs.items():
        (out / f"e1_{profile}_runs.txt").write_text("\n".join(paths) + "\n")
        print(f"{profile}: {len(paths)} runs")
    print(f"total: {sum(len(v) for v in runs.values())} configs written to {out}/")


if __name__ == "__main__":
    main()
