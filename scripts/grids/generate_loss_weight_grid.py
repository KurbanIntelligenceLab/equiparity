"""Expand the loss-weight sweep: NequIP SO(3) on the augmented piezoelectric set.

Six runs: zero-row loss weights {10, 100} x 3 seeds (W=1 reuses the committed augmentation runs).
Each config points at the augmented npz/split (unchanged) but carries a distinct ``dataset`` label
``mp_piezoelectric_augmented_w{W}`` so its ``run_key`` is unique and never shadows the
augmentation study runs. ``target_scale`` is frozen at 0.749134, exactly as the augmentation
study, so W=1 would reproduce the augmentation study run and the weighted runs stay on the same
scale.

    uv run --extra nequip python scripts/grids/generate_loss_weight_grid.py
"""

from __future__ import annotations

from pathlib import Path

WEIGHTS = [10, 100]
SEEDS = [0, 1, 2]
FROZEN_TARGET_SCALE = 0.749134


def _config_yaml(seed: int, weight: int) -> str:
    return (
        "\n".join(
            [
                f"seed: {seed}",
                "core: nequip",
                "parity: so3",
                "target: piezoelectric",
                f"dataset: mp_piezoelectric_augmented_w{weight}",
                "processed_npz: data/raw/mp/mp_piezoelectric_augmented_processed.npz",
                "split_npz: data/splits/mp_piezoelectric_augmented_split.npz",
                "output_dir: outputs",
                "model:",
                "  num_layers: 3",
                "  l_max: 3",
                "  num_features: 64",
                "  r_max: 5.0",
                "training:",
                "  batch_size: 16",
                "  epochs: 150",
                "  lr: 0.002",
                "  device: cuda",
                "  precision: float32",
                f"  target_scale: {FROZEN_TARGET_SCALE}",
                f"  zero_row_loss_weight: {weight}",
            ]
        )
        + "\n"
    )


def main() -> None:
    out = Path("configs/loss_weight")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for weight in WEIGHTS:
        for seed in SEEDS:
            name = f"nequip_so3_piezoelectric_aug_w{weight}_seed{seed}.yaml"
            (out / name).write_text(_config_yaml(seed, weight))
            paths.append(f"configs/loss_weight/{name}")
    (out / "runs.txt").write_text("\n".join(paths) + "\n")
    print(f"wrote {len(paths)} configs to {out}/")


if __name__ == "__main__":
    main()
