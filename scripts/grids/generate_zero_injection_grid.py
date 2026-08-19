"""Expand the zero-injection grid: NequIP SO(3) x 3 seeds x N_zero values = 9 runs.

Identical hyperparameters to the augmented runs (which supply the N = 1000
point; the headline SO(3) runs supply N = 0). Each N gets a distinct ``dataset`` label so run keys
can never collide with the headline or the augmentation study runs (the loss-weight sweep
precedent). ``training.target_scale`` stays frozen at the un-augmented value so violation
magnitudes remain comparable across the curve.

The N values are read from ``results/zero_injection_sets.json`` (written by
``prepare_t3_augmented.py``), so a capped-at-availability largest set is picked up automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

SEEDS = [0, 1, 2]
FEATURES = 64
FROZEN_TARGET_SCALE = 0.749134
SETS_JSON = Path("results/zero_injection_sets.json")


def _config_yaml(n: int, seed: int) -> str:
    name = f"mp_piezoelectric_augmented_n{n}"
    return (
        "\n".join(
            [
                f"seed: {seed}",
                "core: nequip",
                "parity: so3",
                "target: piezoelectric",
                f"dataset: {name}",
                f"processed_npz: data/raw/mp/{name}_processed.npz",
                f"split_npz: data/splits/{name}_split.npz",
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
    sets = json.loads(SETS_JSON.read_text())["sets"]
    n_values = sorted(int(k[1:]) for k in sets)
    out = Path("configs/zero_injection")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for n in n_values:
        for seed in SEEDS:
            name = f"nequip_piezoelectric_so3_n{n}_seed{seed}.yaml"
            (out / name).write_text(_config_yaml(n, seed))
            paths.append(f"configs/zero_injection/{name}")
    (out / "nequip_runs.txt").write_text("\n".join(paths) + "\n")
    print(f"t3: {len(paths)} configs written to {out}/ (N values {n_values})")


if __name__ == "__main__":
    main()
