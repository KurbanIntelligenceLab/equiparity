"""Expand the mean-pooled control-arm grid: {core x parity x seed} on piezoelectric + elastic.

Companion to ``scripts/generate_grid.py`` (same ``CORE_PARITY``/``PROFILE``/``SEEDS``/``TARGET``
tables, same YAML shape and the same ``ModelHyperparams``/``io/config.py`` loading path -- no
parallel mechanism). Differences from the headline grid, both deliberate:

1. Restricted to ``piezoelectric`` and ``elastic`` -- the two intensive-property targets the
   reviewer's pooling critique concerns (U0 and dipole are molecular/QM9 targets with no
   supercell notion, so a pooling control there is not meaningful).
2. Every config gets ``model.pooling: mean`` (the task's opt-in intensive-readout arm; the
   default stays ``sum`` for every existing config that does NOT go through this generator).

Target normalization needs no special handling here: ``target_scale`` is left unset (None) in
every generated config, so the trainer's ``scale = config.training.target_scale or
float(train_targets.std())`` (nequip_tensor.py / mace_tensor.py / equiformer_tensor.py) refits
automatically from the training split under the run's own pooling -- see
results/f5_pooling_arms.json for the measured per-seed values; refitting rather than reusing the
sum-pooled scale is required because the mean-pooled target distribution differs.

    python scripts/generate_grid_meanpool.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_grid import CORE_PARITY, FEATURES, PROFILE, SEEDS, TARGET

MEANPOOL_TARGETS = ("elastic", "piezoelectric")


def _config_yaml(core: str, parity: str, target: str, seed: int) -> str:
    t = TARGET[target]
    precision = "float64" if core == "clifford_stf" else "float32"
    lines = [
        f"seed: {seed}",
        f"core: {core}",
        f"parity: {parity}",
        f"target: {target}",
        f"dataset: {t['dataset']}",
        f"processed_npz: {t['npz']}",
        f"split_npz: {t['split']}",
        "output_dir: outputs",
        "model:",
        "  num_layers: 3",
        f"  l_max: {t['lmax']}",
        f"  num_features: {FEATURES}",
        "  r_max: 5.0",
        "  pooling: mean",  # the only line that differs from generate_grid.py's template
        "training:",
        f"  batch_size: {t['batch']}",
        f"  epochs: {t['epochs']}",
        "  lr: 0.002",
        "  device: cuda",
        f"  precision: {precision}",
    ]
    if t["max_train"] is not None:
        lines += [f"  max_train_samples: {t['max_train']}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    out = Path("configs/grid_meanpool")
    out.mkdir(parents=True, exist_ok=True)
    runs = {"nequip": [], "mace": []}
    total = 0
    for core, parities in CORE_PARITY.items():
        for parity in parities:
            for target in MEANPOOL_TARGETS:
                for seed in SEEDS:
                    name = f"{core}_{target}_{parity}_seed{seed}_mean.yaml"
                    (out / name).write_text(_config_yaml(core, parity, target, seed))
                    runs[PROFILE[core]].append(f"configs/grid_meanpool/{name}")
                    total += 1
    for profile, paths in runs.items():
        (out / f"{profile}_runs.txt").write_text("\n".join(paths) + "\n")
        print(f"{profile}: {len(paths)} runs")
    print(f"total: {total} configs written to {out}/")


if __name__ == "__main__":
    main()
