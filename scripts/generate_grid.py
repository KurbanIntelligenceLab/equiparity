"""Expand the {core x parity x target x seed} ablation grid into per-run YAML configs.

Writes configs/grid/*.yaml plus two run-lists (nequip_runs.txt, mace_runs.txt) grouping runs by
docker image/profile. Cores and their valid parity arms:
  nequip/allegro/mace : O(3) and SO(3)      (toggleable, e3nn)
  equiformer_v2       : SO(3) only          (fixed SOTA)
  clifford_stf        : O(3) only, float64  (fixed geometric-algebra)
"""

from __future__ import annotations

from pathlib import Path

# Four-core scope (CliffordSTF dropped from the paper — ill-conditioned cubic head; see
# archive_clifford/ + docs). 7 arms x 4 targets x 3 seeds = 84 runs.
CORE_PARITY = {
    "nequip": ["o3", "so3"],
    "allegro": ["o3", "so3"],
    "mace": ["o3", "so3"],
    "equiformer_v2": ["so3"],
}
PROFILE = {  # which docker image runs each core
    "nequip": "nequip", "allegro": "nequip", "equiformer_v2": "nequip",
    "mace": "mace",
}
SEEDS = [0, 1, 2]
TARGET = {
    "U0": dict(dataset="qm9", npz="data/raw/qm9/qm9_processed.npz",
              split="data/splits/qm9_split.npz", lmax=2, epochs=100, batch=32, max_train=25000),
    "dipole": dict(dataset="qm9", npz="data/raw/qm9/qm9_processed.npz",
                  split="data/splits/qm9_split.npz", lmax=2, epochs=100, batch=32, max_train=25000),
    "elastic": dict(dataset="mp_elastic", npz="data/raw/mp/mp_elastic_processed.npz",
                   split="data/splits/mp_elastic_split.npz", lmax=2, epochs=120, batch=16,
                   max_train=None),
    "piezoelectric": dict(dataset="mp_piezoelectric",
                         npz="data/raw/mp/mp_piezoelectric_processed.npz",
                         split="data/splits/mp_piezoelectric_split.npz", lmax=3, epochs=150,
                         batch=16, max_train=None),
}
# moderate production size (headline verified at 128; 64 keeps the 96-run grid tractable)
FEATURES = 64


def _config_yaml(core: str, parity: str, target: str, seed: int) -> str:
    t = TARGET[target]
    precision = "float64" if core == "clifford_stf" else "float32"
    lines = [
        f"seed: {seed}", f"core: {core}", f"parity: {parity}", f"target: {target}",
        f"dataset: {t['dataset']}", f"processed_npz: {t['npz']}", f"split_npz: {t['split']}",
        "output_dir: outputs",
        "model:", "  num_layers: 3", f"  l_max: {t['lmax']}", f"  num_features: {FEATURES}",
        "  r_max: 5.0",
        "training:", f"  batch_size: {t['batch']}", f"  epochs: {t['epochs']}", "  lr: 0.002",
        "  device: cuda", f"  precision: {precision}",
    ]
    if t["max_train"] is not None:
        lines += [f"  max_train_samples: {t['max_train']}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    out = Path("configs/grid")
    out.mkdir(parents=True, exist_ok=True)
    runs = {"nequip": [], "mace": []}
    total = 0
    for core, parities in CORE_PARITY.items():
        for parity in parities:
            for target in TARGET:
                for seed in SEEDS:
                    name = f"{core}_{target}_{parity}_seed{seed}.yaml"
                    (out / name).write_text(_config_yaml(core, parity, target, seed))
                    runs[PROFILE[core]].append(f"configs/grid/{name}")
                    total += 1
    for profile, paths in runs.items():
        (out / f"{profile}_runs.txt").write_text("\n".join(paths) + "\n")
        print(f"{profile}: {len(paths)} runs")
    print(f"total: {total} configs written to {out}/")


if __name__ == "__main__":
    main()
