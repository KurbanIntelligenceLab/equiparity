"""Expand the epoch-curve grid: the 12 SO(3) piezoelectric runs, re-instrumented, 4 cores x 3 seeds.

Identical configuration to the headline piezoelectric SO(3) runs -- same canonical npz/split, same
hyperparameters -- run under the per-epoch OOD instrumentation added to the tensor trainers
(``ood_false_flag_history`` in metrics.json). The ``dataset`` label is
``mp_piezoelectric_epochcurve`` so the run keys are suffixed and can never shadow the released
headline runs; the domain layer, `find_piezo_runs`, and `analyze_results._is_headline` all key off
it.

O(3) arms are not retrained: their zeros are structural and epoch-independent.
"""

from __future__ import annotations

from pathlib import Path

CORES = ["nequip", "allegro", "equiformer_v2", "mace"]
PROFILE = {"nequip": "nequip", "allegro": "nequip", "equiformer_v2": "nequip", "mace": "mace"}
SEEDS = [0, 1, 2]
FEATURES = 64


def _config_yaml(core: str, seed: int) -> str:
    return (
        "\n".join(
            [
                f"seed: {seed}",
                f"core: {core}",
                "parity: so3",
                "target: piezoelectric",
                "dataset: mp_piezoelectric_epochcurve",
                "processed_npz: data/raw/mp/mp_piezoelectric_processed.npz",
                "split_npz: data/splits/mp_piezoelectric_split.npz",
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
            ]
        )
        + "\n"
    )


def main() -> None:
    out = Path("configs/epoch_curve")
    out.mkdir(parents=True, exist_ok=True)
    runs: dict[str, list[str]] = {"nequip": [], "mace": []}
    for core in CORES:
        for seed in SEEDS:
            name = f"{core}_piezoelectric_so3_epochcurve_seed{seed}.yaml"
            (out / name).write_text(_config_yaml(core, seed))
            runs[PROFILE[core]].append(f"configs/epoch_curve/{name}")
    for profile, paths in runs.items():
        (out / f"h1curve_{profile}_runs.txt").write_text("\n".join(paths) + "\n")
        print(f"{profile}: {len(paths)} runs")
    print(f"total: {sum(len(v) for v in runs.values())} configs written to {out}/")


if __name__ == "__main__":
    main()
