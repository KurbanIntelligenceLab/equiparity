"""Expand the study's fixed experiment matrices into per-run YAML configs.

Each grid writes ``configs/<grid>/*.yaml`` plus run lists grouped by install profile
(``nequip_runs.txt`` for NequIP, Allegro and EquiformerV2; ``mace_runs.txt`` for MACE), since the
two profiles pin incompatible e3nn versions and run in separate environments.

CLI: ``equiparity grid generate {main,meanpool,sumpool,augmentation,loss-weight,zero-injection}``
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

# Matched-pair scope: 7 arms x 4 targets x 3 seeds = 84 runs. EquiformerV2 has no parity toggle.
CORE_PARITY: dict[str, tuple[str, ...]] = {
    "nequip": ("o3", "so3"),
    "allegro": ("o3", "so3"),
    "mace": ("o3", "so3"),
    "equiformer_v2": ("so3",),
}
PROFILE = {"nequip": "nequip", "allegro": "nequip", "equiformer_v2": "nequip", "mace": "mace"}
SEEDS = (0, 1, 2)
FEATURES = 64
# Un-augmented piezoelectric target std; frozen in every augmented run so violation magnitudes
# stay comparable to the headline grid.
FROZEN_TARGET_SCALE = 0.749134


@dataclass(frozen=True)
class TargetSpec:
    """Dataset and schedule for one target property."""

    dataset: str
    npz: str
    split: str
    lmax: int
    epochs: int
    batch: int
    max_train: int | None


_QM9 = {"dataset": "qm9", "npz": "data/raw/qm9/qm9_processed.npz"}
TARGETS: dict[str, TargetSpec] = {
    "U0": TargetSpec(
        **_QM9, split="data/splits/qm9_split.npz", lmax=2, epochs=100, batch=32, max_train=25000
    ),
    "dipole": TargetSpec(
        **_QM9, split="data/splits/qm9_split.npz", lmax=2, epochs=100, batch=32, max_train=25000
    ),
    "elastic": TargetSpec(
        dataset="mp_elastic",
        npz="data/raw/mp/mp_elastic_processed.npz",
        split="data/splits/mp_elastic_split.npz",
        lmax=2,
        epochs=120,
        batch=16,
        max_train=None,
    ),
    "piezoelectric": TargetSpec(
        dataset="mp_piezoelectric",
        npz="data/raw/mp/mp_piezoelectric_processed.npz",
        split="data/splits/mp_piezoelectric_split.npz",
        lmax=3,
        epochs=150,
        batch=16,
        max_train=None,
    ),
}


def _config_yaml(
    *,
    core: str,
    parity: str,
    target: str,
    seed: int,
    spec: TargetSpec,
    pooling: str | None = None,
    target_scale: float | None = None,
    zero_row_loss_weight: int | None = None,
) -> str:
    lines = [
        f"seed: {seed}",
        f"core: {core}",
        f"parity: {parity}",
        f"target: {target}",
        f"dataset: {spec.dataset}",
        f"processed_npz: {spec.npz}",
        f"split_npz: {spec.split}",
        "output_dir: outputs",
        "model:",
        "  num_layers: 3",
        f"  l_max: {spec.lmax}",
        f"  num_features: {FEATURES}",
        "  r_max: 5.0",
    ]
    if pooling is not None:
        lines.append(f"  pooling: {pooling}")
    lines += [
        "training:",
        f"  batch_size: {spec.batch}",
        f"  epochs: {spec.epochs}",
        "  lr: 0.002",
        "  device: cuda",
        "  precision: float32",
    ]
    if spec.max_train is not None:
        lines.append(f"  max_train_samples: {spec.max_train}")
    if target_scale is not None:
        lines.append(f"  target_scale: {target_scale}")
    if zero_row_loss_weight is not None:
        lines.append(f"  zero_row_loss_weight: {zero_row_loss_weight}")
    return "\n".join(lines) + "\n"


# A grid yields (file name, profile, yaml text).
GridRows = Iterator[tuple[str, str, str]]


def _main(_root: Path) -> GridRows:
    for core, parities in CORE_PARITY.items():
        for parity in parities:
            for target, spec in TARGETS.items():
                for seed in SEEDS:
                    text = _config_yaml(
                        core=core, parity=parity, target=target, seed=seed, spec=spec
                    )
                    yield f"{core}_{target}_{parity}_seed{seed}.yaml", PROFILE[core], text


def _pooling_rows(pooling: str) -> GridRows:
    """Dedicated pooling-control pairs on the two intensive crystal targets."""
    for core, parities in CORE_PARITY.items():
        for parity in parities:
            for target in ("elastic", "piezoelectric"):
                for seed in SEEDS:
                    text = _config_yaml(
                        core=core,
                        parity=parity,
                        target=target,
                        seed=seed,
                        spec=TARGETS[target],
                        pooling=pooling,
                    )
                    name = f"{core}_{target}_{parity}_seed{seed}_{pooling}.yaml"
                    yield name, PROFILE[core], text


def _meanpool(_root: Path) -> GridRows:
    return _pooling_rows("mean")


def _sumpool(_root: Path) -> GridRows:
    return _pooling_rows("sum")


def _augmented_spec(dataset: str, stem: str = "mp_piezoelectric_augmented") -> TargetSpec:
    base = TARGETS["piezoelectric"]
    return TargetSpec(
        dataset=dataset,
        npz=f"data/raw/mp/{stem}_processed.npz",
        split=f"data/splits/{stem}_split.npz",
        lmax=base.lmax,
        epochs=base.epochs,
        batch=base.batch,
        max_train=None,
    )


def _augmentation(_root: Path) -> GridRows:
    """SO(3) arms retrained with 1,000 zero-labelled centrosymmetric crystals added."""
    spec = _augmented_spec("mp_piezoelectric_augmented")
    for core in CORE_PARITY:
        for seed in SEEDS:
            text = _config_yaml(
                core=core,
                parity="so3",
                target="piezoelectric",
                seed=seed,
                spec=spec,
                target_scale=FROZEN_TARGET_SCALE,
            )
            yield f"{core}_piezoelectric_so3_aug_seed{seed}.yaml", PROFILE[core], text


def _loss_weight(_root: Path) -> GridRows:
    """NequIP SO(3) with zero-row loss weights 10 and 100 (weight 1 is the augmentation run)."""
    for weight in (10, 100):
        spec = _augmented_spec(f"mp_piezoelectric_augmented_w{weight}")
        for seed in SEEDS:
            text = _config_yaml(
                core="nequip",
                parity="so3",
                target="piezoelectric",
                seed=seed,
                spec=spec,
                target_scale=FROZEN_TARGET_SCALE,
                zero_row_loss_weight=weight,
            )
            yield f"nequip_so3_piezoelectric_aug_w{weight}_seed{seed}.yaml", "nequip", text


def _zero_injection(root: Path) -> GridRows:
    """NequIP SO(3) over the zero-labelled set sizes in ``results/zero_injection_sets.json``."""
    sets = json.loads((root / "results/zero_injection_sets.json").read_text())["sets"]
    for n in sorted(int(k[1:]) for k in sets):
        name = f"mp_piezoelectric_augmented_n{n}"
        spec = _augmented_spec(name, stem=name)
        for seed in SEEDS:
            text = _config_yaml(
                core="nequip",
                parity="so3",
                target="piezoelectric",
                seed=seed,
                spec=spec,
                target_scale=FROZEN_TARGET_SCALE,
            )
            yield f"nequip_piezoelectric_so3_n{n}_seed{seed}.yaml", "nequip", text


@dataclass(frozen=True)
class Grid:
    """A named experiment matrix and the directory its configs are written to."""

    directory: str
    rows: Callable[[Path], GridRows]
    # The loss-weight sweep has a single profile and one un-prefixed run list.
    single_run_list: bool = False


GRIDS: dict[str, Grid] = {
    "main": Grid("configs/grid", _main),
    "meanpool": Grid("configs/grid_meanpool", _meanpool),
    "sumpool": Grid("configs/grid_sumpool", _sumpool),
    "augmentation": Grid("configs/augmentation", _augmentation),
    "loss-weight": Grid("configs/loss_weight", _loss_weight, single_run_list=True),
    "zero-injection": Grid("configs/zero_injection", _zero_injection),
}


def generate_grid(name: str, root: Path) -> dict[str, list[str]]:
    """Write one grid's configs and run lists under ``root``; return run paths per profile."""
    if name not in GRIDS:
        raise ValueError(f"unknown grid {name!r}; choose from {sorted(GRIDS)}")
    grid = GRIDS[name]
    out = root / grid.directory
    out.mkdir(parents=True, exist_ok=True)
    runs: dict[str, list[str]] = {}
    for file_name, profile, text in grid.rows(root):
        (out / file_name).write_text(text)
        runs.setdefault(profile, []).append(f"{grid.directory}/{file_name}")
    for profile, paths in runs.items():
        list_name = "runs.txt" if grid.single_run_list else f"{profile}_runs.txt"
        (out / list_name).write_text("\n".join(paths) + "\n")
    return runs
