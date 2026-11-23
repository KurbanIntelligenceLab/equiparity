"""Generate training configurations for the crystal and molecular experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

CORE_PARITY: dict[str, tuple[str, ...]] = {
    "nequip": ("o3", "so3"),
    "allegro": ("o3", "so3"),
    "mace": ("o3", "so3"),
}
PROFILE = {"nequip": "nequip", "allegro": "nequip", "mace": "mace"}
SEEDS = (0, 1, 2)
FEATURES = 64


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
        dataset="mp_elastic_ieee",
        npz="data/raw/mp/mp_elastic_ieee_processed.npz",
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
    return "\n".join(lines) + "\n"


# A grid yields (file name, profile, yaml text).
GridRows = Iterator[tuple[str, str, str]]


def _final_epoch(targets: tuple[str, ...]) -> GridRows:
    for core, parities in CORE_PARITY.items():
        for parity in parities:
            for target in targets:
                spec = TARGETS[target]
                for seed in SEEDS:
                    text = _config_yaml(
                        core=core, parity=parity, target=target, seed=seed, spec=spec
                    )
                    yield f"{core}_{target}_{parity}_seed{seed}.yaml", PROFILE[core], text


def _tensor_configs(_root: Path) -> GridRows:
    """Model and optimizer settings shared by crystal regression runs."""
    import yaml

    for core in ("nequip", "allegro", "mace"):
        for target in ("piezoelectric", "elastic"):
            for parity in ("o3", "so3"):
                for seed in SEEDS:
                    spec = TARGETS[target]
                    raw = yaml.safe_load(
                        _config_yaml(
                            core=core,
                            parity=parity,
                            target=target,
                            seed=seed,
                            spec=spec,
                            pooling="mean",
                        )
                    )
                    raw["protocol"] = "validation_selected"
                    raw["dataset"] = f"mp_{target}_ieee"
                    raw["processed_npz"] = f"data/raw/mp/mp_{target}_ieee_processed.npz"
                    raw["output_dir"] = "outputs/crystal"
                    raw["model"]["tensor_head"] = "complete"
                    raw["training"].update(
                        lr=0.001,
                        loss="mse",
                        scheduler="plateau",
                        scheduler_patience=10,
                        min_lr=1e-6,
                        gradient_clip=10.0,
                        evaluate_test=False,
                    )
                    yield (
                        f"{core}_{target}_{parity}_seed{seed}.yaml",
                        PROFILE[core],
                        yaml.safe_dump(raw, sort_keys=False),
                    )


def _crystal(root: Path) -> GridRows:
    """Piezoelectric Huber and MSE comparisons and elastic MSE runs."""
    import yaml

    for name, profile, text in _tensor_configs(root):
        original = yaml.safe_load(text)
        losses = (
            ("mse", "tensor_huber")
            if original["target"] == "piezoelectric" and original["core"] == "nequip"
            else ("tensor_huber",)
            if original["target"] == "piezoelectric"
            else ("mse",)
        )
        for loss in losses:
            raw = yaml.safe_load(text)
            raw["output_dir"] = "outputs/crystal"
            if raw["target"] == "elastic":
                raw["model"]["use_magnetic_magnitudes"] = True
            raw["training"]["loss"] = loss
            if loss == "tensor_huber":
                raw["training"]["huber_delta"] = 0.5
            yield f"{loss}_{name}", profile, yaml.safe_dump(raw, sort_keys=False)


def _zero_controls(root: Path) -> GridRows:
    """Zero-label supervision and loss weights for NequIP."""
    import yaml

    for name, profile, text in _crystal(root):
        raw = yaml.safe_load(text)
        if (
            raw["core"] != "nequip"
            or raw["target"] != "piezoelectric"
            or raw["training"]["loss"] != "tensor_huber"
        ):
            continue
        for weight in (1, 10, 100) if raw["parity"] == "so3" else (1,):
            cfg = yaml.safe_load(text)
            cfg["dataset"] = "mp_piezoelectric_ieee_augmented"
            cfg["processed_npz"] = "data/raw/mp/mp_piezoelectric_ieee_augmented_processed.npz"
            cfg["split_npz"] = "data/splits/mp_piezoelectric_ieee_augmented_split.npz"
            cfg["output_dir"] = "outputs/zero_labels"
            cfg["training"]["target_scale"] = 0.7491590371727936
            cfg["training"]["zero_row_loss_weight"] = weight
            yield f"w{weight}_{name}", profile, yaml.safe_dump(cfg, sort_keys=False)


@dataclass(frozen=True)
class Grid:
    """A named experiment matrix and the directory its configs are written to."""

    directory: str
    rows: Callable[[Path], GridRows]


GRIDS: dict[str, Grid] = {
    "main": Grid("configs/crystal", _crystal),
    "zero-labels": Grid("configs/zero_labels", _zero_controls),
    "molecular": Grid("configs/molecular", lambda _: _final_epoch(("U0", "dipole"))),
    "sum-pooled": Grid("configs/sum_pooled", lambda _: _final_epoch(("piezoelectric",))),
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
        list_name = f"{profile}_runs.txt"
        (out / list_name).write_text("\n".join(paths) + "\n")
    return runs
