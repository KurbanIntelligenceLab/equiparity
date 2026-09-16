"""Build the release execution manifest: every run the study trained, with its configuration,
data, software and hardware.

The manifest is assembled only from committed inputs, so it rebuilds identically from a clean
checkout: the per-run YAML configs under ``configs/``, the dataset and split manifests, the
exact package versions in ``uv.lock``, and the compute record in ``results/appendix_stats.json``.

CLI: ``equiparity manifest`` (writes ``results/run_manifest.json``).
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

from equiparity.workflows.grids import GRIDS, PROFILE

MANIFEST_PATH = "results/run_manifest.json"

# Study each grid belongs to, as reported.
_STUDY = {
    "main": "matched-pair grid",
    "meanpool": "readout pooling control",
    "sumpool": "readout pooling control",
    "augmentation": "zero-label augmentation",
    "loss-weight": "zero-row loss weighting",
    "zero-injection": "zero-injection learning curve",
}

# Packages whose versions determine model numerics, per install profile.
_PACKAGES = {
    "common": ("numpy", "torch", "ase", "pyyaml"),
    "nequip": ("e3nn", "nequip", "nequip-allegro", "torch-geometric"),
    "mace": ("e3nn", "mace-torch"),
    "data": ("pymatgen", "spglib", "mp-api"),
}

# Pins that separate the two install profiles within one lockfile, and the CUDA torch build the
# GPU runs used (the lock also resolves a CPU wheel for macOS).
_PROFILE_PINS = {"nequip": {"e3nn": "0.6.0"}, "mace": {"e3nn": "0.4.4"}}
_TRAINING_TORCH_SUFFIX = "+cu128"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _locked_versions(root: Path) -> dict[str, dict[str, str]]:
    """Resolved versions from ``uv.lock``; a package may resolve per platform or per extra."""
    lock = tomllib.loads((root / "uv.lock").read_text())
    found: dict[str, set[str]] = {}
    for pkg in lock.get("package", []):
        found.setdefault(pkg["name"], set()).add(pkg["version"])
    out: dict[str, dict[str, str]] = {}
    for profile, names in _PACKAGES.items():
        resolved: dict[str, str] = {}
        for name in names:
            versions = sorted(found.get(name, ()))
            if name == "torch":
                versions = [v for v in versions if v.endswith(_TRAINING_TORCH_SUFFIX)] or versions
            pin = _PROFILE_PINS.get(profile, {}).get(name)
            if pin is not None:
                versions = [v for v in versions if v == pin]
            if len(versions) != 1:
                raise ValueError(f"cannot resolve one {name} version for {profile}: {versions}")
            resolved[name] = versions[0]
        out[profile] = resolved
    return out


def _python_requirement(root: Path) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    return str(project["requires-python"])


def _relative(path: Path, root: Path) -> str | None:
    return path.relative_to(root).as_posix() if path.exists() else None


def _run_entry(root: Path, grid: str, config_path: str) -> dict[str, Any]:
    path = root / config_path
    cfg = yaml.safe_load(path.read_text())
    # Loss-weight runs relabel the augmented dataset with a `_w<weight>` suffix for unique run keys.
    dataset = re.sub(r"_w\d+$", "", cfg["dataset"])
    dataset_manifest = root / f"data/manifests/{dataset}.yaml"
    split_manifest = root / f"data/splits/{dataset}.yaml"
    training = cfg.get("training", {})
    return {
        "run_label": f"{cfg['core']}_{cfg['parity']}_{cfg['target']}_seed{cfg['seed']}",
        "study": _STUDY[grid],
        "config": config_path,
        "config_sha256": _sha256(path),
        "core": cfg["core"],
        "parity": cfg["parity"],
        "target": cfg["target"],
        "seed": cfg["seed"],
        "dataset": cfg["dataset"],
        "processed_npz": cfg["processed_npz"],
        "split_npz": cfg["split_npz"],
        "dataset_manifest": _relative(dataset_manifest, root),
        "split_manifest": _relative(split_manifest, root),
        "pooling": cfg.get("model", {}).get("pooling", "sum"),
        "precision": training.get("precision"),
        "epochs": training.get("epochs"),
        "install_profile": PROFILE[cfg["core"]],
    }


def build_run_manifest(root: Path) -> dict[str, Any]:
    """Assemble the manifest from committed configs and records under ``root``."""
    compute = json.loads((root / "results/appendix_stats.json").read_text())["compute"]
    runs: list[dict[str, Any]] = []
    for name, grid in GRIDS.items():
        directory = root / grid.directory
        for run_list in sorted(directory.glob("*runs.txt")):
            for line in run_list.read_text().splitlines():
                if line.strip():
                    runs.append(_run_entry(root, name, line.strip()))
    for run in runs:
        if run["study"] == _STUDY["main"]:
            run["gpu"] = compute["per_core"][run["core"]]["gpu"]
    by_study: dict[str, int] = {}
    for run in runs:
        by_study[run["study"]] = by_study.get(run["study"], 0) + 1
    return {
        "description": (
            "Execution manifest for every training run in the study. Each entry names the exact "
            "config file (with its SHA-256), the dataset and split it read, and its install "
            "profile. Rebuild with `equiparity manifest`."
        ),
        "counts": {"total": len(runs), **by_study},
        "software": {
            "python": _python_requirement(root),
            "lockfile": "uv.lock",
            "lockfile_sha256": _sha256(root / "uv.lock"),
            "packages": _locked_versions(root),
            "cuda": "12.8 (PyTorch cu128 wheels)",
        },
        "hardware": {
            "gpu_by_core": {core: v["gpu"] for core, v in compute["per_core"].items()},
            "note": (
                "Per-core wall-clock is not a like-for-like architecture comparison: two GPU "
                "classes were used."
            ),
        },
        "compute": {
            "total_train_hours": compute["total_train_hours"],
            "train_hours_by_gpu": compute["train_hours_by_gpu"],
            "per_core": compute["per_core"],
            "scope": "matched-pair grid only; sweep and pooling runs excluded",
        },
        "seeding": "each run seeds Python, NumPy and PyTorch from its config `seed`",
        "runs": runs,
    }


def write_run_manifest(root: Path) -> Path:
    """Build the manifest and write it to ``results/run_manifest.json``; return its path."""
    out = root / MANIFEST_PATH
    out.write_text(json.dumps(build_run_manifest(root), indent=2) + "\n")
    return out
