"""Run one experiment from a config: train, evaluate, and write provenance + metrics.

Composes seeding, the core-specific trainer, and provenance writing (CODING_RULES.md Section E).
Every run produces an ``outputs/<experiment_id>/`` directory with a manifest, a config snapshot,
and metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from equiparity.domain.experiment import ExperimentConfig
from equiparity.reproducibility import collect_provenance, seed_everything, write_manifest
from equiparity.training.nequip_scalar import RunResult, train_scalar

_MANIFEST_DIRS = {
    "qm9": ("data/manifests/qm9.yaml", "data/splits/qm9.yaml"),
    "mp_elastic": ("data/manifests/mp_elastic.yaml", "data/splits/mp_elastic.yaml"),
    "mp_piezoelectric": (
        "data/manifests/mp_piezoelectric.yaml",
        "data/splits/mp_piezoelectric.yaml",
    ),
}


def _config_snapshot(config: ExperimentConfig) -> dict[str, object]:
    """Serialize a config to a plain, YAML-ready mapping."""
    return {
        "seed": config.seed,
        "core": config.core,
        "parity": config.parity.value,
        "target": config.target,
        "dataset": config.dataset,
        "processed_npz": str(config.processed_npz),
        "split_npz": str(config.split_npz),
        "model": {
            "num_layers": config.model.num_layers,
            "l_max": config.model.l_max,
            "num_features": config.model.num_features,
            "r_max": config.model.r_max,
        },
        "training": {
            "batch_size": config.training.batch_size,
            "epochs": config.training.epochs,
            "lr": config.training.lr,
            "weight_decay": config.training.weight_decay,
            "max_train_samples": config.training.max_train_samples,
            "max_eval_samples": config.training.max_eval_samples,
        },
    }


def run_experiment(config: ExperimentConfig, *, allow_dirty: bool = False) -> Path:
    """Train per ``config``, then write manifest, config snapshot, and metrics. Returns the run dir.

    Args:
        config: The validated experiment configuration.
        allow_dirty: Permit a dirty git tree (debug/smoke runs). Final-result runs must be clean.

    Returns:
        The ``outputs/<experiment_id>/`` directory path.
    """
    if config.core != "nequip":
        raise NotImplementedError(f"only the nequip core is wired so far, got {config.core!r}")

    seed_everything(config.seed)
    result: RunResult = train_scalar(config)

    snapshot = _config_snapshot(config)
    config_text = yaml.safe_dump(snapshot, sort_keys=True)
    dataset_manifest, split_manifest = _MANIFEST_DIRS.get(config.dataset, ("", ""))
    manifest = collect_provenance(
        config_text=config_text,
        config_path=f"<{config.run_label}>",
        dataset_manifest=dataset_manifest,
        split_manifest=split_manifest,
        seed=config.seed,
    )
    run_dir = config.output_dir / manifest.experiment_id
    write_manifest(manifest, run_dir, allow_dirty=allow_dirty)
    (run_dir / "config_snapshot.yaml").write_text(config_text)
    metrics = {
        "run_label": config.run_label,
        "target": config.target,
        "parity": config.parity.value,
        "n_params": result.n_params,
        "epochs_run": result.epochs_run,
        "val": result.val.to_dict(),
        "test": result.test.to_dict(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return run_dir
