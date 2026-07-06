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
from equiparity.training.nequip_tensor import train_tensor
from equiparity.training.nequip_vector import train_dipole

# Scalar targets use the energy readout; vector -> L=1 head; tensor -> tensor head.
_VECTOR_TARGETS = frozenset({"dipole"})
_TENSOR_TARGETS = frozenset({"elastic", "piezoelectric"})
_OOD_NPZ = "data/raw/mp/mp_ood_centrosymmetric_processed.npz"

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
            "precision": config.training.precision,
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
    seed_everything(config.seed)
    ood = _OOD_NPZ if config.target == "piezoelectric" else None
    if config.core == "mace":
        # MACE has a separate data pipeline (mace.data), so its heads live in mace_* trainers.
        if config.target in _TENSOR_TARGETS:
            from equiparity.training.mace_tensor import train_mace_tensor

            result: RunResult = train_mace_tensor(config, ood_npz=ood)
        elif config.target in _VECTOR_TARGETS:
            from equiparity.training.mace_tensor import train_mace_dipole

            result = train_mace_dipole(config)
        else:
            from equiparity.training.mace_scalar import train_mace_scalar

            result = train_mace_scalar(config)
    elif config.core == "equiformer_v2":
        # EquiformerV2 is a fixed SO(3) representative with a torch_geometric (PBC) pipeline.
        if config.target in _TENSOR_TARGETS:
            from equiparity.training.equiformer_tensor import train_equiformer_tensor

            result = train_equiformer_tensor(config, ood_npz=ood)
        elif config.target in _VECTOR_TARGETS:
            from equiparity.training.equiformer_tensor import train_equiformer_dipole

            result = train_equiformer_dipole(config)
        else:
            from equiparity.training.equiformer_tensor import train_equiformer_scalar

            result = train_equiformer_scalar(config)
    elif config.core == "clifford_stf":
        # CliffordSTF: O(3) geometric-algebra representative (requires float64) across all targets.
        if config.target in _TENSOR_TARGETS:
            from equiparity.training.clifford_tensor import train_clifford_tensor

            result = train_clifford_tensor(config, ood_npz=ood)
        elif config.target in _VECTOR_TARGETS:
            from equiparity.training.clifford_tensor import train_clifford_dipole

            result = train_clifford_dipole(config)
        else:
            from equiparity.training.clifford_tensor import train_clifford_scalar

            result = train_clifford_scalar(config)
    elif config.target in _TENSOR_TARGETS:
        result = train_tensor(config, ood_npz=ood)  # nequip or allegro
    elif config.target in _VECTOR_TARGETS:
        result = train_dipole(config)  # nequip or allegro
    else:
        result = train_scalar(config)  # nequip or allegro energy readout

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
    # Tensor runs carry the OOD violation stats (piezoelectric headline).
    for field in ("ood_violation_median", "ood_violation_max", "ood_false_flag_fraction"):
        value = getattr(result, field, None)
        if value is not None:
            metrics[field] = value
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return run_dir
