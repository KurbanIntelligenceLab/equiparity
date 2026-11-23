"""Independently reload selected crystal checkpoints with a guarded test protocol."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from equiparity.domain.target import TARGETS
from equiparity.evaluation.paired_errors import paired_mae_summary
from equiparity.inference.reload import load_trained
from equiparity.io.config import load_experiment_config
from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset, load_split
from equiparity.training.crystal_regression import tensor_metrics
from equiparity.training.nequip_tensor import _irreps_targets
from equiparity.training.tensor_optimization import atomic_json
from equiparity.workflows.evaluation_protocol import verify_test_authorization
from equiparity.workflows.prepare_data import sha256_file
from equiparity.workflows.tensor_datasets import validate_tensor_dataset


def evaluate_checkpoint(
    root: Path,
    run_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    partition: str = "val",
    protocol_lock: Path | None = None,
) -> dict[str, Any]:
    """Copy one atomic selected checkpoint, verify provenance, and save its predictions."""
    if partition not in ("val", "test"):
        raise ValueError("partition must be val or test")
    if partition == "test" and protocol_lock is None:
        raise ValueError("held-out evaluation requires a protocol lock")
    config = load_experiment_config(run_dir / "config_snapshot.yaml")
    if config.protocol != "validation_selected":
        raise ValueError("validation reload is restricted to validation_selected runs")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("validation reload requested CUDA but no GPU is available")
    data_path, split_path = root / config.processed_npz, root / config.split_npz
    audit = validate_tensor_dataset(data_path, split_path, config.target)
    lock_hash = (
        verify_test_authorization(config, audit, protocol_lock)
        if partition == "test" and protocol_lock is not None
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    # The live run can replace its selected checkpoint atomically while this copy
    # proceeds; opening the source once gives this evaluation a stable snapshot.
    shutil.copyfile(run_dir / "checkpoint_selected.pt", output_dir / "checkpoint_selected.pt")
    shutil.copyfile(run_dir / "config_snapshot.yaml", output_dir / "config_snapshot.yaml")
    selected = torch.load(
        output_dir / "checkpoint_selected.pt", map_location="cpu", weights_only=False
    )
    if partition == "test":
        latest = torch.load(
            run_dir / "checkpoint_latest.pt", map_location="cpu", weights_only=False
        )
        if latest["epoch"] != config.training.epochs or latest["best_epoch"] != selected["epoch"]:
            raise ValueError("held-out evaluation requires a completed, validation-selected run")
    for key in ("data_sha256", "split_sha256"):
        if selected["identity"][key] != audit[key]:
            raise ValueError(f"selected checkpoint {key} differs from validated data")
    trained = load_trained(output_dir, repo_root=root, device=torch.device(device))
    data = load_crystal_dataset(data_path, (config.target,))
    train_ids = load_split(split_path, "train")[: config.training.max_train_samples]
    val_ids = load_split(split_path, partition)[: config.training.max_eval_samples]
    train, val = CrystalDataset(data, train_ids), CrystalDataset(data, val_ids)
    train_target = _irreps_targets(train, config.target, config.target)
    target = _irreps_targets(val, config.target, config.target)
    prediction = trained.predict([val[i].structure for i in range(len(val))])
    metrics = tensor_metrics(prediction, target, TARGETS[config.target].irreps)
    if partition == "val":
        history = json.loads((run_dir / "history.json").read_text())
        expected = history[int(selected["epoch"]) - 1]["val"]
        for name in ("mae", "rmse", "frobenius_rmse", "per_irrep_mae"):
            if not np.allclose(metrics[name], expected[name], rtol=1e-5, atol=1e-7):
                raise ValueError(f"reloaded {name} disagrees with the selected epoch")
    comparisons = {
        name: paired_mae_summary(prediction, target, baseline)
        for name, baseline in (
            ("zero", np.zeros(target.shape[1])),
            ("training_mean", train_target.mean(axis=0)),
            ("training_median", np.median(train_target, axis=0)),
        )
    }
    np.savez_compressed(
        output_dir / f"{'validation' if partition == 'val' else 'test'}_predictions.npz",
        ids=val_ids,
        prediction=prediction,
        target=target,
    )
    result = {
        "source_run": str(run_dir),
        "selected_epoch": int(selected["epoch"]),
        "checkpoint_sha256": sha256_file(output_dir / "checkpoint_selected.pt"),
        "data_sha256": audit["data_sha256"],
        "split_sha256": audit["split_sha256"],
        "partition": partition,
        "protocol_lock_sha256": lock_hash,
        "interpretation": (
            "validation-selected descriptive analysis; not held-out test evidence"
            if partition == "val"
            else "held-out test under frozen protocol"
        ),
        "reproduced_logged_metrics": partition == "val",
        "metrics": metrics,
        "comparisons": comparisons,
    }
    atomic_json(
        result, output_dir / f"{'validation' if partition == 'val' else 'test'}_diagnostics.json"
    )
    return result
