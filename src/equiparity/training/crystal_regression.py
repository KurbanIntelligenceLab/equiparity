"""Validated, complete-head tensor training for the crystal protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from equiparity.domain.experiment import ExperimentConfig
from equiparity.domain.target import TARGETS
from equiparity.evaluation.metrics import MetricSummary, regression_metrics
from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset, load_split
from equiparity.training.nequip_tensor import _irreps_targets, build_tensor_model, periodic_type_map
from equiparity.training.run_instrument import parameter_counts
from equiparity.training.tensor_optimization import atomic_json, fit_tensor
from equiparity.workflows.tensor_datasets import validate_tensor_dataset


@dataclass
class CrystalRegressionResult:
    val: MetricSummary
    test: MetricSummary | None
    n_params: int
    parameter_counts: dict[str, int]
    epochs_run: int
    selected_epoch: int
    baselines: dict[str, Any]
    checkpoint_best: dict[str, Any]
    checkpoint_latest: dict[str, Any]


def build_crystal_model(
    config: ExperimentConfig, metadata: dict[str, Any], device: torch.device
) -> tuple[Any, dict[str, Any]]:
    """Shared construction for training and checkpoint reload; no dataset recomputation."""
    if config.core != "mace":
        type_names, z_map = periodic_type_map()
        model = build_tensor_model(
            config, type_names, metadata["avg_neighbors"], TARGETS[config.target].irreps, device
        )
        if config.model.use_magnetic_magnitudes:
            from equiparity.models.site_scalars import MagnitudeConditionedModel

            model = MagnitudeConditionedModel(model, config.core).to(device)
        return model, {"z_map": z_map, "dtype": torch.float64}
    from mace import tools

    from equiparity.models.mace import MACEConfig, MACETensorModel

    cfg = MACEConfig(
        r_max=config.model.r_max,
        atomic_numbers=tuple(metadata["elements"]),
        num_interactions=config.model.num_layers,
        l_max=config.model.l_max,
        num_features=config.model.num_features,
        avg_num_neighbors=metadata["avg_neighbors"],
        seed=config.seed,
        model_dtype=config.training.precision,
        pooling=config.model.pooling,
        tensor_head=config.model.tensor_head,
    )
    dtype = torch.float32 if config.training.precision == "float32" else torch.float64
    model = MACETensorModel(cfg, config.parity, TARGETS[config.target].irreps).to(device, dtype)
    if config.model.use_magnetic_magnitudes:
        from equiparity.models.site_scalars import MagnitudeConditionedModel

        model = MagnitudeConditionedModel(model, config.core).to(device)
    return model, {
        "z_table": tools.AtomicNumberTable(metadata["elements"]),
        "dtype": dtype,
    }


def tensor_metrics(pred: np.ndarray, target: np.ndarray, irreps: str) -> dict[str, Any]:
    from e3nn import o3

    result: dict[str, Any] = regression_metrics(pred, target).to_dict()
    result["frobenius_rmse"] = float(np.sqrt(np.mean(np.sum((pred - target) ** 2, axis=1))))
    result["per_irrep_mae"] = [
        float(np.abs(pred[:, sl] - target[:, sl]).mean()) for sl in o3.Irreps(irreps).slices()
    ]
    return result


def train_crystal_regression(
    config: ExperimentConfig, run_dir: Path, snapshot: dict[str, Any], resume: Path | None = None
) -> CrystalRegressionResult:
    """Train using cached graphs and validation selection; test evaluation is opt-in."""
    audit = validate_tensor_dataset(config.processed_npz, config.split_npz, config.target)
    device = torch.device(config.training.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("crystal training requested CUDA but no GPU is available")
    torch.serialization.add_safe_globals([slice])
    dtype = torch.float32 if config.training.precision == "float32" else torch.float64
    data = load_crystal_dataset(config.processed_npz, (config.target,))
    if config.model.use_magnetic_magnitudes != (data.magnetic_magnitudes is not None):
        raise ValueError("model magnetic magnitude input does not match dataset")
    datasets = {}
    parts = ["train", "val"] + (["test"] if config.training.evaluate_test else [])
    for part in parts:
        ids = load_split(config.split_npz, part)
        cap = (
            config.training.max_train_samples
            if part == "train"
            else config.training.max_eval_samples
        )
        datasets[part] = CrystalDataset(data, ids[:cap] if cap is not None else ids)
        if not len(datasets[part]):
            raise ValueError(f"empty {part} split")
    targets = {p: _irreps_targets(ds, config.target, config.target) for p, ds in datasets.items()}
    scale = config.training.target_scale or float(np.sqrt(np.mean(targets["train"] ** 2))) or 1.0
    metadata: dict[str, Any] = {}
    if config.core == "mace":
        from mace import tools

        from equiparity.training.mace_scalar import _batches, _to_mace_data
        from equiparity.training.mace_tensor import _avg_neighbors

        metadata["elements"] = list(range(1, 101))
        table = tools.AtomicNumberTable(metadata["elements"])
        graphs = {
            p: [_to_mace_data(ds[i].structure, table, config.model.r_max) for i in range(len(ds))]
            for p, ds in datasets.items()
        }
        metadata["avg_neighbors"] = _avg_neighbors(graphs["train"])

        def collate(items: list[Any]) -> dict[str, Any]:
            batch = next(iter(_batches(items, len(items), dtype)))
            return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    else:
        from nequip.data import AtomicDataDict

        from equiparity.training.nequip_data import avg_num_neighbors, move_batch, to_atomic_data

        _, z_map = periodic_type_map()
        graphs = {
            p: [
                to_atomic_data(ds[i].structure, z_map, config.model.r_max, torch.float64)
                for i in range(len(ds))
            ]
            for p, ds in datasets.items()
        }
        metadata["avg_neighbors"] = avg_num_neighbors(graphs["train"])

        def collate(items: list[Any]) -> dict[str, Any]:
            return move_batch(AtomicDataDict.batched_from_list(items), device, torch.float64)

    model, _ = build_crystal_model(config, metadata, device)
    batch_size = config.training.batch_size
    norm_targets = torch.tensor(targets["train"] / scale, dtype=dtype, device=device)
    weights = torch.ones(len(norm_targets), dtype=dtype, device=device)
    weights[norm_targets.abs().amax(dim=1) == 0] = config.training.zero_row_loss_weight

    def batches():  # noqa: ANN202
        order = np.random.permutation(len(norm_targets))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            yield collate([graphs["train"][i] for i in idx]), norm_targets[idx], weights[idx]

    def predict(part: str) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return (
                np.concatenate(
                    [
                        model(collate(graphs[part][i : i + batch_size])).cpu().numpy()
                        for i in range(0, len(graphs[part]), batch_size)
                    ]
                )
                * scale
            )

    irreps = TARGETS[config.target].irreps

    def evaluate() -> dict[str, Any]:
        train_prediction = predict("train")
        diagnostics = {
            "train": tensor_metrics(train_prediction, targets["train"], irreps),
            "val": tensor_metrics(predict("val"), targets["val"], irreps),
        }
        zero = np.max(np.abs(targets["train"]), axis=1) == 0
        for name, mask in (("train_zero", zero), ("train_nonzero", ~zero)):
            if mask.any():
                diagnostics[name] = tensor_metrics(
                    train_prediction[mask], targets["train"][mask], irreps
                )
        if zero.any():
            norms = np.linalg.norm(train_prediction[zero], axis=1)
            diagnostics["train_zero"]["fraction_above_0.01"] = float((norms > 0.01).mean())
        return diagnostics

    baselines = {}
    for part in parts:
        baselines[part] = {
            name: tensor_metrics(np.broadcast_to(value, targets[part].shape), targets[part], irreps)
            for name, value in (
                ("zero", np.zeros(targets[part].shape[1])),
                ("training_mean", targets["train"].mean(axis=0)),
                ("training_median", np.median(targets["train"], axis=0)),
            )
        }
    atomic_json(baselines, run_dir / "baselines.json")
    identity = {
        "config": snapshot,
        "data_sha256": audit["data_sha256"],
        "split_sha256": audit["split_sha256"],
        "scale": scale,
        "model_metadata": metadata,
    }
    fit = fit_tensor(model, config.training, batches, evaluate, run_dir, identity, resume)
    val_pred = predict("val")
    test = regression_metrics(predict("test"), targets["test"]) if "test" in targets else None
    atomic_json(
        {
            "val": tensor_metrics(val_pred, targets["val"], irreps),
            "selected_epoch": fit["best_epoch"],
        },
        run_dir / "selected_diagnostics.json",
    )
    return CrystalRegressionResult(
        val=regression_metrics(val_pred, targets["val"]),
        test=test,
        n_params=sum(p.numel() for p in model.parameters()),
        parameter_counts=parameter_counts(model),
        epochs_run=config.training.epochs,
        selected_epoch=fit["best_epoch"],
        baselines=baselines,
        checkpoint_best=fit["checkpoint_best"],
        checkpoint_latest=fit["checkpoint_latest"],
    )
