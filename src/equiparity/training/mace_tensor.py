"""MACE tensor and dipole heads: train ``MACETensorModel`` / ``MACEDipoleModel``.

MACE uses its own data pipeline (Configuration + vendored torch_geometric), so unlike the
nequip/allegro cores it needs a dedicated tensor trainer. The head reads MACE's per-node
equivariant features and sums them per structure (:mod:`equiparity.models.mace`). Trains in
float32 mixed precision by default (MACE is natively float32).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from equiparity.domain.experiment import ExperimentConfig
from equiparity.domain.target import TARGETS
from equiparity.evaluation.metrics import MetricSummary, regression_metrics
from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset, load_split
from equiparity.io.qm9_dataset import QM9Dataset, load_qm9
from equiparity.io.qm9_dataset import load_split as load_qm9_split
from equiparity.training.mace_scalar import _batches, _to_mace_data
from equiparity.training.nequip_scalar import RunResult
from equiparity.training.nequip_tensor import (
    TensorRunResult,
    _irreps_targets,
    _raw_variant,
    violation_magnitudes,
)
from equiparity.training.ood_eval import evaluate_ood_variants


def _elements_of(*datasets) -> list[int]:  # noqa: ANN002
    """Sorted unique atomic numbers across one or more crystal/molecule datasets."""
    zs: list[int] = []
    for ds in datasets:
        for i in range(len(ds)):
            zs.extend(int(z) for z in ds[i].structure.atomic_numbers)
    return sorted(set(zs))


def _avg_neighbors(graphs) -> float:  # noqa: ANN001
    """Mean neighbour count over MACE graphs (for the model's message normalization)."""
    edges = sum(int(g.edge_index.shape[1]) for g in graphs)
    nodes = sum(int(g.num_nodes) for g in graphs)
    return edges / nodes if nodes else 1.0


def _predict(model, graphs, batch_size, dtype, device):  # noqa: ANN001, ANN202
    """Per-structure tensor predictions (irreps form), shape (n, dim)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in _batches(graphs, batch_size, dtype):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


def train_mace_tensor(config: ExperimentConfig, *, ood_npz: str | None = None) -> TensorRunResult:
    """Train a MACE tensor head (elastic/piezoelectric), evaluate it, and measure OOD violation."""
    from mace import tools

    from equiparity.models.mace import MACEConfig, MACETensorModel

    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if config.training.precision == "float32" else torch.float64
    kind = config.target
    r_max = config.model.r_max
    data = load_crystal_dataset(config.processed_npz, (config.target,))

    def subset(part: str, cap: int | None) -> CrystalDataset:
        ds = CrystalDataset(data, load_split(config.split_npz, part))
        if cap is not None:
            ids = np.array([ds[i].identifier for i in range(min(cap, len(ds)))])
            ds = CrystalDataset(data, ids)
        return ds

    train_ds = subset("train", config.training.max_train_samples)
    val_ds = subset("val", config.training.max_eval_samples)
    test_ds = subset("test", config.training.max_eval_samples)
    element_sets = [train_ds, val_ds, test_ds]
    # The OOD set may contain elements unseen in training; MACE's z_table must cover them (their
    # embeddings stay untrained, which is fine — the O(3) violation cancels by symmetry regardless).
    if ood_npz is not None and kind == "piezoelectric":
        element_sets.append(CrystalDataset(load_crystal_dataset(ood_npz)))
    elements = _elements_of(*element_sets)
    z_table = tools.AtomicNumberTable(elements)

    def graphs_of(ds: CrystalDataset) -> list:
        return [_to_mace_data(ds[i].structure, z_table, r_max) for i in range(len(ds))]

    train_graphs = graphs_of(train_ds)
    train_targets = _irreps_targets(train_ds, config.target, kind)
    scale = config.training.target_scale or float(train_targets.std()) or 1.0
    norm_targets = torch.tensor(train_targets / scale, dtype=dtype, device=device)

    mace_cfg = MACEConfig(
        r_max=r_max,
        atomic_numbers=tuple(elements),
        num_interactions=config.model.num_layers,
        l_max=config.model.l_max,
        num_features=config.model.num_features,
        avg_num_neighbors=_avg_neighbors(train_graphs),
        seed=config.seed,
        model_dtype=config.training.precision,
    )
    model = MACETensorModel(mace_cfg, config.parity, TARGETS[config.target].irreps).to(device)
    n_params = sum(int(p.numel()) for p in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    batch_size = config.training.batch_size

    def evaluate(ds: CrystalDataset) -> MetricSummary:
        preds = _predict(model, graphs_of(ds), batch_size, dtype, device) * scale
        return regression_metrics(preds, _irreps_targets(ds, config.target, kind))

    # H-1 instrumentation: per-epoch false-flag fraction on the idealized OOD variant. Graphs
    # are built once and reused; one forward pass over the population per epoch.
    epoch_ood_graphs = None
    if ood_npz is not None and kind == "piezoelectric":
        ood_ds = CrystalDataset(load_crystal_dataset(ood_npz))
        if config.training.max_eval_samples is not None:
            cap = min(config.training.max_eval_samples, len(ood_ds))
            ids = np.array([ood_ds[i].identifier for i in range(cap)])
            ood_ds = CrystalDataset(load_crystal_dataset(ood_npz), ids)
        epoch_ood_graphs = graphs_of(ood_ds)
    ood_history: list[dict[str, float]] | None = None if epoch_ood_graphs is None else []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train = time.perf_counter()
    ckpt_every = max(1, config.training.epochs // 10)
    best_val, checkpoint_best = float("inf"), None
    for epoch in range(config.training.epochs):
        model.train()
        offset = 0
        for batch in _batches(train_graphs, batch_size, dtype):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            n = int(batch["ptr"].shape[0]) - 1
            pred = model(batch).to(dtype)
            loss = loss_fn(pred, norm_targets[offset : offset + n])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            offset += n
        if (epoch + 1) % ckpt_every == 0 or epoch == config.training.epochs - 1:
            vm = evaluate(val_ds).mae
            if vm < best_val:
                best_val = vm
                checkpoint_best = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        if epoch_ood_graphs is not None and ood_history is not None:
            v = violation_magnitudes(
                _predict(model, epoch_ood_graphs, batch_size, dtype, device) * scale
            )
            ood_history.append(
                {
                    "epoch": epoch + 1,
                    "false_flag_at_0.01": float((v > 0.01).mean()),
                    "median": float(np.median(v)),
                }
            )
    train_seconds = time.perf_counter() - t_train

    t_eval = time.perf_counter()
    val, test = evaluate(val_ds), evaluate(test_ds)
    eval_seconds = time.perf_counter() - t_eval

    ood_median = ood_max = ood_ff = None
    ood_variants = ood_vectors = None
    ood_seconds = 0.0
    if ood_npz is not None and kind == "piezoelectric":

        def _ood_violations(npz_path: str) -> np.ndarray:
            ood = CrystalDataset(load_crystal_dataset(npz_path))
            if config.training.max_eval_samples is not None:
                cap = min(config.training.max_eval_samples, len(ood))
                ids = np.array([ood[i].identifier for i in range(cap)])
                ood = CrystalDataset(load_crystal_dataset(npz_path), ids)
            preds = _predict(model, graphs_of(ood), batch_size, dtype, device) * scale
            return violation_magnitudes(preds)

        t_ood = time.perf_counter()
        results = evaluate_ood_variants(
            _ood_violations, {"idealized": ood_npz, "raw": _raw_variant(ood_npz)}
        )
        ood_seconds = time.perf_counter() - t_ood
        ood_variants = {k: v["stats"] for k, v in results.items()}
        ood_vectors = {k: v["vector"] for k, v in results.items()}
        prim = ood_variants.get("idealized") or next(iter(ood_variants.values()))
        ood_median, ood_max = prim["median"], prim["max"]
        ood_ff = prim["false_flag_at_0.01"]

    peak_mem = float(torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
    timing = {
        "train_seconds": train_seconds,
        "train_seconds_per_epoch": train_seconds / max(1, config.training.epochs),
        "eval_seconds": eval_seconds,
        "ood_seconds": ood_seconds,
        "train_throughput_structs_per_s": len(train_graphs)
        * config.training.epochs
        / max(1e-9, train_seconds),
        "peak_gpu_mem_mb": peak_mem,
    }
    checkpoint_latest = {
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "epoch": config.training.epochs,
    }

    return TensorRunResult(
        val=val,
        test=test,
        n_params=n_params,
        epochs_run=config.training.epochs,
        ood_violation_median=ood_median,
        ood_violation_max=ood_max,
        ood_false_flag_fraction=ood_ff,
        ood_variants=ood_variants,
        ood_vectors=ood_vectors,
        ood_false_flag_history=ood_history,
        timing=timing,
        checkpoint_best=checkpoint_best,
        checkpoint_latest=checkpoint_latest,
    )


def train_mace_dipole(config: ExperimentConfig) -> RunResult:
    """Train a MACE L=1 dipole head on QM9 and evaluate it (scale-only normalization)."""
    from mace import tools

    from equiparity.models.mace import MACEConfig, MACEDipoleModel

    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if config.training.precision == "float32" else torch.float64
    r_max = config.model.r_max
    data = load_qm9(config.processed_npz)
    elements = sorted(int(z) for z in np.unique(data.z))
    z_table = tools.AtomicNumberTable(elements)

    def subset(part: str, cap: int | None) -> QM9Dataset:
        ds = QM9Dataset(data, load_qm9_split(config.split_npz, part))
        if cap is not None:
            ds = QM9Dataset(
                data, np.array([int(ds[i].identifier) for i in range(min(cap, len(ds)))])
            )
        return ds

    def prepare(ds: QM9Dataset):  # noqa: ANN202
        graphs = [_to_mace_data(ds[i].structure, z_table, r_max) for i in range(len(ds))]
        targets = np.stack(
            [
                np.asarray(ds[i].targets[config.target], dtype=np.float64).reshape(3)
                for i in range(len(ds))
            ]
        )
        return graphs, targets

    train_graphs, train_targets = prepare(subset("train", config.training.max_train_samples))
    val_graphs, val_targets = prepare(subset("val", config.training.max_eval_samples))
    test_graphs, test_targets = prepare(subset("test", config.training.max_eval_samples))

    scale = config.training.target_scale or float(train_targets.std()) or 1.0
    norm_targets = torch.tensor(train_targets / scale, dtype=dtype, device=device)

    mace_cfg = MACEConfig(
        r_max=r_max,
        atomic_numbers=tuple(elements),
        num_interactions=config.model.num_layers,
        l_max=max(config.model.l_max, 1),
        num_features=config.model.num_features,
        avg_num_neighbors=_avg_neighbors(train_graphs),
        seed=config.seed,
        model_dtype=config.training.precision,
    )
    model = MACEDipoleModel(mace_cfg, config.parity).to(device)
    n_params = sum(int(p.numel()) for p in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    batch_size = config.training.batch_size

    def evaluate(graphs, targets):  # noqa: ANN001, ANN202
        preds = _predict(model, graphs, batch_size, dtype, device) * scale
        return regression_metrics(preds, targets)

    from equiparity.training.run_instrument import build_latest, build_timing, state_cpu

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train = time.perf_counter()
    ckpt_every = max(1, config.training.epochs // 10)
    best_val, checkpoint_best = float("inf"), None
    for epoch in range(config.training.epochs):
        model.train()
        offset = 0
        for batch in _batches(train_graphs, batch_size, dtype):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            n = int(batch["ptr"].shape[0]) - 1
            loss = loss_fn(model(batch).to(dtype), norm_targets[offset : offset + n])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            offset += n
        if (epoch + 1) % ckpt_every == 0 or epoch == config.training.epochs - 1:
            vm = evaluate(val_graphs, val_targets).mae
            if vm < best_val:
                best_val, checkpoint_best = vm, state_cpu(model)
    train_seconds = time.perf_counter() - t_train

    t_eval = time.perf_counter()
    val, test = evaluate(val_graphs, val_targets), evaluate(test_graphs, test_targets)
    eval_seconds = time.perf_counter() - t_eval
    return RunResult(
        val=val,
        test=test,
        n_params=n_params,
        epochs_run=config.training.epochs,
        timing=build_timing(
            train_seconds=train_seconds,
            eval_seconds=eval_seconds,
            epochs=config.training.epochs,
            n_train=len(train_graphs),
            device=device,
        ),
        checkpoint_best=checkpoint_best,
        checkpoint_latest=build_latest(model, optimizer, config.training.epochs),
    )
