"""MACE tensor and dipole heads: train ``MACETensorModel`` / ``MACEDipoleModel``.

MACE uses its own data pipeline (Configuration + vendored torch_geometric), so unlike the
nequip/allegro cores it needs a dedicated tensor trainer. The head reads MACE's per-node
equivariant features and sums them per structure (:mod:`equiparity.models.mace`). Trains in
float32 mixed precision by default (MACE is natively float32).
"""

from __future__ import annotations

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
    false_flag_fraction,
    violation_magnitudes,
)


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
    scale = float(train_targets.std()) or 1.0
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

    for _ in range(config.training.epochs):
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

    def evaluate(ds: CrystalDataset) -> MetricSummary:
        preds = _predict(model, graphs_of(ds), batch_size, dtype, device) * scale
        return regression_metrics(preds, _irreps_targets(ds, config.target, kind))

    val, test = evaluate(val_ds), evaluate(test_ds)

    ood_median = ood_max = ood_ff = None
    if ood_npz is not None and kind == "piezoelectric":
        ood = CrystalDataset(load_crystal_dataset(ood_npz))
        if config.training.max_eval_samples is not None:
            ids = np.array(
                [ood[i].identifier for i in range(min(config.training.max_eval_samples, len(ood)))]
            )
            ood = CrystalDataset(load_crystal_dataset(ood_npz), ids)
        ood_preds = _predict(model, graphs_of(ood), batch_size, dtype, device) * scale
        v = violation_magnitudes(ood_preds)
        ood_median, ood_max = float(np.median(v)), float(v.max())
        ood_ff = false_flag_fraction(v, threshold=0.01)

    return TensorRunResult(
        val=val,
        test=test,
        n_params=n_params,
        epochs_run=config.training.epochs,
        ood_violation_median=ood_median,
        ood_violation_max=ood_max,
        ood_false_flag_fraction=ood_ff,
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

    scale = float(train_targets.std()) or 1.0
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

    for _ in range(config.training.epochs):
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

    def evaluate(graphs, targets):  # noqa: ANN001, ANN202
        preds = _predict(model, graphs, batch_size, dtype, device) * scale
        return regression_metrics(preds, targets)

    return RunResult(
        val=evaluate(val_graphs, val_targets),
        test=evaluate(test_graphs, test_targets),
        n_params=n_params,
        epochs_run=config.training.epochs,
    )
