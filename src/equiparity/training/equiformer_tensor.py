"""EquiformerV2 heads: train the fixed SO(3) representative on tensor/dipole/scalar targets.

EquiformerV2 uses a torch_geometric data pipeline (PBC-correct edges via ``to_pyg_data``), so it
needs its own trainer. It is SO(3)-only: ``parity`` is fixed to SO(3), and the point is that it
VIOLATES parity on centrosymmetric crystals (nonzero piezo). Trains in float32 (native).
"""

from __future__ import annotations

import numpy as np
import torch

from equiparity.domain.experiment import ExperimentConfig
from equiparity.domain.parity import ParityMode
from equiparity.domain.target import TARGETS
from equiparity.evaluation.metrics import MetricSummary, regression_metrics
from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset, load_split
from equiparity.io.qm9_dataset import QM9Dataset, load_qm9
from equiparity.io.qm9_dataset import load_split as load_qm9_split
from equiparity.models.equiformer import (
    EquiformerV2Config,
    EquiformerV2DipoleModel,
    EquiformerV2TensorModel,
    to_pyg_data,
)
from equiparity.training.nequip_scalar import RunResult
from equiparity.training.nequip_tensor import (
    TensorRunResult,
    _irreps_targets,
    false_flag_fraction,
    violation_magnitudes,
)


def _config(config: ExperimentConfig) -> EquiformerV2Config:
    return EquiformerV2Config(
        r_max=config.model.r_max,
        lmax=config.model.l_max,
        num_layers=config.model.num_layers,
        sphere_channels=config.model.num_features,
        attn_hidden_channels=max(16, config.model.num_features // 2),
        ffn_hidden_channels=config.model.num_features,
        num_heads=4,
        edge_channels=max(16, config.model.num_features // 2),
        seed=config.seed,
    )


def _batched(structures, r_max, batch_size, device):  # noqa: ANN001, ANN202
    from torch_geometric.data import Batch

    for start in range(0, len(structures), batch_size):
        graphs = [to_pyg_data(s, r_max) for s in structures[start : start + batch_size]]
        yield Batch.from_data_list(graphs).to(device)


def _predict(model, structures, r_max, batch_size, device):  # noqa: ANN001, ANN202
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in _batched(structures, r_max, batch_size, device):
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


def train_equiformer_tensor(
    config: ExperimentConfig, *, ood_npz: str | None = None
) -> TensorRunResult:
    """Train an EquiformerV2 tensor head (elastic/piezoelectric), eval + OOD violation."""
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    r_max = config.model.r_max
    kind = config.target
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
    train_structs = [train_ds[i].structure for i in range(len(train_ds))]
    train_targets = _irreps_targets(train_ds, config.target, kind)
    scale = float(train_targets.std()) or 1.0
    norm_targets = torch.tensor(train_targets / scale, dtype=torch.float32, device=device)

    model = EquiformerV2TensorModel(_config(config), ParityMode.SO3, TARGETS[config.target].irreps)
    model = model.to(device)
    n_params = sum(int(p.numel()) for p in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    batch_size = config.training.batch_size

    for _ in range(config.training.epochs):
        model.train()
        offset = 0
        for batch in _batched(train_structs, r_max, batch_size, device):
            n = int(batch.natoms.shape[0])
            pred = model(batch).to(torch.float32)
            loss = loss_fn(pred, norm_targets[offset : offset + n])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            offset += n

    def evaluate(ds: CrystalDataset) -> MetricSummary:
        structs = [ds[i].structure for i in range(len(ds))]
        preds = _predict(model, structs, r_max, batch_size, device) * scale
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
        structs = [ood[i].structure for i in range(len(ood))]
        ood_preds = _predict(model, structs, r_max, batch_size, device) * scale
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


def _qm9_subset(config: ExperimentConfig, part: str, cap: int | None):  # noqa: ANN202
    data = load_qm9(config.processed_npz)
    ds = QM9Dataset(data, load_qm9_split(config.split_npz, part))
    if cap is not None:
        ds = QM9Dataset(data, np.array([int(ds[i].identifier) for i in range(min(cap, len(ds)))]))
    return ds


def train_equiformer_dipole(config: ExperimentConfig) -> RunResult:
    """Train an EquiformerV2 L=1 dipole head on QM9 (scale-only normalization)."""
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    r_max = config.model.r_max
    batch_size = config.training.batch_size

    def prep(part, cap):  # noqa: ANN001, ANN202
        ds = _qm9_subset(config, part, cap)
        structs = [ds[i].structure for i in range(len(ds))]
        targets = np.stack(
            [
                np.asarray(ds[i].targets[config.target], dtype=np.float64).reshape(3)
                for i in range(len(ds))
            ]
        )
        return structs, targets

    train_structs, train_targets = prep("train", config.training.max_train_samples)
    scale = float(train_targets.std()) or 1.0
    norm_targets = torch.tensor(train_targets / scale, dtype=torch.float32, device=device)

    model = EquiformerV2DipoleModel(_config(config), ParityMode.SO3).to(device)
    n_params = sum(int(p.numel()) for p in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    for _ in range(config.training.epochs):
        model.train()
        offset = 0
        for batch in _batched(train_structs, r_max, batch_size, device):
            n = int(batch.natoms.shape[0])
            loss = loss_fn(model(batch).to(torch.float32), norm_targets[offset : offset + n])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            offset += n

    def evaluate(part):  # noqa: ANN001, ANN202
        structs, targets = prep(part, config.training.max_eval_samples)
        preds = _predict(model, structs, r_max, batch_size, device) * scale
        return regression_metrics(preds, targets)

    return RunResult(
        val=evaluate("val"),
        test=evaluate("test"),
        n_params=n_params,
        epochs_run=config.training.epochs,
    )


def train_equiformer_scalar(config: ExperimentConfig) -> RunResult:
    """Train an EquiformerV2 scalar head on QM9 U0 (mean/std normalization)."""
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    r_max = config.model.r_max
    batch_size = config.training.batch_size

    def prep(part, cap):  # noqa: ANN001, ANN202
        ds = _qm9_subset(config, part, cap)
        structs = [ds[i].structure for i in range(len(ds))]
        targets = np.array([float(ds[i].targets[config.target][0]) for i in range(len(ds))])
        return structs, targets

    train_structs, train_targets = prep("train", config.training.max_train_samples)
    mean, std = float(train_targets.mean()), float(train_targets.std() or 1.0)
    norm_targets = torch.tensor((train_targets - mean) / std, dtype=torch.float32, device=device)

    model = EquiformerV2TensorModel(_config(config), ParityMode.SO3, "1x0e").to(device)
    n_params = sum(int(p.numel()) for p in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    for _ in range(config.training.epochs):
        model.train()
        offset = 0
        for batch in _batched(train_structs, r_max, batch_size, device):
            n = int(batch.natoms.shape[0])
            pred = model(batch).view(-1).to(torch.float32)
            loss = loss_fn(pred, norm_targets[offset : offset + n])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            offset += n

    def evaluate(part):  # noqa: ANN001, ANN202
        structs, targets = prep(part, config.training.max_eval_samples)
        preds = _predict(model, structs, r_max, batch_size, device).reshape(-1) * std + mean
        return regression_metrics(preds.reshape(-1, 1), targets.reshape(-1, 1))

    return RunResult(
        val=evaluate("val"),
        test=evaluate("test"),
        n_params=n_params,
        epochs_run=config.training.epochs,
    )
