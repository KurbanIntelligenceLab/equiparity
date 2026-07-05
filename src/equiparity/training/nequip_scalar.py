"""Train a NequIP model to predict a scalar target (QM9 U0) in one parity mode.

This is the Task 2.1 control path: the SO(3) and O(3) arms should reach the same U0 MAE
(a parity-even scalar has no parity gap). The model's scalar energy readout is the U0 head.
Targets are z-score normalized over the train split for stability; metrics are reported in the
original units. Full runs target the A100 cluster; this path also supports small smoke runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ase.data import chemical_symbols

from equiparity.domain.experiment import ExperimentConfig
from equiparity.evaluation.metrics import MetricSummary, regression_metrics
from equiparity.io.qm9_dataset import QM9Dataset, load_qm9, load_split
from equiparity.models.nequip import NequIPConfig, build_nequip_matched


@dataclass(frozen=True, slots=True)
class RunResult:
    """Outcome of a training run: per-split metrics in the target's units."""

    val: MetricSummary
    test: MetricSummary
    n_params: int
    epochs_run: int


def _type_map(z_values: np.ndarray) -> tuple[tuple[str, ...], dict[int, int]]:
    """Return (type_names, z->type-index) for the elements present in the data."""
    unique = sorted(int(z) for z in np.unique(z_values))
    type_names = tuple(chemical_symbols[z] for z in unique)
    return type_names, {z: i for i, z in enumerate(unique)}


def _to_atomic_data(structure, symbol_to_type, r_max, dtype):  # noqa: ANN001, ANN202
    from ase import Atoms
    from nequip.data import AtomicDataDict, compute_neighborlist_, from_ase

    atoms = Atoms(numbers=structure.atomic_numbers, positions=structure.positions)
    data = compute_neighborlist_(from_ase(atoms), r_max=r_max)
    data[AtomicDataDict.ATOM_TYPE_KEY] = torch.tensor(
        [[symbol_to_type[int(z)]] for z in structure.atomic_numbers], dtype=torch.long
    )
    for key, value in data.items():
        if torch.is_tensor(value) and value.is_floating_point():
            data[key] = value.to(dtype)
    return data


def _prepare(dataset, target, symbol_to_type, r_max, dtype):  # noqa: ANN001, ANN202
    graphs = [
        _to_atomic_data(dataset[i].structure, symbol_to_type, r_max, dtype)
        for i in range(len(dataset))
    ]
    targets = np.array([float(dataset[i].targets[target][0]) for i in range(len(dataset))])
    return graphs, targets


def _move(batch, device, dtype):  # noqa: ANN001, ANN202
    """Move a batched dict to device, casting floating tensors to the model dtype."""
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.is_floating_point():
            out[key] = value.to(device=device, dtype=dtype)
        elif torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _avg_num_neighbors(graphs) -> float:  # noqa: ANN001
    from nequip.data import AtomicDataDict

    edges = sum(int(g[AtomicDataDict.EDGE_INDEX_KEY].shape[1]) for g in graphs)
    atoms = sum(int(g[AtomicDataDict.POSITIONS_KEY].shape[0]) for g in graphs)
    return max(edges / max(atoms, 1), 1.0)


def _evaluate(model, graphs, targets, mean, std, device, dtype, batch_size) -> MetricSummary:  # noqa: ANN001
    from nequip.data import AtomicDataDict

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = _move(
                AtomicDataDict.batched_from_list(graphs[start : start + batch_size]), device, dtype
            )
            energy = model(batch)[AtomicDataDict.TOTAL_ENERGY_KEY].view(-1).cpu().numpy()
            preds.append(energy * std + mean)  # un-normalize to original units
    predictions = np.concatenate(preds).reshape(-1, 1)
    return regression_metrics(predictions, targets.reshape(-1, 1))


def train_scalar(config: ExperimentConfig) -> RunResult:
    """Train and evaluate a NequIP scalar model per ``config``. Returns per-split metrics."""
    from nequip.data import AtomicDataDict
    from nequip.utils.global_state import set_global_state

    torch.serialization.add_safe_globals([slice])
    set_global_state(allow_tf32=False)

    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    # nequip's default precision is float64; train in float64 to keep model buffers, inputs,
    # and targets aligned (avoids float32/float64 clashes in the backward pass).
    dtype = torch.float64

    data = load_qm9(config.processed_npz)
    type_names, symbol_to_type = _type_map(data.z)

    def subset(part: str, cap: int | None) -> QM9Dataset:
        ds = QM9Dataset(data, load_split(config.split_npz, part))
        if cap is not None:
            ds = QM9Dataset(
                data, np.array([int(ds[i].identifier) for i in range(min(cap, len(ds)))])
            )
        return ds

    train_ds = subset("train", config.training.max_train_samples)
    val_ds = subset("val", config.training.max_eval_samples)
    test_ds = subset("test", config.training.max_eval_samples)

    r_max = config.model.r_max
    train_graphs, train_targets = _prepare(train_ds, config.target, symbol_to_type, r_max, dtype)
    val_graphs, val_targets = _prepare(val_ds, config.target, symbol_to_type, r_max, dtype)
    test_graphs, test_targets = _prepare(test_ds, config.target, symbol_to_type, r_max, dtype)

    mean, std = float(train_targets.mean()), float(train_targets.std() or 1.0)
    norm_targets = torch.tensor((train_targets - mean) / std, dtype=dtype, device=device)

    model_cfg = NequIPConfig(
        r_max=r_max,
        type_names=type_names,
        num_layers=config.model.num_layers,
        l_max=config.model.l_max,
        num_features=config.model.num_features,
        type_embed_num_features=config.model.num_features,
        avg_num_neighbors=_avg_num_neighbors(train_graphs),
        seed=config.seed,
        model_dtype="float64",
    )
    model = build_nequip_matched(model_cfg, config.parity).to(device)
    n_params = sum(int(p.numel()) for p in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    batch_size = config.training.batch_size

    for _ in range(config.training.epochs):
        model.train()
        order = np.random.permutation(len(train_graphs))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = _move(
                AtomicDataDict.batched_from_list([train_graphs[i] for i in idx]), device, dtype
            )
            pred = model(batch)[AtomicDataDict.TOTAL_ENERGY_KEY].view(-1)
            loss = loss_fn(pred, norm_targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    val = _evaluate(model, val_graphs, val_targets, mean, std, device, dtype, batch_size)
    test = _evaluate(model, test_graphs, test_targets, mean, std, device, dtype, batch_size)
    return RunResult(val=val, test=test, n_params=n_params, epochs_run=config.training.epochs)
