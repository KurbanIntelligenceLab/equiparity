"""CliffordSTF head: train the fixed O(3) geometric-algebra representative on piezoelectric.

CliffordSTF is O(3)-correct via graded geometric algebra (not e3nn). NOTE: it requires **float64** —
its geometric-product + STF chains accumulate float32 error that spoils the parity cancellation on
centrosymmetric crystals (float64 violation ~1e-9 vs float32 ~1e-1), unlike the e3nn cores which are
clean in float32. So the config should set ``precision: float64``.
"""

from __future__ import annotations

import numpy as np
import torch

from equiparity.domain.experiment import ExperimentConfig
from equiparity.domain.parity import ParityMode
from equiparity.domain.target import TARGETS
from equiparity.evaluation.metrics import MetricSummary, regression_metrics
from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset, load_split
from equiparity.models.clifford_head import (
    CliffordSTFConfig,
    CliffordSTFTensorModel,
    to_clifford_graph,
)
from equiparity.training.nequip_tensor import (
    TensorRunResult,
    _irreps_targets,
    false_flag_fraction,
    violation_magnitudes,
)


def _batch(structures, r_max, dtype, device):  # noqa: ANN001, ANN202
    graphs = [to_clifford_graph(s, r_max, dtype) for s in structures]
    edge_index, batch_idx, offset = [], [], 0
    for k, g in enumerate(graphs):
        edge_index.append(g["edge_index"] + offset)
        batch_idx.append(torch.full((g["n_atoms"],), k, dtype=torch.long))
        offset += g["n_atoms"]
    return {
        "atomic_numbers": torch.cat([g["atomic_numbers"] for g in graphs]).to(device),
        "pos": torch.cat([g["pos"] for g in graphs]).to(device),
        "edge_index": torch.cat(edge_index, dim=1).to(device),
        "edge_vec": torch.cat([g["edge_vec"] for g in graphs]).to(device),
        "batch": torch.cat(batch_idx).to(device),
    }


def _batches(structures, r_max, batch_size, dtype, device):  # noqa: ANN001, ANN202
    for start in range(0, len(structures), batch_size):
        yield _batch(structures[start : start + batch_size], r_max, dtype, device)


def _predict(model, structures, r_max, batch_size, dtype, device):  # noqa: ANN001, ANN202
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in _batches(structures, r_max, batch_size, dtype, device):
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


def train_clifford_tensor(
    config: ExperimentConfig, *, ood_npz: str | None = None
) -> TensorRunResult:
    """Train the CliffordSTF piezoelectric head (float64), eval + OOD violation."""
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if config.training.precision == "float64" else torch.float32
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
    norm_targets = torch.tensor(train_targets / scale, dtype=dtype, device=device)

    cfg = CliffordSTFConfig(
        r_max=r_max,
        n_channels=config.model.num_features,
        n_interactions=config.model.num_layers,
        seed=config.seed,
    )
    model = CliffordSTFTensorModel(cfg, ParityMode.O3, TARGETS[config.target].irreps)
    model = model.to(device).to(dtype)
    n_params = sum(int(p.numel()) for p in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    batch_size = config.training.batch_size

    for _ in range(config.training.epochs):
        model.train()
        offset = 0
        for batch in _batches(train_structs, r_max, batch_size, dtype, device):
            n = int(batch["batch"].max().item()) + 1
            loss = loss_fn(model(batch).to(dtype), norm_targets[offset : offset + n])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            offset += n

    def evaluate(ds: CrystalDataset) -> MetricSummary:
        structs = [ds[i].structure for i in range(len(ds))]
        preds = _predict(model, structs, r_max, batch_size, dtype, device) * scale
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
        ood_preds = _predict(model, structs, r_max, batch_size, dtype, device) * scale
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
