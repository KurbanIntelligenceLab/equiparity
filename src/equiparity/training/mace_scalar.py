"""Train a MACE model to predict a scalar target (QM9 U0) in one parity mode.

MACE uses its own data pipeline (Configuration + vendored torch_geometric) and energy readout,
distinct from the nequip framework. The O(3)/SO(3) toggle is MACE's native ``use_so3`` (all-even
SH), handled by build_mace_matched. Trained in float64.
"""

from __future__ import annotations

import numpy as np
import torch

from equiparity.domain.experiment import ExperimentConfig
from equiparity.evaluation.metrics import regression_metrics
from equiparity.io.qm9_dataset import QM9Dataset, load_qm9, load_split
from equiparity.models.mace import MACEConfig, build_mace_matched
from equiparity.training.nequip_scalar import RunResult


def _to_mace_data(structure, z_table, r_max):  # noqa: ANN001, ANN202
    from mace import data

    # Pass the cell + PBC for periodic crystals so MACE builds a PERIODIC neighborlist. Without
    # this MACE treats a crystal as a finite molecule, whose cut boundary breaks centrosymmetry
    # and leaks parity in the O(3) tensor head (nequip/allegro use PBC and cancel exactly).
    periodic = bool(getattr(structure, "pbc", False)) and structure.cell is not None
    cell_kwargs = (
        {"cell": np.asarray(structure.cell), "pbc": (True, True, True)} if periodic else {}
    )
    config = data.Configuration(
        atomic_numbers=np.asarray(structure.atomic_numbers),
        positions=structure.positions,
        properties={},
        property_weights={},
        **cell_kwargs,
    )
    return data.AtomicData.from_config(config, z_table=z_table, cutoff=r_max)


def _batches(atomic_data_list, batch_size, dtype):  # noqa: ANN001, ANN202
    from mace.tools import torch_geometric

    loader = torch_geometric.dataloader.DataLoader(
        dataset=atomic_data_list, batch_size=batch_size, shuffle=False
    )
    for batch in loader:
        d = batch.to_dict()
        for key, value in d.items():
            if torch.is_tensor(value) and value.is_floating_point():
                d[key] = value.to(dtype)
        yield d


def _energy(model, batch):  # noqa: ANN001, ANN202
    return model(batch, compute_force=False, compute_virials=False, compute_stress=False)["energy"]


def train_mace_scalar(config: ExperimentConfig) -> RunResult:
    """Train and evaluate a MACE scalar model per ``config``. Returns per-split metrics."""
    from mace import tools

    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    # MACE is natively float32; float32 is its fast default. float64 kept for verification runs.
    dtype = torch.float32 if config.training.precision == "float32" else torch.float64
    data = load_qm9(config.processed_npz)
    elements = sorted(int(z) for z in np.unique(data.z))
    z_table = tools.AtomicNumberTable(elements)
    r_max = config.model.r_max

    def subset(part: str, cap: int | None) -> QM9Dataset:
        ds = QM9Dataset(data, load_split(config.split_npz, part))
        if cap is not None:
            ds = QM9Dataset(
                data, np.array([int(ds[i].identifier) for i in range(min(cap, len(ds)))])
            )
        return ds

    def prepare(ds: QM9Dataset):  # noqa: ANN202
        graphs = [_to_mace_data(ds[i].structure, z_table, r_max) for i in range(len(ds))]
        targets = np.array([float(ds[i].targets[config.target][0]) for i in range(len(ds))])
        return graphs, targets

    train_graphs, train_targets = prepare(subset("train", config.training.max_train_samples))
    val_graphs, val_targets = prepare(subset("val", config.training.max_eval_samples))
    test_graphs, test_targets = prepare(subset("test", config.training.max_eval_samples))

    mean, std = float(train_targets.mean()), float(train_targets.std() or 1.0)
    norm_targets = torch.tensor((train_targets - mean) / std, dtype=dtype, device=device)

    mace_cfg = MACEConfig(
        r_max=r_max,
        atomic_numbers=tuple(elements),
        num_interactions=config.model.num_layers,
        l_max=config.model.l_max,
        num_features=config.model.num_features,
        seed=config.seed,
        model_dtype=config.training.precision,
    )
    model = build_mace_matched(mace_cfg, config.parity).to(device)
    n_params = sum(int(p.numel()) for p in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    batch_size = config.training.batch_size

    def evaluate(graphs, targets):  # noqa: ANN001, ANN202
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in _batches(graphs, batch_size, dtype):
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                preds.append(_energy(model, batch).view(-1).cpu().numpy() * std + mean)
        return regression_metrics(np.concatenate(preds).reshape(-1, 1), targets.reshape(-1, 1))

    import time

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
            pred = _energy(model, batch).view(-1)
            loss = loss_fn(pred, norm_targets[offset : offset + n])
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
