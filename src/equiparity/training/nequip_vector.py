"""Train a NequIP model with a direct L=1 head to predict the QM9 dipole vector.

This is the first parity signal: the dipole is parity-odd. The O(3) arm reads out a ``1o`` polar
vector; the SO(3) arm reads out a ``1e`` (even-labelled) vector that cannot honour the odd-parity
structure. Targets are scaled (not mean-shifted) by their component std so the normalization stays
equivariant. Metric is the component MAE in Debye.
"""

from __future__ import annotations

import numpy as np
import torch

from equiparity.domain.experiment import ExperimentConfig
from equiparity.evaluation.metrics import regression_metrics
from equiparity.io.qm9_dataset import QM9Dataset, load_qm9, load_split
from equiparity.models.nequip import NequIPConfig, NequIPDipoleModel
from equiparity.training.nequip_data import (
    avg_num_neighbors,
    element_type_map,
    move_batch,
    to_atomic_data,
)
from equiparity.training.nequip_scalar import RunResult


def _prepare(dataset, target, symbol_to_type, r_max, dtype):  # noqa: ANN001, ANN202
    graphs = [
        to_atomic_data(dataset[i].structure, symbol_to_type, r_max, dtype)
        for i in range(len(dataset))
    ]
    targets = np.stack([dataset[i].targets[target] for i in range(len(dataset))]).astype(np.float64)
    return graphs, targets


def _evaluate(model, graphs, targets, scale, device, dtype, batch_size):  # noqa: ANN001, ANN202
    from nequip.data import AtomicDataDict

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = move_batch(
                AtomicDataDict.batched_from_list(graphs[start : start + batch_size]), device, dtype
            )
            vec = model(batch).cpu().numpy() * scale  # un-normalize to Debye
            preds.append(vec)
    predictions = np.concatenate(preds)
    return regression_metrics(predictions, targets)


def train_dipole(config: ExperimentConfig) -> RunResult:
    """Train and evaluate a NequIP dipole (L=1) model per ``config``. Returns per-split metrics."""
    from nequip.data import AtomicDataDict
    from nequip.utils.global_state import set_global_state

    torch.serialization.add_safe_globals([slice])
    set_global_state(allow_tf32=False)
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float64  # geometry stays float64 (nequip mixed precision)
    target_dtype = torch.float32 if config.training.precision == "float32" else torch.float64

    data = load_qm9(config.processed_npz)
    type_names, symbol_to_type = element_type_map(data.z)

    def subset(part: str, cap: int | None) -> QM9Dataset:
        ds = QM9Dataset(data, load_split(config.split_npz, part))
        if cap is not None:
            ds = QM9Dataset(
                data, np.array([int(ds[i].identifier) for i in range(min(cap, len(ds)))])
            )
        return ds

    r_max = config.model.r_max
    train_graphs, train_targets = _prepare(
        subset("train", config.training.max_train_samples),
        config.target,
        symbol_to_type,
        r_max,
        dtype,
    )
    val_graphs, val_targets = _prepare(
        subset("val", config.training.max_eval_samples), config.target, symbol_to_type, r_max, dtype
    )
    test_graphs, test_targets = _prepare(
        subset("test", config.training.max_eval_samples),
        config.target,
        symbol_to_type,
        r_max,
        dtype,
    )

    # Scale-only normalization keeps the target an equivariant vector (no mean shift).
    scale = float(train_targets.std()) or 1.0
    norm_targets = torch.tensor(train_targets / scale, dtype=target_dtype, device=device)

    avg_neigh = avg_num_neighbors(train_graphs)
    if config.core == "allegro":
        from equiparity.models.allegro import AllegroConfig, AllegroDipoleModel

        allegro_cfg = AllegroConfig(
            r_max=r_max,
            type_names=tuple(type_names),
            num_layers=config.model.num_layers,
            l_max=max(config.model.l_max, 1),
            num_scalar_features=config.model.num_features,
            num_tensor_features=max(8, config.model.num_features // 4),
            avg_num_neighbors=avg_neigh,
            seed=config.seed,
            model_dtype=config.training.precision,
            pooling=config.model.pooling,
        )
        model = AllegroDipoleModel(allegro_cfg, config.parity).to(device)
    else:
        model_cfg = NequIPConfig(
            r_max=r_max,
            type_names=type_names,
            num_layers=config.model.num_layers,
            l_max=max(config.model.l_max, 1),
            num_features=config.model.num_features,
            type_embed_num_features=config.model.num_features,
            avg_num_neighbors=avg_neigh,
            seed=config.seed,
            model_dtype=config.training.precision,
            pooling=config.model.pooling,
        )
        model = NequIPDipoleModel(model_cfg, config.parity).to(device)
    n_params = sum(int(p.numel()) for p in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    batch_size = config.training.batch_size

    import time

    from equiparity.training.run_instrument import build_latest, build_timing, state_cpu

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train = time.perf_counter()
    ckpt_every = max(1, config.training.epochs // 10)
    best_val, checkpoint_best = float("inf"), None
    for epoch in range(config.training.epochs):
        model.train()
        order = np.random.permutation(len(train_graphs))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = move_batch(
                AtomicDataDict.batched_from_list([train_graphs[i] for i in idx]), device, dtype
            )
            pred = model(batch).to(target_dtype)
            loss = loss_fn(pred, norm_targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % ckpt_every == 0 or epoch == config.training.epochs - 1:
            vm = _evaluate(model, val_graphs, val_targets, scale, device, dtype, batch_size).mae
            if vm < best_val:
                best_val, checkpoint_best = vm, state_cpu(model)
    train_seconds = time.perf_counter() - t_train

    t_eval = time.perf_counter()
    val = _evaluate(model, val_graphs, val_targets, scale, device, dtype, batch_size)
    test = _evaluate(model, test_graphs, test_targets, scale, device, dtype, batch_size)
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
