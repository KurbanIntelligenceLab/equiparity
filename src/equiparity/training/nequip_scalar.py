"""Train a NequIP model to predict a scalar target (QM9 U0) in one parity mode.

This is the control path: the SO(3) and O(3) arms should reach the same U0 MAE, because a
parity-even scalar has no parity gap. A gap here would mean the matched pair differs in something
other than parity labelling, and invalidates the comparison on every other target. The model's
scalar energy readout is the U0 head. Targets are z-score normalized over the train split for
stability; metrics are reported in the original units. Full runs target a GPU; this path also runs
at reduced size for a quick check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from equiparity.domain.experiment import ExperimentConfig
from equiparity.evaluation.metrics import MetricSummary, regression_metrics
from equiparity.io.qm9_dataset import QM9Dataset, load_qm9, load_split
from equiparity.training.nequip_data import (
    avg_num_neighbors,
    element_type_map,
    move_batch,
    to_atomic_data,
)


def build_energy_model(config: ExperimentConfig, type_names, avg_neigh, mode):  # noqa: ANN001, ANN201
    """Build the energy-readout model for a nequip-framework core (nequip or allegro)."""
    m = config.model
    if config.core == "nequip":
        from equiparity.models.nequip import NequIPConfig, build_nequip_matched

        cfg = NequIPConfig(
            r_max=m.r_max,
            type_names=type_names,
            num_layers=m.num_layers,
            l_max=m.l_max,
            num_features=m.num_features,
            type_embed_num_features=m.num_features,
            avg_num_neighbors=avg_neigh,
            seed=config.seed,
            model_dtype=config.training.precision,
        )
        return build_nequip_matched(cfg, mode)
    if config.core == "allegro":
        from equiparity.models.allegro import AllegroConfig, build_allegro_matched

        cfg = AllegroConfig(
            r_max=m.r_max,
            type_names=type_names,
            num_layers=m.num_layers,
            l_max=m.l_max,
            num_scalar_features=m.num_features,
            num_tensor_features=max(m.num_features // 2, 1),
            avg_num_neighbors=avg_neigh,
            seed=config.seed,
            model_dtype=config.training.precision,
        )
        return build_allegro_matched(cfg, mode)
    raise NotImplementedError(f"energy model not wired for core {config.core!r}")


@dataclass(frozen=True, slots=True)
class RunResult:
    """Outcome of a training run: per-split metrics in the target's units."""

    val: MetricSummary
    test: MetricSummary
    n_params: int
    epochs_run: int
    # Run instrumentation: wall-clock timing + resumable/best checkpoints.
    timing: dict[str, float] | None = None
    checkpoint_best: dict[str, object] | None = None
    checkpoint_latest: dict[str, object] | None = None


def _prepare(dataset, target, symbol_to_type, r_max, dtype):  # noqa: ANN001, ANN202
    graphs = [
        to_atomic_data(dataset[i].structure, symbol_to_type, r_max, dtype)
        for i in range(len(dataset))
    ]
    targets = np.array([float(dataset[i].targets[target][0]) for i in range(len(dataset))])
    return graphs, targets


def _evaluate(model, graphs, targets, mean, std, device, dtype, batch_size) -> MetricSummary:  # noqa: ANN001
    from nequip.data import AtomicDataDict

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = move_batch(
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
    # Geometry (positions/edges) stays float64 — nequip's mixed-precision design. The model
    # weights/compute run in `precision` (float32 by default; ~6.8x faster on consumer GPUs) and
    # the energy output is cast to `target_dtype` to match the targets/loss.
    dtype = torch.float64
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

    train_ds = subset("train", config.training.max_train_samples)
    val_ds = subset("val", config.training.max_eval_samples)
    test_ds = subset("test", config.training.max_eval_samples)

    r_max = config.model.r_max
    train_graphs, train_targets = _prepare(train_ds, config.target, symbol_to_type, r_max, dtype)
    val_graphs, val_targets = _prepare(val_ds, config.target, symbol_to_type, r_max, dtype)
    test_graphs, test_targets = _prepare(test_ds, config.target, symbol_to_type, r_max, dtype)

    mean, std = float(train_targets.mean()), float(train_targets.std() or 1.0)
    norm_targets = torch.tensor((train_targets - mean) / std, dtype=target_dtype, device=device)

    model = build_energy_model(
        config, type_names, avg_num_neighbors(train_graphs), config.parity
    ).to(device)
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
            pred = model(batch)[AtomicDataDict.TOTAL_ENERGY_KEY].view(-1).to(target_dtype)
            loss = loss_fn(pred, norm_targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % ckpt_every == 0 or epoch == config.training.epochs - 1:
            vm = _evaluate(model, val_graphs, val_targets, mean, std, device, dtype, batch_size).mae
            if vm < best_val:
                best_val, checkpoint_best = vm, state_cpu(model)
    train_seconds = time.perf_counter() - t_train

    t_eval = time.perf_counter()
    val = _evaluate(model, val_graphs, val_targets, mean, std, device, dtype, batch_size)
    test = _evaluate(model, test_graphs, test_targets, mean, std, device, dtype, batch_size)
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
