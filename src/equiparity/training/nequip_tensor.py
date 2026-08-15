"""Tensor-head prediction and the piezoelectric OOD violation metric.

The headline (Task 2.3): a model trained on non-centrosymmetric piezoelectric crystals is
evaluated on centrosymmetric ones, whose true tensor is exactly zero by symmetry. An O(3) model
with an odd-parity head predicts exact zero there (by construction); an SO(3) model predicts a
spurious nonzero tensor. The violation magnitude is ``||predicted tensor||`` per structure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ase.data import chemical_symbols

from equiparity.domain.experiment import ExperimentConfig
from equiparity.domain.target import TARGETS
from equiparity.evaluation.metrics import MetricSummary, regression_metrics
from equiparity.features.tensor_irreps import voigt_to_irreps
from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset, load_split
from equiparity.models.nequip import NequIPConfig, NequIPTensorModel
from equiparity.training.nequip_data import (
    avg_num_neighbors,
    element_type_map,
    move_batch,
    to_atomic_data,
)
from equiparity.training.ood_eval import evaluate_ood_variants

_VOIGT_SHAPE = {"piezoelectric": (3, 6), "elastic": (6, 6)}


def _raw_variant(ood_npz: str) -> str | None:
    """Sibling raw OOD npz: ``..._processed.npz`` -> ``..._processed_raw.npz`` (None if absent)."""
    p = Path(ood_npz)
    raw = p.with_name(p.stem + "_raw" + p.suffix)
    return str(raw) if raw.exists() else None


def periodic_type_map(max_z: int = 100) -> tuple[tuple[str, ...], dict[int, int]]:
    """Fixed type map over the first ``max_z`` elements (covers train and OOD elements alike)."""
    type_names = tuple(chemical_symbols[z] for z in range(1, max_z + 1))
    return type_names, {z: z - 1 for z in range(1, max_z + 1)}


def predict_tensors(
    model,  # noqa: ANN001
    dataset: CrystalDataset,
    symbol_to_type: dict[int, int],
    *,
    r_max: float,
    device,  # noqa: ANN001
    dtype,  # noqa: ANN001
    batch_size: int = 16,
) -> np.ndarray:
    """Return the model's per-structure tensor predictions (in irreps form), shape (n, dim)."""
    from nequip.data import AtomicDataDict

    model.eval()
    graphs = [
        to_atomic_data(dataset[i].structure, symbol_to_type, r_max, dtype)
        for i in range(len(dataset))
    ]
    preds = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = move_batch(
                AtomicDataDict.batched_from_list(graphs[start : start + batch_size]), device, dtype
            )
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


def violation_magnitudes(predictions: np.ndarray) -> np.ndarray:
    """Frobenius norm of each predicted tensor, shape (n,).

    On centrosymmetric structures the true value is zero, so this is the violation magnitude:
    ~0 for a correct O(3) model, nonzero for an SO(3) model.
    """
    return np.sqrt((predictions**2).sum(axis=1))


def false_flag_fraction(magnitudes: np.ndarray, threshold: float) -> float:
    """Fraction of structures whose violation magnitude exceeds a materials-relevance threshold."""
    if magnitudes.size == 0:
        return 0.0
    return float((magnitudes > threshold).mean())


def dataset_type_map(dataset: CrystalDataset) -> tuple[tuple[str, ...], dict[int, int]]:
    """Build (type_names, z->index) covering every element in a crystal dataset."""
    z_values = np.concatenate([dataset[i].structure.atomic_numbers for i in range(len(dataset))])
    return element_type_map(z_values)


@dataclass(frozen=True, slots=True)
class TensorRunResult:
    """Outcome of a tensor training run, plus optional OOD violation stats (piezoelectric)."""

    val: MetricSummary
    test: MetricSummary
    n_params: int
    epochs_run: int
    ood_violation_median: float | None = None
    ood_violation_max: float | None = None
    ood_false_flag_fraction: float | None = None
    # Reviewer instrumentation (Checkpoint 7): both OOD variants w/ threshold curves + distros,
    # per-structure vectors (offline histograms), wall-clock timing, and checkpoints.
    ood_variants: dict[str, object] | None = None  # {variant: violation_stats(...)}
    ood_vectors: dict[str, object] | None = None  # {variant: np.ndarray of magnitudes}
    # H-1 instrumentation: idealized-variant false-flag fraction and violation median per epoch.
    ood_false_flag_history: list[dict[str, float]] | None = None
    timing: dict[str, float] | None = None  # train/eval/ood seconds, throughput, peak mem
    checkpoint_best: dict[str, object] | None = None  # best-val model state_dict
    checkpoint_latest: dict[str, object] | None = None  # model+optimizer+epoch (resumable)


def _irreps_targets(dataset: CrystalDataset, target: str, kind: str) -> np.ndarray:
    shape = _VOIGT_SHAPE[kind]
    return np.stack(
        [
            voigt_to_irreps(dataset[i].targets[target].reshape(shape), kind)
            for i in range(len(dataset))
        ]
    ).astype(np.float64)


def build_tensor_model(config, type_names, avg_neigh, o3_output_irreps, device):  # noqa: ANN001, ANN201
    """Build a tensor-head model for the ``nequip`` or ``allegro`` core (shared data pipeline)."""
    if config.core == "allegro":
        from equiparity.models.allegro import AllegroConfig, AllegroTensorModel

        cfg = AllegroConfig(
            r_max=config.model.r_max,
            type_names=tuple(type_names),
            num_layers=config.model.num_layers,
            l_max=config.model.l_max,
            num_scalar_features=config.model.num_features,
            num_tensor_features=max(8, config.model.num_features // 4),
            avg_num_neighbors=avg_neigh,
            seed=config.seed,
            model_dtype=config.training.precision,
            pooling=config.model.pooling,
        )
        return AllegroTensorModel(cfg, config.parity, o3_output_irreps).to(device)
    cfg = NequIPConfig(
        r_max=config.model.r_max,
        type_names=type_names,
        num_layers=config.model.num_layers,
        l_max=config.model.l_max,
        num_features=config.model.num_features,
        type_embed_num_features=config.model.num_features,
        avg_num_neighbors=avg_neigh,
        seed=config.seed,
        model_dtype=config.training.precision,
        pooling=config.model.pooling,
    )
    return NequIPTensorModel(cfg, config.parity, o3_output_irreps).to(device)


def train_tensor(config: ExperimentConfig, *, ood_npz: str | None = None) -> TensorRunResult:
    """Train a NequIP/Allegro tensor head (elastic or piezoelectric) and evaluate it.

    For the piezoelectric headline, pass ``ood_npz`` to also evaluate the violation magnitude on
    the centrosymmetric OOD set (exactly zero for O(3), spurious nonzero for SO(3)).
    """
    from nequip.data import AtomicDataDict
    from nequip.utils.global_state import set_global_state

    torch.serialization.add_safe_globals([slice])
    set_global_state(allow_tf32=False)
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float64  # geometry stays float64 (nequip mixed precision)
    target_dtype = torch.float32 if config.training.precision == "float32" else torch.float64
    kind = config.target
    r_max = config.model.r_max
    type_names, z_map = periodic_type_map()

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

    def graphs_of(ds: CrystalDataset) -> list:
        return [to_atomic_data(ds[i].structure, z_map, r_max, dtype) for i in range(len(ds))]

    train_graphs = graphs_of(train_ds)
    train_targets = _irreps_targets(train_ds, config.target, kind)
    scale = config.training.target_scale or float(train_targets.std()) or 1.0
    norm_targets = torch.tensor(train_targets / scale, dtype=target_dtype, device=device)

    model = build_tensor_model(
        config, type_names, avg_num_neighbors(train_graphs), TARGETS[config.target].irreps, device
    )
    n_params = sum(int(p.numel()) for p in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    # Per-row loss weight: exactly-zero-target rows get zero_row_loss_weight, all others 1. At the
    # default weight 1.0 this is identical to torch.nn.MSELoss (verified in tests). H3 raises it.
    zero_mask = np.abs(train_targets).max(axis=1) == 0.0
    row_weight = torch.ones(len(train_targets), dtype=target_dtype, device=device)
    row_weight[torch.tensor(zero_mask, device=device)] = config.training.zero_row_loss_weight

    def weighted_mse(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (w[:, None] * (pred - target) ** 2).mean()

    batch_size = config.training.batch_size

    def evaluate(ds: CrystalDataset) -> MetricSummary:
        preds = predict_tensors(
            model, ds, z_map, r_max=r_max, device=device, dtype=dtype, batch_size=batch_size
        )
        preds = preds * scale
        targets = _irreps_targets(ds, config.target, kind)
        return regression_metrics(preds, targets)

    # H-1 instrumentation: per-epoch false-flag fraction on the idealized OOD variant. One
    # forward pass over the evaluation population per epoch; piezoelectric runs only.
    epoch_ood_ds = None
    if ood_npz is not None and kind == "piezoelectric":
        epoch_ood_ds = CrystalDataset(load_crystal_dataset(ood_npz))
        if config.training.max_eval_samples is not None:
            cap = min(config.training.max_eval_samples, len(epoch_ood_ds))
            ids = np.array([epoch_ood_ds[i].identifier for i in range(cap)])
            epoch_ood_ds = CrystalDataset(load_crystal_dataset(ood_npz), ids)
    ood_history: list[dict[str, float]] | None = None if epoch_ood_ds is None else []

    # ---- train with timing + best-val checkpoint tracking ----
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
            loss = weighted_mse(model(batch).to(target_dtype), norm_targets[idx], row_weight[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % ckpt_every == 0 or epoch == config.training.epochs - 1:
            vm = evaluate(val_ds).mae
            if vm < best_val:
                best_val = vm
                checkpoint_best = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        if epoch_ood_ds is not None and ood_history is not None:
            v = violation_magnitudes(
                predict_tensors(
                    model,
                    epoch_ood_ds,
                    z_map,
                    r_max=r_max,
                    device=device,
                    dtype=dtype,
                    batch_size=batch_size,
                )
                * scale
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

    # ---- OOD: evaluate on BOTH idealized + raw variants, keep threshold curves + vectors ----
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
            preds = (
                predict_tensors(
                    model,
                    ood,
                    z_map,
                    r_max=r_max,
                    device=device,
                    dtype=dtype,
                    batch_size=batch_size,
                )
                * scale
            )
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
    n_train = len(train_graphs)
    timing = {
        "train_seconds": train_seconds,
        "train_seconds_per_epoch": train_seconds / max(1, config.training.epochs),
        "eval_seconds": eval_seconds,
        "ood_seconds": ood_seconds,
        "train_throughput_structs_per_s": n_train
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
