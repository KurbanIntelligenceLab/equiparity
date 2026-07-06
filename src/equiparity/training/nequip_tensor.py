"""Tensor-head prediction and the piezoelectric OOD violation metric.

The headline (Task 2.3): a model trained on non-centrosymmetric piezoelectric crystals is
evaluated on centrosymmetric ones, whose true tensor is exactly zero by symmetry. An O(3) model
with an odd-parity head predicts exact zero there (by construction); an SO(3) model predicts a
spurious nonzero tensor. The violation magnitude is ``||predicted tensor||`` per structure.
"""

from __future__ import annotations

from dataclasses import dataclass

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

_VOIGT_SHAPE = {"piezoelectric": (3, 6), "elastic": (6, 6)}


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
    scale = float(train_targets.std()) or 1.0
    norm_targets = torch.tensor(train_targets / scale, dtype=target_dtype, device=device)

    model = build_tensor_model(
        config, type_names, avg_num_neighbors(train_graphs), TARGETS[config.target].irreps, device
    )
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
            batch = move_batch(
                AtomicDataDict.batched_from_list([train_graphs[i] for i in idx]), device, dtype
            )
            loss = loss_fn(model(batch).to(target_dtype), norm_targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def evaluate(ds: CrystalDataset) -> MetricSummary:
        preds = predict_tensors(
            model, ds, z_map, r_max=r_max, device=device, dtype=dtype, batch_size=batch_size
        )
        preds = preds * scale
        targets = _irreps_targets(ds, config.target, kind)
        return regression_metrics(preds, targets)

    val, test = evaluate(val_ds), evaluate(test_ds)

    ood_median = ood_max = ood_ff = None
    if ood_npz is not None and kind == "piezoelectric":
        ood = CrystalDataset(load_crystal_dataset(ood_npz))
        if config.training.max_eval_samples is not None:
            ids = np.array(
                [ood[i].identifier for i in range(min(config.training.max_eval_samples, len(ood)))]
            )
            ood = CrystalDataset(load_crystal_dataset(ood_npz), ids)
        ood_preds = (
            predict_tensors(
                model, ood, z_map, r_max=r_max, device=device, dtype=dtype, batch_size=batch_size
            )
            * scale
        )
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
