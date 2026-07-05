"""Tensor-head prediction and the piezoelectric OOD violation metric.

The headline (Task 2.3): a model trained on non-centrosymmetric piezoelectric crystals is
evaluated on centrosymmetric ones, whose true tensor is exactly zero by symmetry. An O(3) model
with an odd-parity head predicts exact zero there (by construction); an SO(3) model predicts a
spurious nonzero tensor. The violation magnitude is ``||predicted tensor||`` per structure.
"""

from __future__ import annotations

import numpy as np
import torch

from equiparity.io.mp_dataset import CrystalDataset
from equiparity.training.nequip_data import element_type_map, move_batch, to_atomic_data


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
