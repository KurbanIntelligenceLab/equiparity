"""The piezoelectric OOD violation property (requires nequip + MP OOD data; skipped otherwise).

The headline structural result: an O(3) model with an odd-parity piezoelectric head predicts
exactly zero on centrosymmetric crystals BY CONSTRUCTION (even untrained), while an SO(3) model
predicts spurious nonzero tensors. This holds before any training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nequip")

_OOD = Path("data/raw/mp/mp_ood_centrosymmetric_processed.npz")
if not _OOD.exists():
    pytest.skip("MP OOD data not present; run scripts/prepare_mp.py ood", allow_module_level=True)

import torch  # noqa: E402
from nequip.utils.global_state import set_global_state  # noqa: E402

from equiparity.domain.parity import ParityMode  # noqa: E402
from equiparity.domain.target import PIEZOELECTRIC  # noqa: E402
from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset  # noqa: E402
from equiparity.models.nequip import NequIPConfig, NequIPTensorModel  # noqa: E402
from equiparity.training.nequip_tensor import (  # noqa: E402
    dataset_type_map,
    predict_tensors,
    violation_magnitudes,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _violations(mode: ParityMode) -> np.ndarray:
    set_global_state(allow_tf32=False)
    full = CrystalDataset(load_crystal_dataset(_OOD))
    idxs = [
        i
        for i in range(len(full))
        if len(set(int(z) for z in full[i].structure.atomic_numbers)) <= 4
        and full[i].structure.n_atoms <= 40
    ][:8]
    subset = CrystalDataset(
        load_crystal_dataset(_OOD), np.array([full[i].identifier for i in idxs])
    )
    type_names, z_map = dataset_type_map(subset)
    torch.manual_seed(0)
    config = NequIPConfig(
        r_max=5.0,
        type_names=type_names,
        num_layers=3,
        l_max=3,
        num_features=12,
        type_embed_num_features=12,
        avg_num_neighbors=30.0,
        model_dtype="float64",
    )
    model = NequIPTensorModel(config, mode, PIEZOELECTRIC.irreps)
    preds = predict_tensors(
        model,
        subset,
        z_map,
        r_max=5.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
        batch_size=8,
    )
    return violation_magnitudes(preds)


def test_o3_predicts_exact_zero_on_centrosymmetric() -> None:
    # Odd-parity output on a centrosymmetric structure cancels to machine zero, untrained.
    assert _violations(ParityMode.O3).max() < 1e-10


def test_so3_predicts_nonzero_on_centrosymmetric() -> None:
    # No parity constraint -> spurious nonzero piezoelectric tensor.
    assert np.median(_violations(ParityMode.SO3)) > 1e-6
