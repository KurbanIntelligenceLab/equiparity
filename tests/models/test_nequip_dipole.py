"""The NequIP dipole head has the right output parity per mode (requires nequip).

O(3): the dipole is a proper polar vector (1o) and satisfies dipole(Mx) = M.dipole(x) under a
reflection M. SO(3): the dipole is even-labelled (1e) and violates this. This is the dipole
parity signal, guaranteed by construction (independent of training).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("nequip")

import torch
from ase import Atoms
from nequip.data import AtomicDataDict, compute_neighborlist_, from_ase
from nequip.utils.global_state import set_global_state

from equiparity.domain.parity import ParityMode
from equiparity.models.nequip import NequIPConfig, NequIPDipoleModel

pytestmark = pytest.mark.integration

_REFLECTION = np.diag([-1.0, 1.0, 1.0])
_Z = [8, 1, 1, 6]
_POS = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.3], [0.0, 1.1, 0.0], [0.0, 0.0, 1.2]])
_TYPE = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}


def _batch(positions: np.ndarray):  # noqa: ANN202
    atoms = Atoms(numbers=_Z, positions=positions)
    data = compute_neighborlist_(from_ase(atoms), r_max=5.0)
    data[AtomicDataDict.ATOM_TYPE_KEY] = torch.tensor([[_TYPE[z]] for z in _Z])
    for key, value in data.items():
        if torch.is_tensor(value) and value.is_floating_point():
            data[key] = value.double()
    return AtomicDataDict.batched_from_list([data])


def _reflection_error(mode: ParityMode) -> float:
    set_global_state(allow_tf32=False)
    torch.manual_seed(0)
    config = NequIPConfig(
        r_max=5.0,
        type_names=("H", "C", "N", "O", "F"),
        num_layers=3,
        l_max=1,
        num_features=16,
        type_embed_num_features=16,
        avg_num_neighbors=20.0,
        model_dtype="float64",
    )
    model = NequIPDipoleModel(config, mode).double().eval()
    with torch.no_grad():
        base = model(_batch(_POS))[0].numpy()
        reflected = model(_batch(_POS @ _REFLECTION.T))[0].numpy()
    return float(np.abs(reflected - _REFLECTION @ base).max())


def test_o3_dipole_is_polar_vector() -> None:
    assert _reflection_error(ParityMode.O3) < 1e-10


def test_so3_dipole_violates_reflection_parity() -> None:
    assert _reflection_error(ParityMode.SO3) > 1e-8
