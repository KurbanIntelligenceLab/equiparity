"""The mean-pooling control arm: EquiformerV2's atom-pooled tensor readout.

EquiformerV2 is documented as "a fixed SO(3) representative" (module docstring in
``equiparity.models.equiformer``): its backbone features are always all-even, so it has no
real O(3) arm to hold to an exact-zero structural test the way NequIP/Allegro/MACE do (a
``ParityMode.O3``-labelled EquiformerV2TensorModel would read out through an ``o3.Linear`` with
no valid odd-parity paths from an all-even source, making its output identically zero
everywhere, not specifically on centrosymmetric crystals -- a degenerate check, not the
Theorem-1 property). This file therefore checks the two properties that DO apply here:

1. ``pooling: sum`` (default) is bit-identical to the pre-existing ``index_add_`` readout.
2. ``pooling: mean`` still passes the rotation-equivariance gate (the property EquiformerV2 is
   verified to hold; float32, since its Wigner buffers are float32 -- module docstring).

Requires the ``nequip`` extra (for ``torch_geometric``, EquiformerV2's data pipeline).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")

from e3nn import o3

from equiparity.domain.parity import ParityMode
from equiparity.models.equiformer import (
    EquiformerV2Config,
    EquiformerV2TensorModel,
    to_pyg_data,
)
from equiparity.verification.equivariance import _random_orthogonal

pytestmark = pytest.mark.integration

POSITIONS = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.3], [0.0, 1.1, 0.0], [0.0, 0.0, 1.2]])
ATOMIC_NUMBERS = np.array([1, 1, 6, 8])
O3_IRREPS = "1x1o"  # a direct L=1 dipole head is enough to probe the pooled readout


class _Structure:
    def __init__(self, positions: np.ndarray, atomic_numbers: np.ndarray) -> None:
        self.atomic_numbers = atomic_numbers
        self.positions = positions
        self.cell = None
        self.pbc = False


def _cfg(pooling: str) -> EquiformerV2Config:
    return EquiformerV2Config(
        r_max=4.0,
        lmax=2,
        num_layers=1,
        sphere_channels=8,
        attn_hidden_channels=8,
        ffn_hidden_channels=8,
        num_heads=2,
        edge_channels=8,
        seed=0,
        pooling=pooling,
    )


def _build_model(pooling: str) -> EquiformerV2TensorModel:
    torch.manual_seed(0)
    return EquiformerV2TensorModel(_cfg(pooling), ParityMode.SO3, O3_IRREPS).eval()


def _predict(
    model: EquiformerV2TensorModel, positions: np.ndarray, seed: int = 9000
) -> torch.Tensor:
    """One forward pass, reseeded first.

    EquiformerV2 draws a fresh random per-edge frame on every forward call (module docstring in
    ``equiparity.inference.reload``), so a rotation-equivariance check across two separate calls
    needs the SAME frame draw on both -- reseed identically before each (the repo's own
    ``seeded_predict`` convention).
    """
    from torch_geometric.data import Batch

    g = to_pyg_data(_Structure(positions, ATOMIC_NUMBERS), 4.0, dtype=torch.float32)
    batch = Batch.from_data_list([g])
    torch.manual_seed(seed)
    with torch.no_grad():
        return model(batch)[0]


def test_pooling_defaults_to_sum() -> None:
    assert EquiformerV2Config(r_max=4.0).pooling == "sum"


def test_sum_pooling_is_bit_identical_to_index_add() -> None:
    """Regression guard, within a SINGLE forward call.

    EquiformerV2 draws a fresh random per-edge frame on every forward pass (module docstring
    in ``equiparity.inference.reload``), so two separate ``model(batch)`` calls on identical
    input are not comparable. Hook the pre-pooling per-atom readout during ONE call and check
    it against the pooled output from that same call.
    """
    from torch_geometric.data import Batch

    model = _build_model(pooling="sum")
    g = to_pyg_data(_Structure(POSITIONS, ATOMIC_NUMBERS), 4.0, dtype=torch.float32)
    batch = Batch.from_data_list([g])

    store: dict[str, torch.Tensor] = {}
    handle = model.readout.register_forward_hook(lambda _m, _i, o: store.__setitem__("per_atom", o))
    try:
        with torch.no_grad():
            actual = model(batch)
    finally:
        handle.remove()

    per_atom = store["per_atom"]
    batch_index = batch.batch
    n_graphs = int(batch_index.max().item()) + 1
    expected = torch.zeros(n_graphs, per_atom.shape[1], dtype=per_atom.dtype).index_add_(
        0, batch_index, per_atom
    )
    assert torch.equal(actual, expected)


def test_mean_pooling_preserves_rotation_equivariance() -> None:
    model = _build_model(pooling="mean")
    rotation = _random_orthogonal(0, improper=False)
    out0 = _predict(model, POSITIONS)
    out_rot = _predict(model, POSITIONS @ rotation.T)
    irreps = o3.Irreps(model.output_irreps)
    d_matrix = irreps.D_from_matrix(torch.as_tensor(rotation, dtype=out0.dtype))
    err = (out_rot - out0 @ d_matrix.T).abs().max().item()
    # EquiformerV2 trains natively in float32 (module docstring); use the parity gate's
    # float32 rotation bound (tests/verification/*_gate.py pattern: rotation_max=1e-5).
    assert err < 1e-5
