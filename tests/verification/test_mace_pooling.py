"""The mean-pooling control arm: MACE's atom-pooled tensor readout.

Same three guarantees as ``tests/verification/test_nequip_pooling.py``, for the MACE core:

1. ``pooling: sum`` (default) is bit-identical to the pre-existing ``index_add_`` readout.
2. ``pooling: mean`` still passes the rotation/reflection equivariance gate, float64.
3. ``pooling: mean`` still gives an exact structural zero for the O(3) arm on a centrosymmetric
   crystal.

Requires the ``mace`` extra; skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("mace")

# e3nn 0.4.4 (the MACE profile) unpickles a Wigner-constants file at *import* time under torch
# 2.6+'s weights_only default; must run before anything imports e3nn (mirrors inference/reload.py).
torch.serialization.add_safe_globals([slice])

from e3nn import o3  # noqa: E402
from mace import tools  # noqa: E402

from equiparity.domain.parity import ParityMode  # noqa: E402
from equiparity.inference.structures import rutile  # noqa: E402
from equiparity.models.mace import MACEConfig, MACETensorModel  # noqa: E402
from equiparity.training.mace_scalar import _to_mace_data  # noqa: E402
from equiparity.training.mace_tensor import _batches  # noqa: E402
from equiparity.training.nequip_data import element_type_map  # noqa: E402
from equiparity.verification.equivariance import _random_orthogonal  # noqa: E402

pytestmark = pytest.mark.integration

POSITIONS = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.3], [0.0, 1.1, 0.0], [0.0, 0.0, 1.2]])
ATOMIC_NUMBERS = np.array([1, 1, 6, 8])
ELEMENT_TABLE = (1, 6, 8)
O3_IRREPS = "2x1o+1x2o+1x3o"


class _Structure:
    def __init__(self, positions: np.ndarray, atomic_numbers: np.ndarray) -> None:
        self.atomic_numbers = atomic_numbers
        self.positions = positions
        self.cell = None
        self.pbc = False


def _build_model(pooling: str) -> MACETensorModel:
    cfg = MACEConfig(
        r_max=4.0,
        atomic_numbers=ELEMENT_TABLE,
        num_interactions=2,
        l_max=2,
        num_features=16,
        model_dtype="float64",
        pooling=pooling,
        seed=0,
    )
    torch.manual_seed(0)
    return MACETensorModel(cfg, ParityMode.O3, O3_IRREPS).eval()


def _predict(model: MACETensorModel, positions: np.ndarray) -> torch.Tensor:
    z_table = tools.AtomicNumberTable(list(ELEMENT_TABLE))
    g = _to_mace_data(_Structure(positions, ATOMIC_NUMBERS), z_table, 4.0)
    batch = next(iter(_batches([g], 1, torch.float64)))
    with torch.no_grad():
        return model(batch)[0]


def test_pooling_defaults_to_sum() -> None:
    assert MACEConfig(r_max=4.0, atomic_numbers=ELEMENT_TABLE).pooling == "sum"


def test_sum_pooling_is_bit_identical_to_index_add() -> None:
    model = _build_model(pooling="sum")
    z_table = tools.AtomicNumberTable(list(ELEMENT_TABLE))
    g = _to_mace_data(_Structure(POSITIONS, ATOMIC_NUMBERS), z_table, 4.0)
    batch = next(iter(_batches([g], 1, torch.float64)))

    store: dict[str, torch.Tensor] = {}
    handle = model._probe_module.register_forward_hook(
        lambda _m, _i, o: store.__setitem__("feat", o if torch.is_tensor(o) else o[0])
    )
    try:
        with torch.no_grad():
            model.backbone(batch, compute_force=False, compute_virials=False, compute_stress=False)
    finally:
        handle.remove()

    per_atom = model.readout(store["feat"])
    batch_index = batch["batch"]
    n_graphs = int(batch_index.max().item()) + 1
    expected = torch.zeros(n_graphs, per_atom.shape[1], dtype=per_atom.dtype).index_add_(
        0, batch_index, per_atom
    )
    with torch.no_grad():
        actual = model(batch)
    assert torch.equal(actual, expected)


def test_mean_pooling_preserves_rotation_equivariance() -> None:
    model = _build_model(pooling="mean")
    rotation = _random_orthogonal(0, improper=False)
    out0 = _predict(model, POSITIONS)
    out_rot = _predict(model, POSITIONS @ rotation.T)
    irreps = o3.Irreps(model.output_irreps)
    d_matrix = irreps.D_from_matrix(torch.as_tensor(rotation, dtype=out0.dtype))
    err = (out_rot - out0 @ d_matrix.T).abs().max().item()
    # MACE's symmetric-contraction tensors stay float32 even under a float64 model (module
    # docstring in equiparity.models.mace), so its equivariance error floors ~1e-7; the parity
    # gate (tests/verification/test_mace_gate.py) uses the float32 threshold (1e-5) for the
    # same reason. A tensor HEAD adds one more o3.Linear on top of the probed features, so use
    # the same float32-class bound rather than the float64 one used for NequIP/Allegro.
    assert err < 1e-5


def test_mean_pooling_preserves_the_mirror_law() -> None:
    model = _build_model(pooling="mean")
    reflection = _random_orthogonal(1, improper=True)
    out0 = _predict(model, POSITIONS)
    out_ref = _predict(model, POSITIONS @ reflection.T)
    irreps = o3.Irreps(model.output_irreps)
    d_matrix = irreps.D_from_matrix(torch.as_tensor(reflection, dtype=out0.dtype))
    err = (out_ref - out0 @ d_matrix.T).abs().max().item()
    assert err < 1e-5


def test_o3_arm_predicts_exact_zero_on_centrosymmetric_crystal_under_mean_pooling() -> None:
    # P4_2/mnm (136), centrosymmetric, subgroup 422 (rotation-subgroup-safe control)
    structure = rutile("TiO2")
    _, _symbol_to_type = element_type_map(structure.atomic_numbers)
    elements = tuple(sorted(int(z) for z in np.unique(structure.atomic_numbers)))
    z_table = tools.AtomicNumberTable(list(elements))

    cfg = MACEConfig(
        r_max=5.0,
        atomic_numbers=elements,
        num_interactions=2,
        l_max=2,
        num_features=16,
        model_dtype="float64",
        pooling="mean",
        seed=0,
    )
    torch.manual_seed(0)
    model = MACETensorModel(cfg, ParityMode.O3, O3_IRREPS).eval()

    g = _to_mace_data(structure, z_table, 5.0)
    batch = next(iter(_batches([g], 1, torch.float64)))
    with torch.no_grad():
        out = model(batch)[0]
    # MACE's float32-internal symmetric contractions floor its residual around 1e-7 to 1e-6
    # (module docstring; Supplementary Table stab:distribution reports a trained-model O(3)
    # floor of 2.7e-6 median for MACE, vs 1e-7 for NequIP/Allegro at the same table). This is
    # untrained and tiny, but keep the MACE-specific bound rather than the float64 one.
    assert float(out.norm()) < 1e-6
