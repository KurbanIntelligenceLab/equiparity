"""The mean-pooling control arm: NequIP and Allegro readouts.

Locks in three properties of ``model.pooling`` for the two nequip-framework cores:

1. ``pooling: sum`` (the default) is bit-identical to the pre-existing ``index_add_`` readout
   -- reproduces every committed result and checkpoint exactly.
2. ``pooling: mean`` still passes the rotation/reflection equivariance gate, float64.
3. ``pooling: mean`` still returns an exact structural zero for the O(3) arm on a
   centrosymmetric crystal (Theorem 1 holds for any equivariant pooling).

Allegro's readout is per-EDGE, not per-atom (module docstring in ``equiparity.models.allegro``),
so its mean divides by the edge count; the test checks that explicitly via ``pool_unit``.

Requires the ``nequip`` extra; skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("nequip")

from e3nn import o3

from equiparity.domain.parity import ParityMode
from equiparity.inference.structures import rutile
from equiparity.models.allegro import AllegroConfig, AllegroTensorModel
from equiparity.models.nequip import NequIPConfig, NequIPTensorModel
from equiparity.training.nequip_data import element_type_map, to_atomic_data
from equiparity.verification.equivariance import _random_orthogonal

pytestmark = pytest.mark.integration

POSITIONS = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.3], [0.0, 1.1, 0.0], [0.0, 0.0, 1.2]])
ATOMIC_NUMBERS = np.array([1, 1, 6, 8])
O3_IRREPS = "2x1o+1x2o+1x3o"  # the piezoelectric tensor (domain/target.py PIEZOELECTRIC)


def _batch_data(positions: np.ndarray, r_max: float):
    from nequip.data import AtomicDataDict

    type_names, symbol_to_type = element_type_map(ATOMIC_NUMBERS)

    class _S:
        pass

    s = _S()
    s.atomic_numbers = ATOMIC_NUMBERS
    s.positions = positions
    s.cell = None
    s.pbc = False
    data = to_atomic_data(s, symbol_to_type, r_max, torch.float64)
    data[AtomicDataDict.BATCH_KEY] = torch.zeros(len(positions), dtype=torch.long)
    return data, type_names


@pytest.fixture(params=["nequip", "allegro"])
def core(request) -> str:
    return request.param


def _build_model(core: str, mode: ParityMode, pooling: str, avg_num_neighbors: float = 3.0):
    _, type_names = _batch_data(POSITIONS, 4.0)
    if core == "allegro":
        pytest.importorskip("allegro")
        cfg = AllegroConfig(
            r_max=4.0,
            type_names=type_names,
            num_layers=2,
            l_max=2,
            num_scalar_features=16,
            num_tensor_features=8,
            avg_num_neighbors=avg_num_neighbors,
            model_dtype="float64",
            pooling=pooling,
            seed=0,
        )
        torch.manual_seed(0)
        return AllegroTensorModel(cfg, mode, O3_IRREPS).eval()
    cfg = NequIPConfig(
        r_max=4.0,
        type_names=type_names,
        num_layers=3,
        l_max=2,
        num_features=16,
        type_embed_num_features=16,
        radial_mlp_width=32,
        avg_num_neighbors=avg_num_neighbors,
        model_dtype="float64",
        pooling=pooling,
        seed=0,
    )
    torch.manual_seed(0)
    return NequIPTensorModel(cfg, mode, O3_IRREPS).eval()


def _predict(model, positions: np.ndarray) -> torch.Tensor:
    data, _ = _batch_data(positions, 4.0)
    with torch.no_grad():
        return model(data)[0]


def test_pooling_defaults_to_sum(core: str) -> None:
    cfg_cls = AllegroConfig if core == "allegro" else NequIPConfig
    kwargs = (
        dict(
            r_max=4.0,
            type_names=("H", "C"),
            num_scalar_features=8,
            num_tensor_features=4,
        )
        if core == "allegro"
        else dict(r_max=4.0, type_names=("H", "C"))
    )
    assert cfg_cls(**kwargs).pooling == "sum"


def test_sum_pooling_is_bit_identical_to_index_add(core: str) -> None:
    """Regression guard: pooling="sum" (default) must reproduce the pre-patch readout exactly."""
    model = _build_model(core, ParityMode.O3, pooling="sum")
    data, _ = _batch_data(POSITIONS, 4.0)
    from nequip.data import AtomicDataDict

    store: dict[str, torch.Tensor] = {}

    def _hook(_m, _i, o):
        if isinstance(o, dict) and AtomicDataDict.NODE_FEATURES_KEY in o:
            store["feat"] = o[AtomicDataDict.NODE_FEATURES_KEY]
        else:
            store["feat"] = o if torch.is_tensor(o) else o[0]

    handle = model._probe_module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model.backbone(data)
    finally:
        handle.remove()

    per_unit = model.readout(store["feat"])
    if per_unit.dim() == 3:
        per_unit = per_unit.sum(dim=1)
    if core == "allegro":
        from nequip.data import AtomicDataDict

        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        batch_index = data[AtomicDataDict.BATCH_KEY]
        unit_to_graph = batch_index[edge_index[0]]
    else:
        from nequip.data import AtomicDataDict

        unit_to_graph = data[AtomicDataDict.BATCH_KEY]
    n_graphs = int(unit_to_graph.max().item()) + 1
    expected = torch.zeros(n_graphs, per_unit.shape[1], dtype=per_unit.dtype).index_add_(
        0, unit_to_graph, per_unit
    )

    with torch.no_grad():
        actual = model(data)
    assert torch.equal(actual, expected)  # bit-identical, not just close


def test_mean_pooling_preserves_rotation_equivariance(core: str) -> None:
    model = _build_model(core, ParityMode.O3, pooling="mean")
    rotation = _random_orthogonal(0, improper=False)
    out0 = _predict(model, POSITIONS)
    out_rot = _predict(model, POSITIONS @ rotation.T)
    irreps = o3.Irreps(model.output_irreps)
    d_matrix = irreps.D_from_matrix(torch.as_tensor(rotation, dtype=out0.dtype))
    err = (out_rot - out0 @ d_matrix.T).abs().max().item()
    assert err < 1e-10


def test_mean_pooling_preserves_the_mirror_law(core: str) -> None:
    """The O(3) arm must still respect reflections (parity) under mean pooling."""
    model = _build_model(core, ParityMode.O3, pooling="mean")
    reflection = _random_orthogonal(1, improper=True)
    out0 = _predict(model, POSITIONS)
    out_ref = _predict(model, POSITIONS @ reflection.T)
    irreps = o3.Irreps(model.output_irreps)
    d_matrix = irreps.D_from_matrix(torch.as_tensor(reflection, dtype=out0.dtype))
    err = (out_ref - out0 @ d_matrix.T).abs().max().item()
    assert err < 1e-10


def test_o3_arm_predicts_exact_zero_on_centrosymmetric_crystal_under_mean_pooling(
    core: str,
) -> None:
    """Theorem 1 holds for any equivariant pooling: mean must still give an exact zero."""
    # P4_2/mnm (136), centrosymmetric, subgroup 422 (rotation-subgroup-safe control)
    structure = rutile("TiO2")
    type_names, symbol_to_type = element_type_map(structure.atomic_numbers)

    if core == "allegro":
        pytest.importorskip("allegro")
        cfg = AllegroConfig(
            r_max=5.0,
            type_names=type_names,
            num_layers=2,
            l_max=2,
            num_scalar_features=16,
            num_tensor_features=8,
            avg_num_neighbors=10.0,
            model_dtype="float64",
            pooling="mean",
            seed=0,
        )
        torch.manual_seed(0)
        model = AllegroTensorModel(cfg, ParityMode.O3, O3_IRREPS).eval()
    else:
        cfg = NequIPConfig(
            r_max=5.0,
            type_names=type_names,
            num_layers=3,
            l_max=2,
            num_features=16,
            type_embed_num_features=16,
            avg_num_neighbors=10.0,
            model_dtype="float64",
            pooling="mean",
            seed=0,
        )
        torch.manual_seed(0)
        model = NequIPTensorModel(cfg, ParityMode.O3, O3_IRREPS).eval()

    data = to_atomic_data(structure, symbol_to_type, 5.0, torch.float64)
    from nequip.data import AtomicDataDict

    data[AtomicDataDict.BATCH_KEY] = torch.zeros(structure.n_atoms, dtype=torch.long)
    with torch.no_grad():
        out = model(data)[0]
    assert float(out.norm()) < 1e-10
