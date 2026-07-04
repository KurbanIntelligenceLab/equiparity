"""Integration tests for the NequIP parity-toggle builder.

These require the ``nequip`` extra (``uv sync --extra nequip``) and are skipped otherwise.
They lock in the verified fact that the O(3)/SO(3) toggle only manifests at >=3 layers.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nequip")

from equiparity.domain.parity import ParityMode
from equiparity.models.nequip import (
    NequIPConfig,
    build_nequip,
    count_parameters,
    realized_hidden_irreps,
)

pytestmark = pytest.mark.integration


def _config(num_layers: int) -> NequIPConfig:
    return NequIPConfig(
        r_max=4.0,
        type_names=("H", "C", "O"),
        num_layers=num_layers,
        l_max=2,
        num_features=16,
        radial_mlp_width=32,
        avg_num_neighbors=10.0,
    )


def test_deep_network_o3_has_more_params_than_so3() -> None:
    cfg = _config(num_layers=4)
    o3 = count_parameters(build_nequip(cfg, ParityMode.O3))
    so3 = count_parameters(build_nequip(cfg, ParityMode.SO3))
    assert o3 > so3, f"O(3) should carry more channels than SO(3): {o3} vs {so3}"


def test_o3_middle_layer_realizes_both_parities() -> None:
    cfg = _config(num_layers=4)
    o3_irreps = realized_hidden_irreps(build_nequip(cfg, ParityMode.O3))
    so3_irreps = realized_hidden_irreps(build_nequip(cfg, ParityMode.SO3))
    # A middle layer in O(3) mode reaches even-l-odd / odd-l-even channels (e.g. 1e, 2o)
    # that natural-parity SO(3) never populates.
    o3_middle = o3_irreps["model.func.layer1_convnet"]
    so3_middle = so3_irreps["model.func.layer1_convnet"]
    assert "1e" in o3_middle and "2o" in o3_middle
    assert "1e" not in so3_middle and "2o" not in so3_middle


def test_shallow_network_toggle_is_a_noop() -> None:
    # The verified gotcha: at 2 layers the toggle changes nothing. Guards against a
    # verification gate that silently uses too few layers.
    with pytest.warns(UserWarning, match="parity toggle"):
        cfg = _config(num_layers=2)
    o3 = count_parameters(build_nequip(cfg, ParityMode.O3))
    so3 = count_parameters(build_nequip(cfg, ParityMode.SO3))
    assert o3 == so3
