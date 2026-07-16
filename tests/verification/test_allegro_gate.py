"""Integration test: the Allegro O(3)/SO(3) matched pair passes the parity gate.

Requires the ``nequip`` extra (nequip-allegro). Locks in that Allegro has the same parity
mechanism as NequIP: only the SO(3) arm (all-even edge SH) breaks reflections.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("allegro")

from equiparity.domain.parity import ParityMode
from equiparity.models.allegro import AllegroConfig, allegro_featurizer, build_allegro_matched
from equiparity.verification.equivariance import (
    EquivarianceReport,
    check_equivariance,
    count_parameters,
)

pytestmark = pytest.mark.integration

POSITIONS = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.3], [0.0, 1.1, 0.0], [0.0, 0.0, 1.2]])


def _report(mode: ParityMode) -> EquivarianceReport:
    config = AllegroConfig(
        r_max=4.0,
        type_names=("H", "C", "O"),
        num_layers=2,
        l_max=2,
        num_scalar_features=16,
        num_tensor_features=8,
        avg_num_neighbors=10.0,
        model_dtype="float64",
    )
    model = build_allegro_matched(config, mode).eval()
    featurize = allegro_featurizer(model, "H2CO", config.type_names, r_max=4.0, dtype="float64")
    return check_equivariance(
        featurize, POSITIONS, label=f"Allegro {mode.label}", n_params=count_parameters(model)
    )


def test_o3_arm_is_parity_respecting() -> None:
    report = _report(ParityMode.O3)
    assert report.verdict == "O3"
    assert report.reflection_error < 1e-12


def test_so3_arm_breaks_reflection_only() -> None:
    report = _report(ParityMode.SO3)
    assert report.verdict == "SO3"
    assert report.rotation_error < 1e-12  # rotations preserved
    assert report.reflection_error > 1e-4  # reflections broken
