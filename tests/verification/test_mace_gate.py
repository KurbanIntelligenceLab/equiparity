"""Integration test: the MACE O(3)/SO(3) matched pair passes the parity gate.

Requires the ``mace`` extra (``uv sync --extra mace``); skipped otherwise. MACE's correct
SO(3) toggle is ``use_so3=True`` (all-even SH). MACE runs at ~1e-7 precision, so the gate
uses the float32 thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mace")

from equiparity.domain.parity import ParityMode
from equiparity.models.mace import MACE_GATE_DTYPE, MACEConfig, build_mace_matched, mace_featurizer
from equiparity.verification.equivariance import (
    EquivarianceReport,
    check_equivariance,
    count_parameters,
)

pytestmark = pytest.mark.integration

POSITIONS = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.3], [0.0, 1.1, 0.0], [0.0, 0.0, 1.2]])
ATOM_NUMBERS = (1, 1, 6, 8)  # per-atom (H2CO)
ELEMENT_TABLE = (1, 6, 8)


def _report(mode: ParityMode) -> EquivarianceReport:
    config = MACEConfig(
        r_max=4.0, atomic_numbers=ELEMENT_TABLE, num_features=16, model_dtype="float64"
    )
    model = build_mace_matched(config, mode).eval()
    featurize = mace_featurizer(model, ATOM_NUMBERS, ELEMENT_TABLE, r_max=4.0, dtype="float64")
    return check_equivariance(
        featurize,
        POSITIONS,
        label=f"MACE {mode.label}",
        n_params=count_parameters(model),
        dtype=MACE_GATE_DTYPE,
    )


def test_o3_arm_is_parity_respecting() -> None:
    report = _report(ParityMode.O3)
    assert report.verdict == "O3"
    assert report.reflection_error < 1e-5


def test_so3_arm_breaks_reflection_only() -> None:
    report = _report(ParityMode.SO3)
    assert report.verdict == "SO3"
    assert report.rotation_error < 1e-5  # rotations preserved
    assert report.reflection_error > 1e-2  # reflections broken
