"""Smoke test for the NequIP scalar trainer (requires nequip + QM9 data; skipped otherwise)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nequip")

_QM9 = Path("data/raw/qm9/qm9_processed.npz")
_SPLIT = Path("data/splits/qm9_split.npz")
if not (_QM9.exists() and _SPLIT.exists()):
    pytest.skip(
        "QM9 processed data not present; run scripts/data/prepare_qm9.py", allow_module_level=True
    )

from equiparity.domain.experiment import (  # noqa: E402
    ExperimentConfig,
    ModelHyperparams,
    TrainingParams,
)
from equiparity.domain.parity import ParityMode  # noqa: E402
from equiparity.training.nequip_scalar import train_scalar  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _config(mode: ParityMode) -> ExperimentConfig:
    return ExperimentConfig(
        seed=0,
        core="nequip",
        parity=mode,
        target="U0",
        dataset="qm9",
        processed_npz=_QM9,
        split_npz=_SPLIT,
        output_dir=Path("outputs"),
        model=ModelHyperparams(num_layers=3, l_max=1, num_features=8, r_max=5.0),
        training=TrainingParams(
            batch_size=8,
            epochs=2,
            lr=5e-3,
            device="cuda",
            max_train_samples=40,
            max_eval_samples=20,
        ),
    )


def test_trains_and_returns_finite_metrics() -> None:
    result = train_scalar(_config(ParityMode.O3))
    assert result.n_params > 0
    assert result.epochs_run == 2
    assert np.isfinite(result.test.mae)
    assert result.test.n_samples == 20
