"""Smoke test for the MACE scalar trainer (requires mace + QM9 data; skipped otherwise)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mace")

_QM9 = Path("data/raw/qm9/qm9_processed.npz")
_SPLIT = Path("data/splits/qm9_split.npz")
if not (_QM9.exists() and _SPLIT.exists()):
    pytest.skip("QM9 processed data not present", allow_module_level=True)

from equiparity.domain.experiment import (  # noqa: E402
    ExperimentConfig,
    ModelHyperparams,
    TrainingParams,
)
from equiparity.domain.parity import ParityMode  # noqa: E402
from equiparity.training.mace_scalar import train_mace_scalar  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_mace_trains_both_modes() -> None:
    for mode in (ParityMode.O3, ParityMode.SO3):
        config = ExperimentConfig(
            seed=0,
            core="mace",
            parity=mode,
            target="U0",
            dataset="qm9",
            processed_npz=_QM9,
            split_npz=_SPLIT,
            output_dir=Path("outputs"),
            model=ModelHyperparams(num_layers=2, l_max=1, num_features=8, r_max=5.0),
            training=TrainingParams(
                batch_size=8,
                epochs=2,
                lr=5e-3,
                device="cuda",
                max_train_samples=32,
                max_eval_samples=16,
            ),
        )
        result = train_mace_scalar(config)
        assert result.n_params > 0
        assert np.isfinite(result.test.mae)
