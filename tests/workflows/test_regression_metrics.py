"""Seed variability and incomplete evidence must not be misreported."""

import json
from pathlib import Path

import numpy as np
import pytest

from equiparity.workflows.regression_metrics import summarize_regressions, three_seed_statistics


def test_seed_statistics_and_fixed_baselines() -> None:
    rows = [
        {
            "seed": seed,
            "metrics": {"mae": value, "rmse": value * 2, "frobenius_rmse": value * 3},
            "comparisons": {
                name: {"mean_mae_reduction": 1 - value}
                for name in ["zero", "training_mean", "training_median"]
            },
        }
        for seed, value in [(2, 0.3), (0, 0.1), (1, 0.2)]
    ]
    result = three_seed_statistics(rows)
    assert np.isclose(result["mae"]["mean"], 0.2)
    assert np.isclose(result["mae"]["sample_sd"], 0.1)
    assert result["mae"]["by_seed"] == [0.1, 0.2, 0.3]
    assert result["baselines"]["zero"]["all_three_seeds_improve"]
    with pytest.raises(ValueError, match="exactly seeds"):
        three_seed_statistics([rows[0], rows[0], rows[1]])
    rows[0]["comparisons"]["zero"]["mean_mae_reduction"] += 1
    with pytest.raises(ValueError, match="baseline changes"):
        three_seed_statistics(rows)


def test_missing_evaluations_cannot_create_summary(tmp_path: Path) -> None:
    lock = tmp_path / "protocol.json"
    lock.write_text(json.dumps({"configs": [{"scientific_sha256": "required"}]}))
    output = tmp_path / "summary.json"
    with pytest.raises(ValueError, match="incomplete protocol"):
        summarize_regressions(lock, [], output)
    assert not output.exists()
