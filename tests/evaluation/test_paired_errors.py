"""Crystal-level error comparisons must retain within-tensor correlations."""

import numpy as np
import pytest

from equiparity.evaluation.paired_errors import paired_mae_summary


def test_paired_summary_exact_gain_and_no_artificial_component_sample_size() -> None:
    target = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
    prediction = target / 2
    result = paired_mae_summary(prediction, target, np.zeros(2))
    repeated = paired_mae_summary(np.tile(prediction, (1, 4)), np.tile(target, (1, 4)), np.zeros(8))
    assert result == repeated
    assert result["n_crystals"] == 3
    assert result["mean_mae_reduction"] == 1.5
    assert result["fraction_strictly_improved"] == 1.0
    equal = paired_mae_summary(np.zeros_like(target), target, np.zeros(2))
    assert equal["descriptive_bootstrap_95_percent_interval"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="nonempty matching"):
        paired_mae_summary(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(2))
