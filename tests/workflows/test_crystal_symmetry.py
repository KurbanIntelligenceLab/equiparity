"""Per-class diagnostics must not hide empty classes or conflate norms with dimensions."""

import numpy as np
import pytest

from equiparity.workflows.crystal_symmetry import class_summary


def test_all_eleven_classes_with_exact_counts_and_empty_markers() -> None:
    rows = class_summary(np.array(["-1", "-1", "m-3m"]), np.array([0.0, 0.02, 1e-8]))
    assert len(rows) == 11 and sum(r["n"] for r in rows) == 3
    by_class = {r["point_group"]: r for r in rows}
    assert by_class["-1"]["gamma3"] == 18
    assert by_class["-1"]["false_flag_fraction"] == 0.5
    assert by_class["-1"]["median_norm"] == 0.01
    assert by_class["m-3m"]["gamma3"] == 0
    assert by_class["m-3m"]["false_flag_fraction"] == 0.0
    assert by_class["mmm"]["n"] == 0 and by_class["mmm"]["median_norm"] is None


def test_reject_noncentrosymmetric_population_or_nonfinite_predictions() -> None:
    with pytest.raises(ValueError, match="non-centrosymmetric"):
        class_summary(np.array(["1"]), np.array([0.0]))
    with pytest.raises(ValueError, match="finite"):
        class_summary(np.array(["-1"]), np.array([np.nan]))
