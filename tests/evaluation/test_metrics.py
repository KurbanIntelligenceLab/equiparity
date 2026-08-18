"""Tests for regression metrics."""

from __future__ import annotations

import numpy as np
import pytest

from equiparity.evaluation.metrics import frobenius_error, regression_metrics


def test_perfect_prediction() -> None:
    y = np.array([[1.0], [2.0], [3.0]])
    m = regression_metrics(y, y)
    assert m.mae == 0.0
    assert m.rmse == 0.0
    assert m.n_samples == 3


def test_known_errors() -> None:
    pred = np.array([[1.0], [2.0]])
    target = np.array([[2.0], [4.0]])  # errors 1 and 2
    m = regression_metrics(pred, target)
    assert m.mae == pytest.approx(1.5)
    assert m.rmse == pytest.approx(np.sqrt((1 + 4) / 2))


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        regression_metrics(np.zeros((2, 1)), np.zeros((3, 1)))


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        regression_metrics(np.zeros((0, 1)), np.zeros((0, 1)))


def test_frobenius_error() -> None:
    pred = np.array([[3.0, 4.0], [0.0, 0.0]])
    target = np.array([[0.0, 0.0], [0.0, 0.0]])
    fe = frobenius_error(pred, target)
    np.testing.assert_allclose(fe, [5.0, 0.0])  # 3-4-5 triangle
