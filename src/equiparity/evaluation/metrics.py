"""Regression metrics for scalar, vector, and tensor targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Summary regression metrics over a set of predictions."""

    mae: float
    rmse: float
    n_samples: int

    def to_dict(self) -> dict[str, float | int]:
        """Serialize to a plain JSON-ready dict."""
        return asdict(self)


def regression_metrics(
    predictions: npt.NDArray[np.float64], targets: npt.NDArray[np.float64]
) -> MetricSummary:
    """Compute MAE and RMSE between predictions and targets.

    Both arrays are flattened, so this works for scalar, vector, or tensor targets. For
    multi-component targets the metrics are per-component averages.

    Raises:
        ValueError: If the shapes differ or the input is empty.
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"shape mismatch: {predictions.shape} vs {targets.shape}")
    if predictions.size == 0:
        raise ValueError("cannot compute metrics on empty input")
    error = predictions.reshape(-1) - targets.reshape(-1)
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt((error**2).mean()))
    return MetricSummary(mae=mae, rmse=rmse, n_samples=int(targets.shape[0]))


def frobenius_error(
    predictions: npt.NDArray[np.float64], targets: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Per-sample Frobenius norm of the prediction error (for tensor targets).

    Args:
        predictions: Shape ``(n_samples, n_components)``.
        targets: Same shape.

    Returns:
        Shape ``(n_samples,)`` of ``||pred - target||_F`` per sample.
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"shape mismatch: {predictions.shape} vs {targets.shape}")
    diff = predictions - targets
    result: npt.NDArray[np.float64] = np.sqrt((diff**2).sum(axis=1))
    return result
