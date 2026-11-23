"""Descriptive, crystal-level comparisons on a fixed evaluation partition."""

from __future__ import annotations

import numpy as np


def paired_mae_summary(
    prediction: np.ndarray, target: np.ndarray, baseline: np.ndarray
) -> dict[str, object]:
    """Resample crystals, retaining all correlated tensor components together.

    These intervals describe the supplied partition. When that partition selected
    the checkpoint, they are not independent held-out evidence of generalization.
    """
    if prediction.shape != target.shape or prediction.ndim != 2 or not len(prediction):
        raise ValueError("nonempty matching prediction and target matrices required")
    baseline = np.broadcast_to(baseline, target.shape)
    if not all(np.isfinite(a).all() for a in (prediction, target, baseline)):
        raise ValueError("paired errors require finite inputs")
    model_error = np.abs(prediction - target).mean(axis=1)
    baseline_error = np.abs(baseline - target).mean(axis=1)
    gain = baseline_error - model_error
    rng = np.random.default_rng(0)
    means = np.concatenate(
        [gain[rng.integers(len(gain), size=(100, len(gain)))].mean(axis=1) for _ in range(100)]
    )
    return {
        "n_crystals": len(gain),
        "mean_mae_reduction": float(gain.mean()),
        "descriptive_bootstrap_95_percent_interval": np.percentile(means, [2.5, 97.5]).tolist(),
        "fraction_strictly_improved": float(np.mean(gain > 0)),
        "median_model_crystal_mae": float(np.median(model_error)),
        "median_baseline_crystal_mae": float(np.median(baseline_error)),
        "model_crystal_mae_quantiles_50_90_99": np.percentile(model_error, [50, 90, 99]).tolist(),
        "baseline_crystal_mae_quantiles_50_90_99": np.percentile(
            baseline_error, [50, 90, 99]
        ).tolist(),
        "resampling_unit": "crystal",
        "bootstrap_seed": 0,
        "bootstrap_replicates": 10000,
    }
