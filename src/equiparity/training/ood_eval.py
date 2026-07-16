"""OOD piezoelectric-violation evaluation: both eval variants + full threshold curve.

The headline used a single 0.01 C/m^2 threshold on the idealized (exact space group) OOD set.
Reviewers require (a) both eval variants side by side -- idealized (tests the structural guarantee)
and raw DFT-relaxed (tests robustness on real data) -- and (b) the full false-flag-fraction vs
threshold curve plus the violation-magnitude distribution, not one operating point. This computes
both from the per-structure violation magnitudes ``||predicted tensor||`` (exactly zero, up to model
error, for a truly centrosymmetric crystal).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Log-spaced thresholds 1e-4 .. 1 C/m^2 (reviewer spec) for the false-flag-fraction curve.
THRESHOLDS = np.logspace(-4, 0, 25)


def violation_stats(mags: np.ndarray) -> dict[str, object]:
    """Distribution summary + false-flag-vs-threshold curve for one variant's magnitudes."""
    mags = np.asarray(mags, dtype=float)
    p5, p25, p50, p75, p95 = (float(x) for x in np.percentile(mags, [5, 25, 50, 75, 95]))
    return {
        "n": int(mags.size),
        "median": p50,
        "iqr": [p25, p75],
        "percentiles_5_25_50_75_95": [p5, p25, p50, p75, p95],
        "max": float(mags.max()),
        "false_flag_at_0.01": float((mags > 0.01).mean()),
        "thresholds": THRESHOLDS.tolist(),
        "false_flag_fraction": [float((mags > t).mean()) for t in THRESHOLDS],
    }


def evaluate_ood_variants(
    predict_violations: Callable[[str], np.ndarray],
    variants: dict[str, str | None],
) -> dict[str, object]:
    """Evaluate every provided OOD variant.

    Args:
        predict_violations: closure mapping an OOD npz path to the per-structure violation
            magnitudes (``||predicted piezoelectric tensor||``) for the trained model.
        variants: variant name -> npz path (``None`` entries skipped), e.g.
            ``{"idealized": ".../processed.npz", "raw": ".../processed_raw.npz"}``.

    Returns:
        ``{variant: {"stats": <violation_stats>, "vector": <np.ndarray>}}``. The caller persists
        ``stats`` into metrics.json and ``vector`` as an ``.npy`` for offline curves/histograms.
    """
    out: dict[str, object] = {}
    for name, path in variants.items():
        if not path:
            continue
        mags = np.asarray(predict_violations(path), dtype=float)
        out[name] = {"stats": violation_stats(mags), "vector": mags}
    return out
