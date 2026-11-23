"""Evaluate piezoelectric tensor norms on raw and idealized crystals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def violation_stats(mags: np.ndarray) -> dict[str, object]:
    """Summarize the norm distribution and fraction above 0.01 C/m²."""
    mags = np.asarray(mags, dtype=float)
    p5, p25, p50, p75, p95 = (float(x) for x in np.percentile(mags, [5, 25, 50, 75, 95]))
    return {
        "n": int(mags.size),
        "median": p50,
        "iqr": [p25, p75],
        "percentiles_5_25_50_75_95": [p5, p25, p50, p75, p95],
        "max": float(mags.max()),
        "false_flag_at_0.01": float((mags > 0.01).mean()),
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
        ``stats`` into metrics.json and ``vector`` as an ``.npy`` for per-crystal analysis.
    """
    out: dict[str, object] = {}
    for name, path in variants.items():
        if not path:
            continue
        mags = np.asarray(predict_violations(path), dtype=float)
        out[name] = {"stats": violation_stats(mags), "vector": mags}
    return out
