"""Measure external piezoelectric predictions under the two tensor norm conventions."""

import hashlib
import io
import zipfile
from collections.abc import Iterator
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


def mask_conditions() -> Iterator[str]:
    """The model, geometry and mask combinations in the external prediction set."""
    for model, geometry, state in product(
        ("gmtnet", "ceitnet"), ("idealized", "raw"), ("off", "on")
    ):
        yield f"{model}_{geometry}_mask{state}"


def norm_summary(norms: np.ndarray) -> dict[str, float | int]:
    """Compute the fraction exceeding 0.01 C/m² and the norm distribution."""
    return {
        "n": len(norms),
        "false_flag_at_0.01": float((norms > 0.01).mean()),
        "median": float(np.median(norms)),
        "max": float(norms.max()),
    }


def measure_masks(predictions: Path) -> dict[str, Any]:
    """Calculate each external checkpoint's statistics directly from its saved tensors."""
    rows = {}
    with zipfile.ZipFile(predictions) as archive:
        for key in mask_conditions():
            data = archive.read(f"external/{key}_tensors.npy")
            tensor = np.load(io.BytesIO(data), allow_pickle=False)
            if tensor.shape != (2000, 3, 6) or not np.isfinite(tensor).all():
                raise ValueError(f"invalid tensor array for {key}")
            stored = np.linalg.norm(tensor.reshape(len(tensor), -1), axis=1)
            squared = tensor.astype(float) ** 2
            cartesian = np.sqrt(np.sum(squared * [1, 1, 1, 2, 2, 2], axis=(1, 2)))
            rows[key] = {
                "voigt_array_norm": norm_summary(stored),
                "full_cartesian_frobenius_norm": norm_summary(cartesian),
                "tensor_sha256": hashlib.sha256(data).hexdigest(),
            }
    return rows
