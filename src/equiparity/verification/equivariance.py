"""Core-agnostic equivariance probe and parity classification.

A model is probed through a ``Featurizer``: a callable that maps atomic positions to the model's
internal equivariant node features plus their e3nn irreps. The gate transforms the input by a
random proper rotation and a random improper reflection, then measures how far the recomputed
features deviate from the parity-aware prediction ``features @ D(g).T``.

- rotation error small in BOTH arms (every model must be rotation-equivariant),
- reflection error small => O(3) (parity respected),
- reflection error large => genuine SO(3) (parity violated).

Thresholds: float64 O(3) < 1e-12, SO(3) > 1e-4; float32 O(3) < 1e-5, SO(3) > 1e-2. Anything
between the two reflection bounds, or a failed rotation check, is a FAIL requiring investigation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from e3nn.o3 import Irreps

# positions (n_atoms, 3) -> (node features (n_atoms, irreps_dim), irreps of those features)
Featurizer = Callable[[np.ndarray], "tuple[torch.Tensor, Irreps]"]


@dataclass(frozen=True, slots=True)
class ParityThresholds:
    """Reflection/rotation error bounds for classifying a model's symmetry."""

    rotation_max: float
    o3_reflection_max: float
    so3_reflection_min: float


THRESHOLDS: dict[str, ParityThresholds] = {
    "float64": ParityThresholds(
        rotation_max=1e-12, o3_reflection_max=1e-12, so3_reflection_min=1e-4
    ),
    "float32": ParityThresholds(rotation_max=1e-5, o3_reflection_max=1e-5, so3_reflection_min=1e-2),
}


def classify(rotation_error: float, reflection_error: float, thresholds: ParityThresholds) -> str:
    """Classify a model as ``O3``, ``SO3``, or ``FAIL`` from its equivariance errors."""
    if rotation_error > thresholds.rotation_max:
        return "FAIL"
    if reflection_error < thresholds.o3_reflection_max:
        return "O3"
    if reflection_error > thresholds.so3_reflection_min:
        return "SO3"
    return "FAIL"


@dataclass(frozen=True, slots=True)
class EquivarianceReport:
    """Result of probing one model in one parity mode."""

    label: str
    irreps: str
    rotation_error: float
    reflection_error: float
    n_params: int
    dtype: str

    @property
    def verdict(self) -> str:
        """``O3`` / ``SO3`` / ``FAIL`` under the thresholds for this report's dtype."""
        return classify(self.rotation_error, self.reflection_error, THRESHOLDS[self.dtype])


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of parameters in a torch model."""
    return sum(int(p.numel()) for p in model.parameters())


def _random_orthogonal(seed: int, *, improper: bool) -> np.ndarray:
    """Deterministic random orthogonal 3x3 matrix; ``improper`` gives det = -1."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))  # fix QR sign ambiguity for determinism
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]  # normalize to a proper rotation (det +1)
    if improper:
        q[:, 0] = -q[:, 0]  # flip one axis -> reflection (det -1)
    return q


def _transform_error(
    before: torch.Tensor, after: torch.Tensor, irreps: Irreps, matrix: np.ndarray
) -> float:
    """Max abs deviation of ``after`` from the parity-aware prediction ``before @ D(g).T``."""
    d_matrix = irreps.D_from_matrix(torch.as_tensor(matrix, dtype=before.dtype))
    predicted = before @ d_matrix.T
    return (after - predicted).abs().max().item()


def check_equivariance(
    featurize: Featurizer,
    positions: np.ndarray,
    *,
    label: str,
    n_params: int,
    dtype: str = "float64",
    seed: int = 0,
) -> EquivarianceReport:
    """Probe rotation and reflection equivariance of a model's internal features.

    Args:
        featurize: Maps positions to (node features, irreps) for one model.
        positions: Base structure, shape (n_atoms, 3).
        label: Human-readable name for the probed configuration.
        n_params: Model parameter count, recorded for the report.
        dtype: ``float64`` or ``float32``; selects the classification thresholds.
        seed: Seeds the random proper/improper transforms.

    Returns:
        An :class:`EquivarianceReport` with rotation/reflection errors and a verdict.
    """
    base, irreps = featurize(positions)
    rotation = _random_orthogonal(seed, improper=False)
    reflection = _random_orthogonal(seed + 1, improper=True)
    rotated_feat, _ = featurize(positions @ rotation.T)
    reflected_feat, _ = featurize(positions @ reflection.T)
    return EquivarianceReport(
        label=label,
        irreps=str(irreps),
        rotation_error=_transform_error(base, rotated_feat, irreps, rotation),
        reflection_error=_transform_error(base, reflected_feat, irreps, reflection),
        n_params=n_params,
        dtype=dtype,
    )
