"""Inversion parity modes for orthogonal and proper-rotation models."""

from __future__ import annotations

from enum import StrEnum


class ParityMode(StrEnum):
    """Whether a model's internal irreps carry parity labels.

    O3: full O(3) equivariance, irreps tagged even/odd (e.g. ``1o``, ``2e``).
    SO3: SO(3) equivariance only, no parity labels (all-even-equivalent).
    """

    O3 = "o3"
    SO3 = "so3"

    @property
    def has_parity(self) -> bool:
        """True when features carry parity labels (O(3) mode)."""
        return self is ParityMode.O3

    @property
    def label(self) -> str:
        """Human-readable label for tables and figures."""
        return "O(3)" if self is ParityMode.O3 else "SO(3)"
