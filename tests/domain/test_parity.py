"""Tests for the ParityMode domain enum."""

from __future__ import annotations

from equiparity.domain.parity import ParityMode


def test_o3_has_parity() -> None:
    assert ParityMode.O3.has_parity is True
    assert ParityMode.SO3.has_parity is False


def test_labels() -> None:
    assert ParityMode.O3.label == "O(3)"
    assert ParityMode.SO3.label == "SO(3)"


def test_string_values_are_stable() -> None:
    # Config files reference these string values; they must not drift.
    assert ParityMode.O3.value == "o3"
    assert ParityMode.SO3.value == "so3"
