"""Tests for the centrosymmetric space-group filter."""

from __future__ import annotations

from equiparity.domain.spacegroup import CENTROSYMMETRIC_SPACE_GROUPS, is_centrosymmetric


def test_count_is_92() -> None:
    # The 11 centrosymmetric Laue classes contain exactly 92 space groups.
    assert len(CENTROSYMMETRIC_SPACE_GROUPS) == 92


def test_known_centrosymmetric() -> None:
    assert is_centrosymmetric(2)  # P-1
    assert is_centrosymmetric(225)  # Fm-3m (rocksalt, e.g. NaCl)
    assert is_centrosymmetric(221)  # Pm-3m (perovskite cubic)


def test_known_non_centrosymmetric() -> None:
    assert not is_centrosymmetric(1)  # P1
    assert not is_centrosymmetric(216)  # F-43m (zincblende)
    assert not is_centrosymmetric(186)  # P6_3mc (wurtzite)
    assert not is_centrosymmetric(160)  # R3m
