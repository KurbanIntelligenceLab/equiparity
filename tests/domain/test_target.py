"""Tests for the target parity spectrum."""

from __future__ import annotations

from equiparity.domain.target import DIPOLE, ELASTIC, PIEZOELECTRIC, TARGETS, U0, TargetParity


def test_parity_spectrum() -> None:
    # The paper's spectrum: even scalar -> odd vector -> even tensor -> odd tensor.
    assert U0.parity is TargetParity.EVEN
    assert DIPOLE.parity is TargetParity.ODD
    assert ELASTIC.parity is TargetParity.EVEN
    assert PIEZOELECTRIC.parity is TargetParity.ODD


def test_component_counts_match_irreps() -> None:
    # 2x1o + 1x2o + 1x3o = 2*3 + 5 + 7 = 18
    assert PIEZOELECTRIC.n_components == 18
    # 2x0e + 2x2e + 1x4e = 2 + 2*5 + 9 = 21
    assert ELASTIC.n_components == 21
    assert DIPOLE.n_components == 3
    assert U0.n_components == 1


def test_odd_targets_have_odd_irreps() -> None:
    # Parity-odd targets must carry only odd (o) irreps in their output head.
    assert "o" in DIPOLE.irreps and "e" not in DIPOLE.irreps
    assert "o" in PIEZOELECTRIC.irreps and "e" not in PIEZOELECTRIC.irreps
    # Parity-even targets carry only even (e) irreps.
    assert "e" in U0.irreps and "o" not in U0.irreps
    assert "e" in ELASTIC.irreps and "o" not in ELASTIC.irreps


def test_registry() -> None:
    assert set(TARGETS) == {"U0", "dipole", "elastic", "piezoelectric"}
