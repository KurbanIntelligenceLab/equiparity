"""Numerical parity-gap table: group orders, known values, and the released record."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from equiparity.verification.parity_gap import (
    PROPER_GENERATORS,
    check_parity_gap_record,
    close_group,
    invariant_dimension,
    parity_gap,
)

REPO = Path(__file__).resolve().parents[2]

ORDERS = {
    "1": 1,
    "2": 2,
    "222": 4,
    "4": 4,
    "422": 8,
    "3": 3,
    "32": 6,
    "6": 6,
    "622": 12,
    "23": 12,
    "432": 24,
}


@pytest.mark.parametrize(("subgroup", "order"), sorted(ORDERS.items()))
def test_proper_subgroups_have_crystallographic_orders(subgroup: str, order: int) -> None:
    group = close_group(PROPER_GENERATORS[subgroup])
    assert len(group) == order
    assert all(np.isclose(np.linalg.det(g), 1.0) for g in group)


def test_trivial_group_leaves_full_tensor_spaces() -> None:
    identity = [np.eye(3)]
    assert invariant_dimension(identity, 1) == 3
    assert invariant_dimension(identity, 3) == 18


def test_known_gaps() -> None:
    assert parity_gap("432", 3) == 0  # m-3m forbids piezoelectricity by rotations alone
    assert parity_gap("23", 3) == 1
    assert parity_gap("1", 3) == 18


def test_released_table_matches_computation() -> None:
    assert check_parity_gap_record(REPO / "results/parity_gap_table.csv") == []


def test_mismatched_record_is_reported(tmp_path: Path) -> None:
    text = (
        (REPO / "results/parity_gap_table.csv")
        .read_text()
        .replace("m-3m,432,cubic,0,0", "m-3m,432,cubic,0,1")
    )
    bad = tmp_path / "gap.csv"
    bad.write_text(text)
    assert check_parity_gap_record(bad) == ["gap.csv: m-3m gap_rank3_piezoelectric = 1, computed 0"]
