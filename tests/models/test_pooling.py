"""Unit tests for the shared sum/mean pooling helper (``equiparity.models.pooling``).

No core dependency: these test the pure-tensor accumulation logic directly, independent of
which framework's readout calls it.
"""

from __future__ import annotations

import pytest
import torch

from equiparity.models.pooling import pool_per_structure, validate_pooling


def test_validate_pooling_accepts_sum_and_mean() -> None:
    assert validate_pooling("sum") == "sum"
    assert validate_pooling("mean") == "mean"


def test_validate_pooling_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="pooling must be one of"):
        validate_pooling("max")


def test_sum_pooling_matches_original_index_add() -> None:
    """Regression guard: "sum" must be bit-identical to the pre-existing index_add_ readout."""
    torch.manual_seed(0)
    per_unit = torch.randn(7, 4, dtype=torch.float64)
    unit_to_graph = torch.tensor([0, 0, 0, 1, 1, 2, 2], dtype=torch.long)
    n_graphs = 3

    original = torch.zeros(n_graphs, 4, dtype=torch.float64).index_add_(0, unit_to_graph, per_unit)
    pooled = pool_per_structure(per_unit, unit_to_graph, n_graphs, "sum")
    assert torch.equal(pooled, original)  # bit-identical, not just close


def test_mean_pooling_divides_by_per_structure_unit_count() -> None:
    per_unit = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [10.0, 0.0], [-6.0, 6.0], [0.0, 0.0]], dtype=torch.float64
    )
    unit_to_graph = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)  # counts: 2, 3
    pooled = pool_per_structure(per_unit, unit_to_graph, 2, "mean")
    expected = torch.tensor([[2.0, 3.0], [4.0 / 3, 2.0]], dtype=torch.float64)
    assert torch.allclose(pooled, expected, atol=1e-14)


def test_mean_pooling_recovers_sum_when_every_structure_has_one_unit() -> None:
    """One unit per structure: mean == sum (denominator 1), a sanity anchor."""
    torch.manual_seed(1)
    per_unit = torch.randn(5, 3, dtype=torch.float64)
    unit_to_graph = torch.arange(5, dtype=torch.long)
    summed = pool_per_structure(per_unit, unit_to_graph, 5, "sum")
    meaned = pool_per_structure(per_unit, unit_to_graph, 5, "mean")
    assert torch.equal(summed, meaned)


def test_mean_pooling_is_size_invariant_under_replication() -> None:
    """K identical units all mapped to the same structure: mean of K copies == the unit value,
    matching the supercell size-consistency property scripts/f3_size_consistency.py measures
    end-to-end (this is the same algebraic fact restated as a pure-tensor unit test)."""
    unit_value = torch.tensor([2.0, -1.0, 5.0], dtype=torch.float64)
    for k in (1, 2, 3, 8):
        per_unit = unit_value.unsqueeze(0).repeat(k, 1)
        unit_to_graph = torch.zeros(k, dtype=torch.long)
        pooled = pool_per_structure(per_unit, unit_to_graph, 1, "mean")
        assert torch.allclose(pooled[0], unit_value, atol=1e-14)
        summed = pool_per_structure(per_unit, unit_to_graph, 1, "sum")
        assert torch.allclose(summed[0], k * unit_value, atol=1e-14)


def test_pool_per_structure_rejects_unknown_mode() -> None:
    per_unit = torch.zeros(2, 3, dtype=torch.float64)
    unit_to_graph = torch.tensor([0, 0], dtype=torch.long)
    with pytest.raises(ValueError, match="pooling must be one of"):
        pool_per_structure(per_unit, unit_to_graph, 1, "max")
