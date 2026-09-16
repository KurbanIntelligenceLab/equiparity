"""The grid generator must reproduce the committed configs byte-for-byte."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from equiparity.workflows.grids import GRIDS, generate_grid

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_generated_grid_matches_committed_configs(name: str, tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    shutil.copy(REPO / "results/zero_injection_sets.json", tmp_path / "results")
    generate_grid(name, tmp_path)
    directory = GRIDS[name].directory
    generated = {p.name: p.read_text() for p in (tmp_path / directory).iterdir()}
    committed = {p.name: p.read_text() for p in (REPO / directory).iterdir()}
    assert generated == committed


def test_main_grid_is_the_84_run_matched_pair_grid(tmp_path: Path) -> None:
    runs = generate_grid("main", tmp_path)
    assert sum(len(v) for v in runs.values()) == 84
    assert len(runs["mace"]) == 24


def test_unknown_grid_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown grid"):
        generate_grid("no-such-grid", tmp_path)
