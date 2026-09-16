"""The results index must list exactly the released records."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from equiparity.io.records import check_index, find_repo_root, get_record, load_records

REPO = Path(__file__).resolve().parents[2]


def test_index_is_complete() -> None:
    assert check_index(REPO) == []


def test_every_manuscript_display_item_has_a_record() -> None:
    items = {i for r in load_records(REPO) for i in r.manuscript_items}
    for item in (
        "Table 1",
        "Figure 1c",
        "Figure 2a",
        "Figure 2b",
        "Figure 3a",
        "Figure 3b",
        "Figure 3c",
        "Supplementary Figure 1",
        "Supplementary Table 1",
        "Supplementary Table 2",
        "Supplementary Table 3",
        "Supplementary Table 4",
    ):
        assert any(i.startswith(item) for i in items), item


def test_lookup_accepts_stem_and_file_name() -> None:
    assert get_record(REPO, "stats") == get_record(REPO, "stats.json")
    with pytest.raises(KeyError):
        get_record(REPO, "no_such_record")


def test_root_discovery_walks_up(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    shutil.copy(REPO / "results/index.json", tmp_path / "results")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == tmp_path.resolve()


def test_unindexed_and_missing_files_are_reported(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    index = {
        "records": [{"file": "gone.json", "contents": "", "manuscript_items": [], "producer": ""}]
    }
    (tmp_path / "results/index.json").write_text(json.dumps(index))
    (tmp_path / "results/extra.csv").write_text("")
    assert check_index(tmp_path) == [
        "indexed but missing: results/gone.json",
        "not indexed: results/extra.csv",
    ]


def test_every_record_backs_a_manuscript_item() -> None:
    assert [r.file for r in load_records(REPO) if not r.manuscript_items] == []
