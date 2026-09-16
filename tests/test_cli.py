"""CLI surface: help for every command, JSON output, exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from equiparity.cli import main

REPO = Path(__file__).resolve().parents[1]

COMMANDS = [
    ["run"],
    ["verify"],
    ["records", "list"],
    ["records", "show"],
    ["grid", "generate"],
    ["data", "prepare"],
    ["aggregate"],
    ["manifest"],
]


@pytest.mark.parametrize("command", COMMANDS, ids=" ".join)
def test_every_command_has_help(command: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main([*command, "--help"])
    assert exc.value.code == 0
    assert "usage: equiparity" in capsys.readouterr().out


def test_records_list_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["records", "list", "--root", str(REPO), "--json"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert {"file", "contents", "manuscript_items"} <= set(records[0])


def test_records_show_unknown_exits_nonzero() -> None:
    assert main(["records", "show", "no_such_record", "--root", str(REPO)]) == 2


def test_verify_theory_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify", "--theory", "--root", str(REPO), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == [
        {"check": "theory", "passed": True, "problems": []}
    ]


def test_missing_root_is_a_clean_error(tmp_path: Path) -> None:
    assert main(["records", "list", "--root", str(tmp_path)]) == 1
