"""The released execution manifest must match what the committed inputs rebuild."""

from __future__ import annotations

import json
from pathlib import Path

from equiparity.workflows.run_manifest import MANIFEST_PATH, build_run_manifest

REPO = Path(__file__).resolve().parents[2]


def test_released_manifest_is_current() -> None:
    released = json.loads((REPO / MANIFEST_PATH).read_text())
    assert released == json.loads(json.dumps(build_run_manifest(REPO)))


def test_manifest_covers_every_run_with_existing_inputs() -> None:
    manifest = build_run_manifest(REPO)
    assert manifest["counts"]["matched-pair grid"] == 84
    assert manifest["counts"]["total"] == len(manifest["runs"])
    labels = {(r["config"]) for r in manifest["runs"]}
    assert len(labels) == len(manifest["runs"])
    for run in manifest["runs"]:
        assert (REPO / run["config"]).is_file()
        assert run["dataset_manifest"] is not None
    assert manifest["software"]["packages"]["mace"]["e3nn"] == "0.4.4"
    assert manifest["software"]["packages"]["nequip"]["e3nn"].startswith("0.6")
