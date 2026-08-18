"""Tests for seed control and provenance collection."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from equiparity.reproducibility import (
    collect_provenance,
    hash_config,
    seed_everything,
    write_manifest,
)


def test_seed_everything_is_deterministic() -> None:
    seed_everything(123)
    a_py, a_np = random.random(), np.random.rand(3).tolist()
    seed_everything(123)
    b_py, b_np = random.random(), np.random.rand(3).tolist()
    assert a_py == b_py
    assert a_np == b_np


def test_hash_config_is_prefixed_and_stable() -> None:
    h = hash_config("seed: 42\n")
    assert h.startswith("sha256:")
    assert h == hash_config("seed: 42\n")
    assert h != hash_config("seed: 43\n")


def test_collect_provenance_populates_fields() -> None:
    ts = datetime(2026, 7, 4, 12, 0, tzinfo=UTC)
    manifest = collect_provenance(
        config_text="seed: 42\n",
        config_path="configs/exp.yaml",
        dataset_manifest="data/manifests/qm9.yaml",
        split_manifest="data/splits/qm9.yaml",
        seed=42,
        timestamp=ts,
    )
    assert manifest.seed == 42
    assert manifest.config_hash.startswith("sha256:")
    assert manifest.timestamp_utc == ts.isoformat()
    assert manifest.python_version.startswith("3.12")


def test_write_manifest_refuses_dirty_tree(tmp_path: Path) -> None:
    ts = datetime(2026, 7, 4, 12, 0, tzinfo=UTC)
    manifest = collect_provenance(
        config_text="seed: 42\n",
        config_path="configs/exp.yaml",
        dataset_manifest="data/manifests/qm9.yaml",
        split_manifest="data/splits/qm9.yaml",
        seed=42,
        timestamp=ts,
    )
    # Force the dirty flag to exercise the guard regardless of the working tree state.
    dirty = type(manifest)(**{**manifest.to_dict(), "git_dirty": True})  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        write_manifest(dirty, tmp_path)
    written = write_manifest(dirty, tmp_path, allow_dirty=True)
    assert written.exists()
    assert (tmp_path / "manifest.json").read_text().strip().startswith("{")
