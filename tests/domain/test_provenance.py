"""Tests for RunManifest provenance record."""

from __future__ import annotations

from equiparity.domain.provenance import RunManifest


def _manifest(**overrides: object) -> RunManifest:
    base: dict[str, object] = {
        "git_sha": "abcdef1234567890",
        "git_dirty": False,
        "config_hash": "sha256:0011223344556677",
        "config_path": "configs/exp.yaml",
        "dataset_manifest": "data/manifests/qm9.yaml",
        "split_manifest": "data/splits/qm9.yaml",
        "seed": 42,
        "python_version": "3.12.3",
        "package_version": "0.1.0",
        "timestamp_utc": "2026-07-04T12:00:00+00:00",
        "hostname": "host",
        "gpu_model": "RTX 5090",
        "cuda_version": "12.8",
        "driver_version": "610.43.02",
    }
    base.update(overrides)
    return RunManifest(**base)  # type: ignore[arg-type]


def test_experiment_id_format() -> None:
    manifest = _manifest()
    # <short_git_sha>_<config_hash_prefix>_<safe_timestamp>
    assert manifest.experiment_id == "abcdef12_00112233_20260704T1200000000"


def test_to_dict_roundtrip_keys() -> None:
    data = _manifest().to_dict()
    assert data["seed"] == 42
    assert data["git_dirty"] is False
    assert set(data) == {
        "git_sha",
        "git_dirty",
        "config_hash",
        "config_path",
        "dataset_manifest",
        "split_manifest",
        "seed",
        "python_version",
        "package_version",
        "timestamp_utc",
        "hostname",
        "gpu_model",
        "cuda_version",
        "driver_version",
    }
