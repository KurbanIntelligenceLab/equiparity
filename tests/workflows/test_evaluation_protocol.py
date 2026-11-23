"""Test authorization must bind both scientific settings and data bytes."""

from dataclasses import replace
from pathlib import Path

import pytest

from equiparity.io.config import load_experiment_config
from equiparity.workflows.evaluation_protocol import (
    lock_evaluation_protocol,
    scientific_digest,
    verify_test_authorization,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/crystal/tensor_huber_nequip_piezoelectric_o3_seed0.yaml"


def test_lock_rejects_changes_and_overwrite(tmp_path: Path) -> None:
    cfg = load_experiment_config(CONFIG)
    for path in (cfg.processed_npz, cfg.split_npz):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixed-data")
    lock = tmp_path / "protocol.json"
    result = lock_evaluation_protocol(tmp_path, [CONFIG], lock)
    audit = result["configs"][0]
    assert len(verify_test_authorization(cfg, audit, lock)) == 64
    assert scientific_digest(replace(cfg, output_dir=Path("elsewhere"))) == scientific_digest(cfg)
    with pytest.raises(ValueError, match="not uniquely registered"):
        verify_test_authorization(replace(cfg, seed=2), audit, lock)
    with pytest.raises(ValueError, match="mismatch"):
        verify_test_authorization(cfg, dict(audit, data_sha256="changed"), lock)
    with pytest.raises(FileExistsError):
        lock_evaluation_protocol(tmp_path, [CONFIG], lock)


def test_test_partition_requires_lock_before_loading_any_data(tmp_path: Path) -> None:
    from equiparity.workflows.evaluate_checkpoint import evaluate_checkpoint

    with pytest.raises(ValueError, match="requires a protocol lock"):
        evaluate_checkpoint(tmp_path, tmp_path, tmp_path / "result", "cpu", "test")
    assert not (tmp_path / "result").exists()
