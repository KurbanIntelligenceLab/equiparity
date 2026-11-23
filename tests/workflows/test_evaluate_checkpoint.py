"""Selected-checkpoint evaluation must be reproducible and validation-only."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from equiparity.domain.experiment import ExperimentConfig, ModelHyperparams, TrainingParams
from equiparity.domain.parity import ParityMode
from equiparity.domain.target import TARGETS
from equiparity.training.crystal_regression import tensor_metrics
from equiparity.workflows import evaluate_checkpoint as workflow


def test_reload_checks_data_identity_and_never_requests_test_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("e3nn")
    config = ExperimentConfig(
        seed=0,
        core="nequip",
        parity=ParityMode.O3,
        target="piezoelectric",
        dataset="fixture",
        processed_npz=Path("data.npz"),
        split_npz=Path("split.npz"),
        output_dir=Path("unused"),
        protocol="validation_selected",
        model=ModelHyperparams(pooling="mean", tensor_head="complete"),
        training=TrainingParams(device="cpu"),
    )
    monkeypatch.setattr(workflow, "load_experiment_config", lambda _: config)
    audit = {"data_sha256": "data", "split_sha256": "split"}
    monkeypatch.setattr(workflow, "validate_tensor_dataset", lambda *_: audit.copy())
    monkeypatch.setattr(workflow, "load_crystal_dataset", lambda *_: None)
    labels = {
        "train-a": np.ones(18),
        "train-b": np.ones(18) * 3,
        "val-a": np.ones(18) * 2,
        "val-b": np.ones(18) * 4,
    }
    requested = []

    def split(_: Path, partition: str) -> np.ndarray:
        assert partition in ("train", "val")
        requested.append(partition)
        return np.array([f"{partition}-a", f"{partition}-b"])

    class Dataset:
        def __init__(self, _: object, ids: np.ndarray) -> None:
            self.ids = ids

        def __len__(self) -> int:
            return len(self.ids)

        def __getitem__(self, index: int) -> SimpleNamespace:
            return SimpleNamespace(structure=self.ids[index])

    monkeypatch.setattr(workflow, "load_split", split)
    monkeypatch.setattr(workflow, "CrystalDataset", Dataset)
    monkeypatch.setattr(
        workflow, "_irreps_targets", lambda ds, *_: np.stack([labels[x] for x in ds.ids])
    )
    monkeypatch.setattr(
        workflow,
        "load_trained",
        lambda *_, **__: SimpleNamespace(
            predict=lambda ids: np.stack([labels[x] / 2 for x in ids])
        ),
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "config_snapshot.yaml").write_text("fixture: true\n")
    torch.save({"identity": audit.copy(), "epoch": 1}, run / "checkpoint_selected.pt")
    target = np.stack([labels["val-a"], labels["val-b"]])
    metrics = tensor_metrics(target / 2, target, TARGETS["piezoelectric"].irreps)
    (run / "history.json").write_text(json.dumps([{"val": metrics}]))
    output = tmp_path / "evaluation"
    result = workflow.evaluate_checkpoint(tmp_path, run, output, "cpu")
    assert requested == ["train", "val"]
    assert result["reproduced_logged_metrics"]
    assert result["metrics"]["mae"] == 1.5
    with np.load(output / "validation_predictions.npz") as saved:
        assert saved["ids"].tolist() == ["val-a", "val-b"]
    audit["data_sha256"] = "changed"
    with pytest.raises(ValueError, match="data_sha256 differs"):
        workflow.evaluate_checkpoint(tmp_path, run, tmp_path / "tampered", "cpu")
