"""Checkpoint selection, exact resume, and invariant robust loss."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from equiparity.domain.experiment import TrainingParams
from equiparity.training.tensor_optimization import fit_tensor, tensor_loss


def test_radial_huber_is_rotation_invariant_and_finite_at_zero() -> None:
    torch.manual_seed(7)
    params = TrainingParams(loss="tensor_huber")
    pred = torch.randn(8, 9, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(pred)
    q, _ = torch.linalg.qr(torch.randn(9, 9, dtype=torch.float64))
    weights = torch.arange(1, 9, dtype=torch.float64)
    torch.testing.assert_close(
        tensor_loss(pred, target, weights, params),
        tensor_loss(pred @ q, target @ q, weights, params),
    )
    loss = tensor_loss(pred, pred.detach(), weights, params)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert loss == 0


def _seed() -> None:
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)


def test_best_weights_restored_and_resume_reproduces_uninterrupted_run(tmp_path: Path) -> None:
    params = TrainingParams(epochs=4, lr=0.02, scheduler_patience=0)
    identity = {"config": "test", "scale": 1.0}

    def run(
        path: Path, interrupt: bool = False, resume: Path | None = None
    ) -> tuple[torch.nn.Module, dict]:
        model = torch.nn.Linear(3, 1)
        calls = 0

        def batches():  # noqa: ANN202
            nonlocal calls
            calls += 1
            if interrupt and calls == 3:
                raise RuntimeError("simulated interruption")
            # All three RNG streams affect training, testing their recovery.
            x = torch.randn(4, 3) + float(np.random.normal()) + random.random()
            yield x, torch.ones(4, 1), torch.ones(4)

        def evaluate() -> dict:
            model.eval()
            # Make the first epoch best; final weights must not be selected.
            step = len(list(path.glob("history.json")))
            if step:
                import json

                step = len(json.loads((path / "history.json").read_text()))
            if resume:
                # Resumed path has no history initially; recovered checkpoint has 2 epochs.
                step = max(step, 2)
            return {"val": {"mae": 1.0 + step}, "train": {"mae": 0.0}}

        result = fit_tensor(model, params, batches, evaluate, path, identity, resume)
        return model, result

    _seed()
    full_model, full = run(tmp_path / "full")
    _seed()
    with pytest.raises(RuntimeError, match="simulated"):
        run(tmp_path / "interrupted", interrupt=True)
    partial = tmp_path / "interrupted/checkpoint_latest.pt"
    resumed_model, resumed = run(tmp_path / "resumed", resume=partial)
    assert full["best_epoch"] == resumed["best_epoch"] == 1
    for k, v in full_model.state_dict().items():
        torch.testing.assert_close(v, resumed_model.state_dict()[k], rtol=0, atol=0)
        torch.testing.assert_close(v, full["checkpoint_best"][k], rtol=0, atol=0)
    for k, v in full["checkpoint_latest"]["model"].items():
        torch.testing.assert_close(v, resumed["checkpoint_latest"]["model"][k], rtol=0, atol=0)
    assert full["history"] == resumed["history"]
    with pytest.raises(ValueError, match="identity differs"):
        fit_tensor(
            torch.nn.Linear(3, 1),
            params,
            lambda: [],
            lambda: {},
            tmp_path / "bad",
            {"scale": 2},
            partial,
        )
