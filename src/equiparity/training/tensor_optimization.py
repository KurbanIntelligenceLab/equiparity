"""Crystal tensor objectives, validation selection, logging, and recoverable state."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from equiparity.domain.experiment import TrainingParams
from equiparity.training.run_instrument import state_cpu


def tensor_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    params: TrainingParams,
) -> torch.Tensor:
    """Use a Frobenius/radial loss so changing tensor orientation leaves the loss unchanged."""
    residual = pred - target
    squared = residual.square().mean(dim=-1)
    if params.loss == "mse":
        per_row = squared
    else:
        radius = torch.sqrt(squared.clamp_min(torch.finfo(squared.dtype).tiny))
        delta = params.huber_delta
        per_row = torch.where(radius <= delta, 0.5 * squared, delta * (radius - 0.5 * delta))
    return (per_row * weights).mean()


def atomic_save(payload: Any, path: Path) -> None:  # noqa: ANN401
    """Only expose complete checkpoint files to external backup processes."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json(payload: Any, path: Path) -> None:  # noqa: ANN401
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def fit_tensor(
    model: torch.nn.Module,
    params: TrainingParams,
    batches: Callable[[], Iterable[tuple[Any, torch.Tensor, torch.Tensor]]],
    evaluate: Callable[[], dict[str, Any]],
    run_dir: Path,
    identity: dict[str, Any],
    resume: Path | None = None,
) -> dict[str, Any]:
    """Fit to training batches, select only by validation MAE, and restore best weights.

    ``evaluate`` must return train/val diagnostics, never test metrics. Identity binds
    the resume file to configuration, data, normalization and model construction.
    Checkpoints contain optimizer/scheduler/RNG state at completed epoch boundaries.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=params.scheduler_patience,
            min_lr=params.min_lr,
        )
        if params.scheduler == "plateau"
        else None
    )
    best, best_epoch, start = float("inf"), 0, 0
    best_state = None
    history: list[dict[str, Any]] = []
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("checkpoint identity differs from config/data/model; resume rejected")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler:
            scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        best, best_epoch, best_state = state["best_val"], state["best_epoch"], state["best_model"]
        history, start = state["history"], state["epoch"]
        if start >= params.epochs:
            raise ValueError("checkpoint already completed the requested epoch budget")
    latest: dict[str, Any] = {}
    for epoch in range(start, params.epochs):
        model.train()
        loss_sum, n_rows, grad_max = 0.0, 0, 0.0
        for batch, targets, weights in batches():
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch).to(targets.dtype)
            loss = tensor_loss(pred, targets, weights, params)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), params.gradient_clip, error_if_nonfinite=True
            )
            grad_max = max(grad_max, float(grad))
            optimizer.step()
            loss_sum += float(loss.detach()) * len(targets)
            n_rows += len(targets)
        diagnostics = evaluate()
        val = float(diagnostics["val"]["mae"])
        if not np.isfinite(val):
            raise RuntimeError("nonfinite validation MAE")
        if val < best:
            best, best_epoch, best_state = val, epoch + 1, state_cpu(model)
        lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step(val)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": loss_sum / max(n_rows, 1),
                "lr": lr,
                "gradient_norm_max_before_clip": grad_max,
                **diagnostics,
            }
        )
        latest = {
            "model": state_cpu(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "rng": rng_state(),
            "epoch": epoch + 1,
            "identity": identity,
            "best_val": best,
            "best_epoch": best_epoch,
            "best_model": best_state,
            "history": history,
        }
        # The latest file is self-contained even if interrupted between the two writes.
        atomic_save(latest, run_dir / "checkpoint_latest.pt")
        atomic_save(
            {"model": best_state, "epoch": best_epoch, "identity": identity},
            run_dir / "checkpoint_selected.pt",
        )
        atomic_json(history, run_dir / "history.json")
    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    return {
        "history": history,
        "best_epoch": best_epoch,
        "checkpoint_best": best_state,
        "checkpoint_latest": latest,
    }
