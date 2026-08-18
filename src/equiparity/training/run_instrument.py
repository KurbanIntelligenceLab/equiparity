"""Shared timing + checkpoint helpers for every trainer.

Keeps the scalar/vector trainers (U0, dipole) consistent with the tensor trainers: wall-clock
timing + throughput + peak GPU memory, and best-val / latest (resumable) checkpoints -- without
duplicating the boilerplate in each trainer.
"""

from __future__ import annotations

import torch


def state_cpu(model) -> dict:  # noqa: ANN001
    """Detached CPU copy of a model's state_dict (safe to stash for the best checkpoint)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def build_timing(
    *,
    train_seconds: float,
    eval_seconds: float,
    epochs: int,
    n_train: int,
    device,  # noqa: ANN001
    ood_seconds: float = 0.0,
) -> dict[str, float]:
    """Timing/throughput/peak-memory summary for one run."""
    peak = (
        float(torch.cuda.max_memory_allocated() / 1e6)
        if getattr(device, "type", "") == "cuda"
        else 0.0
    )
    return {
        "train_seconds": train_seconds,
        "train_seconds_per_epoch": train_seconds / max(1, epochs),
        "eval_seconds": eval_seconds,
        "ood_seconds": ood_seconds,
        "train_throughput_structs_per_s": n_train * epochs / max(1e-9, train_seconds),
        "peak_gpu_mem_mb": peak,
    }


def build_latest(model, optimizer, epoch: int) -> dict:  # noqa: ANN001
    """Resumable checkpoint: model + optimizer state + epoch (for 'continue training' later)."""
    return {"model": state_cpu(model), "optimizer": optimizer.state_dict(), "epoch": epoch}
