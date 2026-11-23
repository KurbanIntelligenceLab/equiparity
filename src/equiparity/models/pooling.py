"""Sum or mean of per-atom or per-edge tensor contributions within each crystal."""

from __future__ import annotations

import torch

POOLING_MODES = ("sum", "mean")


def validate_pooling(pooling: str) -> str:
    """Raise ``ValueError`` unless ``pooling`` is one of :data:`POOLING_MODES`."""
    if pooling not in POOLING_MODES:
        raise ValueError(f"pooling must be one of {POOLING_MODES}, got {pooling!r}")
    return pooling


def pool_per_structure(
    per_unit: torch.Tensor,
    unit_to_graph: torch.Tensor,
    n_graphs: int,
    pooling: str,
) -> torch.Tensor:
    """Accumulate ``per_unit`` contributions into per-structure totals, summed or averaged.

    Args:
        per_unit: ``(n_units, dim)`` per-atom or per-edge contributions.
        unit_to_graph: ``(n_units,)`` long tensor mapping each unit to its structure index.
        n_graphs: Number of structures in the batch.
        pooling: ``"sum"`` or
            ``"mean"`` (divide by the per-structure unit count -- an intensive readout).

    Returns:
        ``(n_graphs, dim)`` per-structure tensor.
    """
    out = torch.zeros(n_graphs, per_unit.shape[1], dtype=per_unit.dtype, device=per_unit.device)
    out = out.index_add_(0, unit_to_graph, per_unit)
    if pooling == "sum":
        return out
    if pooling == "mean":
        counts = torch.bincount(unit_to_graph, minlength=n_graphs).to(out.dtype)
        counts = counts.clamp(min=1).unsqueeze(
            -1
        )  # guard an empty structure, never hit in practice
        return out / counts
    raise ValueError(f"pooling must be one of {POOLING_MODES}, got {pooling!r}")
