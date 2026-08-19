"""Shared per-structure pooling for readout heads: sum (extensive, default) or mean (intensive).

Every core's tensor head accumulates a per-atom (NequIP, MACE, EquiformerV2) or per-edge (Allegro)
contribution into a per-structure total via ``index_add_``. That sum is the repo's committed
behaviour, and it is extensive: ``scripts/experiments/size_consistency.py`` shows the prediction
on a K-replica supercell of a periodic crystal is exactly K times the primitive-cell prediction
(max deviation < 7e-14 across the sweep in ``results/size_consistency.json``). Piezoelectric and
elastic tensors are physically intensive properties, so a mean-pooled readout is offered as an
explicit, opt-in alternative (``pooling: mean``): the same accumulated total divided by the
per-structure unit count -- atoms for NequIP/MACE/EquiformerV2, but EDGES for Allegro, whose
readout is edge-centric (no per-atom message passing; see ``equiparity.models.allegro``).
``pool_per_structure`` is unit-agnostic: the caller passes whichever ``unit_to_graph`` mapping
matches its readout (``batch_index`` for atom-pooled cores, ``edge_struct`` for Allegro), so
"mean" always divides by the correct denominator for that core.

``pooling: sum`` is the default and reproduces every existing committed result and checkpoint
bit-identically: the sum branch below is exactly the original ``out.index_add_(0, unit_to_graph,
per_unit)`` with no extra division.
"""

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
        pooling: ``"sum"`` (default; bit-identical to the original ``index_add_`` behaviour) or
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
