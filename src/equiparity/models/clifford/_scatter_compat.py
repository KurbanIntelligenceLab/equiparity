"""Pure-Python torch_scatter replacements (torch_geometric.utils), so the vendored Clifford code
runs without the compiled ``torch_scatter`` extension (no cu128 wheels). ``scatter`` is a drop-in;
``scatter_softmax(src, index, dim)`` maps to ``softmax(src, index, dim=dim)``."""

from __future__ import annotations

import torch
from torch_geometric.utils import scatter, softmax


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return softmax(src, index, dim=dim)


__all__ = ["scatter", "scatter_softmax"]
