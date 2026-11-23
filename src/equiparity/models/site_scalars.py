"""Condition geometric messages on time-even, parity-even site magnitudes."""

from __future__ import annotations

from typing import Any

import torch

MAGNITUDE_KEY = "magnetic_magnitude"


class MagnitudeConditionedModel(torch.nn.Module):
    """Inject magnetic moment norms before angular message construction.

    The norm is a spatial scalar and is unchanged by global spin reversal.
    This is a conditional elasticity model, not prediction of magnetic state
    from geometry. Both parity arms use identical scalar conditioning.
    """

    def __init__(self, base: torch.nn.Module, core: str) -> None:
        super().__init__()
        self.base = base
        self.core = core
        self.embedding_name = {
            "nequip": "model.func.type_embed",
            "allegro": "model.func.radial_chemical_embed.type_embed",
            "mace": "node_embedding",
        }[core]
        embedding = dict(base.backbone.named_modules())[self.embedding_name]
        if core == "mace":
            width = embedding.linear.irreps_out.dim
        else:
            key = "node_features" if core == "nequip" else "edge_embedding"
            width = embedding.irreps_out[key].dim
        self.condition = torch.nn.Sequential(
            torch.nn.Linear(2 if core == "allegro" else 1, width, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width, bias=False),
        ).to(dtype=next(base.parameters()).dtype)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """Gate scalar embeddings; tensor transformation laws remain unchanged."""
        if MAGNITUDE_KEY not in batch:
            raise ValueError("conditional model requires per-site magnetic magnitudes")
        moments = batch[MAGNITUDE_KEY].reshape(-1, 1)
        if not torch.isfinite(moments).all() or (moments < 0).any():
            raise ValueError("invalid magnetic magnitudes")
        values = torch.log1p(moments).to(next(self.condition.parameters()).dtype)
        if self.core == "allegro":
            edges = batch["edge_index"]
            values = torch.cat([values[edges[0]], values[edges[1]]], dim=1)
        gate = 1 + self.condition(values)

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:  # noqa: ANN401
            if self.core == "mace":
                return output * gate
            result = dict(output)
            key = "node_features" if self.core == "nequip" else "edge_embedding"
            result[key] = output[key] * gate
            if self.core == "nequip":
                result["node_attrs"] = output["node_attrs"] * gate
            return result

        module = dict(self.base.backbone.named_modules())[self.embedding_name]
        handle = module.register_forward_hook(hook)
        try:
            return self.base(batch)
        finally:
            handle.remove()
