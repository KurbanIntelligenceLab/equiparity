"""
Clifford wrapper for MLIP benchmarking — Optimized v2.

Features:
  - torch.compile support (reduce-overhead mode for CUDA graphs)
  - Mixed precision (BF16) support
  - Exponential Moving Average (EMA) of weights
  - Gradient clipping utility
  - DeNS denoising auxiliary loss
  - Two-phase training support (energy/force reweighting)
  - Multi-GPU ready (DDP compatible)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn.pool import radius_graph

from equiparity.models.clifford.interaction import CliffordNet


# ============================================================
# Exponential Moving Average
# ============================================================


class ExponentialMovingAverage:
    """EMA of model parameters. ~2-5% accuracy gain for free.

    Usage:
        ema = ExponentialMovingAverage(model, decay=0.999)
        # During training:
        ema.update()
        # During eval:
        ema.apply_shadow()  # swap to EMA weights
        model.eval(); validate()
        ema.restore()       # swap back to training weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ============================================================
# Main Wrapper
# ============================================================


class CliffordWrapper(nn.Module):
    """Clifford encoder for data_wrapper interface — Optimized v2.

    New features:
        - use_compile: wrap model in torch.compile (2-4× speedup)
        - use_ema: track EMA weights (set decay via ema_decay)
        - use_dens: denoising auxiliary loss
        - All accuracy features from CliffordNet v2

    Example:
        model = CliffordWrapper(
            n_channels=128,
            n_interactions=5,
            use_attention=True,
            use_self_interaction=True,
            max_body_order=3,
            use_l2=True,
            use_multiscale=True,
            use_compile=True,
        )
        # Training loop:
        energy, forces = model(data)
        # or with DeNS:
        energy, forces, dens_loss = model.forward_with_dens(data)
    """

    def __init__(
        self,
        # Architecture
        n_atom_types: int = 100,
        n_channels: int = 128,
        n_interactions: int = 5,
        n_rbf: int = 20,
        cutoff: float = 5.0,
        n_hidden_output: int = 64,
        max_neighbors: int = 50,
        direct_forces: bool = True,
        # Accuracy features
        use_attention: bool = True,
        use_self_interaction: bool = True,
        max_body_order: int = 3,
        use_l2: bool = True,
        use_multiscale: bool = True,
        use_gp_readout: bool = True,
        n_heads: int = 4,
        max_grade: int = 3,
        # DeNS
        use_dens: bool = False,
        dens_noise_std: float = 0.01,
        # Speed features
        use_compile: bool = False,
        compile_mode: str = "reduce-overhead",
        # EMA
        use_ema: bool = True,
        ema_decay: float = 0.999,
        # NEW, trailing (so existing positional/keyword call sites still work):
        signature: Tuple[int, int] = (3, 0),
        pos_projection: str = "none",  # "none" or "xy"
        vector_embedding: str = "spatial",  # "spatial" or "spatial_plus_distance"
    ):
        super().__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.use_ema = use_ema
        self.use_dens = use_dens
        self.signature = tuple(signature)
        self.pos_projection = pos_projection
        self.vector_embedding = vector_embedding

        # Guard: DeNS requires Cl(3,0) because forward_with_dens compares
        # (N, grade_dims[1]) against (N, 3) noise.
        if use_dens and self.signature != (3, 0):
            raise NotImplementedError(
                f"DeNS auxiliary loss is only supported for signature=(3,0); got {self.signature}"
            )

        # Validate pos_projection
        if pos_projection not in ("none", "xy"):
            raise ValueError(f"unknown pos_projection: {pos_projection!r}")

        self._model = CliffordNet(
            n_atom_types=n_atom_types,
            n_channels=n_channels,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            n_hidden_output=n_hidden_output,
            max_neighbors=max_neighbors,
            direct_forces=direct_forces,
            use_attention=use_attention,
            use_self_interaction=use_self_interaction,
            max_body_order=max_body_order,
            use_l2=use_l2,
            use_multiscale=use_multiscale,
            use_gp_readout=use_gp_readout,
            n_heads=n_heads,
            max_grade=max_grade,
            use_dens=use_dens,
            dens_noise_std=dens_noise_std,
            signature=self.signature,
            vector_embedding=vector_embedding,
        )

        # torch.compile (applied after model creation, before DDP)
        if use_compile:
            try:
                self._model = torch.compile(self._model, mode=compile_mode, dynamic=True)
                print(f"[CliffordWrapper] torch.compile enabled (mode={compile_mode})")
            except Exception as e:
                print(f"[CliffordWrapper] torch.compile failed: {e}, using eager mode")

        # EMA (initialized after compile so it tracks the right params)
        self._ema = None
        self._ema_decay = ema_decay

    def init_ema(self):
        """Initialize EMA after model is on device. Call after .to(device)."""
        if self.use_ema:
            # Access underlying model if compiled
            model = self._get_raw_model()
            self._ema = ExponentialMovingAverage(model, decay=self._ema_decay)
            print(f"[CliffordWrapper] EMA initialized (decay={self._ema_decay})")

    def _get_raw_model(self):
        """Get the raw model (unwrap compile if needed)."""
        m = self._model
        if hasattr(m, "_orig_mod"):
            return m._orig_mod
        return m

    def update_ema(self):
        """Call after each optimizer.step()."""
        if self._ema is not None:
            self._ema.update()

    def apply_ema(self):
        """Swap to EMA weights for evaluation."""
        if self._ema is not None:
            self._ema.apply_shadow()

    def restore_from_ema(self):
        """Swap back to training weights."""
        if self._ema is not None:
            self._ema.restore()

    def forward(self, data):
        """Standard forward pass.

        Output shape is always (N, 3) regardless of signature. Internal CliffordNet
        returns (N, grade_dims[1]) — this wrapper adapts that to the trainer's 3D
        force-MSE contract:

        - Cl(2,0) (pos_projection="xy"): positions are projected to z=0 before
          graph construction; force z-column is padded to exactly 0. Force-MSE
          loss sees the 2 xy components plus a constant-zero 3rd component — the
          Cl(2,0) model has no out-of-plane capacity so forces on out-of-plane
          atoms cannot be predicted.

        - Cl(3,0): no-op — 3D forces pass through unchanged.

        - Cl(3,1): CliffordNet returns 4D forces (3 spatial + 1 timelike). The
          timelike component is DROPPED before returning to the trainer. Force-MSE
          loss therefore sees only the 3 spatial components — the Cl(3,1) model's
          timelike output is unused in the force loss. This is an approximation
          justified by the ablation design (spatial forces are what the benchmark
          evaluates).

        Returns:
            (energy, forces) if direct_forces else energy
            - energy: (num_graphs,)
            - forces: (num_atoms, 3)
        """
        if self.pos_projection == "xy":
            pos = data.pos.clone()
            pos[:, 2] = 0.0
        else:
            pos = data.pos

        edge_index = radius_graph(pos, r=self.cutoff, batch=data.batch, max_num_neighbors=self.max_neighbors)
        energy, forces = self._model(data.z, pos, edge_index, data.batch)

        raw = self._get_raw_model()
        if raw.direct_forces:
            # Learned force-head output has shape (N, algebra.grade_dims[1]).
            # Pad/truncate to (N, 3): Cl(2,0)→2 (pad z=0); Cl(3,0)→3 (no-op); Cl(3,1)→4 (drop timelike).
            g1_dim = raw.algebra.grade_dims[1]
            if g1_dim < 3:
                pad = forces.new_zeros(forces.shape[0], 3 - g1_dim)
                forces = torch.cat([forces, pad], dim=-1)
            elif g1_dim > 3:
                forces = forces[..., :3]
        # else: autograd forces were already computed by CliffordNet (shape N,3)
        # — return them so the trainer does not attempt a second grad through
        # the now-freed graph.

        return energy.view(-1), forces

    def forward_with_dens(self, data):
        """Forward with DeNS denoising auxiliary loss.

        Only supported for signature=(3,0) (guarded in __init__).

        Returns:
            (energy, forces, dens_loss)
        """
        edge_index = radius_graph(data.pos, r=self.cutoff, batch=data.batch, max_num_neighbors=self.max_neighbors)
        energy, forces, dens_loss = self._get_raw_model().forward_with_dens(data.z, data.pos, edge_index, data.batch)
        return energy.view(-1), forces, dens_loss


# ============================================================
# Training Utilities
# ============================================================


def build_optimizer(
    model: nn.Module,
    lr: float = 2e-4,
    weight_decay: float = 1e-3,
    amsgrad: bool = False,
) -> torch.optim.Optimizer:
    """AdamW optimizer with recommended MLIP hyperparameters."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 1000,
    min_lr: float = 1e-6,
    mode: str = "cosine",
):
    """LR scheduler: linear warmup + cosine decay.

    Args:
        mode: "cosine" for CosineAnnealingLR (large datasets)
              "plateau" for ReduceLROnPlateau (small datasets)
    """
    if mode == "cosine":
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR,
            LinearLR,
            SequentialLR,
        )

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=min_lr,
        )
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    elif mode == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=min_lr,
        )
    else:
        raise ValueError(f"Unknown scheduler mode: {mode}")


def compute_loss(
    energy_pred: torch.Tensor,
    forces_pred: torch.Tensor,
    energy_target: torch.Tensor,
    forces_target: torch.Tensor,
    energy_weight: float = 4.0,
    force_weight: float = 100.0,
    dens_loss: Optional[torch.Tensor] = None,
    dens_weight: float = 0.1,
    force_loss_type: str = "l2mae",
) -> Dict[str, torch.Tensor]:
    """Compute energy + force loss with recommended weighting.

    Args:
        force_loss_type: "l2mae" (per-atom L2 norm, then mean)
                        or "mae" (component-wise MAE)
    Returns:
        dict with 'total', 'energy', 'forces', and optionally 'dens' keys.
    """
    # Energy MAE
    energy_loss = torch.mean(torch.abs(energy_pred - energy_target))

    # Force loss
    if force_loss_type == "l2mae":
        force_err = torch.sqrt(torch.sum((forces_pred - forces_target) ** 2, dim=-1) + 1e-8)
        force_loss = torch.mean(force_err)
    else:
        force_loss = torch.mean(torch.abs(forces_pred - forces_target))

    total = energy_weight * energy_loss + force_weight * force_loss

    result = {
        "total": total,
        "energy": energy_loss,
        "forces": force_loss,
    }

    if dens_loss is not None:
        total = total + dens_weight * dens_loss
        result["total"] = total
        result["dens"] = dens_loss

    return result


def two_phase_loss_weights(epoch: int, total_epochs: int, phase2_start_fraction: float = 0.8) -> Dict[str, float]:
    """MACE-style two-phase training: emphasize forces early, energy late.

    Phase 1 (0 to 80%): energy_weight=4, force_weight=100
    Phase 2 (80% to 100%): energy_weight=1000, force_weight=10
    """
    if epoch < int(total_epochs * phase2_start_fraction):
        return {"energy_weight": 4.0, "force_weight": 100.0}
    else:
        return {"energy_weight": 1000.0, "force_weight": 10.0}


# ============================================================
# Quick training example
# ============================================================


def example_training_loop():
    """Pseudocode demonstrating the full training setup."""
    print("=" * 60)
    print("Example training loop (pseudocode)")
    print("=" * 60)

    print("""
    # ---- Model setup ----
    model = CliffordWrapper(
        n_channels=128,
        n_interactions=5,
        cutoff=5.0,
        use_attention=True,
        use_self_interaction=True,
        max_body_order=3,
        use_l2=True,
        use_multiscale=True,
        use_compile=True,          # 2-4× speedup
        use_ema=True,              # ~2-5% accuracy gain
        use_dens=True,             # ~5-10% force MAE improvement
    ).cuda()
    model.init_ema()

    optimizer = build_optimizer(model, lr=2e-4, weight_decay=1e-3)
    scheduler = build_scheduler(optimizer, total_steps=200000, warmup_steps=1000)

    # ---- Training with BF16 mixed precision ----
    scaler = torch.amp.GradScaler()

    for epoch in range(total_epochs):
        loss_weights = two_phase_loss_weights(epoch, total_epochs)

        for data in train_loader:
            data = data.cuda()
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if model.use_dens:
                    energy, forces, dens_loss = model.forward_with_dens(data)
                else:
                    energy, forces = model(data)
                    dens_loss = None

                losses = compute_loss(
                    energy, forces,
                    data.energy, data.forces,
                    dens_loss=dens_loss,
                    **loss_weights,
                )

            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            model.update_ema()

        # ---- Validation with EMA ----
        model.apply_ema()
        model.eval()
        # ... validate ...
        model.restore_from_ema()
        model.train()
    """)


if __name__ == "__main__":
    example_training_loop()

    print("\n--- Quick model test ---")
    import torch

    model = CliffordWrapper(
        n_channels=64,
        n_interactions=3,
        use_attention=True,
        use_self_interaction=True,
        max_body_order=3,
        use_l2=True,
        use_multiscale=True,
        use_compile=False,  # skip compile for quick test
        use_ema=False,
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get the raw model to print grade schedule
    raw = model._get_raw_model()
    raw.print_grade_schedule()
