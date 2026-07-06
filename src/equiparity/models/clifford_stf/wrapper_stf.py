"""
CliffordSTF Clifford Wrapper — Ablation-Ready.

Backward compatible: ABLATION_CONFIGS["baseline"] = original CliffordWrapper.
All training utilities (optimizer, scheduler, loss) unchanged.
"""

import copy
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch_geometric.nn.pool import radius_graph

from equiparity.models.clifford.wrapper import (
    ExponentialMovingAverage,
    build_optimizer,
    build_scheduler,
    compute_loss,
    two_phase_loss_weights,
)
from .interaction_stf import CliffordSTF


# ============================================================
# Ablation Configurations
# ============================================================


ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Original model — exact reproduction of CliffordWrapper
    "baseline": dict(
        stf_mode="none",
        use_hodge_forces=False,
        use_adaptive_routing=False,
        routing_mode="none",
        use_cross_track=False,
    ),
    # Phase 1 only: Hodge dual force readout
    "hodge_only": dict(
        stf_mode="none",
        use_hodge_forces=True,
        use_adaptive_routing=False,
        routing_mode="none",
        use_cross_track=False,
    ),
    # STF₂ without Hodge (isolates feature quality from readout)
    "stf2_no_hodge": dict(
        stf_mode="stf2",
        use_hodge_forces=False,
        use_adaptive_routing=False,
        routing_mode="none",
        use_cross_track=True,
    ),
    # STF₂ + Hodge (recommended default)
    "stf2": dict(
        stf_mode="stf2",
        use_hodge_forces=True,
        use_adaptive_routing=False,
        routing_mode="none",
        use_cross_track=True,
    ),
    # STF₂ without cross-track coupling
    "stf2_no_cross": dict(
        stf_mode="stf2",
        use_hodge_forces=True,
        use_adaptive_routing=False,
        routing_mode="none",
        use_cross_track=False,
    ),
    # Full 20D: STF₂ + STF₃
    "stf2_stf3": dict(
        stf_mode="stf2+stf3",
        use_hodge_forces=True,
        use_adaptive_routing=False,
        routing_mode="none",
        use_cross_track=True,
    ),
    # STF₂ + static routing (atom-type based)
    "stf2_static_routing": dict(
        stf_mode="stf2",
        use_hodge_forces=True,
        use_adaptive_routing=True,
        routing_mode="static",
        use_cross_track=True,
    ),
    # STF₂ + learned routing
    "stf2_learned_routing": dict(
        stf_mode="stf2",
        use_hodge_forces=True,
        use_adaptive_routing=True,
        routing_mode="learned",
        use_cross_track=True,
    ),
    # Full model: STF₂+STF₃ + Hodge + learned routing + cross-track
    "full": dict(
        stf_mode="stf2+stf3",
        use_hodge_forces=True,
        use_adaptive_routing=True,
        routing_mode="learned",
        use_cross_track=True,
    ),
    # Full without routing
    "full_no_routing": dict(
        stf_mode="stf2+stf3",
        use_hodge_forces=True,
        use_adaptive_routing=False,
        routing_mode="none",
        use_cross_track=True,
    ),
    # Full without cross-track
    "full_no_cross": dict(
        stf_mode="stf2+stf3",
        use_hodge_forces=True,
        use_adaptive_routing=False,
        routing_mode="none",
        use_cross_track=False,
    ),
    # Clifford-track removed from OUTPUT head. The Clifford track still runs
    # inside the message-passing (it generates STF₂ via grade-1 ⊗ grade-1),
    # but the output head reads only STF invariants for energy and only the
    # per-edge STF₂·r̂ contraction for forces. Requires stf_mode != 'none'.
    # Isolates whether the STF head alone can carry the output signal.
    "stf_only_output": dict(
        stf_mode="stf2+stf3",
        use_hodge_forces=False,       # no grade-2 Hodge dual (clifford-track)
        use_adaptive_routing=False,
        routing_mode="none",
        use_cross_track=True,         # keep internal MP coupling between tracks
        skip_clifford_output=True,    # only new flag: output reads STF tracks only
    ),
}


# ============================================================
# Wrapper
# ============================================================


class CliffordSTFWrapper(nn.Module):
    """CliffordSTF Clifford encoder for data_wrapper interface.

    Drop-in replacement for CliffordWrapper with augmented features.

    Example:
        model = CliffordSTFWrapper(
            n_channels=128,
            n_interactions=5,
            stf_mode="stf2",
            use_hodge_forces=True,
        )
        energy, forces = model(data)
    """

    def __init__(
        self,
        # Architecture (original)
        n_atom_types: int = 100,
        n_channels: int = 128,
        n_interactions: int = 5,
        n_rbf: int = 20,
        cutoff: float = 5.0,
        n_hidden_output: int = 64,
        max_neighbors: int = 50,
        direct_forces: bool = True,
        use_attention: bool = True,
        use_self_interaction: bool = True,
        max_body_order: int = 3,
        use_l2: bool = True,
        use_multiscale: bool = True,
        use_gp_readout: bool = True,
        n_heads: int = 4,
        use_dens: bool = False,
        dens_noise_std: float = 0.01,
        # CliffordSTF features
        stf_mode: str = "stf2",
        use_hodge_forces: bool = True,
        use_adaptive_routing: bool = False,
        routing_mode: str = "none",
        use_cross_track: bool = True,
        skip_clifford_output: bool = False,
        # Speed / EMA
        use_compile: bool = False,
        compile_mode: str = "reduce-overhead",
        use_ema: bool = True,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.use_ema = use_ema
        self.use_dens = use_dens

        self._model = CliffordSTF(
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
            use_dens=use_dens,
            dens_noise_std=dens_noise_std,
            stf_mode=stf_mode,
            use_hodge_forces=use_hodge_forces,
            use_adaptive_routing=use_adaptive_routing,
            routing_mode=routing_mode,
            use_cross_track=use_cross_track,
            skip_clifford_output=skip_clifford_output,
        )

        if use_compile:
            try:
                self._model = torch.compile(
                    self._model, mode=compile_mode, dynamic=True
                )
                print(f"[CliffordSTFWrapper] torch.compile enabled (mode={compile_mode})")
            except Exception as e:
                print(f"[CliffordSTFWrapper] torch.compile failed: {e}")

        self._ema = None
        self._ema_decay = ema_decay

    def init_ema(self):
        if self.use_ema:
            model = self._get_raw_model()
            self._ema = ExponentialMovingAverage(model, decay=self._ema_decay)

    def _get_raw_model(self):
        m = self._model
        if hasattr(m, "_orig_mod"):
            return m._orig_mod
        return m

    def update_ema(self):
        if self._ema is not None:
            self._ema.update()

    def apply_ema(self):
        if self._ema is not None:
            self._ema.apply_shadow()

    def restore_from_ema(self):
        if self._ema is not None:
            self._ema.restore()

    def _build_edges(self, data):
        return radius_graph(
            data.pos, r=self.cutoff, batch=data.batch,
            max_num_neighbors=self.max_neighbors,
        )

    def forward(self, data):
        edge_index = self._build_edges(data)
        energy, forces = self._model(
            data.z, data.pos, edge_index, data.batch
        )
        if self._get_raw_model().direct_forces:
            return energy.view(-1), forces
        return energy.view(-1)

    def forward_with_dens(self, data):
        edge_index = self._build_edges(data)
        energy, forces, dens_loss = self._get_raw_model().forward_with_dens(
            data.z, data.pos, edge_index, data.batch
        )
        return energy.view(-1), forces, dens_loss


# ============================================================
# Factory
# ============================================================


def build_from_ablation(
    config_name: str,
    n_channels: int = 128,
    n_interactions: int = 5,
    **overrides,
) -> CliffordSTFWrapper:
    """Build model from named ablation config.

    Args:
        config_name: key from ABLATION_CONFIGS
        overrides: any parameter to override

    Example:
        model = build_from_ablation("stf2", n_channels=64, cutoff=6.0)
    """
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Available: {list(ABLATION_CONFIGS.keys())}"
        )

    cfg = {**ABLATION_CONFIGS[config_name]}
    cfg.update(overrides)
    cfg["n_channels"] = n_channels
    cfg["n_interactions"] = n_interactions

    return CliffordSTFWrapper(**cfg)


# ============================================================
# Smoke Test
# ============================================================


def smoke_test():
    print("=" * 60)
    print("CliffordSTF Clifford — Smoke Test")
    print("=" * 60)

    device = "cpu"
    torch.manual_seed(42)

    N, C = 10, 32
    n_interactions = 3

    atomic_numbers = torch.randint(1, 50, (N,), device=device)
    pos = torch.randn(N, 3, device=device) * 2.0
    batch = torch.zeros(N, dtype=torch.long, device=device)

    cutoff = 5.0
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j and (pos[i] - pos[j]).norm() < cutoff:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)

    print(f"\nAtoms: {N}, Edges: {edge_index.shape[1]}, Channels: {C}")

    configs_to_test = [
        "baseline", "hodge_only", "stf2_no_hodge", "stf2",
        "stf2_no_cross", "stf2_stf3", "full_no_routing",
    ]

    all_pass = True
    for cfg_name in configs_to_test:
        print(f"\n--- Config: {cfg_name} ---")
        try:
            model = build_from_ablation(
                cfg_name,
                n_channels=C,
                n_interactions=n_interactions,
                cutoff=cutoff,
                use_compile=False,
                use_ema=False,
            ).to(device)

            raw = model._get_raw_model()
            raw.print_config()

            # Forward
            energy, forces = raw(atomic_numbers, pos, edge_index, batch)
            print(f"  Energy: {energy.item():.6f}")
            print(f"  Forces: {forces.shape}, norm={forces.norm():.6f}")

            # Gradient check
            model.zero_grad()
            loss = energy.sum() + forces.norm()
            loss.backward()
            total_p, active_p = 0, 0
            for name, p in model.named_parameters():
                if p.requires_grad:
                    total_p += 1
                    if p.grad is not None and p.grad.abs().sum() > 0:
                        active_p += 1
            grad_ok = active_p == total_p
            print(f"  Gradients: {active_p}/{total_p} {'✓' if grad_ok else '✗'}")

            if not grad_ok:
                all_pass = False
                # List dead parameters
                for name, p in model.named_parameters():
                    if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0):
                        print(f"    DEAD: {name}")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    # Equivariance test on key configs
    print(f"\n--- Equivariance Tests ---")
    for cfg_name in ["baseline", "hodge_only", "stf2", "stf2_stf3"]:
        model = build_from_ablation(
            cfg_name, n_channels=C, n_interactions=n_interactions,
            cutoff=cutoff, use_compile=False, use_ema=False,
        ).to(device)
        raw = model._get_raw_model()
        raw.eval()

        Q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.det(Q) < 0:
            Q[:, 0] *= -1

        with torch.no_grad():
            e1, f1 = raw(atomic_numbers, pos, edge_index, batch)
            pos_rot = pos @ Q.T
            sr, dr = [], []
            for i in range(N):
                for j in range(N):
                    if i != j and (pos_rot[i] - pos_rot[j]).norm() < cutoff:
                        sr.append(i)
                        dr.append(j)
            ei_rot = torch.tensor([sr, dr], dtype=torch.long, device=device)
            e2, f2 = raw(atomic_numbers, pos_rot, ei_rot, batch)

        e_err = (e1 - e2).abs().item()
        f_err = (f2 - f1 @ Q.T).abs().max().item()
        e_ok = e_err < 1e-3
        f_ok = f_err < 1e-3
        print(f"  {cfg_name:20s}: E_err={e_err:.2e} {'✓' if e_ok else '✗'}  "
              f"F_err={f_err:.2e} {'✓' if f_ok else '✗'}")
        if not (e_ok and f_ok):
            all_pass = False

    print("\n" + "=" * 60)
    print(f"{'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    smoke_test()