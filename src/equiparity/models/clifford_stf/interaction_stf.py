"""
CliffordSTF Message Passing — Dual-Track Architecture.

Extends interaction.py with:
  - STF₂/STF₃ tracks propagating as equivariant features through MP
  - Hodge dual force readout (grade-1 + ★grade-2)
  - Per-edge STF₂ force decomposition (L=2 ⊗ L=1 → L=1)
  - Environment-adaptive angular momentum routing
  - Cross-track coupling via equivariant bilinear products only

Backward compatible: stf_mode="none" + use_hodge_forces=False = original model.
"""

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from equiparity.models.clifford._scatter_compat import scatter, scatter_softmax

from equiparity.models.clifford.clifford import (
    ALL_GRADES,
    CliffordAlgebra,
    CliffordLinear,
    CliffordNorm,
    CliffordGateActivation,
    compute_gp_output_grades,
    compute_layer_grades,
    make_scalar_mv,
    make_grades01_mv,
    GRADE_RANGES,
)

from .clifford_stf import (
    CliffordSTFAlgebra,
    CliffordSTFLinear,
    CliffordSTFNorm,
    CliffordSTFGateActivation,
    CL_DIM,
    STF2_DIM,
    STF3_DIM,
    clifford_stf_dim,
    split_aug_mv,
    make_aug_mv,
    pad_cl_to_aug,
    compute_stf2_product,
    compute_stf3_product,
    contract_stf2_vec,
    contract_stf3_vec_to_stf2,
    stf2_norm_sq,
    stf3_norm_sq,
    stf2_inner,
    hodge_star_g2_to_vec,
)

from equiparity.models.clifford.interaction import (
    RadialBasisFunctions,
    CosineCutoff,
    CliffordAttention,
    PerLayerEnergyReadout,
)


# ============================================================
# Edge Embedding — now produces equivariant STF₂ edge features
# ============================================================


class CliffordSTFEdgeEmbedding(nn.Module):
    """Edge features with equivariant L=2 STF₂ track.

    Clifford track: grade-(0,1) multivectors (same as original)
    STF₂ track: direction⊗direction as proper L=2 equivariant features
    """

    def __init__(
        self,
        n_rbf: int = 20,
        n_channels: int = 128,
        cutoff: float = 5.0,
        use_l2: bool = True,
        stf_mode: str = "stf2",
    ):
        super().__init__()
        self.n_channels = n_channels
        self.use_l2 = use_l2
        self.stf_mode = stf_mode
        self.rbf = RadialBasisFunctions(n_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)

        in_dim = n_rbf + (1 if use_l2 else 0)

        self.scalar_net = nn.Sequential(
            nn.Linear(in_dim, n_channels), nn.SiLU(),
            nn.Linear(n_channels, n_channels),
        )
        self.vector_net = nn.Sequential(
            nn.Linear(in_dim, n_channels), nn.SiLU(),
            nn.Linear(n_channels, n_channels),
        )

        if stf_mode != "none":
            self.stf2_net = nn.Sequential(
                nn.Linear(in_dim, n_channels), nn.SiLU(),
                nn.Linear(n_channels, n_channels),
            )

    def forward(self, dist: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        rbf = self.rbf(dist)
        env = self.cutoff_fn(dist)

        if self.use_l2:
            from equiparity.models.clifford.clifford import compute_l2_features
            l2 = compute_l2_features(direction)  # (E, 5): [S_xx, S_xy, S_xz, S_yy, S_yz]
            s_xx, s_xy, s_xz, s_yy, s_yz = l2.unbind(-1)
            s_zz = -s_xx - s_yy
            l2_norm_sq = (
                s_xx ** 2 + s_yy ** 2 + s_zz ** 2
                + 2.0 * (s_xy ** 2 + s_xz ** 2 + s_yz ** 2)
            )
            l2_norm = torch.sqrt(l2_norm_sq.unsqueeze(-1) + 1e-8)  # (E, 1)
            feat = torch.cat([rbf, l2_norm], dim=-1)
        else:
            feat = rbf

        # Grade-0: scalar weights
        g0 = (self.scalar_net(feat) * env.unsqueeze(-1)).unsqueeze(-1)
        # Grade-1: direction × weight
        v_w = self.vector_net(feat) * env.unsqueeze(-1)
        g1 = v_w.unsqueeze(-1) * direction.unsqueeze(-2)

        cl_mv = make_grades01_mv(g0, g1)

        if self.stf_mode == "none":
            return cl_mv

        # STF₂ edge features: equivariant L=2 from direction⊗direction
        # Each channel gets the same geometric STF₂, modulated by learned scalar
        dir_stf2 = compute_stf2_product(direction, direction)  # (E, 5)
        stf2_w = self.stf2_net(feat) * env.unsqueeze(-1)  # (E, C)
        stf2_feat = stf2_w.unsqueeze(-1) * dir_stf2.unsqueeze(-2)  # (E, C, 5)

        parts = [cl_mv, stf2_feat]

        if self.stf_mode == "stf2+stf3":
            # STF₃ initialized to zero in edges (generated during MP)
            stf3_feat = cl_mv.new_zeros(*cl_mv.shape[:-1], STF3_DIM)
            parts.append(stf3_feat)

        return torch.cat(parts, dim=-1)


# ============================================================
# Atom Embedding
# ============================================================


class CliffordSTFAtomEmbedding(nn.Module):
    """Atomic numbers → scalar-only augmented multivectors.

    STF tracks initialized to small noise (not zeros) for gradient flow.
    """

    def __init__(
        self,
        n_atom_types: int = 100,
        n_channels: int = 128,
        stf_mode: str = "stf2",
        init_noise: float = 1e-3,
    ):
        super().__init__()
        self.stf_mode = stf_mode
        self.init_noise = init_noise
        self.embed = nn.Embedding(n_atom_types, n_channels)

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        s = self.embed(atomic_numbers)  # (N, C)
        cl = make_scalar_mv(s)  # (N, C, 8)

        if self.stf_mode == "none":
            return cl

        # Small noise for STF tracks (helps gradient flow)
        D = clifford_stf_dim(self.stf_mode)
        pad_size = D - CL_DIM
        if self.training:
            padding = torch.randn(
                *cl.shape[:-1], pad_size, device=cl.device, dtype=cl.dtype
            ) * self.init_noise
        else:
            padding = cl.new_zeros(*cl.shape[:-1], pad_size)

        return torch.cat([cl, padding], dim=-1)


# ============================================================
# Adaptive Routing
# ============================================================


class AdaptiveRouting(nn.Module):
    """Per-atom angular momentum routing.

    Modes:
        "none": all atoms use full augmented features
        "static": atom-type lookup (H/C/N/O → low, metals → high)
        "learned": MLP from invariant local descriptors

    Output: per-atom soft gate in [0, 1] for each STF track.
    Applied as multiplicative mask (torch.compile friendly).
    """

    def __init__(
        self,
        n_channels: int,
        mode: str = "none",
        stf_mode: str = "stf2",
        n_atom_types: int = 100,
    ):
        super().__init__()
        self.mode = mode
        self.stf_mode = stf_mode
        self.n_tracks = 1 + (1 if stf_mode != "none" else 0) + (
            1 if stf_mode == "stf2+stf3" else 0
        )

        if mode == "static":
            # Learnable per-atom-type routing weights
            # n_tracks - 1 because Clifford track always active
            n_gates = self.n_tracks - 1
            self.type_gates = nn.Embedding(n_atom_types, n_gates)
            nn.init.ones_(self.type_gates.weight)  # default: all tracks on

        elif mode == "learned":
            # Invariant descriptors → gate
            # Inputs: coordination number (1), angular variance (1),
            #         mean distance (1), grade-0 norm (1) = 4
            n_gates = self.n_tracks - 1
            self.gate_net = nn.Sequential(
                nn.Linear(4, n_channels // 4),
                nn.SiLU(),
                nn.Linear(n_channels // 4, n_gates),
                nn.Sigmoid(),
            )

    def forward(
        self,
        aug: torch.Tensor,
        atomic_numbers: Optional[torch.Tensor] = None,
        invariant_desc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply routing masks to augmented multivector.

        Args:
            aug: (N, C, D) augmented multivector
            atomic_numbers: (N,) for static routing
            invariant_desc: (N, 4) for learned routing
        Returns:
            (N, C, D) gated augmented multivector
        """
        if self.mode == "none":
            return aug

        cl, s2, s3 = split_aug_mv(aug, self.stf_mode)

        if self.mode == "static":
            gates = torch.sigmoid(self.type_gates(atomic_numbers))  # (N, n_gates)
        elif self.mode == "learned":
            gates = self.gate_net(invariant_desc)  # (N, n_gates)
        else:
            return aug

        # Apply gates: expand to (N, 1, 1) for broadcasting with (N, C, D_track)
        parts = [cl]  # Clifford always active
        if s2 is not None:
            g2 = gates[:, 0:1].unsqueeze(-1)  # (N, 1, 1)
            parts.append(s2 * g2)
        if s3 is not None:
            g3 = gates[:, 1:2].unsqueeze(-1)
            parts.append(s3 * g3)

        return torch.cat(parts, dim=-1)

    def precompute_geometric_invariants(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        dist: torch.Tensor,
    ) -> torch.Tensor:
        """Compute layer-independent portion of invariant descriptors.

        These terms depend only on positions / edges / distances, not on the
        per-layer features ``h``. Caching them and calling ``combine_with_g0``
        inside the interaction loop avoids recomputing 3 scatter ops per layer.

        Returns: (N, 3) tensor of [coord_num, angular_var, mean_dist]
        """
        src, dst = edge_index
        N = pos.shape[0]

        coord_num = scatter(
            torch.ones_like(dst, dtype=torch.float),
            dst, dim=0, dim_size=N, reduce="sum",
        ).unsqueeze(-1)  # (N, 1)

        mean_dist = scatter(
            dist, dst, dim=0, dim_size=N, reduce="mean",
        ).unsqueeze(-1)  # (N, 1)

        direction = (pos[src] - pos[dst]) / (dist.unsqueeze(-1) + 1e-8)
        mean_dir = scatter(
            direction, dst, dim=0, dim_size=N, reduce="mean",
        )
        mean_dir_norm = torch.sqrt(
            torch.sum(mean_dir ** 2, dim=-1, keepdim=True) + 1e-8
        )
        angular_var = 1.0 - mean_dir_norm  # high if directions spread out

        return torch.cat([coord_num, angular_var, mean_dist], dim=-1)

    def combine_with_g0(
        self,
        geometric_invariants: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Append per-layer g0_norm to cached geometric invariants.

        Returns: (N, 4) tensor of [coord_num, angular_var, mean_dist, g0_norm]
        """
        g0_norm = torch.sqrt(
            torch.sum(h[..., 0] ** 2, dim=-1, keepdim=True) + 1e-8
        ).mean(dim=-1, keepdim=True)  # (N, 1)
        return torch.cat([geometric_invariants, g0_norm], dim=-1)

    def compute_invariant_descriptors(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        dist: torch.Tensor,
        h: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute per-atom invariant descriptors for learned routing.

        Kept for backward compatibility. Prefer
        ``precompute_geometric_invariants`` + ``combine_with_g0`` in hot paths.

        Returns: (N, 4) tensor of [coord_num, angular_var, mean_dist, g0_norm]
        """
        geom = self.precompute_geometric_invariants(pos, edge_index, dist)
        return self.combine_with_g0(geom, h)


# ============================================================
# Message Function — Dual Track
# ============================================================


class CliffordSTFMessageFunction(nn.Module):
    """Dual-track message with GP on Clifford + bilinear on STF tracks.

    Clifford track: same as original (GP dispatch + attention/gating)
    STF₂ track: generated from grade-1 × edge direction (L=1⊗L=1→L=2)
                + propagated from sender STF₂ via scalar modulation
    Cross-track: STF₂ ⊗ grade-1 → grade-1 (L=2⊗L=1→L=1 contraction)
    """

    def __init__(
        self,
        n_channels: int,
        n_rbf: int,
        edge_grades: Tuple[int, ...],
        node_input_grades: Tuple[int, ...],
        gp_output_grades: Tuple[int, ...],
        use_attention: bool = True,
        n_heads: int = 4,
        stf_mode: str = "stf2",
        use_cross_track: bool = True,
    ):
        super().__init__()
        self.alg = CliffordSTFAlgebra()
        self.n_channels = n_channels
        self.node_input_grades = node_input_grades
        self.edge_grades = edge_grades
        self.use_attention = use_attention
        self.stf_mode = stf_mode
        self.use_cross_track = use_cross_track

        # Clifford track projections
        self.pre_edge = CliffordLinear(
            n_channels, n_channels, bias=False, active_grades=edge_grades
        )
        self.proj_he = CliffordLinear(
            n_channels, n_channels, bias=False, active_grades=gp_output_grades
        )
        self.proj_eh = CliffordLinear(
            n_channels, n_channels, bias=False, active_grades=gp_output_grades
        )
        self.skip = nn.Linear(n_channels, n_channels, bias=False)

        if use_attention:
            self.attention = CliffordAttention(n_channels, n_heads, n_rbf)
        else:
            self.rbf = RadialBasisFunctions(n_rbf)
            self.radial_gate = nn.Sequential(
                nn.Linear(n_rbf, n_channels), nn.SiLU(),
                nn.Linear(n_channels, n_channels), nn.Sigmoid(),
            )

        # STF₂ track projections
        if stf_mode != "none":
            self.stf2_proj_sender = nn.Linear(n_channels, n_channels, bias=False)
            self.stf2_proj_gen = nn.Linear(n_channels, n_channels, bias=False)
            # Radial modulation for STF₂ messages (uses pre-computed RBF from block)
            self.stf2_radial = nn.Sequential(
                nn.Linear(n_rbf, n_channels), nn.SiLU(),
                nn.Linear(n_channels, n_channels),
            )

            if use_cross_track:
                # Cross-track: STF₂ contraction with grade-1 → grade-1 feedback
                self.cross_proj = nn.Linear(n_channels, n_channels, bias=False)

        # STF₃ track
        if stf_mode == "stf2+stf3":
            self.stf3_proj_sender = nn.Linear(n_channels, n_channels, bias=False)
            self.stf3_proj_gen = nn.Linear(n_channels, n_channels, bias=False)

    def forward(
        self,
        h_j: torch.Tensor,  # (E, C, D) sender
        h_i: torch.Tensor,  # (E, C, D) receiver
        edge_mv: torch.Tensor,  # (E, C, D) edge features
        dist: torch.Tensor,  # (E,)
        direction: torch.Tensor,  # (E, 3)
        dst: torch.Tensor,  # (E,)
        n_nodes: int,
        rbf_feats: Optional[torch.Tensor] = None,  # (E, n_rbf) pre-computed
    ) -> torch.Tensor:
        D = clifford_stf_dim(self.stf_mode)

        # === Clifford track ===
        h_j_cl = h_j[..., :CL_DIM]
        h_i_cl = h_i[..., :CL_DIM]
        edge_cl = edge_mv[..., :CL_DIM]

        e = self.pre_edge(edge_cl)
        gp_he = self.alg.dispatch_gp(h_j_cl, e, self.node_input_grades, self.edge_grades)
        gp_eh = self.alg.dispatch_gp(e, h_j_cl, self.edge_grades, self.node_input_grades)

        skip_s = self.skip(h_j_cl[..., 0])
        skip_mv = make_scalar_mv(skip_s)

        msg_cl = self.proj_he(gp_he) + self.proj_eh(gp_eh) + skip_mv

        # Cross-track feedback: STF₂ × grade-1 → grade-1
        if self.stf_mode != "none" and self.use_cross_track:
            h_j_s2 = h_j[..., CL_DIM:CL_DIM + STF2_DIM]
            h_j_vec = h_j_cl[..., 1:4]  # grade-1 vectors

            # Contract sender's STF₂ with sender's grade-1 → L=1 contribution
            cross_vec = contract_stf2_vec(h_j_s2, h_j_vec)  # (E, C, 3)
            cross_vec = self.cross_proj(
                cross_vec.transpose(-1, -2)
            ).transpose(-1, -2)  # project over channels

            # Inject into grade-1 of Clifford message
            msg_cl[..., 1:4] = msg_cl[..., 1:4] + cross_vec

        # Attention or radial gating (on Clifford track)
        if self.use_attention:
            attn = self.attention(
                h_i_cl[..., 0], h_j_cl[..., 0], dist, dst, n_nodes
            )
            msg_cl = msg_cl * attn
        else:
            gate = self.radial_gate(self.rbf(dist)).unsqueeze(-1)
            msg_cl = msg_cl * gate

        if self.stf_mode == "none":
            return msg_cl

        # === STF₂ track ===
        assert rbf_feats is not None, (
            "rbf_feats required when stf_mode != 'none'. "
            "Pass pre-computed RBF features from the interaction block."
        )
        h_j_s2 = h_j[..., CL_DIM:CL_DIM + STF2_DIM]
        edge_s2 = edge_mv[..., CL_DIM:CL_DIM + STF2_DIM]

        # Generate L=2 from grade-1 × edge direction
        h_j_vec = h_j_cl[..., 1:4]
        dir_exp = direction.unsqueeze(-2).expand_as(h_j_vec)
        s2_gen = compute_stf2_product(h_j_vec, dir_exp)  # (E, C, 5)

        # Radial modulation (use pre-computed RBF features)
        rbf_w = self.stf2_radial(rbf_feats)  # (E, C)

        # Combine: generated + propagated sender STF₂
        s2_gen_proj = torch.einsum(
            "oc,...ci->...oi", self.stf2_proj_gen.weight, s2_gen
        )
        s2_send_proj = torch.einsum(
            "oc,...ci->...oi", self.stf2_proj_sender.weight, h_j_s2
        )
        msg_s2 = (s2_gen_proj + s2_send_proj) * rbf_w.unsqueeze(-1)

        parts = [msg_cl, msg_s2]

        # === STF₃ track ===
        if self.stf_mode == "stf2+stf3":
            h_j_s3 = h_j[..., CL_DIM + STF2_DIM:]

            # Generate L=3 from sender STF₂ × edge direction
            s3_gen = compute_stf3_product(h_j_s2, dir_exp)  # (E, C, 7)
            s3_gen_proj = torch.einsum(
                "oc,...ci->...oi", self.stf3_proj_gen.weight, s3_gen
            )
            s3_send_proj = torch.einsum(
                "oc,...ci->...oi", self.stf3_proj_sender.weight, h_j_s3
            )
            msg_s3 = (s3_gen_proj + s3_send_proj) * rbf_w.unsqueeze(-1)
            parts.append(msg_s3)

        return torch.cat(parts, dim=-1)


# ============================================================
# Multi-Body Interaction — generates STF from iterated products
# ============================================================


class CliffordSTFMultiBodyInteraction(nn.Module):
    """Multi-body with STF generation from iterated products.

    2-body: W2 · agg
    3-body: GP(agg, agg) on Clifford + STF₂ from grade-1 × grade-1
    4-body: GP(3body, agg) + STF₃ from STF₂ × grade-1
    """

    def __init__(
        self,
        n_channels: int,
        active_grades: Tuple[int, ...] = ALL_GRADES,
        max_body_order: int = 3,
        stf_mode: str = "stf2",
    ):
        super().__init__()
        self.alg = CliffordSTFAlgebra()
        self.max_body_order = max_body_order
        self.stf_mode = stf_mode

        self.w2 = CliffordSTFLinear(
            n_channels, n_channels, bias=False,
            active_grades=active_grades, stf_mode=stf_mode,
        )
        self.w3 = CliffordSTFLinear(
            n_channels, n_channels, bias=False,
            active_grades=active_grades, stf_mode=stf_mode,
        )
        if max_body_order >= 4:
            self.w4 = CliffordSTFLinear(
                n_channels, n_channels, bias=False,
                active_grades=active_grades, stf_mode=stf_mode,
            )

    def forward(self, agg: torch.Tensor) -> torch.Tensor:
        D = clifford_stf_dim(self.stf_mode)
        out = self.w2(agg)

        if self.max_body_order >= 3:
            if self.stf_mode == "none":
                agg_cl = agg
                three_body_cl = self.alg.geometric_product(agg_cl, agg_cl)
                three_body = three_body_cl
            else:
                three_body = self.alg.augmented_product(
                    agg, agg, stf_mode=self.stf_mode
                )
            out = out + self.w3(three_body)

            if self.max_body_order >= 4:
                if self.stf_mode == "none":
                    four_body = self.alg.geometric_product(three_body, agg_cl)
                else:
                    four_body = self.alg.augmented_product(
                        three_body, agg, stf_mode=self.stf_mode
                    )
                out = out + self.w4(four_body)

        return out


# ============================================================
# Self-Interaction
# ============================================================


class CliffordSTFSelfInteraction(nn.Module):
    """Per-node self-interaction via augmented product."""

    def __init__(
        self,
        n_channels: int,
        active_grades: Tuple[int, ...] = ALL_GRADES,
        stf_mode: str = "stf2",
    ):
        super().__init__()
        self.alg = CliffordSTFAlgebra()
        self.stf_mode = stf_mode
        self.proj = CliffordSTFLinear(
            n_channels, n_channels, bias=False,
            active_grades=active_grades, stf_mode=stf_mode,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_proj = self.proj(h)
        if self.stf_mode == "none":
            return self.alg.geometric_product(h_proj, h)
        return self.alg.augmented_product(h_proj, h, stf_mode=self.stf_mode)


# ============================================================
# Update Function
# ============================================================


class CliffordSTFUpdateFunction(nn.Module):
    """Node update with augmented features + residual."""

    def __init__(
        self,
        n_channels: int,
        input_grades: Tuple[int, ...],
        output_grades: Tuple[int, ...],
        use_self_interaction: bool = True,
        stf_mode: str = "stf2",
    ):
        super().__init__()
        self.use_self_interaction = use_self_interaction
        self.stf_mode = stf_mode

        cat_mult = 3 if use_self_interaction else 2
        self.linear_in = CliffordSTFLinear(
            cat_mult * n_channels, n_channels, bias=True,
            active_grades=output_grades, stf_mode=stf_mode,
        )
        self.activation = CliffordSTFGateActivation(
            n_channels, active_grades=output_grades, stf_mode=stf_mode,
        )
        self.linear_out = CliffordSTFLinear(
            n_channels, n_channels, bias=True,
            active_grades=output_grades, stf_mode=stf_mode,
        )
        self.norm = CliffordSTFNorm(
            n_channels, active_grades=output_grades, stf_mode=stf_mode,
        )

        if use_self_interaction:
            self.self_int = CliffordSTFSelfInteraction(
                n_channels, active_grades=output_grades, stf_mode=stf_mode,
            )

    def forward(self, h: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        if self.use_self_interaction:
            si = self.self_int(h)
            combined = torch.cat([h, agg, si], dim=-2)
        else:
            combined = torch.cat([h, agg], dim=-2)

        out = self.linear_in(combined)
        out = self.activation(out)
        out = self.linear_out(out)

        if h.shape[-2] == out.shape[-2]:
            return self.norm(out + h)
        return self.norm(out)


# ============================================================
# Interaction Block
# ============================================================


class CliffordSTFInteractionBlock(nn.Module):
    """Single augmented Clifford MP layer."""

    def __init__(
        self,
        n_channels: int,
        n_rbf: int,
        edge_grades: Tuple[int, ...],
        node_input_grades: Tuple[int, ...],
        node_output_grades: Tuple[int, ...],
        use_attention: bool = True,
        use_self_interaction: bool = True,
        max_body_order: int = 3,
        n_heads: int = 4,
        stf_mode: str = "stf2",
        use_cross_track: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.stf_mode = stf_mode

        # Shared RBF for all sub-modules that need radial features
        if stf_mode != "none":
            self.shared_rbf = RadialBasisFunctions(n_rbf)

        gp_out = compute_gp_output_grades(node_input_grades, edge_grades)

        self.message_fn = CliffordSTFMessageFunction(
            n_channels, n_rbf,
            edge_grades=edge_grades,
            node_input_grades=node_input_grades,
            gp_output_grades=gp_out,
            use_attention=use_attention,
            n_heads=n_heads,
            stf_mode=stf_mode,
            use_cross_track=use_cross_track,
        )

        union_grades = tuple(sorted(set(node_input_grades) | set(gp_out)))

        self.multi_body = CliffordSTFMultiBodyInteraction(
            n_channels,
            active_grades=union_grades,
            max_body_order=max_body_order,
            stf_mode=stf_mode,
        )

        self.update_fn = CliffordSTFUpdateFunction(
            n_channels,
            input_grades=union_grades,
            output_grades=node_output_grades,
            use_self_interaction=use_self_interaction,
            stf_mode=stf_mode,
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mv: torch.Tensor,
        dist: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        h_j = h[src]
        h_i = h[dst]
        N = h.shape[0]

        # Pre-compute RBF features once for all sub-modules
        rbf_feats = self.shared_rbf(dist) if self.stf_mode != "none" else None

        messages = self.message_fn(
            h_j, h_i, edge_mv, dist, direction, dst, N,
            rbf_feats=rbf_feats,
        )

        agg = scatter(messages, dst, dim=0, dim_size=N, reduce="sum")

        agg = self.multi_body(agg)

        return self.update_fn(h, agg)


# ============================================================
# Output Block — Hodge dual + per-edge STF₂ forces
# ============================================================


class CliffordSTFOutputBlock(nn.Module):
    """Energy/force output with Hodge dual and STF₂ force decomposition.

    Energy: grade-0 scalars + GP readout + STF₂ norm invariants + STF₃ norm invariants
    Forces: grade-1 vectors + ★grade-2 (Hodge dual) + per-edge STF₂·r̂ contraction

    If ``skip_clifford_output=True`` the Clifford-track contribution is removed
    from both energy (no grade-0 scalars, no GP readout) and forces (no grade-1
    vectors, no Hodge dual). Only STF invariants and the per-edge STF₂ force
    decomposition drive the output. Requires ``stf_mode != 'none'`` because
    otherwise the output head would have no inputs.
    """

    def __init__(
        self,
        n_channels: int = 128,
        n_hidden: int = 64,
        use_gp_readout: bool = True,
        use_hodge_forces: bool = True,
        stf_mode: str = "stf2",
        skip_clifford_output: bool = False,
    ):
        super().__init__()
        if skip_clifford_output and stf_mode == "none":
            raise ValueError(
                "skip_clifford_output=True requires stf_mode != 'none': "
                "with no Clifford track AND no STF track, there is nothing to "
                "read out."
            )
        self.use_gp_readout = use_gp_readout
        self.use_hodge_forces = use_hodge_forces
        self.stf_mode = stf_mode
        self.skip_clifford_output = skip_clifford_output
        self.alg = CliffordAlgebra()

        if not skip_clifford_output:
            self.pre_mix = CliffordLinear(
                n_channels, n_channels, bias=False, active_grades=ALL_GRADES
            )

        # Energy head input dim
        if skip_clifford_output:
            energy_in = 0
        else:
            energy_in = 2 * n_channels if use_gp_readout else n_channels
        if stf_mode != "none":
            energy_in += n_channels  # STF₂ norm invariant
        if stf_mode == "stf2+stf3":
            energy_in += n_channels  # STF₃ norm invariant

        self.energy_head = nn.Sequential(
            nn.Linear(energy_in, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, 1),
        )

        # Force head: grade-1 (+ Hodge dual of grade-2) = C or 2C vector channels.
        # Skipped entirely when clifford output is off — forces come from per-edge STF₂.
        if not skip_clifford_output:
            force_vec_channels = 2 * n_channels if use_hodge_forces else n_channels
            self.force_head = nn.Linear(force_vec_channels, 1, bias=False)

        # Per-edge STF₂ force: scalar magnitude per edge
        if stf_mode != "none":
            self.stf2_force_head = nn.Sequential(
                nn.Linear(n_channels, n_hidden), nn.SiLU(),
                nn.Linear(n_hidden, 1),
            )

    def forward(
        self,
        h: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        dist: Optional[torch.Tensor] = None,
        direction: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
    ):
        h_cl = h[..., :CL_DIM]

        # === Energy ===
        if self.skip_clifford_output:
            # STF-only output: no grade-0 scalars, no GP readout.
            # scalars starts empty and gets the STF invariants appended below.
            scalars = h_cl.new_empty(*h_cl.shape[:-2], 0)
        elif self.use_gp_readout:
            h_pre = self.pre_mix(h_cl)
            h_mixed = self.alg.geometric_product(h_pre, h_cl)
            scalars = torch.cat([h_cl[..., 0], h_mixed[..., 0]], dim=-1)
        else:
            scalars = h_cl[..., 0]

        # STF invariant norms → energy (cache for force path)
        s2_norm_sq_cached = None
        if self.stf_mode != "none":
            h_s2 = h[..., CL_DIM:CL_DIM + STF2_DIM]
            s2_norm_sq_cached = stf2_norm_sq(h_s2)  # (N, C, 1)
            s2_inv = torch.sqrt(s2_norm_sq_cached + 1e-8).squeeze(-1)  # (N, C)
            scalars = torch.cat([scalars, s2_inv], dim=-1)

        if self.stf_mode == "stf2+stf3":
            h_s3 = h[..., CL_DIM + STF2_DIM:]
            s3_inv = torch.sqrt(stf3_norm_sq(h_s3) + 1e-8).squeeze(-1)
            scalars = torch.cat([scalars, s3_inv], dim=-1)

        atom_energy = self.energy_head(scalars).squeeze(-1)

        if batch is not None:
            # Callers in the hot path should pass num_graphs to avoid a
            # GPU->CPU sync from batch.max().item(). Fallback computes it here.
            if num_graphs is None:
                num_graphs = int(batch.max().item()) + 1
            energy = scatter(atom_energy, batch, dim=0, dim_size=num_graphs, reduce="sum")
        else:
            energy = atom_energy.sum(dim=0, keepdim=True)

        # === Forces ===
        if self.skip_clifford_output:
            # No Clifford-track forces: start from zero, rely on per-edge STF₂ below.
            forces = h_cl.new_zeros(h_cl.shape[0], 3)
        else:
            # Grade-1 vectors
            v_force = h_cl[..., 1:4]  # (N, C, 3)

            if self.use_hodge_forces:
                # Hodge dual of grade-2 bivectors → second L=1 channel
                pv_force = hodge_star_g2_to_vec(h_cl)  # (N, C, 3)
                combined_v = torch.cat([v_force, pv_force], dim=-2)  # (N, 2C, 3)
            else:
                combined_v = v_force

            # Linear projection over channels → (N, 1, 3) → (N, 3)
            forces = self.force_head(
                combined_v.transpose(-1, -2)
            ).squeeze(-1)  # (N, 3)

        # Per-edge STF₂ force contribution
        if self.stf_mode != "none" and edge_index is not None and direction is not None:
            src, dst = edge_index
            h_s2_src = h[src, :, CL_DIM:CL_DIM + STF2_DIM]  # (E, C, 5)
            h_s2_dst = h[dst, :, CL_DIM:CL_DIM + STF2_DIM]

            # Contract sender STF₂ with edge direction → L=1 force contribution
            dir_exp = direction.unsqueeze(-2).expand_as(h_s2_src[..., :3])
            f_s2_src = contract_stf2_vec(h_s2_src, dir_exp)  # (E, C, 3)
            f_s2_dst = contract_stf2_vec(h_s2_dst, dir_exp)

            # Scalar magnitude from invariant features (reuse cached node norms)
            s2_inv_edge = torch.sqrt(
                s2_norm_sq_cached[src].squeeze(-1) + 1e-8
            )  # (E, C)
            s2_weight = self.stf2_force_head(s2_inv_edge).unsqueeze(-1)  # (E, 1, 1)

            edge_force = s2_weight * (f_s2_src + f_s2_dst)  # (E, C, 3)
            # Sum over channels → (E, 3), then aggregate per atom
            edge_force = edge_force.mean(dim=-2)  # (E, 3)
            stf2_forces = scatter(
                edge_force, dst, dim=0, dim_size=h.shape[0], reduce="sum"
            )
            forces = forces + stf2_forces

        return energy, forces


# ============================================================
# Full Model
# ============================================================


class CliffordSTF(nn.Module):
    """CliffordSTF GNN — Cl(3,0) + STF₂ + STF₃.

    Ablation flags:
        stf_mode: "none" / "stf2" / "stf2+stf3"
        use_hodge_forces: grade-1 + ★grade-2 force readout
        use_adaptive_routing: per-atom L_max selection
        routing_mode: "none" / "static" / "learned"
        use_cross_track: Clifford ↔ STF bilinear coupling
        + all original flags from CliffordGNN
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
        # Original accuracy features
        use_attention: bool = True,
        use_self_interaction: bool = True,
        max_body_order: int = 3,
        use_l2: bool = True,
        use_multiscale: bool = True,
        use_gp_readout: bool = True,
        n_heads: int = 4,
        # DeNS
        use_dens: bool = False,
        dens_noise_std: float = 0.01,
        # === New augmented features ===
        stf_mode: str = "stf2",
        use_hodge_forces: bool = True,
        use_adaptive_routing: bool = False,
        routing_mode: str = "none",
        use_cross_track: bool = True,
        skip_clifford_output: bool = False,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.direct_forces = direct_forces
        self.n_channels = n_channels
        self.use_multiscale = use_multiscale
        self.use_dens = use_dens
        self.dens_noise_std = dens_noise_std
        self.stf_mode = stf_mode
        self.use_adaptive_routing = use_adaptive_routing
        self.routing_mode = routing_mode
        self.skip_clifford_output = skip_clifford_output

        self.atom_embed = CliffordSTFAtomEmbedding(
            n_atom_types, n_channels, stf_mode=stf_mode
        )
        self.edge_embed = CliffordSTFEdgeEmbedding(
            n_rbf, n_channels, cutoff, use_l2=use_l2, stf_mode=stf_mode,
        )

        # Grade schedule
        edge_grades = (0, 1)
        layer_output_grades = compute_layer_grades(n_interactions, edge_grades)

        # Interaction blocks
        self.interactions = nn.ModuleList()
        node_grades: Tuple[int, ...] = (0,)
        for i in range(n_interactions):
            out_grades = layer_output_grades[i]
            self.interactions.append(
                CliffordSTFInteractionBlock(
                    n_channels, n_rbf,
                    edge_grades=edge_grades,
                    node_input_grades=node_grades,
                    node_output_grades=out_grades,
                    use_attention=use_attention,
                    use_self_interaction=use_self_interaction,
                    max_body_order=max_body_order,
                    n_heads=n_heads,
                    stf_mode=stf_mode,
                    use_cross_track=use_cross_track,
                )
            )
            node_grades = out_grades

        # Multi-scale readout
        if use_multiscale:
            self.layer_readouts = nn.ModuleList([
                PerLayerEnergyReadout(n_channels, n_hidden_output)
                for _ in range(n_interactions)
            ])

        self.output = CliffordSTFOutputBlock(
            n_channels, n_hidden_output,
            use_gp_readout=use_gp_readout,
            use_hodge_forces=use_hodge_forces,
            stf_mode=stf_mode,
            skip_clifford_output=skip_clifford_output,
        )

        # Adaptive routing
        if use_adaptive_routing:
            self.router = AdaptiveRouting(
                n_channels, mode=routing_mode,
                stf_mode=stf_mode, n_atom_types=n_atom_types,
            )

        self._grade_schedule = layer_output_grades

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_vec: Optional[torch.Tensor] = None,
    ):
        if not self.direct_forces:
            pos = pos.clone().requires_grad_(True)

        src, dst = edge_index
        # PBC: honor a precomputed edge_vec (periodic-image displacement) instead of raw
        # pos[dst]-pos[src], which breaks centrosymmetry for crystals (the MACE lesson).
        rel_pos = edge_vec if edge_vec is not None else pos[dst] - pos[src]
        dist = torch.sqrt(torch.sum(rel_pos ** 2, dim=-1) + 1e-8)
        direction = rel_pos / (dist.unsqueeze(-1) + 1e-8)

        h = self.atom_embed(atomic_numbers)
        edge_mv = self.edge_embed(dist, direction)

        # Compute num_graphs once and thread it through the scatter-using
        # output / per-layer readouts, so the per-layer `batch.max().item()`
        # GPU->CPU sync happens at most once per forward.
        num_graphs = None
        if batch is not None and (self.use_multiscale or True):
            num_graphs = int(batch.max().item()) + 1

        # Precompute geometric routing invariants once per forward. Three of
        # the four descriptor terms depend only on (pos, edge_index, dist) and
        # were being recomputed in every interaction layer.
        geometric_invariants_cache = None
        if self.use_adaptive_routing and self.routing_mode == "learned":
            geometric_invariants_cache = (
                self.router.precompute_geometric_invariants(pos, edge_index, dist)
            )

        layer_energies = []
        for i, interaction in enumerate(self.interactions):
            h = interaction(h, edge_index, edge_mv, dist, direction)

            # Adaptive routing after each layer
            if self.use_adaptive_routing:
                if self.routing_mode == "static":
                    h = self.router(h, atomic_numbers=atomic_numbers)
                elif self.routing_mode == "learned":
                    inv_desc = self.router.combine_with_g0(
                        geometric_invariants_cache, h[..., :CL_DIM]
                    )
                    h = self.router(h, invariant_desc=inv_desc)

            if self.use_multiscale:
                # Multi-scale reads from Clifford grade-0 only
                layer_energies.append(
                    self.layer_readouts[i](h[..., :CL_DIM], batch, num_graphs=num_graphs)
                )

        energy, forces = self.output(
            h, batch, edge_index, dist, direction, num_graphs=num_graphs
        )

        if self.use_multiscale and layer_energies:
            for le in layer_energies:
                energy = energy + le

        if not self.direct_forces:
            forces = -torch.autograd.grad(
                energy.sum(), pos,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]

        return energy, forces

    def forward_with_dens(
        self,
        atomic_numbers: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ):
        """Forward with DeNS denoising auxiliary loss."""
        energy, forces = self.forward(atomic_numbers, pos, edge_index, batch)

        dens_loss = torch.tensor(0.0, device=pos.device)
        if self.training and self.use_dens:
            noise = torch.randn_like(pos) * self.dens_noise_std
            pos_noisy = pos + noise

            rel_pos_n = pos_noisy[edge_index[1]] - pos_noisy[edge_index[0]]
            dist_n = torch.sqrt(torch.sum(rel_pos_n ** 2, dim=-1) + 1e-8)
            direction_n = rel_pos_n / (dist_n.unsqueeze(-1) + 1e-8)

            h_n = self.atom_embed(atomic_numbers)
            edge_mv_n = self.edge_embed(dist_n, direction_n)

            for interaction in self.interactions:
                h_n = interaction(h_n, edge_index, edge_mv_n, dist_n, direction_n)

            # Predict noise from all L=1 channels
            h_cl_n = h_n[..., :CL_DIM]
            noise_pred = h_cl_n[..., 1:4].mean(dim=-2)  # grade-1 → (N, 3)

            if self.use_hodge_forces:
                hodge_pred = hodge_star_g2_to_vec(h_cl_n).mean(dim=-2)
                noise_pred = noise_pred + hodge_pred

            dens_loss = F.mse_loss(noise_pred, -noise)

        return energy, forces, dens_loss

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def use_hodge_forces(self):
        return self.output.use_hodge_forces

    def print_config(self):
        print(f"CliffordSTF GNN:")
        print(f"  stf_mode:       {self.stf_mode}")
        print(f"  hodge_forces:   {self.use_hodge_forces}")
        print(f"  adaptive_route: {self.use_adaptive_routing} ({self.routing_mode})")
        print(f"  clifford_stf_dim:        {clifford_stf_dim(self.stf_mode)}")
        print(f"  parameters:     {self.num_params:,}")
        print(f"  Grade schedule:")
        print(f"    Edge:  (0, 1)")
        print(f"    Atoms: (0,)")
        for i, g in enumerate(self._grade_schedule):
            print(f"    Layer {i}: {g}")