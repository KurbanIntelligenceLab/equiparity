"""CliffordSTF core: a fixed O(3) representative (geometric algebra) with a piezo tensor head.

CliffordSTF (vendored under :mod:`equiparity.models.clifford_stf`) is O(3)-correct via graded
geometric algebra, NOT e3nn irreps — so it demonstrates the parity finding transcends the e3nn
framework. Its per-node ``(N, C, 20)`` feature carries Cl(3,0) grades + STF tracks; the parity-odd
content usable here is grade-1 (Cartesian vector = ``1o``) and grade-2 (bivector; its Hodge dual
is an axial vector = ``1e``). The piezo target ``2x1o + 1x2o + 1x3o`` is assembled from these two:

- ``1o`` <- grade-1 (Cartesian basis IS e3nn's ``1o`` up to scale — verified).
- ``2o`` <- TensorProduct(grade-1 ``1o``, grade-2 ``1e``)          (odd x even = odd).
- ``3o`` <- TensorProduct(grade-1 ``1o``, (grade-2 ⊗ grade-2 -> ``2e``)).

Each step was verified rotation-equivariant to ~1e-8 before wiring. The odd outputs cancel to ~0 on
centrosymmetric crystals (O(3) correctness); the STF tracks still enrich the grade features through
the message passing. Crystals need PBC-correct edge vectors (:func:`to_clifford_graph`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from equiparity.domain.parity import ParityMode
from equiparity.models.irreps import output_irreps

# grade-2 bivector [e12,e13,e23] -> axial vector (e3nn 1e): a = [e23,-e13,e12]. Verified (Step 3).
_HODGE = torch.tensor([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


@dataclass(frozen=True, slots=True)
class CliffordSTFConfig:
    """Typed CliffordSTF hyperparameters (O(3)-only via geometric algebra)."""

    r_max: float
    n_channels: int = 64
    n_interactions: int = 3
    n_atom_types: int = 100
    seed: int = 42


def to_clifford_graph(structure, r_max: float, dtype=torch.float32):  # noqa: ANN001, ANN201
    """Build (atomic_numbers, pos, edge_index, edge_vec) with PBC-correct edges via ASE."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    periodic = bool(getattr(structure, "pbc", False)) and structure.cell is not None
    if periodic:
        atoms = Atoms(
            numbers=structure.atomic_numbers,
            positions=structure.positions,
            cell=structure.cell,
            pbc=True,
        )
    else:
        atoms = Atoms(numbers=structure.atomic_numbers, positions=structure.positions)
    i, j, disp = neighbor_list("ijD", atoms, cutoff=r_max)
    return {
        "atomic_numbers": torch.tensor(np.asarray(structure.atomic_numbers), dtype=torch.long),
        "pos": torch.tensor(structure.positions, dtype=dtype),
        "edge_index": torch.tensor(np.stack([i, j]), dtype=torch.long),
        "edge_vec": torch.tensor(disp, dtype=dtype),  # PBC displacement i<-j
        "n_atoms": len(structure.atomic_numbers),
    }


class CliffordSTFTensorModel(torch.nn.Module):
    """CliffordSTF with a piezoelectric (parity-odd) tensor head built from Cl(3,0) grades."""

    def __init__(self, config: CliffordSTFConfig, mode: ParityMode, o3_output_irreps: str) -> None:
        super().__init__()
        from e3nn import o3

        from equiparity.models.clifford_stf.interaction_stf import CliffordSTF

        torch.manual_seed(config.seed)
        c = config.n_channels
        self.backbone = CliffordSTF(
            n_atom_types=config.n_atom_types,
            n_channels=c,
            n_interactions=config.n_interactions,
            cutoff=config.r_max,
            stf_mode="stf2+stf3",
            direct_forces=True,
            use_adaptive_routing=False,
            use_multiscale=False,
            use_dens=False,
        )
        self.output_irreps = o3.Irreps(output_irreps(o3_output_irreps, mode))
        self.register_buffer("hodge", _HODGE.clone())
        self.lin_1o = o3.Linear(f"{c}x1o", "2x1o")
        self.tp_2o = o3.FullyConnectedTensorProduct(f"{c}x1o", f"{c}x1e", "1x2o")
        self.tp_2e = o3.FullyConnectedTensorProduct(f"{c}x1e", f"{c}x1e", f"{c}x2e")
        self.tp_3o = o3.FullyConnectedTensorProduct(f"{c}x1o", f"{c}x2e", "1x3o")

    def forward(self, batch):  # noqa: ANN001, ANN201
        store: dict[str, torch.Tensor] = {}
        handle = self.backbone.output.register_forward_pre_hook(
            lambda _m, args: store.__setitem__("h", args[0])
        )
        try:
            self.backbone(
                batch["atomic_numbers"],
                batch["pos"],
                batch["edge_index"],
                batch["batch"],
                edge_vec=batch["edge_vec"],
            )
        finally:
            handle.remove()
        h = store["h"]  # (N, C, 20)
        n = h.shape[0]
        g1 = h[..., 1:4].reshape(n, -1)  # Cx1o
        g2 = torch.einsum("ij,ncj->nci", self.hodge.to(h.dtype), h[..., 4:7]).reshape(n, -1)  # Cx1e
        e2 = self.tp_2e(g2, g2)  # Cx2e (from 1e (x) 1e)
        # concat in the target order 2x1o + 1x2o + 1x3o
        per_atom = torch.cat([self.lin_1o(g1), self.tp_2o(g1, g2), self.tp_3o(g1, e2)], dim=1)
        batch_index = batch["batch"]
        n_graphs = int(batch_index.max().item()) + 1
        out = torch.zeros(n_graphs, per_atom.shape[1], dtype=per_atom.dtype, device=per_atom.device)
        return out.index_add_(0, batch_index, per_atom)
