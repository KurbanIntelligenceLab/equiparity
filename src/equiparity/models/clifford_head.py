"""CliffordSTF core: a fixed O(3) representative (geometric algebra) with equivariant heads.

CliffordSTF (vendored under :mod:`equiparity.models.clifford_stf`) is O(3)-correct via graded
geometric algebra, NOT e3nn irreps — so it demonstrates the parity finding transcends the e3nn
framework. We run the backbone with ``stf_mode="none"`` (pure Cl(3,0), per-node ``(N, C, 8)``):
the STF (spherical-tensor) tracks' ``augmented_product`` couples grades in a parity-blind way and
breaks O(3)-equivariance (perfect-centro leak ~0.09 vs ~1e-15 without STF), and the head does not
use them anyway. We build the target irreps from two verified base blocks (grade-1 = Cartesian
``1o``; grade-2 bivector Hodge-dual = axial ``1e``) plus grade-0 (``0e``), combined by e3nn tensor
products:

- ``0e`` <- grade-0        ``1o`` <- grade-1        ``1e`` <- grade-2 · Hodge
- ``2e`` <- TP(1e, 1e)     ``2o`` <- TP(1o, 1e)     ``3o`` <- TP(1o, 2e)     ``4e`` <- TP(2e, 2e)

Each construction was verified rotation-equivariant to ~1e-8. Odd outputs cancel to ~0 on
centrosymmetric crystals (O(3) correctness). NOTE: requires **float64** — the
geometric-product/STF chains lose the parity cancellation in float32. Crystals need PBC edges
(:func:`to_clifford_graph`).
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
    """CliffordSTF with an equivariant head that builds a target's irreps from Cl(3,0) grades."""

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
            stf_mode="none",
            direct_forces=True,
            use_adaptive_routing=False,
            use_multiscale=False,
            use_dens=False,
        )
        self.output_irreps = o3.Irreps(output_irreps(o3_output_irreps, mode))
        self.register_buffer("hodge", _HODGE.clone())

        # Decide which base blocks (l, p) to build from the target irreps, plus dependencies:
        # 3o and 4e are built from the 2e block, so require it.
        needed = {(ir.l, ir.p) for _, ir in self.output_irreps}
        if (3, -1) in needed or (4, 1) in needed:
            needed.add((2, 1))
        self._need = needed
        # tensor-product blocks (grades 0e/1o/1e are read directly, no module needed)
        if (2, 1) in needed:
            self.tp_2e = o3.FullyConnectedTensorProduct(f"{c}x1e", f"{c}x1e", f"{c}x2e")
        if (2, -1) in needed:
            self.tp_2o = o3.FullyConnectedTensorProduct(f"{c}x1o", f"{c}x1e", f"{c}x2o")
        if (3, -1) in needed:
            self.tp_3o = o3.FullyConnectedTensorProduct(f"{c}x1o", f"{c}x2e", f"{c}x3o")
        if (4, 1) in needed:
            self.tp_4e = o3.FullyConnectedTensorProduct(f"{c}x2e", f"{c}x2e", f"{c}x4e")
        # combined irreps of the built blocks, in a fixed order, then a linear map to the target.
        order = [(0, 1), (1, -1), (1, 1), (2, 1), (2, -1), (3, -1), (4, 1)]
        label = {
            (0, 1): "0e",
            (1, -1): "1o",
            (1, 1): "1e",
            (2, 1): "2e",
            (2, -1): "2o",
            (3, -1): "3o",
            (4, 1): "4e",
        }
        self._blocks = [lp for lp in order if lp in needed]
        combined = o3.Irreps("+".join(f"{c}x{label[lp]}" for lp in self._blocks))
        self.readout = o3.Linear(combined, self.output_irreps)

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
        g0 = h[..., 0:1].reshape(n, -1)  # Cx0e
        g1 = h[..., 1:4].reshape(n, -1)  # Cx1o
        g2 = torch.einsum("ij,ncj->nci", self.hodge.to(h.dtype), h[..., 4:7]).reshape(n, -1)  # Cx1e
        e2 = self.tp_2e(g2, g2) if (2, 1) in self._need else None
        built = {
            (0, 1): g0,
            (1, -1): g1,
            (1, 1): g2,
            (2, 1): e2,
            (2, -1): self.tp_2o(g1, g2) if (2, -1) in self._need else None,
            (3, -1): self.tp_3o(g1, e2) if (3, -1) in self._need else None,
            (4, 1): self.tp_4e(e2, e2) if (4, 1) in self._need else None,
        }
        combined = torch.cat([built[lp] for lp in self._blocks], dim=1)
        per_atom = self.readout(combined)
        batch_index = batch["batch"]
        n_graphs = int(batch_index.max().item()) + 1
        out = torch.zeros(n_graphs, per_atom.shape[1], dtype=per_atom.dtype, device=per_atom.device)
        return out.index_add_(0, batch_index, per_atom)
