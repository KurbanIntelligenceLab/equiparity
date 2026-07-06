"""EquiformerV2 core: a fixed SO(3) representative with a tensor-output head.

EquiformerV2 (vendored under :mod:`equiparity.models.equiformer_v2`) is an SO(3)-only spherical
harmonic transformer — its per-node ``SO3_Embedding`` features are all-even (no parity). We read
them out per structure to a target tensor. Because the features carry no parity, an odd target
(e.g. piezoelectric) is represented all-even (:func:`output_irreps` in SO(3) mode) and therefore
does NOT cancel on centrosymmetric crystals — the SO(3) failure signal.

Two integration facts (verified): the backbone runs in float32 (its Wigner buffers are float32),
and the ``SO3_Embedding`` (l,m) ordering is e3nn-compatible, so the layout map below is
rotation-equivariant (checked: ||v(Rx) - R v(x)|| ~ 1e-4). Crystals need PBC-correct edge vectors
(``edge_distance_vec`` with periodic shifts) or centrosymmetry breaks — see :func:`to_pyg_data`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from equiparity.domain.parity import ParityMode
from equiparity.models.irreps import output_irreps


@dataclass(frozen=True, slots=True)
class EquiformerV2Config:
    """Typed EquiformerV2 hyperparameters (SO(3)-only)."""

    r_max: float
    lmax: int = 4
    num_layers: int = 4
    sphere_channels: int = 64
    attn_hidden_channels: int = 32
    ffn_hidden_channels: int = 64
    num_heads: int = 4
    edge_channels: int = 32
    max_num_elements: int = 90
    seed: int = 42


def to_pyg_data(structure, r_max: float, dtype=torch.float32):  # noqa: ANN001, ANN201
    """Build a torch_geometric ``Data`` with PBC-correct edge vectors from a structure.

    Uses ASE's neighborlist (``'ijD'``) which returns periodic-image displacement vectors, so
    centrosymmetry is preserved. ``edge_distance_vec`` matches the model's ``pos[row]-pos[col]``
    convention (D is the vector from j to i; the model uses row=i, col=j).
    """
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    from torch_geometric.data import Data

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
    edge_index = torch.tensor(np.stack([i, j]), dtype=torch.long)
    edge_vec = torch.tensor(disp, dtype=dtype)  # (n_edges, 3), vector i<-j (= pos[i]-pos[j]+shift)
    n = len(structure.atomic_numbers)
    return Data(
        atomic_numbers=torch.tensor(np.asarray(structure.atomic_numbers), dtype=torch.long),
        pos=torch.tensor(structure.positions, dtype=dtype),
        natoms=torch.tensor([n]),
        batch=torch.zeros(n, dtype=torch.long),
        edge_index=edge_index,
        edge_distance_vec=edge_vec,
    )


class EquiformerV2TensorModel(torch.nn.Module):
    """EquiformerV2 with a direct equivariant tensor head (SO(3), all-even features)."""

    def __init__(self, config: EquiformerV2Config, mode: ParityMode, o3_output_irreps: str) -> None:
        super().__init__()
        from e3nn import o3

        from equiparity.models.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

        torch.manual_seed(config.seed)
        self.lmax = config.lmax
        self.channels = config.sphere_channels
        self.backbone = EquiformerV2_OC20(
            None,
            None,
            1,
            otf_graph=False,
            regress_forces=False,
            use_pbc=True,
            max_radius=config.r_max,
            num_layers=config.num_layers,
            sphere_channels=config.sphere_channels,
            attn_hidden_channels=config.attn_hidden_channels,
            ffn_hidden_channels=config.ffn_hidden_channels,
            num_heads=config.num_heads,
            lmax_list=[config.lmax],
            mmax_list=[config.lmax],
            max_num_elements=config.max_num_elements,
            edge_channels=config.edge_channels,
        )
        # EquiformerV2 is SO(3): features are all-even. The output is relabeled per `mode` (odd
        # targets become all-even -> violate parity). Source irreps are all-even up to lmax.
        self.output_irreps = o3.Irreps(output_irreps(o3_output_irreps, mode))
        src = o3.Irreps("+".join(f"{self.channels}x{deg}e" for deg in range(self.lmax + 1)))
        self.readout = o3.Linear(src, self.output_irreps)

    def _so3_to_e3nn(self, emb: torch.Tensor) -> torch.Tensor:
        """(n_atoms,(lmax+1)^2,C) coeff(l,m)-major -> e3nn 'Cx0e+..+CxLe' (mul-major). Verified."""
        out = []
        offset = 0
        for deg in range(self.lmax + 1):
            n = 2 * deg + 1
            block = emb[:, offset : offset + n, :].transpose(1, 2).reshape(emb.shape[0], -1)
            out.append(block)
            offset += n
        return torch.cat(out, dim=1)

    def forward(self, data):  # noqa: ANN001, ANN201
        store: dict[str, torch.Tensor] = {}
        handle = self.backbone.norm.register_forward_hook(
            lambda _m, _i, o: store.__setitem__("f", o)
        )
        try:
            self.backbone(data)
        finally:
            handle.remove()
        per_atom = self.readout(self._so3_to_e3nn(store["f"]))  # (n_atoms, output_dim)
        batch_index = data.batch
        n_graphs = int(batch_index.max().item()) + 1
        out = torch.zeros(n_graphs, per_atom.shape[1], dtype=per_atom.dtype, device=per_atom.device)
        return out.index_add_(0, batch_index, per_atom)


class EquiformerV2DipoleModel(EquiformerV2TensorModel):
    """EquiformerV2 with an L=1 dipole head (relabeled ``1e`` in SO(3) mode)."""

    def __init__(self, config: EquiformerV2Config, mode: ParityMode) -> None:
        super().__init__(config, mode, "1x1o")
