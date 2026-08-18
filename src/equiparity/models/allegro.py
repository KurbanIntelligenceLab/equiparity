"""Allegro core: build the O(3)/SO(3) matched pair from a typed config.

Allegro (nequip-allegro) has the same parity mechanism as NequIP: its preset ``AllegroModel``
always builds natural-parity edge spherical harmonics, so the ``parity`` boolean does NOT
produce an SO(3) model. The genuine SO(3) arm relabels the edge SH (and the allowed tensor
irreps) as all-even, reached through the raw-irreps route ``FullAllegroModel``. Both arms
share every hyperparameter and differ only in that parity labeling.

Allegro is edge-centric: the probe reads its per-edge tensor features (:func:`allegro_featurizer`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from equiparity.domain.parity import ParityMode
from equiparity.models.irreps import degree_irreps, output_irreps
from equiparity.models.pooling import pool_per_structure, validate_pooling


@dataclass(frozen=True, slots=True)
class AllegroConfig:
    """Typed Allegro hyperparameters shared across parity modes."""

    r_max: float
    type_names: tuple[str, ...]
    num_layers: int = 2
    l_max: int = 2
    num_scalar_features: int = 32
    num_tensor_features: int = 16
    num_bessels: int = 8
    bessel_trainable: bool = False
    polynomial_cutoff_p: int = 6
    avg_num_neighbors: float = 50.0
    seed: int = 42
    model_dtype: str = "float32"
    do_derivatives: bool = False
    # "sum" (default) reproduces every committed result bit-identically. "mean" divides the
    # summed per-edge readout by the structure's EDGE count, not atom count -- Allegro is
    # edge-centric (no per-atom message passing), so an intensive readout here means the
    # per-edge contributions average out over edges, not atoms (see equiparity.models.pooling).
    pooling: str = "sum"

    def __post_init__(self) -> None:
        validate_pooling(self.pooling)


def _ensure_global_state() -> None:
    import torch.serialization

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    from nequip.utils.global_state import set_global_state

    set_global_state(allow_tf32=False)


def build_allegro_matched(config: AllegroConfig, mode: ParityMode):  # noqa: ANN201
    """Build one arm of the Allegro O(3)/SO(3) matched pair via the raw-irreps route.

    The arms differ ONLY in the parity labeling of the edge spherical harmonics and the
    allowed tensor-track irreps (O(3) natural parity vs SO(3) all-even).
    """
    _ensure_global_state()
    from allegro.model.allegro_models import FullAllegroModel

    edge_sh = degree_irreps(config.l_max, mult=1, mode=mode)
    radial_chemical_embed = {
        "_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
        "num_bessels": config.num_bessels,
        "bessel_trainable": config.bessel_trainable,
        "polynomial_cutoff_p": config.polynomial_cutoff_p,
    }
    return FullAllegroModel(
        r_max=config.r_max,
        type_names=list(config.type_names),
        irreps_edge_sh=edge_sh,
        tensor_track_allowed_irreps=edge_sh,
        radial_chemical_embed=radial_chemical_embed,
        num_layers=config.num_layers,
        num_scalar_features=config.num_scalar_features,
        num_tensor_features=config.num_tensor_features,
        avg_num_neighbors=config.avg_num_neighbors,
        seed=config.seed,
        model_dtype=config.model_dtype,
        do_derivatives=config.do_derivatives,
    )


def allegro_featurizer(  # noqa: ANN201 (returns a Featurizer closure over untyped allegro objects)
    model,  # noqa: ANN001
    symbols: str,
    type_names: tuple[str, ...],
    *,
    r_max: float,
    probe_layer: str = "model.func.allegro.tps.0",
    dtype: str = "float64",
):
    """Return a ``Featurizer`` closure that reads Allegro's per-edge tensor features.

    The reflection/rotation transforms preserve interatomic distances and atom indices, so
    the neighborlist edge ordering is stable across transforms and per-edge features
    correspond directly.
    """
    import numpy as np
    from ase import Atoms
    from nequip.data import AtomicDataDict, compute_neighborlist_, from_ase

    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    symbol_to_type = {name: idx for idx, name in enumerate(type_names)}

    def featurize(positions: np.ndarray):  # noqa: ANN202
        atoms = Atoms(symbols, positions=positions)
        data = compute_neighborlist_(from_ase(atoms), r_max=r_max)
        data[AtomicDataDict.ATOM_TYPE_KEY] = torch.tensor(
            [[symbol_to_type[s]] for s in atoms.get_chemical_symbols()], dtype=torch.long
        )
        for key, value in data.items():
            if torch.is_tensor(value) and value.is_floating_point():
                data[key] = value.to(torch_dtype)

        store: dict[str, object] = {}

        def hook(module, _inputs, output):  # noqa: ANN001, ANN202
            tensor = output if torch.is_tensor(output) else output[0]
            store["feat"] = tensor.detach().clone()
            store["irreps"] = module.irreps_out

        handle = None
        for name, module in model.named_modules():
            if name == probe_layer:
                handle = module.register_forward_hook(hook)
                break
        with torch.no_grad():
            model(data)
        if handle is not None:
            handle.remove()
        return store["feat"], store["irreps"]

    return featurize


class AllegroTensorModel(torch.nn.Module):
    """An Allegro model with a direct equivariant tensor head of arbitrary output irreps.

    Allegro is edge-centric: it produces per-edge equivariant tensor features (no per-node
    message passing). The target is read out from the deepest tensor-product layer by a per-edge
    ``o3.Linear`` and summed over the structure's edges. Summing same-irrep per-edge features is
    equivariant (a global rotation rotates every edge feature identically), and for a
    centrosymmetric structure the O(3) odd-parity output cancels to exact zero (edges come in
    inversion-related pairs) while the SO(3) output does not — verified: O(3) ||T|| ~1e-14 vs
    SO(3) ~1e+2. The O(3) arm uses the target's true irreps; the SO(3) arm relabels them all even.
    """

    def __init__(self, config: AllegroConfig, mode: ParityMode, o3_output_irreps: str) -> None:
        super().__init__()
        from e3nn import o3

        self.backbone = build_allegro_matched(config, mode)
        self.pooling = config.pooling
        self.output_irreps = o3.Irreps(output_irreps(o3_output_irreps, mode))
        # Deepest per-edge tensor-product layer (model.func.allegro.tps.<i>) that still carries
        # l>0 features. Allegro's FINAL tps collapses to scalars (1x0e) for the energy readout,
        # so reading it would give exact zero for any l>0 target — pick the deepest non-scalar one.
        tps = [
            (int(name.split(".")[-1]), name, module)
            for name, module in self.backbone.named_modules()
            if name.startswith("model.func.allegro.tps.") and name.split(".")[-1].isdigit()
        ]
        tensor_tps = [(i, name, m) for i, name, m in tps if o3.Irreps(m.irreps_out).lmax > 0]
        if not tensor_tps:
            raise RuntimeError(
                "no allegro tensor-product layer with l>0 features for the tensor head"
            )
        _, self._probe_name, self._probe_module = max(tensor_tps, key=lambda t: t[0])
        self.readout = o3.Linear(self._probe_module.irreps_out, self.output_irreps)
        readout_dtype = torch.float32 if config.model_dtype == "float32" else torch.float64
        self.readout = self.readout.to(readout_dtype)

    def _find_probe(self, name: str):  # noqa: ANN202
        for module_name, module in self.backbone.named_modules():
            if module_name == name:
                return module
        raise RuntimeError(f"probe layer {name!r} not found in backbone")

    def forward(self, batch):  # noqa: ANN001, ANN201
        from nequip.data import AtomicDataDict

        store: dict[str, object] = {}

        def hook(module, _inputs, output):  # noqa: ANN001, ANN202
            store["feat"] = output if torch.is_tensor(output) else output[0]

        handle = self._probe_module.register_forward_hook(hook)
        try:
            self.backbone(batch)
        finally:
            handle.remove()
        # Allegro tensor features are (n_edges, num_tensor_features, per_channel_irrep_dim). Read
        # each channel out to the target irreps and sum the channels (equivariant: same irrep).
        per_edge = self.readout(store["feat"])  # (n_edges, [n_channels,] output_dim)
        if per_edge.dim() == 3:
            per_edge = per_edge.sum(dim=1)  # reduce the tensor-feature channels
        edge_index = batch[AtomicDataDict.EDGE_INDEX_KEY]  # (2, n_edges)
        n_atoms = int(edge_index.max().item()) + 1
        batch_index = batch.get(
            AtomicDataDict.BATCH_KEY,
            torch.zeros(n_atoms, dtype=torch.long, device=per_edge.device),
        )
        edge_struct = batch_index[edge_index[0]]  # (n_edges,) edge -> structure
        n_graphs = int(batch_index.max().item()) + 1
        # "mean" divides by the per-structure EDGE count (edge_struct), not atom count: Allegro's
        # readout is per-edge, so an intensive average must be taken over edges (module docstring).
        return pool_per_structure(per_edge, edge_struct, n_graphs, self.pooling)


class AllegroDipoleModel(AllegroTensorModel):
    """An Allegro model with a direct L=1 dipole head (``1o`` for O(3), ``1e`` for SO(3))."""

    def __init__(self, config: AllegroConfig, mode: ParityMode) -> None:
        super().__init__(config, mode, "1x1o")
