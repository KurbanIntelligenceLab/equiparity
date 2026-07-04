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
from equiparity.models.irreps import degree_irreps


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
