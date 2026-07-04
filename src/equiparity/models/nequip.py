"""NequIP core: build the O(3)/SO(3) matched pair from a typed config.

The correct O(3)/SO(3) toggle (Checkpoint-1 decision, see
docs/reports/checkpoint1_offcycle_parity_toggle.md) is NOT the preset ``parity`` boolean:
``parity=False`` keeps honest natural-parity irreps and stays fully O(3)-equivariant. The
genuine SO(3) arm relabels the edge spherical harmonics (and hidden irreps) as all-even,
which removes parity as an e3nn selection rule. Both arms are built through the raw-irreps
route (``FullNequIPGNNModel``) with identical multiplicities, ``l_max``, and layer count so
the ONLY difference is the parity labeling:

- ``O3``  -> natural-parity SH ``0e+1o+2e``, hidden ``Nx0e+Nx1o+Nx2e``.
- ``SO3`` -> all-even SH ``0e+1e+2e``,      hidden ``Nx0e+Nx1e+Nx2e`` (same values, mislabeled).

Use :func:`build_nequip_matched` for the study. :func:`build_nequip` (preset ``parity`` flag)
is retained only for the shallow-net capacity demonstration and must NOT be used as the SO(3)
arm. The preset toggle is also a no-op below 3 layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from equiparity.domain.parity import ParityMode
from equiparity.models.irreps import degree_irreps


@dataclass(frozen=True, slots=True)
class NequIPConfig:
    """Typed NequIP hyperparameters shared across parity modes.

    A single config instance is built into both parity modes so that the only difference
    between an O(3) and an SO(3) run is the parity toggle.
    """

    r_max: float
    type_names: tuple[str, ...]
    num_layers: int = 4
    l_max: int = 2
    num_features: int = 32
    type_embed_num_features: int | None = None
    radial_mlp_depth: int = 1
    radial_mlp_width: int = 64
    num_bessels: int = 8
    polynomial_cutoff_p: int = 6
    avg_num_neighbors: float = 50.0
    seed: int = 42
    model_dtype: str = "float32"
    do_derivatives: bool = False

    def __post_init__(self) -> None:
        if self.num_layers < 3:
            # Not an error: shallow nets are valid models, but the parity toggle is a
            # no-op below 3 layers, which silently breaks the verification gate.
            import warnings

            warnings.warn(
                f"num_layers={self.num_layers} < 3: the O(3)/SO(3) parity toggle does not "
                "affect features or parameter counts in shallow NequIP networks.",
                stacklevel=2,
            )


# e3nn stores Wigner constants via torch.load; torch 2.6+ needs `slice` allow-listed,
# and nequip requires its global state initialized before building a model.
def _ensure_global_state() -> None:
    import torch.serialization

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    from nequip.utils.global_state import set_global_state

    set_global_state(allow_tf32=False)


def build_nequip(config: NequIPConfig, mode: ParityMode):  # noqa: ANN201 (nequip GraphModel is untyped)
    """Build a NequIP model in the requested parity mode.

    Args:
        config: Shared NequIP hyperparameters.
        mode: O(3) or SO(3); selects the ``parity`` boolean.

    Returns:
        A nequip ``GraphModel`` ready for a forward pass.
    """
    _ensure_global_state()
    from nequip.model import NequIPGNNModel

    type_embed = (
        config.type_embed_num_features
        if config.type_embed_num_features is not None
        else config.num_features
    )
    return NequIPGNNModel(
        r_max=config.r_max,
        type_names=list(config.type_names),
        num_layers=config.num_layers,
        l_max=config.l_max,
        parity=mode.has_parity,
        num_features=config.num_features,
        type_embed_num_features=type_embed,
        radial_mlp_depth=config.radial_mlp_depth,
        radial_mlp_width=config.radial_mlp_width,
        num_bessels=config.num_bessels,
        polynomial_cutoff_p=config.polynomial_cutoff_p,
        avg_num_neighbors=config.avg_num_neighbors,
        seed=config.seed,
        model_dtype=config.model_dtype,
        do_derivatives=config.do_derivatives,
    )


def build_nequip_matched(config: NequIPConfig, mode: ParityMode):  # noqa: ANN201
    """Build one arm of the O(3)/SO(3) matched pair via the raw-irreps route.

    Both arms share every hyperparameter; the arms differ ONLY in the parity labeling of the
    edge spherical harmonics and hidden irreps (O(3) natural parity vs SO(3) all-even). This
    is the study's toggle; see the module docstring.

    Args:
        config: Shared NequIP hyperparameters.
        mode: O(3) or SO(3).

    Returns:
        A nequip ``GraphModel``.
    """
    _ensure_global_state()
    from nequip.model import FullNequIPGNNModel

    type_embed = (
        config.type_embed_num_features
        if config.type_embed_num_features is not None
        else config.num_features
    )
    edge_sh = degree_irreps(config.l_max, mult=1, mode=mode)
    hidden = degree_irreps(config.l_max, mult=config.num_features, mode=mode)
    scalar_out = f"{config.num_features}x0e"
    hidden_per_layer = [hidden] * (config.num_layers - 1) + [scalar_out]
    return FullNequIPGNNModel(
        r_max=config.r_max,
        type_names=list(config.type_names),
        radial_mlp_depth=[config.radial_mlp_depth] * config.num_layers,
        radial_mlp_width=[config.radial_mlp_width] * config.num_layers,
        feature_irreps_hidden=hidden_per_layer,
        irreps_edge_sh=edge_sh,
        type_embed_num_features=type_embed,
        num_bessels=config.num_bessels,
        polynomial_cutoff_p=config.polynomial_cutoff_p,
        avg_num_neighbors=config.avg_num_neighbors,
        seed=config.seed,
        model_dtype=config.model_dtype,
        do_derivatives=config.do_derivatives,
    )


def nequip_featurizer(  # noqa: ANN201 (returns a Featurizer closure over untyped nequip objects)
    model,  # noqa: ANN001
    symbols: str,
    type_names: tuple[str, ...],
    *,
    r_max: float,
    probe_layer: str = "layer1_convnet",
    dtype: str = "float64",
):
    """Return a ``Featurizer`` closure for the equivariance gate.

    The closure maps positions to the model's node features at ``probe_layer`` and their
    irreps, by running a forward pass with a hook. Atom species come from ``symbols`` (an ASE
    formula) mapped to model type indices via ``type_names``.
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
            if isinstance(output, dict) and AtomicDataDict.NODE_FEATURES_KEY in output:
                store["feat"] = output[AtomicDataDict.NODE_FEATURES_KEY].detach().clone()
                store["irreps"] = module.irreps_out[AtomicDataDict.NODE_FEATURES_KEY]

        handle = None
        for name, module in model.named_modules():
            if name.endswith(probe_layer):
                handle = module.register_forward_hook(hook)
                break
        with torch.no_grad():
            model(data)
        if handle is not None:
            handle.remove()
        return store["feat"], store["irreps"]

    return featurize


def realized_hidden_irreps(model) -> dict[str, str]:  # noqa: ANN001
    """Return the realized ``node_features`` irreps of each convolution layer.

    These are the irreps the network can actually populate (after unreachable parity
    channels are pruned), keyed by module name. This is what the parity verification gate
    inspects to confirm a model is in the intended mode.
    """
    layers: dict[str, str] = {}
    for name, module in model.named_modules():
        irreps_out = getattr(module, "irreps_out", None)
        if name.endswith("_convnet") and isinstance(irreps_out, dict):
            features = irreps_out.get("node_features")
            if features is not None:
                layers[name] = str(features)
    return layers
