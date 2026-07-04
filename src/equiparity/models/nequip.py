"""NequIP core: build the O(3)/SO(3)-toggleable NequIP model from a typed config.

Wraps nequip 0.18's preset ``NequIPGNNModel`` builder. The parity toggle maps
:class:`~equiparity.domain.parity.ParityMode` to the ``parity`` boolean:

- ``O3`` -> ``parity=True``: hidden features declare both mirror parities per degree.
- ``SO3`` -> ``parity=False``: hidden features carry only natural spherical-harmonic
  parity ``(-1)**l`` (``0e, 1o, 2e, ...``).

Verified gotcha (see docs/parity_work_plan.md Task 0.3): the two modes realize *identical*
features and parameter counts in shallow networks. The extra O(3) channels only become
reachable from the second convolution layer onward, so any parity check must use
``num_layers >= 3`` or it will silently show no difference.
"""

from __future__ import annotations

from dataclasses import dataclass

from equiparity.domain.parity import ParityMode


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


def count_parameters(model) -> int:  # noqa: ANN001 (nequip GraphModel is untyped)
    """Return the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


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
