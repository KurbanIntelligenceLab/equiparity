"""MACE core: build the O(3)/SO(3) matched pair from a typed config.

Unlike NequIP/Allegro, MACE exposes a correct native SO(3) toggle: ``use_so3=True`` builds
the edge spherical harmonics as all-even (``p=1``), removing parity as an e3nn selection
rule (mace ``modules/models.py``). The matched pair is therefore:

- O(3):  ``use_so3=False`` + natural-parity hidden ``Nx0e+Nx1o+Nx2e``.
- SO(3): ``use_so3=True``  + all-even hidden     ``Nx0e+Nx1e+Nx2e``.

Numerical note: MACE's symmetric-contraction tensors stay float32 even when the model is
cast to float64, so its equivariance error floors around 1e-7. The verification gate uses the
float32 thresholds for MACE (``dtype="float32"``), not the float64 ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from equiparity.domain.parity import ParityMode
from equiparity.models.irreps import degree_irreps, output_irreps
from equiparity.models.pooling import pool_per_structure, validate_pooling

# MACE's reduced-precision internals: probe/classify MACE with these thresholds.
MACE_GATE_DTYPE = "float32"
MACE_PROBE_LAYER = "interactions.0.linear"


@dataclass(frozen=True, slots=True)
class MACEConfig:
    """Typed MACE hyperparameters shared across parity modes."""

    r_max: float
    atomic_numbers: tuple[int, ...]
    num_interactions: int = 2
    l_max: int = 2
    num_features: int = 16
    num_bessel: int = 8
    num_polynomial_cutoff: int = 6
    correlation: int = 3
    avg_num_neighbors: float = 8.0
    seed: int = 42
    model_dtype: str = "float64"
    # "sum" (default) reproduces every committed result bit-identically. "mean" divides the
    # summed per-atom readout by the structure's atom count (see equiparity.models.pooling).
    pooling: str = "sum"

    def __post_init__(self) -> None:
        validate_pooling(self.pooling)


def build_mace_matched(config: MACEConfig, mode: ParityMode):  # noqa: ANN201
    """Build one arm of the MACE O(3)/SO(3) matched pair.

    The arms differ only in ``use_so3`` (all-even SH) and the hidden-irreps parity labeling.
    """
    from e3nn import o3
    from mace import modules

    torch.manual_seed(config.seed)
    hidden = degree_irreps(config.l_max, mult=config.num_features, mode=mode)
    interaction = modules.interaction_classes["RealAgnosticResidualInteractionBlock"]
    model = modules.MACE(
        r_max=config.r_max,
        num_bessel=config.num_bessel,
        num_polynomial_cutoff=config.num_polynomial_cutoff,
        max_ell=config.l_max,
        interaction_cls=interaction,
        interaction_cls_first=interaction,
        num_interactions=config.num_interactions,
        num_elements=len(config.atomic_numbers),
        hidden_irreps=o3.Irreps(hidden),
        MLP_irreps=o3.Irreps(f"{config.num_features}x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.zeros(len(config.atomic_numbers)),
        avg_num_neighbors=config.avg_num_neighbors,
        atomic_numbers=list(config.atomic_numbers),
        correlation=config.correlation,
        radial_type="bessel",
        use_so3=not mode.has_parity,
    )
    return model.double() if config.model_dtype == "float64" else model


def mace_featurizer(  # noqa: ANN201 (returns a Featurizer closure over untyped mace objects)
    model,  # noqa: ANN001
    atom_numbers: tuple[int, ...],
    element_table: tuple[int, ...],
    *,
    r_max: float,
    probe_layer: str = MACE_PROBE_LAYER,
    dtype: str = "float64",
):
    """Return a ``Featurizer`` closure that reads MACE's per-node interaction features.

    Args:
        model: A built MACE model.
        atom_numbers: Per-atom atomic numbers of the probe structure (e.g. ``(1, 1, 6, 8)``).
        element_table: The unique elements the model was built for (e.g. ``(1, 6, 8)``).
        r_max: Neighbour cutoff.
        probe_layer: Module whose output node features are probed.
        dtype: Working dtype.

    Uses MACE's vendored ``mace.tools.torch_geometric`` for batching and disables force
    autograd (the probe only needs equivariant features).
    """
    from e3nn import o3
    from mace import data, tools
    from mace.tools import torch_geometric

    z_table = tools.AtomicNumberTable(list(element_table))
    torch_dtype = torch.float64 if dtype == "float64" else torch.float32

    def featurize(positions: np.ndarray):  # noqa: ANN202
        config = data.Configuration(
            atomic_numbers=np.array(atom_numbers),
            positions=positions,
            properties={},
            property_weights={},
        )
        atomic_data = data.AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
        loader = torch_geometric.dataloader.DataLoader(dataset=[atomic_data], batch_size=1)  # type: ignore[arg-type]
        batch = next(iter(loader)).to_dict()
        for key, value in batch.items():
            if torch.is_tensor(value) and value.is_floating_point():
                batch[key] = value.to(torch_dtype)

        store: dict[str, object] = {}

        def hook(module, _inputs, output):  # noqa: ANN001, ANN202
            tensor = output if torch.is_tensor(output) else output[0]
            store["feat"] = tensor.detach().clone()
            store["irreps"] = o3.Irreps(str(module.irreps_out))

        handle = None
        for name, module in model.named_modules():
            if name.endswith(probe_layer):
                handle = module.register_forward_hook(hook)
                break
        with torch.no_grad():
            model(batch, compute_force=False, compute_virials=False, compute_stress=False)
        if handle is not None:
            handle.remove()
        return store["feat"], store["irreps"]

    return featurize


class MACETensorModel(torch.nn.Module):
    """A MACE model with a direct equivariant tensor head of arbitrary output irreps.

    MACE exposes per-node equivariant features after its interaction block
    (``interactions.0.linear``, the same probe the parity gate uses). The target is read out
    per-atom by an ``o3.Linear`` and
    summed over the structure's atoms. The O(3) arm uses the target's true irreps; the SO(3) arm
    (``use_so3=True``, all-even hidden) relabels them all even. For a centrosymmetric structure the
    O(3) odd-parity output cancels to exact zero while the SO(3) output does not — the headline.
    """

    def __init__(self, config: MACEConfig, mode: ParityMode, o3_output_irreps: str) -> None:
        super().__init__()
        from e3nn import o3

        self.backbone = build_mace_matched(config, mode)
        self.pooling = config.pooling
        self.output_irreps = o3.Irreps(output_irreps(o3_output_irreps, mode))
        # The parity gate probes interactions.0 (any layer proves equivariance), but a tensor HEAD
        # needs learned angular features: probe the DEEPEST interaction whose linear still carries
        # l>0 (mirrors NequIP's penultimate / Allegro's deepest-with-l>0). Reading interactions.0
        # was ~6x undertrained for the dipole (shallow features); deeper fixes accuracy and cannot
        # affect the O(3) zero (parity cancellation is structural at every layer).
        self._probe_name = self._deepest_tensor_probe()
        self._probe_module = self._find_probe(self._probe_name)
        self.readout = o3.Linear(self._probe_module.irreps_out, self.output_irreps)
        readout_dtype = torch.float32 if config.model_dtype == "float32" else torch.float64
        self.readout = self.readout.to(readout_dtype)

    def _deepest_tensor_probe(self) -> str:
        """Return the deepest ``interactions.{i}.linear`` whose output irreps carry l>0."""
        from e3nn import o3

        best_idx, best_name = -1, None
        for name, module in self.backbone.named_modules():
            if not (".interactions." in f".{name}" and name.endswith(".linear")):
                continue
            irreps = getattr(module, "irreps_out", None)
            if irreps is None:
                continue
            if not any(ir.ir.l > 0 for ir in o3.Irreps(irreps)):
                continue
            idx = int(name.split("interactions.")[1].split(".")[0])
            if idx > best_idx:
                best_idx, best_name = idx, name.split("backbone.")[-1]
        return best_name if best_name is not None else MACE_PROBE_LAYER

    def _find_probe(self, name: str):  # noqa: ANN202
        for module_name, module in self.backbone.named_modules():
            if module_name.endswith(name):  # match the featurizer's lookup (path may be prefixed)
                return module
        raise RuntimeError(f"probe layer {name!r} not found in backbone")

    def forward(self, batch):  # noqa: ANN001, ANN201
        store: dict[str, object] = {}

        def hook(module, _inputs, output):  # noqa: ANN001, ANN202
            store["feat"] = output if torch.is_tensor(output) else output[0]

        handle = self._probe_module.register_forward_hook(hook)
        try:
            self.backbone(batch, compute_force=False, compute_virials=False, compute_stress=False)
        finally:
            handle.remove()
        per_atom = self.readout(store["feat"])  # (n_atoms, output_dim)
        batch_index = batch["batch"]  # (n_atoms,) node -> structure (torch_geometric)
        n_graphs = int(batch_index.max().item()) + 1
        return pool_per_structure(per_atom, batch_index, n_graphs, self.pooling)


class MACEDipoleModel(MACETensorModel):
    """A MACE model with a direct L=1 dipole head (``1o`` for O(3), ``1e`` for SO(3))."""

    def __init__(self, config: MACEConfig, mode: ParityMode) -> None:
        super().__init__(config, mode, "1x1o")
