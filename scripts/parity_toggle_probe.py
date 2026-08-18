"""Diagnostic probe: does a given NequIP construction actually break O(3) parity?

Reproduces the parity-toggle finding (written up in the Supplementary Information,
Supplementary Note "Prevalence audit of released architectures"; the original checkpoint note is
retired; git show a3342ea:docs/reports/checkpoint1_offcycle_parity_toggle.md):
flipping NequIP's ``parity`` boolean does NOT produce a parity-violating SO(3) model, while
relabelling the edge spherical harmonics (and hidden features) as all-even does.

Method: run each construction on a toy structure and its mirror image, extract the intermediate
node features, and measure ``|feat(M x) - D(M) . feat(x)|`` where ``D`` is e3nn's parity-aware
representation of the reflection ``M``. Near-zero => the construction is O(3)-equivariant
(parity-respecting); large => it genuinely violates parity (true SO(3)).

Run: ``uv run --extra nequip python scripts/parity_toggle_probe.py``

This is a provisional diagnostic attached as evidence to the off-cycle report. It is NOT the
package's verification gate; that lands in src/equiparity once the SO(3) mechanism is decided.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.serialization

torch.serialization.add_safe_globals([slice])

from ase import Atoms  # noqa: E402
from nequip.data import AtomicDataDict, compute_neighborlist_, from_ase  # noqa: E402
from nequip.model import FullNequIPGNNModel, NequIPGNNModel  # noqa: E402
from nequip.utils.global_state import set_global_state  # noqa: E402

R_MAX = 4.0
TYPE_NAMES = ["H", "C", "O"]
Z_TO_TYPE = {1: 0, 6: 1, 8: 2}
NUM_LAYERS = 3
PROBE_LAYER = "layer1_convnet"
# A deliberately asymmetric 4-atom structure and its reflection through the x-plane.
POSITIONS = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.3], [0.0, 1.1, 0.0], [0.0, 0.0, 1.2]])
REFLECTION = np.diag([-1.0, 1.0, 1.0])


def _make_input(positions: np.ndarray) -> dict:
    atoms = Atoms("H2CO", positions=positions)
    data = compute_neighborlist_(from_ase(atoms), r_max=R_MAX)
    numbers = data[AtomicDataDict.ATOMIC_NUMBERS_KEY].view(-1).tolist()
    data[AtomicDataDict.ATOM_TYPE_KEY] = torch.tensor(
        [[Z_TO_TYPE[int(z)]] for z in numbers], dtype=torch.long
    )
    for key, value in data.items():
        if torch.is_tensor(value) and value.is_floating_point():
            data[key] = value.double()
    return data


def _grab_features(model, data):
    store: dict = {}

    def hook(module, _inputs, output):
        if isinstance(output, dict) and AtomicDataDict.NODE_FEATURES_KEY in output:
            store["feat"] = output[AtomicDataDict.NODE_FEATURES_KEY].detach().clone()
            store["irreps"] = module.irreps_out[AtomicDataDict.NODE_FEATURES_KEY]

    handle = None
    for name, module in model.named_modules():
        if name.endswith(PROBE_LAYER):
            handle = module.register_forward_hook(hook)
            break
    with torch.no_grad():
        model(data)
    if handle is not None:
        handle.remove()
    return store["feat"], store["irreps"]


def _reflection_error(model) -> tuple[str, float, float]:
    feat, irreps = _grab_features(model, _make_input(POSITIONS))
    feat_reflected, _ = _grab_features(model, _make_input(POSITIONS @ REFLECTION.T))
    d_matrix = irreps.D_from_matrix(torch.tensor(REFLECTION)).double()
    predicted = feat @ d_matrix.T
    err = (feat_reflected - predicted).abs().max().item()
    scale = feat.abs().max().item()
    return str(irreps), err, scale


def _build_preset(parity: bool):
    return NequIPGNNModel(
        r_max=R_MAX,
        type_names=TYPE_NAMES,
        num_layers=NUM_LAYERS,
        l_max=2,
        parity=parity,
        num_features=16,
        type_embed_num_features=16,
        radial_mlp_depth=1,
        radial_mlp_width=32,
        num_bessels=8,
        polynomial_cutoff_p=6,
        avg_num_neighbors=10.0,
        seed=42,
        model_dtype="float64",
        do_derivatives=False,
    ).eval()


def _build_full(edge_sh: str, hidden: str):
    return FullNequIPGNNModel(
        r_max=R_MAX,
        type_names=TYPE_NAMES,
        radial_mlp_depth=[1] * NUM_LAYERS,
        radial_mlp_width=[32] * NUM_LAYERS,
        feature_irreps_hidden=[hidden] * (NUM_LAYERS - 1) + ["16x0e"],
        irreps_edge_sh=edge_sh,
        type_embed_num_features=16,
        num_bessels=8,
        polynomial_cutoff_p=6,
        avg_num_neighbors=10.0,
        seed=42,
        model_dtype="float64",
        do_derivatives=False,
    ).eval()


def main() -> None:
    set_global_state(allow_tf32=False)
    torch.manual_seed(0)

    constructions = [
        ("preset parity=True  (plan O3)", lambda: _build_preset(True)),
        ("preset parity=False (plan SO3 toggle)", lambda: _build_preset(False)),
        (
            "full natural-parity SH (honest)",
            lambda: _build_full("1x0e+1x1o+1x2e", "16x0e+16x1o+16x2e"),
        ),
        (
            "full all-even SH (genuine SO3)",
            lambda: _build_full("1x0e+1x1e+1x2e", "16x0e+16x1e+16x2e"),
        ),
    ]

    print(f"{'construction':<40} {'irreps':<34} {'refl.err':>10}  verdict")
    print("-" * 100)
    for label, builder in constructions:
        irreps, err, scale = _reflection_error(builder())
        equivariant = err < 1e-6 * max(scale, 1.0)
        verdict = "O(3) (parity-respecting)" if equivariant else ">>> parity-VIOLATING (SO3) <<<"
        print(f"{label:<40} {irreps:<34} {err:>10.1e}  {verdict}")


if __name__ == "__main__":
    main()
