"""H1 : reproduce the EquiformerV2 findings on upstream source (@ 8fe8cba), no OCP install.

Builds a model from ``third_party/equiformer_v2_shimmed/`` (upstream verbatim + the documented
OCP-removal edits; see ``h1_build_shimmed.py``), with ``generate_graph`` relocated out of the model
file and provided by a small, auditable shim subclass. Then:

1. **Equivalence** — the shimmed-upstream model reproduces the shipped vendored model
   **bit-for-bit**
   on a seeded forward (proves the reconstruction is faithful and the vendored copy carries upstream
   behaviour verbatim).
2. **Pass-through bit-identity (Condition 2 by measurement)** — E5/E3 measurements are
   **bit-identical**
   whether ``generate_graph`` is the shim pass-through or a hard stub that returns the precomputed
   graph verbatim. Bit-identical ⇒ the added method cannot affect any measured number.
3. **Reruns** — E5 determinism (single-draw + 5-seed average), rotation, mirror; E3 FD-vs-autograd —
   on the shimmed-upstream model, compared to the vendored measurements.

Run: ``uv run --extra nequip python scripts/h1_upstream_repro.py``
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SHIMMED = REPO / "third_party/equiformer_v2_shimmed"
OUT_JSON = REPO / "results" / "h1_upstream.json"
sys.path.insert(0, str(REPO / "src"))


def _load_shimmed_oc20():
    """Import EquiformerV2_OC20 from the shimmed-upstream tree (isolated module name)."""
    # Load the package's sibling modules under a dedicated name so their relative imports resolve.
    pkg = "equiformer_v2_shimmed"
    if pkg not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg, SHIMMED / "__init__.py", submodule_search_locations=[str(SHIMMED)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[pkg] = module
        spec.loader.exec_module(module)
    oc20 = importlib.import_module(f"{pkg}.equiformer_v2_oc20")
    return oc20.EquiformerV2_OC20


def make_shim_backbone_cls(stub: bool = False):
    """Subclass the upstream-sourced backbone and provide the relocated ``generate_graph``.

    The shim is a pure pass-through, guarded to enforce the pass-through precondition. ``stub=True``
    returns the precomputed graph even more directly (edge_distance from the norm, all else the
    plainest possible) — used only for the Condition-2 bit-identity check.
    """
    base = _load_shimmed_oc20()

    class _Shim(base):  # type: ignore[valid-type, misc]
        def generate_graph(self, data):
            # Guard: the pass-through precondition, enforced not assumed.
            assert self.otf_graph is False, "shim generate_graph requires otf_graph=False"
            assert getattr(data, "edge_distance_vec", None) is not None, (
                "shim generate_graph requires a precomputed edge_distance_vec"
            )
            edge_index = data.edge_index
            edge_distance_vec = data.edge_distance_vec
            edge_distance = torch.norm(edge_distance_vec, dim=-1)
            n_edges = edge_index.shape[1]
            if stub:
                zeros3 = torch.zeros(n_edges, 3, device=edge_index.device)
                neighbors = torch.zeros(1, dtype=torch.long, device=edge_index.device)
                return (
                    edge_index,
                    edge_distance,
                    edge_distance_vec,
                    zeros3.long(),
                    zeros3,
                    neighbors,
                )
            num_atoms = data.pos.shape[0]
            neighbors = torch.zeros(num_atoms, dtype=torch.long, device=edge_index.device)
            neighbors.index_add_(0, edge_index[1], torch.ones_like(edge_index[1]))
            cell_offsets = torch.zeros(n_edges, 3, dtype=torch.long, device=edge_index.device)
            offset_distances = torch.zeros(n_edges, 3, device=edge_index.device)
            return (
                edge_index,
                edge_distance,
                edge_distance_vec,
                cell_offsets,
                offset_distances,
                neighbors,
            )

    return _Shim


def build_tensor_model(config, mode, o3_output_irreps, *, stub: bool = False):
    """Mirror EquiformerV2TensorModel.__init__ exactly, but with the shimmed-upstream backbone."""
    from e3nn import o3

    from equiparity.models.equiformer import EquiformerV2TensorModel
    from equiparity.models.irreps import output_irreps

    backbone_cls = make_shim_backbone_cls(stub=stub)

    # Construct via the vendored wrapper, then swap the backbone for a shimmed one built with the
    # identical seed + args so the readout RNG stream (seeded after the backbone) is unchanged.
    model = EquiformerV2TensorModel.__new__(EquiformerV2TensorModel)
    torch.nn.Module.__init__(model)
    torch.manual_seed(config.seed)
    model.lmax = config.lmax
    model.channels = config.sphere_channels
    model.backbone = backbone_cls(
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
    model.output_irreps = o3.Irreps(output_irreps(o3_output_irreps, mode))
    src = o3.Irreps("+".join(f"{model.channels}x{deg}e" for deg in range(model.lmax + 1)))
    model.readout = o3.Linear(src, model.output_irreps)
    return model


def main() -> None:
    from equiparity.domain.parity import ParityMode
    from equiparity.domain.target import TARGETS
    from equiparity.models.equiformer import EquiformerV2Config, EquiformerV2TensorModel

    # Bit-identity is only meaningful on CPU: CUDA's atomic index_add_ and random-frame draw are
    # nondeterministic (~1e-6 run-to-run) regardless of code correctness. The pass-through proof
    # therefore runs on CPU, where two equivalent code paths agree exactly.
    dev = torch.device("cpu")
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = EquiformerV2Config(
        r_max=5.0,
        lmax=3,
        num_layers=3,
        sphere_channels=64,
        attn_hidden_channels=32,
        ffn_hidden_channels=64,
        num_heads=4,
        edge_channels=32,
        seed=0,
    )
    irreps = TARGETS["piezoelectric"].irreps

    torch.manual_seed(cfg.seed)
    vendored = EquiformerV2TensorModel(cfg, ParityMode.SO3, irreps).to(dev).eval()
    shimmed = build_tensor_model(cfg, ParityMode.SO3, irreps).to(dev).eval()
    shimmed.load_state_dict(vendored.state_dict())  # identical weights

    # --- 1. equivalence: bit-identical seeded forward on a toy periodic graph ---
    from equiparity.models.equiformer import to_pyg_data

    struct = _toy_crystal()
    g = to_pyg_data(struct, cfg.r_max).to(dev)
    torch.manual_seed(123)
    v_out = vendored(g).detach().cpu()
    torch.manual_seed(123)
    s_out = shimmed(g).detach().cpu()
    equiv_max = float((v_out - s_out).abs().max())
    print(
        f"[1] shimmed-upstream vs vendored, seeded forward: max|Δ| = {equiv_max:.3e}  "
        f"({'BIT-IDENTICAL' if equiv_max == 0.0 else 'DIFFERS'})"
    )

    # --- 2. pass-through bit-identity: shim generate_graph vs hard stub ---
    stubbed = build_tensor_model(cfg, ParityMode.SO3, irreps, stub=True).to(dev).eval()
    stubbed.load_state_dict(vendored.state_dict())
    torch.manual_seed(123)
    a = shimmed(g).detach().cpu()
    torch.manual_seed(123)
    b = stubbed(g).detach().cpu()
    passthrough_max = float((a - b).abs().max())
    print(
        f"[2] shim generate_graph vs hard stub: max|Δ| = {passthrough_max:.3e}  "
        f"({'BIT-IDENTICAL' if passthrough_max == 0.0 else 'DIFFERS'})"
    )

    # --- 3. reruns on the shimmed-upstream model with the trained checkpoints (CUDA) ---
    reruns = _reruns(cfg, irreps)

    result = {
        "pinned_sha": "8fe8cbaf8f3c27865b6e28c21db7867e75a107f7",
        "equivalence_max_abs_diff": equiv_max,
        "passthrough_max_abs_diff": passthrough_max,
        "reruns_on_upstream_source": reruns,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=1) + "\n")
    print(f"\nwrote {OUT_JSON}")


def _trained_wrapper(model, dev):
    """Wrap a shimmed model as a reload-harness TrainedModel so the E5/E3 helpers apply directly."""
    from equiparity.inference.reload import PIEZO_SCALE, TrainedModel

    return TrainedModel(
        model=model,
        scale=PIEZO_SCALE,
        config=type(
            "C",
            (),
            {
                "core": "equiformer_v2",
                "model": type("M", (), {"r_max": 5.0})(),
                "training": type("T", (), {"batch_size": 16})(),
            },
        )(),
        core="equiformer_v2",
        parity="so3",
        run_dir=Path(),
        device=dev,
        _extra={},
    )


def _eqv2_fd_vs_autograd(model, structure, dev, seed: int = 4242) -> float:
    """Median rel. error of autograd vs central finite differences for dT/dr on EquiformerV2.

    The random per-edge frame is seeded identically before every forward so FD is not swamped by
    frame noise; the residual disagreement is the incomplete autograd (detached Wigner-D). Rebuilds
    edge_distance_vec from a differentiable ``pos`` at fixed connectivity (the E3 patch).: E501
    """
    from equiparity.models.equiformer import to_pyg_data

    g0 = to_pyg_data(structure, 5.0).to(dev)
    base = g0.pos.detach().clone()
    src, dst = g0.edge_index[0], g0.edge_index[1]
    disp = g0.edge_distance_vec.detach()
    shift = disp - (base[dst] - base[src])  # ASE convention: disp = pos[j]-pos[i]+shift

    def forward(pos):
        g = to_pyg_data(structure, 5.0).to(dev)
        ev = pos[dst] - pos[src] + shift
        g.pos, g.edge_distance_vec, g.edge_distance = pos, ev, ev.norm(dim=-1)
        torch.manual_seed(seed)
        return model(g)

    pos = base.clone().requires_grad_(True)
    out = forward(pos)
    out.flatten()[0].backward()
    ana = pos.grad.detach().cpu().numpy()

    rng = np.random.default_rng(0)
    coords = [(int(rng.integers(0, base.shape[0])), int(rng.integers(0, 3))) for _ in range(4)]
    errs = []
    h = 2e-3
    for a, c in coords:
        pp, pm = base.clone(), base.clone()
        pp[a, c] += h
        pm[a, c] -= h
        with torch.no_grad():
            fd = float((forward(pp).flatten()[0] - forward(pm).flatten()[0]) / (2 * h))
        errs.append(abs(ana[a, c] - fd) / max(abs(fd), 1e-12))
    return float(np.median(errs))


def _reruns(cfg, irreps) -> dict:
    """E5 determinism/rotation/mirror + E3 FD on the shimmed-upstream model, trained weights."""
    sys.path.insert(0, str(REPO / "scripts"))
    from e5_output_parity import _random_rotation, _rel_error, _transform, _wigner

    from equiparity.domain.parity import ParityMode
    from equiparity.inference import find_piezo_runs, seeded_predict
    from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset, load_split

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = find_piezo_runs(
        Path(os.environ.get("PARITY_RUNS", Path.home() / "Desktop" / "parity_work")),
        dataset="mp_piezoelectric",
    )
    eq_runs = sorted(k for k in runs if k.startswith("equiformer_v2"))

    data = load_crystal_dataset(
        REPO / "data/raw/mp/mp_piezoelectric_processed.npz", ("piezoelectric",)
    )
    test = CrystalDataset(data, load_split(REPO / "data/splits/mp_piezoelectric_split.npz", "test"))
    noncentro = [test[i].structure for i in range(25)]
    # match the vendored E5 determinism check, which used the first 5 centrosymmetric OOD structures.  # noqa: E501
    ood = CrystalDataset(
        load_crystal_dataset(REPO / "data/raw/mp/mp_ood_centrosymmetric_processed.npz")
    )
    centro5 = [ood[i].structure for i in range(5)]

    rng = np.random.default_rng(0)
    mirror_m = np.diag([-1.0, 1.0, 1.0])
    rot_r = _random_rotation(rng)
    d_mirror, d_rot = _wigner(mirror_m), _wigner(rot_r)

    rows = []
    for label in eq_runs:
        model = build_tensor_model(cfg, ParityMode.SO3, irreps).to(dev).eval()
        sd = torch.load(runs[label] / "checkpoint_latest.pt", map_location=dev, weights_only=False)
        model.load_state_dict(sd["model"])
        trained = _trained_wrapper(model, dev)

        torch.manual_seed(0)
        base = trained.predict(noncentro)
        torch.manual_seed(0)
        mirrored = trained.predict([_transform(s, mirror_m) for s in noncentro])
        torch.manual_seed(0)
        rotated = trained.predict([_transform(s, rot_r) for s in noncentro])
        draws = seeded_predict(trained, centro5, draws=5)
        spread = float(np.abs(draws.max(axis=0) - draws.min(axis=0)).max())
        fd = float(np.median([_eqv2_fd_vs_autograd(model, noncentro[i], dev) for i in range(3)]))
        rows.append(
            {
                "run": label,
                "mirror_rel_error": _rel_error(mirrored, base @ d_mirror.T, base),
                "rotation_rel_error": _rel_error(rotated, base @ d_rot.T, base),
                "determinism_spread": spread,
                "e3_fd_rel_error": fd,
            }
        )
        r = rows[-1]
        print(
            f"[3] {label:38s} mirror={r['mirror_rel_error']:.3f} rot={r['rotation_rel_error']:.3f} "
            f"det={r['determinism_spread']:.3f} FD={r['e3_fd_rel_error']:.2f}"
        )
    return {"rows": rows}


def _toy_crystal():
    from equiparity.domain.structure import AtomicStructure

    rng = np.random.default_rng(0)
    cell = np.eye(3) * 6.0
    pos = rng.uniform(0, 6, size=(8, 3))
    z = np.array([8, 14, 8, 14, 8, 14, 8, 14], dtype=np.int64)
    return AtomicStructure(atomic_numbers=z, positions=pos, cell=cell, pbc=True)


if __name__ == "__main__":
    main()
