"""Tier 2 -- random-init rank-3 structural test + reflection probe for eSEN / UMA / EquiformerV3.

Standalone by design: this script runs INSIDE the model's isolated venv (fairchem-core cannot
co-install with the repo's torch-2.11 env) and therefore imports nothing from `equiparity`.
Data comes straight from the committed npz; the (l,m)-major -> e3nn mul-major remap and the
attach-a-rank-3-head construction replicate `EquiformerV2TensorModel` exactly.

Theorem 1 holds at any parameters, so the structural test runs at RANDOM INIT: attach the
standard rank-3 head (`o3.Linear` onto all-even features, output relabeled even -- the SO(3)
arm construction) and evaluate ||T||_F on the 2,000 idealized centrosymmetric crystals. The
scale-free internal control is the point-group family split: an exactly rotation-equivariant
model is forced to machine zero on the 166 m-3m crystals (rotation ceiling) while nothing
forces the others -- so the m-3m floor vs non-cubic magnitude ratio is the structural result,
independent of the random head's arbitrary scale.

The E5-style output-level probe (mirror / rotation / determinism laws, physical odd-parity
Wigner D) runs on 25 non-centrosymmetric piezoelectric structures.

    third_party/venvs/fairchem1/bin/python scripts/t2_backbone_probe.py --model esen
    third_party/venvs/fairchem/bin/python  scripts/t2_backbone_probe.py --model uma
    third_party/venvs/fairchem1/bin/python scripts/t2_backbone_probe.py --model eqv3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from e3nn import o3

REPO = Path(__file__).resolve().parent.parent
OOD_NPZ = REPO / "data/raw/mp/mp_ood_centrosymmetric_processed.npz"
PIEZO_NPZ = REPO / "data/raw/mp/mp_piezoelectric_processed.npz"
SPACEGROUPS = REPO / "results/ood_spacegroups.json"
OUT_DIR = REPO / "results" / "t2"
OUT_JSON = REPO / "results" / "t2_random_init.json"

LMAX = 2
CHANNELS = 64
PHYSICAL_IRREPS = "2x1o+1x2o+1x3o"
N_PROBE = 25
THRESHOLD = 0.01


class Shim(dict):
    """fairchem data objects are accessed both as dict and by attribute."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:  # hasattr() must see AttributeError, not KeyError
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    def get(self, key, default=None):  # UMA calls data.get(k, default=None)
        return dict.get(self, key, default)


def build_esen():
    from fairchem.core.models.esen.esen import eSEN_Backbone

    return eSEN_Backbone(
        lmax=LMAX,
        mmax=LMAX,
        sphere_channels=CHANNELS,
        num_layers=2,
        otf_graph=True,
        use_pbc=True,
        cutoff=5.0,
        regress_forces=False,
        direct_forces=False,
    ).eval()


def build_uma():
    from fairchem.core.models.uma.escn_md import eSCNMDBackbone

    return eSCNMDBackbone(
        lmax=LMAX,
        mmax=LMAX,
        sphere_channels=CHANNELS,
        num_layers=2,
        otf_graph=True,
        cutoff=5.0,
        direct_forces=False,
        regress_stress=False,
        dataset_mapping={"omat": "omat"},
    ).eval()


def build_eqv3():
    import sys

    sys.path.insert(0, str(REPO / "third_party/equiformer_v3/experimental/models"))
    from equiformer_v3.equiformer_v3 import EquiformerV3_OC  # type: ignore

    class Adapter(torch.nn.Module):
        """EquiformerV3's forward returns only energy/forces; replicate `_forward_direct`
        up to the post-blocks SO3 embedding and expose it as `node_embedding`."""

        def __init__(self, m) -> None:
            super().__init__()
            self.m = m

        def forward(self, data):
            m = self.m
            m.batch_size = len(data.natoms)
            m.dtype = data.pos.dtype
            m.device = data.pos.device
            edge_index, edge_distance, edge_distance_vec, _, _, _ = m.generate_graph(
                data,
                enforce_max_neighbors_strictly=m.enforce_max_neighbors_strictly,
                use_pbc_single=m.use_pbc_single,
            )
            atomic_numbers = data.atomic_numbers.long()
            src_z = atomic_numbers[edge_index[0]]
            tgt_z = atomic_numbers[edge_index[1]]
            edge_distance, w = m._forward_edge(edge_distance, edge_distance_vec)
            x = m._forward_embedding(atomic_numbers, edge_distance, edge_index, w)
            _, x = m._forward_blocks(x, src_z, tgt_z, edge_distance, edge_index, w, data.batch)
            emb = x.embedding if hasattr(x, "embedding") else x
            return {"node_embedding": emb}

    # max_neighbors=300 (not the OC default 20): neighbor truncation is not rotation-
    # invariant, and the probe must measure the network, not the graph builder.
    m = EquiformerV3_OC(
        num_layers=2,
        num_channels=CHANNELS,
        lmax=LMAX,
        mmax=LMAX,
        max_radius=5.0,
        otf_graph=True,
        regress_forces=False,
        max_neighbors=300,
    )
    return Adapter(m).eval()


BUILDERS = {"esen": build_esen, "uma": build_uma, "eqv3": build_eqv3}
VERSIONS = {
    "esen": "fairchem-core 1.10.0",
    "uma": "fairchem-core 2.21.0",
    "eqv3": "atomicarchitects/equiformer_v3 a7300c5",
}


def _structures(npz: Path, limit: int | None = None):
    d = np.load(npz)
    off = np.concatenate([[0], np.cumsum(d["n_atoms"])])
    n = len(d["ids"]) if limit is None else min(limit, len(d["ids"]))
    return [
        {
            "pos": d["positions"][off[i] : off[i + 1]].copy(),
            "cell": d["cells"][i].copy(),
            "z": d["z"][off[i] : off[i + 1]].copy(),
        }
        for i in range(n)
    ]


def _data(s: dict, transform: np.ndarray | None = None) -> Shim:
    pos, cell = s["pos"], s["cell"]
    if transform is not None:
        pos = pos @ transform.T
        cell = cell @ transform.T
    z = torch.tensor(s["z"], dtype=torch.long)
    n = len(z)
    return Shim(
        pos=torch.tensor(pos, dtype=torch.float32),
        atomic_numbers=z,
        atomic_numbers_full=z,
        cell=torch.tensor(cell, dtype=torch.float32).unsqueeze(0),
        natoms=torch.tensor([n]),
        batch=torch.zeros(n, dtype=torch.long),
        batch_full=torch.zeros(n, dtype=torch.long),
        num_graphs=1,
        pbc=torch.tensor([[True, True, True]]),
        charge=torch.zeros(1, dtype=torch.long),
        spin=torch.zeros(1, dtype=torch.long),
        dataset=["omat"],
    )


def _so3_to_e3nn(emb: torch.Tensor) -> torch.Tensor:
    """(n,(lmax+1)^2,C) coeff(l,m)-major -> e3nn 'Cx0e+..+CxLe' mul-major (EquiformerV2 remap)."""
    out, offset = [], 0
    for deg in range(LMAX + 1):
        n = 2 * deg + 1
        out.append(emb[:, offset : offset + n, :].transpose(1, 2).reshape(emb.shape[0], -1))
        offset += n
    return torch.cat(out, dim=1)


class RankThreeHead(torch.nn.Module):
    """The SO(3)-arm construction: all-even source irreps, output relabeled all-even."""

    def __init__(self, backbone) -> None:
        super().__init__()
        self.backbone = backbone
        src = o3.Irreps("+".join(f"{CHANNELS}x{deg}e" for deg in range(LMAX + 1)))
        self.readout = o3.Linear(src, o3.Irreps("2x1e+1x2e+1x3e"))

    def forward(self, data: Shim) -> torch.Tensor:
        emb = self.backbone(data)["node_embedding"]
        per_atom = self.readout(_so3_to_e3nn(emb))
        return per_atom.sum(dim=0)  # single-structure batches


def _predict(model: RankThreeHead, structures: list[dict], seed: int) -> np.ndarray:
    preds = []
    for s in structures:
        torch.manual_seed(seed)  # models with stochastic edge frames get a seeded draw
        with torch.no_grad():
            preds.append(model(_data(s)).numpy())
    return np.stack(preds)


def structural_test(model: RankThreeHead, families: np.ndarray) -> dict:
    structures = _structures(OOD_NPZ)
    preds = _predict(model, structures, seed=0)
    v = np.linalg.norm(preds, axis=1)
    out: dict = {"n": int(v.size)}
    scale = float(np.median(v[families == "non-cubic"]))
    for fam in ("m-3m", "m-3", "non-cubic"):
        m = v[families == fam]
        out[fam] = {
            "n": int(m.size),
            "median": float(np.median(m)),
            "max": float(m.max()),
            "median_over_noncubic_median": float(np.median(m) / scale),
        }
    # Scale-free "false-flag-like" statistic for an untrained model: the fraction of crystals
    # whose ||T|| exceeds 1% of the typical (non-cubic median) magnitude. For an exactly
    # rotation-equivariant backbone the m-3m crystals must sit at machine zero and fail this.
    out["fraction_above_1pct_of_noncubic_median"] = float((v > 0.01 * scale).mean())
    return out, v


def _wigner(matrix: np.ndarray) -> np.ndarray:
    irreps = o3.Irreps(PHYSICAL_IRREPS)
    return irreps.D_from_matrix(torch.tensor(matrix, dtype=torch.float64)).numpy()


def _random_orthogonal(rng: np.random.Generator, improper: bool) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if (np.linalg.det(q) < 0) != improper:
        q = q @ np.diag([-1.0, 1.0, 1.0])
    return q


def output_probe(model: RankThreeHead) -> dict:
    """E5 laws on non-centrosymmetric structures: mirror, rotation, determinism."""
    rng = np.random.default_rng(0)
    structures = _structures(PIEZO_NPZ, limit=N_PROBE)
    mirror = np.diag([-1.0, 1.0, 1.0])
    rotation = _random_orthogonal(rng, improper=False)
    d_mirror, d_rotation = _wigner(mirror), _wigner(rotation)

    def batch(transform=None, seed=0):
        preds = []
        for s in structures:
            torch.manual_seed(seed)
            with torch.no_grad():
                preds.append(model(_data(s, transform)).numpy())
        return np.stack(preds)

    base = batch()
    mirrored = batch(mirror)
    rotated = batch(rotation)
    redraw = batch(seed=1)

    def rel(actual, expected):
        num = np.linalg.norm(actual - expected, axis=1)
        den = np.linalg.norm(base, axis=1)
        # The relative laws are meaningful only where ||T|| is well above the float noise
        # floor: high-symmetry members of the piezo set drive a random-init head to machine
        # zero, and a ratio there divides noise by noise (the E5 caveat). Keep structures
        # within 1e-3 of the largest response.
        keep = den > 1e-3 * den.max()
        return float(np.median(num[keep] / den[keep]))

    return {
        "mirror_rel_error": rel(mirrored, base @ d_mirror.T),
        "rotation_rel_error": rel(rotated, base @ d_rotation.T),
        "determinism_spread": float(np.abs(redraw - base).max()),
        "n_structures": len(structures),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(BUILDERS), required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = RankThreeHead(BUILDERS[args.model]())
    n_params = sum(int(p.numel()) for p in model.parameters())

    records = json.loads(SPACEGROUPS.read_text())["records"]
    families = np.array([r["family"] for r in records])

    probe = output_probe(model)
    print(
        f"{args.model}: mirror={probe['mirror_rel_error']:.3e} "
        f"rot={probe['rotation_rel_error']:.3e} det={probe['determinism_spread']:.3e}"
    )
    struct, v = structural_test(model, families)
    print(
        f"{args.model}: m-3m median={struct['m-3m']['median']:.3e} "
        f"non-cubic median={struct['non-cubic']['median']:.3e} "
        f"ratio={struct['m-3m']['median_over_noncubic_median']:.3e}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{args.model}_violations.npy", v)
    merged = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    merged[args.model] = {
        "version": VERSIONS[args.model],
        "n_params": n_params,
        "lmax": LMAX,
        "channels": CHANNELS,
        "random_init": True,
        "output_probe": probe,
        "structural_test": struct,
    }
    OUT_JSON.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
