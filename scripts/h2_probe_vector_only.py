"""Flag 16 -- reflection-probe the three vector-only prevalence-audit rows.

Declared prediction (before measurement): the single Cartesian vector channel of PaiNN,
TorchMD-Net (ET) and ViSNet is built from edge vectors and is therefore implicitly POLAR --
under any improper operation g the node vector features satisfy X(g.x) = g X(x) (not the
pseudovector law det(g) g X(x)). If so, these architectures carry parity implicitly while
typing nothing, and a correctly built rank-3 readout (odd number of vector factors) would
inherit the structural zero -- unlike the SO(3)-only rows, which no readout can repair.

Probes run at random initialisation (parity is architectural), in float64:

* **TorchMD-Net (ET)** -- third_party/torchmd-net at the audit's pinned commit 2a2c913. The
  commit's warp neighbor kernel is stubbed out and the distance module replaced with a naive
  all-pairs one; neither touches the equivariant path under test.
* **ViSNet** -- torch_geometric 2.8.0 (the audit's pinned source); ``radius_graph`` (needs
  pyg-lib) is monkeypatched with a naive all-pairs implementation.
* **PaiNN** -- the audit row's pinned evidence object is torchmd-net's shared vector channel;
  for an architecture-faithful measurement the probe additionally drives schnetpack's PaiNN in
  an ephemeral env (``--with-schnetpack``, uses ``uv run --with schnetpack``).

    uv run python scripts/h2_probe_vector_only.py --with-schnetpack
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
TORCHMD = REPO / "third_party" / "torchmd-net"
TORCHMD_SHA = "2a2c913"
OUT_JSON = REPO / "results" / "h2_probes.json"

Z = [6, 1, 1, 8, 1]


def _orthogonal(rng: np.random.Generator, improper: bool) -> torch.Tensor:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if (np.linalg.det(q) < 0) != improper:
        q = q @ np.diag([-1.0, 1.0, 1.0])
    return torch.tensor(q, dtype=torch.float64)


def _laws(run, rng: np.random.Generator) -> dict:
    """Test X(g.x) = g X(x) (polar) vs det(g) g X(x) (pseudo) for proper and improper g."""
    pos = torch.tensor(rng.normal(size=(len(Z), 3)), dtype=torch.float64)
    v = run(pos)
    out = {}
    for tag, improper in (("rotation", False), ("improper", True)):
        g = _orthogonal(rng, improper)
        det = float(torch.det(g))
        vg = run(pos @ g.T)
        gv = torch.einsum("ij,ajc->aic", g, v)
        out[f"{tag}_polar_rel"] = float((vg - gv).abs().max() / v.abs().max())
        out[f"{tag}_pseudo_rel"] = float((vg - det * gv).abs().max() / v.abs().max())
    return out


def _verdict(m: dict) -> str:
    if m["rotation_polar_rel"] < 1e-10 and m["improper_polar_rel"] < 1e-10:
        return "vector channel is implicitly polar (1o); parity carried, untyped"
    if m["rotation_polar_rel"] < 1e-10:
        return "rotation-equivariant, vector channel breaks both improper laws"
    return "FAIL (not rotation-equivariant)"


def probe_torchmd_et() -> dict:
    sys.path.insert(0, str(TORCHMD))
    ops = types.ModuleType("torchmdnet.extensions.ops")
    ops.get_neighbor_pairs_kernel = None  # warp kernel unused: distance module replaced below
    sys.modules["torchmdnet.extensions.ops"] = ops
    from torchmdnet.models.torchmd_et import TorchMD_ET  # type: ignore[import-not-found]

    class NaiveDistance(torch.nn.Module):
        def forward(self, pos, batch, box=None):
            d = torch.cdist(pos, pos)
            src, dst = (d <= 5.0).nonzero(as_tuple=True)  # loop=True convention
            return torch.stack([src, dst]), d[src, dst], pos[src] - pos[dst]

    torch.manual_seed(0)
    m = TorchMD_ET(
        hidden_channels=64,
        num_layers=2,
        num_rbf=16,
        cutoff_upper=5.0,
        max_z=20,
        dtype=torch.float64,
    ).eval()
    m.distance = NaiveDistance()
    z = torch.tensor(Z)

    def run(pos):
        with torch.no_grad():
            return m(z, pos, batch=torch.zeros(len(Z), dtype=torch.long))[1]

    laws = _laws(run, np.random.default_rng(0))
    return {
        "model": "TorchMD-Net (ET)",
        "sha": TORCHMD_SHA,
        "measured": True,
        "verdict": _verdict(laws),
        **laws,
    }


def probe_visnet() -> dict:
    import torch_geometric
    import torch_geometric.nn.models.visnet as visnet_mod

    def naive_radius_graph(pos, r, batch=None, loop=False, max_num_neighbors=32):
        d = torch.cdist(pos, pos)
        mask = (d <= r) & ~torch.eye(pos.shape[0], dtype=torch.bool)
        src, dst = mask.nonzero(as_tuple=True)
        return torch.stack([src, dst])

    visnet_mod.radius_graph = naive_radius_graph
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        m = visnet_mod.ViSNetBlock(
            hidden_channels=64, num_layers=2, lmax=1, cutoff=5.0, max_z=20
        ).eval()
        z = torch.tensor(Z)

        def run(pos):
            with torch.no_grad():
                _, vec = m(z, pos, torch.zeros(len(Z), dtype=torch.long))
            return vec

        laws = _laws(run, np.random.default_rng(0))
    finally:
        torch.set_default_dtype(prev)
    return {
        "model": "ViSNet",
        "source": f"torch_geometric {torch_geometric.__version__}",
        "measured": True,
        "verdict": _verdict(laws),
        **laws,
    }


_PAINN_PROBE = r"""
import json, torch, numpy as np
import schnetpack
from schnetpack.representation import PaiNN
from schnetpack.nn.radial import GaussianRBF
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack import properties
torch.manual_seed(0)
m = PaiNN(n_atom_basis=64, n_interactions=2,
          radial_basis=GaussianRBF(n_rbf=16, cutoff=5.0),
          cutoff_fn=CosineCutoff(5.0)).double().eval()
Z = [6, 1, 1, 8, 1]
z = torch.tensor(Z)
def run(pos):
    n = pos.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    idx_i = torch.tensor([p[0] for p in pairs]); idx_j = torch.tensor([p[1] for p in pairs])
    inputs = {properties.Z: z, properties.Rij: (pos[idx_j] - pos[idx_i]).double(),
              properties.idx_i: idx_i, properties.idx_j: idx_j,
              properties.n_atoms: torch.tensor([n])}
    with torch.no_grad():
        return m(inputs)["vector_representation"]
rng = np.random.default_rng(0)
pos = torch.tensor(rng.normal(size=(len(Z), 3)), dtype=torch.float64)
v = run(pos)
out = {"schnetpack_version": schnetpack.__version__}
for tag, improper in (("rotation", False), ("improper", True)):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if (np.linalg.det(q) < 0) != improper:
        q = q @ np.diag([-1.0, 1.0, 1.0])
    g = torch.tensor(q, dtype=torch.float64)
    det = float(torch.det(g))
    vg = run(pos @ g.T)
    gv = torch.einsum('ij,ajc->aic', g, v)
    out[tag + "_polar_rel"] = float((vg - gv).abs().max() / v.abs().max())
    out[tag + "_pseudo_rel"] = float((vg - det * gv).abs().max() / v.abs().max())
print(json.dumps(out))
"""


def probe_painn(with_schnetpack: bool) -> dict:
    base = {"model": "PaiNN"}
    if not with_schnetpack:
        return {
            **base,
            "measured": False,
            "note": "audit row's pinned evidence object is torchmd-net's shared vector channel "
            "(measured above); rerun with --with-schnetpack for an architecture-faithful PaiNN",
        }
    out = subprocess.run(
        ["uv", "run", "--with", "schnetpack", "python", "-c", _PAINN_PROBE],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    if out.returncode != 0:
        return {**base, "measured": False, "error_tail": out.stderr.strip()[-300:]}
    m = json.loads(out.stdout.strip().splitlines()[-1])
    return {
        "model": "PaiNN",
        "source": f"schnetpack {m.pop('schnetpack_version')}",
        "measured": True,
        "verdict": _verdict(m),
        **m,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-schnetpack", action="store_true")
    args = ap.parse_args()

    results = {
        "torchmd_et": probe_torchmd_et(),
        "visnet": probe_visnet(),
        "painn": probe_painn(args.with_schnetpack),
    }
    for r in results.values():
        print(
            f"{r['model']:18s} measured={r['measured']} "
            + (
                f"rot_polar={r['rotation_polar_rel']:.1e} improper_polar="
                f"{r['improper_polar_rel']:.1e} improper_pseudo={r['improper_pseudo_rel']:.1e}"
                f" -> {r['verdict']}"
                if r["measured"]
                else r.get("note", r.get("error_tail", ""))
            )
        )
    merged = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    merged["vector_only"] = results
    OUT_JSON.write_text(json.dumps(merged, indent=1) + "\n")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
