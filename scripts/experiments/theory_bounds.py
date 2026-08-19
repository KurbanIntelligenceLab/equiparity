"""Quantitative test of Proposition 1 and the stability clause of Corollary 3.

Proposition 1 (arithmetic floor): on a centrosymmetric x, with ``delta = ||f(I.x) - f(x)||``
(failure of A1-A2) and ``eta = ||f(I.x) + f(x)||`` (failure of the O(3) law at Q = I), every model
satisfies ``||f(x)|| <= (delta + eta)/2``. Measured here as absolute quantities per structure over
the full 2,000-crystal idealized evaluation population, for every arm.

Corollary 3 (rotation ceiling, stability clause): on class m-3m, with ``delta_plus = max_R
||f(R.x) - f(x)||`` and ``epsilon = max_R ||f(R.x) - D(R) f(x)|| / ||f(x)||`` over the
proper-rotation subgroup G+(x) = 432, an approximately SO(3)-equivariant model satisfies
``||f(x)|| <= delta_plus / (1 - epsilon)`` wherever ``epsilon < 1``. Measured over the full proper
subgroup (spglib rotations of the idealized structure, converted to Cartesian) on the 166 m-3m
crystals, for the SO(3) arms the clause bounds.

Run once per install profile:

    uv run --extra nequip python scripts/experiments/theory_bounds.py \
        --cores nequip allegro equiformer_v2
    uv run --extra mace   python scripts/experiments/theory_bounds.py --cores mace

Results merge into ``results/theory_bounds.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import spglib
import torch

from equiparity.domain.structure import AtomicStructure
from equiparity.inference import find_piezo_runs, load_trained

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inversion_averaging import _invert, _norms
from output_parity import _transform, _wigner

REPO = Path(__file__).resolve().parents[2]
MIRROR = Path(os.environ.get("PARITY_RUNS", Path(__file__).resolve().parents[2] / "runs"))
IDEALIZED_NPZ = REPO / "data" / "raw" / "mp" / "mp_ood_centrosymmetric_processed.npz"
SPACEGROUPS = REPO / "results" / "ood_spacegroups.json"
OUT_JSON = REPO / "results" / "theory_bounds.json"
OUT_MD = REPO / "docs" / "results" / "theory_bounds.md"


def _structures() -> list[AtomicStructure]:
    from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset

    ds = CrystalDataset(load_crystal_dataset(IDEALIZED_NPZ))
    return [ds[i].structure for i in range(len(ds))]


def _m3m_indices() -> list[int]:
    records = json.loads(SPACEGROUPS.read_text())["records"]
    return [r["index"] for r in records if r["family"] == "m-3m"]


def _proper_rotations(s: AtomicStructure) -> list[np.ndarray]:
    """Unique non-identity Cartesian rotations of the proper subgroup G+(x)."""
    cell = s.cell
    assert cell is not None
    frac = s.positions @ np.linalg.inv(cell)
    ds = spglib.get_symmetry_dataset((cell, frac, s.atomic_numbers), symprec=1e-3)
    seen, out = set(), []
    for w in ds.rotations:
        key = w.tobytes()
        if key in seen or np.array_equal(w, np.eye(3, dtype=w.dtype)):
            continue
        seen.add(key)
        if round(float(np.linalg.det(w))) != 1:
            continue
        r_cart = cell.T @ w @ np.linalg.inv(cell).T
        if not np.allclose(r_cart @ r_cart.T, np.eye(3), atol=1e-6):
            continue  # lattice-incompatible numerical case; skip conservatively
        out.append(r_cart)
    return out


def proposition1(trained) -> dict:
    """Absolute delta, eta and the floor check over the full idealized population."""
    structures = _structures()
    torch.manual_seed(0)
    t_x = trained.predict(structures)
    torch.manual_seed(0)
    t_ix = trained.predict([_invert(s) for s in structures])

    mag = _norms(t_x)
    delta = _norms(t_ix - t_x)
    eta = _norms(t_ix + t_x)
    bound = (delta + eta) / 2.0
    return {
        "n": int(mag.size),
        "median_violation": float(np.median(mag)),
        "delta_median": float(np.median(delta)),
        "delta_p95": float(np.percentile(delta, 95)),
        "delta_max": float(delta.max()),
        "eta_median": float(np.median(eta)),
        "eta_p95": float(np.percentile(eta, 95)),
        "eta_max": float(eta.max()),
        "floor_bound_median": float(np.median(bound)),
        # the floor is a triangle-inequality identity, so any excess is float rounding;
        # record it relative to the bound rather than pass/fail against an absolute epsilon
        "fraction_within_floor": float((mag <= bound * (1 + 1e-6) + 1e-12).mean()),
        "floor_max_rel_excess": float(((mag - bound) / np.maximum(bound, 1e-300)).max()),
    }


def corollary3(trained) -> dict:
    """delta_plus, epsilon over the full proper subgroup on the 166 m-3m crystals."""
    structures = _structures()
    idx = _m3m_indices()
    d_plus, eps, g_abs, mags, applicable_ok = [], [], [], [], []
    for i in idx:
        s = structures[i]
        rots = _proper_rotations(s)
        if not rots:
            continue
        torch.manual_seed(0)
        base = trained.predict([s])[0]
        # one rotated copy at a time: 23 copies of a large cubic cell in one batch OOM
        # Allegro's strided tensor-product contraction
        chunks = []
        for r in rots:
            torch.manual_seed(0)
            chunks.append(trained.predict([_transform(s, r)]))
        rotated = np.concatenate(chunks)
        mag = float(np.linalg.norm(base))
        dp = float(np.max(_norms(rotated - base[None, :])))
        expected = np.stack([_wigner(r) @ base for r in rots])
        ga = float(np.max(_norms(rotated - expected)))  # absolute (A3) defect over the group
        ep = ga / mag if mag > 0 else float("inf")
        mags.append(mag)
        d_plus.append(dp)
        eps.append(ep)
        g_abs.append(ga)
        if ep < 1.0:
            applicable_ok.append(mag <= dp / (1.0 - ep) + 1e-12)
    eps_arr = np.asarray(eps)
    mags_arr, dp_arr, ga_arr = np.asarray(mags), np.asarray(d_plus), np.asarray(g_abs)
    return {
        "n": len(mags),
        "n_rotations": "full proper subgroup per structure (23 for point group m-3m)",
        "median_violation": float(np.median(mags_arr)),
        "delta_plus_median": float(np.median(dp_arr)),
        "delta_plus_max": float(np.max(dp_arr)),
        "g_abs_median": float(np.median(ga_arr)),
        "epsilon_median": float(np.median(eps_arr[np.isfinite(eps_arr)])),
        "n_epsilon_ge_1": int((~(eps_arr < 1.0)).sum()),
        "n_bound_applicable": len(applicable_ok),
        "fraction_within_bound_where_applicable": (
            float(np.mean(applicable_ok)) if applicable_ok else float("nan")
        ),
        # the absolute form of the stability clause, ||f|| <= delta_plus + max_R ||g_R||,
        # needs no epsilon < 1 and is checkable on every structure
        "fraction_within_absolute_bound": float((mags_arr <= dp_arr + ga_arr + 1e-12).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", nargs="+", default=["nequip", "allegro", "equiformer_v2"])
    args = parser.parse_args()

    runs = find_piezo_runs(MIRROR)
    merged = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    for label in sorted(runs):
        core = label.split("_o3_")[0].split("_so3_")[0]
        if core not in args.cores or label in merged:
            continue
        trained = load_trained(runs[label], repo_root=REPO)
        entry: dict[str, object] = {"core": core, "parity": trained.parity}
        entry["proposition1"] = proposition1(trained)
        if trained.parity == "so3":
            entry["corollary3_m3m"] = corollary3(trained)
        merged[label] = entry
        p1 = entry["proposition1"]
        print(
            f"{label:38s} delta_med={p1['delta_median']:.3e} eta_med={p1['eta_median']:.3e} "
            f"floor_ok={p1['fraction_within_floor']:.4f}"
        )
        if "corollary3_m3m" in entry:
            c3 = entry["corollary3_m3m"]
            print(
                f"{'':38s} m-3m: d+={c3['delta_plus_median']:.3e} "
                f"eps_med={c3['epsilon_median']:.3e} eps>=1: {c3['n_epsilon_ge_1']} "
                f"bound_ok={c3['fraction_within_bound_where_applicable']:.4f}"
            )
        OUT_JSON.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    OUT_JSON.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT_JSON} ({len(merged)} runs)")


if __name__ == "__main__":
    main()
