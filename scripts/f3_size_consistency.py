"""F3 -- the size-consistency (supercell) control.

A reviewer noted that the readout sums per-atom/per-edge contributions (index_add_ in every
core, see the module docstrings in equiparity.models.{nequip,allegro}), so the predicted tensor
is extensive where the piezoelectric/elastic tensor is intensive: predictions on a primitive
cell and on a supercell of the SAME crystal are not guaranteed to agree. methods.tex and
supplementary.tex currently admit this was not tested. This script tests it.

No training is needed: Theorem 1 (the structural zero) holds at ANY parameter values, so a
randomly initialized model demonstrates the pooling behaviour exactly -- the same logic
scripts/t2_backbone_probe.py and scripts/f2_per_atom.py already rely on.

Route: the committed processed npz (data/raw/mp/mp_ood_centrosymmetric_processed.npz) is NOT
present in this checkout (only the manifest data/manifests/mp_ood_centrosymmetric.yaml and the
split data/splits/*.npz are committed; the raw MP fetch requires network access to the
Materials Project API). This script therefore builds idealized centrosymmetric test crystals
directly with ase.spacegroup.crystal -- exactly space-group-symmetric coordinates, the same
"idealized" convention scripts/idealize_ood.py applies to the MP-derived OOD set (idealize_ood.py
snaps raw MP coordinates onto the space group because "the piezoelectric tensor is exactly zero
for the IDEAL centrosymmetric crystal"). Nine crystals spanning six distinct space groups (225,
221, 227, 136, 141, 167) and the point groups already used as the OOD family split (m-3m,
non-cubic) are ample: this is a deterministic algebraic property of the readout (sum vs. mean
over an index_add_), not a statistic that needs a large sample to resolve.

Cores: NequIP and Allegro (both installed cleanly in a dedicated e3nn-0.6.x env; MACE needs a
separate e3nn-0.4.4 env and is added if that env is available). Both O(3) and SO(3) arms of each
core are tested, at random init, float64.

For each crystal x each multiplicity (1x1x1 primitive, 2x2x2 = 8 replicas, 2x1x1 = 2 replicas,
3x1x1 = 3 replicas) x each core x each arm x each pooling mode (sum via index_add_, mean = sum /
atom_count for NequIP or / edge_count for Allegro):
  1. predict the tensor
  2. record ||T|| and the ratio ||T(supercell)|| / ||T(primitive)||

Prediction: because the graph is periodic at a fixed radial cutoff, every atom of the supercell
has an environment identical to its primitive-cell counterpart (checked explicitly below via the
edges-per-atom / edges-per-edge-partner ratio), so per-atom (or per-edge) contributions are
IDENTICAL between primitive and supercell. Therefore sum pooling scales by the replica count and
mean pooling is exactly invariant.

    python scripts/f3_size_consistency.py                 # NequIP + Allegro
    python scripts/f3_size_consistency.py --core nequip    # one core only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.spacegroup import crystal

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

OUT_JSON = REPO / "results" / "f3_size_consistency.json"
OUT_DIR = REPO / "results" / "f3"

R_MAX = 5.0
L_MAX = 2
SEED = 42
O3_OUTPUT_IRREPS = "2x1o+1x2o+1x3o"  # the piezoelectric tensor (domain/target.py PIEZOELECTRIC)
MULTIPLICITIES = {"1x1x1": (1, 1, 1), "2x2x2": (2, 2, 2), "2x1x1": (2, 1, 1), "3x1x1": (3, 1, 1)}
CENTROSYMMETRIC_SPACE_GROUPS = frozenset(
    (
        2,
        *range(10, 16),
        *range(47, 75),
        *range(83, 89),
        *range(123, 143),
        147,
        148,
        *range(162, 168),
        175,
        176,
        *range(191, 195),
        *range(200, 207),
        *range(221, 231),
    )
)


def build_crystals() -> dict[str, dict]:
    """Seven idealized centrosymmetric test crystals (ase.spacegroup.crystal -- exact symmetry
    by construction, the same "idealized" convention as scripts/idealize_ood.py)."""
    specs = {
        "NaCl_rocksalt": dict(
            symbols=["Na", "Cl"],
            basis=[[0, 0, 0], [0.5, 0.5, 0.5]],
            spacegroup=225,
            cellpar=[5.6402, 5.6402, 5.6402, 90, 90, 90],
            family="m-3m",
        ),
        "MgO_rocksalt": dict(
            symbols=["Mg", "O"],
            basis=[[0, 0, 0], [0.5, 0.5, 0.5]],
            spacegroup=225,
            cellpar=[4.212, 4.212, 4.212, 90, 90, 90],
            family="m-3m",
        ),
        "CsCl": dict(
            symbols=["Cs", "Cl"],
            basis=[[0, 0, 0], [0.5, 0.5, 0.5]],
            spacegroup=221,
            cellpar=[4.123, 4.123, 4.123, 90, 90, 90],
            family="m-3m",
        ),
        "SrTiO3_perovskite": dict(
            symbols=["Sr", "Ti", "O"],
            basis=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]],
            spacegroup=221,
            cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
            family="m-3m",
        ),
        "Si_diamond": dict(
            symbols=["Si"],
            basis=[[0, 0, 0]],
            spacegroup=227,
            cellpar=[5.431, 5.431, 5.431, 90, 90, 90],
            family="m-3m",
        ),
        "CaF2_fluorite": dict(
            symbols=["Ca", "F"],
            basis=[[0, 0, 0], [0.25, 0.25, 0.25]],
            spacegroup=225,
            cellpar=[5.463, 5.463, 5.463, 90, 90, 90],
            family="m-3m",
        ),
        "TiO2_rutile": dict(
            symbols=["Ti", "O"],
            basis=[[0, 0, 0], [0.3053, 0.3053, 0.0]],
            spacegroup=136,
            cellpar=[4.5937, 4.5937, 2.9587, 90, 90, 90],
            family="non-cubic",
        ),
        "TiO2_anatase": dict(
            symbols=["Ti", "O"],
            basis=[[0, 0, 0], [0, 0, 0.2081]],
            spacegroup=141,
            cellpar=[3.785, 3.785, 9.514, 90, 90, 90],
            family="non-cubic",
        ),
        "Al2O3_corundum": dict(
            symbols=["Al", "O"],
            basis=[[0, 0, 0.3522], [0.3062, 0, 0.25]],
            spacegroup=167,
            cellpar=[4.7602, 4.7602, 12.9933, 90, 90, 120],
            family="non-cubic",
        ),
    }
    out = {}
    for name, spec in specs.items():
        family = spec.pop("family")
        atoms = crystal(spec.pop("symbols"), basis=spec.pop("basis"), **spec)
        out[name] = {"atoms": atoms, "family": family, "spacegroup": spec["spacegroup"]}
    return out


def verify_centrosymmetric(crystals: dict[str, dict]) -> None:
    """spglib-verify every test crystal is centrosymmetric, the repo's stated convention
    (equiparity.domain.spacegroup.is_centrosymmetric)."""
    import spglib

    for name, rec in crystals.items():
        atoms = rec["atoms"]
        cell = (atoms.get_cell()[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
        ds = spglib.get_symmetry_dataset(cell, symprec=1e-3)
        number = int(ds.number)
        assert number in CENTROSYMMETRIC_SPACE_GROUPS, (
            f"{name}: spglib space group {number} ({ds.international}) is NOT centrosymmetric"
        )
        rec["spglib_number"] = number
        rec["spglib_symbol"] = str(ds.international)


def type_map(atoms) -> tuple[tuple[str, ...], dict[str, int]]:
    symbols = sorted(set(atoms.get_chemical_symbols()))
    return tuple(symbols), {s: i for i, s in enumerate(symbols)}


def to_atomic_data(atoms, symbol_to_type: dict[str, int], r_max: float, dtype):
    from nequip.data import AtomicDataDict, compute_neighborlist_, from_ase

    data = compute_neighborlist_(from_ase(atoms), r_max=r_max)
    data[AtomicDataDict.ATOM_TYPE_KEY] = torch.tensor(
        [[symbol_to_type[s]] for s in atoms.get_chemical_symbols()], dtype=torch.long
    )
    for key, value in data.items():
        if torch.is_tensor(value) and value.is_floating_point():
            data[key] = value.to(dtype)
    n = len(atoms)
    data[AtomicDataDict.BATCH_KEY] = torch.zeros(n, dtype=torch.long)
    return data


def edges_per_atom(atoms, symbol_to_type, r_max) -> float:
    from nequip.data import AtomicDataDict

    data = to_atomic_data(atoms, symbol_to_type, r_max, torch.float64)
    n_edges = int(data[AtomicDataDict.EDGE_INDEX_KEY].shape[1])
    return n_edges / len(atoms)


# --------------------------------------------------------------------------------------------
# Core builders. Each returns (o3_model, so3_model, forward_sum, forward_mean, pool_unit),
# where pool_unit is "atom" (NequIP) or "edge" (Allegro): mean pooling divides the accumulated
# sum by the per-structure atom count or edge count respectively (task spec).
# --------------------------------------------------------------------------------------------


def build_nequip_pair(type_names: tuple[str, ...], avg_num_neighbors: float):
    from equiparity.domain.parity import ParityMode
    from equiparity.models.nequip import NequIPConfig, NequIPTensorModel

    cfg = NequIPConfig(
        r_max=R_MAX,
        type_names=type_names,
        num_layers=3,
        l_max=L_MAX,
        num_features=16,
        type_embed_num_features=16,
        avg_num_neighbors=avg_num_neighbors,
        seed=SEED,
        model_dtype="float64",
    )
    torch.manual_seed(0)
    o3_model = NequIPTensorModel(cfg, ParityMode.O3, O3_OUTPUT_IRREPS).eval()
    torch.manual_seed(0)
    so3_model = NequIPTensorModel(cfg, ParityMode.SO3, O3_OUTPUT_IRREPS).eval()
    return o3_model, so3_model, "atom"


def build_allegro_pair(type_names: tuple[str, ...], avg_num_neighbors: float):
    from equiparity.domain.parity import ParityMode
    from equiparity.models.allegro import AllegroConfig, AllegroTensorModel

    cfg = AllegroConfig(
        r_max=R_MAX,
        type_names=type_names,
        num_layers=2,
        l_max=L_MAX,
        num_scalar_features=16,
        num_tensor_features=8,
        avg_num_neighbors=avg_num_neighbors,
        seed=SEED,
        model_dtype="float64",
    )
    torch.manual_seed(0)
    o3_model = AllegroTensorModel(cfg, ParityMode.O3, O3_OUTPUT_IRREPS).eval()
    torch.manual_seed(0)
    so3_model = AllegroTensorModel(cfg, ParityMode.SO3, O3_OUTPUT_IRREPS).eval()
    return o3_model, so3_model, "edge"


CORE_BUILDERS = {"nequip": build_nequip_pair, "allegro": build_allegro_pair}


def predict_sum(model, atoms, symbol_to_type) -> np.ndarray:
    data = to_atomic_data(atoms, symbol_to_type, R_MAX, torch.float64)
    with torch.no_grad():
        return model(data).numpy()[0]


def predict_mean(model, atoms, symbol_to_type, pool_unit: str) -> np.ndarray:
    """Same forward pass as predict_sum, but divide the accumulated sum by the per-structure
    atom count (NequIP, node-pooled) or edge count (Allegro, edge-pooled)."""
    from nequip.data import AtomicDataDict

    data = to_atomic_data(atoms, symbol_to_type, R_MAX, torch.float64)
    with torch.no_grad():
        total = model(data).numpy()[0]
    if pool_unit == "atom":
        denom = len(atoms)
    elif pool_unit == "edge":
        denom = int(data[AtomicDataDict.EDGE_INDEX_KEY].shape[1])
    else:
        raise ValueError(pool_unit)
    return total / denom


NOISE_FLOOR = 1e-6  # so3_sum_norm(primitive) below this: random-init output is itself at the
# machine-precision noise floor (the cubic "rotation ceiling" from t2_backbone_probe.py -- an
# exactly rotation-equivariant model is forced near machine zero on m-3m crystals regardless of
# parity), so a supercell/primitive ratio there divides noise by noise and is not informative
# about the pooling law. Flagged per crystal, not discarded -- reported alongside the resolved set.


def run_core(core: str, crystals: dict[str, dict]) -> dict:
    builder = CORE_BUILDERS[core]
    per_crystal = {}
    for name, rec in crystals.items():
        atoms = rec["atoms"]
        type_names, symbol_to_type = type_map(atoms)
        avg_neigh = edges_per_atom(atoms, symbol_to_type, R_MAX)
        o3_model, so3_model, pool_unit = builder(type_names, avg_neigh)
        n_params = sum(int(p.numel()) for p in o3_model.parameters())

        per_mult = {}
        eff_neigh = {}
        for mult_name, mult in MULTIPLICITIES.items():
            super_atoms = atoms * mult
            replicas = mult[0] * mult[1] * mult[2]
            eff_neigh[mult_name] = edges_per_atom(super_atoms, symbol_to_type, R_MAX)

            o3_sum = predict_sum(o3_model, super_atoms, symbol_to_type)
            so3_sum = predict_sum(so3_model, super_atoms, symbol_to_type)
            o3_mean = predict_mean(o3_model, super_atoms, symbol_to_type, pool_unit)
            so3_mean = predict_mean(so3_model, super_atoms, symbol_to_type, pool_unit)

            per_mult[mult_name] = {
                "replicas": replicas,
                "n_atoms": len(super_atoms),
                "o3_sum_norm": float(np.linalg.norm(o3_sum)),
                "so3_sum_norm": float(np.linalg.norm(so3_sum)),
                "o3_mean_norm": float(np.linalg.norm(o3_mean)),
                "so3_mean_norm": float(np.linalg.norm(so3_mean)),
            }

        # ratios relative to the primitive cell (1x1x1)
        prim = per_mult["1x1x1"]
        for rec_m in per_mult.values():
            rec_m["so3_sum_ratio"] = (
                rec_m["so3_sum_norm"] / prim["so3_sum_norm"] if prim["so3_sum_norm"] > 0 else None
            )
            rec_m["so3_mean_ratio"] = (
                rec_m["so3_mean_norm"] / prim["so3_mean_norm"]
                if prim["so3_mean_norm"] > 0
                else None
            )
            rec_m["so3_sum_ratio_deviation_from_replicas"] = (
                None
                if rec_m["so3_sum_ratio"] is None
                else rec_m["so3_sum_ratio"] - rec_m["replicas"]
            )
            rec_m["so3_mean_ratio_deviation_from_1"] = (
                None if rec_m["so3_mean_ratio"] is None else rec_m["so3_mean_ratio"] - 1.0
            )

        per_crystal[name] = {
            "family": rec["family"],
            "spacegroup": rec["spglib_number"],
            "spacegroup_symbol": rec["spglib_symbol"],
            "n_atoms_primitive": len(atoms),
            "type_names": type_names,
            "avg_num_neighbors_primitive": avg_neigh,
            "edges_per_atom_by_multiplicity": eff_neigh,
            "n_params": n_params,
            "pool_unit": pool_unit,
            "so3_sum_norm_primitive": prim["so3_sum_norm"],
            "above_noise_floor": prim["so3_sum_norm"] > NOISE_FLOOR,
            "by_multiplicity": per_mult,
        }
    return per_crystal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", choices=sorted(CORE_BUILDERS), default=None, help="one core only")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    cores = [args.core] if args.core else sorted(CORE_BUILDERS)

    crystals = build_crystals()
    verify_centrosymmetric(crystals)
    print(f"built {len(crystals)} idealized centrosymmetric crystals, spglib-verified:")
    for name, rec in crystals.items():
        print(
            f"  {name:20s} sg={rec['spglib_number']:4d} "
            f"{rec['spglib_symbol']:10s} family={rec['family']}"
        )

    t0 = time.time()
    prior = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else None
    results: dict = {
        "provenance": {
            "route": "idealized_ase_spacegroup_crystal",
            "reason": (
                "data/raw/mp/mp_ood_centrosymmetric_processed.npz is not present in this "
                "checkout (requires a live Materials Project API fetch via "
                "scripts/prepare_mp.py); built idealized centrosymmetric crystals directly "
                "with ase.spacegroup.crystal instead, matching the 'idealized' convention "
                "scripts/idealize_ood.py applies to the MP-derived OOD set"
            ),
            "n_crystals": len(crystals),
            "crystal_names": sorted(crystals),
            "families": {name: rec["family"] for name, rec in crystals.items()},
            "spacegroups": {name: rec["spglib_number"] for name, rec in crystals.items()},
            "r_max": R_MAX,
            "l_max": L_MAX,
            "o3_output_irreps": O3_OUTPUT_IRREPS,
            "multiplicities": {k: list(v) for k, v in MULTIPLICITIES.items()},
            "precision": "float64",
            "random_init": True,
            "seed": SEED,
            "cores_run": cores,
        },
        "cores": {},
    }
    for core in cores:
        print(f"running core={core} ...")
        results["cores"][core] = run_core(core, crystals)
    results["provenance"]["wall_seconds"] = time.time() - t0

    # Merge with any prior run (e.g. --core nequip then --core allegro in separate invocations)
    # so the JSON accumulates every core run so far instead of only the latest.
    if prior is not None and "cores" in prior:
        merged_cores = dict(prior["cores"])
        merged_cores.update(results["cores"])
        results["cores"] = merged_cores
        merged_names = sorted(merged_cores)
        results["provenance"]["cores_run"] = merged_names
        prior_wall = prior.get("provenance", {}).get("wall_seconds", 0.0)
        results["provenance"]["wall_seconds"] += prior_wall

    # Resolved-set summary: which crystals give an informative (above-noise-floor) SO(3) ratio,
    # and the max deviation from the exact prediction on that subset, per core.
    summary = {}
    for core, per_crystal in results["cores"].items():
        resolved = [n for n, r in per_crystal.items() if r["above_noise_floor"]]
        unresolved = [n for n, r in per_crystal.items() if not r["above_noise_floor"]]
        max_sum_dev = max(
            (
                abs(m["so3_sum_ratio_deviation_from_replicas"])
                for n in resolved
                for m in per_crystal[n]["by_multiplicity"].values()
                if m["so3_sum_ratio_deviation_from_replicas"] is not None
            ),
            default=None,
        )
        max_mean_dev = max(
            (
                abs(m["so3_mean_ratio_deviation_from_1"])
                for n in resolved
                for m in per_crystal[n]["by_multiplicity"].values()
                if m["so3_mean_ratio_deviation_from_1"] is not None
            ),
            default=None,
        )
        max_o3_sum_norm = max(
            m["o3_sum_norm"]
            for n in per_crystal
            for m in per_crystal[n]["by_multiplicity"].values()
        )
        max_o3_mean_norm = max(
            m["o3_mean_norm"]
            for n in per_crystal
            for m in per_crystal[n]["by_multiplicity"].values()
        )
        summary[core] = {
            "resolved_crystals": resolved,
            "unresolved_crystals_noise_floor": unresolved,
            "noise_floor": NOISE_FLOOR,
            "max_abs_so3_sum_ratio_deviation_from_replicas_on_resolved": max_sum_dev,
            "max_abs_so3_mean_ratio_deviation_from_1_on_resolved": max_mean_dev,
            "max_o3_sum_norm_over_all_crystals_and_multiplicities": max_o3_sum_norm,
            "max_o3_mean_norm_over_all_crystals_and_multiplicities": max_o3_mean_norm,
        }
    results["summary"] = summary

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT_JSON}")
    for core, s in summary.items():
        print(
            f"{core}: resolved={len(s['resolved_crystals'])} "
            f"unresolved={len(s['unresolved_crystals_noise_floor'])} "
            "max|so3_sum_dev|="
            f"{s['max_abs_so3_sum_ratio_deviation_from_replicas_on_resolved']:.3e} "
            f"max|so3_mean_dev|={s['max_abs_so3_mean_ratio_deviation_from_1_on_resolved']:.3e} "
            f"max_o3_sum={s['max_o3_sum_norm_over_all_crystals_and_multiplicities']:.3e} "
            f"max_o3_mean={s['max_o3_mean_norm_over_all_crystals_and_multiplicities']:.3e}"
        )

    # console summary
    for core in cores:
        print(f"\n=== {core} ===")
        for name, rec in results["cores"][core].items():
            for mult_name in MULTIPLICITIES:
                m = rec["by_multiplicity"][mult_name]
                print(
                    f"  {name:20s} {mult_name:6s} replicas={m['replicas']} "
                    f"o3_sum={m['o3_sum_norm']:.3e} o3_mean={m['o3_mean_norm']:.3e} "
                    f"so3_sum_ratio={m['so3_sum_ratio']:.6f} "
                    f"so3_mean_ratio={m['so3_mean_ratio']:.6f}"
                )


if __name__ == "__main__":
    main()
