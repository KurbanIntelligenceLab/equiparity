"""The size-consistency (supercell) control, MACE core.

MACE uses a different data pipeline (mace.data.AtomicData / torch_geometric batching, not NequIP's
AtomicDataDict) and a conflicting e3nn version (0.4.4 vs NequIP/Allegro's 0.6.x), so it runs in
its own dedicated env and its own thin script; see scripts/experiments/size_consistency.py for the
full docstring, rationale, and the NequIP/Allegro results. Same idealized centrosymmetric test
crystals, same multiplicities, same random-init/float64/no-training logic (Theorem 1 holds at any
parameter values). Merges into the same results/size_consistency.json under cores.mace.

    python scripts/experiments/size_consistency_mace.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.spacegroup import crystal

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

OUT_JSON = REPO / "results" / "size_consistency.json"

R_MAX = 5.0
L_MAX = 2
SEED = 42
O3_OUTPUT_IRREPS = "2x1o+1x2o+1x3o"
MULTIPLICITIES = {"1x1x1": (1, 1, 1), "2x2x2": (2, 2, 2), "2x1x1": (2, 1, 1), "3x1x1": (3, 1, 1)}
NOISE_FLOOR = 1e-6
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
    import spglib

    for name, rec in crystals.items():
        atoms = rec["atoms"]
        cell = (atoms.get_cell()[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
        ds = spglib.get_symmetry_dataset(cell, symprec=1e-3)
        number = int(ds.number)
        assert number in CENTROSYMMETRIC_SPACE_GROUPS, f"{name}: sg {number} not centrosymmetric"
        rec["spglib_number"] = number
        rec["spglib_symbol"] = str(ds.international)


def mace_batch(atoms, z_table, r_max: float, dtype):
    from mace import data
    from mace.tools import torch_geometric

    config = data.Configuration(
        atomic_numbers=np.array(atoms.get_atomic_numbers()),
        positions=atoms.get_positions(),
        cell=np.array(atoms.get_cell()),
        pbc=tuple(atoms.get_pbc()),
        properties={},
        property_weights={},
    )
    atomic_data = data.AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
    loader = torch_geometric.dataloader.DataLoader(dataset=[atomic_data], batch_size=1)
    batch = next(iter(loader)).to_dict()
    for key, value in batch.items():
        if torch.is_tensor(value) and value.is_floating_point():
            batch[key] = value.to(dtype)
    return batch


def edges_per_atom(atoms, z_table, r_max) -> float:
    batch = mace_batch(atoms, z_table, r_max, torch.float64)
    n_edges = int(batch["edge_index"].shape[1])
    return n_edges / len(atoms)


def build_mace_pair(atomic_numbers: tuple[int, ...], avg_num_neighbors: float):
    from equiparity.domain.parity import ParityMode
    from equiparity.models.mace import MACEConfig, MACETensorModel

    cfg = MACEConfig(
        r_max=R_MAX,
        atomic_numbers=atomic_numbers,
        num_interactions=2,
        l_max=L_MAX,
        num_features=16,
        avg_num_neighbors=avg_num_neighbors,
        seed=SEED,
        model_dtype="float64",
    )
    o3_model = MACETensorModel(cfg, ParityMode.O3, O3_OUTPUT_IRREPS).eval()
    so3_model = MACETensorModel(cfg, ParityMode.SO3, O3_OUTPUT_IRREPS).eval()
    return o3_model, so3_model


def predict_sum(model, atoms, z_table) -> np.ndarray:
    batch = mace_batch(atoms, z_table, R_MAX, torch.float64)
    with torch.no_grad():
        return model(batch).numpy()[0]


def predict_mean(model, atoms, z_table) -> np.ndarray:
    batch = mace_batch(atoms, z_table, R_MAX, torch.float64)
    with torch.no_grad():
        total = model(batch).numpy()[0]
    return total / len(atoms)  # MACE readout is node-pooled (per-atom index_add_)


def run_mace(crystals: dict[str, dict]) -> dict:
    from mace import tools

    per_crystal = {}
    for name, rec in crystals.items():
        atoms = rec["atoms"]
        elements = sorted(set(int(z) for z in atoms.get_atomic_numbers()))
        z_table = tools.AtomicNumberTable(elements)
        avg_neigh = edges_per_atom(atoms, z_table, R_MAX)
        o3_model, so3_model = build_mace_pair(tuple(elements), avg_neigh)
        n_params = sum(int(p.numel()) for p in o3_model.parameters())

        per_mult = {}
        eff_neigh = {}
        for mult_name, mult in MULTIPLICITIES.items():
            super_atoms = atoms * mult
            replicas = mult[0] * mult[1] * mult[2]
            eff_neigh[mult_name] = edges_per_atom(super_atoms, z_table, R_MAX)

            o3_sum = predict_sum(o3_model, super_atoms, z_table)
            so3_sum = predict_sum(so3_model, super_atoms, z_table)
            o3_mean = predict_mean(o3_model, super_atoms, z_table)
            so3_mean = predict_mean(so3_model, super_atoms, z_table)

            per_mult[mult_name] = {
                "replicas": replicas,
                "n_atoms": len(super_atoms),
                "o3_sum_norm": float(np.linalg.norm(o3_sum)),
                "so3_sum_norm": float(np.linalg.norm(so3_sum)),
                "o3_mean_norm": float(np.linalg.norm(o3_mean)),
                "so3_mean_norm": float(np.linalg.norm(so3_mean)),
            }

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
            "avg_num_neighbors_primitive": avg_neigh,
            "edges_per_atom_by_multiplicity": eff_neigh,
            "n_params": n_params,
            "pool_unit": "atom",
            "so3_sum_norm_primitive": prim["so3_sum_norm"],
            "above_noise_floor": prim["so3_sum_norm"] > NOISE_FLOOR,
            "by_multiplicity": per_mult,
        }
    return per_crystal


def main() -> None:
    torch.set_default_dtype(torch.float64)
    crystals = build_crystals()
    verify_centrosymmetric(crystals)
    print(f"built {len(crystals)} idealized centrosymmetric crystals, spglib-verified")

    t0 = time.time()
    mace_result = run_mace(crystals)
    wall = time.time() - t0

    prior = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else None
    if prior is None:
        raise SystemExit(
            "results/size_consistency.json not found -- run "
            "scripts/experiments/size_consistency.py "
            "(NequIP/Allegro) first so this script has a base file to merge into"
        )
    prior["cores"]["mace"] = mace_result
    prior["provenance"]["cores_run"] = sorted(prior["cores"])
    prior["provenance"]["wall_seconds"] = prior["provenance"].get("wall_seconds", 0.0) + wall

    resolved = [n for n, r in mace_result.items() if r["above_noise_floor"]]
    unresolved = [n for n, r in mace_result.items() if not r["above_noise_floor"]]
    max_sum_dev = max(
        (
            abs(m["so3_sum_ratio_deviation_from_replicas"])
            for n in resolved
            for m in mace_result[n]["by_multiplicity"].values()
            if m["so3_sum_ratio_deviation_from_replicas"] is not None
        ),
        default=None,
    )
    max_mean_dev = max(
        (
            abs(m["so3_mean_ratio_deviation_from_1"])
            for n in resolved
            for m in mace_result[n]["by_multiplicity"].values()
            if m["so3_mean_ratio_deviation_from_1"] is not None
        ),
        default=None,
    )
    max_o3_sum = max(
        m["o3_sum_norm"] for n in mace_result for m in mace_result[n]["by_multiplicity"].values()
    )
    max_o3_mean = max(
        m["o3_mean_norm"] for n in mace_result for m in mace_result[n]["by_multiplicity"].values()
    )
    prior.setdefault("summary", {})["mace"] = {
        "resolved_crystals": resolved,
        "unresolved_crystals_noise_floor": unresolved,
        "noise_floor": NOISE_FLOOR,
        "max_abs_so3_sum_ratio_deviation_from_replicas_on_resolved": max_sum_dev,
        "max_abs_so3_mean_ratio_deviation_from_1_on_resolved": max_mean_dev,
        "max_o3_sum_norm_over_all_crystals_and_multiplicities": max_o3_sum,
        "max_o3_mean_norm_over_all_crystals_and_multiplicities": max_o3_mean,
    }

    OUT_JSON.write_text(json.dumps(prior, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT_JSON} (merged mace)")
    print(
        f"mace: resolved={len(resolved)} unresolved={len(unresolved)} "
        f"max|so3_sum_dev|={max_sum_dev} max|so3_mean_dev|={max_mean_dev} "
        f"max_o3_sum={max_o3_sum:.3e} max_o3_mean={max_o3_mean:.3e}"
    )


if __name__ == "__main__":
    main()
