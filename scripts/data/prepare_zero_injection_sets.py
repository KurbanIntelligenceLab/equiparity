"""Build the zero-injection training sets: how many exact zeros does SO(3) need to see?

Sweeps the number of exact-zero centrosymmetric crystals injected into the piezoelectric training
set: N_zero in {250, 4000, 16000}. N = 0 is the headline SO(3) run and N = 1000 the augmented run,
both reused. The sets are **nested** -- 250 is a subset of the augmentation set's exact 1,000
(loaded from ``results/augmentation_eval_split.json``), which is a subset of 4,000, which is a
subset of 16,000 -- so the curve is monotone in data and directly consistent with the reused
points.

Same design rules as the augmentation set (``prepare_augmented_piezoelectric.py``, whose helpers
are imported): candidates only from the six SEEN centrosymmetric space groups (keeps the
SEEN/UNSEEN evaluation partition valid for every N), spglib-verified and idealized, disjoint from
the OOD 2,000 and the piezo set, val/test splits inherited unchanged. If fewer than 16,000
candidates survive the filters, the largest set is capped at availability and the achieved count
recorded (the prespecified rule).

    uv run --extra nequip --extra data python scripts/data/prepare_zero_injection_sets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import spglib
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_augmented_piezoelectric import (
    N_AUGMENT as AUG_N,
)
from prepare_augmented_piezoelectric import (
    OUT_EVAL as AUG_EVAL,
)
from prepare_augmented_piezoelectric import (
    SEED,
    SEEN_SPACEGROUPS,
    SYMPREC,
    _idealize,
    _structures_from,
)
from prepare_mp import _load_token

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data/raw/mp"
SPLIT_DIR = REPO / "data/splits"
MANIFEST_DIR = REPO / "data/manifests"

PIEZO_NPZ = RAW_DIR / "mp_piezoelectric_processed.npz"
PIEZO_SPLIT = SPLIT_DIR / "mp_piezoelectric_split.npz"
OOD_NPZ = RAW_DIR / "mp_ood_centrosymmetric_processed.npz"
AUG_NPZ = RAW_DIR / "mp_piezoelectric_augmented_processed.npz"
OUT_SETS = REPO / "results/zero_injection_sets.json"

N_VALUES = [250, 4000, 16000]


def _fetch_pool(exclude: set[str], aug_ids: list[str], n_max: int) -> list[dict]:
    """The augmentation set's 1,000 first (reused from the augmentation npz), then fresh
    verified candidates."""
    _, aug_structs = _structures_from(AUG_NPZ)
    by_id = {s["id"]: s for s in aug_structs}
    pool: list[dict] = []
    for mid in aug_ids:  # the augmentation study's exact crystals, already verified + idealized
        s = by_id[mid]
        sg = int(
            spglib.get_symmetry_dataset(
                (s["cell"], (s["positions"] @ np.linalg.inv(s["cell"])) % 1.0, s["z"]),
                symprec=SYMPREC,
            ).number
        )
        pool.append({**s, "sg": sg})
    if n_max <= len(pool):
        return pool[:n_max]

    from mp_api.client import MPRester

    print(f"fetching centrosymmetric insulators from SGs {SEEN_SPACEGROUPS}...")
    with MPRester(_load_token()) as mpr:
        docs = mpr.materials.summary.search(
            spacegroup_number=SEEN_SPACEGROUPS,
            band_gap=(0.1, None),
            fields=["material_id", "structure"],
        )
    print(f"  {len(docs)} candidates; {len(exclude)} excluded (OOD + piezoelectric + augmentation)")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(docs))
    rejected = 0
    taken = set(aug_ids)
    for i in order:
        doc = docs[int(i)]
        mid = str(doc.material_id)
        if mid in exclude or mid in taken:
            continue
        structure = doc.structure
        cell = np.asarray(structure.lattice.matrix, dtype=np.float64)
        frac = np.asarray(structure.frac_coords, dtype=np.float64)
        z = np.asarray(structure.atomic_numbers, dtype=np.int64)
        number = int(spglib.get_symmetry_dataset((cell, frac % 1.0, z), symprec=SYMPREC).number)
        if number not in SEEN_SPACEGROUPS:
            rejected += 1
            continue
        ideal = _idealize(cell, frac, z)
        if ideal is None:
            rejected += 1
            continue
        lattice, positions, numbers = ideal
        pool.append(
            {"id": mid, "cell": lattice, "positions": positions, "z": numbers, "sg": number}
        )
        taken.add(mid)
        if len(pool) >= n_max:
            break
    print(f"  pool size {len(pool)} (target {n_max}); rejected {rejected} spglib mismatches")
    return pool


def _write_set(n: int, augment: list[dict], piezo_targets, piezo_structs, old_split) -> dict:
    name = f"mp_piezoelectric_augmented_n{n}"
    structures = piezo_structs + augment
    zeros = np.zeros((len(augment), piezo_targets.shape[1]), dtype=np.float64)
    targets = np.concatenate([piezo_targets, zeros])
    out_npz = RAW_DIR / f"{name}_processed.npz"
    np.savez_compressed(
        out_npz,
        ids=np.array([s["id"] for s in structures]),
        n_atoms=np.array([len(s["z"]) for s in structures], dtype=np.int64),
        z=np.concatenate([s["z"] for s in structures]).astype(np.int64),
        positions=np.concatenate([s["positions"] for s in structures]).astype(np.float64),
        cells=np.stack([s["cell"] for s in structures]).astype(np.float64),
        piezoelectric=targets,
    )
    aug_ids = sorted(a["id"] for a in augment)
    train = np.concatenate([old_split["train"], np.array(aug_ids)])
    out_split = SPLIT_DIR / f"{name}_split.npz"
    np.savez(out_split, train=train, val=old_split["val"], test=old_split["test"])
    print(f"wrote {out_npz.name} + {out_split.name} ({len(augment)} zero-labelled)")

    (MANIFEST_DIR / f"{name}.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": name,
                "source": "MP piezoelectric (DFPT) + centrosymmetric insulators labelled zero",
                "n_total": len(structures),
                "n_real_tensors": len(piezo_structs),
                "n_zero_labelled": len(augment),
                "nested": "250 subset of 1000 subset of 4000 subset of 16000",
                "augmentation_spacegroups": sorted({a["sg"] for a in augment}),
                "augmentation_symprec": SYMPREC,
                "idealized": "spglib standardize_cell(to_primitive=True, no_idealize=False)",
                "seed": SEED,
                "contamination_checked": "augmentation ids disjoint from OOD 2000 and piezo set",
            },
            sort_keys=False,
        )
    )
    return {"n_requested": n, "n_achieved": len(augment), "ids": aug_ids}


def main() -> None:
    ood_ids = {str(i) for i in np.load(OOD_NPZ, allow_pickle=False)["ids"]}
    piezo_targets, piezo_structs = _structures_from(PIEZO_NPZ)
    piezo_ids = {s["id"] for s in piezo_structs}
    aug_ids = json.loads(AUG_EVAL.read_text())["augmentation_ids"]
    assert len(aug_ids) == AUG_N

    pool = _fetch_pool(exclude=ood_ids | piezo_ids, aug_ids=aug_ids, n_max=max(N_VALUES))
    pool_ids = {p["id"] for p in pool}
    assert not (pool_ids & ood_ids), "augmentation leaks into the OOD evaluation set"
    assert not (pool_ids & piezo_ids), "augmentation duplicates a training crystal"
    assert [p["id"] for p in pool[:AUG_N]] == aug_ids, (
        "nesting broken: the augmentation 1000 must be prefix"
    )

    old_split = np.load(PIEZO_SPLIT, allow_pickle=False)
    sets = {}
    for n in N_VALUES:
        take = pool[: min(n, len(pool))]
        if len(take) < n:
            print(f"NOTE: N={n} capped at availability: {len(take)} candidates")
        sets[f"n{n}"] = _write_set(n, take, piezo_targets, piezo_structs, old_split)

    OUT_SETS.write_text(
        json.dumps(
            {
                "seen_spacegroups": SEEN_SPACEGROUPS,
                "pool_size": len(pool),
                "augmentation_prefix": True,
                "sets": {
                    k: {kk: v[kk] for kk in ("n_requested", "n_achieved")} for k, v in sets.items()
                },
                "ids": {k: v["ids"] for k, v in sets.items()},
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {OUT_SETS}")


if __name__ == "__main__":
    main()
