"""Build the piezoelectric training set augmented with exact-zero centrosymmetric crystals.

The strongest objection to the headline is "just train the SO(3) model on centrosymmetric examples
labelled zero." This builds the dataset that answers it, with a split design that makes the answer
decisive: learned zeros that only work on the symmetry classes seen in training are not a fix.

- **Train** = the existing 2,649 piezoelectric crystals (real tensors) + ~1,000 *fresh*
  centrosymmetric insulators labelled with exact-zero tensors, drawn only from the six most populous
  centrosymmetric space groups (the SEEN list). None of them appear in the OOD 2,000.
- **SEEN-SG eval** = the OOD crystals whose space group is in that list (never trained on).
- **UNSEEN-SG eval** = the OOD crystals from the other 67 space groups (their space groups never
  appear anywhere in training).

Both eval sets are subsets of the headline 2,000, so the augmentation study's numbers are directly
comparable to the main table, with zero contamination. Val and test partitions are inherited
unchanged from the main piezoelectric split, so the non-centrosymmetric test MAE stays comparable
too.

    uv run --extra nequip --extra data python scripts/data/prepare_augmented_piezoelectric.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import spglib
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_mp import _load_token

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data/raw/mp"
SPLIT_DIR = REPO / "data/splits"
MANIFEST_DIR = REPO / "data/manifests"

PIEZO_NPZ = RAW_DIR / "mp_piezoelectric_processed.npz"
PIEZO_SPLIT = SPLIT_DIR / "mp_piezoelectric_split.npz"
OOD_NPZ = RAW_DIR / "mp_ood_centrosymmetric_processed.npz"
SPACEGROUPS = REPO / "results/ood_spacegroups.json"

OUT_NPZ = RAW_DIR / "mp_piezoelectric_augmented_processed.npz"
OUT_SPLIT = SPLIT_DIR / "mp_piezoelectric_augmented_split.npz"
OUT_EVAL = REPO / "results/augmentation_eval_split.json"

SEEN_SPACEGROUPS = [2, 12, 14, 15, 62, 225]
N_AUGMENT = 1000
SYMPREC = 1e-3
SEED = 42


def _idealize(cell: np.ndarray, pos_frac: np.ndarray, z: np.ndarray):
    """Snap onto the exact space group -- identical treatment to the OOD set (idealize_ood.py)."""
    std = spglib.standardize_cell(
        (cell, pos_frac % 1.0, z), to_primitive=True, no_idealize=False, symprec=SYMPREC
    )
    if std is None:
        return None
    lattice, scaled, numbers = std
    return np.asarray(lattice), scaled @ lattice, np.asarray(numbers, dtype=np.int64)


def _fetch_augmentation(exclude: set[str]) -> list[dict]:
    from mp_api.client import MPRester

    print(f"fetching centrosymmetric insulators from SGs {SEEN_SPACEGROUPS}...")
    with MPRester(_load_token()) as mpr:
        docs = mpr.materials.summary.search(
            spacegroup_number=SEEN_SPACEGROUPS,
            band_gap=(0.1, None),
            fields=["material_id", "structure"],
        )
    print(f"  {len(docs)} candidates; {len(exclude)} ids excluded (the OOD set)")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(docs))
    kept, rejected = [], 0
    for i in order:
        doc = docs[int(i)]
        mid = str(doc.material_id)
        if mid in exclude:
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
        kept.append(
            {"id": mid, "cell": lattice, "positions": positions, "z": numbers, "sg": number}
        )
        if len(kept) >= N_AUGMENT:
            break
    print(f"  kept {len(kept)} verified + idealized; rejected {rejected} spglib mismatches")
    return kept


def _structures_from(npz: Path) -> tuple[np.ndarray, list[dict]]:
    data = np.load(npz, allow_pickle=False)
    offsets = np.concatenate([[0], np.cumsum(data["n_atoms"])])
    out = []
    for i in range(len(data["ids"])):
        start, stop = int(offsets[i]), int(offsets[i + 1])
        out.append(
            {
                "id": str(data["ids"][i]),
                "cell": data["cells"][i],
                "positions": data["positions"][start:stop],
                "z": data["z"][start:stop],
            }
        )
    targets = data.get("piezoelectric")
    return targets, out


def main() -> None:
    ood_ids = {str(i) for i in np.load(OOD_NPZ, allow_pickle=False)["ids"]}
    piezo_targets, piezo_structs = _structures_from(PIEZO_NPZ)
    piezo_ids = {s["id"] for s in piezo_structs}

    augment = _fetch_augmentation(exclude=ood_ids | piezo_ids)
    aug_ids = {a["id"] for a in augment}

    # No contamination, at the id level, in either direction.
    assert not (aug_ids & ood_ids), "augmentation leaks into the OOD evaluation set"
    assert not (aug_ids & piezo_ids), "augmentation duplicates a training crystal"

    structures = piezo_structs + augment
    zeros = np.zeros((len(augment), piezo_targets.shape[1]), dtype=np.float64)
    targets = np.concatenate([piezo_targets, zeros])

    np.savez_compressed(
        OUT_NPZ,
        ids=np.array([s["id"] for s in structures]),
        n_atoms=np.array([len(s["z"]) for s in structures], dtype=np.int64),
        z=np.concatenate([s["z"] for s in structures]).astype(np.int64),
        positions=np.concatenate([s["positions"] for s in structures]).astype(np.float64),
        cells=np.stack([s["cell"] for s in structures]).astype(np.float64),
        piezoelectric=targets,
    )
    print(f"wrote {OUT_NPZ} ({len(structures)} structures, {len(augment)} zero-labelled)")

    # Split: train gains the augmentation; val/test inherited unchanged so MAE stays comparable.
    old = np.load(PIEZO_SPLIT, allow_pickle=False)
    train = np.concatenate([old["train"], np.array(sorted(aug_ids))])
    np.savez(OUT_SPLIT, train=train, val=old["val"], test=old["test"])
    print(f"wrote {OUT_SPLIT} (train {len(train)} = {len(old['train'])} + {len(aug_ids)})")
    assert not (set(train.tolist()) & ood_ids), "train leaks into OOD"

    # SEEN / UNSEEN evaluation partition of the untouched OOD 2,000.
    records = json.loads(SPACEGROUPS.read_text())["records"]
    seen = [r["index"] for r in records if r["spacegroup"] in SEEN_SPACEGROUPS]
    unseen = [r["index"] for r in records if r["spacegroup"] not in SEEN_SPACEGROUPS]
    assert len(seen) + len(unseen) == 2000
    assert not (set(seen) & set(unseen))

    trained_sgs = sorted({a["sg"] for a in augment})
    unseen_sgs = sorted({r["spacegroup"] for r in records if r["index"] in set(unseen)})
    assert not (set(trained_sgs) & set(unseen_sgs)), "an UNSEEN space group appears in training"

    OUT_EVAL.write_text(
        json.dumps(
            {
                "seen_spacegroups": trained_sgs,
                "n_seen": len(seen),
                "n_unseen": len(unseen),
                "seen_indices": seen,
                "unseen_indices": unseen,
                "n_augmentation": len(augment),
                "augmentation_ids": sorted(aug_ids),
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {OUT_EVAL}: SEEN {len(seen)} / UNSEEN {len(unseen)}; train SGs {trained_sgs}")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "mp_piezoelectric_augmented.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": "mp_piezoelectric_augmented",
                "source": "MP piezoelectric (DFPT) + centrosymmetric insulators labelled zero",
                "n_total": len(structures),
                "n_real_tensors": len(piezo_structs),
                "n_zero_labelled": len(augment),
                "augmentation_spacegroups": trained_sgs,
                "augmentation_symprec": SYMPREC,
                "idealized": "spglib standardize_cell(to_primitive=True, no_idealize=False)",
                "seed": SEED,
                "contamination_checked": "augmentation ids disjoint from the OOD 2000",
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
