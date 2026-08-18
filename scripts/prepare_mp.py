"""Fetch and process Materials Project datasets: elastic, piezoelectric, and the OOD set.

Writes per-dataset processed archives (not committed) plus committed manifests and splits:
- MP Elastic (~13k, rank-4 6x6 tensor, parity-even control), 80/10/10 split.
- MP Piezoelectric (~3.3k, DFPT 3x6 tensor, parity-odd headline), 80/10/10 split.
- Centrosymmetric OOD (~2k insulators, spglib-verified, tensor exactly zero by symmetry).

Requires MP_TOKEN in .env. Run: ``uv run python scripts/prepare_mp.py [elastic|piezo|ood|all]``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import yaml

from equiparity.domain.data_manifest import DatasetManifest, SplitManifest
from equiparity.domain.sample import LabeledStructure
from equiparity.domain.spacegroup import CENTROSYMMETRIC_SPACE_GROUPS, is_centrosymmetric
from equiparity.io.materials_project import pymatgen_to_structure, space_group_number, tensor_sample

RAW_DIR = Path("data/raw/mp")
MANIFEST_DIR = Path("data/manifests")
SPLIT_DIR = Path("data/splits")
SPLIT_SEED = 42
OOD_MAX = 2000
OOD_SYMPREC = 1e-3
# Domain sanity bounds: MP holds some failed-DFT tensors with
# physically impossible magnitudes. Exclude and report them; never silently keep them.
MAX_ELASTIC_ABS_GPA = 2000.0  # hardest materials (diamond) reach ~1000 GPa
MAX_PIEZO_ABS = 50.0  # strong piezoelectrics (PZT) reach ~25 C/m^2


def _filter_by_magnitude(samples: list, target_key: str, bound: float) -> tuple[list, int]:
    """Drop samples whose target exceeds ``bound`` in absolute value; return (kept, n_dropped)."""
    kept = [s for s in samples if float(np.abs(s.targets[target_key]).max()) <= bound]
    return kept, len(samples) - len(kept)


def _load_token() -> str:
    for line in Path(".env").read_text().splitlines():
        m = re.match(r"\s*MP_TOKEN\s*=\s*(.+)", line)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    raise RuntimeError("MP_TOKEN not found in .env")


def _save_crystal_dataset(
    name: str, structures: list, target_key: str | None, targets: list
) -> int:
    """Save concatenated crystal geometry (+ optional flat target) to a processed npz."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ids = [s.identifier for s in structures]
    n_atoms = np.array([s.structure.n_atoms for s in structures], dtype=np.int64)
    z = np.concatenate([s.structure.atomic_numbers for s in structures]).astype(np.int64)
    pos = np.concatenate([s.structure.positions for s in structures]).astype(np.float64)
    cells = np.stack([s.structure.cell for s in structures]).astype(np.float64)
    payload: dict[str, np.ndarray] = {
        "ids": np.array(ids),
        "n_atoms": n_atoms,
        "z": z,
        "positions": pos,
        "cells": cells,
    }
    if target_key is not None:
        payload[target_key] = np.stack(targets).astype(np.float64)
    np.savez_compressed(RAW_DIR / f"{name}_processed.npz", **payload)
    print(f"  wrote {RAW_DIR / f'{name}_processed.npz'} ({len(structures)} structures)")
    return len(structures)


def _write_split(name: str, dataset: str, ids: list[str]) -> None:
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(ids))
    n_train, n_val = int(0.8 * len(ids)), int(0.1 * len(ids))
    parts = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }
    id_arr = np.array(ids)
    np.savez(SPLIT_DIR / f"{name}_split.npz", **{k: id_arr[v] for k, v in parts.items()})
    manifest = SplitManifest(
        split_id=f"{name}_random_seed{SPLIT_SEED}",
        dataset=dataset,
        method="random",
        seed=SPLIT_SEED,
        counts={k: len(v) for k, v in parts.items()},
    )
    (SPLIT_DIR / f"{name}.yaml").write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False))
    print(f"  wrote {name} split: {manifest.counts}")


def _processed_hashes(name: str) -> dict[str, str]:
    """SHA-256 of every processed archive this dataset writes.

    The manuscript's data availability statement promises "dataset manifests with content
    hashes", and DatasetManifest documents file_hashes as the digests loaders verify. The
    processed archives are not committed, so the digest is the only way a reader who
    re-derives them can confirm they reproduced the same bytes. Includes the idealized and
    raw variants of the centrosymmetric population when present.
    """
    import hashlib

    out: dict[str, str] = {}
    for path in sorted(RAW_DIR.glob(f"{name}_processed*.npz")):
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[path.name] = h.hexdigest()
    return out


def _write_manifest(
    name: str, source: str, schema: dict[str, str], n: int, extra: str = ""
) -> None:
    manifest = DatasetManifest(
        name=name,
        source=source,
        version="Materials Project API; fetched 2026-07-04",
        license="CC-BY-4.0 (Materials Project)",
        file_hashes=_processed_hashes(name),
        schema=schema,
        structure_format="pymatgen Structure (periodic)",
        query="scripts/prepare_mp.py",
        cleaning=f"{n} entries retained. {extra}",
        limitations="MP DFPT tensors; provenance is the MP material_id + API fetch date.",
    )
    (MANIFEST_DIR / f"{name}.yaml").write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False))
    print(f"  wrote {MANIFEST_DIR / f'{name}.yaml'}")


def _fetch_structures(mpr, ids: list[str], chunk: int = 2000) -> dict:
    """Fetch structures for many material ids, chunked (MP rejects overly long id filters)."""
    out: dict = {}
    for start in range(0, len(ids), chunk):
        batch = ids[start : start + chunk]
        for s in mpr.materials.summary.search(
            material_ids=batch, fields=["material_id", "structure"]
        ):
            out[str(s.material_id)] = s.structure
        print(f"  fetched structures {min(start + chunk, len(ids))}/{len(ids)}")
    return out


def prepare_piezo(mpr) -> None:
    print("fetching piezoelectric...")
    docs = mpr.materials.piezoelectric.search(fields=["material_id", "total"])
    tensors = {str(d.material_id): np.asarray(d.total, dtype=np.float64) for d in docs}
    structures = _fetch_structures(mpr, list(tensors))
    samples = [
        tensor_sample(struct, tensors[mid], "piezoelectric", mid)
        for mid, struct in structures.items()
    ]
    samples, dropped = _filter_by_magnitude(samples, "piezoelectric", MAX_PIEZO_ABS)
    print(f"  excluded {dropped} with |e| > {MAX_PIEZO_ABS} C/m^2")
    n = _save_crystal_dataset(
        "mp_piezoelectric", samples, "piezoelectric", [s.targets["piezoelectric"] for s in samples]
    )
    _write_manifest(
        "mp_piezoelectric",
        "Materials Project materials.piezoelectric",
        {
            "piezoelectric": "DFPT piezoelectric tensor, 3x6 flattened to 18 (parity-odd)",
            "positions": "angstrom",
        },
        n,
        f"Non-centrosymmetric; excluded {dropped} with |e| > {MAX_PIEZO_ABS} C/m^2.",
    )
    _write_split("mp_piezoelectric", "mp_piezoelectric", [s.identifier for s in samples])


def prepare_elastic(mpr) -> None:
    print("fetching elastic...")
    docs = mpr.materials.elasticity.search(fields=["material_id", "elastic_tensor"])
    tensors = {
        str(d.material_id): np.asarray(d.elastic_tensor.ieee_format, dtype=np.float64)
        for d in docs
        if d.elastic_tensor is not None
    }
    structures = _fetch_structures(mpr, list(tensors))
    samples = [
        tensor_sample(struct, tensors[mid], "elastic", mid) for mid, struct in structures.items()
    ]
    samples, dropped = _filter_by_magnitude(samples, "elastic", MAX_ELASTIC_ABS_GPA)
    print(f"  excluded {dropped} with |C| > {MAX_ELASTIC_ABS_GPA} GPa (failed-DFT tensors)")
    n = _save_crystal_dataset(
        "mp_elastic", samples, "elastic", [s.targets["elastic"] for s in samples]
    )
    _write_manifest(
        "mp_elastic",
        "Materials Project materials.elasticity",
        {
            "elastic": "elastic tensor ieee_format 6x6 flattened to 36 (21 unique, parity-even)",
            "positions": "angstrom",
        },
        n,
        f"Excluded {dropped} with |C| > {MAX_ELASTIC_ABS_GPA} GPa (unphysical failed-DFT tensors).",
    )
    _write_split("mp_elastic", "mp_elastic", [s.identifier for s in samples])


def prepare_ood(mpr) -> None:
    print("fetching centrosymmetric OOD insulators...")
    docs = mpr.materials.summary.search(
        spacegroup_number=sorted(CENTROSYMMETRIC_SPACE_GROUPS),
        band_gap=(0.1, None),
        fields=["material_id", "structure", "symmetry"],
    )
    print(f"  {len(docs)} candidates; verifying with spglib...")
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(docs))
    verified, leaks = [], 0
    for i in order:
        d = docs[int(i)]
        sg = space_group_number(d.structure, symprec=OOD_SYMPREC)
        if not is_centrosymmetric(sg):
            leaks += 1
            continue
        verified.append(
            LabeledStructure(
                structure=pymatgen_to_structure(d.structure),
                targets={},
                identifier=str(d.material_id),
            )
        )
        if len(verified) >= OOD_MAX:
            break
    print(f"  verified {len(verified)} centrosymmetric; rejected {leaks} spglib mismatches")
    n = _save_crystal_dataset("mp_ood_centrosymmetric", verified, None, [])
    # NOTE: raw MP coords deviate from perfect inversion by up to ~symprec, which an O(3)-exact
    # model with a sensitive head reports as a spurious nonzero piezo response. Idealize the saved
    # geometries onto their exact space group before use: `python scripts/idealize_ood.py`.
    _write_manifest(
        "mp_ood_centrosymmetric",
        "Materials Project materials.summary (centrosymmetric space groups, band_gap>0.1 eV)",
        {
            "positions": "angstrom",
            "target": "none — piezoelectric tensor is exactly zero by symmetry",
        },
        n,
        f"spglib-verified centrosymmetric (symprec={OOD_SYMPREC}); {leaks} candidates rejected.",
    )


def main(argv: list[str]) -> None:
    from mp_api.client import MPRester

    which = argv[1] if len(argv) > 1 else "all"
    with MPRester(_load_token()) as mpr:
        if which in ("piezo", "all"):
            prepare_piezo(mpr)
        if which in ("elastic", "all"):
            prepare_elastic(mpr)
        if which in ("ood", "all"):
            prepare_ood(mpr)


if __name__ == "__main__":
    main(sys.argv)
