"""Prepare the public datasets: QM9 and the Materials Project elastic, piezoelectric and
centrosymmetric evaluation sets.

Writes processed archives under ``data/raw/`` plus the committed dataset manifests
(``data/manifests/``) and split definitions (``data/splits/``). All paths are resolved against a
repository ``root`` so the CLI can run from any working directory.

CLI: ``equiparity data prepare {qm9,mp,idealize}``
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from equiparity.domain.data_manifest import DatasetManifest, SplitManifest
from equiparity.domain.sample import LabeledStructure

_log = logging.getLogger(__name__)

SPLIT_SEED = 42

# QM9
QM9_ARCHIVE_SHA256 = "3a63848ac80691bdb8d41834b575afad345b9300d7a2db0c38adb7f6eaa8360c"
QM9_N_TRAIN, QM9_N_VAL = 110_000, 10_000

# Materials Project
MP_DATASETS = ("piezo", "elastic", "ood")
OOD_MAX = 2000
OOD_SYMPREC = 1e-3
# Domain sanity bounds: MP holds some failed-DFT tensors with physically impossible magnitudes.
# Exclude and report them; never silently keep them.
MAX_ELASTIC_ABS_GPA = 2000.0  # hardest materials (diamond) reach ~1000 GPa
MAX_PIEZO_ABS = 50.0  # strong piezoelectrics (PZT) reach ~25 C/m^2


# --------------------------------------------------------------------------------------------
# QM9
# --------------------------------------------------------------------------------------------


def prepare_qm9(root: Path) -> None:
    """Parse extracted ``dsgdb9nsd`` .xyz files into the processed archive, manifest and split."""
    from equiparity.io.qm9 import parse_qm9_xyz

    raw_dir = root / "data/raw/qm9"
    excluded = {
        int(m.group(1))
        for line in (raw_dir / "uncharacterized.txt").read_text().splitlines()
        if (m := re.match(r"\s*(\d+)", line))
    }
    _log.info("excluding %d uncharacterized molecules", len(excluded))

    ids, n_atoms, u0, dipole = [], [], [], []
    z_all, pos_all = [], []
    for path in sorted((raw_dir / "xyz").glob("dsgdb9nsd_*.xyz")):
        index = int(path.stem.split("_")[1])
        if index in excluded:
            continue
        record = parse_qm9_xyz(path.read_text())
        s = record.sample.structure
        ids.append(index)
        n_atoms.append(s.n_atoms)
        u0.append(record.sample.targets["U0"][0])
        dipole.append(record.sample.targets["dipole"])
        z_all.append(s.atomic_numbers)
        pos_all.append(s.positions)
        if len(ids) % 20000 == 0:
            _log.info("parsed %d molecules", len(ids))
    if not ids:
        raise FileNotFoundError(f"no dsgdb9nsd_*.xyz files under {raw_dir / 'xyz'}")

    data = {
        "ids": np.array(ids, dtype=np.int64),
        "n_atoms": np.array(n_atoms, dtype=np.int64),
        "U0": np.array(u0, dtype=np.float64),
        "dipole": np.stack(dipole).astype(np.float64),
        "z": np.concatenate(z_all).astype(np.int64),
        "positions": np.concatenate(pos_all).astype(np.float64),
    }
    processed = raw_dir / "qm9_processed.npz"
    np.savez_compressed(processed, **data)
    _log.info("wrote %s (%d molecules)", processed, len(ids))

    manifest = DatasetManifest(
        name="QM9",
        source="https://ndownloader.figshare.com/files/3195389 (dsgdb9nsd, figshare 978904)",
        version="2014 release; downloaded 2026-07-04",
        license="CC0 (public domain)",
        file_hashes={"dsgdb9nsd.xyz.tar.bz2": QM9_ARCHIVE_SHA256},
        schema={
            "U0": "internal energy at 0 K, converted Hartree -> eV (parity-even scalar)",
            "dipole": "sum q_mulliken * r, converted to Debye (parity-odd vector target)",
            "positions": "angstrom",
        },
        structure_format="xyz (non-periodic molecules, elements H C N O F)",
        query="equiparity data prepare qm9; exclude 3054 uncharacterized (figshare 3195404)",
        cleaning=f"kept {len(ids)} of 133885 molecules after uncharacterized exclusion",
        limitations=(
            "Dipole target from Mulliken charges underestimates the reference DFT |mu| by "
            "~10-25%; direction and parity are exact, which is what the parity experiment tests."
        ),
    )
    manifest_path = root / "data/manifests/qm9.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False))
    _log.info("wrote %s", manifest_path)

    id_arr = data["ids"]
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(id_arr))
    idx = {
        "train": perm[:QM9_N_TRAIN],
        "val": perm[QM9_N_TRAIN : QM9_N_TRAIN + QM9_N_VAL],
        "test": perm[QM9_N_TRAIN + QM9_N_VAL :],
    }
    np.savez(root / "data/splits/qm9_split.npz", **{k: id_arr[v] for k, v in idx.items()})
    split = SplitManifest(
        split_id=f"qm9_random_seed{SPLIT_SEED}",
        dataset="QM9",
        method="random",
        seed=SPLIT_SEED,
        counts={k: len(v) for k, v in idx.items()},
        target_distribution={
            k: {"U0_mean": float(data["U0"][v].mean()), "U0_std": float(data["U0"][v].std())}
            for k, v in idx.items()
        },
    )
    (root / "data/splits/qm9.yaml").write_text(yaml.safe_dump(split.to_dict(), sort_keys=False))
    _log.info("wrote qm9 split: %s", split.counts)


# --------------------------------------------------------------------------------------------
# Materials Project
# --------------------------------------------------------------------------------------------


def _mp_token(root: Path) -> str:
    """Read ``MP_TOKEN`` from the environment, falling back to ``<root>/.env``."""
    if token := os.environ.get("MP_TOKEN"):
        return token
    env = root / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if m := re.match(r"\s*MP_TOKEN\s*=\s*(.+)", line):
                return m.group(1).strip().strip('"').strip("'")
    raise RuntimeError("MP_TOKEN is not set (export it or add it to .env)")


def _filter_by_magnitude(
    samples: list[LabeledStructure], target_key: str, bound: float
) -> tuple[list[LabeledStructure], int]:
    kept = [s for s in samples if float(np.abs(s.targets[target_key]).max()) <= bound]
    return kept, len(samples) - len(kept)


def _save_crystal_dataset(
    root: Path, name: str, samples: list[LabeledStructure], target_key: str | None
) -> int:
    raw_dir = root / "data/raw/mp"
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "ids": np.array([s.identifier for s in samples]),
        "n_atoms": np.array([s.structure.n_atoms for s in samples], dtype=np.int64),
        "z": np.concatenate([s.structure.atomic_numbers for s in samples]).astype(np.int64),
        "positions": np.concatenate([s.structure.positions for s in samples]).astype(np.float64),
        "cells": np.stack([s.structure.cell for s in samples]).astype(np.float64),
    }
    if target_key is not None:
        payload[target_key] = np.stack([s.targets[target_key] for s in samples]).astype(np.float64)
    out = raw_dir / f"{name}_processed.npz"
    np.savez_compressed(out, **payload)
    _log.info("wrote %s (%d structures)", out, len(samples))
    return len(samples)


def _write_split(root: Path, name: str, ids: list[str]) -> None:
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(ids))
    n_train, n_val = int(0.8 * len(ids)), int(0.1 * len(ids))
    parts = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }
    id_arr = np.array(ids)
    split_dir = root / "data/splits"
    np.savez(split_dir / f"{name}_split.npz", **{k: id_arr[v] for k, v in parts.items()})
    manifest = SplitManifest(
        split_id=f"{name}_random_seed{SPLIT_SEED}",
        dataset=name,
        method="random",
        seed=SPLIT_SEED,
        counts={k: len(v) for k, v in parts.items()},
    )
    (split_dir / f"{name}.yaml").write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False))
    _log.info("wrote %s split: %s", name, manifest.counts)


def sha256_file(path: Path) -> str:
    """Streamed SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(
    root: Path, name: str, source: str, schema: dict[str, str], n: int, cleaning: str
) -> None:
    raw_dir = root / "data/raw/mp"
    manifest = DatasetManifest(
        name=name,
        source=source,
        version="Materials Project API; fetched 2026-07-04",
        license="CC-BY-4.0 (Materials Project)",
        file_hashes={
            p.name: sha256_file(p) for p in sorted(raw_dir.glob(f"{name}_processed*.npz"))
        },
        schema=schema,
        structure_format="pymatgen Structure (periodic)",
        query="equiparity data prepare mp",
        cleaning=f"{n} entries retained. {cleaning}",
        limitations="MP DFPT tensors; provenance is the MP material_id + API fetch date.",
    )
    path = root / "data/manifests" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False))
    _log.info("wrote %s", path)


def _fetch_structures(mpr: Any, ids: list[str], chunk: int = 2000) -> dict[str, Any]:  # noqa: ANN401
    """Fetch structures for many material ids, chunked (MP rejects overly long id filters)."""
    out: dict[str, Any] = {}
    for start in range(0, len(ids), chunk):
        for s in mpr.materials.summary.search(
            material_ids=ids[start : start + chunk], fields=["material_id", "structure"]
        ):
            out[str(s.material_id)] = s.structure
        _log.info("fetched structures %d/%d", min(start + chunk, len(ids)), len(ids))
    return out


def _prepare_piezo(root: Path, mpr: Any) -> None:  # noqa: ANN401
    from equiparity.io.materials_project import tensor_sample

    docs = mpr.materials.piezoelectric.search(fields=["material_id", "total"])
    tensors = {str(d.material_id): np.asarray(d.total, dtype=np.float64) for d in docs}
    structures = _fetch_structures(mpr, list(tensors))
    samples = [
        tensor_sample(st, tensors[mid], "piezoelectric", mid) for mid, st in structures.items()
    ]
    samples, dropped = _filter_by_magnitude(samples, "piezoelectric", MAX_PIEZO_ABS)
    name = "mp_piezoelectric"
    n = _save_crystal_dataset(root, name, samples, "piezoelectric")
    _write_manifest(
        root,
        name,
        "Materials Project materials.piezoelectric",
        {
            "piezoelectric": "DFPT piezoelectric tensor, 3x6 flattened to 18 (parity-odd)",
            "positions": "angstrom",
        },
        n,
        f"Non-centrosymmetric; excluded {dropped} with |e| > {MAX_PIEZO_ABS} C/m^2.",
    )
    _write_split(root, name, [s.identifier for s in samples])


def _prepare_elastic(root: Path, mpr: Any) -> None:  # noqa: ANN401
    from equiparity.io.materials_project import tensor_sample

    docs = mpr.materials.elasticity.search(fields=["material_id", "elastic_tensor"])
    tensors = {
        str(d.material_id): np.asarray(d.elastic_tensor.ieee_format, dtype=np.float64)
        for d in docs
        if d.elastic_tensor is not None
    }
    structures = _fetch_structures(mpr, list(tensors))
    samples = [tensor_sample(st, tensors[mid], "elastic", mid) for mid, st in structures.items()]
    samples, dropped = _filter_by_magnitude(samples, "elastic", MAX_ELASTIC_ABS_GPA)
    name = "mp_elastic"
    n = _save_crystal_dataset(root, name, samples, "elastic")
    _write_manifest(
        root,
        name,
        "Materials Project materials.elasticity",
        {
            "elastic": "elastic tensor ieee_format 6x6 flattened to 36 (21 unique, parity-even)",
            "positions": "angstrom",
        },
        n,
        f"Excluded {dropped} with |C| > {MAX_ELASTIC_ABS_GPA} GPa (unphysical failed-DFT tensors).",
    )
    _write_split(root, name, [s.identifier for s in samples])


def _prepare_ood(root: Path, mpr: Any) -> None:  # noqa: ANN401
    from equiparity.domain.spacegroup import CENTROSYMMETRIC_SPACE_GROUPS, is_centrosymmetric
    from equiparity.io.materials_project import pymatgen_to_structure, space_group_number

    docs = mpr.materials.summary.search(
        spacegroup_number=sorted(CENTROSYMMETRIC_SPACE_GROUPS),
        band_gap=(0.1, None),
        fields=["material_id", "structure", "symmetry"],
    )
    _log.info("%d candidates; verifying with spglib", len(docs))
    verified: list[LabeledStructure] = []
    leaks = 0
    for i in np.random.default_rng(SPLIT_SEED).permutation(len(docs)):
        d = docs[int(i)]
        if not is_centrosymmetric(space_group_number(d.structure, symprec=OOD_SYMPREC)):
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
    name = "mp_ood_centrosymmetric"
    n = _save_crystal_dataset(root, name, verified, None)
    # Raw MP coordinates deviate from exact inversion by up to ~symprec; `idealize_ood` snaps them
    # onto the detected space group and keeps the raw variant alongside.
    idealize_ood(root)
    _write_manifest(
        root,
        name,
        "Materials Project materials.summary (centrosymmetric space groups, band_gap>0.1 eV)",
        {
            "positions": "angstrom",
            "target": "none - piezoelectric tensor is exactly zero by symmetry",
        },
        n,
        f"spglib-verified centrosymmetric (symprec={OOD_SYMPREC}); {leaks} candidates rejected.",
    )


def prepare_mp(root: Path, which: str = "all") -> None:
    """Fetch and process the Materials Project datasets (requires ``MP_TOKEN``)."""
    if which not in (*MP_DATASETS, "all"):
        raise ValueError(f"unknown MP dataset {which!r}; choose from {(*MP_DATASETS, 'all')}")
    from mp_api.client import MPRester

    steps = {"piezo": _prepare_piezo, "elastic": _prepare_elastic, "ood": _prepare_ood}
    with MPRester(_mp_token(root)) as mpr:
        for key, step in steps.items():
            if which in (key, "all"):
                _log.info("preparing %s", key)
                step(root, mpr)


# --------------------------------------------------------------------------------------------
# Centrosymmetric population idealization
# --------------------------------------------------------------------------------------------


def idealize_ood(root: Path) -> tuple[int, int]:
    """Snap the centrosymmetric evaluation crystals onto their exact space group, in place.

    The raw coordinates are backed up once to ``*_processed_raw.npz``. Returns
    ``(n_structures, n_kept_raw)`` where the second count is structures spglib could not refine.
    """
    import spglib

    npz = root / "data/raw/mp/mp_ood_centrosymmetric_processed.npz"
    backup = npz.with_name("mp_ood_centrosymmetric_processed_raw.npz")
    if not backup.exists():
        shutil.copyfile(npz, backup)
    d = np.load(backup, allow_pickle=True)
    ids, natoms, z, pos, cells = d["ids"], d["n_atoms"], d["z"], d["positions"], d["cells"]
    off = np.concatenate([[0], np.cumsum(natoms)])

    out_n, out_z, out_pos, out_cells = [], [], [], []
    failed = 0
    for i in range(len(ids)):
        zi, pi, cell = z[off[i] : off[i + 1]], pos[off[i] : off[i + 1]], cells[i]
        frac = (pi @ np.linalg.inv(cell)) % 1.0
        std = spglib.standardize_cell(
            (cell, frac, zi), to_primitive=True, no_idealize=False, symprec=OOD_SYMPREC
        )
        if std is None:
            lat, newpos, newz = cell, pi, zi
            failed += 1
        else:
            lat, scaled, nums = std
            lat, newpos, newz = lat, scaled @ lat, np.asarray(nums, dtype=np.int64)
        out_n.append(len(newz))
        out_z.append(newz)
        out_pos.append(newpos)
        out_cells.append(lat)

    np.savez_compressed(
        npz,
        ids=ids,
        n_atoms=np.array(out_n, dtype=np.int64),
        z=np.concatenate(out_z).astype(np.int64),
        positions=np.concatenate(out_pos).astype(np.float64),
        cells=np.stack(out_cells).astype(np.float64),
    )
    _log.info("idealized %d structures; %d kept raw (refine failed)", len(ids), failed)
    return len(ids), failed
