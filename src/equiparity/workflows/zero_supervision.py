"""Prepare disjoint, symmetry-verified zero supervision for the crystal model."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from equiparity.domain.sample import LabeledStructure
from equiparity.domain.structure import AtomicStructure
from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset, load_split
from equiparity.workflows.prepare_data import _mp_token, _save_crystal_dataset, sha256_file
from equiparity.workflows.tensor_datasets import _search_cached, validate_tensor_dataset


def choose_zero_ids(root: Path) -> list[str]:
    """Load the ordered augmentation cohort and verify its independence."""
    base = root / "data/raw/mp/mp_piezoelectric_ieee_processed.npz"
    with np.load(base, allow_pickle=False) as d:
        original = set(d["ids"].tolist())
    with np.load(
        root / "data/raw/mp/mp_ood_centrosymmetric_processed.npz", allow_pickle=False
    ) as d:
        evaluation = set(d["ids"].tolist())
    selected = json.loads((root / "data/splits/zero_label_ids.json").read_text())
    if (
        len(selected) != 1000
        or len(set(selected)) != 1000
        or set(selected) & (original | evaluation)
    ):
        raise ValueError("expected 1000 distinct and disjoint zero-label materials")
    return selected


def prepare_zero_supervision(root: Path, documents: Path | None = None) -> dict[str, Any]:
    """Retain all original labels/splits and add 1000 exact centrosymmetric training inputs."""
    import spglib
    from emmet.core.mpid import AlphaID
    from mp_api.client import MPRester
    from pymatgen.core import Structure

    base_path = root / "data/raw/mp/mp_piezoelectric_ieee_processed.npz"
    base_split = root / "data/splits/mp_piezoelectric_split.npz"
    base_audit = validate_tensor_dataset(base_path, base_split, "piezoelectric")
    ids = choose_zero_ids(root)
    if documents:
        docs = json.loads(
            gzip.decompress(documents.read_bytes())
            if documents.suffix == ".gz"
            else documents.read_bytes()
        )
    else:
        with MPRester(
            _mp_token(root), use_document_model=False, mute_progress_bars=True, timeout=45
        ) as mpr:
            docs = _search_cached(
                root,
                mpr.materials.summary,
                "zero_label_structures",
                ids,
                ["material_id", "structure"],
                "material_ids",
            )
    mapping = {int(AlphaID(d["material_id"])): d for d in docs}
    if len(mapping) != len(ids):
        raise ValueError("Missing or duplicate zero-label source documents")
    base = CrystalDataset(load_crystal_dataset(base_path, ("piezoelectric",)))
    samples = [base[i] for i in range(len(base))]
    audits = []
    for mid in ids:
        structure = Structure.from_dict(mapping[int(AlphaID(mid))]["structure"])
        original_cell = (structure.lattice.matrix, structure.frac_coords, structure.atomic_numbers)
        # No regression target is used in idealization or sample selection.
        standardized = spglib.standardize_cell(
            original_cell, to_primitive=True, no_idealize=False, symprec=1e-3
        )
        if standardized is None:
            raise ValueError(f"{mid}: symmetry standardization failed")
        cell, fractional, numbers = standardized
        symmetry = spglib.get_symmetry((cell, fractional, numbers), symprec=1e-6)
        if symmetry is None or not any(
            np.array_equal(r, -np.eye(3)) for r in symmetry["rotations"]
        ):
            raise ValueError(f"{mid}: refined zero-label structure lacks inversion")
        sample = LabeledStructure(
            AtomicStructure(
                np.asarray(numbers, dtype=np.int64),
                np.asarray(fractional @ cell),
                np.asarray(cell),
                True,
            ),
            {"piezoelectric": np.zeros(18)},
            mid,
        )
        samples.append(sample)
        audits.append(
            {
                "material_id": mid,
                "n_operations": len(symmetry["rotations"]),
                "inversion_verified": True,
            }
        )
    name = "mp_piezoelectric_ieee_augmented"
    _save_crystal_dataset(root, name, samples, "piezoelectric")
    path = root / f"data/raw/mp/{name}_processed.npz"
    split = root / f"data/splits/{name}_split.npz"
    np.savez_compressed(
        split,
        train=np.concatenate([load_split(base_split, "train"), np.asarray(ids)]),
        val=load_split(base_split, "val"),
        test=load_split(base_split, "test"),
    )
    source = root / "data/property_sources/zero_labels.json.gz"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(
        gzip.compress((json.dumps(docs, sort_keys=True, indent=2) + "\n").encode(), mtime=0)
    )
    parents = [
        base_path,
        base_path.with_suffix(".frame.json"),
        base_path.parent / base_audit["source_file"],
    ]
    report = {
        "protocol": "validation_selected",
        "target": "piezoelectric",
        "passed": True,
        "n": len(samples),
        "n_added_zeros": len(ids),
        "zero_ids": ids,
        "original_nonzero_samples_unchanged": True,
        "data_sha256": sha256_file(path),
        "split_sha256": sha256_file(split),
        "source_file": os.path.relpath(source, path.parent),
        "source_sha256": sha256_file(source),
        "parents": [
            {"path": os.path.relpath(p, path.parent), "sha256": sha256_file(p)} for p in parents
        ],
        "zero_audit": audits,
    }
    path.with_suffix(".frame.json").write_text(json.dumps(report, indent=2) + "\n")
    validate_tensor_dataset(path, split, "piezoelectric")
    return {
        "path": str(path),
        "n": len(samples),
        "n_added": len(ids),
        "passed": True,
    }
