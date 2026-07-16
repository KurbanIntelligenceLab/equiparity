"""Process raw QM9 into a validated dataset with committed manifest and split.

Reads the extracted ``dsgdb9nsd`` .xyz files, excludes the 3,054 uncharacterized molecules,
parses each into a validated record, and writes:

- ``data/raw/qm9/qm9_processed.npz`` (not committed): concatenated geometry, charges, and targets.
- ``data/manifests/qm9.yaml`` (committed): dataset provenance.
- ``data/splits/qm9.yaml`` + ``data/splits/qm9_split.npz`` (committed): the 110k/10k/10831 split.

Run: ``uv run python scripts/prepare_qm9.py``
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml

from equiparity.domain.data_manifest import DatasetManifest, SplitManifest
from equiparity.io.qm9 import parse_qm9_xyz

RAW_DIR = Path("data/raw/qm9")
XYZ_DIR = RAW_DIR / "xyz"
ARCHIVE = RAW_DIR / "dsgdb9nsd.xyz.tar.bz2"
ARCHIVE_SHA256 = "3a63848ac80691bdb8d41834b575afad345b9300d7a2db0c38adb7f6eaa8360c"
UNCHARACTERIZED = RAW_DIR / "uncharacterized.txt"
PROCESSED = RAW_DIR / "qm9_processed.npz"

MANIFEST_PATH = Path("data/manifests/qm9.yaml")
SPLIT_YAML = Path("data/splits/qm9.yaml")
SPLIT_NPZ = Path("data/splits/qm9_split.npz")

SPLIT_SEED = 42
N_TRAIN, N_VAL = 110_000, 10_000


def _excluded_indices() -> set[int]:
    text = UNCHARACTERIZED.read_text()
    return {int(m.group(1)) for line in text.splitlines() if (m := re.match(r"\s*(\d+)", line))}


def _process() -> dict[str, np.ndarray]:
    excluded = _excluded_indices()
    print(f"excluding {len(excluded)} uncharacterized molecules")
    files = sorted(XYZ_DIR.glob("dsgdb9nsd_*.xyz"))

    ids, n_atoms, u0, dipole = [], [], [], []
    z_all, pos_all = [], []
    kept = 0
    for path in files:
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
        kept += 1
        if kept % 20000 == 0:
            print(f"  parsed {kept} molecules")
    print(f"kept {kept} molecules")
    return {
        "ids": np.array(ids, dtype=np.int64),
        "n_atoms": np.array(n_atoms, dtype=np.int64),
        "U0": np.array(u0, dtype=np.float64),
        "dipole": np.stack(dipole).astype(np.float64),
        "z": np.concatenate(z_all).astype(np.int64),
        "positions": np.concatenate(pos_all).astype(np.float64),
    }


def _write_dataset_manifest(n_kept: int) -> None:
    manifest = DatasetManifest(
        name="QM9",
        source="https://ndownloader.figshare.com/files/3195389 (dsgdb9nsd, figshare 978904)",
        version="2014 release; downloaded 2026-07-04",
        license="CC0 (public domain)",
        file_hashes={"dsgdb9nsd.xyz.tar.bz2": ARCHIVE_SHA256},
        schema={
            "U0": "internal energy at 0 K, converted Hartree -> eV (parity-even scalar)",
            "dipole": "sum q_mulliken * r, converted to Debye (parity-odd vector target)",
            "positions": "angstrom",
        },
        structure_format="xyz (non-periodic molecules, elements H C N O F)",
        query="scripts/prepare_qm9.py; exclude 3054 uncharacterized (figshare 3195404)",
        cleaning=f"kept {n_kept} of 133885 molecules after uncharacterized exclusion",
        limitations=(
            "Dipole target from Mulliken charges underestimates the reference DFT |mu| by "
            "~10-25%; direction and parity are exact, which is what the parity experiment tests."
        ),
    )
    MANIFEST_PATH.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False))
    print(f"wrote {MANIFEST_PATH}")


def _write_split(data: dict[str, np.ndarray]) -> None:
    ids = data["ids"]
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(ids))
    train, val, test = perm[:N_TRAIN], perm[N_TRAIN : N_TRAIN + N_VAL], perm[N_TRAIN + N_VAL :]
    partitions = {"train": ids[train], "val": ids[val], "test": ids[test]}
    np.savez(SPLIT_NPZ, **partitions)

    u0 = data["U0"]
    distribution = {
        name: {"U0_mean": float(u0[idx].mean()), "U0_std": float(u0[idx].std())}
        for name, idx in {"train": train, "val": val, "test": test}.items()
    }
    manifest = SplitManifest(
        split_id=f"qm9_random_seed{SPLIT_SEED}",
        dataset="QM9",
        method="random",
        seed=SPLIT_SEED,
        counts={name: len(idx) for name, idx in partitions.items()},
        target_distribution=distribution,
    )
    SPLIT_YAML.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False))
    print(f"wrote {SPLIT_YAML} and {SPLIT_NPZ}: {manifest.counts}")


def main() -> None:
    data = _process()
    np.savez_compressed(PROCESSED, **data)
    print(f"wrote {PROCESSED}")
    _write_dataset_manifest(len(data["ids"]))
    _write_split(data)


if __name__ == "__main__":
    main()
