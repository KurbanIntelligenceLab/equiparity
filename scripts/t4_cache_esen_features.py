"""T4-1: cache frozen pretrained eSEN per-atom features for the piezo splits + OOD set.

Standalone by design: runs INSIDE the isolated fairchem-1.10 venv and imports nothing
from `equiparity` (targets in the irreps basis are read from the already-cached
``results/t4/mace_mp0_<set>.npz`` files, which the main env wrote):

    HF_TOKEN=<token> third_party/venvs/fairchem1/bin/python scripts/t4_cache_esen_features.py

CHECKPOINT ACCESS IS GATED. Every released eSEN checkpoint (esen_30m_{oam,omat,mptrj}.pt,
fairchem registry names eSEN-30M-{OAM,OMAT24,MP}) lives in the HuggingFace repo
``facebook/OMAT24`` (mirror ``fairchem/OMAT24``), which requires accepting the FAIR
Chemistry License with a logged-in HF account. Anonymous download returns 401. Steps:
  1. Create/log into a huggingface.co account, open https://huggingface.co/facebook/OMAT24
     and accept the license (access is granted immediately on acceptance).
  2. Create a read token (https://huggingface.co/settings/tokens) and run this script with
     ``HF_TOKEN=<token>`` (or ``huggingface-cli login`` inside the fairchem1 venv).

CUDA does not work in this venv on the RTX 5090 (torch 2.4 cu121, no sm_120): runs on CPU.
This also makes the cached features bit-deterministic (verified below).

Outputs ``results/t4/esen_<set>.npz`` with ``features`` (sum_atoms, (lmax+1)^2 * C)
float32 in coeff-(l,m)-major layout (the raw ``node_embedding`` flattened), ``n_atoms``,
``ids``, and for train/val/test ``targets_irreps`` (N, 18). Layout (lmax, channels) and
checkpoint provenance go to ``results/t4/esen_meta.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
PIEZO_NPZ = REPO / "data/raw/mp/mp_piezoelectric_processed.npz"
SPLIT_NPZ = REPO / "data/splits/mp_piezoelectric_split.npz"
OOD_NPZ = REPO / "data/raw/mp/mp_ood_centrosymmetric_processed.npz"
OUT_DIR = REPO / "results/t4"
CHECKPOINT_REPO = "facebook/OMAT24"
CHECKPOINT_FILE = "esen_30m_oam.pt"  # eSEN-30M-OAM: OMat24 pretrain + sAlex/MPtrj finetune
CHECKPOINT_DIR = REPO / "third_party/checkpoints"


class Shim(dict):
    """fairchem data objects are accessed both as dict and by attribute (t2 probe)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:  # hasattr() must see AttributeError, not KeyError
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    def get(self, key, default=None):
        return dict.get(self, key, default)


def download_checkpoint() -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    try:
        return Path(hf_hub_download(CHECKPOINT_REPO, CHECKPOINT_FILE, local_dir=CHECKPOINT_DIR))
    except GatedRepoError as e:
        raise SystemExit(
            f"{CHECKPOINT_REPO} is gated: accept the FAIR Chemistry License at "
            f"https://huggingface.co/{CHECKPOINT_REPO} with an HF account, then rerun "
            "with HF_TOKEN=<read token>. See this script's docstring."
        ) from e


def load_backbone(path: Path):
    """Load the frozen eSEN backbone + its (lmax, channels) from the trainer checkpoint."""
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator

    calc = OCPCalculator(checkpoint_path=str(path), cpu=True, seed=0)
    model = calc.trainer._unwrapped_model
    backbone = getattr(model, "backbone", model).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    lmax = int(backbone.lmax if hasattr(backbone, "lmax") else backbone.lmax_list[0])
    channels = int(backbone.sphere_channels)
    return backbone, lmax, channels


def load_structures(npz: Path) -> list[dict]:
    d = np.load(npz)
    off = np.concatenate([[0], np.cumsum(d["n_atoms"])])
    return [
        {
            "id": str(d["ids"][i]),
            "z": d["z"][off[i] : off[i + 1]].copy(),
            "pos": d["positions"][off[i] : off[i + 1]].copy(),
            "cell": d["cells"][i].copy(),
        }
        for i in range(len(d["ids"]))
    ]


def as_data(s: dict) -> Shim:
    z = torch.tensor(s["z"], dtype=torch.long)
    n = len(z)
    return Shim(
        pos=torch.tensor(s["pos"], dtype=torch.float32),
        atomic_numbers=z,
        atomic_numbers_full=z,
        cell=torch.tensor(s["cell"], dtype=torch.float32).unsqueeze(0),
        natoms=torch.tensor([n]),
        batch=torch.zeros(n, dtype=torch.long),
        batch_full=torch.zeros(n, dtype=torch.long),
        num_graphs=1,
        pbc=torch.tensor([[True, True, True]]),
        charge=torch.zeros(1, dtype=torch.long),
        spin=torch.zeros(1, dtype=torch.long),
        dataset=["omat"],
    )


def extract(backbone, s: dict) -> np.ndarray:
    """Per-atom node embedding, flattened (n, (lmax+1)^2 * C), coeff-(l,m)-major."""
    with torch.no_grad():
        emb = backbone(as_data(s))["node_embedding"]
    emb = emb.embedding if hasattr(emb, "embedding") else emb
    return emb.reshape(emb.shape[0], -1).cpu().numpy().astype(np.float32)


def main() -> None:
    path = download_checkpoint()
    backbone, lmax, channels = load_backbone(path)
    print(f"loaded {CHECKPOINT_FILE}: lmax={lmax} channels={channels}")

    split = np.load(SPLIT_NPZ)
    piezo = {s["id"]: s for s in load_structures(PIEZO_NPZ)}
    sets: dict[str, list[dict]] = {
        part: [piezo[i] for i in split[part]] for part in ("train", "val", "test")
    }
    sets["ood"] = load_structures(OOD_NPZ)

    # Determinism check: same structure twice must be bit-identical (CPU forward).
    twice = [extract(backbone, sets["train"][0]) for _ in range(2)]
    det = float(np.abs(twice[0] - twice[1]).max())
    print(f"determinism check: max |diff| = {det:.3e}")
    assert det == 0.0, "frozen backbone is not deterministic"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for part, structures in sets.items():
        feats, n_atoms, ids = [], [], []
        for i, s in enumerate(structures):
            feats.append(extract(backbone, s))
            n_atoms.append(len(s["z"]))
            ids.append(s["id"])
            if (i + 1) % 200 == 0:
                print(f"{part}: {i + 1}/{len(structures)}")
        arrays = {
            "features": np.concatenate(feats),
            "n_atoms": np.array(n_atoms, dtype=np.int64),
            "ids": np.array(ids),
        }
        if part != "ood":
            # Targets in the irreps basis were converted by the MAIN env (equiparity's
            # voigt_to_irreps) and cached alongside the MACE features; reuse, id-aligned.
            mace = np.load(OUT_DIR / f"mace_mp0_{part}.npz")
            assert list(mace["ids"]) == ids, f"{part}: id order mismatch vs mace cache"
            arrays["targets_irreps"] = mace["targets_irreps"]
        out = OUT_DIR / f"esen_{part}.npz"
        np.savez_compressed(out, **arrays)
        print(f"{part}: {len(ids)} structures, features {arrays['features'].shape} -> {out}")

    meta = {
        "backbone": "eSEN-30M-OAM (frozen, pretrained)",
        "checkpoint_repo": CHECKPOINT_REPO,
        "checkpoint_file": CHECKPOINT_FILE,
        "checkpoint_path": str(path),
        "lmax": lmax,
        "channels": channels,
        "layout": "coeff-(l,m)-major (n, (lmax+1)^2 * channels); remap in t4_frozen_backbone",
        "dtype": "float32",
        "device": "cpu",
        "determinism_max_abs_diff": det,
    }
    (OUT_DIR / "esen_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {OUT_DIR / 'esen_meta.json'}")


if __name__ == "__main__":
    main()
