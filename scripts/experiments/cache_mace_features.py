"""Cache frozen MACE-MP-0 per-atom features for the piezoelectric splits and OOD set.

Runs in the main repo env with the mace extra:

    uv run --extra mace python scripts/experiments/cache_mace_features.py

The backbone is the released MACE-MP-0 "small" foundation model (frozen, eval mode, float32 on
CUDA). Per-atom features are read from the DEEPEST interaction linear whose output irreps carry
l>0 (mirrors ``MACETensorModel._deepest_tensor_probe``); for this checkpoint that is
``interactions.1.linear`` with irreps ``128x0e+128x1o+128x2e+128x3o``. Structures are fed as
``mace.data.Configuration`` WITH cell+pbc (parity leaks otherwise).

Outputs ``results/frozen_backbone/mace_mp0_<set>.npz`` with keys ``features`` (sum_atoms, D)
float32, ``n_atoms`` (N,), ``ids`` (N,), and for train/val/test ``targets_irreps`` (N, 18) — the
Voigt (3, 6) rows converted to the 2x1o+1x2o+1x3o irreps basis. Metadata (feature irreps, probe
layer, checkpoint, determinism check) goes to ``results/frozen_backbone/mace_mp0_meta.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from equiparity.features.tensor_irreps import voigt_to_irreps

REPO = Path(__file__).resolve().parents[2]
PIEZO_NPZ = REPO / "data/raw/mp/mp_piezoelectric_processed.npz"
SPLIT_NPZ = REPO / "data/splits/mp_piezoelectric_split.npz"
OOD_NPZ = REPO / "data/raw/mp/mp_ood_centrosymmetric_processed.npz"
OUT_DIR = REPO / "results/frozen_backbone"

BATCH_STRUCTURES = 16


def load_structures(npz: Path) -> list[dict]:
    d = np.load(npz)
    off = np.concatenate([[0], np.cumsum(d["n_atoms"])])
    has_target = "piezoelectric" in d.files
    out = []
    for i in range(len(d["ids"])):
        s = {
            "id": str(d["ids"][i]),
            "z": d["z"][off[i] : off[i + 1]].copy(),
            "pos": d["positions"][off[i] : off[i + 1]].copy(),
            "cell": d["cells"][i].copy(),
        }
        if has_target:
            s["target_irreps"] = voigt_to_irreps(
                d["piezoelectric"][i].reshape(3, 6), "piezoelectric"
            )
        out.append(s)
    return out


def build_probe(device: torch.device):
    """Load frozen MACE-MP-0 small; return (model, z_table, r_max, probe_module, irreps)."""
    from e3nn import o3
    from mace.calculators import mace_mp

    calc = mace_mp(model="small", device=str(device), default_dtype="float32")
    model = calc.models[0].eval()
    for p in model.parameters():
        p.requires_grad_(False)

    best_idx, best = -1, None
    for name, module in model.named_modules():
        if not (".interactions." in f".{name}" and name.endswith(".linear")):
            continue
        irreps = getattr(module, "irreps_out", None)
        if irreps is None or not any(ir.ir.l > 0 for ir in o3.Irreps(str(irreps))):
            continue
        idx = int(name.split("interactions.")[1].split(".")[0])
        if idx > best_idx:
            best_idx, best = idx, (name, module)
    assert best is not None
    path = getattr(calc, "model_paths", None) or getattr(calc, "model_path", "mace_mp small")
    return model, calc.z_table, float(model.r_max), best, str(path)


def batches(structures: list[dict], z_table, r_max: float):
    """Yield mace torch_geometric batches of BATCH_STRUCTURES structures, WITH cell+pbc."""
    from mace import data
    from mace.tools import torch_geometric

    for start in range(0, len(structures), BATCH_STRUCTURES):
        chunk = structures[start : start + BATCH_STRUCTURES]
        atomic = [
            data.AtomicData.from_config(
                data.Configuration(
                    atomic_numbers=s["z"],
                    positions=s["pos"],
                    properties={},
                    property_weights={},
                    cell=s["cell"],
                    pbc=(True, True, True),
                ),
                z_table=z_table,
                cutoff=r_max,
            )
            for s in chunk
        ]
        loader = torch_geometric.dataloader.DataLoader(dataset=atomic, batch_size=len(atomic))  # type: ignore[arg-type]
        yield chunk, next(iter(loader))


def extract(model, probe_module, batch, device: torch.device) -> np.ndarray:
    store: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        store["feat"] = output if torch.is_tensor(output) else output[0]

    handle = probe_module.register_forward_hook(hook)
    try:
        d = batch.to(device).to_dict()
        for key, value in d.items():
            if torch.is_tensor(value) and value.is_floating_point():
                d[key] = value.to(torch.float32)
        with torch.no_grad():
            model(d, compute_force=False, compute_virials=False, compute_stress=False)
    finally:
        handle.remove()
    return store["feat"].detach().cpu().numpy().astype(np.float32)


def main() -> None:
    # Caching runs on CPU: the MACE forward on this GPU (sm_120, cuBLAS float32) differs by
    # ~1e-5 between identical calls even under torch.use_deterministic_algorithms(True); the
    # CPU forward is bit-exact and the whole cache takes ~2 minutes anyway.
    device = torch.device("cpu")
    model, z_table, r_max, (probe_name, probe_module), model_path = build_probe(device)
    feature_irreps = str(probe_module.irreps_out)
    print(f"probe={probe_name} irreps={feature_irreps} r_max={r_max}")

    supported = set(z_table.zs)
    split = np.load(SPLIT_NPZ)
    piezo = {s["id"]: s for s in load_structures(PIEZO_NPZ)}
    sets: dict[str, list[dict]] = {
        part: [piezo[i] for i in split[part]] for part in ("train", "val", "test")
    }
    sets["ood"] = load_structures(OOD_NPZ)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict[str, object] = {
        "backbone": "MACE-MP-0 small (ScaleShiftMACE)",
        "model_path": model_path,
        "probe_layer": probe_name,
        "feature_irreps": feature_irreps,
        "r_max": r_max,
        "dtype": "float32",
        "device": str(device),
        "dropped": {},
    }

    # Determinism check: featurize the first train structure twice, must be bit-identical.
    probe_s = sets["train"][:1]
    twice = []
    for _ in range(2):
        _, batch = next(batches(probe_s, z_table, r_max))
        twice.append(extract(model, probe_module, batch, device))
    det = float(np.abs(twice[0] - twice[1]).max())
    meta["determinism_max_abs_diff"] = det
    print(f"determinism check: max |diff| = {det:.3e}")
    assert det == 0.0, "frozen backbone is not deterministic"

    for part, structures in sets.items():
        kept = [s for s in structures if set(s["z"]).issubset(supported)]
        dropped = [s["id"] for s in structures if s["id"] not in {k["id"] for k in kept}]
        meta["dropped"][part] = dropped  # type: ignore[index]
        feats, n_atoms, ids, targets = [], [], [], []
        for chunk, batch in batches(kept, z_table, r_max):
            feats.append(extract(model, probe_module, batch, device))
            for s in chunk:
                n_atoms.append(len(s["z"]))
                ids.append(s["id"])
                if "target_irreps" in s:
                    targets.append(s["target_irreps"])
        arrays = {
            "features": np.concatenate(feats),
            "n_atoms": np.array(n_atoms, dtype=np.int64),
            "ids": np.array(ids),
        }
        if targets:
            arrays["targets_irreps"] = np.stack(targets).astype(np.float32)
        out = OUT_DIR / f"mace_mp0_{part}.npz"
        np.savez_compressed(out, **arrays)
        print(
            f"{part}: {len(kept)} structures ({len(dropped)} dropped), "
            f"features {arrays['features'].shape} -> {out}"
        )

    (OUT_DIR / "mace_mp0_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {OUT_DIR / 'mace_mp0_meta.json'}")


if __name__ == "__main__":
    main()
