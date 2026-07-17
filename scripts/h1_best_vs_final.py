"""H1 (partial) -- OOD false-flag at the best-validation checkpoint vs the final epoch.

The full H-1 curve (false-flag fraction vs training epoch) cannot be reconstructed: per-epoch
metrics were never persisted and only two checkpoints per run survive. But those two are
different training points -- ``checkpoint_best.pt`` is the best-validation state from one of
the ~10 validation evaluations during training, and ``checkpoint_latest.pt`` is the final
epoch, the state behind every released OOD number. Evaluating the false-flag fraction on both
gives a two-point trajectory per run: if they agree at ~0.90, the fraction was flat over the
later part of training, which is direct (if coarse) evidence against the undertrained
objection.

Run once per install profile:

    uv run --extra nequip python scripts/h1_best_vs_final.py \
        --cores nequip allegro equiformer_v2
    uv run --extra mace   python scripts/h1_best_vs_final.py --cores mace
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from equiparity.inference import find_piezo_runs, load_trained

REPO = Path(__file__).resolve().parent.parent
MIRROR = Path.home() / "Desktop" / "parity_work"
OUT_JSON = REPO / "results" / "h1_best_vs_final.json"
THRESHOLD = 0.01


def _ood_structures() -> list:
    from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset

    npz = REPO / "data" / "raw" / "mp" / "mp_ood_centrosymmetric_processed.npz"
    ds = CrystalDataset(load_crystal_dataset(npz))
    return [ds[i].structure for i in range(len(ds))]


def evaluate(cores: list[str]) -> dict:
    structures = _ood_structures()
    runs = find_piezo_runs(MIRROR)
    out: dict[str, dict] = {}
    for label, run_dir in sorted(runs.items()):
        core = label.split("_o3_")[0].split("_so3_")[0]
        if core not in cores or "_so3_" not in label:
            continue
        best_path = run_dir / "checkpoint_best.pt"
        if not best_path.exists():
            continue
        trained = load_trained(run_dir, repo_root=REPO)  # loads checkpoint_latest (final epoch)

        torch.manual_seed(0)
        v_final = trained.violations(structures)

        best_sd = torch.load(best_path, map_location=trained.device, weights_only=False)
        trained.model.load_state_dict(best_sd)
        torch.manual_seed(0)
        v_best = trained.violations(structures)

        out[label] = {
            "core": core,
            "ff_final": float((v_final > THRESHOLD).mean()),
            "ff_best_val": float((v_best > THRESHOLD).mean()),
            "median_final": float(np.median(v_final)),
            "median_best_val": float(np.median(v_best)),
        }
        e = out[label]
        print(
            f"{label:38s} ff(best-val)={e['ff_best_val']:.4f} ff(final)={e['ff_final']:.4f} "
            f"median {e['median_best_val']:.3f} -> {e['median_final']:.3f}"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", nargs="+", default=["nequip", "allegro", "equiformer_v2"])
    args = parser.parse_args()

    merged = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    merged.update(evaluate(args.cores))
    OUT_JSON.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT_JSON} ({len(merged)} runs)")


if __name__ == "__main__":
    main()
