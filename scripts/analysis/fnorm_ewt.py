"""Convert the matched cores' piezoelectric test predictions to Fnorm and EwT.

The crystal-tensor benchmark literature reports the mean Frobenius-norm distance (Fnorm) and the
error-within-threshold rate (EwT m%: fraction of test samples with Fnorm(error)/Fnorm(label) <
m%), on the JARVIS-DFT piezoelectric set with zero tensors removed (CEITNet paper,
arXiv:2602.04323, Table 5). This computes the same two metrics for every arm on our Materials
Project test split, so Methods can state where the models land. The comparison is indicative, not
head-to-head: different source database, different split, and our split keeps its 6 zero-norm test
rows (EwT is computed over the nonzero-label rows only, matching the benchmark's zero-removal
rationale; Fnorm is reported both ways).

Run once per install profile:

    uv run --extra nequip python scripts/analysis/fnorm_ewt.py --cores nequip allegro equiformer_v2
    uv run --extra mace   python scripts/analysis/fnorm_ewt.py --cores mace
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from equiparity.inference import find_piezo_runs, load_trained

REPO = Path(__file__).resolve().parents[2]
MIRROR = Path(os.environ.get("PARITY_RUNS", Path(__file__).resolve().parents[2] / "runs"))
OUT_JSON = REPO / "results" / "fnorm_ewt.json"

# Table 5 of arXiv:2602.04323 (CEITNet), transcribed from the PDF 2026-07-16; ETGNN, GMTNet and
# GoeCTP rows are credited there to Hua et al. 2025 (arXiv:2410.02372). JARVIS-DFT piezoelectric
# dataset, 2,701 samples after zero-tensor removal, test n=270.
BENCHMARK = {
    "ETGNN": {"fnorm": 0.873, "ewt25": 0.0, "ewt10": 0.0, "ewt5": 0.0},
    "GMTNet": {"fnorm": 0.752, "ewt25": 6.29, "ewt10": 1.48, "ewt5": 1.11},
    "GoeCTP": {"fnorm": 0.778, "ewt25": 2.59, "ewt10": 1.14, "ewt5": 0.04},
    "CEITNet": {"fnorm": 0.517, "ewt25": 21.98, "ewt10": 5.80, "ewt5": 2.72},
}
THRESHOLDS = (0.25, 0.10, 0.05)


def _test_data():
    from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset, load_split
    from equiparity.training.nequip_tensor import _irreps_targets

    data = load_crystal_dataset(
        REPO / "data/raw/mp/mp_piezoelectric_processed.npz", ("piezoelectric",)
    )
    test = CrystalDataset(data, load_split(REPO / "data/splits/mp_piezoelectric_split.npz", "test"))
    targets = _irreps_targets(test, "piezoelectric", "piezoelectric")
    structures = [test[i].structure for i in range(len(test))]
    return structures, targets


def evaluate(cores: list[str]) -> dict:
    structures, targets = _test_data()
    label_norm = np.linalg.norm(targets, axis=1)
    nonzero = label_norm > 0

    runs = find_piezo_runs(MIRROR)
    out: dict[str, dict] = {}
    for label in sorted(runs):
        core = label.split("_o3_")[0].split("_so3_")[0]
        if core not in cores:
            continue
        trained = load_trained(runs[label], repo_root=REPO)
        torch.manual_seed(0)
        preds = trained.predict(structures)
        err = np.linalg.norm(preds - targets, axis=1)
        rel = err[nonzero] / label_norm[nonzero]
        entry = {
            "core": core,
            "parity": trained.parity,
            "n_test": int(err.size),
            "n_nonzero_label": int(nonzero.sum()),
            "fnorm_all": float(err.mean()),
            "fnorm_nonzero": float(err[nonzero].mean()),
            **{f"ewt{int(t * 100)}": float((rel < t).mean() * 100.0) for t in THRESHOLDS},
        }
        out[label] = entry
        print(
            f"{label:38s} Fnorm={entry['fnorm_nonzero']:.4f} "
            f"EwT25={entry['ewt25']:.2f}% EwT10={entry['ewt10']:.2f}% EwT5={entry['ewt5']:.2f}%"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", nargs="+", default=["nequip", "allegro", "equiformer_v2"])
    args = parser.parse_args()

    merged = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    merged.setdefault("benchmark_table5_arxiv2602_04323", BENCHMARK)
    merged.setdefault("runs", {}).update(evaluate(args.cores))

    # seed-aggregated per arm
    arms: dict[str, dict] = {}
    for e in merged["runs"].values():
        key = f"{e['core']}/{e['parity']}"
        arms.setdefault(key, []).append(e)
    merged["arms"] = {
        key: {
            "fnorm_nonzero_mean": float(np.mean([e["fnorm_nonzero"] for e in es])),
            "fnorm_nonzero_std": float(np.std([e["fnorm_nonzero"] for e in es], ddof=1))
            if len(es) > 1
            else 0.0,
            **{f"ewt{m}_mean": float(np.mean([e[f"ewt{m}"] for e in es])) for m in (25, 10, 5)},
            "n_seeds": len(es),
        }
        for key, es in arms.items()
    }
    OUT_JSON.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT_JSON}")
    for key, a in sorted(merged["arms"].items()):
        print(
            f"{key:24s} Fnorm={a['fnorm_nonzero_mean']:.4f}±{a['fnorm_nonzero_std']:.4f} "
            f"EwT25={a['ewt25_mean']:.2f}% EwT10={a['ewt10_mean']:.2f}% EwT5={a['ewt5_mean']:.2f}%"
        )


if __name__ == "__main__":
    main()
