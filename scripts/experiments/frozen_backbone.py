"""Train a piezoelectric e3nn head on cached frozen-backbone features.

    uv run python scripts/experiments/frozen_backbone.py --backbone mace_mp0
    uv run python scripts/experiments/frozen_backbone.py --backbone esen

Backbones (features cached by scripts/experiments/cache_{mace,esen}_features.py):

- ``mace_mp0`` (parity-aware): features carry the checkpoint's parity-labelled irreps
  (recorded in ``results/frozen_backbone/mace_mp0_meta.json``); head is the O(3)-correct
  ``o3.Linear(feature_irreps -> 2x1o+1x2o+1x3o)``. On centrosymmetric crystals the odd
  outputs must cancel structurally.
- ``esen`` (parity-blind): features are the (l,m)-major SO(3) embedding; they are remapped
  to mul-major all-even e3nn irreps (the ``t2_backbone_probe._so3_to_e3nn`` remap) and the
  head is the SO(3)-arm relabel ``o3.Linear(all_even -> 2x1e+1x2e+1x3e)``.

Matched protocol per seed {0,1,2}: Adam lr 2e-3, no weight decay, MSE on targets normalized by the
train std (over rows with a nonzero target), 150 epochs, batch 16 structures, GPU. Per-atom head
outputs are index_add-pooled per structure. After training: val/test MAE, OOD violation vector ->
``results/frozen_backbone/<backbone>_seed<k>_ood.npy``, false-flag @ 0.01 C/m^2 + np.logspace(-4,
0, 25) threshold curve -> merged ``results/frozen_backbone.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from e3nn import o3

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results/frozen_backbone"
OUT_JSON = REPO / "results/frozen_backbone.json"

SEEDS = (0, 1, 2)
LR = 2e-3
EPOCHS = 150
BATCH = 16
THRESHOLDS = np.logspace(-4, 0, 25)
FALSE_FLAG_THRESHOLD = 0.01  # C/m^2


class CachedSet:
    """One structure set: features (sum_atoms, D), per-structure batch index, targets."""

    def __init__(self, npz: Path, device: torch.device) -> None:
        d = np.load(npz)
        self.features = torch.tensor(d["features"], dtype=torch.float32, device=device)
        n_atoms = torch.tensor(d["n_atoms"], dtype=torch.long, device=device)
        self.n = int(n_atoms.shape[0])
        self.batch_index = torch.repeat_interleave(torch.arange(self.n, device=device), n_atoms)
        self.offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), n_atoms.cumsum(0)]
        )
        self.targets = (
            torch.tensor(d["targets_irreps"], dtype=torch.float32, device=device)
            if "targets_irreps" in d.files
            else None
        )

    def slice(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the atoms of the selected structures + a compact batch index."""
        chunks, batch = [], []
        for j, i in enumerate(idx.tolist()):
            a, b = int(self.offsets[i]), int(self.offsets[i + 1])
            chunks.append(self.features[a:b])
            batch.append(torch.full((b - a,), j, dtype=torch.long, device=self.features.device))
        return torch.cat(chunks), torch.cat(batch)


def so3_to_e3nn(features: torch.Tensor, lmax: int, channels: int) -> torch.Tensor:
    """(n, (lmax+1)^2 * C) coeff-(l,m)-major -> e3nn 'Cx0e+..+CxLe' mul-major (t2 remap)."""
    emb = features.reshape(features.shape[0], (lmax + 1) ** 2, channels)
    out, offset = [], 0
    for deg in range(lmax + 1):
        n = 2 * deg + 1
        out.append(emb[:, offset : offset + n, :].transpose(1, 2).reshape(emb.shape[0], -1))
        offset += n
    return torch.cat(out, dim=1)


def pooled(head: torch.nn.Module, feats: torch.Tensor, batch: torch.Tensor, n: int):
    per_atom = head(feats)
    out = torch.zeros(n, per_atom.shape[1], dtype=per_atom.dtype, device=per_atom.device)
    return out.index_add_(0, batch, per_atom)


def train_one(
    backbone: str, seed: int, sets: dict[str, CachedSet], src_irreps: str, out_irreps: str
) -> dict:
    device = sets["train"].features.device
    torch.manual_seed(seed)
    head = o3.Linear(o3.Irreps(src_irreps), o3.Irreps(out_irreps)).to(device)
    n_params = sum(int(p.numel()) for p in head.parameters())
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=0.0)

    train = sets["train"]
    assert train.targets is not None
    real_rows = train.targets.abs().amax(dim=1) > 0
    scale = float(train.targets[real_rows].std())
    norm_targets = train.targets / scale

    rng = np.random.default_rng(seed)
    for _epoch in range(EPOCHS):
        order = torch.tensor(rng.permutation(train.n), device=device)
        for start in range(0, train.n, BATCH):
            idx = order[start : start + BATCH]
            feats, batch = train.slice(idx)
            pred = pooled(head, feats, batch, len(idx))
            loss = ((pred - norm_targets[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def predict(s: CachedSet) -> torch.Tensor:
        preds = []
        for start in range(0, s.n, 256):
            idx = torch.arange(start, min(start + 256, s.n), device=device)
            feats, batch = s.slice(idx)
            preds.append(pooled(head, feats, batch, len(idx)) * scale)
        return torch.cat(preds)

    result: dict[str, object] = {"seed": seed, "n_params": n_params, "scale": scale}
    for part in ("val", "test"):
        s = sets[part]
        assert s.targets is not None
        pred = predict(s)
        result[f"{part}_mae"] = float((pred - s.targets).abs().mean())
        result[f"{part}_rmse"] = float(((pred - s.targets) ** 2).mean().sqrt())
    v = predict(sets["ood"]).norm(dim=1).cpu().numpy().astype(np.float64)
    np.save(OUT_DIR / f"{backbone}_seed{seed}_ood.npy", v)
    result["ood_violation_median"] = float(np.median(v))
    result["ood_violation_max"] = float(v.max())
    result["ood_false_flag_at_0.01"] = float((v > FALSE_FLAG_THRESHOLD).mean())
    result["thresholds"] = THRESHOLDS.tolist()
    result["false_flag_curve"] = [float((v > t).mean()) for t in THRESHOLDS]
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=("mace_mp0", "esen"), required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((OUT_DIR / f"{args.backbone}_meta.json").read_text())
    sets = {
        part: CachedSet(OUT_DIR / f"{args.backbone}_{part}.npz", device)
        for part in ("train", "val", "test", "ood")
    }

    if args.backbone == "mace_mp0":
        src_irreps = meta["feature_irreps"]  # parity-labelled, from the checkpoint
        out_irreps = "2x1o+1x2o+1x3o"  # O(3)-correct piezoelectric head
    else:
        lmax, channels = int(meta["lmax"]), int(meta["channels"])
        for part in sets.values():  # (l,m)-major -> mul-major all-even, once
            part.features = so3_to_e3nn(part.features, lmax, channels)
        src_irreps = "+".join(f"{channels}x{deg}e" for deg in range(lmax + 1))
        out_irreps = "2x1e+1x2e+1x3e"  # SO(3)-arm relabel

    per_seed = [train_one(args.backbone, seed, sets, src_irreps, out_irreps) for seed in SEEDS]
    for r in per_seed:
        print(
            f"{args.backbone} seed {r['seed']}: test MAE {r['test_mae']:.4f}, "
            f"OOD median {r['ood_violation_median']:.3e}, "
            f"false-flag@0.01 {r['ood_false_flag_at_0.01']:.4f}"
        )

    ff = np.array([r["ood_false_flag_at_0.01"] for r in per_seed])
    summary = {
        "feature_irreps": src_irreps,
        "output_irreps": out_irreps,
        "feature_dim": int(sets["train"].features.shape[1]),
        "n_params": per_seed[0]["n_params"],
        "scale": [r["scale"] for r in per_seed],
        "protocol": {"lr": LR, "epochs": EPOCHS, "batch": BATCH, "optimizer": "Adam", "wd": 0.0},
        "backbone_meta": meta,
        "seeds": per_seed,
        "false_flag_mean": float(ff.mean()),
        "false_flag_std": float(ff.std()),
        "ood_violation_median_mean": float(np.mean([r["ood_violation_median"] for r in per_seed])),
        "test_mae_mean": float(np.mean([r["test_mae"] for r in per_seed])),
    }
    merged = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    merged[args.backbone] = summary
    OUT_JSON.write_text(json.dumps(merged, indent=1) + "\n")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
