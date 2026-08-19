"""The crystals SO(3) gets right are the ones rotations already forbid.

SO(3) equivariance alone forces ``T(x) = R.T(x)`` for every proper rotation ``R`` in a crystal's
point group. For point group **m-3m** the proper-rotation subgroup is **432**, under which the
only invariant rank-3 tensor is zero -- so an exactly SO(3)-equivariant model must predict exactly
zero, with no parity label anywhere. For **m-3** the rotation subgroup is **23**, a piezoelectric
class: invariants exist, and nothing forces a zero.

So SO(3)'s zeros are the ones rotations already guarantee; O(3)'s zeros are the strictly larger
set that parity guarantees. This is a supplementary analysis: it reuses the committed violation
vectors and changes no headline number.

The group theory itself is a repo test (``tests/test_physics_claims.py``), not an assertion here.

    uv run --extra nequip --extra data python scripts/experiments/rotation_subgroup.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from aggregate_grid import CORE_LABEL, CORES, SEEDS, load_vectors

REPO = Path(__file__).resolve().parents[2]
SPACEGROUPS = REPO / "results" / "ood_spacegroups.json"
OUT_JSON = REPO / "results" / "rotation_subgroup.json"
OUT_MD = REPO / "docs" / "results" / "rotation_subgroup.md"

THRESHOLD = 0.01
FAMILIES = ["m-3", "m-3m", "non-cubic"]
ROTATION_SUBGROUP = {"m-3": "23 (order 12)", "m-3m": "432 (order 24)", "non-cubic": "various"}
RANK3_ALLOWED = {"m-3": "yes", "m-3m": "**no**", "non-cubic": "yes"}
_TABLE_HEADER = (
    "| core | arm | family | rotation subgroup | rank-3 invariant? | n | median ‖T‖ | false-flag |"
)


def main() -> None:
    records = json.loads(SPACEGROUPS.read_text())["records"]
    family = np.array([r["family"] for r in records])
    masks = {f: family == f for f in FAMILIES}

    vectors = load_vectors()
    stats: dict[str, dict] = {}

    for core in CORES:
        for parity in ("o3", "so3"):
            if (core, parity, SEEDS[0], "idealized") not in vectors:
                continue
            per_seed = [vectors[(core, parity, s, "idealized")] for s in SEEDS]
            entry: dict[str, object] = {}
            for f, mask in masks.items():
                medians = [float(np.median(v[mask])) for v in per_seed]
                flags = [float((v[mask] > THRESHOLD).mean()) for v in per_seed]
                entry[f] = {
                    "n": int(mask.sum()),
                    "median_min": min(medians),
                    "median_max": max(medians),
                    "median_mean": float(np.mean(medians)),
                    "false_flag_min": min(flags),
                    "false_flag_max": max(flags),
                    "false_flag_mean": float(np.mean(flags)),
                }
            stats[f"{core}_{parity}"] = entry

    # Where do the SO(3) "successes" live? Seed-averaged, per core.
    successes: dict[str, dict] = {}
    for core in CORES:
        if (core, "so3", SEEDS[0], "idealized") not in vectors:
            continue
        mean_v = np.mean([vectors[(core, "so3", s, "idealized")] for s in SEEDS], axis=0)
        unflagged = mean_v <= THRESHOLD
        n = int(unflagged.sum())
        successes[core] = {
            "n_unflagged": n,
            "n_unflagged_cubic": int((unflagged & (masks["m-3"] | masks["m-3m"])).sum()),
            "n_unflagged_m3m": int((unflagged & masks["m-3m"]).sum()),
            "n_unflagged_m3": int((unflagged & masks["m-3"]).sum()),
            "fraction_cubic": (
                float((unflagged & (masks["m-3"] | masks["m-3m"])).sum() / n) if n else 0.0
            ),
        }

    OUT_JSON.write_text(
        json.dumps({"threshold": THRESHOLD, "by_family": stats, "successes": successes}, indent=1)
        + "\n"
    )
    _render(stats, successes, masks)


def _fmt_range(lo: float, hi: float) -> str:
    if lo == hi:
        return f"{lo:.3e}"
    return f"{lo:.2e} – {hi:.2e}"


def _render(stats: dict, successes: dict, masks: dict) -> None:
    lines = [
        "# The rotation subgroup explains SO(3)'s correct zeros (supplementary)",
        "",
        "SO(3) equivariance forces `T(x) = R·T(x)` for every proper rotation `R` in the crystal's",
        "point group. Under **432** (the proper subgroup of m-3̄m) the only invariant rank-3 ",
        "tensor is",
        "zero, so an exactly SO(3)-equivariant model is *forced* to predict zero — no parity label",
        "required. Under **23** (the proper subgroup of m-3̄, a piezoelectric class) invariants ",
        "exist",
        "and nothing is forced. Both group-theoretic facts are asserted as tests in",
        "`tests/test_physics_claims.py`.",
        "",
        f"Counts in the OOD set: m-3̄ **{int(masks['m-3'].sum())}**, "
        f"m-3̄m **{int(masks['m-3m'].sum())}**,",
        f"non-cubic **{int(masks['non-cubic'].sum())}** (total 2,000; spglib at symprec 1e-3).",
        "",
        "## Violation by point-group family (idealized variant)",
        "",
        "Range over the 3 training seeds. False-flag threshold 0.01 C/m².",
        "",
        _TABLE_HEADER,
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(stats):
        core, parity = key.rsplit("_", 1)
        for f in FAMILIES:
            e = stats[key][f]
            lines.append(
                f"| {CORE_LABEL[core]} | {parity} | {f} | {ROTATION_SUBGROUP[f]} "
                f"| {RANK3_ALLOWED[f]} | {e['n']} "
                f"| {_fmt_range(e['median_min'], e['median_max'])} "
                f"| {_fmt_range(e['false_flag_min'], e['false_flag_max'])} |"
            )

    lines += [
        "",
        "## Where SO(3)'s correct zeros live",
        "",
        "Structures the SO(3) arm does **not** false-flag (seed-averaged ‖T‖ ≤ 0.01).",
        "",
        "| core | unflagged | of which cubic | m-3̄m | m-3̄ | cubic fraction |",
        "|---|---|---|---|---|---|",
    ]
    for core, s in successes.items():
        lines.append(
            f"| {CORE_LABEL[core]} | {s['n_unflagged']} | {s['n_unflagged_cubic']} "
            f"| {s['n_unflagged_m3m']} | {s['n_unflagged_m3']} | {s['fraction_cubic']:.3f} |"
        )

    lines += [
        "",
        "## Reading",
        "",
        "For the three exactly-SO(3) e3nn cores (NequIP, Allegro, MACE) the m-3̄m false-flag rate ",
        "is",
        "**0.000** in every seed and the median ‖T‖ sits at machine zero (1.8e-07 – 1.1e-06): the ",
        "432",
        "constraint is satisfied exactly, by rotation alone. All 166 m-3̄m crystals are unflagged, ",
        "for",
        "every core and every seed.",
        "",
        "The m-3̄ crystals, whose rotation subgroup 23 does permit a rank-3 tensor, are ",
        "false-flagged at",
        "**0.889 – 1.000** depending on seed (one MACE seed leaves 2 of the 18 below threshold). ",
        "No",
        "m-3̄ crystal survives seed-averaging unflagged.",
        "",
        "The correspondence is therefore strong but not exhaustive: the 432 constraint accounts ",
        "for 166",
        "of each core's 175–200 unflagged crystals (83%–95%). The remainder are non-cubic ",
        "structures",
        "on which the model simply happened to predict a small magnitude — no symmetry protects ",
        "them,",
        "and they carry no guarantee.",
        "",
        "**EquiformerV2 is the exception, and it is not a symmetry-group difference.** It is only",
        "*approximately* SO(3)-equivariant (the output-parity audit: rotation error 7e-2 – 1.1e-1; "
        "its forward redraws ",
        "a",
        "random per-edge frame each call), so on m-3̄m — where the exact SO(3) answer is zero — ",
        "the",
        "entire predicted signal is equivariance error. Its m-3̄m numbers are single stochastic ",
        "draws",
        "and straddle the 0.01 threshold; they should not be read as a statement about SO(3).",
        "",
        "**Consequence for the headline.** 1,816 non-cubic + 18 m-3̄ = 1,834 of 2,000 (91.7%) of ",
        "the",
        "OOD set lies outside what rotation can enforce, and the observed SO(3) false-flag rate is",
        "0.89–0.91 — close to that ceiling, the small shortfall being non-cubic crystals with",
        "incidentally small predictions. SO(3)'s zeros are the ones rotations already guarantee;",
        "O(3)'s zeros are the strictly larger set that parity guarantees. This sharpens the ",
        "paper's",
        "claim rather than weakening it, and it costs no new inference.",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
