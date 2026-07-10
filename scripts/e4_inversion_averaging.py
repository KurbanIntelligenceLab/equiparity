"""E4 -- test-time inversion averaging: what symmetrising an SO(3) model's output actually buys.

The proposed reviewer fix is the odd projection ``T_sym(x) = [T(x) - T(I.x)] / 2``.

For an exactly equivariant model this is vacuous on *both* OOD variants, and provably so for either
arm: if ``x`` is centrosymmetric then ``I.x`` is the same periodic crystal up to a permutation of
atoms and a translation, so any permutation- and translation-invariant model returns
``T(I.x) = T(x)`` and ``T_sym == 0`` identically. Measurement confirms this even on the raw
(DFT-relaxed) variant, whose coordinates satisfy inversion only to within a tolerance -- we
expected a residual signal there and there is none. Symmetrisation removes the false flags, for
a reason that carries no information about parity.

The exception is EquiformerV2, which violates the identity the fix depends on (it is only
approximately rotation-equivariant), so its false-flag rate falls only from ~0.95 to ~0.82.

On an O(3) arm the same projection is a *correctness check*: the model already satisfies
``T(I.x) = -T(x)``, so ``T_sym == T``. That law is verified on non-centrosymmetric crystals,
where ``T`` is a real signal -- on centrosymmetric input it is machine noise and no identity is
measurable.

Run per profile, then render:

    uv run --extra nequip python scripts/e4_inversion_averaging.py \
        --cores nequip allegro equiformer_v2
    uv run --extra mace   python scripts/e4_inversion_averaging.py --cores mace
    python scripts/e4_inversion_averaging.py --render
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from equiparity.domain.structure import AtomicStructure
from equiparity.inference import find_piezo_runs, load_trained

REPO = Path(__file__).resolve().parent.parent
MIRROR = Path.home() / "Desktop" / "parity_work"
OUT_JSON = REPO / "results" / "e4_inversion_averaging.json"
OUT_MD = REPO / "docs" / "results" / "e4_inversion_averaging.md"

VARIANTS = {
    "idealized": "data/raw/mp/mp_ood_centrosymmetric_processed.npz",
    "raw": "data/raw/mp/mp_ood_centrosymmetric_processed_raw.npz",
}
THRESHOLD = 0.01
_N = "&#124;T&#124;"
_NSYM = "&#124;T_sym&#124;"
_MAIN_HEADER = (
    f"| run | core | parity | variant | median {_N} | median {_NSYM} "
    f"| false-flag | false-flag (sym) |"
)
_MINUS = f"{_N[:6]}T(I·x) − T(x){_N[6:]}/{_N}"
_PLUS = f"{_N[:6]}T(I·x) + T(x){_N[6:]}/{_N}"
_LAW_HEADER = f"| run | parity | variant | {_MINUS} | {_PLUS} |"
_CONTROL_HEADER = f"| run | parity | n | {_MINUS} | {_PLUS} |"


def _invert(structure: AtomicStructure) -> AtomicStructure:
    """Inversion through the cell origin: x -> -x, lattice unchanged."""
    return AtomicStructure(
        atomic_numbers=structure.atomic_numbers.copy(),
        positions=-structure.positions,
        cell=None if structure.cell is None else structure.cell.copy(),
        pbc=structure.pbc,
    )


def _norms(tensors: np.ndarray) -> np.ndarray:
    return np.sqrt((tensors**2).sum(axis=1))


def _identity_control(trained) -> dict:
    """On non-centrosymmetric crystals, which inversion law does the model obey?"""
    from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset, load_split

    data = load_crystal_dataset(
        REPO / "data/raw/mp/mp_piezoelectric_processed.npz", ("piezoelectric",)
    )
    split = REPO / "data/splits/mp_piezoelectric_split.npz"
    test = CrystalDataset(data, load_split(split, "test"))
    structures = [test[i].structure for i in range(25)]

    torch.manual_seed(0)
    t_x = trained.predict(structures)
    torch.manual_seed(0)
    t_ix = trained.predict([_invert(s) for s in structures])
    den = np.maximum(_norms(t_x), 1e-30)
    return {
        "n": len(structures),
        "median_rel_T_of_Ix_equals_plus_T": float(np.median(_norms(t_ix - t_x) / den)),
        "median_rel_T_of_Ix_equals_minus_T": float(np.median(_norms(t_ix + t_x) / den)),
    }


def evaluate(cores: list[str]) -> dict:
    from equiparity.io.mp_dataset import CrystalDataset, load_crystal_dataset

    runs = find_piezo_runs(MIRROR)
    results: dict[str, dict] = {}
    for label in sorted(runs):
        core = label.split("_o3_")[0].split("_so3_")[0]
        if core not in cores:
            continue
        trained = load_trained(runs[label], repo_root=REPO)
        entry: dict[str, object] = {"core": core, "parity": trained.parity}

        # Control on NON-centrosymmetric structures. There T(x) is a real signal, so the O(3)
        # parity law T(I.x) = -T(x) is testable; on centrosymmetric input T is machine noise and
        # neither identity means anything (both residuals land near 1.4).
        entry["noncentrosymmetric_control"] = _identity_control(trained)

        for variant, npz in VARIANTS.items():
            ds = CrystalDataset(load_crystal_dataset(REPO / npz))
            structures = [ds[i].structure for i in range(len(ds))]

            torch.manual_seed(0)
            t_x = trained.predict(structures)
            torch.manual_seed(0)
            t_ix = trained.predict([_invert(s) for s in structures])

            t_sym = (t_x - t_ix) / 2.0
            raw_norm, sym_norm = _norms(t_x), _norms(t_sym)

            # How close is T(I.x) to +T(x) (permutation-invariance) vs -T(x) (O(3) parity law)?
            den = np.maximum(raw_norm, 1e-30)
            entry[variant] = {
                "median_violation": float(np.median(raw_norm)),
                "median_violation_symmetrized": float(np.median(sym_norm)),
                "false_flag": float((raw_norm > THRESHOLD).mean()),
                "false_flag_symmetrized": float((sym_norm > THRESHOLD).mean()),
                "median_rel_T_of_Ix_equals_plus_T": float(np.median(_norms(t_ix - t_x) / den)),
                "median_rel_T_of_Ix_equals_minus_T": float(np.median(_norms(t_ix + t_x) / den)),
            }
            v = entry[variant]
            print(
                f"{label:38s} {variant:10s} |T|={v['median_violation']:.3e} "
                f"|T_sym|={v['median_violation_symmetrized']:.3e} "
                f"ff={v['false_flag']:.4f} -> ff_sym={v['false_flag_symmetrized']:.4f}"
            )
        results[label] = entry
    return results


def render() -> None:
    data = json.loads(OUT_JSON.read_text())
    rows = sorted(data.items(), key=lambda kv: (kv[1]["core"], kv[1]["parity"]))

    lines = [
        "# E4 — test-time inversion averaging",
        "",
        "`T_sym(x) = [T(x) - T(I·x)] / 2`, evaluated on all 2,000 centrosymmetric crystals.",
        "",
        "## Symmetrisation is vacuous for an exactly equivariant model — on *both* variants",
        "",
        "If `x` is exactly centrosymmetric then `I·x` is the same periodic crystal up to a",
        "permutation of atoms and a translation. Any permutation- and translation-invariant model",
        "therefore returns `T(I·x) = T(x)`, and `T_sym ≡ 0` identically — regardless of parity.",
        "",
        "The measurement confirms this and **corrects an expectation we held going in**: we ",
        "assumed",
        "the raw (DFT-relaxed) variant would retain a residual signal because its coordinates ",
        "satisfy",
        "inversion only to within a tolerance. It does not. For all three e3nn cores the SO(3)",
        "false-flag rate collapses from ~0.90 to ~0.000 on the raw variant as well. Symmetrisation",
        "does remove the false flags here; it simply does so for a reason that carries no ",
        "information",
        "about parity. **Do not present either zero as evidence that symmetrisation works.**",
        "",
        "## The exception: symmetrisation fails for EquiformerV2",
        "",
        "EquiformerV2's false-flag rate only falls from ~0.95 to ~0.82, and its `|T_sym|` stays at",
        "~2.8e-02. It violates the very identity the fix relies on: `|T(I·x) − T(x)| / |T|` is",
        "0.10–0.15 rather than ~1e-6, because it is only approximately rotation-equivariant (see ",
        "E5).",
        "The one deployed SO(3) model in this study is the one the proposed fix does not repair.",
        "",
        "## Tables",
        "",
        _MAIN_HEADER,
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, r in rows:
        for variant in VARIANTS:
            v = r[variant]
            lines.append(
                f"| {label} | {r['core']} | {r['parity']} | {variant} "
                f"| {v['median_violation']:.3e} | {v['median_violation_symmetrized']:.3e} "
                f"| {v['false_flag']:.4f} | {v['false_flag_symmetrized']:.4f} |"
            )

    lines += [
        "",
        "## Which identity does `T(I·x)` obey?",
        "",
        "Median relative residual of the two candidate laws, on the centrosymmetric set.",
        "Permutation- and translation-invariance force the first on exactly centrosymmetric input.",
        "",
        "The O(3) rows are **uninformative here** (both residuals ≈ 1.4): their `T` is machine ",
        "noise,",
        "so no identity about its direction can be measured. The O(3) parity law is tested in the",
        "control table below, on non-centrosymmetric crystals where `T` is a real signal.",
        "",
        _LAW_HEADER,
        "|---|---|---|---|---|",
    ]
    for label, r in rows:
        for variant in VARIANTS:
            v = r[variant]
            lines.append(
                f"| {label} | {r['parity']} | {variant} "
                f"| {v['median_rel_T_of_Ix_equals_plus_T']:.3e} "
                f"| {v['median_rel_T_of_Ix_equals_minus_T']:.3e} |"
            )

    lines += [
        "",
        "## Control: the O(3) parity law on non-centrosymmetric crystals",
        "",
        "Here `I·x` is a genuinely different crystal and `T(x)` a genuine nonzero prediction, so ",
        "this",
        "is a statement about the *model*, not the structure. An O(3) model must satisfy",
        "`T(I·x) = -T(x)` (second column ≈ 1e-6): inversion is in its symmetry group. An SO(3) ",
        "model",
        "satisfies **neither** law (both columns O(1)) — inversion is simply not a symmetry it was",
        "built to respect, so nothing constrains `T(I·x)` at all. This is the correctness check ",
        "that",
        "makes the `T_sym = T` identity meaningful for the O(3) arms.",
        "",
        _CONTROL_HEADER,
        "|---|---|---|---|---|",
    ]
    for label, r in rows:
        c = r["noncentrosymmetric_control"]
        lines.append(
            f"| {label} | {r['parity']} | {c['n']} "
            f"| {c['median_rel_T_of_Ix_equals_plus_T']:.3e} "
            f"| {c['median_rel_T_of_Ix_equals_minus_T']:.3e} |"
        )

    lines += [
        "",
        "## What the fix costs",
        "",
        "Symmetrisation (a) doubles inference cost, (b) presupposes knowing the target's parity in",
        "advance — exactly the knowledge an O(3) model's features already encode — and (c) ",
        "repairs one",
        "output while leaving the model's internal representations parity-blind for every other",
        "quantity derived from them.",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", nargs="+", default=["nequip", "allegro", "equiformer_v2"])
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.render:
        render()
        return

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    merged.update(evaluate(args.cores))
    OUT_JSON.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {OUT_JSON} ({len(merged)} runs)")


if __name__ == "__main__":
    main()
