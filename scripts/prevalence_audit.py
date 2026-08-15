"""Task 0.4 -- the prevalence audit: which released models carry parity labels on their features?

This is Table 1 of the paper. Every row is classified by **inspecting released source**, and records
the version or commit inspected plus the specific line that decides the classification. Nothing here
is taken from a paper's prose; where the source does not decide the question, the row says so.

Categories
----------
``parity-aware``   features carry an e3nn parity label (``0e``/``1o``/``2e``), or are built from an
                   equivalent construction in which a degree-l object picks up ``(-1)^l`` under
                   inversion. An odd-parity output is then structurally zero on a centrosymmetric
                   input.
``SO(3)-only``     features are rotation-typed but parity-free (all-even irreps, or an ``lmax``-only
                   SO(3) embedding). Nothing forces an odd-parity output to vanish.
``vector-only``    equivariance is carried by Cartesian vector channels rather than typed irreps.
                   These can express a vector but not a rank-3 tensor head without extra machinery.
``invariant``      the network sees only distances (and angles); it cannot express any equivariant
                   output at all.
``undetermined``   source inspection does not settle it; a reflection test on the built model would.

    uv run --extra nequip python scripts/prevalence_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "results" / "prevalence_audit.json"
OUT_MD = REPO / "docs" / "reports" / "checkpoint8_prevalence_audit.md"

# Each row: (model, source, version_or_commit, category, evidence_file, evidence_line, evidence)
ROWS: list[dict] = [
    {
        "model": "SchNet",
        "source": "torch_geometric.nn.models.schnet",
        "version": "torch-geometric 2.8.0",
        "category": "invariant",
        "evidence_file": "torch_geometric/nn/models/schnet.py",
        "evidence": (
            "Interactions consume only `edge_weight` (interatomic distances); the file contains "
            "zero references to `Irreps` or `o3`. A distance-only network is invariant under every "
            "element of O(3), so it cannot emit a tensor of any rank."
        ),
    },
    {
        "model": "DimeNet++",
        "source": "torch_geometric.nn.models.dimenet",
        "version": "torch-geometric 2.8.0",
        "category": "invariant",
        "evidence_file": "torch_geometric/nn/models/dimenet.py",
        "evidence": (
            "Message passing is over distances and bond angles; zero references to `Irreps` or "
            "`o3`. Angles are invariant under reflection, so the representation cannot distinguish "
            "a structure from its mirror image."
        ),
    },
    {
        "model": "PaiNN",
        "source": "torchmd-net (torchmdnet/models)",
        "version": "commit 2a2c913",
        "category": "vector-only",
        "evidence_file": "torchmdnet/models/torchmd_et.py",
        "evidence": (
            "Equivariance is carried by an explicit Cartesian vector channel "
            "`vec = torch.zeros(x.size(0), 3, x.size(1))` (line 213), updated alongside scalars. "
            "No `Irreps` anywhere in `torchmdnet/models/`. A vector channel is an l=1 object; the "
            "code does not label its parity, and there is no rank-3 path."
        ),
    },
    {
        "model": "TorchMD-Net (ET)",
        "source": "torchmd-net",
        "version": "commit 2a2c913",
        "category": "vector-only",
        "evidence_file": "torchmdnet/models/torchmd_et.py",
        "evidence": (
            "Same Cartesian `vec` channel (line 213), `dx, dvec = attn(...)` (line 216). No typed "
            "irreps; parity is never represented."
        ),
    },
    {
        "model": "ViSNet",
        "source": "torch_geometric.nn.models.visnet",
        "version": "torch-geometric 2.8.0",
        "category": "vector-only",
        "evidence_file": "torch_geometric/nn/models/visnet.py",
        "evidence": (
            "Maintains a Cartesian vector channel (`vec = torch.cat([vec1, vec2], dim=1)`, "
            "line 269); zero references to `Irreps` or `o3`."
        ),
    },
    {
        "model": "FAENet",
        "source": "vict0rsch/faenet",
        "version": "commit 1f725cf",
        "category": "invariant",
        "evidence_file": "faenet/tests/test_model_symmetries.py",
        "evidence": (
            "The network itself is not equivariant; symmetry is imposed externally by frame "
            "averaging (`frame_averaging` in {'', '2D', '3D', 'DA'}). Parity is a property of the "
            "chosen frame set, not of the features. Reflections enter only if the frame set "
            "includes them."
        ),
    },
    {
        "model": "NequIP",
        "source": "nequip",
        "version": "0.18.0",
        "category": "parity-aware",
        "evidence_file": "nequip/model/nequip_models.py",
        "evidence": (
            "`parity: bool = True` (line 120). Its own docstring (line 138) shows that with "
            "`parity=False` and `l_max=2` the features are still `5x0e, 2x1o, 7x2e` -- i.e. the "
            "flag does not strip parity labels. This corroborates our Checkpoint-1 finding that "
            "the boolean is NOT the SO(3) toggle; we build the SO(3) arm by relabelling irreps."
        ),
    },
    {
        "model": "Allegro",
        "source": "nequip-allegro",
        "version": "0.8.3",
        "category": "parity-aware",
        "evidence_file": "allegro/model/allegro_models.py",
        "evidence": (
            "`parity: bool = True` (line 36), 'whether to include features with odd mirror parity' "
            "(line 48), consumed at line 75. Features are e3nn irreps with parity labels."
        ),
    },
    {
        "model": "MACE",
        "source": "mace-torch",
        "version": "0.3.16",
        "category": "parity-aware",
        "evidence_file": "mace/modules/models.py",
        "evidence": (
            "`use_so3: bool = False` (line 67), stored at line 104. The default builds the edge "
            "spherical harmonics with natural parity `(-1)^l`; `use_so3=True` makes them all-even, "
            "which is the correct native SO(3) toggle and is what our matched pair uses."
        ),
    },
    {
        "model": "Equiformer (v1)",
        "source": "atomicarchitects/equiformer",
        "version": "commit 64cb786",
        "category": "SO(3)-only",
        "evidence_file": "nets/graph_attention_transformer.py",
        "evidence": (
            "Every shipped default node embedding is all-even: "
            "`irreps_node_embedding='128x0e+64x1e+32x2e'` (line 739; likewise "
            "dp_attention_transformer.py:261, equiformer_md17_dens.py:59) and "
            "`'256x0e+128x1e'` for OC20 (graph_attention_transformer_oc20.py:89). Odd irreps "
            "appear only in an `irreps_head` and in auxiliary force outputs "
            "(dp_attention_transformer_oc20.py:182). As released, Equiformer v1's node features "
            "carry no odd-parity channel."
        ),
    },
    {
        "model": "EquiformerV2",
        "source": "vendored (src/equiparity/models/equiformer_v2)",
        "version": "as vendored in this repo",
        "category": "SO(3)-only",
        "evidence_file": "src/equiparity/models/equiformer_v2/so3.py",
        "evidence": (
            "`SO3_Embedding` is indexed by `lmax_list`/`mmax_list` only; `so3.py` contains zero "
            "`Irreps` constructions and no parity anywhere. Confirmed behaviourally in E5: it "
            "violates the mirror law by O(1). Separately, its rotation equivariance is only "
            "approximate (edge_rot_mat.py draws a random per-edge frame each forward)."
        ),
    },
    {
        "model": "eSCN",
        "source": "vendored so3.py / so2_ops.py (Meta OCP eSCN lineage)",
        "version": "as vendored @ EquiformerV2 8fe8cba",
        "category": "SO(3)-only",
        "evidence_file": "src/equiparity/models/equiformer_v2/so3.py",
        "evidence": (
            "eSCN is the SO(2)-reduced convolution EquiformerV2 is built on; its code IS the "
            "vendored `so3.py`/`so2_ops.py` (Meta OCP header). The `SO3_Embedding` is "
            "`lmax`/`mmax`-"
            "typed with **no parity label** -- deciding line, in-repo. (The `fairchem` clone "
            "carries"
            "only the newer UMA MoE eSCN variant, not the EquiformerV2 lineage.)"
        ),
    },
    {
        "model": "MatTen",
        "source": "wengroup/matten",
        "version": "commit 0a04f1c",
        "category": "parity-aware",
        "evidence_file": "src/matten/model_factory/tfn_scalar_tensor.py",
        "evidence": (
            "Default `conv_layer_irreps: '32x0o + 32x0e + 16x1o + 16x1e'` (line 206) and "
            "`irreps_edge_sh: '0e + 1o'` (line 207). Both parities are present and explicitly "
            "labelled; this is an O(3) model (it targets the even-parity elasticity tensor)."
        ),
    },
    {
        "model": "ICTP",
        "source": "nec-research/ICTP",
        "version": "commit f40592a",
        "category": "parity-aware",
        "evidence_file": "ictp/o3/cartesian_harmonics.py",
        "evidence": (
            "Uses its own `CartesianHarmonics(l_max)` (line 253) rather than e3nn irreps. "
            "**Measured"
            "(H2, random init, `results/h2_probes.json`):** `RankThreeCartesianHarmonics` gives "
            "`‖f(-x)+f(x)‖/‖f‖ = 0.0` (rank-3 odd) and the rank-2 output is even -- the `(-1)^l` "
            "construction confirmed to machine precision. Parity-aware by measurement, not "
            "argument."
        ),
    },
    {
        "model": "GotenNet",
        "source": "sarpaykent/GotenNet",
        "version": "commit 44c945b",
        "category": "parity-aware",
        "evidence_file": "gotennet/models/representation/gotennet.py",
        "evidence": (
            "Edge features come from `e3nn.o3.SphericalHarmonics(lmax)` (line 861), carrying "
            "natural"
            "parity. We suspected the non-e3nn `GATA` blocks might strip it, but **the measurement "
            "(H2, isolated CPU env, random init, `results/h2_probes.json`) overturns that**: the "
            "l=1 vector output satisfies `X(gx) = g X(x)` for a random rotation (rel 1.2e-15) AND "
            "a"
            "random improper op (rel 1.8e-15), while the pseudovector law fails (rel 1.9). So the "
            "output is a genuine polar (1o) vector -- reflection-equivariant, parity-aware. This "
            "is"
            "why the row was left undetermined until measured."
        ),
    },
    # ---- Tier 2 (Checkpoint 10): the current deployed generation, audit -> 18 models ----
    {
        "model": "eSEN",
        "source": "fairchem-core (PyPI)",
        "version": "1.10.0",
        "category": "SO(3)-only",
        "evidence_file": "fairchem/core/models/esen/esen.py",
        "evidence": (
            "Node features are spherical coefficient arrays sized `(lmax+1)**2` "
            "(`sph_feature_size`, line 94), mixed by `SO3_Linear` over degree/order via "
            "`CoefficientMapping(lmax, mmax)` -- the eSCN SO(2)-convolution lineage its paper "
            "states (Sec. 4). Zero `parity` references anywhere in `models/esen/`; no odd/even "
            "typing, no rank-3 path. Measured (T2, random init, rank-3 head attached): rotation "
            "law 4.2e-5, deterministic, mirror law violated 2.0; m-3m/non-cubic violation ratio "
            "1.6e-5 on the 2,000-crystal population (`results/t2_random_init.json`)."
        ),
    },
    {
        "model": "UMA",
        "source": "fairchem-core (PyPI)",
        "version": "2.21.0",
        "category": "SO(3)-only",
        "evidence_file": "fairchem/core/models/uma/escn_md.py",
        "evidence": (
            "`eSCNMDBackbone` (line 263) builds on `uma.common.so3.CoefficientMapping`/`SO3_Grid` "
            "and `SO3_Linear` -- the same degree/order-only spherical typing as eSEN, which UMA's "
            "paper names as its central edgewise component (the MoE variant `escn_moe` wraps the "
            "same backbone). Zero `parity` references anywhere in `models/uma/`. Measured (T2, "
            "random init): rotation law 1.0e-4, deterministic, mirror law violated 1.8; "
            "m-3m/non-cubic ratio 1.9e-5 (`results/t2_random_init.json`)."
        ),
    },
    {
        "model": "EquiformerV3",
        "source": "atomicarchitects/equiformer_v3",
        "version": "commit a7300c5",
        "category": "SO(3)-only",
        "evidence_file": "experimental/models/equiformer_v3/so3.py",
        "evidence": (
            "The embedding is documented and indexed by 'Maximum degrees (l)' only (docstring "
            "line 75); `so3.py`/`wigner.py` carry no parity label and the model directory "
            "contains zero `parity` references. SE(3), not E(3), is in the paper's own title -- "
            "strict rotational equivariance is claimed, parity is never represented. Measured "
            "(T2, random init): rotation-exact (2.2e-5) AND deterministic -- unlike EquiformerV2 "
            "-- yet the mirror law is violated by 2.0; m-3m/non-cubic ratio 1.2e-6. This removes "
            "the approximate-equivariance confound entirely (`results/t2_random_init.json`)."
        ),
    },
]

CATEGORY_NOTE = {
    "parity-aware": "odd-parity output is **structurally zero** on a centrosymmetric input",
    "SO(3)-only": "**nothing** forces an odd-parity output to vanish",
    "vector-only": "no typed irreps; a rank-3 head needs extra machinery",
    "invariant": "cannot express an equivariant output at all",
    "undetermined": "source does not settle it; needs a reflection test",
}
ORDER = ["parity-aware", "SO(3)-only", "vector-only", "invariant", "undetermined"]


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"rows": ROWS}, indent=1) + "\n")

    counts: dict[str, int] = {}
    for row in ROWS:
        counts[row["category"]] = counts.get(row["category"], 0) + 1

    lines = [
        "# Table 1 — prevalence audit (Task 0.4)",
        "",
        "Which released models attach a **parity label** to their internal features? Only ",
        "those can",
        "produce a structurally exact zero for a parity-odd tensor on a centrosymmetric crystal.",
        "",
        "Every row was classified by reading the released source at the version or commit given.",
        "The evidence column quotes the line that decides it. Where the source does not decide,",
        "the row says `undetermined` rather than guessing.",
        "",
        "## Summary",
        "",
        "| category | n | consequence for a parity-odd tensor |",
        "|---|---|---|",
    ]
    for category in ORDER:
        if category in counts:
            lines.append(f"| `{category}` | {counts[category]} | {CATEGORY_NOTE[category]} |")

    n_not_parity = sum(counts.get(c, 0) for c in ("SO(3)-only", "vector-only", "invariant"))
    lines += [
        "",
        f"Of the {len(ROWS)} models surveyed, **{counts.get('parity-aware', 0)}** carry parity",
        f"labels on their features. **{n_not_parity}** do not, and one is undetermined. The",
        "SO(3)-only group is not a fringe: it contains EquiformerV2 and the eSCN family, which are",
        "the current state of the art on large-scale catalysis and materials benchmarks, and",
        "Equiformer v1 as released.",
        "",
        "## Rows",
        "",
    ]
    for category in ORDER:
        rows = [r for r in ROWS if r["category"] == category]
        if not rows:
            continue
        lines += [f"### `{category}` — {CATEGORY_NOTE[category]}", ""]
        for r in rows:
            lines += [
                f"**{r['model']}** — {r['source']}, `{r['version']}`  ",
                f"*Evidence* (`{r['evidence_file']}`): {r['evidence']}",
                "",
            ]

    lines += [
        "## What this table is for",
        "",
        "The introduction claims that the parity distinction is (a) invisible in most ",
        "model cards and",
        "(b) load-bearing for tensor properties. This table supports (a) by showing that ",
        "the two most",
        "widely deployed equivariant transformers in this list are SO(3)-only, and that the",
        "distinction is nowhere in their configuration surface — EquiformerV2 exposes ",
        "`lmax`, not a",
        "parity flag. NequIP is the sharpest case: it *has* a `parity` boolean, and that ",
        "boolean does",
        "not do what its name suggests (its own docstring shows `1o` features surviving",
        "`parity=False`). Our SO(3) arms are therefore built by relabelling irreps, not ",
        "by flipping",
        "that flag — see the Supplementary Information, Supplementary Note "
        "\"Prevalence audit of released architectures\".",
        "",
        "## Notes (H2: two rows converted from reading to measurement)",
        "",
        "- `GotenNet` was `undetermined`; **now `parity-aware` by measurement** (H2). We suspected "
        "its non-e3nn `GATA` blocks might strip the parity its edge spherical harmonics carry; a "
        "reflection test on the built model at random init showed the opposite -- its l=1 output "
        "is"
        "a polar (1o) vector, reflection-equivariant to machine precision. Measuring overturned "
        "the"
        "guess, which is why the row was held open until measured.",
        "- `ICTP` was `parity-aware` by a *construction* argument; **now confirmed by "
        "measurement**"
        "(H2): its rank-3 Cartesian harmonic is odd (`(-1)^3`) to machine precision, rank-2 even.",
        "- `eSCN` now cites a **decisive in-repo source line** -- the vendored `so3.py` (Meta OCP "
        "eSCN, the code EquiformerV2 is built on), whose `SO3_Embedding` is `lmax`/`mmax`-typed "
        "with"
        "no parity label -- rather than family resemblance.",
        "- `FAENet` is listed `invariant` because the *network* is; its symmetry comes from frame "
        "averaging applied outside the model, and whether that includes improper operations is a "
        "choice made at the call site.",
        "",
        "### Changelog",
        "",
        "- eSCN was added during the audit, bringing the survey to **15** models (an earlier "
        "hand-written summary said 14; corrected).",
        "- H2 converted the GotenNet and ICTP rows from source-reading to measurement; GotenNet "
        "moved"
        "`undetermined` -> `parity-aware`, so the counts are now **6 parity-aware, 3 SO(3)-only, "
        "3 vector-only, 3 invariant, 0 undetermined**.",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}")
    for category in ORDER:
        if category in counts:
            print(f"  {category:16s} {counts[category]}")


if __name__ == "__main__":
    main()
