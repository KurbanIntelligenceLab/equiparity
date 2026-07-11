# Checkpoint 9 — hardening package (H1–H3)

Three hardening items completed before the writing phase. No headline arm was retrained;
`results/stats.json` is byte-identical throughout. Every number below is backed by a `results/`
JSON and was cross-checked before commit. External source (H1/H2) lives in git-ignored
`third_party/` at pinned shas recorded here.

## H1 — EquiformerV2 findings reproduced against pinned upstream

Full report: [`checkpoint9_h1_equiformer_upstream.md`](checkpoint9_h1_equiformer_upstream.md).

The public claims about EquiformerV2 (random per-edge frame ⇒ nondeterministic eval; detached
Wigner-D ⇒ incomplete autograd) are shown to be properties of the released code, not a vendoring
artifact. Pinned to `atomicarchitects/equiformer_v2` @ **`8fe8cba`** (the repo's only commit to
`nets/equiformer_v2/`; frozen since 2023-06-28). Diff manifest: 13 model files, all semantically
upstream (0 substantive), including the two load-bearing ones — `edge_rot_mat.py` (random frame) and
`so3.py` (Wigner detach) — AST-identical.

Per the ruling, the one non-`nets/` method (`generate_graph`, inherited from the OCP `BaseModel`) was
relocated into an auditable shim subclass, and Condition 2 was proven **by measurement** on CPU:

- the shimmed-upstream model reproduces the shipped vendored model **bit-for-bit** (`max|Δ| = 0.0`);
- the shim `generate_graph` is **bit-identical** to a hard stub (`max|Δ| = 0.0`) — it cannot affect
  any measured quantity.

Reruns on upstream source (trained checkpoints) reproduce the committed vendored E5 numbers to three
decimals:

| seed | mirror | rotation | determinism | E3 FD rel. err |
|---|---|---|---|---|
| 0 | 0.982 | 0.073 | 0.111 | 0.36 |
| 1 | 0.929 | 0.094 | 0.143 | 0.84 |
| 2 | 0.713 | 0.106 | 0.144 | 0.66 |

The E3 FD-vs-autograd disagreement (0.36–0.84) confirms the incomplete autograd; the exclusion now
cites upstream at `8fe8cba`. Doc-check: the eSCN paper calls the SO(2) reduction mathematically
exact, while upstream issue #17 reports the random-frame equivariance breaking. Framed strictly as
measured properties of the released code, no defect claim. **No OCP install was needed.**

## H2 — two audit rows converted from reading to measurement (Table 1: 14 → 15)

Full evidence: [`checkpoint8_prevalence_audit.md`](checkpoint8_prevalence_audit.md);
`results/h2_probes.json`. Both probed at random initialisation (parity is architectural).

- **ICTP** (@ `f40592a`): `RankThreeCartesianHarmonics` gives `‖f(−x)+f(x)‖/‖f‖ = 0.0` (rank-3 odd),
  rank-2 even — the `(-1)^l` construction confirmed to machine precision. **Parity-aware by
  measurement**, and a self-contained rank-3 odd-parity assertion added to the CI test suite.
- **GotenNet** (@ `44c945b`) — **the finding that overturned a guess.** Built an isolated CPU venv
  (it needs torch 2.5.1 + PyG compiled extensions, which won't co-install with torch 2.11). We
  suspected its non-e3nn `GATA` blocks strip parity; the reflection test says the opposite — its l=1
  output satisfies `X(gx) = g·X(x)` for a random rotation (**1.2e-15**) *and* a random improper
  operation (**1.8e-15**), failing the pseudovector law (1.9). It is a genuine polar (1o) vector:
  **reflection-equivariant, parity-aware.** Moved `undetermined → parity-aware`. This is exactly why
  the plan held the row open until measured.
- **eSCN**: deciding line now cites the in-repo vendored `so3.py` (Meta OCP eSCN, the code
  EquiformerV2 is built on).

New counts: **6 parity-aware, 3 SO(3)-only, 3 vector-only, 3 invariant, 0 undetermined = 15**. The
stale "14" corrected in `RESULTS.md` and `checkpoint8_report.md`; changelog note in the audit doc.

## H3 — the loss-weight sweep (preempts the first referee question)

Full report: [`../results/h3_loss_weight.md`](../results/h3_loss_weight.md); `results/h3_loss_weight.json`.

"Can you force the zero with a big enough loss weight?" — answered on E1's own design. NequIP SO(3)
on the augmented set, the 1,016 exactly-zero-target rows up-weighted by W in the MSE (W=1 reuses the
committed E1 run; W=10, 100 are 6 new local runs). Mean over 3 seeds:

| W | ff on trained zeros | ff SEEN-SG | ff UNSEEN-SG | test MAE |
|---|---|---|---|---|
| 1 | 0.895 | 0.904 | 0.868 | 0.2278 |
| 10 | 0.873 | 0.903 | 0.842 | 0.2180 |
| 100 | 0.676 | 0.871 | 0.760 | 0.2142 |

**Extreme reweighting does not buy the zero.** Even at 100×, the model still false-flags ~68% of the
crystals it was *trained to call zero* and ~87% of held-out seen ones — far above O(3)'s 0.0000. The
median violation shrinks ~9× but never reaches zero, and non-centrosymmetric test MAE is not harmed
(it slightly improves). Gradient descent pushes towards zero and cannot arrive; only the O(3)
structure delivers exact zero, for free, at any weight. No weight achieved ~0.00 false-flag with
intact MAE, so no off-cycle report — the E1 conclusion stands, sharpened.

## Status

| item | state |
|---|---|
| H1 | complete; bit-identity proofs + upstream reruns match vendored E5 exactly; no OCP install |
| H2 | complete; GotenNet & ICTP measured; count 14→15; GotenNet overturned to parity-aware |
| H3 | complete; 6 local runs; even 100× weighting does not remove the impossible predictions |
| Headline `results/stats.json` | byte-identical (`0e2f0c94…`) |
| Tests / lint / mypy | green |
| Writing phase | opens on review of this package |
