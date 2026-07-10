# Decision: Post-Checkpoint-7 strengthening package — APPROVED SCOPE, execute before writing

Headline results are locked; nothing in this package retrains existing arms. It adds three
experiments (E1–E3), two rebuttal checks (E4–E5), figure assets (E6), and framing work (F1–F2).
No seed-count increases. E1 is the only item involving training; E2–E6 are inference-only and can
run in parallel with it. The off-cycle standing rule applies to every Expected statement below.

## V0 — Validate the advisor's claims before implementing anything (MANDATORY, FIRST)
Do not take the claims in this package on trust. Two have been pre-validated by the advisor with
toy reproductions; you must reproduce both independently in the project environment and extend
them, and you must validate every remaining physical/mathematical claim you rely on as you go.
1. Jacobian parity claim (basis of E3): build a small O(3)-equivariant net with parity-odd output
   irreps (e.g., 2x1o+1x2o+1x3o) on an exactly centrosymmetric point cloud (points in +/- pairs).
   Verify: (a) output is machine-zero at the centrosymmetric configuration; (b) all leading
   singular vectors of the autograd Jacobian dT/dr have parity score -1 (definition in E3).
   Repeat with the all-even SO(3) construction and verify nonzero output and mixed-parity
   singular vectors. Advisor's reference values: O(3) |T| ~ 1e-17, scores -1.000 (top 5);
   SO(3) |T| ~ 2e-1, scores ranging roughly +1.0 to -1.0. If your reproduction disagrees, STOP
   and file an off-cycle report before touching E3.
2. BaTiO3 protocol claim (basis of E2): cubic perovskite BaTiO3 (a = 4.00 Å; Ba(0,0,0),
   Ti(.5,.5,.5), O at the three face centers) must register in spglib as Pm-3m (221) at
   symprec 1e-3 AND 1e-5; the [001] polar distortion (Ti +z, O -z; advisor used fractional
   displacements Ti +0.030δ, O -0.024δ/-0.010δ/-0.010δ) must register as P4mm (99) for all
   δ > 0 tested. Reproduce before building the sweep. Cross-check the displacement pattern and
   lattice constants against the literature values for tetragonal BaTiO3 and record the source.
3. Standing requirement: every physics claim this package asks you to encode (Curie's principle
   phrasing, parity of the piezoelectric irreps, the symmetrization identity in E4) must be
   verified by a test in the repository, not asserted. Extend the CI gate where a claim is
   testable. Your Checkpoint-8 report must contain a "claims validated" section listing each
   claim, the test that checks it, and the result.

## E1 — Augmentation rebuttal with unseen-space-group split (the reviewer-killer)
Why: the strongest objection to the headline is "just train SO(3) with centrosymmetric examples
labeled zero." This experiment answers it with evidence instead of argument, and the split design
is what makes it decisive: learned zeros that only work on seen symmetry classes are not a fix.
- Build the augmented training set: existing piezo training split + ~1,000 centrosymmetric
  insulators labeled with exact-zero tensors, drawn ONLY from a restricted list of centrosymmetric
  space groups. Choose the list so that both halves below are well-populated; record it in the
  manifest.
- Two disjoint evaluation sets, both spglib-verified at symprec 1e-3, idealized variant:
  (a) SEEN-SG: held-out centrosymmetric crystals from the augmentation space groups;
  (b) UNSEEN-SG: centrosymmetric crystals exclusively from space groups absent from training.
- Retrain SO(3) arms only (NequIP, Allegro, MACE, EquiformerV2), 3 seeds, hyperparameters
  identical to the main runs. Do NOT retrain O(3) arms — their zeros are structural and hold
  regardless of training data; cite the existing runs and say exactly that in the paper.
- Report: false-flag at 0.01 C/m² and violation medians on SEEN-SG vs UNSEEN-SG, and the
  non-centrosymmetric test MAE (does augmentation degrade regression quality?).
- Expected: SEEN-SG false-flags drop substantially; UNSEEN-SG remains materially higher. Either
  outcome supports the thesis: full transfer means "the fix requires curated data and still gives
  no guarantee"; failed transfer means "learned zeros do not generalize across symmetry classes."
  A result outside both (e.g., no drop even on SEEN-SG) triggers the standing rule.

## E2 — BaTiO3 symmetry-breaking curve (new main-text figure)
Why: this upgrades the story from "O(3) predicts zero" to "O(3) tracks the physics of symmetry
breaking" — the guarantee turns off exactly when the symmetry does, on the textbook displacive
ferroelectric every materials reader knows. It also connects to the A5 finding (O(3) models
detected genuine residual asymmetry in raw structures): same phenomenon, now shown continuously.
- Use the V0-validated construction. Parameterize x(δ) = x_cubic + δ·Δx for δ in [0, 1.2] with
  at least 25 steps; δ=1 is the physical tetragonal amplitude. spglib-verify every frame: δ=0
  centrosymmetric, δ>0 polar. Store the verification alongside the sweep data.
- Run ALL trained piezo models (every core, both arms where applicable, all seeds) at every δ.
  Plot ||T||_F vs δ, log-y, one panel per core, O(3) and SO(3) arms overlaid.
- Expected: O(3) curves start at machine zero and rise smoothly and monotonically for small δ;
  SO(3) curves start from an O(1) spurious offset that dwarfs the physical signal at small δ.
- Supplementary panel: repeat with a second material (PbTiO3, same protocol) to show the curve
  shape is not BaTiO3-specific.

## E3 — Jacobian order-parameter analysis (the novelty add)
Why: no one has shown the parity guarantee is differentiable. The O(3) model's Jacobian at a
centrosymmetric structure is a learned map of which atomic displacements activate piezoelectric
response — and V0 item 1 shows its singular vectors are purely inversion-odd (polar-mode-like
order parameters). This connects the paper to the Smidt-group line on equivariant networks as
symmetry-breaking order-parameter detectors and elevates the claim from "predicts zero" to
"the architectural guarantee has physically structured derivatives."
- For ~20 centrosymmetric OOD structures spanning several space groups (include cubic BaTiO3):
  compute J = dT/dr at the centrosymmetric geometry via autograd (T as its 18 components; r as
  all atomic Cartesian coordinates). SVD of J.
- Parity score: for each leading right-singular vector u (a displacement pattern), form its
  inversion image u' by permuting atoms with the structure's inversion operation and flipping
  displacement signs (u'_i = -u_{sigma(i)}); score s = <u,u'>/(|u||u'|). s = -1 is purely
  inversion-odd; s = +1 purely even. Use the V0 toy as the unit test for this scorer.
- Report: distribution of parity scores of the top-5 singular vectors per structure, O(3) vs
  SO(3) arms, per core.
- Methods paragraph (state, then verify numerically): T is parity-odd and vanishes identically on
  the centrosymmetric manifold, so its differential at a centrosymmetric point can be nonzero
  only along inversion-odd displacement directions. The O(3) scores must equal -1 to numerical
  precision — treat any deviation beyond the gate thresholds as a bug, not a result.
- Expected: O(3) scores -1.000 across the board; SO(3) scores broadly distributed.

## E4 — Test-time inversion-averaging rebuttal (supplementary table)
Why: the second reviewer escape hatch is "symmetrize SO(3) outputs at test time." Show what that
buys and what it costs.
- Odd projection: T_sym(x) = [T(x) - T(I·x)]/2, where I·x inverts the structure about its
  inversion center (verify the identity on an O(3) arm first: its predictions must already
  satisfy T(I·x) = -T(x), making T_sym = T; this doubles as a correctness check).
- Evaluate symmetrized SO(3) predictions on idealized AND raw OOD variants.
- Expected: exact zeros on idealized (trivially — say so), near-zeros on raw. The table's caption
  carries the argument: the fix (a) doubles inference cost, (b) presupposes knowing the target's
  parity in advance — exactly the knowledge O(3) features encode — and (c) repairs one output
  while leaving the model's internal representations parity-blind for every other quantity.

## E5 — Output-level parity check (owed from Checkpoint 1; closes the verification chain)
- For every trained piezo model and ~50 structures (mixed centro/non-centro): predict T(x) and
  T(Mx) for a mirror M. O(3) arms must reproduce the exact parity-transformed tensor within gate
  precision (float32 thresholds for MACE); SO(3) arms and EquiformerV2 must violate it — this is
  the deferred EquiformerV2 positive control. Wire into the CI gate report.

## E6 — Named materials for the headline figure
- From the SO(3) OOD predictions, shortlist ~8 familiar centrosymmetric materials (prefer
  rocksalt / perovskite / diamond-structure compounds a general reader recognizes) with mp-ids,
  structure type, and predicted ||T||_F per core. Final 3 chosen at review. A named familiar
  material with an impossible predicted response is the abstract's most memorable sentence.

## F1 — Reframing (writing, no compute)
- Introduction: position against the symmetry-breaking-in-equivariant-networks literature
  (Kaba & Ravanbakhsh 2023; Smidt-group equivariant symmetry-breaking sets; probabilistic
  symmetry breaking 2025). That thread relaxes equivariance so outputs CAN break input symmetry;
  we show the physics-side complement — for tensor properties, preserving improper symmetry is
  Curie's principle, and SO(3) models violate it structurally. Verify each characterization
  against the cited papers' abstracts before writing (V0 standing requirement applies).
- Motivation: tensor-property readouts are increasingly built on universal-potential embeddings;
  the backbone's symmetry group silently decides whether physically impossible predictions are
  possible. Cite at least one such readout-on-embeddings work.
- One sentence distinguishing our feature-level diagnosis from readout-level fixes
  (canonicalization / space-group conditioning: GMTNet, GoeCTP lineage). No new baselines.

## F2 — Writing-phase corrections (carry from the Checkpoint-7 review)
- All summary tables: mean ± std over seeds. U0 control wording: "no consistent direction across
  cores"; do not claim a tested null — seed variance is large.
- Fig 2 reframe: the gap appears where parity-odd targets meet inversion-constrained structure;
  the molecular dipole null is supporting evidence for this, not a miss.
- Size-dependence caveat: ||T||_F correlates with n_atoms (rho ~ 0.5–0.7). Report a per-atom
  variant of the headline metric in supplementary and confirm conclusions are unchanged.
- Do NOT claim SO(3) is cheaper for e3nn cores (parameter counts here say otherwise); the
  efficiency motivation belongs to eSCN-style architectures only.
- A5's mp-1227949 gets its own paragraph: the O(3) "false flag" on raw coordinates was a
  genuinely symmetry-broken structure — the guarantee detected a data-pipeline artifact.
- Confirm the Task 0.4 prevalence audit table exists with commit hashes; it is Table 1 and
  blocks the introduction if missing.

## Not approved
- No seed-count increases. No new toggleable cores. No canonicalization or space-group-
  conditioning baselines. No CliffordSTF revival.

## Deliverable
- One consolidated Checkpoint-8 report (standard template) containing: the V0 claims-validated
  section, E1 tables, E2/E3 draft figures, E4/E5 tables, E6 shortlist, F2 confirmations.
  Writing starts on approval of that report.