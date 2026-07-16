# Decision: Checkpoint 10 — final pre-submission experiment package (verified specifications)

Every factual premise below has been verified by the advisor against primary sources (papers,
repos, official docs) over five searches and three fetches; corrections to the original audit are
baked in. V0 discipline applies throughout: every prediction stated here is declared BEFORE its
measurement, and every claim that enters the paper gets a test or a transcription from a primary
source — never a number recalled from an audit document. The off-cycle standing rule applies to
every Expected statement. Manuscript red-flag sections stay red until their experiment lands; no
submission until all flags clear.

## Tier 0 — immediate checks (run first, all cheap)

### T0-1. Contamination check (was B-4)
- Intersect the 2,000 OOD evaluation identifiers with the piezoelectric train/val/test splits
  (2,649/331/332). Report the overlap in Methods either way. If nonzero, additionally report all
  headline false-flag fractions with the overlapping crystals removed.
- Bonus check: the training split contains 16 exactly-zero rows. Look up their point groups.
  Prediction (declared): they are 432-class or similar rotation-forbidden non-centrosymmetric
  crystals — the E7 mechanism appearing in the source data. Report what you find.

### T0-2. False-flag vs training epoch (was H-1)
- From already-logged metrics: plot SO(3)-arm false-flag fraction against epoch for all piezo
  runs. Expected: plateau near 0.90 well before epoch 150. If still falling at 150, off-cycle
  report — the undertrained objection would then have teeth.

### T0-3. Bootstrap CIs (was H-5)
- Percentile bootstrap over the 2,000 structures (parity_metrics.py::bootstrap_ci) for every
  false-flag fraction in the paper. Add to all tables.

## Tier 1 — B-1: the SOTA crystal-tensor predictors (BLOCKING)

Verified context you must use:
- The benchmark family (GMTNet, GoeCTP, CEITNet) shares the ComformerConv INVARIANT backbone;
  equivariance is synthesized only at the tensor head. None carries equivariant features.
- The benchmark protocol REMOVES zero tensors from the piezo dataset ("a simple model that always
  predicts zero could achieve near-SOTA results"), and EwT = Fnorm(err)/Fnorm(label) is undefined
  at zero labels. The field's standard evaluation cannot express our test. This goes in the
  introduction — it is the offensive framing of B-1, not a side note.
- Code: GMTNet in github.com/divelab/AIRS (verified). GoeCTP and EATGNN have released code
  (verified via the field's index + EATGNN's own repo). CEITNet code NOT found — contingent; if
  unavailable at execution time, proceed with three and state it.
- Naming: the original paper spells it GoeCTP (the CEITNet paper writes "GeoCTP") — cite per the
  original authors.
- Benchmark digits: Table 5 (piezo Fnorm/EwT) numbers must be TRANSCRIBED from the arXiv PDF of
  2602.04323 before quoting anywhere; three fetch attempts truncated at that table. Attribute
  every baseline number to the specific paper reporting it — the same baseline differs across
  papers (verified: dielectric ETGNN is 3.92 in CEITNet's table vs 3.40 in GoeCTP's).

The experiment:
1. Declare per-model predictions FIRST, in the report, from head algebra (advisor's derivations —
   verify each before relying on it):
   - CEITNet (if runnable) / EATGNN: heads assemble odd-parity Cartesian/irrep bases with
     invariant weights and invariant pooling → predicted STRUCTURALLY ZERO on exactly
     centrosymmetric inputs, no mask needed. Verify the derivation on the built model with a toy
     centrosymmetric input before the full run.
   - GMTNet: passes via space-group zero-mask lookup when the mask is on.
   - GoeCTP: the open case. Its R&R canonicalization module (polar decomposition, includes
     reflections) claims O(3) equivariance, but canonicalization is provably discontinuous at
     high-symmetry inputs — exactly our evaluation set. No prediction declared; this is the
     genuine measurement.
2. Run each available model on the 2,000 centrosymmetric crystals (idealized AND raw variants),
   WITH and WITHOUT any symmetry-adjustment/mask module. Released checkpoints if they exist
   (JARVIS-trained is fine — the test is label-free); otherwise retrain on their published split
   per their published settings.
3. Report per model: false-flag fraction @ 0.01 C/m² + threshold curve, violation medians, both
   mask states, both variants.
4. Jacobian even-subspace fraction per model ONLY IF autograd validates against finite
   differences first (the EquiformerV2 lesson — run the FD check per model; exclude with cause
   if it fails, as before).
5. Also run PCGNN (Nano Letters 2026; symmetry-sparsity-without-constraints claim, MP test set,
   beats EATGNN) IF its code is available — its learned-compliance claim is precisely what the
   OOD test falsifies. If no code, cite it in the Discussion as the claim our result bounds.
6. Note for Discussion (verified quote to paraphrase): EATGNN's own repository states the plan to
   replace SO(3) tensor products with SO(2) operations for speed — the efficiency migration
   arriving at a parity-odd tensor model. Cite the repo at a pinned commit.

Framing rule regardless of outcome: heterogeneous results classified in advance by head algebra,
then confirmed, is the result. If all pass the output test, the contribution renarrows honestly
to features (Jacobian) + the benchmark-blindness finding + raw-structure behavior. If any fail,
that extends the headline. Both are publishable; say which one happened.

## Tier 2 — B-2: current-generation deployed models (BLOCKING)

Verified context: eSEN uses eSCN's SO2 convolutions (its paper, §4); UMA is built on eSEN with
the eSCN convolution as its central edgewise component (its paper); both have released fairchem/
HuggingFace checkpoints. EquiformerV3 = arXiv 2604.09130, SE(3) in its own title, claims strict
equivariance — if rotation-exact but parity-blind, it removes the "EquiformerV2 is just
approximately equivariant" confound entirely.

1. Source-read + reflection-probe eSEN, UMA, EquiformerV3 at pinned commits → three new
   prevalence-audit rows (audit goes to 18 architectures).
2. Random-init structural test (no training, per Theorem 1): attach the standard rank-3 head to
   each backbone, evaluate on the 2,000 centrosymmetric crystals, report violation distributions.
   Load released checkpoints where head attachment permits; random init is sufficient for the
   structural claim.
3. EquiformerV3 additionally gets the E5 rotation/mirror/determinism checks — declare the
   prediction first: rotation error near machine precision (their strict-equivariance claim),
   mirror violation O(1).

## Tier 3 — B-3: the learning curve (BLOCKING for the title)

- N_zero ∈ {0, 250, 1000, 4000, 16000} zero-labelled centrosymmetric crystals in training;
  NequIP SO(3) arm; 3 seeds each (reuse N=0 and N=1000 runs). First verify ~16,000 qualifying
  centrosymmetric insulators exist in MP after filters; cap at availability and record the count.
- Plot vs N: false-flag on trained-on zeros, false-flag on held-out (SEEN-SG and UNSEEN-SG),
  violation median, non-centro test MAE.
- Title rule: "Not all symmetry can be learned" was never approved and survives ONLY if the
  false-flag curve plateaus well above zero. If it plummets, revert to the approved
  "Rotation is not enough…" and reframe E1 as cost-of-data. The curve decides; report it neutrally.

## Tier 4 — high-value additions

### T4-1. Frozen pretrained backbones (was H-3; dovetails with Tier 2 installs)
- Fine-tune the piezo head with frozen backbones: MACE-MP-0 (parity-aware) vs eSEN or UMA
  (parity-blind), matched protocol, 3 seeds. Report false-flag fractions on the OOD set.
  This is the practitioner scenario the Discussion promises; it must be measured, not asserted.

### T4-2. Fnorm/EwT conversion (was H-2; after Tier 1 transcription)
- Convert the matched cores' piezo predictions to Fnorm and EwT on our split and state where they
  land relative to the transcribed benchmark numbers. In Methods, not a rebuttal.

### T4-3. Additional distortion materials (was H-6, corrected)
- Two more polar paths whose PARENT rotation subgroup permits a rank-3 invariant (the 432 lesson):
  rutile-type SnO2 or GeO2, and anatase TiO2. spglib-verify every frame at symprec 1e-8, as in E2.

## Owner tasks (not the agent's)
- B-5: independent human re-derivation and sign-off of Supplementary Note 1 proofs.
- B-6: independent human mock reviews.
- Methods AI-use disclosure per Nature Portfolio policy; funding/acknowledgements; data/code DOIs
  (Zenodo), licence, suggested reviewers, preprint status — the back-matter red flags.

## Deliverable
One consolidated Checkpoint-10 report: T0 results, Tier-1 per-model tables with declared-vs-
measured predictions and the Table 5 transcription, Tier-2 audit rows + structural-test tables,
Tier-3 learning curve + the title verdict, Tier-4 tables, and the updated red-flag ledger showing
which flags each result clears. All numbers backed by results/ JSON with source attribution.
Submission is gated on this report plus the owner tasks.