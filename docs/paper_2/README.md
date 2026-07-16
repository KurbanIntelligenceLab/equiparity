# Submission bundle — "Not all symmetry can be learned"

Target: **Nature Communications** (Article). Finalization pass completed 2026-07-14.

---


## The title

**"Not all symmetry can be learned: parity-blind equivariant networks predict physically
impossible material properties"** (14 words; NC's limit is 15).

The old title, *"Rotation is not enough: parity-blind equivariant networks predict physically
impossible material properties"*, sold the smallest of the three audiences that will cite this
paper.

| Audience | What they will cite you for | Old title | New title |
|---|---|---|---|
| **The hard-wire-vs-learn debate** (largest) | *"symmetry constraints that are exact are not acquired by training [X]"* | silent | **leads with it** |
| **Architecture authors** | *"models described as E(3)-equivariant are often only SO(3)-equivariant [X]"* | implicit | implicit |
| **Materials-ML tensor prediction** (smallest) | *"parity-blind models predict nonzero piezoelectric tensors for centrosymmetric crystals [X]"* | yes | yes |

Every high-citation title in this literature either **names the debate** (*Does equivariance
matter at scale?*) or states a **fact you can cite in one clause** (*E(3)-equivariant models
cannot learn chirality*; *Forces are not enough*; *The dark side of the forces*). The new title
does both.

It is also **honest**, which the tempting version was not. *"Exact symmetries cannot be learned"*
is an overclaim your own limitations paragraph forbids. *"Not **all** symmetry can be learned"*
is a **partial** negation, and it is exactly what you show: some symmetry can be learned (the
rotational case, which the cited literature establishes), and this one cannot, even when the
answer is handed to the model at a hundredfold loss weight.

A referee who reads the title will hunt for the sentence that licenses it. It is in the
Discussion, and it is findable:

> *Not all symmetry, then, can be learned from data. On the evidence here the dividing line is
> exactness: a constraint that permits a range of values can be approached by fitting, while one
> that permits a single value is not attained by it, even when that value is placed in the
> training set and weighted a hundredfold.*

Running title: **"Not all symmetry can be learned"**.

## The story, after the SOTA pass

The paper was framed as a narrow finding about parity labels. It is in fact a contribution
to the dispute the field is actively having, and it now says so.

**The dispute.** Should the symmetries of physics be hard-wired into the architecture, or
learned from data? The evidence for "learned" is real: augmentation closes the
data-efficiency gap given enough epochs (Brehmer et al. 2024), unconstrained potentials on
large corpora beat physically constrained ones on accuracy and speed (Bigi et al. 2026),
and approximate rotational equivariance costs almost nothing in bulk simulation (Langer et
al. 2024). Creed et al. (2026) list it among six open questions for the field.

**The gap.** That dispute has been conducted on one axis, rotations, and in one currency,
accuracy. This paper opens the reflection axis, where the currency is different.

**The precedent that makes it legible.** Non-conservative force models made exactly this
trade against energy conservation, on exactly this argument ("it can be learned"), and
broke geometry optimisation and molecular dynamics. That paper was an ICML 2025 Oral.
Parity is the same shape of finding against a different law, and the Introduction now says
so explicitly.

**The contribution to the dispute.** The augmentation experiment is a clean *negative*
result for "symmetry can be learned from data" in the regime where the symmetry is exact.
A thousand centrosymmetric crystals carrying the exact answer, with their loss weight
raised a hundredfold, move the false-flag fraction only from 0.90 to 0.68 on the very
crystals the model was shown. The general principle, now stated in the Discussion: where
the physics permits any value, learned symmetry is a reasonable engineering trade; where it
permits exactly one, it is no trade at all, because an approximate constraint delivers an
error the size of the property.

## What the paper claims, with the real numbers in

* O(3) arms return zero at the float32 noise floor (median 3e-7 to 3e-6 C/m²) on **all
  2,000** centrosymmetric crystals: not one of 18,000 structure-runs exceeds the threshold.
* SO(3) arms exceed it for **89.5 / 91.0 / 90.8%**, and unmodified EquiformerV2 for
  **95.7%**, at median magnitudes of 0.44 to 0.96 C/m² against a real-piezoelectric median
  of 0.51.
* Theory predicts a **ceiling of 91.7%** (only class m-3m is forced to zero without
  parity). The matched cores reach **97.6 / 99.2 / 99.0%** of it. EquiformerV2 *exceeds* it,
  because its rotational equivariance is only approximate.
* Zero-labelled retraining and 100x loss up-weighting reduce but never remove the
  violations. Test-time symmetrisation only appears to; Corollary 1 says why.
* Parity labelling costs nothing where the constraint is inactive and helps where it is
  active (3.4 to 4.6 pooled seed s.d. lower piezoelectric MAE).

## The bug I introduced and caught

Adding the proofs as Supplementary Note 1 shifted every other note by +1. The *table*
numbers were updated; nine of eleven *note* references in the Results were not. My first
checker passed them, because it only asked whether the number existed. `check_crossrefs.py`
now asks whether it points at the right note, by requiring a keyword agreed in advance
between the citing sentence and the target heading. Re-running it immediately caught a
second, independent error (accuracy tables cited as "Statistical analysis"). Both fixed;
all 50 references now resolve and are corroborated.

**Run `python check_crossrefs.py` after any renumbering.** LaTeX cannot catch these: the SI
is a separate PDF, so every reference to it is a hard-coded string.



## The theory

Seven statements, all in the main body with their assumptions, all proved in Supplementary
Note 1. They were not complete before; three things were wrong.

**The assumptions were not in the statements.** The old Proposition read "let f be exactly
O(3)-equivariant ... then f(x) = 0", but the two assumptions its proof actually uses,
permutation invariance and translation invariance, lived in the surrounding prose. A theorem
whose hypotheses sit outside it is not a theorem. (A1), (A2) and (A3) are now named in the
main body and invoked by every statement that uses them.

**A novel result was only in the appendix.** The Jacobian identity `J∘P = −J`, which is the
entire theoretical basis for Figure 5b, was proved in the SI and never stated in the main
text. It is now Corollary 2.

**"Machine zero" had no theorem behind it.** The paper asserted a "float32 noise floor"
throughout and noted that MACE's floor is ~8× the others'. That was an anecdote. It is now
**Proposition 1 (Arithmetic floor)**:

> ‖f(x)‖ ≤ (δ + η)/2, where δ is the defect in permutation/translation invariance at x and
> η the defect in O(3)-equivariance at Q = I.

Two lines to prove, three jobs: it recovers the structural zero at δ = η = 0; it proves the
residual floor is set by the *arithmetic*, not the symmetry; and for a parity-blind model
Lemma 1 forces η = 2‖f(x)‖, so the bound collapses to an identity and constrains nothing.
The guarantee and its stability are lost together.

### One engine, four consequences

Everything follows from one identity, now stated first and separately.

| | Statement | What it needs |
|---|---|---|
| **Lemma 1** | Inversion identity: `f(I·x) = f(x)` on a centrosymmetric crystal | (A1), (A2) only. **No parity.** |
| **Theorem 1** | Structural zero: `f(x) = 0` at any parameters | Lemma 1 + O(3)-equivariance |
| **Proposition 1** | Arithmetic floor: `‖f(x)‖ ≤ (δ+η)/2` | the two defects |
| **Corollary 1** | Test-time symmetrization is vacuous | **Lemma 1 alone** |
| **Corollary 2** | The guarantee is differentiable: `J P_even = 0` | Lemma 1 differentiated |
| **Corollary 3** | Rotation ceiling, with a stability clause | the rotation-subgroup analogue |

The paper previously stated three of these in prose, in three different sections, without
noticing they were the same identity.

**New: the stability clause of Corollary 3 turns the EquiformerV2 anomaly into a prediction.**
If (A1)–(A2) hold only to an absolute defect δ and (A3) to a relative defect ε < 1, then on a
crystal whose rotation subgroup admits no invariant, ‖f(x)‖ ≤ δ/(1−ε). The three matched SO(3)
cores have δ, ε at the arithmetic floor, the bound is ~1e-6, four orders below the threshold,
and they flag **none** of the 166 m-3m crystals in any seed. EquiformerV2 has both of order
1e-1, the bound is weak, and it flags **53.4%**. The theory now says why the ceiling is a
ceiling only for exactly equivariant models.

### What the theory does not say

It forbids; it does not compel. Theorem 1 says an O(3) model **must** predict zero. Corollary 3
says an SO(3) model **may** predict nonzero on 91.7% of the population, not that it will. That
it does, on 97.6 to 99.2% of the crystals where it is free to, is an **empirical finding**, not
a theorem. That paragraph is in the paper, in both the main body and the appendix, because a
referee will look for exactly this distinction and it is better to draw it first.

### Placement

7 statements in the main body (complete, with assumptions and proof sketches), 7 in the
appendix (notation table, evidence for each assumption, 6 full proofs, the crystallographic
classification, and the limits of the theory). **0 orphans in either direction**, and the
numbering is identical in the two separately compiled PDFs.

Both new bounds were verified numerically before being written in, and the Reynolds ranks
(432 → 0, 23 → 1, 422 → 1) recomputed from scratch. `parity_metrics.py` does all of it.

## Section-by-section readiness

Run `python check_submission.py`. Current state: **29 pass, 5 open (all author action), 0 fail.**

| Section | State |
|---|---|
| **Title page** | Ready. Title 12 words, abstract 187 (unreferenced), 2 corresponding authors, 4 affiliations. |
| **Introduction** | Ready. Repositioned against the hard-wire-vs-learn dispute; five-part contribution statement. |
| **Results** | Ready as prose, theory, structure and cross-references. The **numbers are yours to confirm** (red flags 10–15). |
| **Discussion** | Ready. Engages the counter-position, answers the pre-filter objection, adds the practitioner rule and the reporting recommendation. |
| **Methods** | Ready except the AI-use disclosure, which Nature Portfolio policy requires and only you can write (red flag 3). |
| **Back matter** | **This section was incomplete.** Acknowledgements did not exist; NC requires funding to be declared. Added, plus Additional information, plus the persistent DOIs NC requires for data and code. |
| **Supplementary** | Ready. 12 notes, 17 tables, 1 figure, all cross-references corroborated. |
| **Cover letter** | Rewritten. It claimed three contributions where the paper claims five, and asserted an MIT licence the paper flags as unverified. |
| **References** | 49 entries, all with titles. Three carry incomplete author lists, flagged, never invented. |
| **Figures/legends** | Ready. All under 350 words, title sentence first, replicate counts and box-plot conventions stated. |
| **Build** | 0 errors, 0 overfull boxes, 0 undefined citations, line numbers on, no ink outside the text block. |

### Two things about this pass worth knowing

**The red-flag switch.** Every unresolved item now goes through `\MISSING{}` / `\VERIFY{}`.
Set `\finaltrue` in `main.tex` (line 38), `sn-supplementary.tex` and `cover_letter.tex` and
every flag vanishes. That is deliberate: an item you forgot then shows up as a **hole in the
sentence**, not as red ink you have stopped seeing.

**Spelling was inconsistent and is now not.** The manuscript mixed 23 × "idealized" against
~30 `-ise` forms, "center" against "centre", "artifact" against "artefact", "license" against
"licence". All normalized to Oxford spelling (British `-our/-re/-ll` with `-ize` endings),
which is what Nature journals use, from a hand-checked word list. The pass over-applied once,
rewriting a **siunitx key** (`table-number-alignment=center`) and breaking the build; the
compile caught it and it is reverted.

## Where each claim lives

| Claim | Main text | Supplementary |
|---|---|---|
| Prevalence: 9 of 15 architectures cannot forbid it | Table 1 | Note 2, Table 1 |
| Why parity gives an exact zero (Proposition 1) | Results, theory subsection | Note 1 |
| Test-time symmetrisation is vacuous (Corollary 1) | Results, "Training cannot buy the zero" | Note 1, Note 9, Table 13 |
| The 91.7% rotation ceiling (Corollary 2) | Results, theory + mechanism | Note 1, Note 10, Table 15 |
| The Jacobian is odd (Corollary 3) | Results, Fig. 5b | Note 1 |
| Headline false-flag fractions | Fig. 2, Results | Notes 4-6, Tables 3-6 |
| Parity costs no accuracy | Fig. 3 | Note 7, Table 7 |
| Training cannot remove it | Fig. 4 | Note 8, Tables 10-12 |
| EquiformerV2 is reported as released | Discussion | Note 11, Tables 16-17 |

---

## How to build

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
pdflatex sn-supplementary.tex && bibtex sn-supplementary && \
  pdflatex sn-supplementary.tex && pdflatex sn-supplementary.tex
```

`main.bbl` and `sn-supplementary.bbl` are shipped, so a single `pdflatex` pass resolves
references even without BibTeX. The Supplementary Information is a **separate PDF**, the
standard Nature Communications arrangement.

Current state: **0 errors, 0 undefined or multiply-defined citations, 0 overfull boxes** in
both documents, and no ink outside the text block on any page. Main text 24 pp, SI 23 pp.

A LaTeX bug in the original bundle is fixed: `sn-jnl.cls` applies `\tablebodyfont`
*inside* its `table` environment, so every `\small`/`\scriptsize` placed after
`\begin{table}` was a silent no-op. That is why six SI tables bled into the margin. The
size commands now sit immediately before each `\begin{tabular}`, where they bind.
Do not wrap these tables in `\resizebox`: `sn-jnl` builds them inside a `threeparttable`
and the combination does not compile.

## Nature Communications compliance

| Requirement | Status |
|---|---|
| Title ≤ 15 words | 12 ✔ |
| Abstract ≤ 200 words, unreferenced | 187 ✔ |
| Main text ≤ 5,000 words (excl. Methods/refs/legends) | 5,448. NC states 5,000 as an ideal, not a limit, and does not enforce format at first submission. The seven formal statements account for ~450 of those words, and ~700 words of prose the theory made redundant were cut. Going lower means removing a theorem. |
| Display items ≤ 10 | 5 figures + 1 table ✔ |
| References ≤ ~70 | 49 ✔ |
| Structure: Introduction / Results / Discussion / Methods | ✔ |
| Data availability, Code availability, Author contributions, Competing interests | ✔ |
| AI-use disclosure in Methods | **red flag 3 — you must complete this** |
| Acknowledgements and funding | **red flag 7 — this section did not exist** |
| Archived DOI for data and code | **red flags 4 and 6** |
| Line numbers in the review PDF | on |

Nature Communications is single-blind by default, so author names and the GitHub URL are
fine as they stand. If you opt into double-blind review, the repository URL and the
`KurbanIntelligenceLab` org name de-anonymise the manuscript and must be replaced.

---

## Analysis code

`parity_metrics.py` (numpy only; SciPy used if present) implements every reported
quantity: the violation magnitude, false-flag fraction, threshold curve, the per-atom
size-normalised variant, bootstrap CIs, the one-sided paired Wilcoxon test with a
rank-biserial effect size, the pooled-seed effect size, Jaccard and Spearman agreement,
the Jacobian even-subspace fraction, and the Reynolds projector.

```bash
python parity_metrics.py     # metrics, statistics, and the group theory from scratch
python check_crossrefs.py    # every Supplementary Note/Table reference, semantically
python check_submission.py   # section-by-section format check against NC
```

Smoke test 4 is worth running before you argue with a referee: it builds the point groups
432, 23 and 422 from scratch and computes the rank of the Reynolds projector on the
18-dimensional piezoelectric space. It returns **0, 1, 1**, independently confirming the
group theory the whole mechanism argument rests on.

---

## Blocking findings — the paper is not submission-ready until these are closed

1. **The prevalence audit is one generation stale.** Its fifteen models stop before eSEN,
   UMA and EquiformerV3, which are built on the same spherical-channel embedding as the
   SO(3)-only rows and are what people actually deploy in 2026. The claim "most released
   architectures cannot forbid impossible predictions" is the paper's prevalence
   backbone, and a referee will ask about UMA by name. Run the same protocol on the
   three and add the rows. Nothing in this bundle asserts their parity structure,
   because we did not test them.
2. **The proofs in Supplementary Note 1 are AI-drafted.** A human author must re-derive
   and sign off on every step. The numerical checks in `parity_metrics.py` support them
   but are not a substitute.
3. **Two to three independent human mock reviews**, with any unresolved finding treated
   as blocking.

## Then, in order

4. Close red flags 1-9 in `RED_FLAG_LIST.txt`.
5. Reconcile the EquiformerV2 0.9550 / 0.9570 discrepancy (red flag 9).
6. Print the PDFs and read the figures on paper. Every table in the SI overflowed the
   text block in the version you sent, and no automated check in the original bundle
   caught it.
