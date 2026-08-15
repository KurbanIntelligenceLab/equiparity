# Reproducing the submission

Everything in this document was executed against this tree. Where something was **not**
executed, it says so.

The submission is `docs/draft/`. Its figure generator, figure-data inputs and validators
live beside it, so the manuscript and the code that produces it move together.

## Figures

```
cd docs/draft
python build_figures.py
```

Writes the seven PDFs in `figures/` that the manuscript includes. Reruns are
byte-identical: `save()` passes `metadata={"CreationDate": None}`, which omits
matplotlib's timestamp, so an unchanged rerun leaves no diff in `git status`. Verified by
running the target twice and comparing SHA-256.

| Figure | Generator function | Data source |
|---|---|---|
| `fig1_concept.pdf` | `fig1()` | `DIST` literals, audited against Supplementary Table 4 |
| `fig2_headline.pdf` | `fig2()` | `DIST` literals + `figdata/fig2b_thresholds.csv` |
| `fig3_accuracy.pdf` | `fig3()` | `ACC` literals, audited against Supplementary Table 7 |
| `fig5_mechanism.pdf` | `fig5()` | `NAMED`/`FAM`/`JAC` literals + `figdata/fig5a_rutile_sweep.csv`, `figdata/fig5b_jacobian_points.csv`; `NAMED` audited against `results/e6_named_materials.json` |
| `figures/figS1_epoch_curves.pdf` | `figS3()` | `figdata/figS3_epoch_curves.csv` |
| `figures/figS2_rutile_sweep.pdf` | `figS2()` | `figdata/fig5a_rutile_sweep.csv` |

The function name `figS3` and the output name `figS1_epoch_curves` differ because the
submission renumbered the Supplementary Figures; the output filename tracks the
manuscript's `\includegraphics`, which is what has to be right.

The default run writes exactly the six panels the manuscript includes — the generated set and
the `\includegraphics` set are equal, verified as a set comparison.

Two panels are retired, neither cited by any `\includegraphics`:

- `figS1()`, the raw-coordinate-variant threshold curves, retired into prose (Supplementary
  Note, "The raw coordinate variant and symmetry tolerance"). Its machine-readable series is
  still released as `figdata/figS1_raw_thresholds.csv`.
- `fig4()`, the trained-on-zeros lollipop and loss-weight sweep, retired into Supplementary
  Table `stab:t3` a/b and the surrounding prose. `verify_figures.py` still audits its `SWEEP`
  and `INTRAIN` literals against that table, so the numbers stay under test even though the
  panel is not drawn.

To regenerate both:

```
python build_figures.py --include-retired
```

They are written into `figures/`; move them out before committing, since the submission's
figure set is the six the manuscript compiles.

## Validators

Run from `docs/draft`. All six are clean on this tree: the four that return a pass/fail verdict
exit 0, and the two run-per-target tools complete without findings.

| Script | What it checks | Result |
|---|---|---|
| `verify_figures.py` | every hard-coded figure number against its source table or record, plus zeros on log axes, clipped points, percentile monotonicity | all figures verified |
| `audit_numbers.py` | every quantity the manuscript reports twice, or reports enough to reconstruct | 48 of 48 consistent |
| `check_submission.py` | Nature Portfolio section, format, legend and integrity rules | 28 pass, 3 open (author action), 0 fail |
| `check_crossrefs.py` | every Supplementary Note/Table/Fig. cross-reference resolves and is topically corroborated | every reference resolves |
| `audit_fig1.py` | pixel-exact label-on-ink overlap audit, per figure | run per figure: `python audit_fig1.py fig1` |
| `parity_metrics.py` | first-principles recomputation of the rank and false-flag statistics from the records | `python parity_metrics.py` |

There were six validators. `check_tables.py` was **retired** rather than kept: it reported
"0 tables checked   0 rule violations" on every run, because its regex looks for
`\begin{tabular}` while every table in this manuscript uses `\begin{tabular*}` for
full-width Nature tables. A validator that passes by matching nothing is worse than no
validator, since its clean output reads as evidence. Recover it with
`git show d4270eb:docs/draft/check_tables.py` if the regex is ever widened.

The 12 tables in `sections/supplementary.tex` (11 `table` floats and one `longtable`) and the
1 in `sections/results.tex` are
therefore not machine-checked against Nature's table rules. There is no table generator
anywhere in the repository (no `.py` file contains `tabular`, `toprule` or `booktabs`), so
the tables are hand-typed and do not regenerate; the provenance of their values is the
record files listed under "Frozen data" below. Nothing in this repository claims they
regenerate.

`check_submission.py` reports 28 pass, 3 open (author action), 0 fail. The three open items
are for the author: the AI-use disclosure, six em dashes that are table "not applicable"
markers rather than prose, and the red-flag list.

## Frozen data

Records the manuscript's own text names, and which script produces each:

| Record | Produced by | Cited in |
|---|---|---|
| `results/f3_size_consistency.json` | `scripts/f3_size_consistency.py` | Supplementary Information, size-consistency note |
| `results/f4_noneN3_control.json` | `scripts/f4_noneN3_control.py` | Supplementary Information, non-e3nn control note |
| `results/qm9_pointgroups.json` | `scripts/qm9_pointgroups.py` | Supplementary Information, QM9 point-group census |
| `results/e6_named_materials.json` | `scripts/e6_named_materials.py` | Fig. 5c provenance; per-compound values promised to the code release |
| `results/e7_rotation_subgroup.json` | `scripts/e7_rotation_subgroup.py` | Fig. 5a family fractions |

Datasets. `data/manifests/*.yaml` carry SHA-256 digests of the processed archives, and
`data/splits/*.npz` carry the split identifier lists. All five recorded digests were
verified against the archives on disk. The processed Materials Project archives
(`data/raw/mp/*.npz`, 6.7 MB) are tracked, because they are the exact arrays the
evaluation reads and no public endpoint returns them. QM9 is not tracked: its source
tarball is pinned by content hash in `data/manifests/qm9.yaml` and
`scripts/prepare_qm9.py` rebuilds the 133,885 `.xyz` files from it.

`scratch_hotpp/` is a vendored copy of HotPP, which has no PyPI release under that name.
`scripts/f4_noneN3_control.py` imports it from there, and the Supplementary Information's
claim that the package carries no e3nn dependency is checkable only against this tree
(`grep -rl e3nn` over its 34 Python files returns nothing).

## Build

The manuscript compiles as a single document: the Supplementary Information is input by
`main.tex` after `\bibliography{references}`, so every cross-reference and citation resolves
within one `.aux`. There is no `xr`, no separate Supplementary Information document, and no
`build.sh`.

```
cd docs/draft
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Latest build: 51 pages, pdfTeX (TeX Live 2026), zero errors, zero undefined references, zero
undefined citations. It embeds the corrected `fig5_mechanism.pdf`.

`main.log` reports 53 underfull boxes and zero overfull. `check_submission.py` gates on
overfull only; the underfull warnings were not individually inspected.

## Provenance of the statements above

The figure, record, digest and validator statements in this document were executed and their
output read. The build facts in the preceding section are read from `main.log` and from file
timestamps: no TeX engine is installed in the environment this work was done in, so the
compile was the author's.
