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
| `fig4_training.pdf` | `fig4()` | `SWEEP` literals, audited against Supplementary Table `stab:t3`b |
| `fig5_mechanism.pdf` | `fig5()` | `NAMED`/`FAM`/`JAC` literals + `figdata/fig5a_rutile_sweep.csv`, `figdata/fig5b_jacobian_points.csv`; `NAMED` audited against `results/e6_named_materials.json` |
| `figures/figS1_epoch_curves.pdf` | `figS3()` | `figdata/figS3_epoch_curves.csv` |
| `figures/figS2_rutile_sweep.pdf` | `figS2()` | `figdata/fig5a_rutile_sweep.csv` |

The function name `figS3` and the output name `figS1_epoch_curves` differ because the
submission renumbered the Supplementary Figures; the output filename tracks the
manuscript's `\includegraphics`, which is what has to be right.

`figS1()` draws the raw-coordinate-variant threshold panel, which the submission retired
into prose (Supplementary Note, "The raw coordinate variant and symmetry tolerance"). No
`\includegraphics` cites it, so it is not part of the default run. Its machine-readable
series is still released as `figdata/figS1_raw_thresholds.csv`. To regenerate the panel:

```
python build_figures.py --include-retired
```

## Validators

Run from `docs/draft`. All five are clean on this tree.

| Script | What it checks | Result |
|---|---|---|
| `verify_figures.py` | every hard-coded figure number against its source table or record, plus zeros on log axes, clipped points, percentile monotonicity | all figures verified |
| `audit_numbers.py` | every quantity the manuscript reports twice, or reports enough to reconstruct | 48 of 48 consistent |
| `check_submission.py` | Nature Portfolio section, format, legend and integrity rules | 30 pass, 3 open (author action), 1 fail (see below) |
| `check_crossrefs.py` | every Supplementary Note/Table/Fig. cross-reference resolves and is topically corroborated | every reference resolves |
| `check_tables.py` | Nature table rules (booktabs, caption above, units in header, decimal consistency) | 0 violations |
| `audit_fig1.py` | pixel-exact label-on-ink overlap audit, per figure | run per figure: `python audit_fig1.py fig1` |

`check_tables.py` reports "0 tables checked". This is a **known limitation, not a pass**:
its regex looks for `\begin{tabular}`, and every table in this manuscript uses
`\begin{tabular*}` for full-width Nature tables. The 12 tables in
`sections/supplementary.tex` and the 1 in `sections/results.tex` are therefore unchecked
by it. There is no table generator anywhere in the repository (no `.py` file contains
`tabular`, `toprule` or `booktabs`), so the tables are hand-typed; the provenance of their
values is the record files listed under "Frozen data" below.

`check_submission.py`'s one failure is `sn-supplementary: no undefined refs/cites`. The
standalone Supplementary Information document has no `.bbl`, so its 37 `\cite` keys are
unresolved in that build. All 37 keys are present in `references.bib`; the main document
resolves every one. Fixing it requires running BibTeX on `sn-supplementary` as part of the
build, which needs a TeX engine (see "Not verified").

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

## Not verified

- **The manuscript was not compiled.** No TeX engine is installed in the environment this
  work was done in (`pdflatex`, `xelatex`, `latexmk` and `tectonic` are all absent). The
  shipped `main.pdf` and `sn-supplementary.pdf` are the author's own builds, from pdfTeX
  (TeX Live 2026) per `main.log`. Every statement here about figures, records and
  validators was executed; no statement about the compiled document was.
- `main.tex` and `sn-supplementary.tex` both reference `./build.sh`, which does not exist
  in the repository. The two-pass label-extraction procedure those comments describe (each
  document's `\newlabel` lines extracted into `*-labels.aux` for `xr` to import) has to be
  reconstructed from the comments, or the script supplied. The `*-labels.aux` files are
  present, so the procedure was run at some point.
- `main.log` reports 26 overfull or underfull boxes. `check_submission.py` counts only
  "Overfull" and passes, so these are underfull warnings; they were not individually
  inspected.
