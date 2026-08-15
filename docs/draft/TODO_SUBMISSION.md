# Open items on the submission

Split by who can settle them.

## Resolved during consolidation

- Figure generator moved to `docs/draft/build_figures.py` and repointed at the submission's
  own `figures/` directory. It previously wrote a staging copy that the manuscript does not
  include.
- Epoch-curve output renamed `figS1_epoch_curves.pdf` to match the manuscript's
  `\includegraphics`. The generating function is still `figS3()`, after the panel's number
  in the superseded draft.
- Figure PDFs are byte-reproducible: `metadata={"CreationDate": None}` on the save path, so
  a rerun on unchanged inputs leaves no diff. Verified across two runs.
- Fig. 5c MgO NequIP SO(3) corrected `4.9e-8` to `4.8e-8`. The seed mean in
  `results/e6_named_materials.json` is `4.8474e-8`. **This changed the figure**, so
  `fig5_mechanism.pdf` was regenerated and the manuscript needs recompiling to pick it up.
- The five validators were recovered from git (`HEAD:docs/paper_2/`) into `docs/draft/`,
  and four crash-level or silent-pass defects in them were fixed. All five now run clean
  against this tree.
- `data/manifests/mp_piezoelectric.yaml`, `mp_elastic.yaml` and
  `mp_ood_centrosymmetric.yaml` shipped `file_hashes: {}`, contradicting the data
  availability statement. Digests populated and verified; `scripts/prepare_mp.py` now
  computes them so regenerated manifests carry them too.
- The raw-threshold panel `figS1_raw_thresholds.pdf` was retired: no `\includegraphics`
  cites it, since the submission moved that content into prose. Its `figdata` CSV is kept,
  as the SI promises the series accompanies the code release.

## Needs a TeX engine

None of this could be checked here; `pdflatex`, `xelatex`, `latexmk` and `tectonic` are all
absent from the environment.

1. **Recompile `main.pdf` and `sn-supplementary.pdf`.** The shipped PDFs predate the
   `fig5_mechanism.pdf` correction, so the compiled Fig. 5c still shows the old MgO value.
2. **`sn-supplementary` has 37 unresolved citations.** All 37 keys are present in
   `references.bib` and the main document resolves every one; the standalone SI has no
   `.bbl`. This is the one remaining `check_submission.py` failure. It needs BibTeX run on
   `sn-supplementary` as part of the build.
3. **`./build.sh` does not exist.** Both `main.tex` and `sn-supplementary.tex` reference it
   for the two-pass `xr` label-extraction procedure (each document's `\newlabel` lines
   extracted into `*-labels.aux`, so `xr` imports labels without importing `\bibcite`
   lines). The `*-labels.aux` files are present, so the procedure was run at some point, but
   the script is not in the repository. Either restore it or write it from the comments —
   without it the cross-references between the two documents cannot be rebuilt from a clean
   checkout, which the reproducibility statement implies they can.
4. **`main.log` reports 26 underfull boxes.** Zero overfull, which is what
   `check_submission.py` gates on. Not individually inspected.

## Needs the author

1. **AI-use disclosure.** `check_submission.py` flags it as present-but-incomplete in
   `sections/methods.tex`. Only the author can complete it.
2. **`check_tables.py` checks nothing.** Its regex looks for `\begin{tabular}`; every table
   here uses `\begin{tabular*}` for full-width Nature tables, so it reports "0 tables
   checked" and passes. The 13 tables are unverified against Nature's table rules. Widening
   the regex was left alone deliberately: it would turn a silent pass into a set of real
   findings on tables that are hand-typed, and triaging those is an editorial decision.
3. **No table generator exists.** No `.py` file in the repository contains `tabular`,
   `toprule` or `booktabs`, so the tables are hand-typed and do not regenerate. If the
   reproducibility appendix implies they do, that claim needs softening, or a generator
   needs writing. The honest intermediate is a provenance table mapping each float to the
   record its values come from.
4. **`handoff/piezo_raw.json` holds 3,322 records** against the 3,312 structures of
   `data/raw/mp/mp_piezoelectric_processed.npz`. Nothing reads it. It is retired to
   `to_be_deleted/handoff/` rather than deleted because the ten-record difference is
   unexplained — if it is a pre-filter snapshot, it can go.
5. **`scratch_hotpp/hotpp_parity_probe.json` differs from `results/noneN3/hotpp_parity_probe.json`.**
   The other five probe files in that pair are byte-identical. Which is authoritative is
   not decidable from the code.
6. **`docs/draft/supplementary_data/Supplementary_Data_2_pooling_per_seed.csv` is orphaned.**
   No `.tex` file in the submission references "Supplementary Data" at all, and there is no
   Supplementary Data 1. Either the manuscript should cite it or it should not ship.
