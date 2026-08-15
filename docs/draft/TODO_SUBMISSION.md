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
- Six validators were recovered from git (`HEAD:docs/paper_2/`) into `docs/draft/`, and four
  crash-level or silent-pass defects in them were fixed. Five run clean against this tree;
  `check_tables.py` was retired (see below).
- `data/manifests/mp_piezoelectric.yaml`, `mp_elastic.yaml` and
  `mp_ood_centrosymmetric.yaml` shipped `file_hashes: {}`, contradicting the data
  availability statement. Digests populated and verified; `scripts/prepare_mp.py` now
  computes them so regenerated manifests carry them too.
- The raw-threshold panel `figS1_raw_thresholds.pdf` was retired: no `\includegraphics`
  cites it, since the submission moved that content into prose. Its `figdata` CSV is kept,
  as the SI promises the series accompanies the code release.

## Build status

The manuscript now compiles as a single document. The Supplementary Information was folded
into `main.tex` (input after `\bibliography{references}`), which retired the two-document
`xr` label-extraction scheme and the `./build.sh` it depended on. The build is a plain
`pdflatex` / `bibtex` / `pdflatex` x2 sequence.

Latest build: 51 pages, pdfTeX (TeX Live 2026), zero errors, zero undefined references and
zero undefined citations. It postdates the corrected `fig5_mechanism.pdf` by 17 minutes and
`main.log` confirms it embedded that file, so the compiled Fig. 5c carries the corrected MgO
value. `main.log` reports 53 underfull boxes and zero overfull; `check_submission.py` gates on
overfull only, and the underfull warnings were not individually inspected.

Three items that previously required a TeX engine are therefore closed: the recompile, the 37
unresolved Supplementary Information citations, and the missing `build.sh`.

## Needs the author

1. **AI-use disclosure.** `check_submission.py` flags it as present-but-incomplete in
   `sections/methods.tex`. Only the author can complete it. This is the one item on this list
   that no amount of tooling can settle.

## Resolved in the follow-up sweep

These were on the author's list and have been settled.

- **`check_tables.py` retired.** It reported "0 tables checked   0 rule violations" on every
  run, because its regex looks for `\begin{tabular}` while every table here uses
  `\begin{tabular*}`. A validator that passes by matching nothing reads as evidence when it
  is not, so it is retired rather than left in place
  (`git show d4270eb:docs/draft/check_tables.py` recovers it). The 13 tables are consequently not machine-checked; `REPRODUCE.md` says so plainly.
- **No table generator, and no claim of one.** Confirmed that no `.py` file in the repository
  contains `tabular`, `toprule` or `booktabs`: the tables are hand-typed. `REPRODUCE.md` now
  states this and points at the record files behind the values, so nothing in the repository
  implies the tables regenerate.
- **The duplicate probe question is answered, and the first pass had it backwards.** Six
  probe files had been retired from `scratch_hotpp/` as duplicates of `results/noneN3/`. That
  was wrong: `probe_escnn.py` inserts `os.path.dirname(__file__)/patched_pkgs` on the path and
  `probe_hotpp.py` inserts `"."`, so both resolve imports relative to their own directory —
  and `patched_pkgs/` and `hotpp/` live in `scratch_hotpp/`. The `scratch_hotpp/` copies are
  the runnable ones and have been restored; the `results/noneN3/` copies are the archived
  record. The differing `hotpp_parity_probe.json` is the same script run twice from two
  directories, differing only in last-bit float noise on machine-epsilon quantities
  (`max_abs_error` 1.39e-16 against 1.67e-16 on a norm of 1.0816652438). Neither is
  authoritative over the other, and both are kept.
- **`Supplementary_Data_2_pooling_per_seed.csv` retired.** No `.tex` file in the submission
  references "Supplementary Data", and there was no Supplementary Data 1. Before retiring it,
  all 21 rows were checked cell by cell against `results/f5_pooling_arms.json`: every value
  re-derives within the precision the CSV prints, so it was a formatted view of the record,
  not independent data. The Supplementary Information already promises the per-seed values
  accompany the code release, which `results/f5_pooling_arms.json` satisfies.
- **`handoff/piezo_raw.json` retired.** Nothing reads it. Retired rather than deleted because
  its 3,322 records against the tracked archive's 3,312 remain unexplained. It was never
  committed, so recovering it would mean going to the Trash before that is emptied.
- **The pooling readiness note was promoted, not retired.** `handoff/meanpool_readiness.md`
  was cited by two tracked files, so it was load-bearing. It is now
  `docs/results/f5_pooling_arms.md`, alongside the other appendices and matching
  `results/f5_pooling_arms.json`; both citations were repointed.
- **`scripts/h1_report.py` would have crashed.** It wrote into `docs/reports/`, retired with
  the checkpoint notes, without creating the parent directory. Repointed at
  `docs/results/h1_equiformer_upstream.md` with an explicit `mkdir`.
- **`scripts/export_figdata.py` wrote to the wrong tree.** It targeted
  `docs/paper_2/figdata/` and would have recreated that directory. Repointed at
  `docs/draft/figdata/`.
