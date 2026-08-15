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
- Seven validators were recovered from git (`HEAD:docs/paper_2/`) into `docs/draft/`, and four
  crash-level or silent-pass defects in them were fixed. The six that were kept all run clean
  against this tree;
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

## Repository state

The repository is scoped to the submission and what the availability statements promise.
`docs/draft/` holds the manuscript with its generator, figdata and validators; `src/` and
`tests/` hold the package and the parity verification gate; `scripts/`, `configs/` and
`results/` hold the experiment drivers, their configs and the frozen measurement records the
manuscript cites; `data/manifests/` and `data/splits/` carry the digests and split
definitions, with the processed Materials Project archives tracked alongside them.

Retired in the cleanup, and how to recover each: the study prose (`INTRO.md`, `METHODS.md`,
`RESULTS.md`), superseded by the manuscript, which carries the Clifford withdrawal in full;
the GPU-rental and container infrastructure, which no availability statement promises; the
18 script-generated appendices under `docs/results/`, which regenerate from
`scripts/analyze_results.py` and `scripts/analyze_ood_symmetry.py`; the patched `escnn`
package and its probe, since `escnn` appears nowhere in the manuscript; the archived scouting
probes under `results/noneN3/`, superseded by `scripts/f4_noneN3_control.py`, which is what
the Supplementary Information cites; and two development plots superseded by
`docs/draft/figures/`. Everything named here is in `to_be_deleted/`, mirroring the repository
layout, so any group is restored by moving its path back.

The three appendices with no generator — `f3_size_consistency.md`, `f4_noneN3_control.md` and
`f5_pooling_arms.md` — were kept for that reason. `scratch_hotpp/hotpp/` was kept because
`scripts/f4_noneN3_control.py` imports it and the Supplementary Information's no-e3nn claim is
checkable only against it. The CliffordSTF implementation was kept: the Supplementary
Information reports two measured quantities from it, and the code availability statement
promises all code without carve-outs.

## Packaging

`pyproject.toml` plus `uv.lock`, `requires-python = "==3.12.*"`. There is no
`requirements.txt` and no `pip install` path anywhere in the repository. The `nequip` and
`mace` extras conflict and cannot co-install, which is why CI runs them as a matrix.
