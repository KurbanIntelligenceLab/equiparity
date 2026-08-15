# Consolidation report: `jp/presub`

## The reframe

The figure build was regenerating a superseded directory. `docs/build_figures.py` wrote to
`docs/figures/`, a staging copy, while the submission includes `docs/draft/figures/`. The
generator also emitted `figS3_epoch_curves.pdf`, but the submission's Supplementary
Information includes `figS1_epoch_curves` — the panel was renumbered when the manuscript
moved, and the generator was never told. Running `make figures` therefore did not update
any figure the submission actually compiles, which quietly contradicts the code
availability statement's promise that a reader can regenerate them.

That is fixed: the generator now lives at `docs/draft/build_figures.py`, writes
`docs/draft/figures/` directly, and emits the filenames the manuscript cites.

## What was verified before anything moved

1. **Figure reproduction and build-input isolation.** The generator was run into an
   isolated scratch directory holding only `build_figures.py` and `figdata/`, with the
   repository untouched. It produced all eight PDFs, which proves build-input isolation:
   the generator needs only those two paths. A single comparison cell then computed
   decompressed-PDF-content-stream digests for both sides — the freshly generated files and
   the pre-edit rendered copies — and printed them side by side. Six of the seven panels the
   submission retains match their pre-edit stream exactly, including `figS1_epoch_curves`,
   whose stream is identical to the pre-edit `figS3_epoch_curves`: the panel was renamed,
   not redrawn.
2. **The one changed panel.** `fig5_mechanism.pdf` does **not** match, and that is
   intended: it carries the MgO correction described below. It is the only figure whose
   content changed in this work.
3. **A separate pre-existing difference.** `docs/paper_3/figures/fig5_mechanism.pdf` also
   differs from the submission's own copy, independently of the correction. It is the sole
   record of the earlier panel, which is why `docs/paper_3` is retired rather than deleted.
4. **Baseline caveat.** `docs/draft/figures/` was never git-tracked (zero files under that
   path on `jp/dev`), so the pre-edit baseline is the staging copy now held in
   the staging copy `docs/figures/`, not git history. Every figure comparison in this report is
   against that baseline, which was captured by content-stream digest before the staging copy
   was retired: `fig1` `6aeb4103987098bd`, `fig2` `b70019736261a939`, `fig3` `9a24807e13bc8fd6`,
   `fig4` `cb606bc0099a5311`, `fig5` `48467cb31ebb6f25`, `figS3_epoch_curves`
   `7eb2e7676bb09911`, `figS2` `435ac6a16bdb0a42`, `figS1_raw_thresholds` `f36dfb7d76d9d087`.
   The digests are recorded here so the comparison remains checkable without the copy.
5. **Validator baseline.** The repository's five validators were recovered from
   `HEAD:docs/paper_2/` (deleted from the worktree but present in git) and run against
   `docs/draft` before any change. Baseline: `audit_numbers` 48/48 consistent,
   `check_crossrefs` clean, `check_submission` crashed on a `SyntaxError`, `verify_figures`
   crashed on a missing label, `check_tables` silently reported "0 tables checked". The
   last three were pre-existing defects, not introduced here.
6. **Manifest digests.** All five content hashes recorded in `data/manifests/*.yaml` were
   verified against the archives on disk. All matched.

## Value-tracing

Applied one-way, as it must be: a distinctive value appearing in a retained document proves
a file load-bearing, while its absence proves nothing.

- `results/e6_named_materials.json` proved load-bearing for Fig. 5c: the ten hard-coded
  compound rows are the per-seed means in that record, matching at the precision each
  literal carries.
- `scratch_hotpp/` proved load-bearing twice: `scripts/f4_noneN3_control.py` imports it by
  absolute path, and the Supplementary Information's claim that HotPP carries no e3nn
  dependency is checkable only against this tree (`grep -rl e3nn` over its 34 Python files
  returns nothing).
- **Family integrity held.** The six probe files duplicated between `scratch_hotpp/` and
  `results/noneN3/` looked like a clean retirement candidate. Five are byte-identical; one,
  `hotpp_parity_probe.json`, differs. Neither copy is disposable until that difference is
  explained, so the family was kept whole.
- `handoff/piezo_raw.json` traces to nothing — no script and no manuscript file names it.
  It holds 3,322 records against the 3,312 structures of the tracked piezoelectric archive.
  Retired, not deleted, because the ten-record difference is unexplained.

## Defects found and fixed

| Defect | Status |
|---|---|
| Figure build wrote to a superseded directory | Fixed: generator moved into `docs/draft/`, writes the submission's `figures/` |
| Generator emitted `figS3_epoch_curves`, manuscript includes `figS1_epoch_curves` | Fixed: output filename now tracks the manuscript |
| Every `make figures` left a phantom diff (matplotlib `CreationDate`) | Fixed: `metadata={"CreationDate": None}`; two consecutive runs are byte-identical |
| `check_submission.py` crashed: backslash in an f-string expression | Fixed |
| `verify_figures.py` crashed: `stab:distributions`, no such label | Fixed to `stab:distribution` |
| `verify_figures.py` silently skipped the EqV2 row (table abbreviates it `EqV2`) | Fixed: 7 of 7 distribution rows now checked, was 6 |
| `verify_figures.py` looked for `stab:h3`, which the submission merged into `stab:t3`b | Fixed |
| `verify_figures.py` audited Fig. 5c against a table the submission retired into prose | Repointed at `results/e6_named_materials.json` |
| Fig. 5c MgO NequIP SO(3) read `4.9e-8`; the record's seed mean is `4.8474e-8` | Fixed to `4.8e-8`; `fig5_mechanism.pdf` regenerated |
| Three MP manifests shipped `file_hashes: {}` against a data statement promising content hashes | Fixed: digests populated and `scripts/prepare_mp.py` now computes them |
| `/data/raw/` directory-form ignore rule made every `!` exception inside it unreachable | Fixed to `/data/raw/**` |
| `.DS_Store` files would have been committed to the submission branch | Fixed: OS-metadata rule added |
| `fig4_training` was generated on every run but cited by no `\includegraphics`; the submission retired it into Supplementary Table `stab:t3` a/b | Gated behind `--include-retired`; the default run now writes exactly the set the manuscript includes |
| `.gitignore` `!` exceptions still named `docs/paper` and `docs/paper_2` | Repointed at `docs/draft` |

## Accounting

Apparent size throughout: the sum of file bytes, not `du` disk usage. Magnitudes are
decimal MB. Counts come from an `os.walk` that includes dotfiles, excluding `.git`.

| | Files | Size |
|---|---|---|
| Before | 134,933 | 351.56 MB |
| Retired | 72 | 8.44 MB |
| Authored or recovered into the tree | 10 | — |
| Bytecode and OS-metadata by-products swept | 242 | 2.79 MB |
| Tree now | 134,640 | 340.59 MB |

The reconciliation:

```
134,933 − 72 retired − 242 by-products + 10 authored = 134,629   observed 134,640   (+11)
```

The eleven are the author's own test and compile by-products, created while this work was in
progress: ten `.pytest_cache` files under the repository root and `tests/`, plus one LaTeX
build product. All are gitignored and none is staged.

Retirement was staged in a gitignored `to_be_deleted/` mirror of the repository layout, which
the author has since moved to the system Trash. That is why the retired files no longer appear
in the tree, and why the recovery paths in the next section point at git rather than at a
holding directory.

133,885 of the 134,640 files are the QM9 `.xyz` set under `data/raw/qm9/`, which is gitignored
and rebuilt by `scripts/prepare_qm9.py` from a tarball pinned by content hash. Excluding it,
the tree is about 750 files.

## Version control

`jp/presub` was branched from `jp/dev`. `.gitignore` was rewritten from a list of
extensions into a stated policy: the manuscript and its figure code are tracked
deliberately and the file says so; `*.pdf` is a build product with narrow exceptions for
the released figures; the processed Materials Project archives are tracked (6.7 MB) while
QM9 is not; the holding directory never reaches the remote. The `*.log` rule was examined
rather than assumed — it currently catches only pdfTeX output, and the file records what to
do if a harvested log is ever added.

## Retired from the submission, and how to get it back

Only material retired from `docs/draft` is recorded here. The superseded manuscript, the
staging figure copy, the review screenshots, the drafting notes and the stale inventory were
also retired; they are development history rather than part of the submission, and nothing in
the manuscript or its build depends on them.

| Retired from the submission | Recover with |
|---|---|
| `check_tables.py` | `git show d4270eb:docs/draft/check_tables.py > docs/draft/check_tables.py` |
| `supplementary_data/Supplementary_Data_2_pooling_per_seed.csv` | `git show d4270eb:docs/draft/supplementary_data/Supplementary_Data_2_pooling_per_seed.csv > <path>` — though every value re-derives from `results/f5_pooling_arms.json` |
| `figures/fig4_training.pdf` | `python build_figures.py --include-retired` (content stream `cb606bc0099a5311`) |
| `figures/figS1_raw_thresholds.pdf` | same flag (content stream `f36dfb7d76d9d087`) |

The two panels regenerate rather than needing a stored copy, which is the point of keeping the
generator beside the manuscript. Move them out of `figures/` afterwards: the submission's figure
set is the six the manuscript compiles, and the default run writes exactly those.

**Not pushed.** The branch exists locally with three commits on top of `a3342ea`:
`71587a8` commits the pooling arms, size-consistency and non-e3nn control work that was
already sitting uncommitted in the worktree, unmodified, so that the consolidation is
separately reviewable; `d4270eb` is the retarget; the third is the follow-up sweep recorded below
(named by subject rather than hash, since the hash changes if the commit is amended). `git fetch origin` failed with a network error in this environment, so the
branch is based on the local `jp/dev` tip, which was level with `origin/jp/dev` at the time.
Pushing to a shared remote is the author's call.

## Follow-up sweep

A second pass over the whole tree, after the retarget commit, looking for references to
retired paths and for material no decision had yet been made about.

**It caught an error in the first pass.** Six probe files had been retired from
`scratch_hotpp/` as duplicates of `results/noneN3/`, on the grounds that five of the six are
byte-identical. That was wrong: `probe_escnn.py` does
`sys.path.insert(0, os.path.join(os.path.dirname(__file__), "patched_pkgs"))` and
`probe_hotpp.py` does `sys.path.insert(0, ".")`, so both resolve their imports **relative to
their own directory**. `patched_pkgs/` and `hotpp/` live in `scratch_hotpp/`, not in
`results/noneN3/`, which means the `scratch_hotpp/` copies are the runnable ones and the
`results/noneN3/` copies are the archived record. All six were restored. This is why the
one-way test matters: identical bytes did not make the file disposable, because what made it
load-bearing was its location.

The differing pair `hotpp_parity_probe.json` is now explained: `probe_hotpp.py` writes to its
own cwd, so the file is the same script run twice from two directories, and the differences
are last-bit float noise in quantities at machine epsilon (`max_abs_error` 1.39e-16 vs
1.67e-16 on a norm of 1.0816652438). Neither copy is wrong and neither is authoritative over
the other.

Other findings, all fixed:

| Finding | Resolution |
|---|---|
| `scripts/export_figdata.py` wrote to `docs/paper_2/figdata/` and would have recreated that directory, putting the generator's inputs where the manuscript does not read them | Repointed at `docs/draft/figdata/` |
| `handoff/meanpool_readiness.md` was retired, but two tracked files cite it — `scripts/generate_grid_meanpool.py` for why the target scale must be refit, and `configs/mp_piezoelectric_meanpool_dryrun.yaml` for why the dry run uses the tiny subset | Promoted to `docs/results/f5_pooling_arms.md`, joining the other 20 appendices and matching `results/f5_pooling_arms.json`; both citations repointed |
| 15 citations across `INTRO.md`, `METHODS.md`, `RESULTS.md`, six scripts, `src/equiparity/__init__.py`, `src/equiparity/models/nequip.py`, `tests/test_physics_claims.py` and `docs/results/e1_augmentation.md` pointed at checkpoint and work-plan documents deleted from the worktree before this work began | The 13 documents were recovered from `a3342ea` into the holding directory, so the retirement is reversible from one place, and the citations repointed at the surviving homes for that content |
| `.DS_Store` was untracked but not ignored, so a Finder visit would add it | OS-metadata rule added (recorded above) |

Three items were examined and deliberately left alone. `scratch_hotpp/patched_pkgs/` (232
files, 3.4 MB) is the patched `escnn` that `probe_escnn.py` imports as a sibling; it is the
only copy. `outputs/` holds two run directories with checkpoints and is gitignored by the
repository's own rule, so it never reaches the remote and costs nothing to leave in place.
`.mp_cache/` is empty.

## Unverified

**The manuscript was not compiled.** No TeX engine is installed in the environment this
work was done in. A retarget that has not been built is not finished. Every claim above
about figures, records, digests and validators was executed; no claim about the compiled
document was. The open items that need a build, and the one that needs the author, are in
`docs/draft/TODO_SUBMISSION.md`.
