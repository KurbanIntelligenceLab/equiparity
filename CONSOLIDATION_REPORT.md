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

1. **Figure reproduction.** The generator was run into an isolated scratch directory
   holding only `build_figures.py` and `figdata/`, with the repository untouched. All eight
   PDFs it produced matched the shipped `docs/draft/figures/`, `docs/figures/` and
   `docs/paper_3/figures/` copies by decompressed PDF content stream — establishing both
   that the figures reproduce and that build-input isolation holds: the generator needs
   only those two paths.
2. **The one exception.** `docs/paper_3/figures/fig5_mechanism.pdf` has a different content
   stream from the submission's. It is the sole record of the earlier panel, which is why
   `docs/paper_3` is retired rather than deleted.
3. **Validator baseline.** The repository's five validators were recovered from
   `HEAD:docs/paper_2/` (deleted from the worktree but present in git) and run against
   `docs/draft` before any change. Baseline: `audit_numbers` 48/48 consistent,
   `check_crossrefs` clean, `check_submission` crashed on a `SyntaxError`, `verify_figures`
   crashed on a missing label, `check_tables` silently reported "0 tables checked". The
   last three were pre-existing defects, not introduced here.
4. **Manifest digests.** All five content hashes recorded in `data/manifests/*.yaml` were
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
| `.gitignore` `!` exceptions still named `docs/paper` and `docs/paper_2` | Repointed at `docs/draft` |

## Accounting

Apparent size throughout: the sum of file bytes, not `du` disk usage. Magnitudes are
decimal MB. Counts come from an `os.walk` that includes dotfiles, excluding `.git`.

| | Files | Size |
|---|---|---|
| Before | 134,933 | 351.56 MB |
| Retired into `to_be_deleted/` | 57 | 7.89 MB |
| Authored here | 9 | — |
| Live tree after | 134,885 | 343.82 MB |
| Whole tree after (holding dir included) | 134,943 | 351.71 MB |

Both reconciliations balance exactly:

```
live:  134,933 − 57 retired + 8 authored-in-live + 1 by-product = 134,885   observed 134,885
whole: 134,885 live + 57 retired + 1 manifest                   = 134,943   observed 134,943
```

The by-product is `docs/draft/__pycache__/build_figures.cpython-311.pyc`, created when the
validators import the generator. Chasing that single-file gap is what surfaced it; it is a
by-product, not cleanup, and is excluded from the retired total along with the 242
`.DS_Store` and bytecode files left in place.

Note that 133,885 of the 134,885 live files are the QM9 `.xyz` set under `data/raw/qm9/`,
which is gitignored and rebuilt by `scripts/prepare_qm9.py` from a tarball pinned by
content hash. Excluding it, the live tree is about 1,000 files.

## Version control

`jp/presub` was branched from `jp/dev`. `.gitignore` was rewritten from a list of
extensions into a stated policy: the manuscript and its figure code are tracked
deliberately and the file says so; `*.pdf` is a build product with narrow exceptions for
the released figures; the processed Materials Project archives are tracked (6.7 MB) while
QM9 is not; the holding directory never reaches the remote. The `*.log` rule was examined
rather than assumed — it currently catches only pdfTeX output, and the file records what to
do if a harvested log is ever added.

**Not pushed.** The branch exists locally with one commit; pushing to a shared remote is
the author's call.

## Unverified

**The manuscript was not compiled.** No TeX engine is installed in the environment this
work was done in. A retarget that has not been built is not finished. Every claim above
about figures, records, digests and validators was executed; no claim about the compiled
document was. The open items that need a build, and the one that needs the author, are in
`docs/draft/TODO_SUBMISSION.md`.
