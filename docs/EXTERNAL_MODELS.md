# External models

Some measurements evaluate models released by other groups. Their code and checkpoints are not
redistributed here — they carry their own licences and, in one case, an access gate. Each is
placed under `third_party/`, which is not tracked.

Nothing in this document is needed to install the package, run the test suite, or check the
claims: `verification/verify_theory.py` and `verification/verify_claims.py` read only what is
committed. The per-structure outputs of every measurement below are already released under
`results/`, so a reported fraction can be recomputed without obtaining any of these models.

## Expected layout

```text
third_party/
  AIRS/OpenMat/GMTNet/GMTNet_piezo/     GMTNet, with its released piezoelectric checkpoint
  ceitnet/piezo/                        CEITNet, with repro_data/pretrained_ckpts/piezo.pt
  equiformer_v2_upstream/               atomicarchitects/equiformer_v2 @ 8fe8cba
  equiformer_v2_shimmed/                built from the above by the upstream-build script
  equiformer_v3/experimental/models/    EquiformerV3
  ICTP/                                 ICTP
  GotenNet/                             GotenNet
  torchmd-net/                          TorchMD-NET, a GotenNet dependency
  checkpoints/                          downloaded checkpoints (eSEN)
  venvs/                                isolated interpreters, see below
```

## Why isolated environments

These models pin mutually incompatible dependency sets, and several conflict with both of this
project's own profiles. Each is installed into its own virtual environment under
`third_party/venvs/` and invoked by absolute interpreter path rather than through `uv run`. The
scripts that need one say so in their docstring and name the interpreter they expect.

| Environment | Used by |
|---|---|
| `third_party/venvs/tier1/` | GMTNet and CEITNet evaluation |
| `third_party/venvs/fairchem/` | UMA random-initialization probe |
| `third_party/venvs/fairchem1/` | eSEN feature caching |

## Per model

**GMTNet** — dedicated crystal-tensor predictor. Clone `divelab/AIRS` and use
`OpenMat/GMTNet/GMTNet_piezo` with its released piezoelectric checkpoint. Nothing inside
`third_party/` is modified; `scripts/experiments/tensor_predictors.py` imports their graph
construction and symmetry-mask helpers directly so the evaluation runs their pipeline on our
structures.

**CEITNet** — dedicated crystal-tensor predictor. Needs
`third_party/ceitnet/piezo/repro_data/pretrained_ckpts/piezo.pt` from the authors' release. Their
forced-zero mask functions are copied verbatim into our script rather than imported, because
importing their `test.py` triggers module-level side effects.

**eSEN** — frozen universal potential backbone. **Access is gated.** The checkpoint
`esen_30m_oam.pt` lives in the `facebook/OMAT24` repository on Hugging Face. Accept the licence at
<https://huggingface.co/facebook/OMAT24> with a Hugging Face account, create a read token at
<https://huggingface.co/settings/tokens>, and pass it as `HF_TOKEN` (or run `huggingface-cli
login` inside the `fairchem1` environment). `scripts/experiments/cache_esen_features.py` fails
with a pointer to these steps if the gate has not been accepted.

**MACE-MP-0** — frozen universal potential backbone, downloaded automatically by
`mace.calculators.mace_mp` on first use. No token required; needs the `mace` profile.

**UMA** — rotation-only potential, probed at random initialization through
`fairchem.core.models.uma`. Install fairchem into `third_party/venvs/fairchem/`.

**EquiformerV3** — rotation-only potential, probed at random initialization. Expected at
`third_party/equiformer_v3/experimental/models/`.

**EquiformerV2 (upstream)** — used to confirm that the copy vendored under
`src/equiparity/models/equiformer_v2/` is faithful to its source. Clone
`atomicarchitects/equiformer_v2` at commit `8fe8cba` into `third_party/equiformer_v2_upstream/`.
`scripts/experiments/equiformer_v2_upstream_build.py` reconstructs a runnable copy by relocating
`generate_graph` into a shim, recording every edit so the claim "measured on upstream source at
8fe8cba" is auditable.

**ICTP** and **GotenNet** — audited architectures whose parity class could not be settled by
reading their source. Each is probed in an isolated environment by
`scripts/experiments/inheritance_probes.py`; GotenNet additionally needs TorchMD-NET. Pass the
isolated interpreter with `--gotennet-python /path/to/venv/bin/python`.

**HotPP** — the non-e3nn O(3) control. This one *is* vendored, at `vendor/hotpp/`, under MIT
(arXiv:2402.15286), because it has no PyPI release.
