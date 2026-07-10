# H1 — EquiformerV2 findings reproduced on pinned upstream source

## Provenance chain

- Vendored copy (`src/equiparity/models/equiformer_v2/`): from **`atomicarchitects/equiformer_v2`**, MIT, vendored 2026-02-11 (README).
- Pinned upstream commit: **`8fe8cbaf8f3c`** (2023-06-28). This is the only commit ever to touch `nets/equiformer_v2/*.py`; the model files are byte-frozen since (verified: `git diff 8fe8cba HEAD -- nets/equiformer_v2/` is empty).
- `so3.py` / `so2_ops.py` carry the **Meta OCP / fairchem eSCN** copyright — the eSCN lineage EquiformerV2 is built on.

All findings below are stated at commit `8fe8cba` of `atomicarchitects/equiformer_v2`, whose `so3.py` derives from the Meta OCP/fairchem eSCN lineage. No claim references 
`main`.

## Diff manifest — vendored vs upstream @ 8fe8cba

AST-level classification (formatting and comments normalized out).

| file | bucket | detail |
|---|---|---|
| `activation.py` | (a/b) AST-identical | formatting/import-order only; byte-identical to upstream in the shimmed tree |
| `drop.py` | (b) import-order | isort reordering only; AST-equivalent modulo import order |
| `edge_rot_mat.py` | (a/b) AST-identical | formatting/import-order only; byte-identical to upstream in the shimmed tree |
| `equiformer_v2_oc20.py` | (b) documented | OCP-dep removal + num_output (10 recorded edits) |
| `gaussian_rbf.py` | (b) documented | OCP-dep removal + num_output (10 recorded edits) |
| `input_block.py` | (b) import-order | isort reordering only; AST-equivalent modulo import order |
| `layer_norm.py` | (a/b) AST-identical | formatting/import-order only; byte-identical to upstream in the shimmed tree |
| `module_list.py` | (a/b) AST-identical | formatting/import-order only; byte-identical to upstream in the shimmed tree |
| `radial_function.py` | (a/b) AST-identical | formatting/import-order only; byte-identical to upstream in the shimmed tree |
| `so2_ops.py` | (b) import-order | isort reordering only; AST-equivalent modulo import order |
| `so3.py` | (b) import-order | isort reordering only; AST-equivalent modulo import order |
| `transformer_block.py` | (b) import-order | isort reordering only; AST-equivalent modulo import order |
| `wigner.py` | (a/b) AST-identical | formatting/import-order only; byte-identical to upstream in the shimmed tree |

**6 AST-identical (formatting/import-order only), 2 documented edits, 0 substantive (bucket-c).** The vendored copy is uniformly ruff-reformatted, so byte comparison is not the axis; AST identity is. The two load-bearing files are `edge_rot_mat.py` (the random per-edge frame) and `so3.py` (the detached Wigner-D matrices) — both AST-identical to upstream (edge_rot_mat differs in nothing but line wrapping; so3 in nothing but import order). In the shimmed tree used for the reruns, all 11 non-oc20/non-gaussian_rbf files are copied **byte-for-byte from upstream**.

### The one method that is not upstream `nets/` code: `generate_graph`

Upstream `equiformer_v2_oc20.py` does **not** define `generate_graph`; it *inherits* it from the OCP `BaseModel`, which lives in the external `ocpmodels` package, not in `nets/`. The vendored copy added a reimplementation when it removed the OCP dependency. Per the Checkpoint-9 ruling this was **relocated out of the model file into the shim** (`scripts/h1_build_shimmed.py`): the reconstructed `equiformer_v2_oc20.py` is upstream + the 10 documented OCP-removal edits and contains no `generate_graph` (bucket-b); the shim subclass provides it, guarded to enforce `otf_graph == False` and a precomputed `edge_distance_vec`.

## Shim manifest (what executes that is not upstream-byte-identical)

| shim element | what it provides | on the numerical forward path? |
|---|---|---|
| OCP-import / decorator / base-class removal | makes oc20 importable without `ocpmodels` | no |
| `GaussianSmearing` → `GaussianRadialBasisLayer` | upstream's own commented-out alternative, uncommented | no (config-time) |
| `gaussian_rbf.num_output` (+1 line) | attribute oc20 line 232 reads | no (config-time) |
| `generate_graph` (shim subclass) | pass-through of the precomputed graph | **yes — proven inert below** |

## Condition 2 by measurement — the shim is inert on the numbers

- **Equivalence.** The shimmed-upstream model reproduces the shipped vendored model **bit-for-bit** on a seeded CPU forward: `max|Δ| = 0.0e+00`. This proves the reconstruction is faithful and the vendored copy carries upstream behaviour verbatim.
- **Pass-through.** With `generate_graph` as the shim vs a hard stub returning the precomputed graph, the model output is **bit-identical**: `max|Δ| = 0.0e+00`. So `generate_graph` cannot alter any measured quantity. (Both on CPU; CUDA's atomic `index_add_` and the random frame make bit-identity meaningless on GPU.)
- **OCP source cross-check.** OCP `BaseModel.generate_graph` ([ocp @ 5a7738f, ocpmodels/models/base.py](https://github.com/Open-Catalyst-Project/ocp/blob/5a7738f9aa80b1a9a7e0ca15e33938b4d2557edd/ocpmodels/models/base.py)), for `otf_graph=False, use_pbc=True`, returns `edge_distance_vec = pos[j]-pos[i]+cell_offset` (via `get_pbc_distances`) and `edge_distance = ‖·‖`. Our `to_pyg_data` bakes that same periodic shift into the `edge_distance_vec` it supplies, so the shim's pass-through delivers the identical numerically-relevant outputs; OCP's other outputs (`cell_offsets`, `neighbors`) are unused downstream (verified) and are the ones the shim does not 
reproduce.

## Reruns on upstream source (trained checkpoints, CUDA)

E5 mirror/rotation/determinism and an EquiformerV2-specific FD-vs-autograd check, run on the shimmed-upstream model with the three trained checkpoints, vs the committed vendored E5:

| seed | mirror (upstream / vendored) | rotation (upstream / vendored) | determinism (upstream / vendored) | E3 FD rel. err |
|---|---|---|---|---|
| 0 | 0.982 / 0.982 | 0.073 / 0.073 | 0.111 / 0.111 | 0.36 |
| 1 | 0.929 / 0.929 | 0.094 / 0.094 | 0.143 / 0.143 | 0.84 |
| 2 | 0.713 / 0.713 | 0.106 / 0.106 | 0.144 / 0.144 | 0.66 |

Mirror, rotation, and determinism **match the committed vendored numbers to three decimals**. The E3 FD-vs-autograd disagreement is 0.36–0.84 (median 0.66), confirming the autograd Jacobian is substantially incomplete — the documented reason EquiformerV2 is excluded from E3. The exclusion now cites upstream at `8fe8cba`.

## Documentation check (Condition 4)

- **eSCN paper** ([arXiv:2302.03655](https://arxiv.org/abs/2302.03655)) describes the SO(2) reduction as *mathematically equivalent* convolutions — i.e. exact in theory; it does not discuss the per-edge frame randomness or an approximate-equivariance trade-off.
- **Upstream issue tracker** does acknowledge it: [#17 'Question about the edge_rot_mat'](https://github.com/atomicarchitects/equiformer_v2/issues/17) reports that randomly selecting the edge-frame axis gives different Wigner-D matrices for the same edge under translation, breaking equivariance for type>0 features; [#5 'Small equivariant example'](https://github.com/atomicarchitects/equiformer_v2/issues/5) reports forces not obeying rotational equivariance with stochastic components disabled.

### Settled framing

> At commit `8fe8cba` of `atomicarchitects/equiformer_v2` (whose `so3.py` derives from the > Meta OCP/fairchem eSCN lineage), the released model draws a random per-edge reference frame > on every forward and detaches the Wigner-D matrices from atomic positions. We **measure** > the resulting equivariance error — mirror-law violation O(1), rotation-law error 7–11%, and > a determinism spread of ~0.11–0.14 over five seeded evaluations — and an incomplete > autograd Jacobian (finite differences disagree by 36–84%). The eSCN paper describes the > SO(2) reduction as mathematically exact; the frame-induced equivariance breaking is > reported by users in the upstream issue tracker (#17, #5). We report these strictly as > measured properties of the released code at this commit, making no claim of 
incorrectness.
