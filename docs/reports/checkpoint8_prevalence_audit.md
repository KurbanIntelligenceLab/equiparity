# Table 1 — prevalence audit (Task 0.4)

Which released models attach a **parity label** to their internal features? Only those can
produce a structurally exact zero for a parity-odd tensor on a centrosymmetric crystal.

Every row was classified by reading the released source at the version or commit given.
The evidence column quotes the line that decides it. Where the source does not decide,
the row says `undetermined` rather than guessing.

## Summary

| category | n | consequence for a parity-odd tensor |
|---|---|---|
| `parity-aware` | 5 | odd-parity output is **structurally zero** on a centrosymmetric input |
| `SO(3)-only` | 3 | **nothing** forces an odd-parity output to vanish |
| `vector-only` | 3 | no typed irreps; a rank-3 head needs extra machinery |
| `invariant` | 3 | cannot express an equivariant output at all |
| `undetermined` | 1 | source does not settle it; needs a reflection test |

Of the 15 models surveyed, **5** carry parity
labels on their features. **9** do not, and one is undetermined. The
SO(3)-only group is not a fringe: it contains EquiformerV2 and the eSCN family, which are
the current state of the art on large-scale catalysis and materials benchmarks, and
Equiformer v1 as released.

## Rows

### `parity-aware` — odd-parity output is **structurally zero** on a centrosymmetric input

**NequIP** — nequip, `0.18.0`  
*Evidence* (`nequip/model/nequip_models.py`): `parity: bool = True` (line 120). Its own docstring (line 138) shows that with `parity=False` and `l_max=2` the features are still `5x0e, 2x1o, 7x2e` -- i.e. the flag does not strip parity labels. This corroborates our Checkpoint-1 finding that the boolean is NOT the SO(3) toggle; we build the SO(3) arm by relabelling irreps.

**Allegro** — nequip-allegro, `0.8.3`  
*Evidence* (`allegro/model/allegro_models.py`): `parity: bool = True` (line 36), 'whether to include features with odd mirror parity' (line 48), consumed at line 75. Features are e3nn irreps with parity labels.

**MACE** — mace-torch, `0.3.16`  
*Evidence* (`mace/modules/models.py`): `use_so3: bool = False` (line 67), stored at line 104. The default builds the edge spherical harmonics with natural parity `(-1)^l`; `use_so3=True` makes them all-even, which is the correct native SO(3) toggle and is what our matched pair uses.

**MatTen** — wengroup/matten, `commit 0a04f1c`  
*Evidence* (`src/matten/model_factory/tfn_scalar_tensor.py`): Default `conv_layer_irreps: '32x0o + 32x0e + 16x1o + 16x1e'` (line 206) and `irreps_edge_sh: '0e + 1o'` (line 207). Both parities are present and explicitly labelled; this is an O(3) model (it targets the even-parity elasticity tensor).

**ICTP** — nec-research/ICTP, `commit f40592a`  
*Evidence* (`ictp/o3/cartesian_harmonics.py`): Uses its own `CartesianHarmonics(l_max)` (line 253) rather than e3nn irreps. A rank-l Cartesian harmonic is a homogeneous degree-l polynomial in the displacement vector, so under inversion it picks up `(-1)^l` -- parity is carried implicitly and correctly, as in MACE. Classified parity-aware on that construction; a reflection test on the built model would confirm it directly and is the honest next step.

### `SO(3)-only` — **nothing** forces an odd-parity output to vanish

**Equiformer (v1)** — atomicarchitects/equiformer, `commit 64cb786`  
*Evidence* (`nets/graph_attention_transformer.py`): Every shipped default node embedding is all-even: `irreps_node_embedding='128x0e+64x1e+32x2e'` (line 739; likewise dp_attention_transformer.py:261, equiformer_md17_dens.py:59) and `'256x0e+128x1e'` for OC20 (graph_attention_transformer_oc20.py:89). Odd irreps appear only in an `irreps_head` and in auxiliary force outputs (dp_attention_transformer_oc20.py:182). As released, Equiformer v1's node features carry no odd-parity channel.

**EquiformerV2** — vendored (src/equiparity/models/equiformer_v2), `as vendored in this repo`  
*Evidence* (`src/equiparity/models/equiformer_v2/so3.py`): `SO3_Embedding` is indexed by `lmax_list`/`mmax_list` only; `so3.py` contains zero `Irreps` constructions and no parity anywhere. Confirmed behaviourally in E5: it violates the mirror law by O(1). Separately, its rotation equivariance is only approximate (edge_rot_mat.py draws a random per-edge frame each forward).

**eSCN** — FAIR-Chem/fairchem, `commit a838178`  
*Evidence* (`fairchem (escn)`): eSCN is the SO(2)-reduced convolution underlying EquiformerV2; its embeddings are `lmax`/`mmax`-typed with no parity label. Same family, same conclusion.

### `vector-only` — no typed irreps; a rank-3 head needs extra machinery

**PaiNN** — torchmd-net (torchmdnet/models), `commit 2a2c913`  
*Evidence* (`torchmdnet/models/torchmd_et.py`): Equivariance is carried by an explicit Cartesian vector channel `vec = torch.zeros(x.size(0), 3, x.size(1))` (line 213), updated alongside scalars. No `Irreps` anywhere in `torchmdnet/models/`. A vector channel is an l=1 object; the code does not label its parity, and there is no rank-3 path.

**TorchMD-Net (ET)** — torchmd-net, `commit 2a2c913`  
*Evidence* (`torchmdnet/models/torchmd_et.py`): Same Cartesian `vec` channel (line 213), `dx, dvec = attn(...)` (line 216). No typed irreps; parity is never represented.

**ViSNet** — torch_geometric.nn.models.visnet, `torch-geometric 2.8.0`  
*Evidence* (`torch_geometric/nn/models/visnet.py`): Maintains a Cartesian vector channel (`vec = torch.cat([vec1, vec2], dim=1)`, line 269); zero references to `Irreps` or `o3`.

### `invariant` — cannot express an equivariant output at all

**SchNet** — torch_geometric.nn.models.schnet, `torch-geometric 2.8.0`  
*Evidence* (`torch_geometric/nn/models/schnet.py`): Interactions consume only `edge_weight` (interatomic distances); the file contains zero references to `Irreps` or `o3`. A distance-only network is invariant under every element of O(3), so it cannot emit a tensor of any rank.

**DimeNet++** — torch_geometric.nn.models.dimenet, `torch-geometric 2.8.0`  
*Evidence* (`torch_geometric/nn/models/dimenet.py`): Message passing is over distances and bond angles; zero references to `Irreps` or `o3`. Angles are invariant under reflection, so the representation cannot distinguish a structure from its mirror image.

**FAENet** — vict0rsch/faenet, `commit 1f725cf`  
*Evidence* (`faenet/tests/test_model_symmetries.py`): The network itself is not equivariant; symmetry is imposed externally by frame averaging (`frame_averaging` in {'', '2D', '3D', 'DA'}). Parity is a property of the chosen frame set, not of the features. Reflections enter only if the frame set includes them.

### `undetermined` — source does not settle it; needs a reflection test

**GotenNet** — sarpaykent/GotenNet, `commit 44c945b`  
*Evidence* (`gotennet/models/representation/gotennet.py`): Edge features come from `e3nn.o3.Irreps.spherical_harmonics(lmax)` (line 860), which does carry natural parity `(-1)^l`. But the node tensor features `X` are propagated through the model's own (non-e3nn) `GATA` attention blocks, which are not parity-typed. Source inspection does not settle whether an odd output is structurally zero on a centrosymmetric input. **A reflection test on the built model is required before this row can be filled in; we do not guess.**

## What this table is for

The introduction claims that the parity distinction is (a) invisible in most model cards and
(b) load-bearing for tensor properties. This table supports (a) by showing that the two most
widely deployed equivariant transformers in this list are SO(3)-only, and that the
distinction is nowhere in their configuration surface — EquiformerV2 exposes `lmax`, not a
parity flag. NequIP is the sharpest case: it *has* a `parity` boolean, and that boolean does
not do what its name suggests (its own docstring shows `1o` features surviving
`parity=False`). Our SO(3) arms are therefore built by relabelling irreps, not by flipping
that flag — see `docs/reports/checkpoint1_offcycle_parity_toggle.md`.

## Honesty notes

- `GotenNet` is left **undetermined on purpose**. Its edge spherical harmonics carry natural
  parity, but its node features flow through custom attention blocks that are not
  parity-typed. Deciding this requires building the model and running a reflection test, not
  reading it.
- `ICTP` is classified `parity-aware` from a *construction* argument (rank-l Cartesian
  harmonics scale as `(-1)^l` under inversion), not from an explicit parity label. That
  argument is sound but is weaker evidence than an `Irreps` string; a reflection test would
  settle it.
- `eSCN` is classified by family resemblance to EquiformerV2 rather than by a decisive line
  of its own. It shares the `lmax`/`mmax` SO(3) embedding.
- `FAENet` is listed `invariant` because the *network* is; its symmetry comes from frame
  averaging applied outside the model, and whether that includes improper operations is a
  choice made at the call site.
