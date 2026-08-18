# F3 — the size-consistency (supercell) control, in numbers

Every readout in this study sums per-atom (NequIP, MACE) or per-edge (Allegro) contributions via `index_add_`, which is extensive: the predicted tensor scales with the number of unit cells in the supercell, not just with the crystal it represents. This checks that claim directly, and checks the mean-pooled alternative a reviewer proposed.

**Route.** `data/raw/mp/mp_ood_centrosymmetric_processed.npz` is not present in this checkout (it requires a live Materials Project API fetch via `scripts/prepare_mp.py`). Built 9 idealized centrosymmetric crystals directly with `ase.spacegroup.crystal` instead — exact space-group symmetry by construction, the same convention `scripts/idealize_ood.py` applies to the MP-derived OOD set. Space groups: 136, 141, 167, 221, 225, 227 (6 cubic *m*-3*m* crystals, 3 non-cubic: rutile/anatase TiO2, corundum Al2O3). Cores: Allegro, MACE, NequIP (three matched-pair cores, each in its own dedicated env — NequIP/Allegro share e3nn 0.6.x, MACE needs e3nn 0.4.4), random init, seed 42, float64, `r_max = 5.0` Å. Both the O(3) and SO(3) arms of each core were run.

**Supercells.** For each crystal, 2×2×2 (8 replicas), 2×1×1 (2 replicas), and 3×1×1 (3 replicas) were built from the primitive cell with `ase`'s `atoms * (nx,ny,nz)`, preserving periodicity. The periodic neighbor list gives an identical edges-per-atom count at every multiplicity (checked explicitly, e.g. rutile: 51.3333 in the primitive cell and in every supercell), confirming every atom of the supercell has an environment identical to its primitive-cell counterpart — the premise the ratio law below rests on.

## The predicted scaling law, measured

The cubic (*m*-3*m*) crystals drive even the SO(3) arm's random-init output to the machine-precision noise floor (the same rotation-ceiling effect documented in `scripts/t2_backbone_probe.py`: an exactly rotation-equivariant model has no rank-3 invariant available in the cubic point group, so its response is forced near zero independent of parity). A supercell/primitive ratio of noise over noise is uninformative, so those crystals are flagged rather than used to test the pooling law. On the resolved, non-cubic subset (rutile TiO2, anatase TiO2, corundum Al2O3), where the SO(3) primitive-cell response is well above the noise floor:

| core | resolved crystals | unresolved (noise floor) | max &#124;SO(3) sum ratio − replicas&#124; | max &#124;SO(3) mean ratio − 1&#124; |
|---|---|---|---|---|
| Allegro | 3 (TiO2_rutile, TiO2_anatase, Al2O3_corundum) | 6 | 4.80e-14 | 6.22e-15 |
| MACE | 3 (TiO2_rutile, TiO2_anatase, Al2O3_corundum) | 6 | 6.57e-14 | 1.20e-14 |
| NequIP | 3 (Al2O3_corundum, TiO2_anatase, TiO2_rutile) | 6 | 7.11e-15 | 8.88e-16 |

All deviations are at the ~1e-14 level — floating-point round-off, not a real discrepancy — and agree across three independent, differently-implemented cores. **Sum pooling reproduces the replica count exactly** (2, 3, or 8×); **mean pooling reproduces exactly 1** (no scaling with cell size) — confirming the reviewer's predicted algebraic identity.

### Full per-crystal ratios (resolved subset)

| core | crystal | multiplicity | replicas | SO(3) sum ratio | SO(3) mean ratio |
|---|---|---|---|---|---|
| Allegro | TiO2_rutile | 2x1x1 | 2 | 2.000000 | 1.000000 |
| Allegro | TiO2_rutile | 3x1x1 | 3 | 3.000000 | 1.000000 |
| Allegro | TiO2_rutile | 2x2x2 | 8 | 8.000000 | 1.000000 |
| Allegro | TiO2_anatase | 2x1x1 | 2 | 2.000000 | 1.000000 |
| Allegro | TiO2_anatase | 3x1x1 | 3 | 3.000000 | 1.000000 |
| Allegro | TiO2_anatase | 2x2x2 | 8 | 8.000000 | 1.000000 |
| Allegro | Al2O3_corundum | 2x1x1 | 2 | 2.000000 | 1.000000 |
| Allegro | Al2O3_corundum | 3x1x1 | 3 | 3.000000 | 1.000000 |
| Allegro | Al2O3_corundum | 2x2x2 | 8 | 8.000000 | 1.000000 |
| MACE | TiO2_rutile | 2x1x1 | 2 | 2.000000 | 1.000000 |
| MACE | TiO2_rutile | 3x1x1 | 3 | 3.000000 | 1.000000 |
| MACE | TiO2_rutile | 2x2x2 | 8 | 8.000000 | 1.000000 |
| MACE | TiO2_anatase | 2x1x1 | 2 | 2.000000 | 1.000000 |
| MACE | TiO2_anatase | 3x1x1 | 3 | 3.000000 | 1.000000 |
| MACE | TiO2_anatase | 2x2x2 | 8 | 8.000000 | 1.000000 |
| MACE | Al2O3_corundum | 2x1x1 | 2 | 2.000000 | 1.000000 |
| MACE | Al2O3_corundum | 3x1x1 | 3 | 3.000000 | 1.000000 |
| MACE | Al2O3_corundum | 2x2x2 | 8 | 8.000000 | 1.000000 |
| NequIP | Al2O3_corundum | 2x1x1 | 2 | 2.000000 | 1.000000 |
| NequIP | Al2O3_corundum | 3x1x1 | 3 | 3.000000 | 1.000000 |
| NequIP | Al2O3_corundum | 2x2x2 | 8 | 8.000000 | 1.000000 |
| NequIP | TiO2_anatase | 2x1x1 | 2 | 2.000000 | 1.000000 |
| NequIP | TiO2_anatase | 3x1x1 | 3 | 3.000000 | 1.000000 |
| NequIP | TiO2_anatase | 2x2x2 | 8 | 8.000000 | 1.000000 |
| NequIP | TiO2_rutile | 2x1x1 | 2 | 2.000000 | 1.000000 |
| NequIP | TiO2_rutile | 3x1x1 | 3 | 3.000000 | 1.000000 |
| NequIP | TiO2_rutile | 2x2x2 | 8 | 8.000000 | 1.000000 |

## The structural zero survives both pooling modes

On every crystal (all 9, both cubic and non-cubic), at every multiplicity, the O(3) arm's predicted piezoelectric tensor stays at the float64 noise floor under **both** sum and mean pooling:

| core | max &#124;O(3)&#124; over all crystals/multiplicities, sum pooling | max &#124;O(3)&#124;, mean pooling |
|---|---|---|
| Allegro | 4.72e-13 | 7.28e-17 |
| MACE | 4.58e-14 | 1.91e-16 |
| NequIP | 1.84e-17 | 1.29e-19 |

The SO(3) arm, by contrast, is nonzero on the resolved non-cubic crystals under both poolings (e.g. Allegro on rutile TiO2, primitive cell: SO(3) sum norm 8.554, clearly above the noise floor) — confirming the O(3)/SO(3) contrast is not an artifact of which pooling is used.

## Reading

Size consistency across primitive-cell and supercell representations is a property of the **readout** (sum vs. mean over `index_add_`), not of parity. A mean-pooled readout would give the same prediction for a primitive cell and any supercell of it; the sum-pooled readout used throughout the main analysis does not, and scales exactly with the replica count. This is orthogonal to Theorem 1: the structural zero on centrosymmetric crystals holds under either pooling, because it follows from the parity labeling of the irreps, not from how the per-atom or per-edge contributions are aggregated. The main-text false-flag numbers (computed with the sum-pooled readout, trained models) are therefore not biased by an untested size-consistency assumption in the direction that would matter for the paper's claim: the O(3) zero is exact regardless, and the SO(3) false-flag is nonzero regardless.

**Provenance.** 9 idealized crystals (Al2O3_corundum, CaF2_fluorite, CsCl, MgO_rocksalt, NaCl_rocksalt, Si_diamond, SrTiO3_perovskite, TiO2_anatase, TiO2_rutile), route `idealized_ase_spacegroup_crystal`, cores allegro, mace, nequip, seed 42, float64, random init (no training — Theorem 1 holds at any parameter values).
