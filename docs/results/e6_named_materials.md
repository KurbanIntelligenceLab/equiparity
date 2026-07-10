# E6 — familiar centrosymmetric materials

The true piezoelectric tensor of every crystal below is **exactly zero**: each is
centrosymmetric, and the tensor is parity-odd. Values are ‖T‖_F, mean over 3 seeds.
Entries are annotated with the proper-rotation subgroup of their point group, 
because that
decides whether SO(3) equivariance *alone* already forbids a response (E7).

| formula | mp-id | structure | SG | point group | rotation subgroup | source | NequIP O(3) | NequIP SO(3) | Allegro SO(3) | MACE SO(3) | EquiformerV2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Al2O3 | mp-1143 | corundum | 167 | non-cubic | permits rank-3 | fresh MP | 2.70e-07 | 1.128 | 0.723 | 0.837 | 0.212 |
| TiO2 | mp-2657 | rutile | 74 | non-cubic | permits rank-3 | fresh MP | 2.84e-07 | 0.260 | 0.582 | 0.245 | 0.187 |
| C | mp-66 | diamond | 227 | m-3m | 432 — forbids rank-3 | OOD 2000 | 5.12e-07 | 0.000 | 0.000 | 0.000 | 0.033 |
| CaF2 | mp-2741 | fluorite | 225 | m-3m | 432 — forbids rank-3 | fresh MP | 3.97e-07 | 0.000 | 0.000 | 0.000 | 0.008 |
| CsCl | mp-22865 | caesium chloride | 221 | m-3m | 432 — forbids rank-3 | fresh MP | 3.31e-09 | 0.000 | 0.000 | 0.000 | 0.000 |
| KCl | mp-23193 | rocksalt | 225 | m-3m | 432 — forbids rank-3 | OOD 2000 | 2.95e-09 | 0.000 | 0.000 | 0.000 | 0.000 |
| MgO | mp-1265 | rocksalt | 225 | m-3m | 432 — forbids rank-3 | fresh MP | 5.74e-08 | 0.000 | 0.000 | 0.000 | 0.000 |
| NaCl | mp-22862 | rocksalt | 225 | m-3m | 432 — forbids rank-3 | fresh MP | 5.40e-08 | 0.000 | 0.000 | 0.000 | 0.000 |
| Si | mp-149 | diamond | 227 | m-3m | 432 — forbids rank-3 | fresh MP | 4.77e-08 | 0.000 | 0.000 | 0.000 | 0.004 |
| SrTiO3 | mp-5229 | perovskite | 221 | m-3m | 432 — forbids rank-3 | fresh MP | 1.35e-07 | 0.000 | 0.000 | 0.000 | 0.011 |

## Reading

The O(3) column is at machine zero for every material, as it must be. The SO(3) columns
predict a nonzero piezoelectric response for materials that cannot have one — 
with a clean
exception that is itself the point: **the m-3̄m entries (diamond, KCl, NaCl, 
MgO, CaF₂,
CsCl, SrTiO₃, Si) are near zero for the exact-SO(3) cores too**, because their rotation
subgroup 432 admits no rank-3 tensor at all. Rotation equivariance already forbids a
response there; parity is not doing the work.

The materials where parity *is* the only thing standing between the model and 
a physically
impossible prediction are the non-cubic ones. Among the familiar compounds those are
**corundum Al₂O₃ (sapphire)** and **TiO₂**. Every SO(3) model predicts a 
substantial
piezoelectric response for sapphire — NequIP 1.13, Allegro 0.72, MACE 0.84, 
EquiformerV2
0.21 — for a crystal whose response is exactly zero by symmetry. Those are 
the rows to quote.

## Caveats

**TiO₂ (mp-2657) is rutile, but its DFT-relaxed coordinates refine to Imma 
(74) at symprec
1e-3**, recovering rutile's P4₂/mnm (136) only at symprec 1e-2. Both groups are
centrosymmetric and non-cubic, so the zero and its interpretation are 
unaffected; the space
group column reports what spglib actually finds at the study's tolerance. 
This is the same
tolerance phenomenon as the mp-1227949 raw-coordinate artifact (appendix A5). 
The idealized
rutile used in E2 is built analytically, not from MP, and is exactly 136.

**Germanium is absent.** MP's GGA band gap for Ge is ~0 eV, below the 0.1 eV 
cut that defines
this study's insulator population. Silicon stands in for the 
diamond-structure semiconductor.

EquiformerV2 values are means over 5 seeded draws (its forward is stochastic; see E5).
