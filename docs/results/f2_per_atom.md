# F2 — the extensivity caveat, in numbers

`‖T‖_F` is built by summing atomic contributions, so it is **extensive**: larger crystals
produce larger violations, and the fixed 0.01 C/m² threshold is therefore size-dependent.
The OOD set's median cell has **28 atoms**.

Below, the headline false-flag rate is recomputed with the size-normalised metric
`‖T‖_F / n_atoms` against a threshold rescaled by the median atom count
(`0.01 / 28` = `3.571e-04`), so the two agree exactly
on a
median-sized crystal. Mean ± std over 3 seeds, idealized variant.

| core | arm | false-flag (absolute ‖T‖) | false-flag (‖T‖/n_atoms) | Δ |
|---|---|---|---|---|
| NequIP | O3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | +0.0000 |
| NequIP | SO3 | 0.8953 ± 0.0008 | 0.9090 ± 0.0000 | +0.0137 |
| Allegro | O3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | +0.0000 |
| Allegro | SO3 | 0.9095 ± 0.0015 | 0.9140 ± 0.0013 | +0.0045 |
| MACE | O3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | +0.0000 |
| MACE | SO3 | 0.9077 ± 0.0008 | 0.9120 ± 0.0013 | +0.0043 |
| EquiformerV2 | SO3 | 0.9570 ± 0.0052 | 0.9747 ± 0.0065 | +0.0177 |

## Reading

Every O(3) arm stays at exactly **0.0000** under both metrics (3 arms). The SO(3) arms move by at most **0.0177** in false-flag rate. The headline conclusion —
O(3) produces structural zeros, SO(3) false-flags ~90% of centrosymmetric crystals — is
unchanged by size normalisation.

This is expected rather than lucky: the O(3) zeros are exact to machine precision, so
no rescaling of a threshold can move them. The caveat matters for interpreting the
*magnitude* of an SO(3) violation, not for whether it is nonzero.
