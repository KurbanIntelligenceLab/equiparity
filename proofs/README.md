# Lean proofs of the parity-gap guarantees

This project formalizes the abstract mathematical guarantees retained in **The parity gap in crystal tensor prediction**. It contains 17 checked theorems, with Lean 4 and mathlib pinned to `v4.33.1`. The exact mathlib revision and its transitive dependencies are recorded in `lake-manifest.json`.

## Reproduce

Install [Lean's elan toolchain manager](https://lean-lang.org/install/manual/), then run from this directory:

```sh
lake exe cache get
python3 verify.py
```

The first command obtains mathlib's precompiled dependencies. The verification command builds the proof library, checks that every theorem is covered by the axiom audit, and rejects unfinished proofs or nonstandard axiom dependencies. It does not require Python packages, a GPU, datasets or trained models. Alternatively, run `lake build` and `lake env lean Audit.lean` separately.

## Theorem-to-manuscript map

The mathematical source is [ParityGap/Guarantees.lean](ParityGap/Guarantees.lean).

| Manuscript result | Lean declarations | Formal meaning |
|---|---|---|
| SI inversion identity, assumptions A1–A2 | `periodic_inversion_identity` | A species-preserving permutation, individual shifts in an additive lattice subgroup and a rigid translation identify the inverted coordinates with the original input. The inversion centre is arbitrary. |
| Theorem 1(i), symmetry-forced zero | `eq_neg_forces_zero`, `symmetry_forced_zero` | Input invariance together with odd output parity forces the output to zero over a real vector space. |
| Polar tensor inversion sign | `polarInversion_sign` | Multiplying all Cartesian tensor indices by the inversion sign gives the factor `(-1)^r`. |
| Theorem 1(ii), allowed invariant space | `value_in_fixedSpace`, `fixedSpace_full_le` | An equivariant function's value belongs to the input stabilizer's invariant space; full-group invariance implies subgroup invariance. |
| Theorem 1(ii), attainability | `orbit_value_well_defined`, `orbitExtension_on_orbit`, `invariant_value_attainable` | Every stabilizer-invariant vector is attained by an equivariant function defined along the orbit and as zero elsewhere. The function is defined on an abstract input space with a group action. |
| Theorem 1(iii), parity gap | `fixedSpace_even`, `fixedSpace_odd`, `gap_even`, `gap_odd`, `parity_gap_even_rank`, `parity_gap_odd_rank` | Under the explicit proper/inversion coset cover and polar inversion law, even-rank full/proper invariant spaces coincide; odd-rank full invariants vanish. Their finite-dimensional difference is consequently zero or the full proper-invariant dimension. |
| SI output inversion projection | `antisymmetrize_zero`, `antisymmetrize_odd` | The half-difference vanishes under the input identity and has odd parity for an involutive inversion map. |

## Scope and assumptions

The input identity is proved directly for coordinate functions, a species-preserving relabelling, an additive lattice subgroup and the two translation invariances. It does not assume the desired output identity as its conclusion. An involutive atom permutation is not needed for that equality; the manuscript's centrosymmetry definition supplies a sufficient special case.

The invariant-space and attainability results use an arbitrary group action and a real linear representation. To interpret the action as the physical one, inputs are structures modulo the stated permutations and translations; the stabilizer then represents the proper point group. The formalization does not build a separate crystallographic quotient or derive a space group from atom coordinates. The rank-specific results take the coset cover and polar inversion law as explicit hypotheses. The Cartesian sign lemma checks the inversion factor, but does not formalize the entire orthogonal-group tensor-power representation. Choosing an invariant physical index-symmetry subspace as `V` preserves the applicability of the generic theorems.

The orbit extension proves existence among unrestricted equivariant functions. It asserts neither continuity nor realizability by the tested neural networks. The dimension proofs use invariant spaces directly; they do not reproduce the manuscript's intermediate finite Reynolds-sum calculation. Computed point-group classifications, the dimensions printed in Table 1, floating-point behavior, and empirical measurements remain separate computational checks. Retired derivative and residual-bound branches are outside this project.

`Audit.lean` prints the transitive axiom dependencies of all 17 theorems. They use only Lean's standard `propext`, `Classical.choice` and `Quot.sound` foundations as needed. There are no unfinished proofs, custom axioms or native-evaluation shortcuts. A successful build checks the stated formal propositions; the theorem map and explicit hypotheses establish what is being claimed about the manuscript.

The project uses [Lean 4](https://doi.org/10.1007/978-3-030-79876-5_37) and [mathlib](https://doi.org/10.1145/3372885.3373824). It is covered by the repository's MIT license.
