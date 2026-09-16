# Lean proofs of the parity-gap guarantees

This project formalizes the mathematical guarantees of **The parity gap in crystal tensor prediction**. It contains 9 theorem declarations, with Lean 4 and mathlib pinned to `v4.33.1`. The exact mathlib revision and its transitive dependencies are recorded in `lake-manifest.json`.

## Reproduce

Install [Lean's elan toolchain manager](https://lean-lang.org/install/manual/), then run from this directory:

```sh
lake exe cache get
python3 verify.py
```

The first command obtains mathlib's precompiled dependencies. The verification command builds the proof library, checks that every theorem is covered by the axiom audit, and rejects unfinished proofs or nonstandard axiom dependencies. It does not require Python packages, a GPU, datasets or trained models. Alternatively, run `lake build` and `lake env lean Audit.lean` separately.

## Theorem-to-manuscript map

The mathematical source is [ParityGap/Guarantees.lean](ParityGap/Guarantees.lean).

| Manuscript argument | Lean declarations | Scope |
|---|---|---|
| Periodic input identity | `periodic_inversion_identity` | Species-preserving relabelling, individual lattice shifts and arbitrary rigid translations identify inverted and original inputs. |
| Inversion sign and forced zero | `polarInversion_sign`, `symmetry_forced_zero` | Cartesian rank sign and zero from input invariance plus odd output parity. |
| Proper-rotation constraint | `value_in_fixedSpace` | Equivariant values lie in the stabilizer-invariant space. |
| Attainability | `invariant_value_attainable` | The orbit construction is well defined and equivariant; its supporting lemmas are local proof steps. |
| Parity gap | `parity_gap_even_rank`, `parity_gap_odd_rank` | Both invariant-space and dimension conclusions, under the explicit polar inversion law and the even-case coset cover. |
| Output enforcement | `antisymmetrize_zero`, `antisymmetrize_odd` | Cancellation on invariant inputs and odd parity for involutive inversion. |

The even-rank theorem exposes equality of invariant spaces as well as the zero gap. The declaration inventory and `Audit.lean` are checked together by `verify.py`.

## Scope and assumptions

The input identity is proved directly for coordinate functions, a species-preserving relabelling, an additive lattice subgroup and the two translation invariances. It does not assume the desired output identity as its conclusion. An involutive atom permutation is not needed for that equality; the manuscript uses the species-preserving relabelling directly.

The invariant-space and attainability results use an arbitrary group action and a real linear representation. To interpret the action as the physical one, inputs are structures modulo the stated permutations and translations; the stabilizer then represents the proper point group. The formalization does not build a separate crystallographic quotient or derive a space group from atom coordinates. The rank-specific results take the coset cover and polar inversion law as explicit hypotheses. The Cartesian sign lemma checks the inversion factor, but does not formalize the entire orthogonal-group tensor-power representation. Choosing an invariant physical index-symmetry subspace as `V` preserves the applicability of the generic theorems.

The orbit extension proves existence among unrestricted equivariant functions. It asserts neither continuity nor realizability by the tested neural networks. The dimension proofs use invariant spaces directly; this matches the manuscript's direct invariant-space argument. The finite Reynolds-projector calculations used for the class table remain separate computational checks. Computed point-group classifications, the dimensions printed in Table 1, floating-point behavior, and empirical measurements remain separate computational checks.

`Audit.lean` prints the transitive axiom dependencies of all 9 theorems. They use only Lean's standard `propext`, `Classical.choice` and `Quot.sound` foundations as needed. There are no unfinished proofs, custom axioms or native-evaluation shortcuts. A successful build checks the stated formal propositions; the theorem map and explicit hypotheses establish what is being claimed about the manuscript.

The project uses [Lean 4](https://doi.org/10.1007/978-3-030-79876-5_37) and [mathlib](https://doi.org/10.1145/3372885.3373824). It is covered by the repository's MIT license.
