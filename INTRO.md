# Introduction

Equivariant neural networks have become the standard tool for predicting properties of molecules and
crystals. Most are described as "E(3)-" or "O(3)-equivariant", but a widely used subclass — including
several deployed state-of-the-art models — is equivariant only to **rotations**, SO(3), and not to
**reflections**. The distinction is usually treated as an implementation detail. This study asks what it
costs.

## The physical constraint

Neumann's principle requires that a crystal's macroscopic property tensors be invariant under every
symmetry operation of its point group. The piezoelectric tensor `e_{ijk}` is parity-**odd**: under
spatial inversion it changes sign. If a crystal is centrosymmetric — its space group contains the
inversion operation — Neumann's principle demands `e = -e`, and therefore

> **e = 0 exactly, for every centrosymmetric crystal.**

Of the 230 space groups, 92 are centrosymmetric. This gives an unusually clean test. It requires no
ground-truth labels: any nonzero prediction on a centrosymmetric crystal is, with certainty, a
physically impossible one.

## The mechanism under test

An O(3)-equivariant network attaches a parity label to every internal feature. The composition rules of
those labels force an odd-parity output to vanish on a centrosymmetric input. The zero is *structural*:
it holds for any weights, including at random initialisation, and cannot be trained away.

An SO(3)-equivariant network is identical except that the parity labels are removed. It remains exactly
rotation-equivariant, and in practice loses little accuracy on standard benchmarks. But nothing forces
its odd-parity output to zero.

## Design

We hold architecture, hyperparameters, data, and random seeds fixed, and vary **only** the parity
labelling. Three architectures — NequIP, Allegro, and MACE — are built as matched O(3)/SO(3) pairs. A
fourth, EquiformerV2, is included unmodified as a deployed model that is SO(3)-only by construction and
has no O(3) arm.

Four targets span the parity spectrum, so that "odd-parity target" and "symmetry-forbidden value" can be
separated rather than confounded:

| Target | Parity | Is a zero ever symmetry-mandated on the eval set? |
|---|---|---|
| QM9 U₀ | even (scalar) | no |
| QM9 dipole μ | **odd** (vector) | no — QM9 has no molecule whose symmetry forces μ = 0 |
| MP elastic C | even (rank 4) | no |
| MP piezoelectric e | **odd** (rank 3) | **yes** — on 2,000 centrosymmetric crystals |

The full grid is 7 arms × 4 targets × 3 seeds = 84 runs. The parity toggle is verified numerically —
by reflection and rotation tests on internal features — before any training, because a mislabelled arm
would invalidate everything downstream.

Every model is evaluated on the centrosymmetric set in two coordinate variants: idealized to the exact
space group, and as-relaxed by DFT. The first tests the structural guarantee; the second tests behaviour
on real data, whose coordinates satisfy inversion symmetry only to within a tolerance.

## Reading this repository

| Document | Contents |
|---|---|
| [`METHODS.md`](METHODS.md) | data, models, parity toggle, metrics, training protocol, verification gate, reproducibility |
| [`RESULTS.md`](RESULTS.md) | all measurements, as tables, with limitations |
| [`docs/results/`](docs/results/) | appendices: per-run values, threshold curves, distributions, compute, symmetry audit |
| [`docs/checkpoint7_report.md`](docs/checkpoint7_report.md) | review response; excluded models; corrections log |
| [`docs/parity_work_plan.md`](docs/parity_work_plan.md) | the original scientific plan |

Figures are not embedded in `RESULTS.md`; the underlying numbers are given as tables, and
`results/threshold_curves.csv` holds the curve data in machine-readable form.
