# E3 — the Jacobian of the parity guarantee

`J = dT/dr` by autograd at the centrosymmetric geometry of 20 OOD crystals,
for every trained arm of the three e3nn cores, 3 seeds each.

For an O(3) model `T(I·x) = −T(x)`, so differentiating at a centrosymmetric point gives
`J∘P = −J`, where `(Pu)_i = −u_{σ(i)}`. Every inversion-even displacement then lies in
`ker J`: the learned response to symmetry breaking is supported entirely on inversion-odd
(polar) modes. The guarantee is differentiable, and its derivative is physically structured.

## Even-subspace energy fraction

`f = ‖J·P_even‖_F / ‖J‖_F` with `P_even = (id + P)/2`. Exactly 0 for O(3) by the identity
above; basis-independent; no singular-vector truncation needed.

| core | arm | median f | min | max | median &#124;J∘P + J&#124;/&#124;J&#124; | median rank | median FD rel. err |
|---|---|---|---|---|---|---|---|
| allegro | o3 | 1.654e-07 | 5.620e-08 | 4.805e-07 | 3.307e-07 | 13 | 9.38e-05 |
| allegro | so3 | 5.192e-01 | 1.504e-01 | 7.754e-01 | 1.059e+00 | 18 | 1.05e-04 |
| mace | o3 | 9.690e-07 | 2.171e-07 | 2.335e-06 | 1.696e-06 | 13 | 1.35e-04 |
| mace | so3 | 5.393e-01 | 8.612e-02 | 8.128e-01 | 1.165e+00 | 18 | 1.13e-04 |
| nequip | o3 | 1.388e-07 | 2.079e-08 | 4.135e-07 | 2.826e-07 | 13 | 6.66e-05 |
| nequip | so3 | 4.202e-01 | 1.046e-01 | 8.805e-01 | 8.495e-01 | 18 | 6.62e-05 |

## Parity scores of the top-5 singular vectors (secondary)

`s = ⟨u, Pu⟩ / (‖u‖‖Pu‖)`; `s = −1` is a purely inversion-odd displacement pattern.

| core | arm | median top-5 score | min | max |
|---|---|---|---|---|
| allegro | o3 | -1.000000 | -1.000000 | -1.000000 |
| allegro | so3 | -0.621868 | -1.000000 | 1.000000 |
| mace | o3 | -1.000000 | -1.000000 | -1.000000 |
| mace | so3 | -0.730861 | -1.000000 | 1.000000 |
| nequip | o3 | -1.000000 | -1.000000 | -1.000000 |
| nequip | so3 | -0.868120 | -1.000000 | 1.000000 |

## Reading

The O(3) arms satisfy `J∘P = −J` to float precision and put a vanishing fraction of the
Jacobian's energy on inversion-even modes. The SO(3) arms put an O(0.1–1) fraction there:
their derivative responds to displacements that cannot produce a piezoelectric response.

The per-vector parity scores separate the arms only partially. Every O(3) singular vector
scores exactly −1.00000 (100% of them within 1e-2 of −1), as the theorem demands. But a
*trained* SO(3) model is approximately odd: its scores have median ≈ −0.76 and 17% of its
leading vectors also sit within 1e-2 of −1. The design expectation that SO(3) scores would
be 'broadly distributed' across [−1, +1] holds for random weights (the V0 toy), not for
trained ones. The energy fraction is the statistic with a clean margin: ~1e-7 versus ~0.5,
five to six orders of magnitude, with no overlap across any structure or seed.

**EquiformerV2 is excluded.** `models/equiformer_v2/so3.py` detaches the Wigner-D matrices,
which depend on atomic positions, so autograd yields only the radial part of `dT/dr` — finite
differences disagree by ~45%, while the same check gives ~9e-4 on NequIP. Its forward pass is
also stochastic (E5). Recovering its Jacobian would require editing vendored rotation code and
ensemble-averaging; out of scope. The `median FD rel. err` column above is the guard that
every Jacobian reported here is the true one.
