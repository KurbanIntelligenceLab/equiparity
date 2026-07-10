"""V0 — every physics claim this study relies on, checked rather than asserted.

Each test corresponds to a claim in ``docs/new_additions.md`` or a correction found while
validating it. Nothing here loads a trained checkpoint; these run on CPU in seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from equiparity.domain.parity import ParityMode
from equiparity.domain.target import TARGETS
from equiparity.inference.structures import (
    max_displacement_angstrom,
    perovskite,
    tetragonal_distortion,
)
from equiparity.models.irreps import degree_irreps, output_irreps

e3nn = pytest.importorskip("e3nn")
spglib = pytest.importorskip("spglib")

PIEZO_IRREPS = "2x1o+1x2o+1x3o"


# --------------------------------------------------------------------------- claim 1
def test_piezoelectric_irreps_are_all_parity_odd() -> None:
    """The piezoelectric tensor decomposes into odd irreps only; that is why inversion kills it."""
    from e3nn import o3

    assert TARGETS["piezoelectric"].irreps.replace(" ", "") == PIEZO_IRREPS
    for multiplicity, irrep in o3.Irreps(PIEZO_IRREPS):
        assert irrep.p == -1, f"{irrep} is not parity-odd"
        assert multiplicity >= 1


def test_so3_output_head_relabels_odd_irreps_even() -> None:
    """The SO(3) arm differs from O(3) only by stripping parity labels off the head."""
    assert output_irreps(PIEZO_IRREPS, ParityMode.O3) == PIEZO_IRREPS
    assert output_irreps(PIEZO_IRREPS, ParityMode.SO3) == "2x1e+1x2e+1x3e"
    assert degree_irreps(2, 4, ParityMode.O3) == "4x0e + 4x1o + 4x2e"
    assert degree_irreps(2, 4, ParityMode.SO3) == "4x0e + 4x1e + 4x2e"


# --------------------------------------------------------------------------- claim 2 (E7 gate)
def _reynolds(rotations: np.ndarray, irreps_str: str) -> np.ndarray:
    """Projector onto the subspace of tensors invariant under a group of proper rotations."""
    from e3nn import o3

    irreps = o3.Irreps(irreps_str)
    projector = np.zeros((irreps.dim, irreps.dim))
    for rot in rotations:
        projector += irreps.D_from_matrix(torch.tensor(rot, dtype=torch.float64)).numpy()
    return projector / len(rotations)


def _reynolds_rank(rotations: np.ndarray, irreps_str: str) -> int:
    """Number of invariant tensors. e3nn's Wigner tables carry ~1e-7 noise, so tol is 1e-5."""
    return int(np.linalg.matrix_rank(_reynolds(rotations, irreps_str), tol=1e-5))


def _rotation_group(generators: list[np.ndarray]) -> np.ndarray:
    """Close a set of generators into the full rotation group (small groups only)."""
    elements = [np.eye(3)]
    frontier = [np.eye(3)]
    while frontier:
        new = []
        for a in frontier:
            for g in generators:
                m = g @ a
                if not any(np.allclose(m, e, atol=1e-9) for e in elements):
                    elements.append(m)
                    new.append(m)
        frontier = new
    return np.stack(elements)


def _rot(axis: str, k: int) -> np.ndarray:
    """Rotation by ``k * 90`` degrees about a Cartesian axis."""
    theta = k * np.pi / 2
    c, s = np.cos(theta), np.sin(theta)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_rotation_subgroup_432_forbids_any_rank_three_tensor() -> None:
    """SO(3) equivariance alone forces a zero piezoelectric tensor for point group m-3m.

    m-3m's proper-rotation subgroup is 432 (order 24). No rank-3 tensor is invariant under it, so
    an exactly SO(3)-equivariant model must predict exactly zero -- no parity label needed. This
    is why the SO(3) arms get the m-3m crystals right (E7).
    """
    o = _rotation_group([_rot("z", 1), _rot("x", 1)])
    assert len(o) == 24, f"432 has order 24, built {len(o)}"
    assert np.allclose([np.linalg.det(r) for r in o], 1.0)  # proper rotations only
    assert np.abs(_reynolds(o, PIEZO_IRREPS)).max() < 1e-5  # projector is the zero map
    assert _reynolds_rank(o, PIEZO_IRREPS) == 0


def test_rotation_subgroup_23_permits_a_rank_three_tensor() -> None:
    """m-3's proper subgroup is 23 (order 12), a piezoelectric class: invariants exist, so SO(3)
    is not forced to zero -- which is why every m-3 crystal is false-flagged."""
    threefold = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)  # (111) 3-fold
    t = _rotation_group([_rot("z", 2), _rot("x", 2), threefold])
    assert len(t) == 12, f"23 has order 12, built {len(t)}"
    assert _reynolds_rank(t, PIEZO_IRREPS) > 0
    # ...and 23 is a subgroup of 432, so the invariant it permits is killed by the extra 4-folds.
    assert _reynolds_rank(_rotation_group([_rot("z", 1), _rot("x", 1)]), PIEZO_IRREPS) == 0


# --------------------------------------------------------------------------- claim 3 (E2 protocol)
def _spacegroup(structure, symprec: float) -> int:  # noqa: ANN001
    cell = (
        structure.cell,
        structure.positions @ np.linalg.inv(structure.cell),
        structure.atomic_numbers,
    )
    return int(spglib.get_symmetry_dataset(cell, symprec=symprec).number)


@pytest.mark.parametrize("name", ["BaTiO3", "PbTiO3"])
def test_cubic_perovskite_is_centrosymmetric_at_every_tolerance(name: str) -> None:
    for symprec in (1e-3, 1e-5, 1e-8):
        assert _spacegroup(perovskite(name), symprec) == 221


@pytest.mark.parametrize("name", ["BaTiO3", "PbTiO3"])
def test_polar_distortion_is_p4mm_at_tight_tolerance(name: str) -> None:
    """Every delta > 0 must register as P4mm (99) -- but only at symprec 1e-8."""
    for delta in (1e-3, 3e-3, 6e-3, 1e-2, 0.05, 0.2, 0.5, 1.0, 1.2):
        assert _spacegroup(tetragonal_distortion(name, delta), 1e-8) == 99


def test_spglib_symprec_is_a_distance_tolerance_not_a_symmetry_test() -> None:
    """Regression guard for the E2 protocol amendment.

    At symprec 1e-3, small-delta polar frames are *wrongly* reported centrosymmetric, because the
    maximum atomic displacement falls below the tolerance. The crossover is near delta ~ 0.006.
    This is the same phenomenon as the mp-1227949 raw-coordinate false flag (appendix A5).
    """
    assert _spacegroup(tetragonal_distortion("BaTiO3", 3e-3), 1e-3) == 221  # wrong, but expected
    assert _spacegroup(tetragonal_distortion("BaTiO3", 3e-3), 1e-8) == 99  # right
    assert _spacegroup(tetragonal_distortion("BaTiO3", 6e-3), 1e-3) == 99  # past the crossover
    assert max_displacement_angstrom("BaTiO3", 3e-3) < 1e-3
    assert max_displacement_angstrom("BaTiO3", 6e-3) > 7e-4


# --------------------------------------------------------------------------- claim 4 (E3 basis)
class _ToyTensorNet(torch.nn.Module):
    """Minimal equivariant net with a parity-odd tensor output, in matched O(3)/SO(3) arms.

    Three properties are load-bearing, and each was found the hard way:

    1. **Directed edges** (messages accumulate at the receiver only). With undirected edges the
       message is symmetric under ``i <-> j`` and every odd irrep cancels identically.
    2. **Species embeddings**, inversion-symmetric (``z[sigma(i)] == z[i]``). Without species the
       readout is ``sum_{i != j} g(x_j - x_i)``, again even, so ``T == 0`` for both arms.
    3. **A bilinear readout** ``TP(h_i, h_i)``. A linear readout off the edge spherical harmonics
       cannot produce a nonzero SO(3) output either: the parity-violating path is ``2e (x) 2e ->
       1e``, which is inversion-*even* and therefore survives the sum over +/- pairs. In the O(3)
       arm the same path is ``2e (x) 2e -> 1e``, which cannot feed a ``1o`` output.

    A toy missing any of them yields ``T == 0`` for *both* arms and would pass a naive test
    vacuously; the ``|T(random)| > 1`` guard below is what catches that.
    """

    def __init__(
        self, mode: ParityMode, n_species: int = 3, channels: int = 8, seed: int = 0
    ) -> None:
        super().__init__()
        from e3nn import nn as e3nn_nn  # noqa: F401
        from e3nn import o3

        torch.manual_seed(seed)
        self.mode = mode
        self.sh_irreps = o3.Irreps(degree_irreps(2, 1, mode))
        self.hidden = o3.Irreps(degree_irreps(2, channels, mode))
        self.out_irreps = o3.Irreps(output_irreps(PIEZO_IRREPS, mode))

        self.embed = torch.nn.Embedding(n_species, channels)
        self.message = o3.FullyConnectedTensorProduct(
            self.sh_irreps, o3.Irreps(f"{channels}x0e"), self.hidden, shared_weights=True
        )
        self.radial = torch.nn.Sequential(
            torch.nn.Linear(1, 16), torch.nn.SiLU(), torch.nn.Linear(16, channels)
        )
        # The bilinear readout: this is where 2e (x) 2e -> 1e lives.
        self.readout = o3.FullyConnectedTensorProduct(
            self.hidden, self.hidden, self.out_irreps, shared_weights=True
        )

    def forward(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        from e3nn import o3

        n = pos.shape[0]
        recv, send = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        mask = recv != send
        recv, send = recv[mask], send[mask]

        rel = pos[recv] - pos[send]  # directed: receiver minus sender
        dist = rel.norm(dim=-1, keepdim=True)
        sh = o3.spherical_harmonics(self.sh_irreps, rel, normalize=True, normalization="component")

        scalars = self.embed(z[send]) * self.radial(dist)
        msg = self.message(sh, scalars)

        h = torch.zeros(n, self.hidden.dim, dtype=pos.dtype)
        h = h.index_add(0, recv, msg)
        return self.readout(h, h).sum(dim=0)


def _centrosymmetric_cloud(
    seed: int = 0, pairs: int = 4
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Points in +/- pairs with inversion-symmetric species; sigma swaps 2k <-> 2k+1."""
    rng = np.random.default_rng(seed)
    half = rng.normal(size=(pairs, 3))
    pos = np.empty((2 * pairs, 3))
    pos[0::2], pos[1::2] = half, -half
    z_half = rng.integers(0, 3, size=pairs)
    z = np.empty(2 * pairs, dtype=np.int64)
    z[0::2], z[1::2] = z_half, z_half  # z[sigma(i)] == z[i]
    sigma = np.arange(2 * pairs) ^ 1
    return (
        torch.tensor(pos, dtype=torch.float64),
        torch.tensor(z),
        sigma,
    )


def _parity_operator(sigma: np.ndarray, n: int) -> np.ndarray:
    """(P u)_i = -u_{sigma(i)} on flattened displacements, as a 3n x 3n matrix. P @ P == I."""
    p = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for c in range(3):
            p[3 * i + c, 3 * sigma[i] + c] = -1.0
    return p


def _jacobian(model: _ToyTensorNet, pos: torch.Tensor, z: torch.Tensor) -> np.ndarray:
    pos = pos.clone().requires_grad_(True)
    out = model(pos, z)
    rows = []
    for k in range(out.shape[0]):
        (g,) = torch.autograd.grad(out[k], pos, retain_graph=True)
        rows.append(g.reshape(-1).detach().numpy())
    return np.stack(rows)


@pytest.mark.parametrize("seed", [0, 1])
def test_o3_toy_output_vanishes_at_centrosymmetric_configuration(seed: int) -> None:
    pos, z, _ = _centrosymmetric_cloud(seed)
    model = _ToyTensorNet(ParityMode.O3, seed=seed).double()
    assert float(model(pos, z).norm()) < 1e-12

    # Non-degeneracy guard: a toy that outputs zero everywhere would pass the line above.
    rng = np.random.default_rng(seed + 100)
    random_pos = torch.tensor(rng.normal(size=pos.shape), dtype=torch.float64)
    assert float(model(random_pos, z).norm()) > 1e-3


@pytest.mark.parametrize("seed", [0, 1])
def test_so3_toy_output_is_nonzero_at_centrosymmetric_configuration(seed: int) -> None:
    """The whole thesis in one line: no parity labels, no symmetry-forced zero."""
    pos, z, _ = _centrosymmetric_cloud(seed)
    model = _ToyTensorNet(ParityMode.SO3, seed=seed).double()
    assert float(model(pos, z).norm()) > 1e-3


@pytest.mark.parametrize("seed", [0, 1])
def test_o3_jacobian_is_purely_inversion_odd(seed: int) -> None:
    """``J . P = -J`` at a centrosymmetric point, so every active singular vector scores -1.

    Proof: for O(3), ``T(I.x) = -T(x)``. At centrosymmetric ``x0`` (``I.x0 == x0`` up to the
    inversion permutation sigma), differentiating gives ``J . P = -J``. Any inversion-even
    displacement (``P u == u``) then satisfies ``J u = -J u``, i.e. ``J u = 0``. So even modes lie
    in ker J and every right-singular vector with sigma > 0 obeys ``P u = -u``.
    """
    pos, z, sigma = _centrosymmetric_cloud(seed)
    model = _ToyTensorNet(ParityMode.O3, seed=seed).double()
    j = _jacobian(model, pos, z)
    p = _parity_operator(sigma, pos.shape[0])

    assert np.abs(j @ p + j).max() < 1e-9  # the theorem itself

    # full_matrices=False so vt's rows line up with the singular values (j is 18 x 3n, 3n > 18).
    _, sv, vt = np.linalg.svd(j, full_matrices=False)
    active = vt[sv > 1e-8 * sv.max()]
    assert len(active) > 0, "degenerate Jacobian: no active singular vectors"

    # Even modes lie in ker J, so the rank cannot exceed the odd subspace's dimension (3 * pairs).
    assert len(active) <= 3 * (pos.shape[0] // 2)

    scores = [float(u @ (p @ u) / (np.linalg.norm(u) * np.linalg.norm(p @ u))) for u in active[:5]]
    assert np.allclose(scores, -1.0, atol=1e-9), scores


@pytest.mark.parametrize("seed", [0, 1])
def test_even_subspace_energy_fraction_separates_the_arms(seed: int) -> None:
    """E3's primary statistic: ``||J . P_even||_F / ||J||_F``.

    Exactly 0 for O(3) by the theorem above. Basis-independent, and unlike per-vector parity
    scores it needs no singular-vector truncation. (On *trained* models the SO(3) arm turns out to
    be approximately odd too, so its top parity scores also sit near -1 -- which is why this
    fraction, not the scores, is the statistic we report.)
    """
    pos, z, sigma = _centrosymmetric_cloud(seed)
    p = _parity_operator(sigma, pos.shape[0])
    p_even = (np.eye(p.shape[0]) + p) / 2

    fractions = {}
    for mode in (ParityMode.O3, ParityMode.SO3):
        j = _jacobian(_ToyTensorNet(mode, seed=seed).double(), pos, z)
        fractions[mode] = np.linalg.norm(j @ p_even) / np.linalg.norm(j)

    assert fractions[ParityMode.O3] < 1e-9
    assert fractions[ParityMode.SO3] > 1e-2


# --------------------------------------------------------------------------- claim 5 (E4 identity)
@pytest.mark.parametrize("mode", [ParityMode.O3, ParityMode.SO3])
def test_inversion_averaging_is_trivially_zero_on_an_exactly_centrosymmetric_input(
    mode: ParityMode,
) -> None:
    """E4's idealized column carries no parity information -- for *either* arm.

    If ``x`` is exactly centrosymmetric then ``I.x`` is the same structure up to a permutation of
    atoms, so any permutation-invariant model gives ``T(I.x) == T(x)`` and the odd projection
    ``T_sym = [T(x) - T(I.x)]/2`` vanishes identically. Only the raw variant is informative.
    """
    pos, z, _ = _centrosymmetric_cloud(0)
    model = _ToyTensorNet(mode, seed=0).double()
    t_x = model(pos, z)
    t_ix = model(-pos, z)  # inversion; the +/- pairing makes this the same structure
    t_sym = (t_x - t_ix) / 2
    assert float(t_sym.norm()) < 1e-10


# --------------------------------------------------- run identity (not a physics claim, but a gate)
def test_run_label_collides_across_datasets_but_run_key_does_not() -> None:
    """Regression guard for a real contamination incident.

    ``run_label`` is ``(core, parity, target, seed)`` and omits the dataset, so the E1 augmented
    piezoelectric runs produced labels identical to the headline runs. Flattening then overwrote
    the headline ``metrics/`` files with side-study numbers and ``results/stats.json`` moved.
    ``run_key`` is the dataset-qualified identifier that must be used wherever runs from different
    datasets can be mixed.
    """
    from equiparity.domain.experiment import CANONICAL_DATASETS
    from equiparity.io.config import parse_experiment_config

    base = {
        "seed": 1,
        "core": "nequip",
        "parity": "so3",
        "target": "piezoelectric",
        "processed_npz": "a",
        "split_npz": "b",
    }
    headline = parse_experiment_config({**base, "dataset": "mp_piezoelectric"})
    side = parse_experiment_config({**base, "dataset": "mp_piezoelectric_augmented"})

    assert headline.run_label == side.run_label  # the collision that caused the incident
    assert headline.run_key != side.run_key  # the fix
    assert headline.run_key == headline.run_label  # canonical datasets keep their bare label
    assert side.run_key.endswith("__mp_piezoelectric_augmented")
    assert "mp_piezoelectric" in CANONICAL_DATASETS
    assert "mp_piezoelectric_augmented" not in CANONICAL_DATASETS
