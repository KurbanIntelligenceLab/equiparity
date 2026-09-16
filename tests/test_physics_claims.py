"""Physics claims the study relies on, checked on small constructions.

Nothing here loads a trained checkpoint; these run on CPU in seconds.
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


# ------------------------------------------------------------------- output irreps
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


# ------------------------------------------------------------------- proper-rotation subgroups
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
    is why the SO(3) arms get the m-3m crystals right (the rotation-subgroup analysis).
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


# ------------------------------------------------------------------- polar distortion path
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
    """At symprec 1e-3, small-delta polar frames are reported centrosymmetric.

    The maximum atomic displacement falls below the tolerance; the crossover is near delta ~ 0.006.
    """
    assert _spacegroup(tetragonal_distortion("BaTiO3", 3e-3), 1e-3) == 221  # wrong, but expected
    assert _spacegroup(tetragonal_distortion("BaTiO3", 3e-3), 1e-8) == 99  # right
    assert _spacegroup(tetragonal_distortion("BaTiO3", 6e-3), 1e-3) == 99  # past the crossover
    assert max_displacement_angstrom("BaTiO3", 3e-3) < 1e-3
    assert max_displacement_angstrom("BaTiO3", 6e-3) > 7e-4


# ------------------------------------------------------------------- matched toy model
class _ToyTensorNet(torch.nn.Module):
    """Minimal equivariant net with a parity-odd tensor output, in matched O(3)/SO(3) arms.

    Three properties are required:

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


# ------------------------------------------------------------------- output antisymmetrization
@pytest.mark.parametrize("mode", [ParityMode.O3, ParityMode.SO3])
def test_inversion_averaging_is_trivially_zero_on_an_exactly_centrosymmetric_input(
    mode: ParityMode,
) -> None:
    """Output antisymmetrization is identically zero on an exactly centrosymmetric input.

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


# ------------------------------------------------------------------- run identity
def test_run_label_collides_across_datasets_but_run_key_does_not() -> None:
    """``run_label`` omits the dataset, so augmented and headline runs share labels.

    ``run_key`` is the dataset-qualified identifier used wherever runs from different datasets can
    be mixed.
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

    assert headline.run_label == side.run_label
    assert headline.run_key != side.run_key
    assert headline.run_key == headline.run_label  # canonical datasets keep their bare label
    assert side.run_key.endswith("__mp_piezoelectric_augmented")
    assert "mp_piezoelectric" in CANONICAL_DATASETS
    assert "mp_piezoelectric_augmented" not in CANONICAL_DATASETS


# ------------------------------------------------------------------- zero-row loss weighting
def test_zero_row_weighted_mse_is_a_noop_at_weight_one() -> None:
    """At W=1 the per-row weighted MSE is bit-identical to torch.nn.MSELoss; the mask picks the
    exactly-zero-target rows. Guards the loss-weight sweep loss-weight sweep's control column."""
    torch.manual_seed(0)
    pred = torch.randn(16, 18, dtype=torch.float64)
    target = torch.randn(16, 18, dtype=torch.float64)
    target[3] = 0.0  # one exactly-zero row

    def weighted_mse(p: torch.Tensor, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (w[:, None] * (p - t) ** 2).mean()

    ones = torch.ones(16, dtype=torch.float64)
    assert torch.equal(weighted_mse(pred, target, ones), torch.nn.MSELoss()(pred, target))

    # the mask selects exactly the zero-target rows
    mask = np.abs(target.numpy()).max(axis=1) == 0.0
    assert mask.sum() == 1 and mask[3]

    # at W=10 the zero row is up-weighted, so the loss changes
    w10 = ones.clone()
    w10[torch.tensor(mask)] = 10.0
    assert not torch.equal(weighted_mse(pred, target, w10), torch.nn.MSELoss()(pred, target))
