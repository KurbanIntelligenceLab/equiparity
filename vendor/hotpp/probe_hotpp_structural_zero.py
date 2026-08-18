"""Theorem-1-style structural-zero test for HotPP (MiaoNet), random init, float64.

Builds an artificial centrosymmetric cluster (every atom at +r has an atom of the SAME
species at -r; inversion center at the origin), then computes the rank-3 (way=3) node
feature at RANDOM INIT and asks whether the SUM-POOLED (system-level) rank-3 tensor is
exactly zero, as Theorem 1 predicts for any genuinely O(3)-equivariant network at ANY
parameter values.

Per-atom features need not vanish (an atom's own environment is not itself centrosymmetric
in general -- the theorem is about the pooled/molecular tensor, which is what the manuscript's
piezoelectric target is). We therefore report BOTH the max per-atom norm (informational) and
the norm of the sum over atoms (the structural quantity subject to Theorem 1).
"""
import sys
sys.path.insert(0, ".")
import json
import numpy as np
import torch

from hotpp.utils import EnvPara
EnvPara.FLOAT_PRECISION = torch.float64
torch.set_default_dtype(torch.float64)

from hotpp.layer.cutoff import CosineCutoff
from hotpp.layer.radial import BesselPoly
from hotpp.layer.embedding import AtomicEmbedding
from hotpp.model.miao import MiaoNet

torch.manual_seed(1)
np.random.seed(1)

R_MAX = 5.0
N_LAYERS = 2
MAX_R_WAY = 3
MAX_OUT_WAY = [3, 3]
OUTPUT_DIM = [8, 8]


def build_model(seed):
    torch.manual_seed(seed)
    cutoff_fn = CosineCutoff(cutoff=R_MAX)
    radial_fn = BesselPoly(r_max=R_MAX, n_max=8, cutoff_fn=cutoff_fn)
    embedding_layer = AtomicEmbedding(atomic_number=list(range(1, 20)), n_channel=8)
    model = MiaoNet(
        embedding_layer=embedding_layer,
        radial_fn=radial_fn,
        n_layers=N_LAYERS,
        max_r_way=[MAX_R_WAY] * N_LAYERS,
        max_out_way=MAX_OUT_WAY,
        output_dim=OUTPUT_DIM,
        activate_fn="silu",
        target_way={"site_energy": 0},
        conv_mode="node_j",
        update_edge=False,
    ).double().eval()
    return model


def make_batch(coordinates, atomic_numbers, r_max):
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    atoms = Atoms(numbers=atomic_numbers, positions=coordinates, pbc=False)
    idx_i, idx_j, offset = neighbor_list("ijS", atoms, r_max, self_interaction=False)
    offset = np.array(offset, dtype=np.float64)
    batch_data = {
        "atomic_number": torch.tensor(atomic_numbers, dtype=torch.long),
        "idx_i": torch.tensor(idx_i, dtype=torch.long),
        "idx_j": torch.tensor(idx_j, dtype=torch.long),
        "coordinate": torch.tensor(coordinates, dtype=torch.float64),
        "offset": torch.tensor(offset, dtype=torch.float64),
        "n_atoms": torch.tensor([len(atoms)], dtype=torch.long),
    }
    return batch_data


def get_node_feature(model, batch_data, way):
    node_info, edge_info = model.get_init_info(batch_data)
    for block in model.en_equivalent_blocks:
        node_info, edge_info = block(node_info, edge_info, batch_data)
    return node_info[way].detach().numpy()


def make_centrosymmetric_cluster(rng, n_pairs=4):
    """n_pairs atoms at +-r_k with matching species at +-r_k -> exact inversion symmetry."""
    species = rng.integers(1, 15, size=n_pairs)
    r = rng.normal(scale=1.8, size=(n_pairs, 3))
    coords = np.concatenate([r, -r], axis=0)
    atomic_numbers = np.concatenate([species, species])
    return coords, atomic_numbers


def sum_pooled_norm(T):
    # T: [n_atoms, n_channel, 3,3,3] -> sum over atoms, then Frobenius-like norm over (channel,3,3,3)
    pooled = T.sum(axis=0)
    return float(np.linalg.norm(pooled)), float(np.abs(pooled).max())


def main():
    rng = np.random.default_rng(7)
    results = {"trials": []}
    for trial, seed in enumerate([1, 2, 3]):
        model = build_model(seed)
        coords, atomic_numbers = make_centrosymmetric_cluster(rng, n_pairs=4)
        # sanity: verify the cluster really is centrosymmetric (inversion is a self-map)
        inv_coords = -coords
        # match each inverted atom to an original atom of same species at same position
        matched = True
        for i in range(len(coords)):
            dists = np.linalg.norm(coords - inv_coords[i], axis=1)
            j = int(np.argmin(dists))
            if dists[j] > 1e-10 or atomic_numbers[j] != atomic_numbers[i]:
                matched = False
        batch = make_batch(coords, atomic_numbers, R_MAX)
        T3 = get_node_feature(model, batch, way=3)
        pooled_norm, pooled_max = sum_pooled_norm(T3)
        per_atom_max = float(np.abs(T3).max())
        results["trials"].append({
            "seed": seed,
            "inversion_verified_exact": matched,
            "per_atom_max_abs_way3": per_atom_max,
            "sum_pooled_norm_way3": pooled_norm,
            "sum_pooled_max_abs_way3": pooled_max,
            "sum_pooled_relative_to_per_atom_scale": pooled_norm / (per_atom_max + 1e-300),
        })

    results["verdict"] = {
        "all_inversion_verified": all(t["inversion_verified_exact"] for t in results["trials"]),
        "all_pooled_norm_below_1e-10": all(t["sum_pooled_norm_way3"] < 1e-10 for t in results["trials"]),
        "note": "Theorem-1-style prediction: sum-pooled rank-3 (odd-parity) output must vanish "
                "structurally on a centrosymmetric cluster, at ANY parameter values, for a genuinely "
                "O(3)-equivariant network. Per-atom values need not vanish (measured for context only)."
    }
    with open("hotpp_structural_zero_probe.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
