"""Minimal feature-level O(3) parity probe for HotPP (MiaoNet), random init, float64.

Builds a small MiaoNet configured with max_out_way=3 so a genuine rank-3 (l=3-equivalent
Cartesian) feature is produced natively. Applies a random proper rotation (det=+1) and a
random improper operation (det=-1) to a small non-centrosymmetric molecule's coordinates,
and checks whether the rank-3 node feature transforms as a genuine Cartesian tensor under
each: T'_{abc} = G_ai G_bj G_ck T_ijk, for G = R (rotation) and G = M (improper).

This is the decisive mirror-law test: an O(3)-equivariant (parity-correct) implementation
must satisfy this to machine precision for BOTH G=R and G=M. An E(n)/SO(3)-only
implementation, if it existed, would satisfy it for G=R but fail for G=M.
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

torch.manual_seed(0)
np.random.seed(0)

R_MAX = 5.0
N_LAYERS = 2
MAX_R_WAY = 3
MAX_OUT_WAY = [3, 3]   # per-layer max output tensor rank; final layer carries way=3
OUTPUT_DIM = [8, 8]

def build_model():
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
    return node_info[way].detach().numpy()  # [n_atoms, n_channel, 3,3,3] for way=3


def transform_rank3(T, G):
    # T: [n_atoms, n_channel, 3, 3, 3], G: [3,3]
    # T'_{abc} = G_ai G_bj G_ck T_ijk  (einsum over each of the 3 spatial legs)
    return np.einsum('ai,bj,ck,nlijk->nlabc', G, G, G, T)


def random_rotation(rng):
    # random proper rotation via QR of a random Gaussian matrix, fix det sign
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    assert abs(np.linalg.det(Q) - 1.0) < 1e-10
    return Q


def random_improper(rng):
    R = random_rotation(rng)
    M = -R   # improper: det(-R) = (-1)^3 det(R) = -1
    assert abs(np.linalg.det(M) + 1.0) < 1e-10
    return M


def main():
    rng = np.random.default_rng(42)
    model = build_model()

    # Small non-centrosymmetric organic-like cluster (no symmetry constraints assumed)
    atomic_numbers = np.array([6, 8, 1, 1, 7, 6])
    coords = rng.normal(scale=1.5, size=(len(atomic_numbers), 3))

    batch0 = make_batch(coords, atomic_numbers, R_MAX)
    T0 = get_node_feature(model, batch0, way=3)  # [n_atoms, n_channel, 3,3,3]

    results = {}
    for name, G in [("rotation", random_rotation(rng)), ("improper", random_improper(rng))]:
        coords_g = coords @ G.T   # apply G to each atom position: x -> G x
        batch_g = make_batch(coords_g, atomic_numbers, R_MAX)
        T_g = get_node_feature(model, batch_g, way=3)  # feature computed directly on transformed input

        T_expected = transform_rank3(T0, G)  # transform the ORIGINAL feature by the Cartesian tensor law

        num = np.abs(T_g - T_expected)
        denom = np.abs(T_expected) + 1e-12
        max_abs_err = float(num.max())
        max_rel_err = float((num / denom).max())
        norm_T0 = float(np.linalg.norm(T0))
        results[name] = {
            "det_G": float(np.linalg.det(G)),
            "max_abs_error": max_abs_err,
            "max_rel_error_eps_denom": max_rel_err,
            "norm_T_way3": norm_T0,
            "relative_to_norm": max_abs_err / (norm_T0 + 1e-12),
        }

    # Positive control: verify the probe discriminates by comparing rotation vs improper.
    # A genuine O(3)-equivariant model must pass BOTH to machine precision (float64).
    results["verdict"] = {
        "rotation_passes": results["rotation"]["relative_to_norm"] < 1e-8,
        "improper_passes": results["improper"]["relative_to_norm"] < 1e-8,
    }
    results["config"] = {
        "n_layers": N_LAYERS,
        "max_out_way": MAX_OUT_WAY,
        "max_r_way": MAX_R_WAY,
        "r_max": R_MAX,
        "dtype": "float64",
        "n_atoms": int(len(atomic_numbers)),
        "seed": 42,
        "hotpp_source": "github.com/yongwongxx/Hotpp @ main branch tarball, no e3nn dependency",
    }

    with open("hotpp_parity_probe.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
