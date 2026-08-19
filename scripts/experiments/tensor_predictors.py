"""Evaluate released GMTNet / CEITNet piezoelectric checkpoints on our OOD structures.

Runs two external crystal-tensor models -- GMTNet (third_party/AIRS/OpenMat/GMTNet/GMTNet_piezo)
and CEITNet (third_party/ceitnet/piezo) -- with their released piezo checkpoints on our 2000
centrosymmetric MP structures (idealized and raw variants), with their symmetry mask ON and OFF.

Must be run with the tier1 python:
    <tier1-python> scripts/experiments/tensor_predictors.py --model gmtnet --toy
    <tier1-python> scripts/experiments/tensor_predictors.py --model gmtnet
    <tier1-python> scripts/experiments/tensor_predictors.py --model ceitnet --toy
    <tier1-python> scripts/experiments/tensor_predictors.py --model ceitnet

Faithfulness notes (their model, their pipeline, our structures):
- Graphs are built with their graphs.py:atoms2graphs exactly as in their eval scripts
  (GMTNet train.py: cutoff=4.0, max_neighbors=16; CEITNet test.py: cutoff=4.0, max_neighbors=64;
  use_canonize=True, reduce=False), from jarvis Atoms obtained via JarvisAtomsAdaptor.
- Symmetry artifacts (feature_mask / feature_mask_ori / ideal_matrix / matrix_equal) replicate
  data.py:get_dataset lines exactly (spglib symprec=1e-5, rm_duplicates, cartesian rotations,
  Wigner-D over '2x0e+2x0o+2x1e+2x1o+2x2e+2x2o+2x3e+2x3o'), importing their own helper
  functions from their data.py. We cannot call their get_dataset directly because it reads
  hardcoded /yourpath pickles; nothing inside third_party is modified.
- GMTNet mask ON  = model.mask=True with per-structure feature_mask (bmm inside forward) plus the
  test-time add_feat_mask = (|feature_mask_ori| > 1).float(), exactly as their test.py comformer
  branch. Mask OFF = model.mask=False (feature-mask path skipped entirely). equality_adjustment
  runs inside forward unconditionally in both states, as released.
- CEITNet mask ON  = their --zero_mask post-hoc forced-zero mask (functions copied verbatim from
  third_party/ceitnet/piezo/test.py, since importing their test.py pulls pandarallel/module-level
  side effects). Mask OFF = raw forward output. equality_adjustment runs inside forward in both
  states (their released eval always passes `equality`).
"""

# ruff: noqa: N803, N806, SIM300  (copied third-party code kept verbatim)
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
MODEL_DIRS = {
    "gmtnet": REPO / "third_party/AIRS/OpenMat/GMTNet/GMTNet_piezo",
    "ceitnet": REPO / "third_party/ceitnet/piezo",
}
CKPTS = {
    "gmtnet": MODEL_DIRS["gmtnet"] / "piezo_model.pt",
    "ceitnet": REPO / "third_party/ceitnet/repro_data/pretrained_ckpts/piezo.pt",
}
# Graph hyperparameters as used in each repo's own eval path.
GRAPH_PARAMS = {
    "gmtnet": {"cutoff": 4.0, "max_neighbors": 16},  # GMTNet train.py structure_to_graphs
    "ceitnet": {"cutoff": 4.0, "max_neighbors": 64},  # ceitnet test.py structure_to_graphs
}
NPZ = {
    "idealized": REPO / "data/raw/mp/mp_ood_centrosymmetric_processed.npz",
    "raw": REPO / "data/raw/mp/mp_ood_centrosymmetric_processed_raw.npz",
}
OUT_DIR = REPO / "results/tensor_predictors"
SUMMARY_PATH = REPO / "results/tensor_predictors.json"
SYMPREC = 1e-5  # data.py get_dataset default
# data.py: indices of the 18 odd components (2x1o + 1x2o + 1x3o) in the 64-dim feature vector
IDX18 = [10, 11, 12, 13, 14, 15, 26, 27, 28, 29, 30, 50, 51, 52, 53, 54, 55, 56]
THRESHOLDS = np.logspace(-4, 0, 25)

G = SimpleNamespace()  # holds third_party modules after sys.path setup


def setup_thirdparty(model_name: str) -> None:
    sys.path.insert(0, str(MODEL_DIRS[model_name]))
    import graphs as refgraphs  # their graphs.py
    from pymatgen.io.jarvis import JarvisAtomsAdaptor

    import data as refdata  # their data.py

    G.refdata = refdata
    G.refgraphs = refgraphs
    G.adaptor = JarvisAtomsAdaptor()
    G.model_name = model_name
    G.graph_params = GRAPH_PARAMS[model_name]


# ---------------------------------------------------------------------------
# CEITNet forced-zero mask -- copied verbatim from third_party/ceitnet/piezo/test.py
# (importing their test.py executes module-level pandarallel/cuda side effects).
# ---------------------------------------------------------------------------
def _transform_rank3(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    return np.einsum("ia,jb,kc,abc->ijk", R, R, R, T)


def infer_forced_zero_mask_symmetric_rank3_jk(
    cart_rots: np.ndarray,
    rcond: float = 1e-10,
    zero_tol: float = 1e-8,
) -> np.ndarray:
    if cart_rots is None or len(cart_rots) == 0:
        return np.zeros((3, 3, 3), dtype=bool)

    basis = []
    for i in range(3):
        for j in range(3):
            for k in range(j, 3):
                B = np.zeros((3, 3, 3), dtype=np.float64)
                B[i, j, k] = 1.0
                if k != j:
                    B[i, k, j] = 1.0
                basis.append(B)

    n = float(len(cart_rots))
    cols = []
    for B in basis:
        avg = np.zeros((3, 3, 3), dtype=np.float64)
        for R in cart_rots:
            avg += _transform_rank3(R, B)
        avg /= n
        cols.append(avg.reshape(-1))

    M = np.stack(cols, axis=1)  # (27, 18)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    if S.size == 0:
        return np.zeros((3, 3, 3), dtype=bool)

    smax = S[0]
    keep = S > (rcond * smax)
    if not np.any(keep):
        return np.zeros((3, 3, 3), dtype=bool)

    inv_basis = U[:, keep]  # (27, k)
    forced_zero = np.max(np.abs(inv_basis), axis=1) < zero_tol
    return forced_zero.reshape(3, 3, 3)


def forced_zero_mask_rank3_to_voigt36(forced_zero_mask_333: np.ndarray) -> np.ndarray:
    """Map a (3,3,3) forced-zero mask to Voigt (3,6) using ceitnet's to_voigt convention."""
    mapping = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]  # xx,yy,zz,xy,yz,zx
    out = np.zeros((3, 6), dtype=bool)
    for i in range(3):
        for J, (j, k) in enumerate(mapping):
            out[i, J] = bool(forced_zero_mask_333[i, j, k])
    return out


# ---------------------------------------------------------------------------
# Per-structure record: graph + symmetry artifacts, replicating data.py:get_dataset
# ---------------------------------------------------------------------------
def build_record(task):
    """task = (cell (3,3), z (n,), cart_pos (n,3)). Runs in a worker process (CPU only)."""
    from pymatgen.core.structure import Structure

    cell, z, pos = task
    structure = Structure(lattice=cell, species=z, coords=pos, coords_are_cartesian=True)
    refdata = G.refdata

    sym = refdata.get_symmetry_dataset(structure, SYMPREC)
    rots_frac = np.array(sym.rotations if hasattr(sym, "rotations") else sym["rotations"])
    equivalent_atoms = np.array(
        sym.equivalent_atoms if hasattr(sym, "equivalent_atoms") else sym["equivalent_atoms"]
    )
    spg_number = int(sym.number if hasattr(sym, "number") else sym["number"])

    # data.py get_dataset: Wigner-D feature mask construction (verbatim math)
    maskvec = torch.arange(64) + 10.0
    maskvec[16:] *= 100
    rots = refdata.rm_duplicates(rots_frac)
    Lat = structure.lattice.matrix.T
    L_inv = np.linalg.inv(Lat)
    tmp_rot = np.matmul(Lat, np.matmul(rots, L_inv))
    group_ok = bool(refdata.is_group(tmp_rot))
    D_tmp = refdata.irreps_output.D_from_matrix(torch.Tensor(tmp_rot))
    D_x = D_tmp.sum(dim=0)
    feature_mask_ori = torch.matmul(D_x, maskvec)
    mask_total = feature_mask_ori[IDX18]
    ideal_matrix = refdata.contract_tensor(refdata.converter.to_cartesian(mask_total))
    D_x = D_x / D_tmp.shape[0]
    zero_mask = (D_x > 1e-5).float()
    D_x = D_x * zero_mask
    matrix_equal = refdata.find_almost_equal_entries(torch.tensor(ideal_matrix))

    # CEITNet --zero_mask artifacts (cart_rots identical to their
    # get_cartesian_rotations_from_sym_dataset output)
    forced_zero_36 = forced_zero_mask_rank3_to_voigt36(
        infer_forced_zero_mask_symmetric_rank3_jk(tmp_rot, rcond=1e-10, zero_tol=1e-8)
    )

    graph = G.refgraphs.atoms2graphs(
        G.adaptor.get_atoms(structure),
        cutoff=G.graph_params["cutoff"],
        max_neighbors=G.graph_params["max_neighbors"],
        reduce=False,
        equivalent_atoms=equivalent_atoms,
        use_canonize=True,
    )

    return {
        "graph": graph,
        "feature_mask": D_x,  # (64,64) float
        "feature_mask_ori": feature_mask_ori,  # (64,) float
        "matrix_equal": matrix_equal,  # (2,18,18) bool
        "forced_zero_36": forced_zero_36,  # (3,6) bool
        "group": spg_number,
        "is_group_ok": group_ok,
    }


def _pool_init(model_name):
    torch.set_num_threads(1)
    setup_thirdparty(model_name)


def build_records(tasks, model_name, nproc):
    from tqdm import tqdm

    if nproc <= 1:
        return [build_record(t) for t in tqdm(tasks, desc="build records")]
    with Pool(nproc, initializer=_pool_init, initargs=(model_name,)) as pool:
        it = pool.imap(build_record, tasks, chunksize=8)
        return list(tqdm(it, total=len(tasks), desc="build records"))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_npz(path):
    d = np.load(path, allow_pickle=True)
    ids, n_atoms = d["ids"], d["n_atoms"]
    z, pos, cells = d["z"], d["positions"], d["cells"]
    offsets = np.concatenate([[0], np.cumsum(n_atoms)])
    tasks = [
        (cells[i], z[offsets[i] : offsets[i + 1]].tolist(), pos[offsets[i] : offsets[i + 1]])
        for i in range(len(ids))
    ]
    return [str(s) for s in ids], tasks


def toy_tasks():
    """Exactly centrosymmetric NaCl rocksalt conventional cell and rutile TiO2 (P4_2/mnm)."""
    a = 5.64
    nacl_cell = np.eye(3) * a
    nacl_frac = np.array(
        [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],  # Na
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5],
            [0.5, 0.5, 0.5],  # Cl
        ],
        dtype=float,
    )
    nacl = (nacl_cell, [11] * 4 + [17] * 4, nacl_frac @ nacl_cell)

    a, c, u = 4.5937, 2.9587, 0.3053
    tio2_cell = np.diag([a, a, c]).astype(float)
    tio2_frac = np.array(
        [
            [0, 0, 0],
            [0.5, 0.5, 0.5],  # Ti
            [u, u, 0],
            [1 - u, 1 - u, 0],
            [0.5 + u, 0.5 - u, 0.5],
            [0.5 - u, 0.5 + u, 0.5],  # O
        ],
        dtype=float,
    )
    tio2 = (tio2_cell, [22] * 2 + [8] * 4, tio2_frac @ tio2_cell)
    return ["NaCl_rocksalt", "TiO2_rutile"], [nacl, tio2]


# ---------------------------------------------------------------------------
# Model loading and batched inference
# ---------------------------------------------------------------------------
def load_model(model_name, device):
    if model_name == "gmtnet":
        from gmtnet import ComformerEquivariant

        # args defaults from GMTNet train.py main()
        model = ComformerEquivariant(argparse.Namespace(use_mask=True, reduce_cell=False))
    else:
        from ceitnet import CEITNet

        model = CEITNet(None)
    state_dict = torch.load(CKPTS[model_name], map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_all(model_name, model, records, device, batch_size):
    """Return (pred_maskoff, pred_maskon), each (N, 3, 6) numpy, in C/m^2."""
    from torch_geometric.data.batch import Batch
    from tqdm import tqdm

    n = len(records)
    out_off, out_on = [], []
    for lo in tqdm(range(0, n, batch_size), desc=f"predict {model_name}"):
        batch = records[lo : lo + batch_size]
        g = Batch.from_data_list([r["graph"] for r in batch]).to(device)
        feat_mask = torch.stack([r["feature_mask"] for r in batch]).float().to(device)
        equality = torch.stack([r["matrix_equal"] for r in batch]).to(device)
        add_feat_mask = (
            (torch.stack([r["feature_mask_ori"] for r in batch]).abs() > 1.0).float().to(device)
        )

        if model_name == "gmtnet":
            # no torch.no_grad(): Piezo_block differentiates through outer_S internally
            model.mask = False
            off = model(g, feat_mask, equality).detach().cpu()
            model.mask = True
            on = model(g, feat_mask, equality, add_feat_mask).detach().cpu()
        else:
            with torch.no_grad():
                off = model(g, feat_mask, equality, add_feat_mask).detach().cpu()
            on = off.clone()
            for i, r in enumerate(batch):
                on[i][torch.tensor(r["forced_zero_36"], dtype=torch.bool)] = 0.0

        out_off.append(off.view(-1, 3, 6))
        out_on.append(on.view(-1, 3, 6))
    return torch.cat(out_off).numpy(), torch.cat(out_on).numpy()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(norms):
    return {
        "n": int(norms.shape[0]),
        "false_flag_at_0.01": float(np.mean(norms > 0.01)),
        "median": float(np.median(norms)),
        "max": float(np.max(norms)),
        "threshold_curve": {
            "thresholds": [float(t) for t in THRESHOLDS],
            "false_flag_rate": [float(np.mean(norms > t)) for t in THRESHOLDS],
        },
    }


def merge_summary(key, entry):
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    summary[key] = entry
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


def run_toy(model_name, model, device):
    names, tasks = toy_tasks()
    records = [build_record(t) for t in tasks]
    off, on = predict_all(model_name, model, records, device, batch_size=2)
    np.set_printoptions(precision=6, suppress=False, linewidth=140)
    print(f"\n===== TOY GATE: {model_name} =====")
    for i, name in enumerate(names):
        rec = records[i]
        print(f"\n--- {name} (spacegroup {rec['group']}, is_group_ok={rec['is_group_ok']}) ---")
        print("mask OFF prediction (3x6, C/m^2):")
        print(off[i])
        print(f"mask OFF Frobenius norm: {np.linalg.norm(off[i]):.6e} C/m^2")
        print("mask ON prediction (3x6, C/m^2):")
        print(on[i])
        print(f"mask ON  Frobenius norm: {np.linalg.norm(on[i]):.6e} C/m^2")


def run_full(model_name, device, batch_size, nproc, variants):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # build all records (worker pool forks) before initializing CUDA in this process
    prepared = {}
    for variant in variants:
        ids, tasks = load_npz(NPZ[variant])
        print(f"\n===== BUILD: {model_name} / {variant} ({len(ids)} structures) =====")
        prepared[variant] = (ids, build_records(tasks, model_name, nproc))
    model = load_model(model_name, device)
    for variant in variants:
        ids, records = prepared[variant]
        print(f"\n===== FULL RUN: {model_name} / {variant} ({len(ids)} structures) =====")
        n_bad_group = sum(1 for r in records if not r["is_group_ok"])
        if n_bad_group:
            print(f"WARNING: is_group check failed for {n_bad_group} structures")
        off, on = predict_all(model_name, model, records, device, batch_size)
        for state, preds in (("off", off), ("on", on)):
            norms = np.linalg.norm(preds.reshape(len(ids), -1), axis=1)
            key = f"{model_name}_{variant}_mask{state}"
            np.save(OUT_DIR / f"{key}.npy", norms)
            np.save(OUT_DIR / f"{key}_tensors.npy", preds)
            entry = summarize(norms)
            entry["is_group_failures"] = n_bad_group
            merge_summary(key, entry)
            print(
                f"{key}: false_flag@0.01={entry['false_flag_at_0.01']:.4f} "
                f"median={entry['median']:.4e} max={entry['max']:.4e}"
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["gmtnet", "ceitnet"], required=True)
    ap.add_argument("--toy", action="store_true", help="run the toy gate (NaCl + rutile TiO2)")
    ap.add_argument("--variant", choices=["idealized", "raw", "both"], default="both")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--nproc", type=int, default=16, help="workers for graph/symmetry building")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float32)  # their scripts' torch config
    torch.manual_seed(42)
    np.random.seed(42)

    setup_thirdparty(args.model)
    batch_size = args.batch_size or (64 if args.model == "gmtnet" else 16)
    if args.toy:
        model = load_model(args.model, args.device)
        run_toy(args.model, model, args.device)
        return
    variants = ["idealized", "raw"] if args.variant == "both" else [args.variant]
    run_full(args.model, args.device, batch_size, args.nproc, variants)


if __name__ == "__main__":
    main()
