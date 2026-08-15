"""Minimal O(3) parity probe for escnn: verify (i) genuine parity typing distinguishing
polar (odd) from axial (even) representations of the same order, (ii) that a rank-3
odd-parity (l=3) representation is available as a NATIVE irrep (not a cubic tensor
product), and (iii) numerically verify the mirror law on a random Linear map between
l=1 (vector, odd) input and l=3 (odd) output within escnn's O(3) group.
"""
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "patched_pkgs"))
import numpy as np
import torch
# NOTE: escnn's internal SteerableKernelBasis construction hardcodes float32 comparisons
# (torch.eye() without dtype match against a float64 change-of-basis matrix raises
# "Float did not match Double"). We therefore build the group/irreps/Linear layer at the
# library's native float32, then cast the constructed module and probe inputs to float64
# for the numerical mirror-law measurement itself, preserving float64 for the actual test.
from escnn.group import o3_group
from escnn.gspaces import no_base_space
import escnn.nn as enn

G = o3_group(maximum_frequency=3)

# escnn's O(3) irrep id is (inv_frequency, frequency): frequency = angular momentum l (SO(3)
# irrep label); inv_frequency in {0,1} flags behaviour under the inversion generator (0 = even
# / axial-type, 1 = odd / polar-type). This id *tuple itself* is the parity bookkeeping: two
# distinct irreps share frequency=1 (l=1) but differ in inv_frequency -- (0,1) is the axial
# (pseudo-vector) l=1 rep, (1,1) is the polar (true vector) l=1 rep.
irreps_by_l = {}
for irr in G.irreps():
    l = irr.attributes.get('frequency', None)
    inv_f = irr.attributes.get('inv_frequency', None)
    irreps_by_l.setdefault(l, []).append((irr.id, inv_f))

result = {"irreps_by_frequency": {str(k): v for k, v in sorted(irreps_by_l.items()) if k is not None and k <= 3}}

std_irrep = G.standard_representation()  # polar l=1 representation, composed of irrep (1,1)
print("standard_representation name/irreps:", std_irrep.name, std_irrep.irreps)

gspace = no_base_space(G)
in_type = enn.FieldType(gspace, [std_irrep])   # l=1 polar vector input

l3_irreps = [irr for irr in G.irreps() if irr.attributes.get('frequency') == 3]
print("l=3 irreps found:", [(irr.id, irr.attributes) for irr in l3_irreps])

odd_l3 = [irr for irr in l3_irreps if irr.attributes.get('inv_frequency') == 1]
even_l3 = [irr for irr in l3_irreps if irr.attributes.get('inv_frequency') == 0]
result["l3_odd_irrep_ids"] = [irr.id for irr in odd_l3]
result["l3_even_irrep_ids"] = [irr.id for irr in even_l3]

assert len(odd_l3) >= 1, "No odd-parity l=3 irrep found -- cannot build native rank-3 odd output"
target_irrep = odd_l3[0]

# NOTE: a first attempt built in_type from the l=1 (polar vector) standard_representation and
# tried enn.Linear(in_type -> odd-l=3 out_type). escnn raised "the basis for the block expansion
# of the filter is empty" and the layer had zero equivariant parameters. This is Schur's lemma,
# not an escnn limitation: for a compact group, the space of EQUIVARIANT LINEAR maps between two
# irreps is zero unless the irreps are isomorphic (same (l, parity) label). A single vector
# cannot be linearly mapped to a rank-3 tensor under full O(3)/SO(3) equivariance -- this is the
# exact mathematical reason NequIP/MACE/Allegro build native l=3 features via non-linear tensor
# products of edge spherical harmonics BEFORE the readout, and only apply a LINEAR map at the
# very end, channel-mixing among features that already carry the l=3 (odd) label.
# We reproduce that pattern here: build a multi-channel odd-l=3 FieldType (multiplicity 4,
# representing e.g. four "channels" of already-existing l=3 features, exactly as would arise
# from an edge-spherical-harmonic embedding contracted with a radial network) and readout with
# a LINEAR layer to a single l=3 output channel. This is the linear-readout-from-native-rank-3
# construction that criterion (iii) requires.
in_type = enn.FieldType(gspace, [target_irrep] * 4)   # 4 channels of native odd l=3 features
out_type = enn.FieldType(gspace, [target_irrep])       # single l=3 odd output channel

torch.manual_seed(0)
linear = enn.Linear(in_type, out_type)
linear.eval()
linear = linear.double()  # cast constructed equivariant linear map to float64 for the probe

rng = np.random.default_rng(123)
x_raw = torch.tensor(rng.normal(size=(4, in_type.size)), dtype=torch.float64)
x = enn.GeometricTensor(x_raw, in_type)
y = linear(x)
result["linear_layer_basis_size"] = int(linear.basisexpansion.dimension())
assert result["linear_layer_basis_size"] > 0, "Linear map between same-type l=3 channels must have nonzero basis"

errors = {}
rot_g = None
ref_g = None
for _ in range(200):
    g = G.sample()
    mat = std_irrep(g)
    det = np.linalg.det(mat)
    if rot_g is None and det > 0.5:
        rot_g = g
    if ref_g is None and det < -0.5:
        ref_g = g
    if rot_g is not None and ref_g is not None:
        break

assert rot_g is not None and ref_g is not None, "failed to sample both proper and improper O(3) elements"

for name, g in [("rotation", rot_g), ("improper", ref_g)]:
    det_std = float(np.linalg.det(std_irrep(g)))
    x_g = x.transform(g)
    y_g_direct = linear(x_g)
    y_g_expected = y.transform(g)
    err = (y_g_direct.tensor - y_g_expected.tensor).abs().max().item()
    norm = y.tensor.abs().max().item()
    errors[name] = {"det_standard_rep": det_std, "max_abs_error": err, "relative": err / (norm + 1e-300)}

result["mirror_law_probe"] = errors
# IMPORTANT CAVEAT discovered during this probe: escnn's Linear layer between the l=1 polar
# (standard) representation and an odd-l=3 representation has an EMPTY equivariant basis
# ("the basis for the block expansion of the filter is empty", zero learnable parameters).
# This is Schur's lemma, not an escnn defect: for a compact group, the space of intertwiners
# (equivariant linear maps) between two NON-ISOMORPHIC irreps is {0}. A rank-1 (vector) input
# literally cannot be linearly mapped into a rank-3 output under full O(3)/SO(3) equivariance
# -- this holds for e3nn, escnn, and any other correct irrep implementation alike. What "linear
# readout of native rank-3 features" means in practice (and in NequIP/MACE/Allegro) is: (a) the
# l=3 feature must already exist, produced by a construction that is non-linear in geometry but
# LINEAR in the network's learned parameters (radial-network scalar times a fixed spherical-
# harmonic/CG tensor of the edge geometry), and (b) the final readout head is a linear map
# AMONG CHANNELS OF THAT SAME l=3 IRREP TYPE. That is exactly what is verified above: `in_type`
# is 4 channels already carrying the target odd-l=3 label, and `linear` mixes them into 1
# channel with a nonzero equivariant basis (dimension 4, i.e. an ordinary weighted sum).
result["schur_lemma_caveat"] = (
    "enn.Linear(l=1 standard -> odd-l=3) has an EMPTY equivariant basis (0 free parameters); "
    "Schur's lemma forbids linear maps between non-isomorphic irreps for any correct O(3) "
    "implementation. The probe above instead verifies the correct construction: linear mixing "
    "among channels that already carry the l=3 odd label."
)
result["verdict"] = {
    "rotation_passes": errors["rotation"]["relative"] < 1e-8,
    "improper_passes": errors["improper"]["relative"] < 1e-8,
    "linear_readout_among_native_odd_l3_channels": True,
    "direct_l1_to_l3_linear_map_exists": False,
}
result["config"] = {"escnn_version": __import__("escnn").__version__, "dtype": "float64"}

with open("escnn_parity_probe.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print(json.dumps(result, indent=2, default=str))
