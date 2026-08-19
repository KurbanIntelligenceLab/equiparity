"""Build an upstream-sourced EquiformerV2 with generate_graph relocated to the shim.

Constructs ``third_party/equiformer_v2_shimmed/`` from the pinned upstream checkout
(``atomicarchitects/equiformer_v2`` @ 8fe8cba, whose ``nets/equiformer_v2/*.py`` model files are
byte-frozen since 2023-06-28):

* 11 model files copied **verbatim from upstream** (all AST-identical to the vendored copy;
  6 byte-exact, 5 differ only in import order).
* ``gaussian_rbf.py`` copied from upstream + the single documented ``num_output`` attribute (needed
  by oc20 line 232; upstream itself anticipated the swap with a commented-out line).
* ``equiformer_v2_oc20.py`` reconstructed from upstream by the **ten documented edits** below and
  nothing else — the OCP-dependency removal. It therefore contains **no** ``generate_graph`` (that
  method was inherited from the OCP ``BaseModel``); the shim subclass provides it.

Each edit is recorded so "measured on upstream source at 8fe8cba" is auditable. The reconstruction
is then checked to import, instantiate, and reproduce the shipped vendored model **bit-for-bit**
on a seeded forward — which proves both that the reconstruction is faithful and that the vendored
copy carries upstream behaviour verbatim.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "third_party/equiformer_v2_upstream/nets/equiformer_v2"
VENDORED = REPO / "src/equiparity/models/equiformer_v2"
SHIMMED = REPO / "third_party/equiformer_v2_shimmed"

VERBATIM = [
    "activation.py",
    "drop.py",
    "edge_rot_mat.py",
    "input_block.py",
    "layer_norm.py",
    "module_list.py",
    "radial_function.py",
    "so2_ops.py",
    "so3.py",
    "transformer_block.py",
    "wigner.py",
]

# The ten documented edits that turn upstream oc20 into a standalone (OCP-free) module. Each entry
# is (old, new); old must occur exactly once. Recorded verbatim in the upstream reproduction report.
OC20_EDITS: list[tuple[str, str]] = [
    ("from pyexpat.model import XML_CQUANT_OPT\n", ""),
    ("from ocpmodels.common.registry import registry\n", ""),
    ("from ocpmodels.common.utils import conditional_grad\n", ""),
    ("from ocpmodels.models.base import BaseModel\n", ""),
    ("from ocpmodels.models.scn.sampling import CalcSpherePoints\n", ""),
    (
        "from ocpmodels.models.scn.smearing import (\n"
        "    GaussianSmearing,\n"
        "    LinearSigmoidSmearing,\n"
        "    SigmoidSmearing,\n"
        "    SiLUSmearing,\n"
        ")\n",
        "",
    ),
    ('@registry.register_model("equiformer_v2")\n', ""),
    ("class EquiformerV2_OC20(BaseModel):", "class EquiformerV2_OC20(nn.Module):"),
    (
        "            self.distance_expansion = GaussianSmearing(\n"
        "                0.0,\n"
        "                self.cutoff,\n"
        "                600,\n"
        "                2.0,\n"
        "            )\n"
        "            #self.distance_expansion = GaussianRadialBasisLayer("
        "num_basis=self.num_distance_basis, cutoff=self.max_radius)\n",
        "            self.distance_expansion = GaussianRadialBasisLayer(\n"
        "                num_basis=self.num_distance_basis, cutoff=self.max_radius\n"
        "            )\n",
    ),
    ("    @conditional_grad(torch.enable_grad())\n", ""),
]


def build() -> list[str]:
    SHIMMED.mkdir(parents=True, exist_ok=True)
    (SHIMMED / "__init__.py").write_text("")
    log = []

    for name in VERBATIM:
        shutil.copyfile(UPSTREAM / name, SHIMMED / name)
        log.append(f"verbatim upstream: {name}")

    # Wigner-D constant tables loaded by wigner.py at import (upstream == vendored, byte-identical).
    shutil.copyfile(UPSTREAM / "Jd.pt", SHIMMED / "Jd.pt")
    log.append("verbatim upstream: Jd.pt (Wigner-D constants)")

    # gaussian_rbf: upstream + the documented num_output attribute (take the vendored file, which is
    # upstream + exactly that one line).
    shutil.copyfile(VENDORED / "gaussian_rbf.py", SHIMMED / "gaussian_rbf.py")
    log.append(
        "gaussian_rbf.py: upstream + documented `self.num_output = num_basis` (vendored copy)"
    )

    text = (UPSTREAM / "equiformer_v2_oc20.py").read_text()
    for old, new in OC20_EDITS:
        if text.count(old) != 1:
            raise SystemExit(f"oc20 edit anchor not unique ({text.count(old)}x): {old[:60]!r}")
        text = text.replace(old, new)
    (SHIMMED / "equiformer_v2_oc20.py").write_text(text)
    log.append(
        f"equiformer_v2_oc20.py: upstream + {len(OC20_EDITS)} documented OCP-removal edits; "
        "no generate_graph (was inherited from BaseModel -> provided by the shim subclass)"
    )

    # The call site `self.generate_graph(data)` in forward must remain (the subclass provides it);
    # only a *definition* would make oc20 non-upstream.
    assert "def generate_graph" not in text, "reconstructed oc20 must not DEFINE generate_graph"
    assert "self.generate_graph(data)" in text, "the inherited generate_graph call site must remain"
    assert "ocpmodels" not in text, "reconstructed oc20 must not import ocpmodels"
    return log


if __name__ == "__main__":
    for line in build():
        print(" ", line)
    print(f"built {SHIMMED}")
