"""Parity audit: run the equivariance gate on each core's O(3)/SO(3) matched pair.

Executes Checkpoint-1 steps 1-2: build both arms via the raw-irreps route and classify each
with the reflection/rotation probe. Prints a result table with parameter counts per arm.

Run: ``uv run --extra nequip python scripts/parity_audit.py``
"""

from __future__ import annotations

import numpy as np

from equiparity.domain.parity import ParityMode
from equiparity.models.nequip import (
    NequIPConfig,
    build_nequip_matched,
    count_parameters,
    nequip_featurizer,
)
from equiparity.verification.equivariance import check_equivariance

# Asymmetric 4-atom toy structure (H2CO), reused across cores.
SYMBOLS = "H2CO"
POSITIONS = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.3], [0.0, 1.1, 0.0], [0.0, 0.0, 1.2]])
DTYPE = "float64"


def audit_nequip() -> list:
    config = NequIPConfig(
        r_max=4.0,
        type_names=("H", "C", "O"),
        num_layers=3,
        l_max=2,
        num_features=16,
        type_embed_num_features=16,
        radial_mlp_width=32,
        avg_num_neighbors=10.0,
        model_dtype=DTYPE,
    )
    reports = []
    for mode in (ParityMode.O3, ParityMode.SO3):
        model = build_nequip_matched(config, mode).eval()
        featurize = nequip_featurizer(
            model, SYMBOLS, config.type_names, r_max=config.r_max, dtype=DTYPE
        )
        report = check_equivariance(
            featurize,
            POSITIONS,
            label=f"NequIP {mode.label}",
            n_params=count_parameters(model),
            dtype=DTYPE,
        )
        reports.append(report)
    return reports


def main() -> None:
    reports = audit_nequip()
    header = (
        f"{'core / arm':<18} {'irreps':<28} {'rot.err':>9} {'refl.err':>9} {'params':>8}  verdict"
    )
    print(header)
    print("-" * len(header))
    for r in reports:
        print(
            f"{r.label:<18} {r.irreps:<28} {r.rotation_error:>9.1e} "
            f"{r.reflection_error:>9.1e} {r.n_params:>8}  {r.verdict}"
        )

    o3, so3 = reports[0], reports[1]
    print()
    print(
        f"param match O(3) vs SO(3): {o3.n_params} vs {so3.n_params} "
        f"(delta {abs(o3.n_params - so3.n_params)})"
    )
    ok = o3.verdict == "O3" and so3.verdict == "SO3"
    print("MATCHED-PAIR GATE:", "PASS" if ok else "FAIL — investigate")


if __name__ == "__main__":
    main()
