# Numerical sources for the manuscript

| Manuscript display | Values in the repository |
|---|---|
| Table 1 and Figure 1a–b | [Point groups and invariant dimensions](parity_gap_table.csv). Panels a and b illustrate the symmetry argument. |
| Figure 1c | [Piezoelectric norm percentiles](norm_percentiles.json), averaged across seeds within each backbone and symmetry arm. |
| Figure 2a and Table S2 | [Class-specific flag percentages and median norms](class_symmetry.json). Use `by_backbone` for the figure and `by_class` for the supplementary table. |
| Figure 2b | [Rutile displacement and tensor norm](rutile_distortion.json), including the space group at each displacement. |
| Figure 3a–b and Table S1 | [Crystal regression measurements](crystal_regression.json). The figure uses the mean and range of the percentage improvement over the strongest constant. The table reports MAE with sample SD. |
| Table 2 and Table S3 | [Zero-label supervision](zero_supervision.json), including selected and final epochs, unseen crystals and regression errors. |
| Table S4 | [External symmetry-mask measurements](symmetry_masks.json) under the stored Voigt and full Cartesian norms. Multiply flag fractions by 100 for percentages. |
| Table S5 | [Allocated model parameters](model_capacity.json). |
| Table S6 | [QM9 energy and dipole MAE](molecular_regression.json). |

## Supporting measurements and run configurations

- [predictions.zip](predictions.zip) — NumPy arrays of held-out coefficients, symmetry norms, external-model tensors and initialization norms.
- [initialization_controls.json](initialization_controls.json) — EquiformerV3, eSEN and UMA initialization norms by crystal family and rates above 1% of the noncubic median.
- [inversion_averaging.json](inversion_averaging.json) — Idealized-crystal norms and flag fractions before and after inversion antisymmetrization.
- [populations.csv](populations.csv) — Material identifiers linking the crystal classes and independent initialization cohort. Empty entries indicate absence from that cohort.
- [training_error_concentration.json](training_error_concentration.json) — Share of zero-predictor squared error in the largest piezoelectric training tensors.
- [run_manifest.json](run_manifest.json) — Training configurations, file hashes, dataset paths and numerical software versions.

Within `predictions.zip`, `heldout/` contains material IDs, predicted coefficients and targets; `symmetry/` contains material IDs, point groups and per-crystal norms in C/m². The `external/` arrays follow the idealized or raw MP evaluation array order; `initialization/` follows `initialization_index` in the material-ID crosswalk.

The [machine-readable index](index.json) lists the displays and text supported by each record.
