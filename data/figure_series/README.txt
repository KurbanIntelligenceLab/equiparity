Figure series
=============

The machine-readable series behind the published figures. Each file is exported from a record in
results/ by scripts/analysis/export_figure_series.py, so every plotted point can be traced to a
measurement without retraining anything.

  fig2b_thresholds.csv            Figure 2b
      False-flag fraction at 25 log-spaced thresholds, idealized coordinate variant.
      Columns: tau, core, arm, false_flag_fraction, sd.            175 rows (25 x 7 arms)

  fig4b_jacobian_points.csv       Figure 4b
      Even-subspace fraction of the input-output Jacobian, per structure and seed.
      Columns: core, arm, structure, seed, value.        360 rows (20 crystals x 3 seeds x 6 arms)

  figS1_epoch_curves.csv          Supplementary Figure 1
      False-flag fraction against training epoch.
      Columns: epoch, core, seed, false_flag.                                        1800 rows

  figS2_rutile_sweep.csv          Supplementary Figure 2
      Predicted tensor norm against polar-distortion amplitude for rutile TiO2.
      Columns: delta, arm, core, magnitude.                 231 rows (33 amplitudes x 7 arms)

  raw_coordinate_thresholds.csv   Supplementary table, raw coordinate variant
      False-flag fraction at the same 25 thresholds on raw DFT-relaxed coordinates rather than
      idealized ones. Columns: tau, core, arm, false_flag_fraction, sd.   175 rows (25 x 7 arms)

Regenerate all five with:

    uv run python scripts/analysis/export_figure_series.py
