THREE SERIES ARE MISSING, AND ONLY YOU HAVE THEM
================================================
build_figures.py rebuilds every figure from the numbers in the Supplementary Tables. Three
series are not tabulated anywhere in the manuscript, so the script cannot draw them and will
NOT guess. Export them from the released run outputs, drop them here, and rerun.

  fig5a_rutile_sweep.csv
      columns: delta, arm, core, magnitude
      33 distortion amplitudes x 7 arms.  (main-text Fig. 5a)

  fig5b_jacobian_points.csv
      columns: core, arm, structure, seed, value
      20 crystals x 3 seeds x 6 arms = 360 rows.  (main-text Fig. 5b)

  figS1_raw_thresholds.csv
      columns: tau, core, arm, false_flag_fraction, sd
      25 thresholds x 7 arms, RAW coordinate variant.  (Supplementary Fig. 1)

OPTIONAL, and worth doing:
  fig2b_thresholds.csv
      columns: tau, core, arm, false_flag_fraction, sd
      All 25 thresholds for the IDEALIZED variant. The manuscript tabulates only five of
      them (Supplementary Table 3), so Fig. 2b currently shows five points where the
      original showed a full curve.

Until these land, figures/fig5_mechanism.pdf and figures/figS1_raw_thresholds.pdf keep their
ORIGINAL artwork, which carries real data. Style was never worth trading data for. The
restyled versions, with the missing panels marked in red, are in figures_nature_pending/ so
you can see what you will get.
