# Aggregated tables (mean +/- s.d. over 3 seeds)

## Accuracy (test MAE)

| Core | Target | O(3) test MAE | SO(3) test MAE | Δ (SO3−O3) | Δ / seed-σ |
|---|---|---|---|---|---|
| NequIP | U0 | 53.6500 ± 22.5034 | 51.6647 ± 9.1886 | -1.9853 | -0.12 |
| NequIP | dipole | 0.0519 ± 0.0027 | 0.0530 ± 0.0026 | +0.0011 | +0.42 |
| NequIP | elastic | 24.3318 ± 0.2667 | 24.4862 ± 0.1402 | +0.1544 | +0.72 |
| NequIP | piezoelectric | 0.2083 ± 0.0077 | 0.2405 ± 0.0080 | +0.0322 | +4.09 |
| Allegro | U0 | 31.0763 ± 12.8944 | 26.9813 ± 4.8769 | -4.0950 | -0.42 |
| Allegro | dipole | 0.0751 ± 0.0023 | 0.0764 ± 0.0021 | +0.0013 | +0.58 |
| Allegro | elastic | 23.7240 ± 0.2460 | 23.8876 ± 0.3074 | +0.1637 | +0.59 |
| Allegro | piezoelectric | 0.2140 ± 0.0058 | 0.2589 ± 0.0176 | +0.0449 | +3.43 |
| MACE | U0 | 16.7629 ± 6.2887 | 26.0900 ± 12.4906 | +9.3271 | +0.94 |
| MACE | dipole | 0.0484 ± 0.0018 | 0.0500 ± 0.0045 | +0.0016 | +0.47 |
| MACE | elastic | 24.9191 ± 0.5436 | 24.9392 ± 0.5982 | +0.0201 | +0.04 |
| MACE | piezoelectric | 0.2222 ± 0.0066 | 0.2567 ± 0.0084 | +0.0345 | +4.55 |
| EquiformerV2 | U0 | - (SO(3)-only) | 20.7515 ± 5.9936 | - | - |
| EquiformerV2 | dipole | - (SO(3)-only) | 0.0379 ± 0.0013 | - | - |
| EquiformerV2 | elastic | - (SO(3)-only) | 35.1417 ± 3.1153 | - | - |
| EquiformerV2 | piezoelectric | - (SO(3)-only) | 0.2157 ± 0.0096 | - | - |

## OOD piezoelectric — both variants

| Core | Parity | Variant | false-flag @1e-2 | median violation | max violation |
|---|---|---|---|---|---|
| NequIP | O3 | idealized | 0.0000 ± 0.0000 | 3.19e-07 ± 5.5e-08 | 0.000456 |
| NequIP | O3 | raw | 1.67e-04 ± 2.9e-04 | 3.44e-07 ± 6.2e-08 | 0.00931 |
| NequIP | SO3 | idealized | 0.8953 ± 0.0008 | 0.6664 ± 0.0541 | 9.59 |
| NequIP | SO3 | raw | 0.8953 ± 0.0008 | 0.6714 ± 0.0543 | 9.59 |
| Allegro | O3 | idealized | 0.0000 ± 0.0000 | 3.84e-07 ± 2.2e-08 | 0.000972 |
| Allegro | O3 | raw | 0.0000 ± 0.0000 | 4.26e-07 ± 2.7e-08 | 0.0061 |
| Allegro | SO3 | idealized | 0.9095 ± 0.0015 | 0.9597 ± 0.1283 | 18.9 |
| Allegro | SO3 | raw | 0.9095 ± 0.0015 | 0.9697 ± 0.1364 | 18.9 |
| MACE | O3 | idealized | 0.0000 ± 0.0000 | 2.76e-06 ± 2.3e-07 | 0.00114 |
| MACE | O3 | raw | 3.33e-04 ± 2.9e-04 | 3.17e-06 ± 2.7e-07 | 0.0115 |
| MACE | SO3 | idealized | 0.9077 ± 0.0008 | 0.9249 ± 0.0545 | 15.8 |
| MACE | SO3 | raw | 0.9077 ± 0.0008 | 0.9301 ± 0.0555 | 15.8 |
| EquiformerV2 | SO3 | idealized | 0.9570 ± 0.0052 | 0.4447 ± 0.0905 | 17 |
| EquiformerV2 | SO3 | raw | 0.9563 ± 0.0054 | 0.4489 ± 0.0901 | 17 |

## Idealized vs raw OOD shift

| Core | Parity | ff(idealized) | ff(raw) | Δ |
|---|---|---|---|---|
| NequIP | O3 | 0.0000 ± 0.0000 | 1.67e-04 ± 2.9e-04 | +0.00017 |
| NequIP | SO3 | 0.8953 ± 0.0008 | 0.8953 ± 0.0008 | +0.00000 |
| Allegro | O3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | +0.00000 |
| Allegro | SO3 | 0.9095 ± 0.0015 | 0.9095 ± 0.0015 | +0.00000 |
| MACE | O3 | 0.0000 ± 0.0000 | 3.33e-04 ± 2.9e-04 | +0.00033 |
| MACE | SO3 | 0.9077 ± 0.0008 | 0.9077 ± 0.0008 | +0.00000 |
| EquiformerV2 | SO3 | 0.9570 ± 0.0052 | 0.9563 ± 0.0054 | -0.00067 |

## Timing (piezoelectric runs)

| Core | train s/epoch | throughput (struct/s) | peak GPU (MB) | OOD eval (s) |
|---|---|---|---|---|
| NequIP | 4.7 ± 1.5 | 616.5 ± 216.5 | 3167 | 5.8 |
| Allegro | 5.4 ± 0.2 | 492.8 ± 16.0 | 16346 | 7.8 |
| MACE | 20.6 ± 2.3 | 130.1 ± 14.5 | 4893 | 17.6 |
| EquiformerV2 | 25.4 ± 10.1 | 114.1 ± 37.9 | 4393 | 44.4 |
