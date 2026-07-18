# H-1 — false-flag fraction vs training epoch (re-instrumented retrains)

12 runs collected. Idealized variant, every epoch, threshold 0.01 C/m².

| core | seeds | ff@e1 | ff@e10 | ff@e50 | ff@e150 | first epoch ≥0.85 |
|---|---|---|---|---|---|---|
| allegro | 3 | 0.895 | 0.858 | 0.895 | 0.908 | 1 |
| equiformer_v2 | 3 | 0.803 | 0.618 | 0.907 | 0.956 | 23 |
| mace | 3 | 0.145 | 0.902 | 0.905 | 0.908 | 4 |
| nequip | 3 | 0.042 | 0.756 | 0.870 | 0.895 | 34 |

## Endpoint vs released headline (per run)

| run | endpoint | headline | diff |
|---|---|---|---|
| allegro_seed0 | 0.9095 | 0.9095 | +0.0000 |
| allegro_seed1 | 0.9080 | 0.9110 | -0.0030 |
| allegro_seed2 | 0.9075 | 0.9080 | -0.0005 |
| equiformer_v2_seed0 | 0.9595 | 0.9525 | +0.0070 |
| equiformer_v2_seed1 | 0.9485 | 0.9600 | -0.0115 |
| equiformer_v2_seed2 | 0.9595 | 0.9525 | +0.0070 |
| mace_seed0 | 0.9080 | 0.9085 | -0.0005 |
| mace_seed1 | 0.9080 | 0.9070 | +0.0010 |
| mace_seed2 | 0.9065 | 0.9075 | -0.0010 |
| nequip_seed0 | 0.8960 | 0.8960 | +0.0000 |
| nequip_seed1 | 0.8955 | 0.8955 | +0.0000 |
| nequip_seed2 | 0.8930 | 0.8945 | -0.0015 |
