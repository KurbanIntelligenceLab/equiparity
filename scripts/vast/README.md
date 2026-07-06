# Cloud training on vast.ai

Launch equiparity training on rented GPUs. Adapted from the E3-GRAND vast pattern; the API key
is read from `../agents-mlip/.env` (`VAST_API_KEY`).

## One-time: build + push the image

The image bakes in the 62 MB processed data, so the processed npz must exist first:

```bash
uv sync --extra nequip --extra data
uv run python scripts/prepare_qm9.py
MP_TOKEN=<key> uv run python scripts/prepare_mp.py
docker build --build-arg PROFILE=nequip -t <user>/equiparity:nequip .
docker push <user>/equiparity:nequip          # (docker login first)
# MACE runs use a separate image: --build-arg PROFILE=mace
# If apt fails with "Temporary failure resolving deb.debian.org", the build container can't
# resolve DNS on that host — add --network=host to the build command.
```

## Launch

```bash
scripts/vast/launch.sh --image <user>/equiparity:nequip \
    --config configs/mp_piezoelectric_smoke.yaml --seeds 0,1,2
# then:
vastai logs <id>                          # watch
scripts/vast/fetch_results.sh <id>        # pull outputs/ back
vastai destroy instance <id>              # stop paying
```

## GPU and precision

- **RTX 5090** (primary) — fastest FP32/TF32, cheapest, CUDA 12.8 ready. Training runs in
  **float32 mixed precision** (default), which is ~6.8x faster than float64 at production size
  (measured: 9.8 vs 66.9 ms/step for a 128-feature, l_max=3 model). Set `precision: float64` in a
  config only for high-precision verification.
- **RTX PRO 6000** (Blackwell 96 GB) — automatic availability fallback (`--gpu either`), not needed
  for memory.
- **Memory is a non-issue.** Models are ~50–120k params; molecules ≤29 atoms; training crystals
  ≤288 atoms; OOD eval ≤444 atoms at small batch. 32 GB is far more than enough — no A100 needed.

## Offer filters (`_common.sh`, all tunable via env or flags)

- `DLPERF_MIN=200` (vast DLPerf compute score) — the confirmed filter.
- `INET_DOWN_MIN=200` Mbps — so the ~3.5 GB image pulls fast.
- `RELIABILITY_MIN=0.99`, `DURATION_MIN_DAYS=7`, `num_gpus=1`, `rentable=true`, sorted cheapest first.

Measured: a single RTX 5090 scores **~199–200 DLPerf** on vast, so `dlperf>200` is right at the
edge — it keeps the fastest 5090s but excludes the many at 199.x. If `launch.sh` finds no offer
(especially with a tight `--max-price`), drop `--dlperf 195` to capture all 5090s; the other gates
(reliability, inet_down, price) still apply. Dry-run confirmed a 5090 at $0.47/hr with `--dlperf 200`.

## Dry run (no instance created)

```bash
source scripts/vast/_common.sh && vast_load_api_key && vast_find_offer either 1.0 40
# -> "OFFER_ID DPH GPU_NAME" for the cheapest qualifying 5090/6000, or nothing.
```
