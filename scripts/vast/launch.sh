#!/bin/bash
# Launch an equiparity training run on the cheapest qualifying vast.ai GPU.
#
# Usage:
#   scripts/vast/launch.sh --image USER/equiparity:nequip --config configs/mp_piezoelectric_smoke.yaml
#   scripts/vast/launch.sh --gpu RTX_5090 --max-price 0.9 --dlperf 200 --seeds 0,1,2 --image ...
#
# Flags (defaults shown):
#   --gpu either            RTX_5090 | RTX_PRO_6000 | either (5090 then 6000)
#   --max-price 1.0         max $/hr
#   --disk 40               instance disk GB (image ~7 GB + data + outputs)
#   --dlperf 200            min vast DLPerf score
#   --config <path>         experiment config (default: configs/mp_piezoelectric_smoke.yaml)
#   --seeds 0               comma-separated seeds
#   --image <ref>           REQUIRED — the pushed docker image ref
#   --label <name>          instance label (default derived from config)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

GPU="either"; MAX_PRICE="1.0"; DISK="40"; CONFIG="configs/mp_piezoelectric_smoke.yaml"
SEEDS="0"; IMAGE=""; LABEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --max-price) MAX_PRICE="$2"; shift 2 ;;
        --disk) DISK="$2"; shift 2 ;;
        --dlperf) DLPERF_MIN="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done
[[ -z "$IMAGE" ]] && { echo "ERROR: --image is required (the pushed docker image ref)" >&2; exit 1; }
[[ -z "$LABEL" ]] && LABEL="equiparity_$(basename "${CONFIG%.yaml}")_${SEEDS//,/_}"

vast_load_api_key

echo ">>> Searching offers (gpu=${GPU} dlperf>${DLPERF_MIN} inet_down>${INET_DOWN_MIN} dph<${MAX_PRICE})..."
read -r OFFER_ID DPH GPU_NAME < <(vast_find_offer "$GPU" "$MAX_PRICE" "$DISK")
if [[ -z "${OFFER_ID:-}" ]]; then
    echo "No offer met the gates. Try raising --max-price or lowering --dlperf." >&2
    exit 1
fi
echo ">>> Cheapest offer: id=${OFFER_ID} \$${DPH}/hr ${GPU_NAME}"

ONSTART="$(vast_inject_onstart "${_COMMON_DIR}/onstart.sh" "CONFIG='${CONFIG}' SEEDS='${SEEDS}'")"
INSTANCE_ID="$(vast_create_start "$OFFER_ID" "$IMAGE" "$DISK" "$LABEL" "$ONSTART")"
[[ -z "${INSTANCE_ID:-}" ]] && { echo "ERROR: instance creation failed" >&2; exit 1; }

echo ">>> Launched instance ${INSTANCE_ID} (${LABEL})."
echo ">>> Watch:   vastai logs ${INSTANCE_ID}"
echo ">>> Fetch:   scripts/vast/fetch_results.sh ${INSTANCE_ID}"
echo ">>> Destroy: vastai destroy instance ${INSTANCE_ID}"
