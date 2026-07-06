#!/bin/bash
# Launch one grid-runner instance for a given profile (nequip or mace) on the cheapest 5090.
#
# Usage:
#   scripts/vast/launch_grid.sh --profile nequip --image johnpolat/equiparity:nequip
#   scripts/vast/launch_grid.sh --profile mace   --image johnpolat/equiparity:mace
# Flags: --profile (nequip|mace), --image <ref>, --max-price 0.9, --disk 40, --gpu either
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

PROFILE=""; IMAGE=""; MAX_PRICE="1.0"; DISK="40"; GPU="either"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --max-price) MAX_PRICE="$2"; shift 2 ;;
        --disk) DISK="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done
[[ -z "$PROFILE" || -z "$IMAGE" ]] && { echo "ERROR: --profile and --image required" >&2; exit 1; }

GIT_SHA="$(git -C "$PARITY_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
vast_load_api_key

echo ">>> Searching offer (gpu=${GPU} dlperf>${DLPERF_MIN} dph<${MAX_PRICE})..."
read -r OFFER_ID DPH GPU_NAME < <(vast_find_offer "$GPU" "$MAX_PRICE" "$DISK")
[[ -z "${OFFER_ID:-}" ]] && { echo "No qualifying offer (raise --max-price or lower --dlperf)" >&2; exit 1; }
echo ">>> Cheapest offer: id=${OFFER_ID} \$${DPH}/hr ${GPU_NAME}"

ONSTART="$(vast_inject_onstart "${_COMMON_DIR}/onstart_grid.sh" "PROFILE='${PROFILE}' GIT_SHA='${GIT_SHA}'")"
INSTANCE_ID="$(vast_create_start "$OFFER_ID" "$IMAGE" "$DISK" "equiparity-grid-${PROFILE}" "$ONSTART")"
[[ -z "${INSTANCE_ID:-}" ]] && { echo "ERROR: instance creation failed" >&2; exit 1; }

echo ">>> Launched grid-${PROFILE} instance ${INSTANCE_ID} (sha ${GIT_SHA})."
echo ">>> Watch:   vastai logs ${INSTANCE_ID}   (or tail /workspace/grid_run.log over SSH)"
echo ">>> Fetch:   scripts/vast/fetch_results.sh ${INSTANCE_ID}   (after GRID_DONE)"
echo ">>> Destroy: vastai destroy instance ${INSTANCE_ID}"
