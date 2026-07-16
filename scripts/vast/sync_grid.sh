#!/bin/bash
# Poll each grid instance: rsync its outputs/ to Desktop/parity_work, capture progress, flatten.
#
# Usage: scripts/vast/sync_grid.sh <ID> [ID ...]
# Dest:  ~/Desktop/parity_work/{raw/box<ID>/, metrics/<run_label>.json, summary.csv, status.txt}
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

DEST="${PARITY_SYNC_DEST:-$HOME/Desktop/parity_work}"
mkdir -p "$DEST/raw"
vast_load_api_key

STATUS="$DEST/status.txt"
: > "$STATUS"
echo "=== grid sync $(date -u '+%Y-%m-%d %H:%M UTC') ===" | tee -a "$STATUS"

for ID in "$@"; do
    read -r HOST PORT < <(vast_ssh_hostport "$ID" 2>/dev/null)
    if [[ -z "${HOST:-}" ]]; then
        echo "box $ID: NO SSH ENDPOINT (down/stopped?)" | tee -a "$STATUS"
        continue
    fi
    SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -p ${PORT}"
    # Progress line: current [i/N] config + whether GRID_DONE exists.
    PROG=$(ssh $SSH_OPTS "root@${HOST}" \
        'D=""; [ -f /workspace/GRID_DONE ] && D=" [GRID_DONE]"; \
         L=$(grep -E "^>>> \[[0-9]" /workspace/grid_run.log 2>/dev/null | tail -1); \
         F=$(ls /workspace/outputs 2>/dev/null | wc -l); \
         echo "at:${L}${D} completed_dirs:${F}"' 2>/dev/null || echo "SSH FAILED")
    echo "box $ID: $PROG" | tee -a "$STATUS"
    rsync -az --timeout=60 -e "ssh ${SSH_OPTS}" \
        "root@${HOST}:/workspace/outputs/" "$DEST/raw/box${ID}/" 2>/dev/null \
        || echo "box $ID: rsync incomplete" | tee -a "$STATUS"
done

"${PARITY_ROOT}/.venv/bin/python" "${_COMMON_DIR}/flatten_results.py" "$DEST" 2>/dev/null \
    || python3 "${_COMMON_DIR}/flatten_results.py" "$DEST"
echo ">>> synced $# box(es) -> $DEST" | tee -a "$STATUS"
