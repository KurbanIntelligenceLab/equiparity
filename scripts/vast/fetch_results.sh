#!/bin/bash
# Pull an instance's outputs/ (per-run manifest + metrics.json) back to the local repo.
#
# Usage: scripts/vast/fetch_results.sh <INSTANCE_ID> [DEST_DIR]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ID="${1:?usage: fetch_results.sh <INSTANCE_ID> [DEST_DIR]}"
DEST="${2:-${PARITY_ROOT}/outputs}"
mkdir -p "$DEST"

vast_load_api_key
read -r HOST PORT < <(vast_ssh_hostport "$ID")
[[ -z "${HOST:-}" ]] && { echo "ERROR: no SSH endpoint for instance ${ID}" >&2; exit 1; }

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${PORT}"
echo ">>> Pulling outputs from root@${HOST}:${PORT} -> ${DEST}"
rsync -az -e "ssh ${SSH_OPTS}" "root@${HOST}:/workspace/outputs/" "${DEST}/"
scp ${SSH_OPTS} "root@${HOST}:/workspace/onstart.log" "${DEST}/onstart_${ID}.log" 2>/dev/null || true
echo ">>> Done. Metrics:"
find "${DEST}" -name metrics.json -newermt '-1 hour' 2>/dev/null | head
