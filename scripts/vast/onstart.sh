#!/bin/bash
# Runs ON the vast.ai instance (base64-injected by launch.sh). Trains one config across seeds.
#
# Env (injected by launch.sh; defaults shown):
#   CONFIG    experiment config path (default: configs/mp_piezoelectric_smoke.yaml)
#   SEEDS     comma-separated seeds (default: 0)
#   EQUIPARITY_DATA_URL  optional — if set, re-fetch data (otherwise the baked npz are used)
set -uo pipefail
cd /workspace
exec > >(tee -a /workspace/onstart.log) 2>&1

# Disable vast's auto-tmux so `ssh host cmd` (non-interactive) returns output instead of exit 127.
touch /root/.no_auto_tmux

CONFIG="${CONFIG:-configs/mp_piezoelectric_smoke.yaml}"
SEEDS="${SEEDS:-0}"

echo ">>> GPU preflight..."
nvidia-smi || { echo "FATAL: nvidia-smi failed — no GPU. Aborting."; exit 1; }
python -c "import torch, sys; print('GPU:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else sys.exit('FATAL: torch.cuda.is_available() is False')" || exit 1

# Data is baked into the image; only fetch if an override URL is provided.
if [[ -n "${EQUIPARITY_DATA_URL:-}" ]]; then
    echo ">>> EQUIPARITY_DATA_URL set — fetching override data..."
    bash scripts/docker/fetch_data.sh
fi
if [[ ! -f data/raw/qm9/qm9_processed.npz ]]; then
    echo "FATAL: processed data missing (not baked in and no EQUIPARITY_DATA_URL). Aborting."; exit 1
fi

FAIL=0
IFS=',' read -ra SEED_ARR <<< "$SEEDS"
for s in "${SEED_ARR[@]}"; do
    echo ">>> Running ${CONFIG} seed=${s}..."
    # seed is fixed in the config; override via a tiny sed on a copy so each seed is a distinct run.
    cfg="/tmp/run_seed${s}.yaml"
    sed "s/^seed: .*/seed: ${s}/" "${CONFIG}" > "${cfg}"
    equiparity run "${cfg}" || { echo ">>> FAILED: seed ${s}"; FAIL=$((FAIL+1)); }
done

echo ">>> Done. Failures: ${FAIL}. Results in /workspace/outputs/"
touch /workspace/PARITY_DONE
