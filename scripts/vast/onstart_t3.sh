#!/bin/bash
# Runs ON a vast.ai instance (base64-injected). Runs the T3 learning-curve sweep (nequip
# profile only) followed by this profile's H-1 epoch-curve retrains, from the :t3-* overlay
# image (instrumented trainers + N_zero npz baked in).
#
# Env (injected by launch_grid.sh):
#   PROFILE       nequip | mace
#   GIT_SHA       source commit, recorded in every manifest via EQUIPARITY_GIT_SHA
#   SHARD_INDEX   this shard's index (0-based); SHARD_COUNT total shards for this profile
set -uo pipefail
cd /workspace
exec > >(tee -a /workspace/grid_run.log) 2>&1
touch /root/.no_auto_tmux
chown -R root:root /root/.ssh 2>/dev/null || true
chmod 700 /root/.ssh 2>/dev/null || true
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true

PROFILE="${PROFILE:-nequip}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
export EQUIPARITY_GIT_SHA="${GIT_SHA:-unknown}"

echo ">>> GPU preflight..."
nvidia-smi || { echo "FATAL: no GPU"; exit 1; }
/workspace/.venv/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'FATAL: no CUDA')" || exit 1

# Regenerate configs from the baked generators (deterministic).
/workspace/.venv/bin/python scripts/generate_h1_grid.py
FULL="/tmp/t3_full_runs.txt"
: > "$FULL"
if [ "$PROFILE" = "nequip" ]; then
    /workspace/.venv/bin/python scripts/generate_t3_grid.py
    cat configs/t3/t3_nequip_runs.txt >> "$FULL"
fi
cat "configs/h1curve/h1curve_${PROFILE}_runs.txt" >> "$FULL"

# Take every SHARD_COUNT-th run for this shard (0-based) so instances split the load.
RUNLIST="/tmp/shard_runs.txt"
awk -v idx="$SHARD_INDEX" -v cnt="$SHARD_COUNT" 'NF && ((NR-1) % cnt == idx)' "$FULL" > "$RUNLIST"
N=$(grep -c . "$RUNLIST")
echo ">>> t3+h1curve ${PROFILE} shard ${SHARD_INDEX}/${SHARD_COUNT}: running ${N} configs (sha ${EQUIPARITY_GIT_SHA})"

i=0; fail=0
while read -r cfg; do
    [ -z "$cfg" ] && continue
    i=$((i + 1))
    echo ">>> [${i}/${N}] $(date -u) ${cfg}"
    /workspace/.venv/bin/equiparity run "$cfg" 2>&1 \
        | grep -iE "run complete|error|traceback|false.flag|violation" | tail -2 \
        || { echo ">>> FAILED: ${cfg}"; fail=$((fail + 1)); }
done < "$RUNLIST"

echo ">>> GRID DONE $(date -u). runs=${i} failures=${fail}"
touch /workspace/GRID_DONE
