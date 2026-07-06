#!/bin/bash
# Runs ON a vast.ai instance (base64-injected). Runs this image's profile subset of the ablation grid.
#
# Env (injected by launch_grid.sh):
#   PROFILE   nequip | mace   (which run-list to execute)
#   GIT_SHA   source commit, recorded in every manifest via EQUIPARITY_GIT_SHA
set -uo pipefail
cd /workspace
exec > >(tee -a /workspace/grid_run.log) 2>&1
touch /root/.no_auto_tmux

PROFILE="${PROFILE:-nequip}"
export EQUIPARITY_GIT_SHA="${GIT_SHA:-unknown}"

echo ">>> GPU preflight..."
nvidia-smi || { echo "FATAL: no GPU"; exit 1; }
/workspace/.venv/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'FATAL: no CUDA')" || exit 1

# Regenerate the grid configs from the baked generator (deterministic), then run this profile's list.
/workspace/.venv/bin/python scripts/generate_grid.py
RUNLIST="configs/grid/${PROFILE}_runs.txt"
[ -f "$RUNLIST" ] || { echo "FATAL: $RUNLIST missing"; exit 1; }
N=$(grep -c . "$RUNLIST")
echo ">>> ${PROFILE}: running ${N} configs (sha ${EQUIPARITY_GIT_SHA})"

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
