#!/bin/bash
# Runs ON a vast.ai instance (base64-injected). Re-runs this profile's PIEZO configs only, on the
# idealized-OOD image — every core's piezo OOD is re-evaluated against the symmetry-idealized
# centrosymmetric set (see scripts/idealize_ood.py) for a consistent, artifact-free false-flag.
#
# Env (injected by launch): PROFILE (nequip|mace), GIT_SHA, SHARD_INDEX, SHARD_COUNT.
set -uo pipefail
cd /workspace
exec > >(tee -a /workspace/grid_run.log) 2>&1
touch /root/.no_auto_tmux
PROFILE="${PROFILE:-nequip}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
export EQUIPARITY_GIT_SHA="${GIT_SHA:-unknown}"

echo ">>> GPU preflight..."; nvidia-smi || { echo "FATAL: no GPU"; exit 1; }
/workspace/.venv/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'FATAL: no CUDA')" || exit 1

/workspace/.venv/bin/python scripts/generate_grid.py
# piezo-only subset of this profile's run-list; clifford (float64) is the slow one, put it last.
FULL="/tmp/piezo_full.txt"
grep 'piezoelectric' "configs/grid/${PROFILE}_runs.txt" | grep -v clifford_stf > "$FULL"
grep 'piezoelectric' "configs/grid/${PROFILE}_runs.txt" | grep clifford_stf >> "$FULL" || true
RUNLIST="/tmp/piezo_runs.txt"
awk -v idx="$SHARD_INDEX" -v cnt="$SHARD_COUNT" 'NF && ((NR-1) % cnt == idx)' "$FULL" > "$RUNLIST"
N=$(grep -c . "$RUNLIST")
echo ">>> ${PROFILE} PIEZO re-run shard ${SHARD_INDEX}/${SHARD_COUNT}: ${N} configs (idealized OOD, sha ${EQUIPARITY_GIT_SHA})"

i=0; fail=0
while read -r cfg; do
    [ -z "$cfg" ] && continue
    i=$((i + 1)); echo ">>> [${i}/${N}] $(date -u) ${cfg}"
    /workspace/.venv/bin/equiparity run "$cfg" 2>&1 \
        | grep -iE "run complete|error|traceback|false.flag|violation" | tail -2 \
        || { echo ">>> FAILED: ${cfg}"; fail=$((fail + 1)); }
done < "$RUNLIST"
echo ">>> PIEZO DONE $(date -u). runs=${i} failures=${fail}"
touch /workspace/GRID_DONE
