#!/bin/bash
# Runs ON a vast.ai instance (base64-injected). Dedicated clifford_stf catch-up box.
#
# Runs ALL 12 clifford_stf runs ordered piezo -> elastic -> dipole -> U0 (headline/fast first),
# so the critical O(3) piezo->0 result lands early instead of buried behind slow float64 QM9 runs.
# Redundant runs vs the main grid boxes are harmless (dedup by run_label at flatten time).
#
# Env (injected by launch): GIT_SHA (recorded via EQUIPARITY_GIT_SHA),
#   SHARD_INDEX / SHARD_COUNT (split the piezo-first clifford list across boxes).
set -uo pipefail
cd /workspace
exec > >(tee -a /workspace/grid_run.log) 2>&1
touch /root/.no_auto_tmux
export EQUIPARITY_GIT_SHA="${GIT_SHA:-unknown}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"

echo ">>> GPU preflight..."
nvidia-smi || { echo "FATAL: no GPU"; exit 1; }
/workspace/.venv/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'FATAL: no CUDA')" || exit 1

/workspace/.venv/bin/python scripts/generate_grid.py
# Priority order: crystal targets first (small, fast), QM9 last (25k samples, float64 = slow).
FULL="/tmp/clifford_full.txt"; : > "$FULL"
for t in piezoelectric elastic dipole U0; do
    for s in 0 1 2; do
        f="configs/grid/clifford_stf_${t}_o3_seed${s}.yaml"
        [ -f "$f" ] && echo "$f" >> "$FULL"
    done
done
# This shard takes every SHARD_COUNT-th run (0-based); piezo-first order keeps the headline early.
RUNLIST="/tmp/clifford_runs.txt"
awk -v idx="$SHARD_INDEX" -v cnt="$SHARD_COUNT" 'NF && ((NR-1) % cnt == idx)' "$FULL" > "$RUNLIST"
N=$(grep -c . "$RUNLIST")
echo ">>> clifford shard ${SHARD_INDEX}/${SHARD_COUNT}: running ${N} configs piezo-first (sha ${EQUIPARITY_GIT_SHA})"

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
