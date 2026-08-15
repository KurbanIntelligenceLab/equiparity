#!/usr/bin/env bash
# NOTE: superseded by the resilient driver used for the split-box grid; kept for reference.
# Run the pooling-control grid: every config in configs/grid_meanpool and
# configs/grid_sumpool, routed to the environment its core requires
# (MACE pins e3nn 0.4.4; NequIP/Allegro/EquiformerV2 need e3nn 0.6.x).
#
# Both arms run on the SAME rebuilt OOD population, so the mean-vs-sum
# comparison is not confounded by the population change that came from
# regenerating the dataset (the original OOD material_ids were never committed).
#
# Usage: bash scripts/run_pooling_grid.sh <workdir>
set -uo pipefail

WORK="${1:-$PWD}"
cd "$WORK"
source /opt/conda/etc/profile.d/conda.sh

mkdir -p logs outputs
SUMMARY="grid_summary.tsv"
printf "config\tcore\tenv\texit\tseconds\trun_dir\n" > "$SUMMARY"

shopt -s nullglob
CONFIGS=(configs/grid_meanpool/*.yaml configs/grid_sumpool/*.yaml)
echo "grid: ${#CONFIGS[@]} configs"

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  core=$(grep -E '^core:' "$cfg" | awk '{print $2}')
  if [ "$core" = "mace" ]; then env_name=eqp-mace; else env_name=eqp-nequip; fi

  conda activate "$env_name"
  start=$(date +%s)
  # allow-dirty: the repo is unpacked from a tarball on the box, no git metadata
  PYTHONPATH="$WORK/src" python -m equiparity.cli run "$cfg" --allow-dirty \
      > "logs/${name}.log" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - start ))

  run_dir=$(ls -td outputs/*/ 2>/dev/null | head -1)
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$core" "$env_name" "$rc" "$elapsed" "$run_dir" >> "$SUMMARY"
  echo "[$(date +%H:%M:%S)] $name  core=$core  exit=$rc  ${elapsed}s"
  conda deactivate
done

echo "=== grid complete ==="
awk -F'\t' 'NR>1 && $4!=0 {n++} END {print "failed runs: " (n+0)}' "$SUMMARY"
column -t -s $'\t' "$SUMMARY" | head -90
