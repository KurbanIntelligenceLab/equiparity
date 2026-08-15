#!/usr/bin/env bash
# Resilient pooling-grid driver.
#
# Two failure modes broke the first attempt, both fixed here:
#
#  1. A single failing run killed the whole grid. The driver ran under
#     `bash -eo pipefail`, so the first non-zero training exit aborted the
#     loop and 28 queued runs never started. Here every run is wrapped so a
#     failure is RECORDED and the loop continues.
#
#  2. MACE piezoelectric SO(3) hit CUDA OOM at batch_size 16 on a 24 GB card.
#     The SO(3) arm carries ~40% more parameters than O(3) (7,453,120 vs
#     5,329,920 for MACE piezoelectric), so the arm that fits in O(3) can
#     exceed memory in SO(3) at the same batch size. On OOM the run is retried
#     at successively smaller batch sizes.
#
# Batch size affects optimisation, so it must stay matched WITHIN a
# mean/sum comparison pair. The retry writes the batch size it actually used
# into the summary, and any run that needed a reduction is reported so its
# partner can be re-run at the same size before the arms are compared.
#
# Usage: bash run_pooling_grid_v2.sh <list-file>
set -uo pipefail

LIST="${1:-mylist.txt}"
WORK="$PWD"
source /opt/conda/etc/profile.d/conda.sh
mkdir -p logs outputs
SUMMARY="grid_summary_v2.tsv"
[ -f "$SUMMARY" ] || printf "config\tcore\texit\tseconds\tbatch_size\tnote\n" > "$SUMMARY"

# Fragmentation is a contributing factor in the observed OOM (5.31 GiB was
# reserved-but-unallocated), so allow the allocator to grow segments.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

total=$(grep -cve '^[[:space:]]*$' "$LIST")
n=0
while read -r cfg; do
  [ -z "$cfg" ] && continue
  n=$((n+1))
  name=$(basename "$cfg" .yaml)

  if grep -qP "^\Q${name}\E\t" "$SUMMARY" 2>/dev/null; then
    echo "[$n/$total] $name  already recorded, skipping"
    continue
  fi

  core=$(grep -E '^core:' "$cfg" | awk '{print $2}')
  if [ "$core" = "mace" ]; then env_name=eqp-mace; else env_name=eqp-nequip; fi
  conda activate "/root/work/envs/$env_name"

  rc=1; used_bs=""; note=""; elapsed=0
  for bs in 16 8 4 2; do
    run_cfg="$cfg"
    if [ "$bs" != "16" ]; then
      run_cfg="retry_${name}_bs${bs}.yaml"
      sed "s/^\(\s*\)batch_size:.*/\1batch_size: ${bs}/" "$cfg" > "$run_cfg"
    fi
    start=$(date +%s)
    PYTHONPATH="$WORK/src" python -m equiparity.cli run "$run_cfg" --allow-dirty \
        > "logs/${name}.log" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start ))
    used_bs="$bs"
    if [ "$rc" -eq 0 ]; then
      [ "$bs" != "16" ] && note="oom_retry_bs${bs}"
      break
    fi
    if grep -q "OutOfMemoryError" "logs/${name}.log"; then
      echo "    OOM at batch_size=$bs, retrying smaller"
      cp "logs/${name}.log" "logs/${name}.oom_bs${bs}.log"
      continue
    fi
    note="failed_non_oom"
    break
  done

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$core" "$rc" "$elapsed" "$used_bs" "$note" >> "$SUMMARY"
  echo "[$n/$total] $name core=$core exit=$rc bs=$used_bs ${elapsed}s ${note}"
  conda deactivate
done < "$LIST"

echo "=== grid finished ==="
awk -F'\t' 'NR>1 && $3!=0 {n++} END {print "failed runs: " (n+0)}' "$SUMMARY"
awk -F'\t' 'NR>1 && $5!="" && $5!="16" {print "  reduced batch: " $1 " -> bs " $5}' "$SUMMARY"
tar czf run_outputs.tar.gz outputs logs "$SUMMARY" 2>/dev/null
ls -lh run_outputs.tar.gz
