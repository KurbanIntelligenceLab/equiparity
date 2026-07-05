#!/usr/bin/env bash
# Package the processed datasets into a tarball for upload to S3 (run on the host after the
# prepare scripts). The cloud image's fetch_data.sh downloads and extracts this tarball.
#
# The tarball holds only the processed npz (a few tens of MB), NOT the ~670 MB of raw QM9 xyz.
#
# Usage:
#   bash scripts/docker/package_data.sh                 # -> equiparity_processed_data.tar.gz
#   Then upload, e.g.:
#     aws s3 cp equiparity_processed_data.tar.gz s3://npj-oc-data/
#     # or: rclone copy equiparity_processed_data.tar.gz r2:my-bucket/
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO}"
OUT="equiparity_processed_data.tar.gz"

FILES=()
for f in \
    data/raw/qm9/qm9_processed.npz \
    data/raw/mp/mp_piezoelectric_processed.npz \
    data/raw/mp/mp_elastic_processed.npz \
    data/raw/mp/mp_ood_centrosymmetric_processed.npz ; do
    if [[ -f "$f" ]]; then FILES+=("$f"); else echo "warning: missing $f (run the prepare scripts first)"; fi
done

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "error: no processed npz found; run scripts/prepare_qm9.py and scripts/prepare_mp.py first" >&2
    exit 1
fi

tar -czf "${OUT}" "${FILES[@]}"
echo "Wrote ${OUT} ($(du -h "${OUT}" | cut -f1)) with ${#FILES[@]} files."
