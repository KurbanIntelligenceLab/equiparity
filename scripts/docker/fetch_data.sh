#!/usr/bin/env bash
# Fetch the processed datasets into data/raw at runtime (they are not baked into the image).
#
# Downloads a tarball of the processed npz files (built by scripts/docker/package_data.sh) from
# an S3/HTTP URL and extracts it at the repo root. Override the URL with EQUIPARITY_DATA_URL.
#
# Usage:
#   bash scripts/docker/fetch_data.sh
#   EQUIPARITY_DATA_URL=https://my-bucket.s3.amazonaws.com/equiparity_processed_data.tar.gz \
#     bash scripts/docker/fetch_data.sh
set -euo pipefail

URL="${EQUIPARITY_DATA_URL:-https://npj-oc-data.s3.us-east-2.amazonaws.com/equiparity_processed_data.tar.gz}"
DEST="${EQUIPARITY_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
TARBALL="/tmp/equiparity_processed_data.tar.gz"

echo "Fetching processed data from: ${URL}"
if command -v aria2c >/dev/null 2>&1; then
    aria2c -x8 -s8 -o "${TARBALL}" "${URL}"
else
    curl -fSL -o "${TARBALL}" "${URL}"
fi

echo "Extracting into: ${DEST}"
tar -xzf "${TARBALL}" -C "${DEST}"
rm -f "${TARBALL}"

echo "Done. Processed data:"
find "${DEST}/data/raw" -name "*_processed.npz" -print 2>/dev/null || echo "  (none found — check the tarball layout)"
