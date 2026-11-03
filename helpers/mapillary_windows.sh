#!/usr/bin/env bash
# The photos of the Mapillary Traffic Sign Dataset (MTSD) cut into windows the size of the car's band, for the
# cutter's training (nnslr mapillary-windows, nnslr_tools.mtsd_windows): <data root>/external/mapillary/windows/
# and windows.jsonl. Each window is 1344x432 around a speed sign, at a scale that makes the sign as wide as the
# signs of our own drives, so SignNet sees it as it sees the road camera's top band. The Mapillary switch of
# scripts/run_signnet.sh (NNSLR_CUTTER_MAPILLARY=1 adds them to our train, =pretrain trains on them first) reads
# them; validation and test stay our drives only.
#
# Only the fully annotated archives are read, opened by name from
#
#   /data/mapillary/zips   (NNSLR_MAPILLARY_DIR/zips; on the server the TrueNAS share)
#
# mtsd_v2_fully_annotated_annotation.zip and mtsd_v2_fully_annotated_images.{train.0,train.1,train.2,val}.zip;
# the partially annotated ones are never opened. By default only the train split is cut (--with-val adds val).
#
# usage: helpers/mapillary_windows.sh [--dry-run] [--limit N] [--workers N] [--with-val]
# No network, no GPU; the CPU for a while, the share read once. Run it again after a read error of the share: the
# windows already cut are not cut again.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv/bin/nnslr"
SOURCE="${NNSLR_MAPILLARY_DIR:-/data/mapillary}"
# On the server /data/mapillary is the NFS share bound into the container: the bind is always a mountpoint, so a
# share that is down shows as the local filesystem (zfs) under it.
if [ "$SOURCE" = /data/mapillary ] && [ "$(findmnt -no FSTYPE "$SOURCE")" != nfs4 ]; then
    echo "mapillary: /data/mapillary is not mounted (the TrueNAS share): ask cluster-admin, do not copy around it" >&2
    exit 1
fi
echo "== Mapillary: the windows for the cutter, from $SOURCE/zips"
nice -n 10 "$NN" mapillary-windows --mtsd "$SOURCE/zips" --data-root "$DATA" "$@"
