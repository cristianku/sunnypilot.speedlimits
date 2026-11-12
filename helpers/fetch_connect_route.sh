#!/usr/bin/env bash
# A drive someone shared in comma connect, downloaded into the shared folder of the comma files and linked into data/
# like a drive fetched from our comma (nnslr_tools.connect_fetch): the road camera, the wide camera and the rlog of every
# segment (an rlog.bz2 is re-packed as rlog.zst). The driver camera stays out. Then prepare it as usual
# (helpers/1_prepare.sh or Data on the site): it goes to Sign review, never to the brain or Speed review, whose answer is
# Cristian's speed (data/raw/connect/<route>.json marks it as a drive of another car).
#
# The owner first uploads every file of the drive in connect (open the drive, "Upload all files", the comma on Wi-Fi;
# by default a comma uploads only qcamera and qlog), switches the drive to public and sends the link.
#
# usage: helpers/fetch_connect_route.sh <link> [--segments=all|0-3|0,2,4-6] [--dry-run]
#   helpers/fetch_connect_route.sh https://connect.comma.ai/0123456789abcdef/0000012a--3f9c1e0b2d
#   helpers/fetch_connect_route.sh '0123456789abcdef|0000012a--3f9c1e0b2d' --dry-run
# Network: api.commadotai.com and the comma's blob store only, about 100 MB a minute of drive. Files already in the shared
# folder are kept, so it can run again after an interrupted download.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ "$#" -lt 1 ]; then
    echo "usage: helpers/fetch_connect_route.sh <link> [--segments=all|0-3|0,2,4-6] [--dry-run]" >&2
    exit 2
fi
cd "$ROOT"
exec "$ROOT/.venv/bin/python" -m nnslr_tools.connect_fetch "$@"
