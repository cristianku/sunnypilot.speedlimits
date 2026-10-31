#!/usr/bin/env bash
# The type of road of every frame of the prepared drives (town, 30 zone, rural, expressway, motorway, ramp, merge lane,
# parking, garage), from the GPS and the speed of the comma and the OpenStreetMap roads and car parks, read offline: the labels of
# the road-type head, prepared without a person (nnslr road-types; the button "Find the road types" of Data › Roads
# starts this). A person corrects them later in the road-type review; nothing here touches the reviews of the signs.
#
#   a. the precise position of the road camera's frames from openpilot.distill's localizer, for the segments without
#      it (nnslr localize --all; about 20 s of a core a segment, 6 side by side; skipped without distill's .venv),
#      then the GPS position of the frames of the drives prepared without it or localized since (nnslr gps --all
#      --missing)
#   b. the car's speed of every second, from the drives' logs: a stretch without GPS at a drive's start or end, or by a
#      garage of the map, is a garage when it is slow                                (nnslr car-motion --all, cached)
#   c. the map roads and car parks near the drives, again when a drive or the map is newer (scripts/osm_roads.py,
#      osmium from uv's cache for this run only: the environments of the site and of the training stay as they are)
#   d. the road type of every frame                                                (nnslr road-types --all)
#   e. the intersections every drive passed: the labels of the road-type network's junction head (nnslr junctions
#      --all, data/derived/junctions; a person's corrections, annotations/junctions, are never touched)
#
# usage: helpers/road_types.sh [--again]   (--again: the map roads read again even when nothing is newer)
# No network but uv's first download of osmium and the localizer's GNSS orbits and ionosphere maps of a new day (laika,
# from comma's gnss-data mirror); about 2 minutes when the map is read, seconds otherwise, plus about a minute for the
# logs of each new drive.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv/bin/nnslr"
again=no
for argument in "$@"; do
    case "$argument" in
        --again) again=yes ;;
        *) echo "usage: helpers/road_types.sh [--again]" >&2; exit 2 ;;
    esac
done
MAPS="$DATA/raw/osm"
ROADS="$DATA/derived/osm/roads.jsonl"
PARKING="$DATA/derived/osm/parking.jsonl"
if [ -z "$(find "$MAPS" -maxdepth 1 -name '*.osm.pbf' -print -quit 2>/dev/null)" ]; then
    echo "The map is missing: no $MAPS/*.osm.pbf (the Geofabrik extracts, see docs/COMMANDS.md)" >&2
    exit 2
fi

echo "== a. GPS of the drives"
# Without openpilot.distill (or for a segment it cannot use) the frames keep the position of the GPS.
"$NN" localize --all --data-root "$DATA" < /dev/null \
    || echo "The localizer did not run for some segments: their frames keep the position of the GPS"
# A drive whose logs cannot be read keeps no position: it is said, and the others go on.
"$NN" gps --all --missing --data-root "$DATA" < /dev/null \
    || echo "GPS position not read for some drives: their frames stay without a road type"

echo "== b. Speed of the drives"
# Without it the stretches without GPS stay unknown (or the map's car park); the rest is as before.
"$NN" car-motion --all --data-root "$DATA" < /dev/null \
    || echo "Speed not read for some drives: their stretches without GPS stay without a garage"

echo "== c. Map roads and car parks near the drives"
newer=no
if [ "$again" = yes ] || [ ! -f "$ROADS" ]; then
    newer=yes
elif [ -n "$(find "$MAPS" -maxdepth 1 -name '*.osm.pbf' -newer "$ROADS" -print -quit)" ]; then
    # A map added or downloaded again.
    newer=yes
elif [ -n "$(find "$DATA/derived/gps" -name positions.jsonl -newer "$ROADS" -print -quit)" ]; then
    newer=yes
elif ! grep -q -m 1 '"maxspeed_type"' "$ROADS"; then
    # A file written before the kind of limit was kept.
    newer=yes
elif [ ! -f "$PARKING" ]; then
    # A map read before the car parks (and the kind of a service road) were kept.
    newer=yes
elif ! grep -q -m 1 '"junction"' "$ROADS"; then
    # A map read before the junction and access tags were kept (the intersections need them).
    newer=yes
fi
if [ "$newer" = yes ]; then
    # The site starts this with the service's PATH, without ~/.local/bin where uv usually is.
    UV="$(command -v uv || true)"
    if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"; fi
    if [ -z "$UV" ]; then
        echo "uv is missing: it reads the map (see docs/INSTALL.md)" >&2
        exit 2
    fi
    (cd "$ROOT" && nice "$UV" run --no-project --with osmium python scripts/osm_roads.py --data-root "$DATA")
else
    echo "The map roads and car parks are up to date."
fi

echo "== d. Road type of every frame"
"$NN" road-types --all --data-root "$DATA" < /dev/null

echo "== e. Junctions of the drives"
"$NN" junctions --all --data-root "$DATA" < /dev/null
echo "Done: the road types are in $DATA/derived/road_types/, the junctions in $DATA/derived/junctions/."
