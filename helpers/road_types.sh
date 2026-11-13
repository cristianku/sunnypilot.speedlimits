#!/usr/bin/env bash
# The map's limit of every frame of the prepared drives, from the GPS and the speed of the comma and the OpenStreetMap
# roads and car parks, read offline (nnslr road-types; the button "Read the map" on Data › Drives starts this): Map now
# and the Server today lane of the drive player, the map as it is today. Nothing learns from it, nothing here touches the
# reviews of the signs, and the comma never reads any of it.
#
#   a. the precise position of the road camera's frames from openpilot.distill's localizer, for the segments without
#      it (nnslr localize --all; about 20 s of a core a segment, 6 side by side; skipped without distill's .venv),
#      then the GPS position of the frames of the drives prepared without it or localized since (nnslr gps --all
#      --missing)
#   b. the maps the drives need: the Geofabrik extracts that hold a drive's positions and are not in raw/osm yet,
#      downloaded (scripts/osm_maps.py, at most 5 GB a run; derived/osm/maps.json for Read the map in Data › Drives). Only Geofabrik's
#      index and the extracts are fetched: no position of the drives leaves the server. A download that fails leaves
#      the maps present, and the job goes on: the drives outside them stay without the map's limit
#   c. the car's speed of every second, from the drives' logs: a stretch without GPS at a drive's start or end, or by a
#      garage of the map, is a garage when it is slow                                (nnslr car-motion --all, cached)
#   d. the map roads and car parks near the drives, again when a drive or the map is newer (scripts/osm_roads.py,
#      osmium from uv's cache for this run only: the environments of the site and of the training stay as they are),
#      and the names of the places that name the drives, again when the map is newer (scripts/osm_places.py),
#      and the traffic lights that Review with AI › Traffic lights picks its frames by (scripts/osm_signals.py)
#   e. the map road of every frame, its limit and its type                         (nnslr road-types --all)
#   f. the tunnels of the scanned drives, where the map and the lost GPS agree, written as marks of the kind tunnel in
#      Speed review (reviewer osm; never over a person's mark, removal or check)      (nnslr tunnels --all)
#
# usage: helpers/road_types.sh [--again]   (--again: the map roads read again even when nothing is newer)
# No network but Geofabrik's index once a month and the extracts the drives miss (download.geofabrik.de), uv's first
# download of osmium and the localizer's GNSS orbits and ionosphere maps of a new day (laika, from comma's gnss-data
# mirror); about 2 minutes when the map is read, seconds otherwise, plus about a minute for the logs of each new drive
# and one or two for each map downloaded.
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
PLACES="$DATA/derived/osm/places.jsonl"
SIGNALS="$DATA/derived/osm/traffic_signals.jsonl"
# The site starts this with the service's PATH, without ~/.local/bin where uv usually is.
UV="$(command -v uv || true)"
if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"; fi

echo "== a. GPS of the drives"
# Without openpilot.distill (or for a segment it cannot use) the frames keep the position of the GPS.
"$NN" localize --all --data-root "$DATA" < /dev/null \
    || echo "The localizer did not run for some segments: their frames keep the position of the GPS"
# A drive whose logs cannot be read keeps no position: it is said, and the others go on.
"$NN" gps --all --missing --data-root "$DATA" < /dev/null \
    || echo "GPS position not read for some drives: their frames stay without the map's limit"

echo "== b. Maps the drives need"
# shapely (the outlines of the extracts) is in the site's .venv with the OCR; without it, from uv's cache.
if "$ROOT/.venv/bin/python" -c 'import shapely' 2>/dev/null; then
    "$ROOT/.venv/bin/python" "$ROOT/scripts/osm_maps.py" --data-root "$DATA" < /dev/null \
        || echo "Not every map the drives need is here: the maps present are used, the job goes on"
elif [ -n "$UV" ]; then
    (cd "$ROOT" && "$UV" run --no-project --with shapely python scripts/osm_maps.py --data-root "$DATA" < /dev/null) \
        || echo "Not every map the drives need is here: the maps present are used, the job goes on"
else
    echo "uv is missing: the maps the drives need are not looked for (see docs/INSTALL.md)"
fi
if [ -z "$(find "$MAPS" -maxdepth 1 -name '*.osm.pbf' -print -quit 2>/dev/null)" ]; then
    echo "The map is missing: no $MAPS/*.osm.pbf, and none could be downloaded (see above)" >&2
    exit 2
fi

echo "== c. Speed of the drives"
# Without it the stretches without GPS stay unknown (or the map's car park); the rest is as before.
"$NN" car-motion --all --data-root "$DATA" < /dev/null \
    || echo "Speed not read for some drives: their stretches without GPS stay without a garage"

echo "== d. Map roads and car parks near the drives"
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
    # A map read before the junction and access tags were kept (the road types need them).
    newer=yes
elif ! grep -q -m 1 '"tunnel"' "$ROADS"; then
    # A map read before the tunnels were kept (f. needs them).
    newer=yes
fi
places=no
if [ "$again" = yes ] || [ ! -f "$PLACES" ] \
        || [ -n "$(find "$MAPS" -maxdepth 1 -name '*.osm.pbf' -newer "$PLACES" -print -quit)" ]; then
    places=yes
fi
if [ "$newer" = yes ] && [ -z "$UV" ]; then
    echo "uv is missing: it reads the map (see docs/INSTALL.md)" >&2
    exit 2
fi
if [ "$newer" = yes ]; then
    (cd "$ROOT" && nice "$UV" run --no-project --with osmium python scripts/osm_roads.py --data-root "$DATA")
else
    echo "The map roads and car parks are up to date."
fi
# Only the names of the drives: without them the limits are the same.
if [ "$places" = yes ] && [ -n "$UV" ]; then
    (cd "$ROOT" && nice "$UV" run --no-project --with osmium python scripts/osm_places.py --data-root "$DATA" \
        < /dev/null) || echo "The names of the places not read: the drives keep the names they had"
elif [ "$places" = yes ]; then
    echo "uv is missing: the names of the places not read, the drives keep the names they had"
fi

# Only Review with AI › Traffic lights: without them it says the map has no traffic lights.
if [ -n "$UV" ] && { [ "$again" = yes ] || [ ! -f "$SIGNALS" ] \
        || [ -n "$(find "$MAPS" -maxdepth 1 -name '*.osm.pbf' -newer "$SIGNALS" -print -quit)" ]; }; then
    (cd "$ROOT" && nice "$UV" run --no-project --with osmium python scripts/osm_signals.py --data-root "$DATA" \
        < /dev/null) || echo "The traffic lights of the map not read: Review with AI keeps the ones it had"
fi

echo "== e. Road type of every frame"
"$NN" road-types --all --data-root "$DATA" < /dev/null
echo "== f. Tunnels"
"$NN" tunnels --all --data-root "$DATA" < /dev/null || echo "The tunnels not written: Speed review keeps the marks it has"
echo "Done: the map's limits are in $DATA/derived/road_types/."
