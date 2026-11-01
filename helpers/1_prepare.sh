#!/usr/bin/env bash
# Step 1 of 3: get every new route ready for the review, with no questions asked.
#   a. links the segments of the shared folder (COMMA_VIDEOS_DIR, the same videos
#      openpilot.distill downloads) into data/raw/routes   (nnslr link-videos)
#   b. extracts the frames (1 per second) of the routes not extracted yet   (nnslr extract-frames)
#      plus the exact frame of every sign the comma read (signRead)         (nnslr sign-reads)
#   c. proposes the signs on the frames of the routes without a report     (nnslr preannotate)
#      and reads where the car was at each frame and in which country       (nnslr gps)
# Already done work is skipped, so it can be run after every fetch. Next: helpers/2_review.sh.
#
# Usage: helpers/1_prepare.sh [--jobs=<n>] [--background] [--again] [<route> ...]
#   helpers/1_prepare.sh --again             (proposes again the frames still to confirm of every route, e.g.
#                                             after choosing another proposer: the confirmed frames stay as they
#                                             are; a route all confirmed is left out)
#   helpers/1_prepare.sh                     (every new route, one at a time)
#   helpers/1_prepare.sh --jobs=4            (4 routes at once: more cores, faster)
#   helpers/1_prepare.sh --background        (goes on after the ssh session is closed)
#   helpers/1_prepare.sh 0000011e--0c8825628a
#
# Without --again a route that already has a pre-annotation report is never processed again. With it,
# the new report gets the confirmed frames of the old one (nnslr preannotate --keep-reviews), copied
# before it becomes the current one; it also adds the frames of the comma's reads. Segments of a route that arrive later stay unused (helpers/status.sh
# shows them).
# Every route writes its own log in data/logs/prepare/<route>.log.
# Who proposes the signs: NNSLR_PROPOSER=vietnam (the Vietnamese YOLO of Hugging Face), ours (our cutter,
# classifier and decider in use, on the GPU) or qwen; unset, the choice of the Prepare tab (data/web/proposer.json),
# vietnam only before any choice.
# -h, --help: this help.
# Env: COMMA_VIDEOS_DIR (default ../comma-videos next to this repo), NNSLR_DATA_ROOT (default data/)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export COMMA_VIDEOS_DIR="${COMMA_VIDEOS_DIR:-$(dirname "$ROOT")/comma-videos}"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv/bin/nnslr"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$DATA_ROOT/logs/prepare"
if [ -z "${NNSLR_PROPOSER:-}" ] && [ -f "$DATA_ROOT/web/proposer.json" ]; then
  NNSLR_PROPOSER="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("proposer") or "")' \
    "$DATA_ROOT/web/proposer.json" 2>/dev/null || true)"
fi
export NNSLR_PROPOSER="${NNSLR_PROPOSER:-vietnam}"

die() { echo "ERROR: $*" >&2; exit 1; }

# Prints the comment block at the top of this file.
usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# Routes still to extract / to pre-annotate (helpers/route_status.py), one per line.
todo() {
  (cd "$ROOT" && "$PY" -m helpers.route_status --todo="$1")
}

# The routes of this run, for the progress shown by the panel (/pannello).
write_batch() {
  local list=""
  local route
  for route in "$@"; do
    if [ -n "$list" ]; then
      list="$list, "
    fi
    list="$list\"$route\""
  done
  printf '{"started": "%s", "routes": [%s]}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$list" > "$LOG_DIR/batch.json.tmp"
  mv "$LOG_DIR/batch.json.tmp" "$LOG_DIR/batch.json"
}

# Extract (when needed) and pre-annotate one route; all its output goes to its log.
# Re-extracting is safe here: only routes without a report get this far.
# < /dev/null: ffmpeg reads commands from stdin and would eat the terminal (or the caller's script).
# The exact frame of every sign the comma read, next to the pictures at 1 a second (nnslr sign-reads): without the
# drive's logs there is none, and the route is proposed all the same.
sign_reads() {
  echo "$1: the frames of the signs the comma read..."
  "$NN" sign-reads "$1" < /dev/null >> "$2" 2>&1 \
    || echo "$1: frames of the comma's reads not added (see $2)"
}

prepare_route() {
  local route="$1"
  local log="$LOG_DIR/$route.log"
  local worked=0
  if [ "${PREPARE_AGAIN:-0}" = 1 ]; then
    # A route with frames still to confirm: the confirmed ones are copied onto the new report.
    todo repropose | grep -qx "$route" || { echo "$route: every frame confirmed, left as it is"; return 0; }
    sign_reads "$route" "$log"
    echo "$route: proposing the signs again (${NNSLR_PROPOSER:-vietnam})..."
    "$NN" preannotate --route "$route" --keep-reviews < /dev/null >> "$log" 2>&1 \
      || { echo "$route: proposing failed (see $log)"; return 1; }
    echo "$route: ready to review"
    return 0
  fi
  if todo extract | grep -qx "$route"; then
    echo "$route: taking the pictures out of the videos..."
    "$NN" extract-frames --route "$route" --overwrite < /dev/null >> "$log" 2>&1 \
      || { echo "$route: taking the pictures out failed (see $log)"; return 1; }
    worked=1
  fi
  if todo preannotate | grep -qx "$route"; then
    sign_reads "$route" "$log"
    echo "$route: proposing the signs (${NNSLR_PROPOSER:-vietnam})..."
    "$NN" preannotate --route "$route" < /dev/null >> "$log" 2>&1 \
      || { echo "$route: proposing failed (see $log)"; return 1; }
    # Where the car was at every frame and in which country (the review proposes it): not fatal.
    "$NN" gps --route "$route" < /dev/null >> "$log" 2>&1 \
      || echo "$route: GPS position not read (see $log): choose the country by hand in the review"
    echo "$route: ready to review"
    worked=1
  fi
  if [ "$worked" -eq 0 ]; then
    echo "$route: nothing to do (already proposed, or no videos linked)"
  fi
}

jobs=1
background=0
one=""
routes=()

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
  elif [ "$1" = "--background" ]; then
    background=1
  elif [ "$1" = "--again" ]; then
    export PREPARE_AGAIN=1
  elif [[ "$1" == --jobs=* ]]; then
    jobs="${1#--jobs=}"
  elif [[ "$1" == --one=* ]]; then
    # Internal: one route of a parallel run (--jobs).
    one="${1#--one=}"
  elif [[ "$1" =~ ^[0-9a-f]{8}--[0-9a-f]{10}$ ]]; then
    routes+=("$1")
  else
    die "unknown option or route: $1 (see --help)"
  fi
  shift
done

[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs takes a number from 1 up"
# Every pre-annotation uses its share of the cores: all of them across the routes prepared
# together (nnslr_tools.onnx_device). The routes of --jobs inherit the value of the first call.
if [ -z "${NNSLR_PREANNOTATE_THREADS:-}" ]; then
  cores="$(nproc 2>/dev/null || echo 4)"
  threads=$((cores / jobs))
  if [ "$threads" -lt 2 ]; then
    threads=2
  fi
  export NNSLR_PREANNOTATE_THREADS="$threads"
fi
[ -x "$NN" ] || die "$NN is missing: create the environment as in docs/INSTALL.md (python -m venv .venv, pip install -e '.[preannotate]')"
mkdir -p "$LOG_DIR"

if [ -n "$one" ]; then
  prepare_route "$one"
  exit $?
fi

if [ "$background" -eq 1 ]; then
  args=(--jobs="$jobs" "${routes[@]}")
  [ "${PREPARE_AGAIN:-0}" != 1 ] || args+=(--again)
  nohup "$0" "${args[@]}" < /dev/null > "$DATA_ROOT/logs/prepare.log" 2>&1 &
  echo "Started in the background (pid $!). Follow it with:  tail -f $DATA_ROOT/logs/prepare.log"
  echo "How far it is:  helpers/status.sh"
  exit 0
fi

# "== a. linking" is the step of the progress (nnslr_tools.job_progress.MARKERS).
echo "== a. linking the videos of $COMMA_VIDEOS_DIR"
"$NN" link-videos < /dev/null > "$LOG_DIR/link-videos.json" 2> "$LOG_DIR/link-videos.err" \
  || die "nnslr link-videos failed: see $LOG_DIR/link-videos.json and .err"

if [ "${#routes[@]}" -eq 0 ]; then
  if [ "${PREPARE_AGAIN:-0}" = 1 ]; then
    mapfile -t routes < <(todo repropose | sed '/^$/d' | sort -u)
  else
    mapfile -t routes < <( (todo extract; todo preannotate) | sed '/^$/d' | sort -u)
  fi
fi
if [ "${#routes[@]}" -eq 0 ]; then
  rm -f "$LOG_DIR/batch.json"
  echo "Nothing to prepare: no new drive."
else
  write_batch "${routes[@]}"
  echo "== b+c. ${#routes[@]} drives to prepare, $jobs at a time (logs in $LOG_DIR/)"
  failed=0
  if [ "$jobs" -eq 1 ]; then
    for route in "${routes[@]}"; do
      prepare_route "$route" || failed=1
    done
  else
    # xargs reads the routes from the pipe; the commands it starts get /dev/null as stdin.
    printf '%s\n' "${routes[@]}" | xargs -P "$jobs" -I{} "$0" --one={} || failed=1
  fi
  [ "$failed" -eq 0 ] || echo "Some drives failed: see their logs in $LOG_DIR/"
fi

# Routes prepared before the GPS was read: their positions too (nnslr gps, quick).
"$NN" gps --all --missing < /dev/null >> "$LOG_DIR/gps.log" 2>&1 \
  || echo "GPS position not read for some drives: see $LOG_DIR/gps.log"

echo
(cd "$ROOT" && "$PY" -m helpers.route_status)
