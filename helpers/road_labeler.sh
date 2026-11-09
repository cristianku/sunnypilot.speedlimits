#!/usr/bin/env bash
# The road labeler, an offline network that labels Speed review, the base layer a person corrects only where it is wrong:
# the road type of every moment, the kinds of road (tunnel and the others) and the roadworks, from the road camera at
# high definition, the OpenStreetMap scene, the GPS and the dashboard speed. It never runs on the car and is never an
# input of the car (docs/ROAD_LABELER.md). The portal runs it from the Road labeler card at the bottom of Models.
#
#   train   nnslr road-labeler-train into a new run folder data/runs/<YYYY-MM-DD_HHMM>-road-labeler (the GPU of
#           .venv-gpu; --dry-run: a couple of steps on the CPU with a tiny encoder, nothing learned, the folder ends in
#           -road-labeler-dry and nothing is written outside it)
#   predict nnslr road-labeler-predict with the run given, Prepare Speed review: derived/road_labeler/<route>.jsonl (the
#           chances), and where it is sure, only for the heads that passed the gate, the marks of the kinds of road and
#           of the roadworks (reviewer net/<run>) and its layer of road types (derived/road_labeler_types/), all
#           definitive. It replaces an older run's marks that no person kept and its layer; a person's marks, removals
#           and checks and the tunnels of the map are never touched. --no-marks writes nothing into Speed review
#
# usage: helpers/road_labeler.sh train [--dry-run] [--route R ...] [--epochs N]
#        helpers/road_labeler.sh predict --run-dir D (--all | --route R ...) [--no-marks] [--dry-run]
# Effects: train uses the GPU (the first run of a drive decodes its videos on 4 cores, about 13 s of CPU a minute of
# drive, and keeps its features in derived/road_labeler_features/); predict writes into Speed review. Only on request.
#
# A real run (not --dry-run) refuses a busy GPU before it starts, with one line and exit code 3: another compute process
# on it, or more than NNSLR_GPU_BUSY_PERCENT (10) percent of its memory taken. NNSLR_ALLOW_BUSY_GPU=1 shares it anyway,
# as in scripts/run_brain.sh. NNSLR_GPU_INDEX: the GPU (0). A dry run runs on the CPU and never looks at the GPU.
#
# The log, for the panel's progress (nnslr_tools.job_progress): "run: <name>", "drives: <N>" (how many drives it goes
# through), "== 1/2 Check the GPU ==", "== 2/2 ... ==", then the lines of nnslr: one "<route>: features of ..." a drive,
# one "road_labeler epoch E/N ..." an epoch (train), one {"route_id": ...} a drive (predict).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv-gpu/bin/nnslr"
PY="$ROOT/.venv-gpu/bin/python"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
BUSY_PERCENT="${NNSLR_GPU_BUSY_PERCENT:-10}"
usage() {
    echo "usage: helpers/road_labeler.sh train [--dry-run] [--route R ...] [--epochs N]" >&2
    echo "       helpers/road_labeler.sh predict --run-dir D (--all | --route R ...) [--no-marks] [--dry-run]" >&2
    exit 2
}
refuse() {
    echo "helpers/road_labeler.sh: $1" >&2
    exit 3
}
[ $# -ge 1 ] || usage
[ -x "$NN" ] || { echo "helpers/road_labeler.sh: no .venv-gpu (scripts/setup_v100_training_env.sh)" >&2; exit 2; }
action="$1"
shift
[[ "$BUSY_PERCENT" =~ ^[0-9]+$ ]] || { echo "helpers/road_labeler.sh: NNSLR_GPU_BUSY_PERCENT is a whole number" >&2; exit 2; }

# What the arguments ask for: a dry run, the drives named, every drive, the run of predict.
dry=0
every=0
run_dir=''
routes=()
previous=''
for argument in "$@"; do
    case "$previous" in
        --route) routes+=("$argument") ;;
        --run-dir) run_dir="$argument" ;;
    esac
    case "$argument" in
        --dry-run) dry=1 ;;
        --all) every=1 ;;
        --route=*) routes+=("${argument#--route=}") ;;
        --run-dir=*) run_dir="${argument#--run-dir=}" ;;
    esac
    previous="$argument"
done

# The GPU free, or the run refused before it starts: no other compute process, and its memory taken under BUSY_PERCENT.
check_gpu() {
    if [ "${NNSLR_ALLOW_BUSY_GPU:-0}" = 1 ]; then
        echo "GPU $GPU_INDEX: shared on request (NNSLR_ALLOW_BUSY_GPU=1), not checked"
        return
    fi
    command -v nvidia-smi > /dev/null 2>&1 || refuse "no nvidia-smi, no GPU to run on (a --dry-run runs on the CPU)"
    local memory used total apps
    memory="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' ' || true)"
    used="${memory%%,*}"
    total="${memory##*,}"
    [[ "$used" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ && "$total" -gt 0 ]] \
        || refuse "the GPU $GPU_INDEX does not answer nvidia-smi (a --dry-run runs on the CPU)"
    apps="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | sed '/^[[:space:]]*$/d' | wc -l || true)"
    if [ "$apps" -gt 0 ] || [ $((used * 100)) -gt $((total * BUSY_PERCENT)) ]; then
        refuse "the GPU $GPU_INDEX is busy ($used of $total MiB used, $apps other compute processes): wait for it to finish, or --dry-run on the CPU"
    fi
    echo "GPU $GPU_INDEX free: $used of $total MiB used, no other compute process"
}

# How many drives it goes through, for the progress: the drives named, and with every=1 every scanned drive with its
# frames (road_labeler_data.labeler_routes); a dry training keeps the first DRY_DRIVES. Nothing when it cannot say.
count_drives() {
    "$PY" -c 'import sys
from pathlib import Path
from nnslr_tools import road_labeler
from nnslr_tools.road_labeler_data import labeler_routes
root, mode, every, *named = sys.argv[1:]
routes = list(dict.fromkeys(named))
if every == "1":
    routes += [route for route in labeler_routes(Path(root)) if route not in routes]
print(min(len(routes), road_labeler.DRY_DRIVES) if mode == "dry-train" else len(routes))' \
        "$DATA" "$1" "$2" ${routes[@]+"${routes[@]}"} 2>/dev/null || true
}

case "$action" in
    train)
        suffix=road-labeler
        [ "$dry" = 1 ] && suffix=road-labeler-dry
        run="$DATA/runs/$(date '+%Y-%m-%d_%H%M')-$suffix"
        echo "run:          $(basename "$run")"
        echo "output:       $run"
        echo "== 1/2 Check the GPU =="
        if [ "$dry" = 1 ]; then
            export CUDA_VISIBLE_DEVICES=
            echo "A dry run: on the CPU, the GPU is not checked."
        else
            check_gpu
        fi
        all=0
        [ ${#routes[@]} -eq 0 ] && all=1
        mode=train
        [ "$dry" = 1 ] && mode=dry-train
        echo "drives:       $(count_drives "$mode" "$all")"
        echo "== 2/2 Read the drives and their pictures, then train =="
        exec "$NN" road-labeler-train --data-root "$DATA" --run-dir "$run" "$@"
        ;;
    predict)
        echo "run:          $(basename "${run_dir:-?}")"
        echo "== 1/2 Check the GPU =="
        if [ "$dry" = 1 ]; then
            export CUDA_VISIBLE_DEVICES=
            echo "A dry run: on the CPU, the GPU is not checked."
        else
            check_gpu
        fi
        echo "drives:       $(count_drives predict "$every")"
        echo "== 2/2 Prepare Speed review drive by drive =="
        exec "$NN" road-labeler-predict --data-root "$DATA" "$@"
        ;;
    *) usage ;;
esac
