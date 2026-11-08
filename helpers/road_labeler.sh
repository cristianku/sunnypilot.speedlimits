#!/usr/bin/env bash
# The road labeler, an offline network that prepares the labels of Speed review so that a person has less to fix: the
# road type of every moment, the kinds of road (tunnel and the others) and the roadworks, from the road camera at high
# definition, the OpenStreetMap scene, the GPS and the dashboard speed. It never runs on the car and is never an input of
# the car (docs/ROAD_LABELER.md).
#
#   train   nnslr road-labeler-train into a new run folder data/runs/<YYYY-MM-DD_HHMM>-road-labeler (the GPU of
#           .venv-gpu; --dry-run: a couple of steps on the CPU with a tiny encoder, nothing learned, the folder ends in
#           -road-labeler-dry and nothing is written outside it)
#   predict nnslr road-labeler-predict with the run given: derived/road_labeler/<route>.jsonl, and where it is sure the
#           marks of the kinds of road in Speed review (reviewer net/<run>; never over a person's mark, removal or
#           check). --write also writes the roadworks and the road types; --no-marks writes no mark at all
#
# usage: helpers/road_labeler.sh train [--dry-run] [--route R ...] [--epochs N]
#        helpers/road_labeler.sh predict --run-dir D (--all | --route R ...) [--no-marks] [--write] [--dry-run]
# Effects: train uses the GPU (the first run of a drive decodes its videos on 4 cores, about 13 s of CPU a minute of
# drive, and keeps its features in derived/road_labeler_features/); predict writes into Speed review. Only on request.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv-gpu/bin/nnslr"
usage() {
    echo "usage: helpers/road_labeler.sh train [--dry-run] [--route R ...] [--epochs N]" >&2
    echo "       helpers/road_labeler.sh predict --run-dir D (--all | --route R ...) [--no-marks] [--write] [--dry-run]" >&2
    exit 2
}
[ $# -ge 1 ] || usage
[ -x "$NN" ] || { echo "helpers/road_labeler.sh: no .venv-gpu (scripts/setup_v100_training_env.sh)" >&2; exit 2; }
action="$1"
shift
case "$action" in
    train)
        suffix=road-labeler
        for argument in "$@"; do
            [ "$argument" = --dry-run ] && suffix=road-labeler-dry
        done
        run="$DATA/runs/$(date '+%Y-%m-%d_%H%M')-$suffix"
        if [ "$suffix" = road-labeler-dry ]; then
            export CUDA_VISIBLE_DEVICES=
        fi
        echo "run: $run" >&2
        exec "$NN" road-labeler-train --data-root "$DATA" --run-dir "$run" "$@"
        ;;
    predict)
        for argument in "$@"; do
            [ "$argument" = --dry-run ] && export CUDA_VISIBLE_DEVICES=
        done
        exec "$NN" road-labeler-predict --data-root "$DATA" "$@"
        ;;
    *) usage ;;
esac
