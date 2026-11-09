#!/usr/bin/env bash
# The signs on every frame of a drive's video, for Speed review: nnslr sign-tracks runs the networks in use (the SignNet
# cutter and the classifier, as Sign review runs them) on every decoded frame of the road camera of each drive and
# writes derived/sign_tracks/<route>.json, the boxes the page draws over the video at 20 frames a second. A drive read by
# the same networks on the same minutes is left as it is; a stopped run goes on from the minutes it finished. The portal
# runs it from the Signs on every frame card in Models.
#
# usage: helpers/sign_tracks.sh [--route R ...] [--dry-run] [--force]
#   no --route: every drive with its video (--all)
#   --dry-run: the first 41 frames of the first minute of one drive on the CPU, into derived/sign_tracks/dry, checked
#              against the frames of Sign review ("parity: ..."); the real files are not touched
# Effects: a real run uses the GPU (about 50 frames a second on the V100, some 9 hours for every drive) and writes
# derived/sign_tracks/. Only on request.
#
# A real run (not --dry-run) refuses a busy GPU before it starts, with one line and exit code 3: another compute process
# on it, or more than NNSLR_GPU_BUSY_PERCENT (10) percent of its memory taken. NNSLR_ALLOW_BUSY_GPU=1 shares it anyway,
# as in scripts/run_brain.sh. NNSLR_GPU_INDEX: the GPU (0). A dry run runs on the CPU and never looks at the GPU.
#
# The log, for the panel's progress (nnslr_tools.job_progress): "== 1/2 Check the GPU ==", "== 2/2 Find the signs on
# every frame ==", then the lines of nnslr: "drives: <N>", one "minute j/M: ..." a minute and one "drive i/N <route>: ..."
# a drive.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv-gpu/bin/nnslr"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
BUSY_PERCENT="${NNSLR_GPU_BUSY_PERCENT:-10}"
refuse() {
    echo "helpers/sign_tracks.sh: $1" >&2
    exit 3
}
[ -x "$NN" ] || { echo "helpers/sign_tracks.sh: no .venv-gpu (scripts/setup_v100_training_env.sh)" >&2; exit 2; }
[[ "$BUSY_PERCENT" =~ ^[0-9]+$ ]] || { echo "helpers/sign_tracks.sh: NNSLR_GPU_BUSY_PERCENT is a whole number" >&2; exit 2; }

# What the arguments ask for: a dry run, the drives named.
dry=0
named=0
for argument in "$@"; do
    case "$argument" in
        --dry-run) dry=1 ;;
        --route|--route=*) named=1 ;;
        --all) named=1 ;;
        -h|--help) exec "$NN" sign-tracks --help ;;
    esac
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

echo "== 1/2 Check the GPU =="
if [ "$dry" = 1 ]; then
    export CUDA_VISIBLE_DEVICES=
    echo "A dry run: on the CPU, the GPU is not checked."
else
    check_gpu
fi
every=()
[ "$named" = 1 ] || every=(--all)
echo "== 2/2 Find the signs on every frame =="
exec "$NN" sign-tracks --data-root "$DATA" ${every[@]+"${every[@]}"} "$@"
