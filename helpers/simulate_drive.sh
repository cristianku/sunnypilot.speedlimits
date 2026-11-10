#!/usr/bin/env bash
# Models › Drive: drives replayed through the car's own vision loop with a set of networks, to see on their video what
# the comma's screen would show (the speed to hold, the sign read). nnslr simulate gives the car's loop (the sunnypilot
# clone's pipeline.py) every frame of the road camera's video and the car's messages of the log, and writes
# derived/simulation/<drive>/<set>.json.gz, a row per frame read. With --certified it replays every certified drive:
# the regression test of a new set. The portal runs it from Models › Drive (Compute, Replay the certified drives).
#
# usage: helpers/simulate_drive.sh --set=car|in-use (--route=R [--route=R ...] | --certified) [--again]
#   --set=car     the release in the sunnypilot clone (what the comma downloads)
#   --set=in-use  the versions in use on Models: exported first as Publish would (PyTorch, in .venv-gpu), unless they
#                 are exported already
#   Either set: the road type head of its brain (type_head, for the training only) is read from the brain's weights
#   (PyTorch, in .venv-gpu) into derived/simulation/type-heads/, and the replay names the road type the brain sees.
#   --again       also a drive replayed already with these networks
# Effects: writes derived/simulation/ and uses a V100: the video is decoded by the GPU and the networks run on it.
# ALWAYS ON THE GPU: there is no CPU run. The replay takes about 1 GB of a card and shares it with what else runs
# there: NNSLR_GPU_INDEX names the card (the panel gives the one with the most free memory), without it the card with the
# most free memory. It refuses a card with less than NNSLR_SIMULATE_GPU_MB (2048) free before it starts, with one line
# and exit code 3.
#
# The log, for the panel's progress (nnslr_tools.job_progress): "== 1/2 Get the networks ready ==", "== 2/2 Replay the
# drive ==", then the lines of nnslr: "drives: <N>", one "minute j/M: ..." a minute and one "drive i/N <route>: ..." a
# drive.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export NNSLR_DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT/data}"
NN="$ROOT/.venv/bin/nnslr"
NN_GPU="${NNSLR_GPU_VENV:-$ROOT/.venv-gpu}/bin/nnslr"
GPU_INDEX="${NNSLR_GPU_INDEX:-}"
NEEDED_MB="${NNSLR_SIMULATE_GPU_MB:-2048}"
refuse() {
    echo "helpers/simulate_drive.sh: $1" >&2
    exit 3
}
bad() {
    echo "helpers/simulate_drive.sh: $1" >&2
    exit 2
}

set_key=""
arguments=()
for argument in "$@"; do
    case "$argument" in
        --set=*) set_key="${argument#--set=}" ;;
        --route=*) arguments+=(--route "${argument#--route=}") ;;
        --certified|--again) arguments+=("$argument") ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) bad "unknown argument: $argument" ;;
    esac
done
case "$set_key" in
    car|in-use) ;;
    *) bad "which networks? --set=car or --set=in-use" ;;
esac
[ "${#arguments[@]}" -gt 0 ] || bad "which drives? --route=<drive> or --certified"
[ -x "$NN" ] || bad "no .venv (helpers/update.sh installs it)"

echo "== 1/2 Get the networks ready =="
if [ "$set_key" = in-use ]; then
    [ -x "$NN_GPU" ] || bad "no GPU environment for the export of the versions in use: $NN_GPU (helpers/3_train.sh installs it)"
    "$NN_GPU" simulate-set --set in-use
elif [ -x "$NN_GPU" ]; then
    "$NN_GPU" simulate-set --set car
else
    "$NN" simulate-set --set car
fi

echo "== 2/2 Replay the drive =="
# A card with the room, or the run refused before it starts: the replay never moves to the CPU.
command -v nvidia-smi > /dev/null 2>&1 || refuse "no nvidia-smi, no GPU to replay on: the replay never runs on the CPU"
if [ -z "$GPU_INDEX" ]; then
    GPU_INDEX="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | sort -t, -k2 -n -r | head -n 1 | cut -d, -f1 | tr -d ' ')"
    [ -n "$GPU_INDEX" ] || refuse "nvidia-smi sees no GPU to replay on: the replay never runs on the CPU"
fi
free_mb="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
[ -n "$free_mb" ] || refuse "nvidia-smi does not read GPU $GPU_INDEX: the replay never runs on the CPU"
[ "$free_mb" -ge "$NEEDED_MB" ] \
    || refuse "GPU $GPU_INDEX has only $free_mb MB free and the replay needs $NEEDED_MB: free some memory on it (the replay never runs on the CPU)"
echo "GPU $GPU_INDEX: $free_mb MB free, shared with what else runs there"
# The run is on the card checked: nvidia-smi's index, which CUDA counts in the same order (by PCI bus).
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_INDEX"
exec "$NN" simulate --set "$set_key" "${arguments[@]}"
