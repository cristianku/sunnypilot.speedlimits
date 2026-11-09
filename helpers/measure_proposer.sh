#!/usr/bin/env bash
# Measures a proposer of Prepare on the reviewed test frames, the roads no training saw (nnslr evaluate-proposer):
# speed signs found and read right, every sign found, false proposals per frame, seconds per frame. The result is
# the Proposers view of the Evaluate tab (its "Measure" buttons start this). Nothing is written into the reviews or
# the routes; the report goes to data/evaluations/proposers/.
#
# usage: helpers/measure_proposer.sh [vietnam|ours|qwen]   (default: NNSLR_PROPOSER, else vietnam)
# ours and qwen use the GPU; vietnam runs on the CPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# The GPU: NNSLR_GPU_INDEX, nvidia-smi's index (the panel gives a free one), which CUDA counts in the same order
# (by PCI bus); without it CUDA's first card.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${NNSLR_GPU_INDEX:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$NNSLR_GPU_INDEX"
fi
proposer="${1:-${NNSLR_PROPOSER:-vietnam}}"
echo "Measuring the proposer $proposer on the test frames..."
"$ROOT/.venv/bin/nnslr" evaluate-proposer --proposer "$proposer" --partition test > /dev/null
echo "Done: see the Proposers view of the Evaluate tab."
