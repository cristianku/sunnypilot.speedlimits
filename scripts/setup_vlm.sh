#!/usr/bin/env bash
set -euo pipefail

# Qwen3-VL-8B-Instruct (Alibaba, Apache-2.0), a model that sees images, on this server: to measure it as a reader of
# the boxes our cutter cuts out, and as a proposer on whole frames (docs/NEURAL_NETWORK.md). It does:
#
#   1. installs transformers, accelerate and qwen-vl-utils (pinned) into the GPU environment (.venv-gpu); nothing
#      already installed there changes version (checked with pip --dry-run on 25 Oct 2026);
#   2. downloads the model at a pinned revision (about 17.5 GB) into $DATA_ROOT/models/vlm/Qwen3-VL-8B-Instruct;
#      the files already there are checked (size, and the sha256 of the four weight files) and kept, so they can be
#      downloaded elsewhere on a faster line and copied in (only the four model-0000N-of-00004.safetensors are
#      large); a wrong file is downloaded again;
#   3. loads it on the GPU in float16 (the V100 has no bfloat16) and asks it about one reviewed frame.
#
# usage: scripts/setup_vlm.sh            (again: only what is missing is done)
#        scripts/setup_vlm.sh --no-test  (skip step 3, e.g. while the GPU is busy)
#
# Environment: NNSLR_DATA_ROOT (data/), NNSLR_GPU_VENV (.venv-gpu), NNSLR_GPU_INDEX (0).
# It never touches the reviews, the dataset, the models in use or the comma.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
REVISION="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TARGET="$DATA_ROOT/models/vlm/Qwen3-VL-8B-Instruct"
TEST=1
[[ "${1:-}" == "--no-test" ]] && TEST=0

[[ -x "$VENV/bin/python" ]] || { echo "missing $VENV: run scripts/setup_v100_training_env.sh first" >&2; exit 1; }

echo "== 1/3 Python packages in $VENV"
"$VENV/bin/python" -m pip install --quiet 'transformers==5.20.2' 'accelerate==1.15.0' 'qwen-vl-utils==0.0.14'
"$VENV/bin/python" -c "import transformers, accelerate; print('transformers', transformers.__version__, '· accelerate', accelerate.__version__)"

echo "== 2/3 $MODEL_ID at $REVISION into $TARGET"
mkdir -p "$TARGET"
"$VENV/bin/python" - "$MODEL_ID" "$REVISION" "$TARGET" <<'PY'
import hashlib, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
model_id, revision, target = sys.argv[1:]
# The sha256 of the weight files at this revision (Hugging Face LFS ids), to accept a copy brought in by hand.
SHA256 = {
    "model-00001-of-00004.safetensors": "d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa",
    "model-00002-of-00004.safetensors": "8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5",
    "model-00003-of-00004.safetensors": "83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192",
    "model-00004-of-00004.safetensors": "0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5",
}
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
files = [item for item in HfApi().list_repo_tree(model_id, revision=revision) if hasattr(item, "size")]
for item in files:
    path = Path(target) / item.path
    if path.is_file() and path.stat().st_size == item.size and (item.path not in SHA256 or sha256(path) == SHA256[item.path]):
        print("kept", item.path)
        continue
    if path.is_file():
        print("wrong or incomplete, downloaded again:", item.path)
        path.unlink()
    hf_hub_download(model_id, item.path, revision=revision, local_dir=target)
    if item.path in SHA256 and sha256(path) != SHA256[item.path]:
        sys.exit("checksum mismatch after download: " + item.path)
    print("downloaded", item.path)
print("model files in", target)
PY
echo "$REVISION" > "$TARGET/REVISION"

if [[ "$TEST" -eq 0 ]]; then
  echo "Installed (not tried on the GPU: --no-test)."
  exit 0
fi

echo "== 3/3 Try it on the GPU"
frame="$(ls "$DATA_ROOT"/derived/frames/*/*/frame_*.png 2>/dev/null | head -1 || true)"
[[ -n "$frame" ]] || { echo "no extracted frame to try it on: installed, not tried"; exit 0; }
# NNSLR_GPU_INDEX is nvidia-smi's index: CUDA counts the cards in the same order, by PCI bus.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${NNSLR_GPU_INDEX:-0}" "$VENV/bin/python" - "$TARGET" "$frame" <<'PY'
import sys, time
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
target, frame = sys.argv[1:]
started = time.time()
model = AutoModelForImageTextToText.from_pretrained(target, dtype=torch.float16, device_map="cuda", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained(target)
print(f"loaded in {time.time() - started:.0f} s, {torch.cuda.memory_allocated() / 1e9:.1f} GB on the GPU")
messages = [{"role": "user", "content": [{"type": "image", "image": frame},
             {"type": "text", "text": "List every traffic sign you see in this dashcam frame, with its speed value if it has one."}]}]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt").to(model.device)
started = time.time()
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
answer = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(f"answer in {time.time() - started:.1f} s on {frame}:\n{answer}")
PY
date -u +%Y-%m-%dT%H:%M:%SZ > "$TARGET/.installed"   # read by nnslr_tools.preannotate.vlm_missing
echo "Installed and tried: $TARGET"
