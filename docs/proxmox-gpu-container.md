# Proxmox GPU container setup

This guide prepares a Debian/Ubuntu Proxmox LXC/container for
`sunnypilot.speedlimits`, local comma video/log processing, and later V100
training.

## Assumptions

- Linux x86_64 container.
- NVIDIA device passthrough is already configured.
- `nvidia-smi` works **inside the container**.
- The NVIDIA driver is managed on the Proxmox host/container setup already.
- This guide does not install or modify the GPU driver.
- Commands below assume root. Remove `sudo`/adjust ownership if you use a
  dedicated user.

Check first:

```bash
nvidia-smi
python3 --version || true
uname -m
```

For two V100s, `nvidia-smi -L` should list both devices.

---

## 1. Base OS packages

On Debian/Ubuntu:

```bash
apt update
apt install -y \
  git \
  git-lfs \
  python3 \
  python3-venv \
  python3-pip \
  build-essential \
  pkg-config \
  ca-certificates \
  curl \
  ffmpeg \
  zstd \
  jq
```

Verify the media tools:

```bash
ffmpeg -version | head -n 1
ffprobe -version | head -n 1
zstd --version
```

---

## 2. Software and data directories

The Git repository lives under `~/sunnypilot.speedlimits`.

The raw comma files (videos and full logs) are stored once in a shared folder
next to the clones, `~/comma-videos` (`COMMA_VIDEOS_DIR`), with the same layout
as on the comma: `~/comma-videos/<route>--<segment>/fcamera.hevc`,
`ecamera.hevc`, `rlog.zst`. `openpilot.distill` (`~/openpilot.distill`) uses
the same folder, so a segment is downloaded only once for both projects.

Extracted frames, datasets, checkpoints and training output live under
`~/sunnypilot.speedlimits/data` (gitignored, created on first use).
`data/raw/routes/<route>/<segment>/` only holds absolute symlinks to the files
in `~/comma-videos`. Neither folder may be committed to Git.

If the videos outgrow the container disk, give `~/comma-videos` its own disk
from the Proxmox host (`pct set <id> -mp0 ...`, run by hand on the host).

On the live server they are on the NAS since 1 Nov 2026: `/root/comma-videos`
is a symlink to `/mnt/comma-raw/comma-videos`, the NAS share the Proxmox host
mounts (NFS 4.2, soft) and passes into the container. The links of the data
root go through it unchanged. With the NAS off, downloading and preparing
drives fail with I/O errors; frames already extracted and the review keep
working, since they are on the container disk.

---

## 3. Clone sunnypilot.speedlimits

```bash
cd ~
git clone https://github.com/cristianku/sunnypilot.speedlimits.git
cd ~/sunnypilot.speedlimits
```

For an existing checkout:

```bash
cd ~/sunnypilot.speedlimits
git pull --ff-only
```

---

## 4. Create the NNSLR Python environment

```bash
cd ~/sunnypilot.speedlimits

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install pytest
```

Verify the CPU/tooling side:

```bash
nnslr --help
nnslr selftest
pytest -q
```

At this stage the core and T2 video tooling do not require PyTorch.

---

## 5. Optional: a different data root or videos folder

Without `NNSLR_DATA_ROOT` the data root is `~/sunnypilot.speedlimits/data`.
Without `COMMA_VIDEOS_DIR` the shared videos folder is `~/comma-videos`, the
folder `comma-videos` next to the clone. Only to keep them somewhere else:

```bash
cat >/etc/profile.d/nnslr.sh <<'EOF'
export NNSLR_DATA_ROOT=/path/to/speed-vision-data
export COMMA_VIDEOS_DIR=/path/to/comma-videos
EOF

chmod 0644 /etc/profile.d/nnslr.sh
source /etc/profile.d/nnslr.sh
```

`openpilot.distill` must see the same `COMMA_VIDEOS_DIR`, otherwise each
project downloads its own copy.

Check:

```bash
echo "$NNSLR_DATA_ROOT" "$COMMA_VIDEOS_DIR"
nnslr env
```

`nnslr env` shows the shared videos folder (`path`, filled also when
`COMMA_VIDEOS_DIR` is not set) and the data root; when `NNSLR_DATA_ROOT` is not
set, its `path` is `null` and the default is in its `note`:

```text
"nnsler_data_root": ... "/path/to/speed-vision-data"
"comma_videos_dir": ... "/path/to/comma-videos"
```

---

## 6. Get the videos and a quick video test

Download a route from the comma into the shared folder; `sync-routes` also
creates the links in `data/raw/routes`:

```bash
source ~/sunnypilot.speedlimits/.venv/bin/activate

nnslr sync-routes --route <route>
```

Segments that are already in `~/comma-videos` (for example downloaded by
`openpilot.distill`) are linked without any network access:

```bash
nnslr link-videos --dry-run          # show what would be linked, and the conflicts
nnslr link-videos                    # every route in ~/comma-videos
nnslr link-videos --route <route>    # one route only
```

`link-videos` links `fcamera.hevc`, `ecamera.hevc`, the full log and the other
raw comma files it finds; it never writes into `~/comma-videos`. A segment
with only the log and no `fcamera.hevc` (the automatic fetch keeps just the log
of the first/last segments it trims) is not linked and is listed in
`log_only`, not an error; links an earlier run made for it are removed
(`removed`), since `extract-frames` stops on a segment folder without video. A
segment with `fcamera.hevc` but without a full log is listed in
`missing_files` (exit code 1); its other files are still linked. Existing links are kept; a
link to a file that no longer exists (e.g. after moving `~/comma-videos`) is
pointed to the shared copy (`relinked`); an old regular file in
`data/raw/routes` that is identical to the shared copy is replaced by a link, a
different one is reported (`conflict` in the dry run, `failed` in the real run)
and left alone.

With `--videos-dir <dir>` also `export COMMA_VIDEOS_DIR=<dir>`: the readers
(`route-manifest`, `extract-frames`, ...) only accept links into
`COMMA_VIDEOS_DIR`.

If `sync-routes` reports `content differs` for a file in `~/comma-videos`
(for example cut off by an interrupted `openpilot.distill` fetch), rerun the
distill fetch or delete that file by hand, then rerun `sync-routes`.

The video is then available as:

```text
~/sunnypilot.speedlimits/data/raw/routes/<route>/<segment>/fcamera.hevc
  -> ~/comma-videos/<route>--<segment>/fcamera.hevc
```

Then:

```bash
source ~/sunnypilot.speedlimits/.venv/bin/activate

nnslr video-probe \
  ~/sunnypilot.speedlimits/data/raw/routes/<route>/<segment>/fcamera.hevc
```

Broad discovery extraction defaults to approximately 1 fps:

```bash
nnslr extract-frames \
  --video ~/sunnypilot.speedlimits/data/raw/routes/<route>/<segment>/fcamera.hevc \
  --output ~/sunnypilot.speedlimits/data/derived/frames/discovery
```

Focused extraction around a known interval:

```bash
nnslr extract-frames \
  --video ~/sunnypilot.speedlimits/data/raw/routes/<route>/<segment>/fcamera.hevc \
  --output ~/sunnypilot.speedlimits/data/derived/frames/encounter-001 \
  --start 120 \
  --end 135 \
  --fps 10
```

No GPU is needed for these steps.

---

## 7. Local qlog/rlog parsing environment

NNSLR does not bundle openpilot/cereal. To parse real local `rlog/qlog`
files, point NNSLR at a local checkout matching the software that generated
the logs.

For Cristian's runtime fork:

```bash
cd /opt
git clone --recursive --branch nn-speed-limit-vision https://github.com/cristianku/sunnypilot.git openpilot-src
```

If you already have the correct openpilot/sunnypilot source locally, use that
instead.

Current openpilot requires Python 3.12.x for its Python environment. A clean
way to keep this separate from the NNSLR venv is `uv`.

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Prepare a Python 3.12 environment for the runtime source:

```bash
cd /opt/openpilot-src

uv python install 3.12
uv sync --frozen
```

Then add these variables:

```bash
cat >>/etc/profile.d/nnslr.sh <<'EOF'
export NNSLR_OPENPILOT_ROOT=/opt/openpilot-src
export NNSLR_OPENPILOT_PYTHON=/opt/openpilot-src/.venv/bin/python
EOF

source /etc/profile.d/nnslr.sh
```

Verify:

```bash
"$NNSLR_OPENPILOT_PYTHON" --version
test -f "$NNSLR_OPENPILOT_ROOT/openpilot/tools/lib/logreader.py" && echo OK
```

Now a local compressed log can be read directly:

```bash
source ~/sunnypilot.speedlimits/.venv/bin/activate

nnslr log-metadata \
  ~/sunnypilot.speedlimits/data/raw/routes/<route>/<segment>/rlog.zst \
  --output ~/sunnypilot.speedlimits/data/manifests/log-metadata.jsonl
```

No manual `zstd -d` step is required when the matching openpilot LogReader
environment is available.

---

## 8. Video/log alignment

Example for the narrow road camera:

```bash
nnslr align-route \
  --video ~/sunnypilot.speedlimits/data/raw/routes/<route>/<segment>/fcamera.hevc \
  --log ~/sunnypilot.speedlimits/data/raw/routes/<route>/<segment>/rlog.zst \
  --stream narrow_road \
  --segment-num <segment> \
  --output ~/sunnypilot.speedlimits/data/derived/alignment/<segment>.jsonl
```

Inspect:

```bash
nnslr alignment-report \
  ~/sunnypilot.speedlimits/data/derived/alignment/<segment>.jsonl
```

The production alignment uses openpilot `EncodeIndex.segmentId`, which is the
camera-file index in presentation order. It does not equate encode order with
decoded presentation order.

---

## 9. Prepare the V100 PyTorch environment

The Tesla V100 is Volta, compute capability `sm_70`. Use the repository's
pinned setup script rather than duplicating package-version commands manually:

```bash
cd ~/sunnypilot.speedlimits
bash scripts/setup_v100_training_env.sh
source .venv-gpu/bin/activate
```

The script creates a separate training venv, installs the pinned PyTorch
CUDA 12.6 profile plus Pillow/ONNX/ONNX Runtime dependencies, installs NNSLR
editable, and prints the detected CUDA build information. It does **not**
replace NVIDIA drivers and does **not** execute a GPU workload.

Do not install a separate CUDA toolkit merely because `nvidia-smi` works.
The selected PyTorch wheel carries its CUDA runtime dependencies; the
host/container NVIDIA driver must be compatible with that runtime.

---

## 10. Explicit V100 smoke test

After installation, run the GPU workload explicitly:

```bash
source ~/sunnypilot.speedlimits/.venv-gpu/bin/activate
nnslr env --gpu-smoke --gpu-device 0
```

For a V100, verify that the report shows capability `[7, 0]`, includes
`sm_70` in the compiled architecture list, and reports
`fp16_forward_backward: true`.

For an additional GPU, select its CUDA index explicitly:

```bash
nnslr env --gpu-smoke --gpu-device 1
```

Also check ownership/occupancy before training:

```bash
nvidia-smi
```

Do not start a real NNSLR training run on a GPU that is still reserved by
another service.

---

## 11. Recommended shell workflow

For video/dataset work:

```bash
source ~/sunnypilot.speedlimits/.venv/bin/activate
source /etc/profile.d/nnslr.sh
```

For GPU training later:

```bash
source ~/sunnypilot.speedlimits/.venv-gpu/bin/activate
source /etc/profile.d/nnslr.sh
```

Do not mix the two virtual environments.

---

## 12. Updating NNSLR

CPU/T2 environment:

```bash
cd ~/sunnypilot.speedlimits
git pull --ff-only

source .venv/bin/activate
python -m pip install -e .
pytest -q
```

GPU environment after training dependencies change:

```bash
cd ~/sunnypilot.speedlimits
source .venv-gpu/bin/activate
python -m pip install -e .
```

---

## 13. Basic troubleshooting

### `nvidia-smi` works but PyTorch says CUDA is unavailable

Check:

```bash
source ~/sunnypilot.speedlimits/.venv-gpu/bin/activate

python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import torch; print(torch.cuda.get_arch_list())'
nvidia-smi
```

On V100, if the installed wheel does not contain `sm_70`, replace it with the
CUDA 12.6 build documented above.

### `ffmpeg` or `ffprobe` not found

```bash
apt install -y ffmpeg
which ffmpeg
which ffprobe
```

### qlog/rlog parser cannot import openpilot

Check:

```bash
echo "$NNSLR_OPENPILOT_ROOT"
echo "$NNSLR_OPENPILOT_PYTHON"

"$NNSLR_OPENPILOT_PYTHON" - <<'PY'
from openpilot.tools.lib.logreader import LogReader
print("LogReader import OK")
PY
```

If this fails, rebuild/sync the separate openpilot environment, not the NNSLR
core environment.

### Keep raw data out of Git

Before doing any development from the container:

```bash
cd ~/sunnypilot.speedlimits
git status
```

Raw videos and logs belong in `~/comma-videos`; datasets, runs and
checkpoints belong under `~/sunnypilot.speedlimits/data` (gitignored), never
in the tracked part of the repository.
