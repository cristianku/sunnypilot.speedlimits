# Installation

Everything a fresh machine needs to use this repository, from the CPU tools to GPU training and publishing. Each part is optional: install only what you use.

Tested on Debian 13 (trixie) and macOS. Ubuntu 24.04 works the same way with `apt`.

## 1. System packages

```sh
apt update
apt install -y git curl rsync openssh-client python3 python3-venv ffmpeg zstd jq
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv: fast venvs and pip (optional, recommended)
```

| Package | Used by |
| --- | --- |
| `python3` (3.11 or newer), `python3-venv` | everything |
| `ffmpeg` (`ffmpeg` + `ffprobe`) | `nnslr extract-frames`, `nnslr video-probe`, `nnslr make-clips` |
| `zstd` | reading `rlog.zst` / `qlog.zst` from the shell |
| `rsync`, `openssh-client` | `nnslr sync-routes` (copies routes from the comma device over SSH) |
| `curl` | `scripts/publish_model.sh` (downloads the model catalog) |
| `jq` | inspecting the JSON/JSONL reports by hand |
| `git` | cloning and updating |

On macOS: `brew install ffmpeg zstd jq uv`.

GPU training also needs a working NVIDIA driver: `nvidia-smi` must work where you train (inside the container, for an LXC). This repository never installs or changes the driver; see [proxmox-gpu-container.md](proxmox-gpu-container.md) for a Proxmox LXC.

## 2. Python environments

The core CLI has no dependencies on purpose. Heavier tools live in optional extras or in a separate GPU environment, so the CPU tools stay small.

### Core CLI and tests (`.venv`)

```sh
cd sunnypilot.speedlimits
uv venv .venv                     # or: python3 -m venv .venv
uv pip install --python .venv/bin/python -e . pytest
.venv/bin/nnslr selftest
.venv/bin/python -m pytest -q
```

### Pre-annotation (same `.venv`, extra `preannotate`)

```sh
uv pip install --python .venv/bin/python -e ".[preannotate]"
```

Adds `onnxruntime`, `Pillow`, `numpy` and `rapidocr==3.4.0` (CPU). On a machine with an NVIDIA GPU (the V100 LXC) use the extra `preannotate-gpu` instead: `onnxruntime-gpu[cuda,cudnn]` below 1.27 with cuDNN 9.10.2.21 (CUDA 12: the CUDA 13 builds and the newer cuDNN no longer support the V100) instead of `onnxruntime`, and the detector, the road-surface model and the OCR run on the GPU (CPU fallback; `NNSLR_PREANNOTATE_DEVICE=cpu` forces the CPU). The two extras install the same `onnxruntime` module, so remove the other one first (`uv pip uninstall --python .venv/bin/python onnxruntime`); `helpers/update.sh` does this by itself when it finds an NVIDIA GPU. Each model uses `NNSLR_PREANNOTATE_THREADS` CPU threads (`helpers/1_prepare.sh`: cores / routes in parallel; default half of the cores). A frame is mostly work for one CPU core (the PNG, the tiles, the road mask, the OCR, the JPEG), so `NNSLR_PREANNOTATE_FRAMES` frames are proposed side by side in separate processes, each with its own models and its share of the threads (default a quarter of the cores, at most 8: on the V100 LXC 0.16 s a frame instead of 0.5 s; 16 is no faster and takes 7 GB more memory). `nnslr preannotate` downloads its models on the first run and checks their SHA-256; they are cached in `<data root>/models/preannotation/<sha256>/model.onnx`:

| Model | Source | License |
| --- | --- | --- |
| Traffic-sign detector (YOLO, 82 classes, Vietnam signs) | `star092304/traffic-sign-detection-vietnam-yolo` on Hugging Face | model card Apache-2.0, embedded model AGPL-3.0 |
| Road segmentation (SegFormer-b0 Cityscapes, ONNX) | Hugging Face, pinned revision in `src/nnslr_tools/road_text.py` | NVIDIA SegFormer: non-commercial research and evaluation only |
| OCR for numbers painted on the road (PP-OCRv4) | shipped with `rapidocr` | Apache-2.0 |

These models only propose boxes for human review; they are not part of the trained speed-limit models.

### Reading comma logs (extra `log-reader`)

```sh
uv pip install --python .venv/bin/python -e ".[log-reader]"
```

Adds `pycapnp` and `zstandard` for `nnslr log-metadata`, `nnslr align-route` and the comma download (`helpers/fetch_comma_segments.sh`), which without it drops every trip.

### Precise frame positions: openpilot.distill's localizer (optional)

`nnslr localize` runs the localizer of [openpilot.distill](https://github.com/cristianku/openpilot.distill) (its `localizer/`) on every segment of the drives: the raw GNSS, the car's motion and the calibration give the position and velocity of every road-camera frame, where the GPS fix is a few metres off and late. It needs a clone of openpilot.distill with its `.venv` (laika, numpy, safetensors, openpilot's LogReader: `git clone --recurse-submodules` and `uv sync` there, as its README says), next to this clone (`../openpilot.distill`) or named by `NNSLR_DISTILL_ROOT`. Nothing is installed here and nothing is written into openpilot.distill: its Python runs `src/nnslr_tools/localize_child.py` on the segment and the output goes to `data/derived/localizer/`.

```sh
.venv/bin/nnslr localize --all          # about 20 s of a core and 220 MB a segment, 6 side by side
.venv/bin/nnslr gps --all --missing     # the frames of the localized drives take the localizer's position
```

The first run of a day downloads that day's GNSS orbits and ionosphere maps (laika, from comma's public gnss-data mirror on GitHub, no account) into `/tmp/gnss`. Without openpilot.distill's `.venv`, `nnslr localize` says so and the frames keep the position of the GPS, as before. From the portal it runs as step a of **Read the map** (Data › Drives, `helpers/road_types.sh`).

### The OpenStreetMap maps (downloaded by themselves)

The map's limits (**Read the map**, on Data › Drives: Map now and the Server today lane of the drive player) read the Geofabrik extracts of OpenStreetMap in `data/raw/osm/`. You no longer download them by hand. Each run first looks for the extracts the drives need (`scripts/osm_maps.py`, step b. of `helpers/road_types.sh`) and downloads the missing ones from download.geofabrik.de: a Swiss drive needs Switzerland, a drive near Basel needs Alsace or Baden-Württemberg too, and a drive in the US needs its state. A few positions on a lake or the sea, or a GPS fix that jumped there, never start the download of a whole region such as us-midwest or the US. Only Geofabrik's index and the extracts are fetched, so no position of the drives leaves the server. A run downloads at most 5 GB and leaves 2 GB free on the disk. An extract is kept only when it is whole and its md5 matches. The space needed is the size of the extracts: about 550 MB for Switzerland, 650 MB for Baden-Württemberg, 200 to 400 MB for most US states.

What it needs:

- `uv` (section 1): `osmium` (pyosmium) reads the maps from uv's cache, for that run only, and is never installed in a `.venv`;
- `shapely` for the outlines of the extracts: it is in the `.venv` with the `preannotate` extra (rapidocr brings it); without it, `helpers/road_types.sh` takes it from uv's cache too.

Without the network, the maps already there are used: the drives outside them stay without a road type until a run can download their map. An extract can still be added by hand (any `*.osm.pbf` in `data/raw/osm/`); it is used as it is and never deleted.

### Training on a GPU (`.venv-gpu`)

```sh
bash scripts/setup_v100_training_env.sh
.venv-gpu/bin/wandb login          # once; without it training logs to wandb offline
CUDA_DEVICE_ORDER=PCI_BUS_ID .venv-gpu/bin/nnslr env --gpu-smoke --gpu-device 0
CUDA_DEVICE_ORDER=PCI_BUS_ID .venv-gpu/bin/nnslr env --gpu-smoke --gpu-device 1   # the second card, on a server with two
```

Installs `torch==2.14.0` and `torchvision==0.29.0` from the CUDA 12.6 index (the last line that still supports the Tesla V100, sm_70), `onnx`, `onnxscript`, `onnxruntime` and the `training` extra (`wandb`). On a newer GPU you can install the default PyPI `torch` instead. `--gpu-device` is nvidia-smi's number of the card (here `0` = bus 81:00.0, `1` = C1:00.0); `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes CUDA count the cards the same way, as the training scripts do.

### A model that sees images: Qwen3-VL-8B (optional)

```sh
scripts/setup_vlm.sh               # again: only what is missing; --no-test while the GPU is busy
```

Adds `transformers`, `accelerate` and `qwen-vl-utils` to `.venv-gpu` (no installed package changes version) and downloads [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (Apache-2.0, about 17.5 GB) at a pinned revision into `data/models/vlm/Qwen3-VL-8B-Instruct`. It then loads it on the GPU of `NNSLR_GPU_INDEX` (0; from the panel a free card) in float16 (the V100 has no bfloat16, about 17 GB of memory) and asks it about one frame.

On a slow line, download the four weight files elsewhere (`model-00001-of-00004.safetensors` … `model-00004-of-00004.safetensors` of that revision, 17.5 GB together) and copy them into `data/models/vlm/Qwen3-VL-8B-Instruct/`; then run the script: it checks the sha256 of each file, keeps the right ones and downloads only what is missing or wrong.

The same is the button **Install Qwen3-VL** under **Found by** in Data › Drives, shown until the model has been checked and tried on the GPU (the script leaves `.installed` in the model's folder). Once installed, it is the third choice of **Find the signs → Found by** in Data › Drives (`nnslr preannotate --proposer qwen`); until then that choice says what is missing.

### An AI reviewer: any OpenAI-compatible server (optional)

Nothing to install: the portal talks to the server with the Python standard library. A review can then be done by an AI that sees images, frame by frame on the review page (**Ask AI**) or a whole route at a time (**Review with AI** on the route list). Any server with the OpenAI API works (`/v1/models` and `/v1/chat/completions`): a hosted provider, or a llama.cpp `llama-server` on another machine, started with a vision model and its `--mmproj` file (for example Qwen3-VL-32B-Instruct), so that it sees the frames.

Add it in the portal, under **Account → Settings → AI reviewer**: a name, the address ending in `/v1`, the API key if the server wants one, and the model. Then press **Test** and choose it as the reviewer. The same from a terminal, for example for strata:

```sh
.venv/bin/nnslr llm-provider add --name strata --base-url https://strata.farm.14bodhi.com/v1 --key-file /root/.config/strata-api-key
.venv/bin/nnslr llm-provider test strata
.venv/bin/nnslr llm-provider use strata
```

The providers and their keys are kept in `data/web/private/llm-providers.json`, with mode 600 in a folder with mode 700, outside git. `NNSLR_LLM_PROVIDERS=/full/path.json` puts the file elsewhere. The key is read from `--key-file` and never printed. No page, API answer, log or report shows it: the browser only learns that a key is set. **Test** says whether the model answers in JSON and reads the speed sign of a frame that a person reviewed. A model that the server lists as text only cannot see the frames, and Test says so. An AI review saved from **Review and save** is final: nobody approves it again, and the next import puts it into training as an AI review (`ai/<provider>/<model>`). The commands are in [COMMANDS.md](COMMANDS.md).

### Publishing a model on Hugging Face

Nothing to install: `scripts/publish_model.sh` and `scripts/publish_networks.sh` (our three networks, Models › Publish) run the Hugging Face library through `uvx`.

From the portal: **Settings → Hugging Face** is a three-step wizard. It opens Hugging Face's page for a new token with Write access, takes the pasted token, asks Hugging Face whose it is and whether it can write, and keeps it in `data/web/private/huggingface.json` (mode 600, outside git). The token never goes back to a browser or into a log. **Publish on Hugging Face** passes it to its job in the environment (`HF_TOKEN`).

From a terminal instead, log in once with a token that can write the target repository:

```sh
uvx --from "huggingface_hub>=1,<3" hf auth login
```

## 3. Data folders

| Folder | Default | Override | Content |
| --- | --- | --- | --- |
| Data root | `<clone>/data` | `NNSLR_DATA_ROOT` or `--data-root` | frames, annotations, datasets, runs, cached models |
| Comma videos | `<clone>/../comma-videos` | `COMMA_VIDEOS_DIR` | `fcamera.hevc`, `rlog.zst`, ... per `<route>--<segment>/`, shared with other projects |

Both stay out of Git. Videos are stored once in the comma-videos folder and linked into the data root, so another project on the same machine (for example openpilot.distill) can use the same files.

### data/derived on the local disk

On the LXC, `data/derived` (the frames, videos and sign reads that Prepare rebuilds, about 185 GB) is on the local disk, beside the databases in `data/annotations`.

From the morning to the evening of 9 Nov 2026 it was on the TrueNAS share (`/data/sunnypilot/SPEEDLIMITS/derived`, NFS). There, the lists of Speed review and Sign review read every drive's files over the network and took from 2 s to over 20 s (#12). The mount unit was also the likely reason the LXC hung at boot that day. It came back with the site stopped:

```sh
rsync -a data/derived/ data/derived.local/            # with the site running, the long part
rsync -a --delete data/derived/ data/derived.local/   # with the site stopped
systemctl disable --now root-sunnypilot.speedlimits-data-derived.mount
# the .mount unit and the speedlimits-web drop-in derived-mount.conf moved out of /etc/systemd/system, then
systemctl daemon-reload
rmdir data/derived && mv data/derived.local data/derived
```

`scripts/install_derived_mount.sh` (the bind mount of the share, with a drop-in that makes `speedlimits-web` wait for it) can put it on the share again, and `--uninstall` takes it off. The copy on TrueNAS was left as it was on 9 Nov 2026 and is not kept up to date.

### The comma, from zero

The site downloads the drives over ssh (`helpers/fetch_comma_segments.sh`, the **comma** pill at the top right of every
page, which also guides these steps and has **Test now**):

1. On the comma: Settings → Network: turn on **Enable SSH**, and under **SSH Keys** enter a GitHub username. The comma
   takes the public keys of that user (`https://github.com/<user>.keys`).
2. On GitHub, as that user: add this server's public key (`~/.ssh/id_ed25519.pub`; the comma panel shows it with
   **Copy**) under Settings → SSH keys → New. Nothing to install on the comma.
3. The comma's address: its IP (Settings → Network → Advanced on the comma), set in the comma panel (**Change**;
   saved in `data/web/comma.json` and passed to the download as `--host`). Without it, the ssh host `comma` of
   `~/.ssh/config` is used (`Host comma`, `HostName <ip>`, `User comma`).
4. **Test now**: the server opens the ssh port and logs in running `true`; the pill turns green.

### Automated comma route filtering (shared fetch helper)

To run `bash helpers/fetch_comma_segments.sh` (automatic route selection and download), install the optional log reader and set schema paths:

```sh
uv pip install --python .venv/bin/python -e '.[log-reader]'
export NNSLR_OPENPILOT_ROOT=/path/to/compatible/openpilot
export NNSLR_OPENDBC_ROOT=/path/to/compatible/opendbc
bash helpers/fetch_comma_segments.sh
```

The script connects via SSH/rsync to `COMMA_HOST` (default `comma`), uses `COMMA_VIDEOS_DIR` (shared with Distill), and links accepted files into `NNSLR_DATA_ROOT/raw/routes` (default `data/raw/routes`). In automatic mode it takes every route still on the comma (dashcam or not, no date cutoff) with at least 300 seconds from first to last qlog, and downloads every segment of it, the first/last two and those with reverse gear too. The same script in a Distill checkout excludes the first/last two available segments plus any segment with reverse gear in the full rlog. A missing/unreadable qlog/rlog causes conservative exclusion. Manual route/`--segments` selections are exempt so explicit transfers remain possible. This is distinct from `nnslr sync-routes`, which does not auto-filter. For synthetic offline tests: `python -m unittest discover -s helpers -p 'test_*.py'`.

## 4. Check the installation

```sh
ffmpeg -version | head -n 1
.venv/bin/nnslr env
.venv/bin/nnslr selftest
.venv/bin/python -m pytest -q
```

## 5. Always-on review site and daily use

Install the review site (with its control panel) as the systemd service `speedlimits-web`, then use the helpers; both are described step by step in [GUIDE.md](GUIDE.md):

```sh
scripts/install_review_service.sh --dry-run
scripts/install_review_service.sh
```

The access token ends up in the `.env` of the clone (`NNSLR_REVIEW_TOKEN=...`, mode 600, ignored by git). Later updates: `helpers/update.sh`.

Reviewers from outside need a proxy with Authentik in front of the site, then `scripts/setup_guest_access.sh --public-hosts smartspeed.14bodhi.com` ([GUIDE.md](GUIDE.md), "People from outside"). Without it nobody comes in from outside. The public page is not part of this site: the cluster's administrator deploys `docs/smartspeed/` from a commit.

### The machine of the outside reviewers

The people from outside work on a machine of their own, so that nothing they reach can touch this server. Everything moves from this side: this server sends the drives given in Data › Reviewers and reads what people did, over an account of the machine that can only copy files inside one folder (rrsync, no shell). That machine holds no key towards this server.

On the machine, as root, once the copy account and its folder exist:

```sh
apt-get install -y git python3-venv rsync ffmpeg
git clone https://github.com/cristianku/sunnypilot.speedlimits /opt/speedlimits
/opt/speedlimits/scripts/install_review_box.sh --proxy-ips <proxy> --public-hosts <public address>
```

The site runs as the copy account, and the env file `/etc/speedlimits-reviews.env` holds a new access token and a new proxy secret. They are never printed: the proxy secret goes to the proxy from file to file.

On this server:

```sh
.venv/bin/nnslr review-box link reviewsync@<machine> SHA256:<its ed25519 host key>
sudo scripts/install_review_box_sync.sh
```

Then **Data › Reviewers** shows the machine, its drives and the last copy. Half an hour of driving takes about 5 GB there: its frames, its road video and the reduced logs of the pages, with the track moved away from where it was and without road names. Its raw logs and `derived/gps` stay here. The video can still show where a drive was. What comes back is checked as a stranger's work: only the drives there, only the kinds given, only under the names of the people who held them. If something changed on both sides, this server's version stays and the other is kept in `data/web/review_box/aside/`.

## Reference machine

The maintainer's setup, for comparison: Debian 13 LXC on Proxmox with two Tesla V100 32 GB (GPU 0 on bus 81:00.0, GPU 1 on C1:00.0; NVIDIA driver 580.178.04), `ffmpeg` 7.1.5, `zstd` 1.5.7, `jq` 1.7, `uv`, and a `.venv` on Python 3.13 with the `preannotate` extra and `pytest`.
