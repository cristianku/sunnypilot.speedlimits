# openpilot-nnslr-tools

**NNSLR** — *NN Vision Speed Limit Recognition* training and tooling repository.

> **Status: advisory-only.** This project builds an *independent* speed-sign
> perception source for the Swiss driving context. It is **never** part of the
> vehicle control path: it carries no target speed, no actuation request, and no
> cruise set-point. The runtime integration (Sunnypilot / openpilot) lives in a
> **separate repository** and is **out of scope here** (see `AGENTS.md`).

This is the **TRAIN_REPO**. It owns:
- the pure, hardware-independent domain contract (`speed_vision_core`);
- dataset preparation, alignment, annotation, splitting (T2/T3, synthetic-only
  for now);
- training / evaluation / export tooling (T4–T8). The first reviewed-crop
  reader baseline is implemented; detector training, evaluation and export are
  still pending.

The core/data CLI remains dependency-light and never imports PyTorch implicitly.
Training dependencies are isolated and loaded only by an explicit training or
GPU-smoke command; `nnslr train --dry-run` remains CPU-only.

---

## Repository layout

```
src/speed_vision_core/     Pure contract: types, invariants, validation, (de)serialization.
                           Stdlib-only. Imported by nothing outside this repo.
src/nnslr_tools/           CLI (`nnslr`), synthetic fixtures, and (later) data /
                           alignment / annotation / dataset tooling.
tests/core/                Contract tests (pure CPU, deterministic).
tests/cli/                 CLI entry-point tests + import-isolation guard.
src/nnslr_tools/fixtures/  Synthetic ObservationBatch documents (RFC 8259 JSON).
docs/                      Plan, decisions, audit.
requirements/              CPU dependency lock (empty while stdlib-only).
```

---

## Installation on Proxmox GPU container

For a Debian/Ubuntu Proxmox container with NVIDIA passthrough already working
(`nvidia-smi` succeeds), see:

- [docs/proxmox-gpu-container.md](docs/proxmox-gpu-container.md)

It covers cloning NNSLR, ffmpeg/ffprobe, the local qlog/rlog parser environment,
the data-root layout, and the V100-compatible PyTorch CUDA 12.6 environment.

---

## Quickstart (CPU, clean clone)

No GPU, no model, no data. The whole T1 surface runs on the standard library.

```sh
# 1) Create a clean environment (Python >= 3.11).
python3 -m venv .venv
source .venv/bin/activate

# 2) Install editable. T1 has no third-party dependencies.
pip install -e .

# 3) Check the command surface.
nnslr --help

# 4) Run the bundled synthetic self-test (validation + serialization round-trip).
nnslr selftest

# 5) Run the test suite.
pytest -q
```

<!-- [nnslr-sync] - START -->
Expected: `nnslr selftest` prints `selftest OK: 18 checks` and `pytest -q`
passes. Sync integration tests use local fixtures and real rsync, with SSH
transport replaced; they are skipped if rsync is not installed. No test
connects to a comma device.
<!-- [nnslr-sync] - END -->

> `nnslr env` reports the environment machine-readably. It marks the GPU as
> **untested** and never probes it implicitly — GPU support is only established
> by an explicitly authorized probe (plan §12.4 `check_environment`).

### Data root

<!-- [nnslr-sync] - START -->
Set **`NNSLR_DATA_ROOT`** to your private data directory (decision D4), e.g.:
<!-- [nnslr-sync] - END -->

```sh
export NNSLR_DATA_ROOT=/path/to/speed-vision-data
```

<!-- [nnslr-sync] - START -->
<!-- [route-extract] - START -->
Without this variable every command uses `data/` inside the clone
(gitignored). An explicit `--data-root` overrides the variable.
<!-- [route-extract] - END -->

The local processing commands operate on recordings already present on disk.
Only an explicit `sync-routes` invocation connects to the comma. For qlog/rlog
parsing, point the tooling at a matching local openpilot/sunnypilot checkout:
<!-- [nnslr-sync] - END -->

```sh
export NNSLR_OPENPILOT_ROOT=/path/to/openpilot
# The child interpreter needs only the optional log-reader dependencies:
export NNSLR_OPENPILOT_PYTHON=/path/to/parser-venv/bin/python
# Only if compatible car.capnp is not embedded in the checkout:
export NNSLR_OPENDBC_ROOT=/path/to/opendbc
```

<!-- [schema-reader] - START -->
Create the optional parser environment once, separately from the core CLI:

```sh
python3 -m venv /path/to/parser-venv
/path/to/parser-venv/bin/python -m pip install -r requirements/log-reader.txt
```

The reader loads `openpilot/cereal/log.capnp` (or `cereal/log.capnp`) directly
in a child process; it does not import or install the openpilot runtime.
`log-metadata`, `align-route` and `find-candidates` also accept
`--openpilot-root`, `--openpilot-python` and `--opendbc-root` overrides.
Use compatible `log.capnp` and `car.capnp` versions: a schema mismatch or
truncated log fails the command instead of returning partial metadata.
<!-- [schema-reader] - END -->

---

## What is implemented (T1)

| Subcommand        | Purpose                                                                 |
| ----------------- | ----------------------------------------------------------------------- |
| `nnslr version`   | Print package and core versions.                                        |
| `nnslr env`       | Machine-readable environment report (no implicit GPU probe).            |
| `nnslr validate-batch FILE --now-mono-ns N --expected-session-id S` | Validate an `ObservationBatch` JSON against the §7 contract; stable reason codes; exit `0` accepted / `1` rejected / `2` missing file. |
| `nnslr selftest`  | Run the bundled synthetic fixtures end-to-end (CPU quickstart).         |

The §7 contract (`speed_vision_core.types`) enforces, among other things:
- `value_kph: int | None` — **0 is never a speed limit**; absence stays absent;
  `unknown` / `unreadable` / `not_applicable` / `unavailable` remain distinct.
- Strict numeric deserialization — no silent coercion (`"50"` → `50`, `True` →
  `1`, `50.9` → `50` are all rejected with a deterministic reason code).
- Monotonic timestamps are comparable only within the same session/clock domain.
- A detection must reference the **exact** batch frame (identity + timing +
  native fields), not just the frame key.
- Linked supplementary panels obey the same geometry rules as the primary bbox.
- Deterministic, pure validation reporting a finite set of documented reason codes.
- **No type carries a target speed or actuation request.**

---

<!-- [nnslr-sync] - START -->
## Copy a route from the comma

With an existing SSH alias `comma`, only the route is required:

```sh
nnslr sync-routes --route "$ROUTE"
```

Defaults:

| Setting | Default |
| --- | --- |
| SSH host | `NNSLR_COMMA_HOST`, otherwise `comma` |
| Segments | All available segments of the specified route only |
| Camera | `narrow`: `fcamera.hevc` (front road camera) |
| Log | Full `rlog`, preferring `rlog.zst`, then `rlog`, then `rlog.bz2` |
| Data root | `NNSLR_DATA_ROOT`, otherwise `data/` inside the clone |
| Destination | `<data-root>/raw/routes/<route>/<segment>/` |

Override only what you need:

```sh
# Inspect the remote inventory without downloading or writing local files.
nnslr sync-routes --route "$ROUTE" --dry-run

# Use a different SSH destination, segment selection, or both road cameras.
nnslr sync-routes --route "$ROUTE" --host comma@192.0.2.10 --segments 0-3
nnslr sync-routes --route "$ROUTE" --camera both --data-root /path/to/speed-vision-data

# Set preferences once in your shell configuration instead of repeating options.
export NNSLR_COMMA_HOST=comma
export NNSLR_DATA_ROOT=/path/to/speed-vision-data
```

`--camera wide` selects `ecamera.hevc`, the other forward-facing, wider road
camera. Cabin video and the reduced `qcamera.ts` are not copied. The full log
is retained for frame/capture-time alignment and map-transition candidate
discovery; a `qlog` is not silently substituted for it.

Requires `ssh` and `rsync` locally and `rsync` on the comma. Existing SSH keys,
host verification and `~/.ssh/config` are used; no credentials are stored by
NNSLR. SSH runs in batch mode, so authentication must already work without an
interactive password prompt. No device software is installed or modified.

Rerun the same command to resume: interrupted files stay in `.nnslr-partial/`
until rsync completes them. Existing destination files are checked by rsync
content checksum and reused if identical; different files are reported as
conflicts and never overwritten. Nothing is deleted from either dataset.

Progress is written to stderr. The JSON report on stdout lists selected files,
local SHA-256 hashes, copy/resume/reuse status and missing files/segments. Exit
`0` means success, `1` means missing data or a file transfer/conflict failure,
and `2` means invalid input or a preflight/SSH inventory failure. Available
files can still be copied when other requested data is missing; after a file
transfer failure, remaining entries stay `planned` for the next invocation.
Dry-run contacts the host for inventory, but does not verify existing local
content or transfer files.
<!-- [nnslr-sync] - END -->

<!-- [route-extract] - START -->
## Extract images from a complete route

Once a route has been downloaded, one command processes every available local
segment automatically. No shell loop or segment list is needed:

```sh
nnslr extract-frames --route "$ROUTE"
```

Defaults are `fcamera.hevc` (front narrow road camera), one PNG per second at
the original resolution, and `NNSLR_DATA_ROOT` or `data/` inside the clone as the data
root. Images and a `frames.jsonl` manifest are written for each segment:

```text
<data-root>/derived/frames/<route>/<segment>/frame_00000000.png
<data-root>/derived/frames/<route>/<segment>/frames.jsonl
```

The usual download-to-images workflow is therefore:

```sh
nnslr sync-routes --route "$ROUTE"
nnslr extract-frames --route "$ROUTE"
```

Only specify options when changing a default:

```sh
nnslr extract-frames --route "$ROUTE" --fps 5
nnslr extract-frames --route "$ROUTE" --all-frames
nnslr extract-frames --route "$ROUTE" --data-root /path/to/speed-vision-data
nnslr extract-frames --route "$ROUTE" --output /path/to/images
```

`--output` is the parent of the per-segment directories. `--manifest FILE`
optionally adds one combined JSONL in addition to the per-segment manifests.
`--start` and `--end`, when supplied, apply within **each segment**, not to a
route-wide clock. Segment discovery follows the local folders created by
`sync-routes`; missing indices are not renumbered. A missing `fcamera.hevc`
or pre-existing frame output is reported before extraction starts. Existing
images are not overwritten unless `--overwrite` is explicitly supplied.

This step extracts images for review and annotation. It does not recognize
signs. Capture timestamps are attached only when a matching, hash-bound
video/log alignment is available; otherwise timing remains explicitly unresolved. Single-video processing remains
available with `--video FILE --output DIRECTORY`.
<!-- [route-extract] - END -->

<!-- [preannotation] - START -->
## Pretrained proposals and visual review (CPU)

Install the optional inference dependencies once, from this repository in the
active nnslr virtual environment:

```sh
python -m pip install '.[preannotate]'
```

After extracting a route, these commands process all its frames and open the
review server. No environment variables, output directories or segment loop
are required; the default data root is `data/` inside the clone.

```sh
nnslr preannotate --route 00000089--0ac1c0fdec
nnslr review --route 00000089--0ac1c0fdec
```

Open `http://127.0.0.1:8765` in the browser. When running in the container, start
review from your Mac through an SSH tunnel instead:

```sh
ssh -L 8765:127.0.0.1:8765 openpilot-nnslr 'bash -lic "nnslr review --route 00000089--0ac1c0fdec"'
```

Then open the same local browser address. The server binds to loopback by
default and serves only the report and its image previews. Stop with Ctrl+C.

The first preannotation run downloads checksum-pinned ONNX weights under
`models/preannotation/` inside the data root; subsequent runs use that cache.
Inference uses CPU only in an isolated subprocess. Every run preserves the
previous results and writes a new directory under `derived/preannotations/`.
The review command selects the latest completed run automatically.

The page shows native-resolution boxes and proposed values. You can change
the type/value, delete false positives, draw missed boxes, and confirm each
frame. Edits persist in that browser. **Download the reviewed JSONL** to keep
them outside browser storage. Only explicitly confirmed frames are exported;
a frame with no proposals is never automatically a negative training example.
Exports retain the original proposals and remain `training_ready: false` until
`import-annotations` validates them. Missing capture timing remains explicitly
unknown: reviewed images may support classification, but not passage/latency claims.
The review page records a reviewer name/pseudonym and confirmation time; reopening
`nnslr review` updates older report pages without rerunning the model.

### What the models recognize

- Vertical signs: a pretrained 82-class Vietnam sign detector, applied to the
  whole frame plus overlapping native-resolution tiles. Swiss accuracy is
  **not established**. A numeric OCR reading is not used as a vertical sign.
- Asphalt numbers: PP-OCRv4 reads the lower half with 3x vertical expansion.
  A separate Cityscapes SegFormer model identifies the road surface. At least
  95% of the number box **including surrounding context** must lie on pixels
  classified as road with score at least 0.9. Missing/uncertain road evidence
  rejects the reading. Wall, advertising, vehicle and sidewalk regions are
  excluded when segmentation identifies them correctly.
- An accepted asphalt number is a `road_marking_candidate`, not an established
  speed limit. OCR score is not a probability of being a speed sign. Surface
  segmentation can still fail; arbitrary numbers painted on roads, adjacent
  roads and ambiguous markings require human review. This is dataset
  preparation, with no vehicle-control or runtime integration.

### Model provenance and usage constraints

Model repositories, revisions, checksums, preprocessing and provenance are
recorded in each run and each frame. External weights and private images are
not committed to this repository.

| Component | Source | License information |
| --- | --- | --- |
| Sign detector | [star092304/traffic-sign-detection-vietnam-yolo](https://huggingface.co/star092304/traffic-sign-detection-vietnam-yolo) | Model card says Apache-2.0; embedded ONNX metadata says AGPL-3.0. This discrepancy is recorded, not resolved by this tool. |
| Road segmentation | [Xenova SegFormer ONNX](https://huggingface.co/Xenova/segformer-b0-finetuned-cityscapes-1024-1024), from NVIDIA SegFormer | [NVIDIA license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) limits use to non-commercial research/evaluation. |
| OCR | [RapidOCR 3.4.0](https://pypi.org/project/rapidocr/3.4.0/) bundled PP-OCRv4 | Apache-2.0; bundled model hashes are checked before use. |

These are research/evaluation aids for producing reviewed annotations, not a
validated model bundle for deployment. The pretrained components have their
own terms; the repository's MIT license does not replace them.
<!-- [preannotation] - END -->

## T2 local comma video/log pipeline

These commands operate only on files already copied under local storage. They
do **not** SSH to, download from, or modify a comma device.

<!-- [route-extract] - START -->
| Command | Purpose |
| --- | --- |
| `nnslr route-manifest` | Inventory local route segments, hashes, missing/partial segments. |
| `nnslr inspect-manifest` | Summarize a route manifest. |
| `nnslr video-probe` | Probe fcamera/ecamera/qcamera with ffprobe. |
| `nnslr extract-frames` | Use `--route ID` for all local segments automatically, or `--video FILE --output DIR` for one video; defaults to 1 fps, with `--all-frames` available. |
| `nnslr log-metadata` | Read local qlog/rlog(.zst) camera/map metadata through a matching local openpilot checkout. |
| `nnslr align-route` | Join ffprobe presentation order to openpilot `EncodeIndex.segmentId`; preserve unresolved frames explicitly. |
| `nnslr alignment-report` | Print deterministic alignment diagnostics. |
| `nnslr find-candidates` | Use map speed transitions as search hints only, never ground truth. |
| `nnslr preannotate` | Propose sign boxes and road-surface-gated asphalt numbers on extracted route frames, using optional CPU models. |
| `nnslr review` | Review the latest route proposals in a local browser and export confirmed annotations. |
| `nnslr import-annotations` | Validate and import confirmed review JSONL; defaults to the annotation inbox and canonical objects file. |
| `nnslr validate-dataset` | Check schema, images, hashes, boxes, review/provenance and timing; optionally verify frozen splits. |
| `nnslr build-splits` | Freeze route/site/encounter/image-connected partitions with deterministic defaults and leakage checks. |
| `nnslr make-clips` | Build local review clips around projected candidate times. |
<!-- [route-extract] - END -->

The production alignment path deliberately distinguishes:

```text
video PTS / presentation order
        !=
camera frameId
        !=
EncodeIndex.segmentIdEncode (encode order)
        !=
camera capture timestamp
```

<!-- [t2-validation] - START -->
The adapter uses `EncodeIndex.segmentId` as its presentation-order join key
only when the complete index domain matches decoded positions `0..N-1`.
Real logger startup can leave a nonzero first `segmentId`; truncated or missing
data can also break that domain. Those segments remain unresolved with
`presentation_index_domain_mismatch` (CLI exit 1); the tool never silently
subtracts an offset. Duplicate indices and missing capture timestamps also
remain unresolved. The synthetic `manifest.align_frames()` helper is fixture-only.

Raw HEVC often has no PTS. The frame/alignment output exposes a separate
`media_time_s` / `video_media_time_s` and provenance
`raw_hevc_frame_duration` when every decoded frame has a valid duration.
PTS stays null; camera capture time still comes only from the log. Other
containers, mixed timestamps or missing durations do not use that fallback.

`make-clips` automatically reencodes raw `.hevc`/`.h265` to H.264 MP4 and
performs seeking after opening the input; packet-copy seeks produced empty
files on real recordings. `extract-frames --start` also decodes before seeking.
These operations can take longer because they decode from the beginning.
Empty output is an error. For timestamped containers the default clip mode is
packet copy (keyframe boundaries); use `--reencode` for a precise cut.

See [the real-route validation report](docs/T2_REAL_ROUTE_VALIDATION.md) for
verified coverage and outstanding T2 checks.
<!-- [t2-validation] - END -->

Typical local flow:

```sh
export NNSLR_DATA_ROOT=/path/to/speed-vision-data
export NNSLR_OPENPILOT_ROOT=/path/to/openpilot

nnslr route-manifest raw/routes/000001a3--c20ba54385 \
  --output manifests/routes.jsonl

nnslr video-probe \
  "$NNSLR_DATA_ROOT/raw/routes/000001a3--c20ba54385/0/fcamera.hevc"

nnslr align-route \
  --video "$NNSLR_DATA_ROOT/raw/routes/000001a3--c20ba54385/0/fcamera.hevc" \
  --log "$NNSLR_DATA_ROOT/raw/routes/000001a3--c20ba54385/0/rlog.zst" \
  --stream narrow_road \
  --segment-num 0 \
  --output "$NNSLR_DATA_ROOT/derived/alignment/segment-0.jsonl"

nnslr find-candidates \
  "$NNSLR_DATA_ROOT/raw/routes/000001a3--c20ba54385/0/rlog.zst" \
  --alignment "$NNSLR_DATA_ROOT/derived/alignment/segment-0.jsonl" \
  --output "$NNSLR_DATA_ROOT/manifests/candidates.jsonl"

nnslr make-clips \
  --video "$NNSLR_DATA_ROOT/raw/routes/000001a3--c20ba54385/0/fcamera.hevc" \
  --candidates "$NNSLR_DATA_ROOT/manifests/candidates.jsonl" \
  --output "$NNSLR_DATA_ROOT/derived/clips"

nnslr extract-frames \
  --video "$NNSLR_DATA_ROOT/raw/routes/000001a3--c20ba54385/0/fcamera.hevc" \
  --output "$NNSLR_DATA_ROOT/derived/frames/discovery"
```

---

<!-- [route-extract] - START -->
For the complete option list of any implemented command, use its built-in help:

```sh
nnslr --help
nnslr sync-routes --help
nnslr extract-frames --help
```
<!-- [route-extract] - END -->

<!-- [reviewed-dataset] - START -->
## Import reviewed annotations and build splits

After confirming frames and downloading the JSONL from the review page:

```sh
nnslr import-annotations annotations-RUN_ID.jsonl
nnslr validate-dataset
nnslr build-splits
```

No environment variables or output folder arguments are required. All three
commands use the configured data root, defaulting to `data/` inside the clone.
The default canonical dataset is `annotations/objects.jsonl`. Alternatively,
put review files into `annotations/inbox/` under that root and run
`nnslr import-annotations` with no arguments. Multiple input files can be
imported together; reimporting duplicate identities is rejected.

`build-splits` creates an immutable content-addressed JSON under `splits/`, with
`latest.json` pointing to it. Seed defaults to `0`. It requires at least three
independent connected groups and populates train, validation and test (policy
`train-validation-test-v2`): validation and test each get whole groups adding up
to about 15% of the active rows, everything else is train, and reviewed
negatives stay with their group. The test prefers the groups with the most
frames confirmed by a person; `--test-route ROUTE` (repeatable) pins routes to
test. A route is never split by frame; known shared sites, encounters and
duplicate images also stay together. With only two routes, this command reports
insufficient groups instead of manufacturing an apparently independent test
set. Unknown repeated physical sites remain a reported grouping limitation.
Split files of the earlier five-partition policy (route-held-out, hard-negative)
still validate.

An invalid dataset returns a JSON error report and nonzero exit code. For an
existing split, `nnslr validate-dataset --splits /path/to/split.json` also checks
membership, dataset identity and leakage. See [the canonical dataset contract](docs/DATASET.md).

Extraction now records source decoded index, native geometry, image/video hashes,
media time and provenance. Sampling picks the first available source frame at
each requested interval and never duplicates it. `output_index` remains a
separate PNG counter. This may retain one more frame at the end of a segment
than the previous ffmpeg FPS filter. `--overwrite` stages and validates the new
images before replacing old images and removing stale PNGs.

`extract-frames --alignment FILE` attaches capture provenance for a single
video; with `--route`, pass a directory containing `<segment>.jsonl` files.
Route mode also discovers these files at `derived/alignment/ROUTE_ID/` by
default. Generate alignments with the current `align-route` command: older
files without source hashes are rejected rather than trusted implicitly.
<!-- [reviewed-dataset] - END -->

## Detector + reader baseline training (T4)

The first complete perception-training surface is now available:

```text
full frame
   ↓
SSDLite320 + MobileNetV3 detector
   ↓
reviewed sign crop
   ↓
MobileNetV3-Small reader
```

The detector is currently a single-class vertical speed-sign detector. Road
markings remain separate, and `other_sign` annotations are not silently treated
as whole-frame negatives. These are baselines and do not produce an on-device
model bundle yet.

Validate the exact dataset and frozen split without importing PyTorch or touching
a GPU:

```sh
nnslr train --task reader --dry-run
nnslr train --task detector --dry-run
```

By default this uses the gold human-reviewed dataset and follows the
hash-verified `splits/latest.json` pointer. Model-reviewed data is isolated:

```sh
nnslr train --dry-run --dataset-kind training-candidate
```

Before the first V100 run, explicitly verify the CUDA environment, on each card
(the server has two V100s: `0` on bus 81:00.0, `1` on C1:00.0, as nvidia-smi
numbers them; `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes CUDA count them the same way):

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID nnslr env --gpu-smoke --gpu-device 0
CUDA_DEVICE_ORDER=PCI_BUS_ID nnslr env --gpu-smoke --gpu-device 1
```

Unlike plain `nnslr env`, `--gpu-smoke` intentionally imports PyTorch,
allocates GPU memory, checks that the installed binary contains the GPU's CUDA
architecture, and executes a real FP16 forward/backward operation. For a Tesla
V100 the capability must be `(7, 0)` / `sm_70`.

A real training run requires a new output directory:

```sh
nnslr train --task detector \
  --output "$NNSLR_DATA_ROOT/runs/detector-baseline-001" \
  --device cuda --epochs 20

nnslr train --task reader \
  --output "$NNSLR_DATA_ROOT/runs/reader-baseline-001" \
  --device cuda --epochs 20
```

Each run writes an immutable training plan, best checkpoint and JSON result,
including dataset/split hashes and GPU identity.

The detector resizes the full frame inside the model. `--input-size WIDTHxHEIGHT`
(default `320x320`, torchvision's SSDLite320) sets that size, for example
`--input-size 640x360`; `--anchor-min-ratio` sets the smallest default box as a
fraction of the image side (default `0.2`, torchvision's: at least 155 x 88
pixels of a 1344 x 760 frame at any input size, since the default boxes scale
with the image and a larger input only puts them on a finer grid). Neither
changes a weight. Both are recorded in `detector-best.pt`, `training-state.pt`
and `result.json`, and evaluation, ONNX export (its contract's
`model_internal_resize` and `default_boxes`) and replay rebuild the detector
from the checkpoint; checkpoints without them are 320 x 320 with `0.2`.
Training frames get a photometric augmentation (brightness leaning to darker,
contrast down to half for mist and spray, gamma, saturation, a light colour cast
and, on some frames, a light blur for rain on the windscreen; never a flip: the
side of the road matters); `--no-augment` turns it off, and `result.json` records the
ranges under `augmentation`.

torchvision's SSD takes its negative boxes only inside frames with signs, so a
frame without signs gives no loss. Each detector epoch therefore uses every frame
with signs and a share of the frames without (`--negative-frame-fraction`,
default `0.1`, drawn again each epoch; `1.0` uses every frame as before); the
validation uses the same share as one fixed draw, so its epochs compare.
`result.json` records the counts under `frames`. The detector was the cutter of
the pipeline until 30 October 2026 (SignNet is now, see below), so its best epoch (`detector-best.pt`) and early stopping follow
the validation recall at score 0.1, ties to fewer false boxes per frame
(`--select-by recall`, the default; `--select-by loss` keeps the lowest
validation loss). Every epoch line also prints `val_recall`, and `result.json`
holds `best_epoch`, `best_validation_recall`, the validation loss of that epoch
as `best_validation_loss`, and `selection`.

What the detector learns to cut is `--cutter-families`: `all` (the default for a
new detector) is the speed-sign families and the other traffic signs
(`other_sign`), so that the classifier gets them and says what they are;
`speed` is the speed signs only, as every detector before the option (a
checkpoint, training state or report without `families` is `speed`, and a run
saved before it resumes so). Road markings and boxes marked "not a sign" are
never boxes of the cutter. The detector keeps one class (still named
`speed_sign` in its checkpoint); the plan, the checkpoint, the training state
and `result.json` record `families`, the plan and `result.json` the boxes per
partition and family (`boxes_by_family`). `nnslr evaluate --task detector`
measures a checkpoint on the families it learnt and adds `recall_by_family`
(`speed`, `other_sign`) overall and at every threshold, so the speed-sign recall
of an `all` cutter compares with the older ones; the size comparison shows it
as the `speed` column. Before the other-sign rule of 24 October 2026 an AI
reviewer left the other signs without a box: in those frames they are
background for the cutter.

`--mode resume` needs exactly the settings the run started with;
`scripts/run_v100_baseline.sh` and `scripts/run_detector_sizes.sh` read them
from the run's `training-state.pt` (`nnslr_tools.training.resume_arguments_file`),
`--pretrained` of a start from ImageNet included.

To compare input sizes of the old Detector (kept to compare: since 30 October
2026 a Detector cannot be put in use), `scripts/run_detector_sizes.sh NAME [SIZES...]`
(default `320x320 640x360 960x544`) trains one detector per size with the same
settings into `$NNSLR_DATA_ROOT/runs/NAME/detector-sizes/<size>/`, evaluates
each on test and validation and prints the comparison, saved as
`comparison.txt`. Its smallest default box is `0.02` of the image side
(`NNSLR_DETECTOR_ANCHOR_MIN_RATIO`), its share of frames without signs `0.1`
(`NNSLR_DETECTOR_NEGATIVE_FRACTION`), and the best epoch follows the validation
recall; its boxes are the families `NNSLR_CUTTER_FAMILIES` (`all` by default, or
`speed`). It refuses a busy GPU (the card of `NNSLR_GPU_INDEX`) unless
`NNSLR_ALLOW_BUSY_GPU=1`, like the baseline pipeline, and finishes an
interrupted comparison when run again with the same name.

### The cutter: SignNet (since 30 October 2026)

SignNet (`nnslr_tools.signnet`, `nnslr train-signnet`) is the one cutter, the
same network the comma runs (`sunnypilot/speed_limits/vision.py`). It reads the
band at the top of the frame: the frame scaled to the band width and its top
rows kept (1344 x 416 and 672 x 208 keep the top 416 of 760 rows, which hold 97%
of the reviewed signs; the car's 896 x 288 keeps the top 432), RGB 0..1. Plain 3
x 3 convolutions (`--widths`, the channels at strides 2, 4, 8 and 16) give a
heat map of sign centres at `--stride` 8 or 4 with each sign's size and offset,
no NMS: the decode keeps the 3 x 3 local maxima, best first. `--fuse-kernel 1`
joins the strides with 1 x 1 convolutions. It takes the detector's plan
(`prepare_detector_training`: the same frames, boxes, families, split and
`--dataset-kind`); a switched-off variable display is an ignore box, neither
learnt nor counted.

Switzerland has few end-of-limit signs: on 30 October 2026 the dataset had 154
train crops of them, from a handful of physical signs, and both networks learnt
those by heart (of 24 in validation and buffer, SignNet boxed 9 for the car and
the classifier read 5 sure enough). So both trainings draw end-of-limit signs
over their own speed signs (`nnslr_tools.synthetic_signs`): each time a train
frame is read, every speed sign of 9 pixels or more (`maximum_speed`, a lit
`variable_display`) becomes an end of limit with probability 0.25, in its place,
at its size, with the colours of its own face and digits and as blurred as it
(a round sign the Swiss sign with thin stripes, 15% of them the end of all
restrictions; a lit display the white LED ring, number and slash on its black
panel, or a round sign when its box has no dark panel). SignNet keeps the box;
the classifier labels the crop `cancellation` (see below). The flip and the
light of the band come after. `result.json` records it under
`synthetic_end_of_limit` (with `paintable_train_boxes`).

The defaults are the car's setup, the one **Train the cutter**
(`scripts/run_signnet.sh NAME`, about 80 minutes on the server's GPU) runs:
`--band-width 896 --stride 8 --fuse-kernel 1 --widths 16,24,32,48`, 40 epochs,
batch 16, learning rate 0.003, `--families all` (`NNSLR_CUTTER_FAMILIES=speed`
in the script for the speed signs only).

```sh
CUDA_VISIBLE_DEVICES= .venv-gpu/bin/nnslr train-signnet --run-dir /tmp/signnet-dry --dry-run  # the plan only
bash scripts/run_signnet.sh <YYYY-MM-DD_HHMM-cutter>
CUDA_DEVICE_ORDER=PCI_BUS_ID NNSLR_GPU_INDEX=1 bash scripts/run_signnet.sh <YYYY-MM-DD_HHMM-cutter>  # on the second card
```

The dry run prints one JSON line: `counts_by_split`, `boxes_by_split`,
`families`, `split_sha256`, `trainable`; it exits 1 when the plan has nothing to
train or validate on (a training refuses such a plan too), so the script stops
at its step 1/3, before the GPU. Each epoch is measured on validation, and the
epoch with the best validation recall among the best 5 boxes a frame, at any
score, is kept as `<run>/signnet-best.pt` (the weights with `widths`, `stride`,
`fuse_kernel`, `band`, `epoch`, `dataset_sha256`, `split_sha256` and `families`;
a checkpoint saved before 30 October 2026 has the last three only in the
`result.json` beside it). Its epoch line reads `signnet epoch N/M train_loss L
val_recall R val_false_per_frame F Ns`, the validation among the best 5 boxes at
any score. At the end the best epoch is measured on validation and test, and
`<run>/result.json` (`task` `signnet`) holds `band`, `stride`, `fuse_kernel`,
`widths`, `parameters`, `best_epoch`, `history`, `families`, `dataset_sha256`,
`split_sha256`, `counts_by_split`, `boxes_by_split` and, for `validation` and
`test`, `car` (what the car reads: the best 5 boxes of score 0.3 or more),
`by_score_threshold` (0.05, 0.1, 0.2, 0.3 and 0.5) and `by_box_budget` (the best
3, 5 and 10 boxes of a frame at any score), each with recall, false boxes per
frame, `recall_by_width` and `recall_by_family`. They are scored on the boxes of
the whole frame, so a sign below the band counts as missed. There is no resume:
a run without `result.json` is unfinished, and `scripts/run_signnet.sh` trains
such a NAME again from scratch, after moving its old `signnet.log` aside so the
old epochs do not mix with the new ones (it refuses a finished one).

The Models tab puts in use (**Use this one**) only a finished SignNet with band
896 x 288 and stride 8 (`signnet.fits_the_car`; `CAR_BAND`, `CAR_STRIDE` mirror
the car's `vision.py`); its id is `cutter:<run>/signnet`. The SignNet in use
gives the car's best 5 boxes a frame, from score 0.1 (the car keeps those of 0.3
or more), to Prepare's Our networks, `mine-negatives` and `evaluate-chain`: the
heat map decoded best first, at most 5 a frame (`CAR_MAX_BOXES`), scaled to
frame pixels by frame width / band width, with no size filter (the car has
none), above the score floor of each caller (0.1 for Prepare and mining); the
car keeps those scoring 0.3 or more (`CAR_MIN_SCORE`), so only `evaluate-chain`
judged at 0.3 is exactly the car. A reader of what the car finds takes `car`
from the result; for an older `result.json` without it,
`by_score_threshold['0.3']` stands in (about 0.35 boxes a frame there, so the
cap of 5 almost never bites). **Publish** (to the car) and **Try on the comma now**
refuse a SignNet that does not fit the car or has not finished. A sign below the
band (a road marking, a low sign close by) is never found. `nnslr export-onnx
--task signnet` writes the ONNX for the car, with one output `out`.

### The classifier: "not a sign" and the chain end to end

The reader has two class schemes (`--classes`, recorded in the plan, the
training state, the checkpoint and `result.json`; a checkpoint without one is
`v1`). `v1` is the reader as before. `v2` is the classifier of the pipeline
(cutter -> classifier -> decider, [NEURAL_NETWORK.md](NEURAL_NETWORK.md)): the
same labels, the kind of an other sign (`other_sign:give_way` when the review
gives it; a kind with fewer than 3 training crops is folded into `other_sign`
everywhere, `folded_labels`) and `not_a_sign`, learnt from:

- the proposals a reviewer removed (`review_state` `rejected`) that overlap no
  kept box of their frame (IoU < 0.1: a duplicate on a sign, or a box drawn
  again, still frames a sign), in the partition of their frame;
- the boxes a reviewer kept and marked "not a sign";
- the cutter's false boxes, mined beforehand on the GPU:

```sh
.venv/bin/nnslr mine-negatives --dry-run          # CPU: frames, sign crops, caps, the dataset's own negatives
.venv-gpu/bin/nnslr mine-negatives --cutter active --output data/runs/<run>/negatives.jsonl
```

`mine-negatives` runs the cutter (`active`: the cutter in use of the Models
tab, `models/active.json`; or a `signnet-best.pt`, or an old `detector-best.pt`)
on every reviewed frame of train, validation and test (a SignNet gives the car's
boxes: at most 5 a frame, no size filter) and keeps its boxes of score >= 0.1
(`--score-threshold`) that overlap no box a reviewer saw there (IoU < 0.1 with
every kept box and every removed proposal), at most 3 per frame, the highest
scores (`--per-frame`), and at most 3 x the sign crops of each partition
(`--total-factor`), drawn with `--seed` from the frames a person reviewed first,
then those an AI reviewed after the other-sign rule of 24 October 2026
(`OTHER_SIGN_RULE_AT`), then the older AI reviews, which may hold other signs
without a box. Each partition takes its negatives from its own frames only, the
buffer none. The file starts with a header (dataset and split, the cutter and
its sha256, the settings, the counts per partition and per reviewer tier); the
training refuses it on another dataset or split. Its `cutter` block has the id
(`cutter:<run>/signnet` for a SignNet), `kind` (`signnet` or `detector`) and
`input_size`; for a SignNet also `band` and `stride`, and `input_size` is the
band.

```sh
.venv/bin/nnslr train --task reader --classes v2 --dry-run
.venv-gpu/bin/nnslr train --task reader --classes v2 --negatives data/runs/<run>/negatives.jsonl \
  --pretrained --epochs 30 --output data/runs/<run>/classifier
```

The `v2` training crops are cut with a jittered box (each side scaled 0.85–1.2,
the centre moved up to 10% of the box, `CLASSIFIER_BOX_JITTER`: the cutter's
boxes are never exact) and photometrically augmented like the detector's
frames (`CLASSIFIER_AUGMENTATION`); a quarter of the speed-sign crops, drawn
again every epoch, carry an end-of-limit sign drawn over the sign and are
labelled `cancellation` (`synthetic_signs`, see SignNet above: drawn on the
crop cache's window before the crop is cut, when the plan has the class
`cancellation`; the log says `classifier draws an end of limit over 25% of the
N speed signs of train each epoch`, and `result.json` records it under
`augmentation.synthetic_end_of_limit`); `--no-augment`
trains without all three. The `not_a_sign` crops together weigh at most as much as
all the sign crops in the loss (`loss_weights`). The best epoch is the one with
the highest validation balanced accuracy: the mean of the sign crops read right
and the `not_a_sign` crops rejected. Its epoch lines read
`classifier epoch N/M ... val_sign_accuracy X val_rejection Y Ns`.

`nnslr evaluate --task reader --classes v2 --negatives <file>` measures it
alone: besides the usual numbers on every crop, `signs` (the sign crops only,
with the crops a person confirmed, the signs it called not a sign, and
`family_accuracy`, comparable with the old reader) and `not_a_sign` (the share
rejected, by source and on the frames a person reviewed).

`nnslr evaluate-chain --cutter active --classifier <reader-best.pt> --partition
test --output <run>/evaluation/chain-test.json` measures the two networks end
to end: the cutter on every reviewed frame of the partition, each of its boxes
cut from the frame at full resolution and classified; at cutter scores 0.1,
0.2 and 0.3 with a SignNet (judged at 0.3, the car's floor; the report records
it as `main_score_threshold`), or 0.05, 0.1 and 0.25 with an old Detector
(judged at 0.1), the speed signs found AND read right (overall, by the width of the
reviewed box, on the signs a person confirmed), the cutter's own share found,
the boxes and false boxes per frame, the false boxes per frame that survive
(answered anything but `not_a_sign`) and the confusion of every box passed on.
An old reader (`v1`) rejects nothing: every false box survives.

`scripts/run_classifier.sh NAME` (also the **Train the classifier** button of
the Train tab) does all of it on the GPU into `data/runs/NAME/`: checks, mining
with the cutter in use (the SignNet in use: the car's best 5 boxes a frame, from score 0.1), training from ImageNet, the classifier alone on test and
validation, the chain on test and validation, the chain on test with the old
reader (`NNSLR_OLD_READER_RUN`, default `2026-10-24_1452`) and a comparison
saved as `comparison.txt`. It refuses a busy GPU (the card of
`NNSLR_GPU_INDEX`; from the panel a free one) unless `NNSLR_ALLOW_BUSY_GPU=1`
and finishes an interrupted run when run again with the same name.

For the complete guarded pipeline on the training host (`NNSLR_GPU_INDEX` is
nvidia-smi's number of the card, `0` or `1`; the script exports
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, so CUDA counts the cards the same way, and the
panel gives each run a free card):

```sh
export NNSLR_GPU_INDEX=0
bash scripts/run_v100_baseline.sh baseline-001
```

The script refuses an already-busy GPU by default, then runs CPU-only
population checks, CUDA smoke, detector+reader training, held-out evaluation,
hard-example mining, ONNX parity export, immutable bundle packaging and final
bundle verification. It never kills competing GPU processes and never deploys
anything to Sunnypilot or a comma device.

### wandb

Training logs to [wandb](https://wandb.ai), the same way openpilot.distill
does. `scripts/setup_v100_training_env.sh` installs it (the `training` extra of
`pyproject.toml`); the core, the CLI and the CPU tests never import it.

Log in once on the training host; the key is saved in `~/.netrc`:

```sh
.venv-gpu/bin/wandb login
```

- **Project:** `sunnypilot.speedlimits`, or `WANDB_PROJECT=<name>`.
- **Grouping:** detector and reader of one pipeline run share the wandb group
  `<run id>` (`job_type` `detector` / `reader`, run names
  `<run id>-detector` / `<run id>-reader`). `run_v100_baseline.sh` exports
  `NNSLR_RUN_ID`; a manual `nnslr train --output <data-root>/runs/<run id>/<task>`
  takes the run id from the folder, any other `--output` folder is its own run id.
- **Files:** `<output>/wandb/` next to the checkpoint and `result.json`.
- **Logged:** per step the training loss (detector: plus its
  `bbox_regression` / `classification` parts; reader: plus batch accuracy),
  learning rate and step time; per epoch the epoch loss, validation loss (reader:
  plus train/validation accuracy) and the best value so far. On the first epoch,
  every 5 epochs and on the last one: a few validation frames with predicted
  (score >= 0.25) and reviewed boxes, and the first validation crops with
  `true <label> / pred <label>` in the caption.
- **Offline fallback:** with no `WANDB_MODE`, no `WANDB_API_KEY` and no
  `api.wandb.ai` in `~/.netrc` the run goes offline (one message, training is
  never blocked). Upload it later with
  `wandb sync <output>/wandb/offline-run-*`.
- **Off:** `WANDB_MODE=disabled` turns wandb off completely. If wandb is not
  installed, training prints one warning and runs without it.

## Offline evaluation and hard-example mining (T5/T6)

Inspect the frozen evaluation population without importing PyTorch:

```sh
nnslr evaluate --task detector --partition test --dry-run
nnslr evaluate --task reader --partition test --dry-run
```

After training:

```sh
nnslr evaluate --task detector \
  --checkpoint "$NNSLR_DATA_ROOT/runs/detector-baseline-001/detector-best.pt" \
  --partition test \
  --output "$NNSLR_DATA_ROOT/evaluation/detector-test.json"

nnslr evaluate --task reader \
  --checkpoint "$NNSLR_DATA_ROOT/runs/reader-baseline-001/reader-best.pt" \
  --partition test \
  --output "$NNSLR_DATA_ROOT/evaluation/reader-test.json"
```

Reader evaluation records accuracy, per-class support and confusion plus
sample-level mistakes. Detector evaluation records IoU-based precision, recall,
F1 and false positives per image while retaining the individual error evidence.
Most labels are AI reviews, so both also report `counts_by_label_basis` and a
`human_reviewed` object with the same metrics restricted to the labels a person
confirmed (`provenance.label_basis` `human_review`), or `null` when the
partition has none; the top-level metrics stay those of the whole partition.
`--dry-run` prints `counts_by_label_basis` and `human_reviewed_sample_count`.
The detector was the cutter of the pipeline until 30 October 2026 (cutter, then
the reader as classifier, then whom the sign applies to), so its report also measures recall
first: `recall_by_width` (the reviewed boxes found by their width in frame
pixels: `lt12`, `12-20`, `20-40`, `ge40`, the buckets of the site's Evaluate
tab), `predictions_per_image` (the boxes passed on to the classifier) and, in
`by_score_threshold`, every measure again at 0.05, 0.1, 0.3, 0.5 and the
report's own `--score-threshold`, each with its `human_reviewed` part.
`inference` holds the milliseconds per frame of the model alone (batch 1,
float32, on the device of the evaluation), `input_size` and `anchor_min_ratio`
the geometry of the checkpoint.
COCO-style mAP and time-based false positives/hour remain later evaluation work.

Feed those errors directly into the next dataset iteration:

```sh
nnslr mine-hard-examples \
  "$NNSLR_DATA_ROOT/evaluation/detector-test.json" \
  --output "$NNSLR_DATA_ROOT/evaluation/detector-hard.jsonl"
```

No checkpoint is automatically copied to Sunnypilot or the Comma.

## Offline end-to-end replay

Once both checkpoints exist, the extracted route can be processed end to end:

```sh
nnslr replay --route ROUTE_ID \
  --detector-checkpoint "$NNSLR_DATA_ROOT/runs/detector-baseline-001/detector-best.pt" \
  --reader-checkpoint "$NNSLR_DATA_ROOT/runs/reader-baseline-001/reader-best.pt" \
  --output "$NNSLR_DATA_ROOT/evaluation/ROUTE_ID-replay.jsonl"
```

Before any model exists, validate the route inventory with no PyTorch/GPU:

```sh
nnslr replay --route ROUTE_ID --dry-run
```

Replay preserves decoded-frame identity, media timing and verified capture
provenance from the extraction manifests. It emits detector boxes and reader
labels only. It deliberately does **not** perform temporal consensus, road
ownership, passage/current-limit inference, or any vehicle-control action.

## Reference export and immutable bundle

Both baseline checkpoints now have reference ONNX exporters:

```sh
nnslr export-onnx --task detector \
  --checkpoint "$NNSLR_DATA_ROOT/runs/detector-baseline-001/detector-best.pt" \
  --output "$NNSLR_DATA_ROOT/runs/detector-baseline-001/detector.onnx"

nnslr export-onnx --task reader \
  --checkpoint "$NNSLR_DATA_ROOT/runs/reader-baseline-001/reader-best.pt" \
  --output "$NNSLR_DATA_ROOT/runs/reader-baseline-001/reader.onnx"
```

Each export writes a hash-bound I/O contract and, by default, compares the
PyTorch result with ONNX Runtime. The detector export includes its torchvision
post-processing/NMS path and therefore still requires an actual trained-model
export run before it can be called target-compatible.

Only parity-verified exports can be packaged:

```sh
nnslr package-model \
  --detector-onnx "$NNSLR_DATA_ROOT/runs/detector-baseline-001/detector.onnx" \
  --detector-contract "$NNSLR_DATA_ROOT/runs/detector-baseline-001/detector.onnx.contract.json" \
  --reader-onnx "$NNSLR_DATA_ROOT/runs/reader-baseline-001/reader.onnx" \
  --reader-contract "$NNSLR_DATA_ROOT/runs/reader-baseline-001/reader.onnx.contract.json" \
  --output "$NNSLR_DATA_ROOT/runs/model-bundle-001"

nnslr verify-bundle "$NNSLR_DATA_ROOT/runs/model-bundle-001"
```

The bundle freezes model files, I/O contracts, class mapping, capability
limits, dataset/split provenance and per-file SHA-256 hashes. A one-byte change
fails verification.

The portable runtime contract can also be exported independently:

```sh
nnslr export-core --output /tmp/speed_vision_core
```

This produces the same stdlib-only file set and tree-digest scheme consumed by
the Sunnypilot vendored core snapshot.

## What is still pending

The CLI surface for T4–T8 is now implemented, but **implementation is not the
same as model evidence**. Still required:

- a sufficiently large human-reviewed Swiss gold corpus;
- real detector and reader training on the V100;
- held-out evaluation and hard-example retraining;
- successful detector and reader ONNX parity runs on real checkpoints;
- timed replay metrics (false positives/hour and latency);
- model bundle benchmark on the target Comma runtime backend;
- physical-track association, road-ownership geometry and passage-evidence algorithms;
- evaluation/freeze of the implemented temporal 3-of-5 / 0.8 s profile;
- target-device shadow/observation/advisory validation.

The portable core already includes bounded temporal consensus, fail-closed
perception-health gating and the separate CAR/MAP/VISION advisory comparator.
These are source-level reference semantics; they are not yet evidence that road
ownership or passage works on real driving scenes.

See `docs/plan.md` for the full task breakdown and `docs/vision-speed-limit/`
for decisions (D1–D8) and the T0 audit.

---

## Rules of the road

- **Advisory only** — nothing here may be wired into a control path.
- **No auto-download / train / push / deploy** — those require explicit
  authorization and are gated by task (see `AGENTS.md`).
- **No private data in Git** — paths, tokens, and machine-specific values are
  kept out of the tree (decision D4).
- **Stdlib-only until a task says otherwise** — adding a dependency is an
  explicit, logged change.
