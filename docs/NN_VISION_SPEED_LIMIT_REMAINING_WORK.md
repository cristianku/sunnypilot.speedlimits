# NNSLR — Remaining implementation and Sunnypilot/Openpilot integration

**Updated:** 2026-09-19  
**TRAIN_REPO:** `cristianku/openpilot-nnslr-tools`  
**RUNTIME_REPO:** `cristianku/sunnypilot`  
**Runtime branch:** `nn-speed-limit-vision`

## 1. Goal

NNSLR must remain an independent, advisory-only vision source:

```text
fcamera / VisionIPC
        ↓
preprocessing
        ↓
detector + reader
        ↓
ObservationBatch
        ↓
temporal tracking / road applicability / passage
        ↓
speedVisionState
        ↓
UI / replay / diagnostics

CAR + MAP → existing SpeedLimitResolver / SLA
VISION   ─X→ no direct control path
```

Permanent invariants:

- never write `CarStateSP.speedLimit`;
- never write or impersonate `LiveMapDataSP`;
- Vision reaches `SpeedLimitResolver` only as its own source `vision`, opt-in (Speed Limit Source "Map Data +
  Vision", 28 Oct 2026: see the last section); it never writes `carStateSP` or the map;
- no CAN, acceleration, steering, torque, radar, safety, or cruise-button output;
- do not change the driving model just to make NNSLR work;
- do not initialize Chestnut from a second process;
- map speed may help find annotation candidates, but is never training ground truth;
- runtime inference must be local to the vehicle.

---

## 2. TRAIN_REPO — current status

Baseline before this local T2 continuation:

```text
6cb124f
```

Already implemented:

- pure `speed_vision_core` contracts and validation;
- `FrameRef`, `Detection`, `ObservationBatch`, `RoadContext`, `SignTrack`, `PerceptionHealth`, `LimitHypothesis`, `SourceSnapshot`, `AdvisoryComparison`;
- explicit unknown/unreadable/not-applicable/unavailable states;
- strict speed-value and timestamp validation;
- local Comma route/video/log tooling:
  - `sync-routes` (default SSH host `comma`, all route segments, narrow camera + full rlog)
  - `route-manifest`
  - `inspect-manifest`
  - `video-probe`
  - `extract-frames` (supports a complete `--route` without a shell loop)
  - `log-metadata`
  - `align-route`
  - `alignment-report`
  - `find-candidates`
  - `make-clips`;
- support for local `fcamera.hevc`, `ecamera.hevc`, `qcamera.ts`, `rlog(.zst)`, `qlog(.zst)`;
- EncodeIndex-based video/log alignment.

A real route is now available locally:

```text
00000089--0ac1c0fdec
segments: 0,1,2,3
```

with `fcamera.hevc` and `rlog.zst` for each segment. Other cameras/logs were
removed and the narrow-camera/full-rlog set was downloaded again as requested.

### Immediate TRAIN_REPO work

<!-- [t2-validation] - START -->
1. Route copy and route-wide frame extraction are implemented:
   ```bash
   nnslr sync-routes --route ROUTE
   nnslr extract-frames --route ROUTE
   ```
   Default SSH host: `comma`; `--host` / `NNSLR_COMMA_HOST` override it.
   Data root: `NNSLR_DATA_ROOT`, otherwise `data/` inside the clone.
   The proposed `copy-comma-route.sh` is superseded by this CLI.
<!-- [t2-validation] - END -->

2. Validate the complete T2 pipeline on the real route:
   ```text
   route-manifest
   video-probe
   log-metadata
   align-route
   alignment-report
   find-candidates
   make-clips
   extract-frames
   ```

<!-- [t2-validation] - START -->
   **Current validation:** local end-to-end execution found and fixed parser
   schema loading, HEVC media clock handling, false exact joins and empty HEVC
   clips. Segments 1–3 provide 2,785 aligned full-rate frames; segment 0 has
   1,180 unresolved frames because the logged indices start at 20. Five map
   candidates are projected. The 198 discovery PNGs extracted previously are
   media-only; they are not retroactively capture-aligned by these changes.

   **T2 remains partial.** See [T2_REAL_ROUTE_VALIDATION.md](T2_REAL_ROUTE_VALIDATION.md).
   Resolve/verify the first-segment mapping, perform the ten-clip manual audit,
   and enrich extracted-frame manifests before declaring the gate complete.
<!-- [t2-validation] - END -->

3. Investigate real-world edge cases:
   - HEVC decode behavior;
   - qlog/rlog parser compatibility;
   - unresolved EncodeIndex joins;
   - fcamera/ecamera/qcamera mapping;
   - partial/missing segments;
   - timestamp consistency.

---

## 3. Dataset / annotation source implemented; corpus coverage still pending

The canonical reviewed-dataset tooling is implemented on `main`:

```text
nnslr import-annotations
nnslr validate-dataset
nnslr build-splits
```

The schema records route/segment/frame identity, image/source hashes, bbox,
sign family, value state/value, applicability, supplementary panel,
visibility/occlusion, review state and timing provenance. Human gold data and
model-reviewed training candidates are kept as distinct dataset kinds.

Splits are deterministic and leakage-safe across connected route, physical-site,
encounter and duplicate-image groups; adjacent video frames are never randomly
mixed just to satisfy ratios. Rejected proposals remain audit-only records.

Still required as evidence rather than source code:

- a sufficiently large human-reviewed Swiss gold corpus;
- broader negative/challenge coverage (ramps, parallel roads, junctions,
  advertising, rear of signs, glare, blur, rain, night, tunnels, temporary and
  electronic signs, supplementary panels);
- reviewed repeated-site labels where different routes revisit the same sign;
- completion of the independent synchronization evidence for the T2 gate.

---

## 4. Model training/evaluation source baseline implemented; real run pending

Implemented on `main` as of 2026-09-20:

```text
nnslr train --task detector
nnslr train --task reader
nnslr evaluate --task detector
nnslr evaluate --task reader
nnslr mine-hard-examples
nnslr env --gpu-smoke
```

The detector baseline is single-class vertical speed-sign detection using
SSDLite320 + MobileNetV3. The reader baseline is MobileNetV3-Small over reviewed
sign crops. Both consume the hash-bound canonical dataset and frozen split;
`--dry-run` validates the exact population without importing PyTorch or
touching a GPU. Model-reviewed candidate data remains explicitly distinct from
human gold data.

The V100 environment is pinned to PyTorch 2.14 / CUDA 12.6, and the explicit
GPU smoke check verifies compiled architecture support plus an FP16
forward/backward operation. No real NNSLR detector/reader training run has yet
been executed on the V100, so there are no accuracy claims or final model hashes.

Current evaluation source supports reader accuracy/per-class confusion and
detector IoU precision/recall/F1 plus false positives per image. Sample-level
errors feed `mine-hard-examples`. Still required before model acceptance:

- real gold-corpus training and held-out evaluation;
- detector COCO-style mAP;
- false positives per hour from timed replay;
- numeric/sign-family accuracy over representative Swiss coverage;
- other-road false-positive rate;
- time-to-first-detection and stable-value latency;
- difficult confusion analysis such as 30/80, 60/80, 80/100, 100/120;
- retraining after mined hard examples.

The model must recognize evidence; it must not directly output an operational
vehicle speed target.

---

## 5. Export, replay and immutable model bundle source implemented

Reference ONNX export exists for both baselines:

```text
nnslr export-onnx --task detector --checkpoint detector-best.pt --output detector.onnx
nnslr export-onnx --task reader   --checkpoint reader-best.pt   --output reader.onnx
```

Each export freezes its I/O/preprocessing contract, records checkpoint/model
hashes, validates the ONNX graph and by default performs numerical parity with
ONNX Runtime. The detector contract includes boxes/scores/labels and its
post-processing/NMS path. Real checkpoint exports still need to be executed
before target compatibility can be claimed.

Offline detector→reader route replay is implemented and preserves decoded-frame,
media-time and capture-time provenance:

```text
nnslr replay --route ROUTE --detector-checkpoint ... --reader-checkpoint ... --output ...
```

Immutable packaging and verification are also implemented:

```text
nnslr package-model ...
nnslr verify-bundle BUNDLE
nnslr export-core --output DIRECTORY
```

`package-model` refuses unverified ONNX contracts or detector/reader artifacts
with different dataset/split provenance. The bundle freezes model files,
contracts, classes, capability limits and per-file SHA-256 hashes; verification
rejects tampering. `export-core` independently snapshots the stdlib-only
`speed_vision_core` with the same file-hash/tree-digest style consumed by
Sunnypilot.

Still required:

- execute detector and reader ONNX parity on real trained checkpoints;
- benchmark the selected model bundle on the actual target backend;
- freeze the first accepted model/core/bundle hashes in the runtime artifact lock.

Expected logical bundle:

```text
speed_limit_recognition/
├── detector.*
├── reader.*
├── config.json
├── classes.json
├── model_card.json
├── capability.json
└── sha256sums.json
```

The bundle must freeze:

- input dimensions and dtype;
- color format;
- crop/resize/letterbox behavior;
- normalization;
- tensor layout;
- output names;
- class vocabulary;
- schema/model/training versions;
- hashes.

ONNX should be treated as an interchange/reference format. The final on-device format depends on the verified runtime backend.

---

## 6. Sunnypilot runtime — current source state

### 6.0 What runs on the comma now (28 Oct 2026)

This replaces the older `nn-speed-limit-vision` scaffolding below for the comma 4 (mici, no Developer UI):

- Branch `cristianku/sunnypilot:speed-limits`, injected by `OP-setup-psa-torque-testing sunny` into
  `cristianku/openpilot:psa-torque-sunny-testing` (first build with the reader: `41a497509`).
- Networks in `cristianku/speed-limits-models` (submodule `openpilot/sunnypilot/speed_limits/models`, format 2,
  commit `62dca08`): SignNet `2026-10-28_1028-signnet-twothirds-s8-f1` epoch 27 as the finder and the classifier
  `2026-10-26_1921-classifier`. No decider.
- `openpilot/sunnypilot/speed_limits/speedvisiond.py` reads the road camera (VisionIPC, conflated) 5 times a second:
  SignNet finds up to 5 boxes (score 0.3), the classifier reads each (70 % confidence), a value becomes the limit when
  2 frames within 1.5 s agree and stays until another; it publishes `speedVisionSP` (m/s, 0: no reading).
- The preprocessing (`vision.py`) is the training's bit for bit: swscale's BT.601 YUV to RGB (x86 fixed point) and
  Pillow's bilinear resize, in numpy. On the comma's little cores: 100 ms a frame (offroad the big cores are off).
- The networks are compiled on the device with tinygrad (`runner.py`, the QCOM flags of SConscript) into
  `/data/media/0/models/speed_limits/`, by the build at boot after an update (`speed_limits/SConscript`, since 29 Oct
  2026: driving off right after an install left speedvisiond compiling for minutes of the drive), else by the offroad
  `speed_limits_warmup` (SignNet 50 s, classifier 96 s) or by speedvisiond itself. Checked on the comma: the same boxes
  and classes as onnxruntime on the server.
- SignNet runs right after a `modelV2` (the GPU is shared with the driving model at the same KGSL level);
  the process runs at niceness 10.
- Shown by the sunnylink toggle Cruise › Speed Limits › Show Vision Speed Limit (`SpeedLimitVisionDisplay`), the
  round sign at the top right of the mici driving screen; used for the speed only with Speed Limit Source › Map Data +
  Vision or, since 1 Nov 2026, Vision Only (`SpeedLimitPolicy` 6: the network's limit alone, no limit without it).
  speedvisiond runs when any of them is on.
- Known limits: no 120 km/h class, some 40 and 50 missed by the classifier on the test route, and no decider, so a
  sign of a side road can be read.
- Speed Limit Assist (31 Oct 2026, `speed_limit_assist.py`, `longitudinal_planner.py`): with openpilot longitudinal on
  a car whose cruise is the PCM's (the PSA), the limit applies as soon as it is known, with no confirmation and no
  alert (the PSA has no button to confirm with); the set speed stays the driver's maximum. The sunnylink toggle
  Set Speed Only Engages (Alpha) (`SpeedLimitSetSpeedEngageOnly`, only with Speed Limit Mode Assist) lets the limit
  replace the set speed, above it too, but only while the big driving model runs (`modelV2.big`, the external GPU);
  the screen then shows a green AUTO square with an eye above the final-limit square and no longer the set speed.
  The rules are listed in the README, "The rules on the car".

Older runtime branch:

```text
cristianku/sunnypilot:nn-speed-limit-vision
```

Initial upstream base:

```text
sunnypilot/sunnypilot:master
a5f44653d7f43ad57fef2f546f3916ec4cbf3c56
```

Already present in the runtime branch:

```text
openpilot/sunnypilot/speed_vision/
├── adapters.py
├── runner.py
├── publisher.py
├── lifecycle.py
├── diagnostics.py
├── speedvisiond.py
├── artifacts.lock.json
└── _vendor/speed_vision_core/
```

Current runtime artifact lock explicitly says:

```json
"model": {
  "state": "unavailable",
  "bundle_digest": null
}
```

So the source scaffolding exists, but there is not yet a real NNSLR model connected to the runtime.

Also already introduced:

- dedicated `speedVisionState` messaging;
- separate service registration;
- default-off NNSLR params;
- optional `speedvisiond`;
- UI subscription and separate VISION badge work;
- portable-core snapshot;
- diagnostics scaffolding.

The operational `speed_limit_resolver.py` currently does **not** consume `speedVisionState`. Keep it that way for the advisory release.

---

## 7. Critical Sunnypilot work to actually run the model

### 7.1 Make runtime CI green

A source-level workflow exists:

```text
.github/workflows/nnslr-source.yml
```

The latest observed runtime CI is not green. The remaining failures observed are currently test-path/root issues rather than a demonstrated NNSLR algorithm failure.

Before calling the runtime source-ready:

- fix repo-root resolution in source tests;
- run the full NNSLR runtime test suite;
- validate cereal/schema generation;
- run relevant existing Sunnypilot speed-limit tests;
- run a real source/build check.

### 7.2 Connect `speedvisiond` to VisionIPC

Current daemon behavior is still bootstrap/fail-closed.

Implement:

```text
camerad
  ↓
VisionIpcClient
  ↓
VISION_STREAM_NARROW_ROAD
  ↓
owned frame / verified buffer lifetime
  ↓
NNSLR preprocessing
  ↓
runner
```

Required behaviors:

- connect/reconnect;
- preserve frame ID and SOF/EOF timestamps;
- bounded pending queue;
- discard stale frames;
- track dropped frames;
- invalidate state after producer restart;
- report camera unavailable explicitly.

### 7.3 Implement runtime preprocessing parity

Add a dedicated runtime preprocessing layer that exactly matches the model bundle:

```text
NV12
→ color conversion
→ crop
→ resize/letterbox
→ normalization
→ tensor layout/dtype
```

Training/reference preprocessing and runtime preprocessing must be compared on golden fixtures.

### 7.4 Select and implement the real inference backend

This is still open.

Possible profiles:

**Independent backend**
```text
speedvisiond → NNSLR inference
```

**Accelerator-owner integration**
```text
modeld
 ├─ driving model
 └─ NNSLR auxiliary inference
        ↓
speedVisionObservations
        ↓
speedvisiond
```

Important source constraint already identified: if Chestnut is used, it must remain owned by `modeld`; do not initialize a second Chestnut process.

Before selecting the backend, verify the actual device/runner, operators, memory, latency, thermal headroom, and model format support.

### 7.5 Finish the real runner

The backend should implement:

```python
prepare()
submit_latest(frame)
poll()
health()
close()
```

`prepare()` must verify bundle hashes/config/classes and warm up off-road.

`submit_latest()` must never create an unbounded queue.

`poll()` must produce a real `ObservationBatch` associated with the original capture timestamp.

`health()` must expose availability, faults, latency, last processed capture, dropped frames and overflow.

### 7.6 Implement postprocessing

Needed:

- detector decode;
- NMS;
- crop extraction;
- reader decode;
- confidence handling;
- unsupported-domain rejection.

Do not put road-law interpretation inside the detector/reader.

---

## 8. Temporal semantics — bounded consensus/state gates implemented

The portable core now contains a conservative temporal reference profile:

```text
3 coherent unique captures
out of at most 5 recent observations
inside an 0.8 s window
```

It operates only on observations already associated to the same candidate
physical track. Duplicate captures cannot count twice, old evidence falls out
of the window, and a single frame can never establish a numeric track
consensus.

The hypothesis gate is explicit:

```text
stable value + ownership UNKNOWN → OBSERVED
stable value + ownership OTHER   → UNCERTAIN
stable value + OWN + AHEAD       → AHEAD
stable value + OWN + PASSED      → CURRENT
```

`CURRENT` is usable for advisory only in the final case. Sign disappearance
alone is never treated as passage evidence.

A fail-closed perception-health gate is also implemented. Backend unavailable,
fault, detection overflow, missing/future processed capture, or capture age over
the configured freshness threshold invalidates the prior value and yields
`UNAVAILABLE`.

Still missing as real algorithms/evidence:

- physical sign track association from sequential detector observations;
- calibrated road ownership using camera geometry/context, with OWN/OTHER/UNKNOWN;
- verified passage/crossing evidence using ego motion/context;
- evaluation/freeze of the proposed 3-of-5 / 0.8 s profile on real routes.

---

## 9. CAR / MAP / VISION comparison — advisory core implemented

The portable core now has a separate read-only comparator. It never mutates
CAR/MAP snapshots and has no target-speed or control output.

Behavior follows the project table:

```text
Vision absent/stale/unavailable → no current claim
Vision OBSERVED                 → separate observed indication
Vision AHEAD                    → separate ahead indication
Vision CURRENT + CAR/MAP agree  → AGREE
Vision CURRENT + source differs → CONFLICT
Vision CURRENT + sources absent → current Vision indication, insufficient_data
```

Only fresh/valid operational snapshots participate. Agreement tolerance and age
limits are explicit profile fields rather than hidden constants, and source
timestamps are preserved.

This remains diagnostic/advisory evidence only. Vision is still not inserted
into the operational `SpeedLimitResolver`, and source agreement never boosts
model confidence or becomes training truth.

---

## 10. UI and settings still to finish

A separate VISION HUD element has been started.

Remaining:

- real-device layout validation;
- metric/imperial display;
- observed/ahead/current/unavailable/stale/conflict rendering;
- shadow mode hidden from on-road UI;
- Settings UI for:
  ```text
  Off
  Shadow
  Observation
  Advisory
  ```
- warning toggle;
- diagnostics toggle;
- snapshot/layout tests.

The first visible release should show a clearly separate source, e.g.:

```text
VISION 80
```

It must not silently replace the operational SLA/MAX badge.

---

## 11. Privacy / diagnostics remaining work

Audit these Sunnypilot paths before enabling real diagnostics:

```text
loggerd
uploader
sunnylink
crash logs
backup
file sharing
```

Default behavior:

- no raw frames;
- no route upload;
- no GPS dump;
- bounded private diagnostics;
- diagnostics failure must not block the driving system.

---

## 12. Source and device validation

Before any on-road use, verify non-interference against the newly selected upstream baseline.

At minimum compare:

```text
actuator requests
cruise requests
set speed
SpeedLimitResolver output
SLA source
engagement behavior
radar behavior
safety configuration
```

Inject:

```text
Vision absent
Vision valid
Vision incorrect
Vision stale
Vision malformed
backend unavailable
backend crash
camera unavailable
```

None may affect operational control.

Target benchmark later:

- capture-to-result p95 ≤ 150 ms;
- capture-to-result p99 ≤ 200 ms;
- pending queue ≤ 1;
- no new primary deadline misses;
- measure RAM, temperature, throttling, dropped frames and main-model latency regression.

---

## 13. Replay and release progression

Recommended progression:

```text
offline replay
  ↓
source-ready
  ↓
target benchmark
  ↓
shadow
  ↓
observation UI
  ↓
advisory UI
```

Replay must use recorded camera/log data as runtime input; annotations remain evaluation-only and must never act as oracle runtime inputs.

---

## 14. Reapply-on-new-master tooling

A dedicated TRAIN_REPO branch exists:

```text
nnslr/sunnypilot-reapply-v1
```

It already contains the reapply design/tooling, including:

```text
integration/sunnypilot/feature.lock.json
integration/sunnypilot/hook-map.json
integration/sunnypilot/protected-surfaces.json
integration/sunnypilot/schema-bindings.json
scripts/nnslr_reapply.py
.github/skills/nnslr-reapply/SKILL.md
```

The latest observed CI on that tooling branch is green.

However, the feature lock is still:

```text
state = bootstrap
source_feature_commit = null
patches = []
```

Therefore the runtime feature is **not yet sealed as a reproducible patchset**.

After the runtime stabilizes:

1. identify the exact known-good runtime commit;
2. organize the NNSLR commits into a clean logical patch series;
3. change the feature lock to `sealed`;
4. record source base, feature commit, patch list, core/model/schema hashes;
5. reapply to the same base and compare the resulting tree;
6. reapply to a newer real Sunnypilot master;
7. run compatibility/build/source tests;
8. generate diff/range-diff/report;
9. keep the previous known-good version intact.

Final maintenance workflow:

```text
current Sunnypilot master
        ↓
nnslr-reapply skill/tooling
        ↓
new candidate branch
        ├─ source tests
        ├─ build
        ├─ compatibility report
        └─ no automatic deployment
```

---

## 15. Recommended implementation order from here

### A — Real data
<!-- [t2-validation] - START -->
- [x] `sync-routes` with default host `comma`, resume and segment discovery;
- [x] route-wide `extract-frames`;
- [x] run the local T2 commands on the four-segment route and fix observed failures;
- [x] project five map candidates and generate review clips;
- [ ] establish a verified mapping for the nonzero first-segment index;
- [ ] complete the ten-clip manual alignment audit and timestamp/geometry frame manifests;
- [ ] extend validation to missing/truncated recordings and other camera streams.
<!-- [t2-validation] - END -->

### B — Dataset
- annotation schema;
- annotation import/review;
- negative/challenge examples;
- dataset validation;
- leakage-safe splits.

### C — Model
- detector baseline;
- reader baseline;
- train on V100;
- evaluate;
- hard-example mining;
- retrain/finalize.

### D — Export
- ONNX/reference export;
- parity tests;
- immutable bundle;
- hashes/capabilities.

### E — Sunnypilot runtime
- fix runtime CI;
- VisionIPC input;
- preprocessing parity;
- real inference backend;
- postprocessing;
- temporal tracking;
- ownership/passage;
- UI/settings/privacy.

### F — Device
- build;
- off-road smoke test;
- runtime benchmark;
- fault tests;
- shadow;
- observation;
- advisory.

### G — Maintenance
- seal patchset;
- same-base reproducibility test;
- newer-master reapply test;
- rollback documentation.

---

## 16. Definition of Done

The project is not complete until all of the following are established:

**Training**
- versioned dataset and leakage-safe split;
- evaluated detector/reader;
- hard negatives;
- final model identified by hash;
- export parity.

**Runtime**
- real camera input;
- exact preprocessing;
- verified target backend;
- bounded queue/fault handling;
- real `speedVisionState`;
- temporal state;
- ownership and passage;
- separate UI.

**Safety**
- operational CAR/MAP resolver unchanged;
- no CAN/control output;
- default off;
- fail-closed;
- regression/non-interference suite green.

**Target**
- build passes;
- benchmark passes;
- shadow evidence collected;
- thermal/memory behavior acceptable.

**Maintenance**
- sealed patchset;
- working reapply skill/tool;
- reapply tested against a newer master;
- old version preserved;
- rollback documented.

## 17. Immediate next checkpoint

The next useful milestone is:

```text
real route
  → deterministic aligned dataset inputs
  → first annotated dataset
  → first trained detector/reader
  → verified model bundle
  → Sunnypilot VisionIPC inference
```

Until a real model bundle exists, the Sunnypilot runtime should remain fail-closed and report Vision as unavailable rather than fabricating a speed limit.

<!-- [preannotation] - START -->
### Pretrained annotation assistance

`nnslr preannotate --route ID` and `nnslr review --route ID` now provide
CPU-only model proposals and browser review with the standard data-root
default. Sign detection uses a pinned Vietnam YOLO ONNX export. Asphalt OCR
uses PP-OCRv4 and a pinned Cityscapes road-segmentation model: numeric text
must pass an independent road-surface/context check before becoming a
`road_marking_candidate`. It is never automatically an applicable limit.

The UI supports box addition/removal, class/value corrections, explicit frame
confirmation, browser persistence and export of confirmed frames with original
proposal provenance. Unreviewed and empty model outputs are not ground truth.

Local CPU validation (2026-09-19): all 198 extracted route frames completed,
with 100 proposals across 69 frames. Four proposals were road-number candidates
(all 30), including the requested segment-0 frame 27 and two following frames.
The additional road proposal was visually checked on a separate zone-30 marking.
This is successful execution and a small positive check, not an accuracy metric.
Synthetic perfect-score OCR readings placed in actual wall/advertisement/parked
vehicle mask regions were rejected; the asphalt region was accepted.

The regression run passed 165 tests, including optional real Chrome review/export
and storage-failure checks; seven optional log-parser tests were skipped because
their separate environment was not configured. The core selftest passed all 18
checks. A real-image browser check also exercised box drawing/removal, value
correction, confirmation reset, reload persistence and export. No container or
comma installation was performed for this extension.

Still outstanding: curated positive/negative validation across scenes,
Swiss sign-domain assessment, sequence consistency and road applicability,
review export import into the canonical annotation schema, source-frame/time
alignment, dataset validation/splits and actual training. The OCR/segmentation
scores do not establish these properties. See README for commands, model
provenance and the external weights' license constraints.
<!-- [preannotation] - END -->

<!-- [reviewed-dataset] - START -->
## Update: reviewed annotation tooling and frame provenance

`import-annotations`, `validate-dataset` and `build-splits` now have actual
implementations, default data paths and deterministic tests. The canonical
schema and grouping policy are documented in [DATASET.md](DATASET.md).
Review captures reviewer/time, preserves proposal provenance and keeps
unknown attributes explicit. Whole-route/site/encounter/image-hash grouping
prevents frame-neighbor splitting; fewer than five independent groups cannot
populate all requested partitions. Capture-aware manifests are available, but
segment 0 and the independent ten-point synchronization audit remain unresolved.
No reviewed Swiss corpus, baseline training, export bundle or runtime model is
claimed by this local increment.
<!-- [reviewed-dataset] - END -->

## Proposal (27 Oct 2026): the model as a plugin, memory inside the ONNX

Decided direction, not built yet. Sunnypilot gets one fixed runner, written once; everything that reads
signs or applies rules is in the ONNX bundle downloaded from the Hugging Face catalog (`speed_models.json`).
A new network, a new rule or a new country is a new bundle, never a sunnypilot change.

The bundle: cutter (with a road-type head on its backbone) + sign classifier + memory, exported as one ONNX.
The memory is plain tensor operations (`Where`, `Max`, ...; no `If` or `Loop`, so tinygrad compiles it),
with its state fed back by the runner, as openpilot already does for the driving model.

Contract between runner and bundle (fixed; `state` has spare room for future rules):

| name | dir | shape | meaning |
|---|---|---|---|
| `frame` | in | road camera frame, as the cutter takes it today | |
| `state_in` | in | `[16]` float | `state_out` of the previous frame; zeros at start |
| `general_limits` | in | `[4]` float | town, rural, expressway, motorway of the current country (the runner picks them from GPS and a borders file) |
| `limit_kph` | out | `[1]` float | limit in force; 0 when the model is not sure (sunnypilot then takes the map) |
| `state_out` | out | `[16]` float | to pass back as `state_in` |

The runner does only: camera → ONNX → `speedVisionSP`; `state_out` → `state_in`; the 4 general limits.

### What the model returns (decided 27 Oct 2026)

- A speed sign read → that limit, held until the next sign (the memory inside the ONNX).
- The sign that starts a general limit read (CH: "50 generell", the motorway or expressway start; DE: the town
  entry), and no speed sign since, with the road type sure → that general limit of the country.
- Anything else → 0: a side road joined without any sign, an unclear picture, the start of a drive. The road-type
  head alone never gives a limit: a sure motorway is 120 only half of the time on our drives (OSM: 266 stretches at
  120, 187 at 100, 79 at 80), a sure town can be a Zona 30. The map knows those, so 0 lets sunnypilot use it.
- The value must be steady: a limit that drops to 0 and back makes sunnypilot jump between the network and the map.

The order, strongest first:

1. a speed sign read now → that limit (inside a car park see 2);
2. a car park recognised (the road-type head says parking for a few seconds) → **10 km/h**, a sign read there
   only when lower (5 stays 5, 20 becomes 10); the limit remembered from the road before is dropped, not
   suspended: leaving the car park the model gives 0 until it reads a sign, because the road and the direction it
   re-enters are not known;
3. the limit remembered from the last sign → that limit;
4. the general limit after the sign that starts it, no speed sign since, road type sure → 50 / 80 / 100 / 120;
5. anything else → 0, the map decides.

Road types of the head: town, rural, expressway, motorway, parking. A car park has no legal general limit, but the
model returns 10 km/h there (decided by Cristian on 27 Oct 2026, replacing 0; 10 instead of 15 since 29 Oct 2026): with 0 sunnypilot takes the map, which
often has no limit in a car park or the limit of the road next to it, and the car could speed up where one must drive
at walking pace.

Open: the memory across drives (parked inside a Zona 30, the sign is not seen again when leaving): sunnypilot saves
`state_out` and the parking position at the end of a drive; at the next start the model gets it back only near the
same place, and the limit counts as from the previous drive.

### sunnypilot side: branch `speed-limits` of cristianku/sunnypilot

From master `db27a433` (synced with sunnypilot/sunnypilot master on 27 Oct 2026), commits `638211d6` and `e89824fa`:

- Speed Limit Source has a sixth choice, **Map Data + Vision** in sunnylink (`Map+Vision` on the device,
  `SpeedLimitPolicy` 5): the network's limit when it has one, else the map. The five upstream policies stay as they
  are and ignore the network; the car's own limit is not used by this one. (On 27 Oct it was a separate setting,
  Signs Read By, `SpeedLimitSignReader`; replaced on 28 Oct because Cristian wants the choice in Speed Limit Source.)
- `speedVisionSP` (in `customReserved10 @136`): `speedLimit` m/s, 0 when not sure; `modelName`.
- `SpeedLimitResolver`: source `vision`, skipped at 0 like the car; a stale `speedVisionSP` is never used.
- The models: the public repo `cristianku/speed-limits-models` as a submodule at
  `openpilot/sunnypilot/speed_limits/models` (commit `2f2700fb`), the way sunnypilot takes its lateral models from
  `neural_network_data`, not a Hugging Face download. A new model is a commit there and a move of the pointer.
- Missing: `speedvisiond` (camera → ONNX → `speedVisionSP`) and the tinygrad compile of the ONNX in the build
  (SConscript, as modeld does). Until then the network's limit is always 0, so Map Data + Vision is the map.

The old branch `nn-speed-limit-vision` (September, 277 commits behind master) is not carried over: it published only
an "unavailable" state, had no inference, and its tests forbade the resolver source decided now.

## Measured on the Chestnut (27 Oct 2026)

The three networks in use (package `chestnut-test-20261027`, exported locally, nothing uploaded) run on the comma's
Chestnut with tinygrad (`DEV=AMD`, the device sunnypilot's modeld uses), comma offroad, test files only in
`/data/tmp/sl-chestnut`, openpilot untouched. Outputs compared with onnxruntime on the server, same inputs:

| network | ONNX as exported | tinygrad | max difference | with TinyJit |
|---|---|---|---|---|
| classifier (160 px crop) | as is | runs | 6e-6, same class | 8–13 ms a crop |
| decider (8 boxes) | `RoiAlign` not supported | runs with RoIAlign written with gathers (2e-7 from torchvision) | 5e-6, same classes | 14 ms |
| cutter (1344 × 760, the old Detector) | `NonMaxSuppression` not supported (data-dependent shapes) | runs without its post-processing: boxes and scores of all 32,292 default boxes, the filter and NMS after it on the CPU (same top detection as the full model) | boxes 0.004 px, scores 3e-6 | 119 ms a frame |

The Chestnut stayed at 48–52 °C (memory 63–67 °C), 22–25 W, fan 0–2 rpm. In the drive `00000141` the driving model
alone used it at about 40 % (49 W, 52–55 °C, 27 ms a frame).

To do before the car: export the cutter without its post-processing and the decider with the plain RoIAlign in
`nnslr_tools.exporting` (both were test scripts); feed the cutter uint8 and normalise on the GPU (the 12 MB float32
upload over USB is likely most of the 119 ms); then the memory, and one ONNX (or one JIT) for the chain.

### Together with the driving model, and on the comma's own GPU (27 Oct 2026, evening)

- **Chestnut, one process with the driving model** (Mountain Dew Model v3 loaded as modeld loads it, run at 20 Hz on
  its warmup inputs; our chain on real frames, the cutter every 3 frames): the driving model alone ran at 40 ms
  median (2.5 % of frames over 50 ms); with our chain only 52 frames ran in 60 s instead of 1,200 and 36 % were late
  (cutter 400 ms median there). Taking the picture to the Chestnut over USB costs 115–160 ms (float32 160, uint8 115).
  Not viable as it is: the Chestnut stays the driving model's. Inside modeld, using the frame already on the
  Chestnut, would avoid the upload but ties us to every driving-model release and to its 10–13 ms of spare time.
- **Adreno (the comma's own GPU, tinygrad `DEV=QCOM`, a process of its own like dmonitoringmodeld)**, outputs
  identical to the server: classifier 5.8 ms a crop, decider 213 ms for 8 boxes (fine when a sign appears, not every
  frame), **cutter (the old Detector) 1,697 ms a frame at 1344 × 760**: too slow.

Next: a cutter made for the Adreno: lower input (e.g. 672 × 380) and/or only the part of the picture where signs
stand, a lighter backbone, at 2–5 frames a second (a sign stays in view 1–3 s); measured on the Adreno before training
it fully.

### SignNet, a sign finder made for the Adreno (28 Oct 2026)

`nnslr train-signnet` (`nnslr_tools.signnet`): plain 3 × 3 convolutions on the top band of the frame (the top 416 of
760 rows hold 97 % of the reviewed signs), a heat map of sign centres with each sign's size (CenterNet), no NMS. Trained
from scratch on the cutter's plan and scored like it, on the same 1,133 test frames (381 signs); the best N boxes a
frame. Times on the Adreno with openpilot's tinygrad flags (`IMAGE=1 FLOAT16=1 NOLOCALS=1 JIT_BATCH_SIZE=0
OPENPILOT_HACKS=1`), TinyJit, outputs checked against PyTorch:

| finder | picture | parameters | Adreno | all signs, 5 boxes | speed signs, 5 boxes | far (< 12 px) |
|---|---|---|---|---|---|---|
| old Detector (SSDlite), the cutter then | 1344 × 760 | — | 540 ms | 61 % | 62 % | 29 % |
| SignNet stride 8 | 1344 × 416 | 163 k | 138 ms | 58 % | 67 % | 40 % |
| SignNet stride 4 | 1344 × 416 | 81 k | 119 ms | 59 % | 66 % | 45 % |
| SignNet stride 4 | 896 × 288 (2/3) | 81 k | 63 ms | 54 % | 64 % | 34 % |
| SignNet stride 4 | 672 × 208 (half) | 156 k | 87 ms | 49 % | 59 % | 22 % |

Untrained, timed only: 672 × 208 at stride 8 51 ms, 448 × 144 at stride 4 49 ms; the old Detector at 640 × 360 122 ms.
SignNet is weaker than the old Detector on the big signs (≥ 40 px: 43–54 % against 67 %) and the other signs; to look at
before the next round.

**Where those milliseconds go** (profiled the same day, 896 × 288 at stride 4): of the 63 ms, 20 are the upload of
the picture from the CPU and 22 the read of the two outputs; the GPU itself takes 21 ms when the picture is already
there (as in the car: camerad's buffer is on the GPU, dmonitoringmodeld reads it with `Tensor.from_blob`). Two
things halve that again, both measured with the picture on the GPU and TinyJit:

| SignNet, GPU only | 896 × 288 | 1344 × 416 |
|---|---|---|
| two outputs (heat, shape), fuse 3 × 3 | 40 ms | 90 ms |
| **one output** (heat and shape in one tensor), fuse 3 × 3 | 21 ms | — |
| one output, **fuse 1 × 1** (`--fuse-kernel 1`) | **12 ms** | 26 ms |
| one output, fuse 1 × 1, stride 8 | 12 ms | 27 ms |

So the ONNX for the car has one output, and the 1 × 1 fuse was trained the same day (runs
`2026-10-28_1028-signnet-twothirds-s4-f1`, `-s8-f1`, 63 k parameters): at stride 8 it keeps the signs (55 % of all,
62 % of the speed signs at 5 boxes, against 54 % and 64 % for the 3 × 3 fuse at stride 4), at stride 4 it loses three
points (51 % and 62 %). So the car gets stride 8 with the 1 × 1 fuse, 12 ms of GPU, once retrained on the corrected
labels below. The classifier is 5 ms a box on the same measure; the decider, 213 ms, runs only when a sign appears. Runs: `data/runs/2026-10-28_0718-signnet-s8`, `-s4`,
`2026-10-28_0854-signnet-twothirds-s4`, `-half-s4`.

### Is SignNet limited by the data, by the labels or by the net? (28 Oct 2026)

Measured on `2026-10-28_0854-signnet-twothirds-s4` (896 × 288, stride 4, the cutter's plan and scoring, 5 boxes a
frame), on the 1,133 test frames and on 1,200 training frames drawn at random. Scripts of the day in the session's
scratchpad (`analyze.py`, `negatives.py`, `leakage.py`, `candidates.py`, `learning_curve.py`).

| recall at 5 boxes | test | train sample |
|---|---|---|
| all signs | 54 % | 76 % |
| speed signs | 64 % | 86 % |
| speed signs, how the rest is missed | 43 not found, 23 box too loose (IoU 0.3–0.5), 9 ranked 6th or later | 35 not found, 107 too loose, 8 ranked later |
| false alarms (score ≥ 0.3 on nothing boxed) | 0.06 a frame | 0.25 a frame |

What it says:

- **The net does not even fit its training frames** (86 % of the speed signs). The gap to the test (64 %) is real but
  not the main loss: a bigger net or longer training is worth less than fixing what follows.
- **The labels are the first limit.** A quarter of the training "alarms" are real signs with no box: SignNet is
  confident on 717 frames where no reviewed box covers its sign (score ≥ 0.3; 96 frames at ≥ 0.5, nearly all real
  signs on the contact sheets: give way, roundabout, no entry, width limits, some speed signs and lit displays; 493
  of the 717 frames are confirmed as "no sign"). Every such frame teaches "sign = background". Recall on the training
  frames also follows who labelled them: 87–89 % on frames reviewed by Claude Code and Codex, 78 % by Cristian,
  67 % by the Qwen reviewer on the V100 and 46 % by strata's small Qwen (818 signs of the 2,284 in the sample).
- **Loose boxes count.** 107 of the 150 speed signs missed on training frames are found with a box of IoU 0.3–0.5:
  SignNet's own size head is coarse, and some reviewed boxes are loose too.
- **Switched-off displays** are 92 of the 360 boxes of `variable_display` without a value (the other 268 are lit but
  unreadable). An off panel looks like the gantry around it: on the test the family scores 2.5 % (40 boxes).
- **Which signs are worst:** other signs (43 % on the test, direction panels 0 %, no entry 21 %) and signs under
  12 px (41 % of the speed signs, 34 % of all); clear speed values are fine (60 → 95 %, 40 → 94 %, 100 → 8 of 8).
- **How much data there is per value** (confirmed speed signs, whole dataset): 60 → 279, 80 → 110, 100 → 102, 50 →
  77, 40 → 41, 20 → 28, 30 → 13, 120 → none (two boxes, not confirmed yet); 557 with a number that cannot be read.
  80 and 100 come from two routes.
- **The learning curve** (a quarter, half and all of the training segments, same net, same test, 5 boxes a frame;
  runs `2026-10-28_1100-signnet-twothirds-s4-share25`, `-share50`; half is 4,057 of 8,188 training frames and
  1,547 of 3,289 boxes): speed signs 56 → 63 → 64 %, all signs 40 → 50 → 54 %, other signs 21 → 34 → 43 %, big
  signs (≥ 40 px) 27 → 41 → 46 %. Doubling the data from half to all buys one point on the speed signs and nine on
  the other signs: for the speed signs the limit is no longer the quantity but the labels and the net's precision;
  the other signs, whose boxes are the ones missing most, still gain from every frame. More of the same roads will
  not move the speed signs; new values (30, 40, 120) and corrected labels will.
- **The split leaked between train and validation**, not between train and test: `place-v3` put a buffer around the
  test only, so the validation shared its areas (and signs) with the training; the epoch it picked and the recall it
  reported were optimistic. The test (one route, 15 %, buffered) was sound.

Done the same day:

- **Split `place-v4`** (`nnslr build-splits`): a buffer around the validation too; the validation is two compact
  stretches of road, each started where it adds the most speed values, so the buffer costs 640 rows instead of 4,086
  for scattered areas (train 9,389, validation 1,998 with the values 15–80, test unchanged at 1,284). Split
  `e17e66fa…`.
- **Off displays are ignore boxes** (`training.off_display`, `signnet.encode`): a `variable_display` box without a
  value whose note says off, blank, unlit, spento or vuoto is neither a sign to find nor background; SignNet's loss
  skips its cells (one cell around) and `measure` drops predictions on it. The cutter reads boxes only, so for it
  the panel stays background.

Still to do:

- **Box the 717 frames by hand** (602 AI-reviewed, 115 human-reviewed): the list per route, highest score first, is
  `data/evaluations/unboxed-signs-2026-10-28/frames.md` on the server. They are reviewed by hand on the review site,
  not by agents in bulk. After the next import, retrain and rebuild the split (the new boxes change the groups by
  image).
- **A new independent test** after the corrections: the current test route was reviewed before them and its
  frames among the 717 (37) are corrected too, so the numbers above are not comparable with the next ones. One or
  two new drives on roads never driven, confirmed by Cristian, pinned in `data/annotations/test-routes.txt`.
- **Targeted collection**, in this order of value: towns and villages with 30 and 40 (Zone 30 entries, school
  streets, town entries and exits: 30 has 13 boxes, 40 has 41); motorway entries and ends of roadworks for 120 and
  `motorway_start` (no confirmed 120, 9 motorway signs); 80 and 100 on roads other than the two that hold them now; gantries with lit
  displays (the test's 40 are mostly off or unreadable); dusk, rain and low sun (signs of poor visibility score 40 %
  against 58 % clear). Each new drive on a new road lands whole in one area of the split; drives repeated on the
  same roads add little.

### One cutter: SignNet (30 Oct 2026)

SignNet is now the one cutter, on the server as on the car; the SSDlite Detector is kept only to compare (its
versions stay on the Models tab and cannot be put in use; one already in `models/active.json` stays until a SignNet
is chosen). Why: the classifier learns the false boxes of the cutter, so they must be those of the cutter the car
runs; and there is one network to train and measure.

- **Train the cutter** runs `scripts/run_signnet.sh NAME`: `nnslr train-signnet` with the car's setup (band
  896 × 288, stride 8, fuse 1 × 1, widths 16,24,32,48; families all or speed), about 80 minutes on the server's GPU,
  into `data/runs/NAME/signnet-best.pt`, `result.json` and `signnet.log`. A NAME stopped half way is trained again
  from scratch, its old `signnet.log` moved aside first; the dry run of step 1/3 exits 1, and stops the script before
  the GPU, when the plan has nothing to train or validate on.
- **Use this one** takes only a finished SignNet whose band is 896 × 288 and stride 8 (`signnet.fits_the_car`, the
  `SIGNNET_WIDTH`, `SIGNNET_HEIGHT` and `SIGNNET_STRIDE` of the car's `vision.py`); a Detector is refused
  (`compare_only`), another band or stride too (`not_for_the_car`). **Send to the car** and **Try on the comma now**
  refuse such a SignNet, or an unfinished one, before starting.
- Prepare's Our networks, `mine-negatives` and `evaluate-chain` take the car's best 5 boxes a frame, from score 0.1
  (the car keeps those of 0.3 or more), from the SignNet in use: the heat map decoded best first, at most 5 a frame
  (`CAR_MAX_BOXES`), scaled to frame pixels, no size filter.
  `evaluate-chain` measures at cutter scores 0.1, 0.2 and 0.3 and judges at 0.3 (`CAR_MIN_SCORE`); mining keeps
  score 0.1, 3 a frame and sign score 0.3, on those 5 boxes.
- The trade-off, on the split of 26 October and the best 5 boxes a frame at any score: the SignNet on the car
  (`2026-10-28_1028-signnet-twothirds-s8-f1`) has 55 % of all test signs, 62 % of the speed signs and 26 % of those
  under 12 px, against 61 %, 62 % and 29 % for the 1344 × 760 Detector. What the car reads, at score 0.3: 40 % of
  all test signs (152 of 381), 58 % of the speed signs (120 of 208), 0.22 false boxes a frame. Signs below the band
  (road markings, low signs close by) are not found, as on the car, so chain and proposer numbers are not comparable
  with the old ones.
- Every SignNet so far was trained on the split of 26 October (`b0c47a58…`); a new **Train the cutter** on the
  current split is needed before one is put in use.
