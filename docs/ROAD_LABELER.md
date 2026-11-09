# The road labeler

## What it is for

The road labeler labels the drives of Speed review so that a person has less to do. For every moment of a drive it says
three things:

- the road type: one of `road_types.ROAD_TYPES`, which are town, rural, expressway, motorway, ramp, parking, zone30,
  garage and merge;
- the kinds of road: tunnel and the other kinds of `speed_road_kinds` (narrow, country, gravel, potholes...);
- whether the car is inside roadworks, and so where they start and end.

It is an offline network. It never runs on the car and is never an input of the car. What it writes is the base layer of
Speed review, definitive (9 Nov 2026): there is no confirmation step. Speed review shows it, the brain learns it as it
learns a person's word, and a person corrects it only where it is wrong; a person's marks, removals and checks always
win. Those corrections are what it learns from next time. Because it is offline, the
OpenStreetMap map is one of its inputs. The rule "vision never reads the map" (AGENTS.md) is about the car, and the car
never sees this network.

It is started only on request:

- `helpers/road_labeler.sh train`
- `helpers/road_labeler.sh predict`

These wrap `nnslr road-labeler-train` and `nnslr road-labeler-predict`. The portal starts both from the Road
labeler card on Models (see Portal below).

Code:

- `src/nnslr_tools/road_labeler_data.py`: the steps, the inputs and the answers. No torch.
- `src/nnslr_tools/road_labeler_net.py`: the encoder, the decoding and the network.
- `src/nnslr_tools/road_labeler.py`: the training, the measure, the prediction and the writes into Speed review.

## The design limits

| | value | what it costs |
|---|---|---|
| step rate | 2 a second (`STEP_HZ`): the 1 Hz frames of the brain's split, plus the video frame half a second later | A start or an end is placed to about ±0.5 s. A stretch shorter than 2 s (`MIN_MARK_S`) is never written. |
| picture | the road camera `fcamera.hevc` (1344x760, or 2688x1520 at full definition), decoded at 2 a second and scaled to 768x432 | A far sign or a single cone is a few pixels. A full-definition video is slower to decode for the same picture (not measured). |
| encoder | MobileNetV3-large on ImageNet, frozen; its 960 channels pooled to a 2x4 grid, so 7680 numbers a step, kept as fp16 | Only the coarse place in the frame survives (left, right, top, bottom). The encoder never learns. The cache is keyed by `encoder_id`, so a new encoder means new features. |
| memory | a bidirectional GRU, 2 layers of 128 each way, after a 256-wide projection of the picture plus 48 inputs (47 of the map, the GPS and the speed, and whether the step has its picture); 2.6 M weights | It learns on windows of 960 steps (8 minutes). At prediction every step is said with at least 2 minutes (`MARGIN`) on each side, and up to 6, except at a drive's ends. Nothing further away than about 6 minutes reaches a step. |
| answers | a road type at every step that has one; a kind of road or the roadworks only inside a mark, before that kind's "checked to here", or where a person removed a mark; the tunnel also where the GPS has a fix on a road of the map that is no tunnel | No drive had a "checked to here" on 8 Nov 2026, so only the road type and the tunnel can learn yet (see below). |

The cost per drive was measured on this server's CPU. The V100 rows are estimates, because no GPU was used.

- Decoding takes about 13 s of one core for each minute of drive, with 4 segments in parallel. A 30-minute drive is about
  6.5 core-minutes, or about 1.6 minutes of wall time. Only the first run of a drive pays this.
- The encoder runs at 5.4 frames a second on the 20 CPU threads. That is about 11 minutes for the 3600 steps of a
  30-minute drive, so the CPU is only good for a dry run. On the V100 with fp16 it should be well under a minute a
  drive.
- The features take about 55 MB for a 30-minute drive, kept in `derived/road_labeler_features/<encoder_id>/`. The 57
  drives of today come to about 175,000 steps (24 hours), about 2.7 GB.
- The memory takes 45 s for one CPU step of 8 windows of 960 steps. On the V100, cuDNN should make that a fraction of a
  second, so an epoch of today's train (about 120,000 steps) should take seconds and 30 epochs a few minutes.
- The first full training would therefore take about 1.5 hours of decoding plus some minutes of GPU. Later trainings
  take minutes.

## Inputs

All inputs are at 2 a second (`road_labeler_data.INPUT_NAMES`):

- **The map scene.** These come from `derived/road_types/<route>.jsonl`, the map's own lines, never with a person's
  corrections over them:
  - `map_type` and `highway`, one-hot;
  - `maxspeed` in km/h, with an mph tag converted, and whether there is one;
  - the family of the `basis`;
  - whether the map is sure;
  - whether the map's way has the tunnel tag in `derived/osm/roads.jsonl`.
- **The GPS.** Whether there is a fix within 1 s, and the seconds since the last fix and to the next one. A garage or a
  tunnel is where the GPS is lost.
- **The car's speed.** As the dashboard shows it (`drive_logs.dashboard_speed`).
- **The picture flag.** Whether the step has its picture.

A quarter of the training windows lose the map's inputs (`MAP_DROP_SHARE`), so that the pictures have to say the road
too where the map is wrong or missing.

## Answers and their weight

- **Road type.** A person's correction or confirmation (`annotations/road_types`) weighs 1. The map's own type is a weak
  answer: 0.3 where the map is sure, 0.1 where it is not. "Can't tell" is masked.
- **Kinds of road.** These come from `speed_road` and `speed_road_checked`, read the same way as for the brain
  (`brain_targets.road_answers`):
  - a step inside a mark with an end answers 1;
  - a step outside the marks answers 0, but only before that kind's "checked to here";
  - elsewhere the step is masked.

  The marks of `nnslr tunnels` (reviewer `osm`) count like a person's.
- **Roadworks.** The same rules, from `speed_works` and `speed_works_checked`.
- **A person's removal.** A person who removes a mark says that stretch is not of that kind, so its steps answer 0 at
  weight 1 where nothing else answers, also past "checked to here" (`removed_marks`). Three cases are left out:
  - the removals of the labeler and of `nnslr tunnels`, which are their own undo;
  - a mark removed before it had an end, which was a start tapped by mistake;
  - the steps within 2 s (`EDGE_SECONDS`) of a mark of that kind that is there now, because moving a mark means
    removing it and setting it again.
- **The tunnel's weak "no".** Where the GPS has a fix and our road of the map is not a tunnel, the tunnel column
  answers 0 at weight 0.3 (`TUNNEL_FREE_WEIGHT`). "Our road" is the way `nnslr road-types` found with
  `map_context.our_road`. It applies only where nothing else answers, and never within 5 s (`TUNNEL_MARGIN_S`) of a
  tunnel mark of any reviewer or of a second whose road of the map is a tunnel. The map makes this label offline. It
  never reaches the car.
- **The labeler's own writes.** Everything with reviewer `net/<run>` is an answer of the brain, with no confirmation,
  but never an answer of this network nor of the road-type network: a network never learns to say again what it said.
  Here its marks are left out and their steps, with 2 s (`EDGE_SECONDS`) around them, masked (`own_steps`), so that
  neither their inside nor a "checked to here" past them teaches it its own word. Its layer of road types is read with
  `road_types.load_road_types(..., labeler=False)` here and in `road_type_data`, so both use the map's line under it,
  with a person's corrections over it.
- **One side alone.** When a kind, or the roadworks, has only one side in train (all inside or all outside), it leaves
  the loss of train and the log says so (`keep_both_sides`). A constant answer would teach nothing.

On 8 Nov 2026 the tunnels of the map had 1960 steps inside and none outside, because no drive had a "checked to here"
for tunnel. The weak "no" gives the tunnel head its other side. The other kinds and the roadworks still need a "checked
to here", or a person's removals, on a few drives in Speed review before they learn.

## Split

The labeler uses the brain's frozen split (`brain_targets.read_split`), so the places held out of the brain are held out
here too. A training window lies inside one stretch of train. No picture of validation or test runs in a training step,
nor in the memory of one.

## Measure (`result.json`)

The measure is made for each partition:

- **`types`.** The share of steps whose road type is right:
  - on all the steps that have a type (`right`);
  - on the steps a person set (`right_on_person`), beside `map_on_person`, the map's own type on the same steps.

  The steps a person set are where the map was looked at, so `map_on_person` is the number to beat. Most other answers
  are the map itself, which is also an input.
- **`kinds`, `works`.** Precision and recall at 0.5 on the steps that have an answer. The value is None where a
  partition has only one side.
- **`gate`.** Which heads may write into Speed review, and why the others may not (`earned_heads`). The heads are each
  kind, `works` and `type:<road type>`. It is measured on validation, on the answers of full weight only: a person's,
  or a mark of `nnslr tunnels`, but never the map's weak answers.
  - A kind or the roadworks earns its writes with at least 120 steps inside and 120 outside (`GATE_MIN_ANSWERS`, a
    minute of steps each). It must also be sure (0.9) on at least 20 steps (`GATE_MIN_SAID`) and right on at least
    90% of those (`GATE_PRECISION`).
  - A road type is measured on the steps a person set. It counts as sure where the network says that type at 0.9
    while the map says another, because that is where its layer changes what the brain learns.

  The gate is saved in the checkpoint (`earned`). A checkpoint made before the gate existed writes nothing.

## Chances and Prepare Speed review

`road-labeler-predict` writes the chances to `derived/road_labeler/<route>.jsonl`:

- a first line `{run, step_hz, road_types, kinds, made}`;
- then one line a step: `{ns, segment, decoded, frame_key, partition, road_type, road_type_p, map_type, kinds: {kind:
  chance}, works}`.

Then it prepares Speed review on each drive (`road_labeler.prepare_drive`), unless `--no-marks`. Only the heads that
earned it write (`writable`, `held_back`):

- they were trained with both sides (`trained` in the checkpoint). A head that did not learn says chances from its
  first weights;
- they passed the gate of validation (`earned` in the checkpoint, the `gate` of `result.json`).

A head held back writes nothing; its chances are still in the chances file. The log and each drive's summary name it
with why (`held_back`). What it writes, with reviewer `net/<run>`:

- **Kinds of road and roadworks.** A mark is a stretch where the chance stays at 0.9 or more (`SURE`), at least 2 s
  long, with stretches less than 3 s apart joined. A hole in time (a minute without its encode index or its frames)
  ends a stretch, so a mark never spans what the network did not see. A mark never goes over a person's mark or a mark
  of `nnslr tunnels`, nor where a person removed a mark, nor before that kind's "checked to here". A removal counts as
  a person's word as for the answers (`road_types.person_removals`): a mark with an end, removed by a person, never the
  undo of the labeler or of `nnslr tunnels`.
- **Road types.** A layer of its own, `derived/road_labeler_types/<route>.jsonl`: a first line `{run, made, types}` (the
  road types whose head passed the gate), then `{frame_key, road_type, p}` a frame. A frame is in it when the network's
  most likely type is one of those, at 0.9 or more (`TYPE_SURE`), in a run of at least 5 frames in a row
  (`TYPE_MIN_FRAMES`, `layer_types`), also where it agrees with the map (the line then says whose it is). Elsewhere the
  map shows through. `road_types.load_road_types` reads the map, the layer over it, a person's corrections over both,
  and gives each line its `source` (`map`, `net/<run>` or the person); `typed_by_person` in the brain is true only for
  a person. A correction with reviewer `net/<run>` in `annotations/road_types` (written before 9 Nov 2026) is ignored.

**A newer run replaces the older one.** Every mark of the labeler that no person kept comes out, of any run and any
kind, also of a head the new run holds back (so a head that lost its gate leaves nothing behind). A mark is kept by a
person when it starts before its kind's "checked to here": a person went past it and left it. A kept mark, a person's
mark and a mark of `nnslr tunnels` are never touched. The new run's layer of road types replaces the older one whole,
also with no frame at all.

**One write.** A drive's removals, its new marks (a start and its end, or nothing of it) and a row `types_labeled` for
its layer (state: the run, note: how many frames of which types) go into `speed_history` as one write
(`SpeedReviewStore.batch`), at consecutive revisions, all or nothing. The layer's file is written last inside it, so a
layer that cannot be written takes the marks back with it. A drive changed meanwhile is tried again, up to 3 times.
Preparing again with the same run, and the same answers, changes nothing.

**What a Prepare wrote.** `<run>/prepared.json`: `{run, at, drives, failed, writes (the heads that wrote), heads: {head:
{written, removed, kept, left}, road_types}}` (`removed`: the older marks taken out; `kept`: the labeler's marks a person
kept; `left`: a sure stretch left out for a person's word or refused by the store; `road_types`: frames of its layer).
The run prepared last is the one Speed review shows. It is written again after each drive prepared, so a Prepare stopped
part-way (also by the portal's Stop) says what it wrote, and only once a drive is prepared: a Prepare that prepared no
drive (each one failed or had no scan) leaves the run Speed review shows as it was.

A dry run never writes into Speed review: `--dry-run`, or any checkpoint made by a dry run. Its chances go to
`<run>/chances/`, and it writes no `prepared.json`.

## Portal

The Road labeler card is at the bottom of Models (`/models`, `web/src/control/RoadLabelerCard.vue`).

- **Dry run** and **Train**: `helpers/road_labeler.sh train`, with `--dry-run` for the first. The panel action is
  `road_labeler_train`, its one option `dry` (`"on"`). Both ask first. It needs `.venv-gpu` and at least one drive with its
  logs read and its frames; it needs no confirmed sign.
- **Prepare Speed review** on a run with its checkpoint, and its own **Dry run**: `helpers/road_labeler.sh predict
  --run-dir data/runs/<run> --all`. The panel action is `road_labeler_predict`, with the options `labeler` (the run's
  folder name, which must hold `road-labeler.pt` and the run's `result.json`), `dry` and `marks` (`"off"` adds
  `--no-marks`: the box **Chances only**, nothing written into Speed review). It writes every head that passed its
  gate: the kinds of road, the roadworks and the layer of road types. There is no terminal-only write.
- Any other option, or one that is not a string (a dry run sent as `true`), is refused with 400 and nothing starts.
- **The GPU.** A real run is refused while another process has the GPU: the server answers 409 (`gpu_taken`), and the
  helper checks again before it starts (`nvidia-smi`: another compute process, or more than `NNSLR_GPU_BUSY_PERCENT`, 10,
  percent of its memory taken), with one line and exit code 3. A dry run is never refused for the GPU: it runs on the
  CPU, so the **Dry run** buttons stay usable on a busy GPU, and **Train** and **Prepare Speed review** are greyed.
- **Progress** (`nnslr_tools.job_progress`), from the helper's own lines:
  - for train, Check the GPU, Read the drives and their pictures (drive k of N, from `drives: N` and one
    `<route>: features of ...` or `left out` line a drive), then Train and measure (the `road_labeler epoch` lines, with
    a curve of `train_loss`, `val_types` and `val_on_person`);
  - for predict, Check the GPU, then Prepare Speed review drive by drive (one `{"route_id": ...}` line a drive).
  - A busy GPU shows as the reason the job failed.
- **The runs** (`GET /api/panel/road-labeler`, `nnslr_tools.labeler_runs`, files only). The newest 8 runs with a
  `result.json` of the road labeler, each with:
  - its drives, steps and epochs;
  - the road type on validation: right overall, and on the steps a person set beside the map's share there;
  - the road types it writes (`types.writing`), or that none passed its gate and the map stays;
  - each kind of road and the roadworks: precision and recall on validation, and whether it writes or why it is held
    back. It is the same rule as `road-labeler-predict` (`held_back` of the run's train sides and gate). A dry run never
    writes;
  - `prepared`: what its last Prepare Speed review wrote, head by head (`prepared.json`), and `current` on the run
    prepared last, the one Speed review shows ("in Speed review" on the card).
- **Speed review.** The Road type lane shows the type as the brain takes it: light where it is the map's, a mid tone
  where it is the labeler's, solid where a person set it; the lane's name and the tag say whose the stretch shown is
  (map, net, yours), with the run on the labeler's. There is nothing to confirm: a person sets a type only where it is
  wrong.

## Commands

```
helpers/road_labeler.sh train --dry-run --route R      # CPU, a tiny encoder, the first 48 steps of 1 or 2 drives
helpers/road_labeler.sh train                          # GPU: every scanned drive with frames, 30 epochs
helpers/road_labeler.sh predict --run-dir data/runs/<run> --all --no-marks   # the chances only
helpers/road_labeler.sh predict --run-dir data/runs/<run> --all              # Prepare Speed review: every head earned
```

## What is left

- The portal's buttons, the runs' endpoint and the source of each road type in Speed review go live after a restart of
  `speedlimits-web`.
- A first real training on the V100.
