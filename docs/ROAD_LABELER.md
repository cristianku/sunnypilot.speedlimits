# The road labeler

## What it is for

The road labeler prepares the labels of Speed review so that a person has less to fix. For every moment of a drive it
says three things:

- the road type: one of `road_types.ROAD_TYPES`, which are town, rural, expressway, motorway, ramp, parking, zone30,
  garage and merge;
- the kinds of road: tunnel and the other kinds of `speed_road_kinds` (narrow, country, gravel, potholes...);
- whether the car is inside roadworks, and so where they start and end.

It is an offline network. It never runs on the car and is never an input of the car. It writes proposals, Speed review
shows them, a person fixes what is wrong, and the brain learns from Speed review. Because it is offline, the
OpenStreetMap map is one of its inputs. The rule "vision never reads the map" (AGENTS.md) is about the car, and the car
never sees this network.

It is started only on request:

- `helpers/road_labeler.sh train`
- `helpers/road_labeler.sh predict`

These wrap `nnslr road-labeler-train` and `nnslr road-labeler-predict`. There is no portal button and no Models-tab
entry yet.

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
| answers | a road type at every step that has one; a kind of road or the roadworks only inside a mark, or before that kind's "checked to here" | No drive has a "checked to here" today, so no kind and no roadworks can be learned yet (see below). |

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
- **The labeler's own marks.** Marks with reviewer `net/<run>` are never an answer of its own training. The only
  exception is a mark that ends before that kind's "checked to here", because a person went through it and kept it.
  Its own road-type corrections are never an answer; the map's line under them is used instead.
- **One side alone.** When a kind, or the roadworks, has only one side in train (all inside or all outside), it leaves
  the loss of train and the log says so (`keep_both_sides`). A constant answer would teach nothing.

Today the tunnels of the map have 1960 steps inside and none outside, because no drive has a "checked to here" for
tunnel. So the tunnel head, the other kinds and the roadworks do not learn yet, and only the road type learns. A
"checked to here" on a few drives in Speed review is what turns them on.

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

## Proposals and writes

`road-labeler-predict` writes `derived/road_labeler/<route>.jsonl`:

- a first line `{run, step_hz, road_types, kinds, made}`;
- then one line a step: `{ns, segment, decoded, frame_key, partition, road_type, road_type_p, map_type, kinds: {kind:
  chance}, works}`.

Where the network is sure, it also writes into Speed review, as `nnslr tunnels` does. The reviewer is `net/<run>`. It
never writes over a person's mark, nor where a person removed a mark, nor before that kind's "checked to here".

Only the heads trained with both sides write (`trained` in the checkpoint). A kind, or the roadworks, that left the loss
of train still has its chances in the proposals, but never a mark: a head that did not learn says chances from its first
weights.

- **Kinds of road.** These are written by default. A mark is a stretch where the chance stays at 0.9 or more (`SURE`),
  at least 2 s long, with stretches less than 3 s apart joined. A hole in time (a minute without its encode index or
  its frames) ends a stretch, so a mark never spans what the network did not see. `--no-marks` turns this off.
- **Roadworks.** These are written only with `--write`, by the same rule.
- **Road types.** These are written only with `--write`, and only on map lines no person set. The network has to say
  another type than the map's at 0.9 or more for at least 5 frames in a row. The type is kept with
  `speed_review.set_road_type`, as a correction in `annotations/road_types`.

**Caution on `--write`.** `road_types.corrected` marks every correction as reviewed. So the brain's `type_head` and the
road-type network take a type the labeler wrote as a person's answer, because their readers do not look at the
reviewer. Until `road_types` leaves `net/` corrections out, write road types only on drives a person will go through.

A dry run never writes into Speed review: `--dry-run`, or any checkpoint made by a dry run. Its proposals go to
`<run>/proposals/`.

## Commands

```
helpers/road_labeler.sh train --dry-run --route R      # CPU, a tiny encoder, the first 48 steps of 1 or 2 drives
helpers/road_labeler.sh train                          # GPU: every scanned drive with frames, 30 epochs
helpers/road_labeler.sh predict --run-dir data/runs/<run> --all --no-marks   # the proposals only
helpers/road_labeler.sh predict --run-dir data/runs/<run> --route R          # with the sure marks
```

## What is left

- A Models-tab entry and a portal button. Every operation has to be startable from the portal, and this one is not yet.
- Showing the proposals in Speed review.
- A filter in `road_types` for the `net/` reviewer, so that `--write` stops looking like a person.
- A first real training on the V100.
