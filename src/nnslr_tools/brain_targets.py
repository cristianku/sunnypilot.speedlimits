"""From the drives to the brain's training: what every frame of a drive teaches, and what it may look at.

What it is for. The brain learns the speed to hold from the driver: at every frame his speed as the dashboard showed it,
unless a person blocked it in Speed review (annotations/speed_review.sqlite), and the next limit with the metres to it from
the signs our networks read, with what a person confirmed in Sign review (annotations/review.sqlite) over them
(reviewed_signs), passed where Sign distance says. None of this
changes anything by itself. It reaches the network only as an answer on a frame: the training compares the answer with what
the network said, and that error moves the weights (brain_net.brain_loss). This module makes those answers. It reads the
databases and writes nothing: no training, no GPU.

The drives (training_routes): every dashcam drive with a finished scan (drive_review.sqlite, its span known) and its
frames (derived/frames), always, each whole (his rule of 9 Nov 2026 evening: the switch "OK for brain training" of #9
went again, since the signs the brain is given are the networks' anyway and get better as they learn). A dashcam drive
(speed_review.dashcam_drive) is one where openpilot was passive in every carParams (sunnylink › Toggles › "Enable
sunnypilot" off, or dashcamOnly) and never engaged: he drove alone, so his speed is his. Any other drive is left out
whole, with its reason in dataset's log, and so is one whose logs show openpilot, MADS or controlsd on anyway
(speed_review.engaged_series). The sign classifier learns from every drive. A drive is labelled by being driven: one
without a block or a mark is a drive like the others.

A drive here is its frames in drive order, each on the comma's clock by the encode index (drive_logs.frame_time): labels
and frames meet on logMonoTime, never on a count of seconds. Two rates (dataset's rate):

- 1 Hz, the brains before 7 Nov 2026: the frames of derived/frames, the road-type network's pictures (PIL);
- RATE_HZ (4 a second): every reading of speedvisiond, as the brain steps on the car. The frames of a second are its 1 Hz
  frame and the three readings after it (VIDEO_STEP decoded frames apart), each with its partition and area, so that the
  frozen split stays the one it was frozen on. Their pictures are vision.BrainInput's of the same NV12 frame as the car's,
  made while the signs are read (road_type_signs, pictures=True): a frame without its picture has no answer.

The speed (speed_answers): the driver's speed at the frame (speed_review.driver_speed: the dashboard's, vEgoCluster or vEgo
+ the car's offset, in whole km/h, the same numbers Speed review draws), or the speed of a block that holds there. Class i
is i + 1 km/h, SPEED_CLASSES of them up to BRAIN_MAX_KPH, and no class 0: the car reads 0 as "unknown". Out of the loss,
each with its reason:

- no_speed: the logs give no speed there (before the first carState, a hole);
- stopped: the answer is below 1 km/h, which has no class;
- cruise: a machine held the speed, not his foot: openpilot's longitudinal (carControl.longActive), or the car's cruise
  (carState.cruiseState.enabled) with the gas not pressed (speed_review.machine_held). A block of his still answers there;
- left_out: inside a block with no speed, a person took those moments out;
- boundary: EDGE_SECONDS around the edge of a block where the answer jumps, since a tap is only so exact (a block with the
  dashboard's own speed at its edge is no jump, and the dashboard's speed changing from step to step is never one);
- no_frame: no picture, or a frame the encode index lacks.

The roadworks (works_answers), a target of 1 or 0 and a weight of 1 or 0, apart from the speed:

- inside a mark with an end: 1;
- from the start of the drive to speed_works_checked, outside the marks: 0, a person went through it;
- anywhere else: out of the loss. A stretch with no mark is not verified, it is not "no works";
- from the start of a mark with no end, and EDGE_SECONDS around every tap: out of the loss.

A mark is an answer whoever wrote it: a person, nnslr tunnels (reviewer 'osm') or the road labeler (reviewer net/<run>,
nnslr_tools.road_labeler, only where the head of that kind passed its gate), his rule of 9 Nov 2026: nobody confirms the
labeler's marks, a person corrects them where they are wrong (removes, moves, marks again), and a person's word always
wins because the labeler never writes over it. The labeler's own training leaves its marks out (road_labeler_data).

The two answers never touch: the end of works says nothing of the speed, and no code turns one into the other.

The road type (type_answers): one of road_types.ROAD_TYPES a frame (at 1 Hz: the readings of a second take the type of
its frame), as road_types.load_road_types gives it and Speed review shows it: the map's (nnslr road-types,
derived/road_types), the road labeler's over it where it is sure (derived/road_labeler_types, no confirmation), a
person's corrections over both (annotations/road_types, set in Speed review). typed_by_person is true only where a
person set it. It is an answer of a head only the training has (brain_net's type_head), never an input: the car never
reads the map, and here the map only makes labels offline, as AGENTS.md wants. With types (dataset's types, his rule of
8 Nov 2026, "only the drives where we have the road type"), a frame without a road type answers nothing at all: the
map's search not run on the drive, a road the map does not name (a car park nobody set), or "can't tell". All its
weights are 0, its speed out as no_road_type; it still runs through the memory, as the rest of the drive does.

The kinds of road (road_answers): the same answer, kind by kind (road_kinds: a narrow road, a country road, a gravel road,
potholes, and those a person added in Speed review), from that kind's marks and its own "checked to here". A frame can be of
several kinds at once: one column a kind, each with its target and its weight, and a kind nobody went through has none.

The limit ahead (reviewed_signs, ahead_from_signs), what the brain says beside the speed so that sunnypilot can slow down
for it as for the map's: the next limit (a class of the speed, or NONE_AHEAD) and the metres to it. The signs are the
boxes our networks in use read on every picture of the drive (brain_signs: cutter, classifier and applicability, at 1 a
second), with the person's over them, his rule of 8 Nov 2026: a frame confirmed in Sign review, on the drive's current
run (review.latest_report), never a site AI review still waiting for a person (review_db.NOT_WAITING_SQL), is his and
the networks' boxes there are not read; a sign he boxed is his, whatever road he gave it: the networks' sign of the
same kind (_kind) within SIGN_GAP_SECONDS of one of his boxes is his, with his reading on their pictures where it is of
our road (for a number, those where they read his number), left out where it is another road's. Of the networks' signs,
one the applicability network gave our road on any picture is ours, and a sign nobody could read within
SIGN_GAP_SECONDS of a number is that number seen from far. Only those of our own road:

- a maximum_speed or a variable_display with its number is a sign: boxes of one family at most SIGN_GAP_SECONDS apart are
  one, and two numbers among them are two signs;
- a cancellation is an end: a sign with no number, so the frames before it have no next limit to learn;
- a zone, a town's entry or exit, a motorway's or an expressway's start or end, a sign nobody could read, and a number
  with a plate (time, wet, present or unresolved) say a limit with no number of its own: their approach is masked, out of
  the loss;
- another road's signs, or the unresolved ones a person confirmed, are not ours: left out; a speed sign, an end or one
  of the above the applicability network could not give a road (unresolved) may be ours: its approach is masked.

A sign is passed at the zero a person marked in Sign distance (annotations/sign_distance/<route>.jsonl, the mark in effect
of the vision's run on it), else PASSAGE_SECONDS after its last box. The metres are the car's own, plain vEgo integrated
from the frame to the passing, never the dashboard's speed (it reads higher).

The next is the first sign whose number is not the limit in force, the number of the last sign passed: a repeat is no
next, since the car holds the driver's own speed to the km/h and would publish a repeat as a limit ahead. An end is
always a next. At the start, after an end or after a masked sign nothing is in force and the first sign of all is the
next. Within AHEAD_METRES of it: its number and its metres. No such sign and nothing masked within AHEAD_MARGIN_METRES:
none, and no metres. Anything else, between the two or before an end: out of the loss.

Both need what was read (covered: consecutive 1 Hz pictures confirmed by a person or read by the networks, a gap over
COVER_GAP_SECONDS breaks it), since a part nobody read may hide a sign: a number only where one span holds the frame,
the sign it answers and the sign that put the limit in force; none only where every frame of the AHEAD_MARGIN_METRES
ahead was read. Where a gap lies between the last sign and the frame, what is in force is not known and a repeat
cannot be told from a change: no number, and none only with no sign at all within the margin. Only on frames with an
answer of the speed. A drive the networks never read keeps only what a person confirmed. The networks' mistakes are
answers too: they get fewer as the networks get better, and every training reads the drives again with those in use.

The split (frame_partitions) is the frozen one, data/splits/speed-brain-split.json, never computed again: a drive of the
freeze joins it frame by frame through speed-brain-frames.json, and one whose frames changed is left out of the dataset with
a line in the log (FramesChanged), never put back by its areas; a later drive goes by the area of each frame, and an area
the freeze never saw is train, or buffer next to a held-out one or without any position. It is a mask on the loss: a drive
stays whole, so that its memory runs through it as on the car.

Works sites stay whole (keep_sites_whole): a site with a frame in test or validation, SITE_MARGIN_S around it, has no
frame in train, and neither has any other site on those areas, on another drive too. So the measure is on sites the
network never learned, approach and exit included.

The inputs (drive_inputs), one group each so that a comparison can leave one out, all made only from what comes at or
before the frame:

- signs, plate, plate_hours, plate_wet, hour, lead, country, pedals, speed, engaged, sign_distance, motion: what the car has,
  described in sunnypilot/speed_limits/brain_inputs.py. A drive not read yet has its signs unknown throughout. The hour and the country come from the last fix at or before the frame
  (merged['gps']), the country through the borders (gps.py): a frame never takes the country of a later fix. The lead comes
  from the series of the logs, which leave out a change under 2 m or 2 km/h (drive_extract): a little coarser than the car's.
  The pedals are in the last second, from the same carState (_pressed, as brain_inputs.PedalInputs does on the car).
  engaged goes through the car's own brain_inputs.EngagedInputs with the changes of sd_enabled, gas and brake in their
  order (engaged_inputs). sign_distance is the car's vision.SignDistance given every reading with its boxes and vEgo on a
  grid of SPEED_GRID_SECONDS, asked at each frame as speedvisiond asks it at each step (sign_distance_inputs); unknown on a
  drive read before the boxes were kept. motion goes through the car's brain_inputs.MotionInputs with the changes of the
  turn signals, the steering wheel and the yaw rates (drive_extract: steer within 1 degree, yaw within 0.5 deg/s), the values
  still held given again at each frame (motion_inputs); unknown on a drive whose logs were read before they were kept.
  road_signs is every other sign of those readings through the same SignInputs (road_sign_inputs): no kind picked out.

They are made by the car's own file, sunnypilot/speed_limits/brain_inputs.py, loaded here by its path (BRAIN_INPUTS): what
the network learns on is what it gets on the car, and the checkpoint keeps the file's hash.

pedals, speed and engaged are made but never learned (LEFT_OUT): the answer is the driver's own speed, so with his pedals
and the car's speed among the inputs the network learns to follow the car instead of the road, and the car's speed is what
the network itself decides on the car. The brain trained with them on 10 Nov 2026 did just that on the road: it followed
his speed. The measure still reads them (drive['inputs'], never packed): the frames with no gas, and the ones with the car's
speed far from the answer. The measure's own speed is the dashboard's (drive_logs.dashboard_speed: vEgoCluster, or vEgo +
3 km/h on a Peugeot log without it), in the units of the answers.

The sequences (drive_chunks, learning_pieces): the whole drive in order, state carried from piece to piece, so that
approach, crossing and exit are one sequence and the network sees only the past, as on the car. The pieces only bound the
gradient, and they never mix train with the rest: nnslr_tools.brain_train. The memory starts again after MEMORY_GAP_SECONDS
without a frame (memory_cuts), as on the car.

The hour in training (epoch_drives). Nearly every drive is by day, so the real hour says which drive a frame is from and
nothing of the speed, unless a plate gives it a meaning. So every epoch a drive takes a clock moved by a random number of
hours, one for all its frames, since on the car the clock runs on. A drive with a plate of hours on frames of train keeps its
real clock: there the hour is part of the answer. Validation and test always have the real one.

The night copies (night_copy). A plate of hours read outside them, with his speed above the number, says only that the
number does not hold now. Nothing shows when it does, and drives after 22:00 are few. So such a drive goes through the epoch
once more with its clock moved into the plate's hours, and on those frames the answer is the number: the same pictures, made
only here, nothing on the car, and no answer on roadworks. The plate's hours are the ones the reader read (plate_hours), the
night (NIGHT_FROM to NIGHT_TO) when it read none. Only frames of train with his speed above the number, a plate of hours, a
number that is a sign's (a multiple of SIGN_STEP_KPH up to BRAIN_MAX_KPH) and was not ended, and a known hour outside those
hours. A speed under the number has another reason, roadworks for one, and a copy would raise it. A speed within
SWAP_TOLERANCE_KPH of the number, on one such frame of the passage (the frames in a row with that number and plate), says it
holds then: such a plate has other hours, and the passage has no copy. The limits: a plate of other hours with none read,
driven outside them (7-17 at 19:00), gets a night answer that is wrong, a number stays until the next sign also where the
road was left, and the picture is of the day. The plate comes from the class the reader gives (vision.class_plate and
class_hours: maximum_speed_hours:<number>[:<from>-<to>], a sign with its plate in one box), so with a classifier without that
class there is no copy at all: plate is unknown on every frame, and the export says so (plates_in_train, hours_in_train).

A plate for wet roads has no copy. Whether it holds the network sees in the picture, and the picture of a copy is of a dry
road: the number as its answer would teach it to slow down on a dry one. So it learns only from the plate passed on a wet road
and on a dry one, with his speed of each. The plate comes from the class the reader gives (vision.class_plate:
maximum_speed_wet:<number>, a sign with its plate in one box), so with a classifier without that class it is unknown on every
frame, and the export says so (wet_in_train).

The numbers swapped (swap_passages, swapped). On a few drives the picture tells the road and a road seen in training tells its
speed, so the network has no reason to read the sign: the second run said 50 past a 60, and took no notice of the signs. So
every epoch half of the passages (the frames in a row with one number) whose answers all stay within SWAP_TOLERANCE_KPH of the
number go with that number changed to the other of its family (SWAPS), and those answers, and a next limit of that number,
move by the same difference: the same place with two limits, and only the sign tells which. A next limit of the other
number becomes a repeat of the one in force, no next: those frames say nothing of the limit ahead. A family is numbers the
same kind of road can have, so that no scene turns absurd. Only answers of train count, and a passage with a block in it stays: a person
said the speed there. A passage with an answer off the number stays as it is: the sign was wrong there, or he drove another
speed, and that too is for the brain to learn. A frame with no number in force keeps its answer, since the road alone says
it there, and an end read keeps the answer after it. Besides the number, the single readings equal to it change, on the
frames of the passage. Nothing of this reaches a measure, but brain_train's test_swapped and validation_swapped, which ask
exactly this of their frames.

What the training refuses (check_drives), so that nothing goes wrong without a word: a drive not read or read by another
reader than the car's today, since an unread drive looks like a drive with no sign and then "no reading" is learned as a
speed; a drive with no frame; at RATE_HZ, a drive without the brain's pictures.
"""
from __future__ import annotations

import bisect
import hashlib
import importlib.util
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

from nnslr_tools import brain_signs, drive_logs, drive_review, sign_distance
from nnslr_tools.drive_extract import GAP_SECONDS
from nnslr_tools.gps import frame_country, load_positions
from nnslr_tools.media import FRAME_SIZE
from nnslr_tools.review_db import NOT_WAITING_SQL, db_path as sign_review_path
from nnslr_tools.road_type_data import FRAMES_RELATIVE, frame_cells, frame_path
from nnslr_tools.road_types import ROAD_TYPES, load_road_types
from nnslr_tools.road_type_signs import (SEGMENT_SECONDS, STEPS_PER_SECOND, VIDEO_FPS, VIDEO_STEP, fingerprint, load_signs,
                                         load_vision, package_path, picture_path, pictures_ready)
from nnslr_tools.sign_reads import READ_INDEX_BASE
from nnslr_tools.speed_review import (FIRST_ROAD_KINDS, NS, dashcam_drive, db_path, driver_speed, engaged_series,
                                      machine_held)

# The brain's own table, never taken from Speed review: a class a km/h since 8 Nov 2026 (5 km/h before).
BRAIN_STEP_KPH = 1
BRAIN_MAX_KPH = 200
SIGN_STEP_KPH = 5  # a sign's number is a multiple of it: what a night copy can answer
SPEED_CLASSES = BRAIN_MAX_KPH // BRAIN_STEP_KPH

SPLIT_RELATIVE = Path('splits') / 'speed-brain-split.json'
SPLIT_FRAMES_RELATIVE = Path('splits') / 'speed-brain-frames.json'
EDGE_SECONDS = 2
SITE_MARGIN_S = 60
READ_COMMAND = 'nnslr road-type-signs --labelled'
IN_USE = object()  # reviewed_signs' stamp by default: the reading of Our networks in use counts
SIGN_PARTS = brain_signs.PARTS
NO_SPEED, STOPPED, BLOCKED_OUT, BOUNDARY, NO_FRAME, CRUISE, NO_TYPE = range(1, 8)
OUT_REASONS = {NO_SPEED: 'no_speed', STOPPED: 'stopped', BLOCKED_OUT: 'left_out', BOUNDARY: 'boundary', NO_FRAME: 'no_frame',
               CRUISE: 'cruise', NO_TYPE: 'no_road_type'}
NO_LABELS = {'blocks': [], 'works': [], 'checked_ns': None, 'road': [], 'road_checked': {}}
HELD_OUT = ('test', 'validation')
NIGHT_FROM, NIGHT_TO = 22, 6
SWAPS = {30: 40, 40: 30, 50: 60, 60: 50, 70: 80, 80: 70, 100: 120, 120: 100}
SWAP_TOLERANCE_KPH = 5  # an answer this near the number says it: a driver never holds a number to the km/h
RATE_HZ = float(STEPS_PER_SECOND)  # the brain's steps a second since 7 Nov 2026: every reading of speedvisiond
AHEAD_METRES = 100.0
AHEAD_MARGIN_METRES = 130.0
METRES_SCALE = 100.0  # the metres output is the metres over this (contract output.metres_scale)
NONE_AHEAD = SPEED_CLASSES  # the last class of next: no limit ahead
SPEED_GRID_SECONDS = 0.05  # vEgo to vision.SignDistance on this grid; the car gives it every carState, 100 a second

# The signs of Sign review (reviewed_signs).
LIMIT_FAMILIES = ('maximum_speed', 'variable_display')
MASKED_FAMILIES = ('zone', 'unreadable')
MASKED_SIGN_TYPES = ('town_entry', 'town_exit', 'motorway_start', 'motorway_end', 'expressway_start', 'expressway_end')
MASKED_PANELS = ('time', 'wet', 'present', 'unresolved')
PASSAGE_SECONDS = 0.5  # a sign's last box to its passing, without a mark (median of the marks on 0000016e: 0.4 s)
SIGN_GAP_SECONDS = 3.0  # boxes of one family at most this far apart are one sign
MARK_SECONDS = 5.0  # a zero mark this long after a sign's last box can still be its passing (sign_distance.WINDOW_S)
COVER_GAP_SECONDS = 1.5  # 1 Hz frames read (confirmed or by the networks) further apart than this break a span


def _load_brain_inputs(path: Path):
    """The car's brain_inputs.py (numpy only) as a module of its own."""
    spec = importlib.util.spec_from_file_location('comma_brain_inputs', path)
    if spec is None or spec.loader is None or not path.is_file():
        raise ValueError(f"no {path}: the car's file of the brain's inputs, in the clone of the branch speed-limits")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BRAIN_INPUTS_PATH = package_path() / 'brain_inputs.py'
BRAIN_INPUTS = _load_brain_inputs(BRAIN_INPUTS_PATH)
VISION = load_vision(package_path())
COUNTRIES = BRAIN_INPUTS.COUNTRIES
INPUT_SIZES = BRAIN_INPUTS.INPUT_SIZES
# Groups the car still makes that no brain learns from: engaged (openpilot on or off, 7 Nov 2026) only confused the
# network, his call the same night; pedals and speed (the gas, the brake and vEgo, 10 Nov 2026): the brain trained with
# them followed his speed on the road, his call that day. A brain built before keeps them: its groups are in its checkpoint
# and contract, and Models no longer lets one that takes the driver be chosen (model_roles.DRIVER_GROUPS).
LEFT_OUT = ('engaged', 'pedals', 'speed')
pack_inputs = BRAIN_INPUTS.pack_inputs


class FramesChanged(ValueError):
    """A drive of the frozen split whose frames are not the ones it was frozen on."""


def speed_class(kph):
    """The class of a speed in km/h, or of an array of them: rounded to the table's step, at most SPEED_CLASSES; -1 (no
    class) below one step, and for None or NaN."""
    if kph is None:
        return -1
    if np.ndim(kph) == 0:
        kph = float(kph)
        return int(min(max(round(kph / BRAIN_STEP_KPH), 1), SPEED_CLASSES)) - 1 if kph >= BRAIN_STEP_KPH else -1
    kph = np.asarray(kph, dtype=np.float64)
    found = np.clip(np.rint(np.nan_to_num(kph, nan=0.0) / BRAIN_STEP_KPH), 1, SPEED_CLASSES) - 1
    return np.where(kph >= BRAIN_STEP_KPH, found, -1).astype(np.int64)


def class_kph(index):
    """The km/h of a class, or of an array or a tensor of them."""
    return (index + 1) * BRAIN_STEP_KPH


# Why a scanned drive is not one the brain learns from (training_routes' left_out).
SUNNYPILOT_ON = 'sunnypilot was on (no dashcam drive)'


def _connect_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def read_labels(root: Path) -> dict[str, dict]:
    """{route: {'blocks', 'works', 'checked_ns', 'road', 'road_checked'}} of every drive with a block or a mark of Speed
    review (road: the marks of the kinds of road, each with its kind; road_checked: {kind: end_ns}). Read only: this
    database holds what only a person can write. A file of before 8 Nov 2026 has no blocks yet, nor kinds of road."""
    path = db_path(root)
    if not path.is_file():
        return {}
    connection = _connect_read_only(path)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        labels: dict[str, dict] = {}

        def drive(route: str) -> dict:
            return labels.setdefault(route, {'blocks': [], 'works': [], 'checked_ns': None, 'road': [], 'road_checked': {}})

        for table, key in (('speed_blocks', 'blocks'), ('speed_works', 'works'), ('speed_road', 'road')):
            if table in tables:
                for row in connection.execute(f'SELECT * FROM {table} ORDER BY route, start_ns'):
                    drive(row['route'])[key].append(dict(row))
        if 'speed_works_checked' in tables:
            for row in connection.execute('SELECT * FROM speed_works_checked'):
                drive(row['route'])['checked_ns'] = row['end_ns']
        if 'speed_road_checked' in tables:
            for row in connection.execute('SELECT * FROM speed_road_checked'):
                drive(row['route'])['road_checked'][row['kind']] = row['end_ns']
        return labels
    finally:
        connection.close()


def road_labels(root: Path) -> dict[str, str]:
    """{kind: label} of the kinds of road of Speed review in their order, read only. A file without them yet has the first
    ones (speed_review.FIRST_ROAD_KINDS)."""
    path = db_path(root)
    if path.is_file():
        connection = _connect_read_only(path)
        try:
            if connection.execute("SELECT 1 FROM sqlite_master WHERE name = 'speed_road_kinds'").fetchone():
                return {row['kind']: row['label'] for row in
                        connection.execute('SELECT kind, label FROM speed_road_kinds ORDER BY position, kind')}
        finally:
            connection.close()
    return dict(FIRST_ROAD_KINDS)


def road_kinds(root: Path) -> tuple[str, ...]:
    """The kinds of road of Speed review in their order: the columns of road_answers."""
    return tuple(road_labels(root))


def _dashcam_spans(root: Path, left_out: dict | None = None) -> dict[str, tuple[int, int]]:
    """{route: (t0_ns, end_ns)} of every dashcam drive (speed_review.dashcam_drive) with a finished scan (its start and its
    end on the comma's clock, in drive_review.sqlite, read only) and its frames (derived/frames). left_out, when given,
    gets SUNNYPILOT_ON for each scanned drive that is no dashcam drive, by route."""
    path = drive_review.db_path(root)
    if not path.is_file():
        return {}
    connection = _connect_read_only(path)
    try:
        rows = connection.execute('SELECT route, summary FROM scans').fetchall()
    finally:
        connection.close()
    found = {}
    for row in rows:
        try:
            summary = json.loads(row['summary']) if row['summary'] else None
        except ValueError:
            summary = None
        span = summary.get('span_ns') if isinstance(summary, dict) else None
        if summary is None or summary.get('t0_ns') is None or not span or span[-1] is None:
            continue
        if not dashcam_drive(summary):
            if left_out is not None:
                left_out[row['route']] = SUNNYPILOT_ON
            continue
        if any((Path(root) / FRAMES_RELATIVE / row['route']).glob('*/frames.jsonl')):
            found[row['route']] = (summary['t0_ns'], span[-1])
    return found


def training_routes(root: Path, left_out: dict | None = None) -> list[str]:
    """The drives the brain learns from, in order, each whole: every dashcam drive with a finished scan and its frames
    (_dashcam_spans). left_out, when given, gets SUNNYPILOT_ON for each scanned drive that is no dashcam drive, by
    route."""
    return sorted(_dashcam_spans(root, left_out))


def training_counts(root: Path) -> dict:
    """What the brain learns from today, from the scans alone (no dataset): {drives, hours: their length from the start
    of their logs to the end, sunnypilot_on: the scanned drives with sunnypilot on}."""
    left_out: dict[str, str] = {}
    spans = _dashcam_spans(root, left_out)
    return {'drives': len(spans), 'hours': round(sum(end - start for start, end in spans.values()) / NS / 3600, 2),
            'sunnypilot_on': sum(reason == SUNNYPILOT_ON for reason in left_out.values())}


def training_counts_text(counts: dict) -> str:
    """training_counts for a log: the dashcam drives and their hours, the drives with sunnypilot on left out."""
    return (f"Dashcam drives: {counts['drives']}, {counts['hours']:.1f} h; {counts['sunnypilot_on']} "
            f"drive{'' if counts['sunnypilot_on'] == 1 else 's'} with sunnypilot on, left out whole.")


def speed_answers(ns: np.ndarray, driver_kph: list, blocks: list[dict],
                  held: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(class per step, -1 without an answer; why a step is out of the loss, 0 when it is in) at the times ns, given the
    driver's speed at each (speed_review.driver_speed, whole km/h or None), the blocks of the drive and where a machine
    held the speed (speed_review.machine_held; None: nowhere)."""
    ns = np.asarray(ns, dtype=np.int64)
    answer = np.array([-1 if kph is None else kph for kph in driver_kph], dtype=np.int64).reshape(len(ns))
    out = np.where(answer < 0, NO_SPEED, 0)
    if held is not None:
        machine = np.asarray(held, dtype=bool) & (out == 0)
        out[machine], answer[machine] = CRUISE, -1
    known = ns >= 0
    ordered = sorted(blocks, key=lambda block: block['start_ns'])
    for block in ordered:
        inside = known & (ns >= block['start_ns']) & (ns < block['end_ns'])
        if block['kph'] is None:
            out[inside], answer[inside] = BLOCKED_OUT, -1
        else:
            out[inside], answer[inside] = 0, block['kph']
    out[(out == 0) & (answer < 1)] = STOPPED
    # What an edge separates: the answer of the last step before it and of the first at or after it, None when out.
    value = np.where(np.isin(out, (0, STOPPED)), answer, -1)
    steps = np.flatnonzero(known)
    steps = steps[np.argsort(ns[steps], kind='stable')]
    times = ns[steps]
    for edge in sorted({block[key] for block in ordered for key in ('start_ns', 'end_ns')}):
        at = int(np.searchsorted(times, edge, side='left'))
        if at == 0 or at == len(times) or value[steps[at - 1]] == value[steps[at]]:
            continue
        near = known & (np.abs(ns - edge) < EDGE_SECONDS * NS) & (out == 0)
        out[near] = BOUNDARY
    classes = speed_class(answer)
    classes[out != 0] = -1
    return classes, out


def works_answers(ns: np.ndarray, works: list[dict], checked_ns: int | None) -> tuple[np.ndarray, np.ndarray]:
    """(target, weight) float32 per frame at the times ns: see the module's text."""
    target = np.zeros(len(ns), dtype=np.float32)
    weight = np.zeros(len(ns), dtype=np.float32)
    if checked_ns is not None:
        weight[ns < checked_ns] = 1
    edges = []
    for mark in works:
        edges.append(mark['start_ns'])
        if mark['end_ns'] is None:
            weight[ns >= mark['start_ns']] = 0
            continue
        edges.append(mark['end_ns'])
        inside = (ns >= mark['start_ns']) & (ns < mark['end_ns'])
        target[inside] = 1
        weight[inside] = 1
    for edge in edges:
        weight[np.abs(ns - edge) < EDGE_SECONDS * NS] = 0
    return target, weight


def type_answers(root: Path, route: str, frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(class of road_types.ROAD_TYPES per frame, -1 without one; a person set it there) at the frames of a drive, from
    load_road_types: see the module's text. A reading takes the type of the 1 Hz frame of its second."""
    types = {line['frame_key']: line for line in load_road_types(root, route)}
    found = np.full(len(frames), -1, dtype=np.int64)
    person = np.zeros(len(frames), dtype=bool)
    for at, frame in enumerate(frames):
        line = types.get(f"{frame['segment']}/{frame['index']}")
        if line is not None and line.get('road_type') in ROAD_TYPES:
            found[at] = ROAD_TYPES.index(line['road_type'])
            person[at] = line.get('reviewed') is True
    return found, person


def road_answers(ns: np.ndarray, road: list[dict], checked: dict[str, int], kinds: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """(target, weight) float32 (frames, kinds) at the times ns: works_answers of each kind, from its marks and its own
    "checked to here". A mark of a kind not in kinds is not read."""
    target = np.zeros((len(ns), len(kinds)), dtype=np.float32)
    weight = np.zeros((len(ns), len(kinds)), dtype=np.float32)
    for column, kind in enumerate(kinds):
        target[:, column], weight[:, column] = works_answers(ns, [mark for mark in road if mark['kind'] == kind],
                                                             checked.get(kind))
    return target, weight


def drive_frames(root: Path, route: str, merged: dict) -> list[dict]:
    """The 1 Hz frames of a drive in drive order: key, segment, index, seconds (from the start of the drive), decoded (the
    frame of the segment's video), ns (the exposure start; None when the encode index lacks it), latitude, longitude,
    image, present."""
    index = drive_logs.frame_index(merged)
    positions = load_positions(root, route)
    frames = []
    for path in sorted((Path(root) / FRAMES_RELATIVE / route).glob('*/frames.jsonl')):
        for text in path.read_text(encoding='utf-8').splitlines():
            if not text.strip():
                continue
            row = json.loads(text)
            segment, second = row['segment_index'], row['output_index']
            key = f'{segment}/{second}'
            image = frame_path(root, route, segment, second)
            gps = positions.get(key, {})
            frames.append({'route': route, 'key': key, 'segment': segment, 'index': second,
                           'seconds': float(segment * SEGMENT_SECONDS + second), 'decoded': row['decoded_frame_index'],
                           'ns': drive_logs.frame_time(index, segment, row['decoded_frame_index']),
                           'latitude': gps.get('latitude'), 'longitude': gps.get('longitude'), 'image': str(image),
                           'present': image.is_file()})
    frames.sort(key=lambda frame: (frame['segment'], frame['index']))
    return frames


def _exposures(merged: dict) -> dict[int, dict[int, int]]:
    """{segment: {decoded frame: exposure start}} of the encode index, as drive_logs.frame_time reads it."""
    found = {}
    for _first, segment, sof, ids in drive_logs.frame_index(merged):
        times = found.setdefault(segment, {})
        for decoded, t in zip(ids, sof):
            times.setdefault(decoded, t)
    return found


def brain_frames(root: Path, frames: list[dict], partitions: list[str], cells: list, merged: dict,
                 record: dict | None) -> tuple[list[dict], list[str], list]:
    """The frames at RATE_HZ from the 1 Hz ones: each 1 Hz frame and the STEPS_PER_SECOND - 1 readings after it, with its
    partition and area; step (0 for the 1 Hz frame), picture (the row of the segment's brain pictures) and present (that
    picture is there)."""
    exposures, counts = _exposures(merged), (record or {}).get('pictures') or {}
    files: dict[int, bool] = {}
    found, parts, places = [], [], []
    for frame, partition, cell in zip(frames, partitions, cells):
        segment = frame['segment']
        if frame['decoded'] % VIDEO_STEP:
            raise ValueError(f"{frame['route']} {frame['key']}: a 1 Hz frame off the readings ({frame['decoded']})")
        if segment not in files:
            files[segment] = picture_path(root, frame['route'], segment).is_file()
        for step in range(STEPS_PER_SECOND):
            decoded = frame['decoded'] + step * VIDEO_STEP
            row = decoded // VIDEO_STEP
            found.append({**frame, 'key': f'{segment}/{decoded / VIDEO_FPS:g}', 'decoded': decoded, 'step': step,
                          'seconds': segment * SEGMENT_SECONDS + decoded / VIDEO_FPS,
                          'ns': exposures.get(segment, {}).get(decoded), 'picture': row,
                          'present': files[segment] and row < int(counts.get(str(segment)) or 0)})
            parts.append(partition)
            places.append(cell)
    return found, parts, places


def read_split(root: Path) -> dict:
    """The frozen split with the frame keys of its drives."""
    split = json.loads((Path(root) / SPLIT_RELATIVE).read_text(encoding='utf-8'))
    frames = json.loads((Path(root) / SPLIT_FRAMES_RELATIVE).read_text(encoding='utf-8'))
    if frames['split_sha256'] != split['sha256']:
        raise ValueError('the frames of the split are not the ones it was frozen on')
    return {**split, 'frame_keys': frames['frames']}


def _cell_partition(cell: tuple[int, int] | None, cells: dict[str, str]) -> str:
    """The partition of an area the way a drive after the freeze takes it."""
    if cell is None:
        return 'buffer'
    found = cells.get(f'{cell[0]},{cell[1]}')
    if found is not None:
        return found
    around = (cells.get(f'{cell[0] + a},{cell[1] + b}') for a in (-1, 0, 1) for b in (-1, 0, 1))
    return 'buffer' if any(name in HELD_OUT for name in around) else 'train'


def frame_partitions(route: str, frames: list[dict], split: dict) -> list[str]:
    """The partition of each frame: see the module's text."""
    frozen = split['partitions'].get(route)
    if frozen is None:
        return [_cell_partition(cell, split['cells']) for cell in frame_cells(frames)]
    by_key = dict(zip(split['frame_keys'][route], frozen))
    missing = [frame['key'] for frame in frames if frame['key'] not in by_key]
    if missing or len(by_key) != len(frames):
        raise FramesChanged(f'{route}: the frames changed since the split was frozen ({len(frames)} now, {len(by_key)} then)')
    return [by_key[frame['key']] for frame in frames]


def _last_at(points: list, ns: np.ndarray) -> list:
    """The value of a step series at each time: its last point at or before it, None before the first."""
    times = [point[0] for point in points]
    found = []
    for t in ns:
        at = bisect.bisect_right(times, int(t)) - 1
        found.append(points[at][1] if at >= 0 else None)
    return found


def _pressed(points: list, ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pressed at some moment of the last second, known) of a step series of booleans at each time."""
    times = [point[0] for point in points]
    pressed = np.zeros(len(ns), dtype=np.float32)
    known = np.zeros(len(ns), dtype=np.float32)
    for row, t in enumerate(ns):
        last = bisect.bisect_right(times, int(t)) - 1
        first = max(bisect.bisect_right(times, int(t) - NS) - 1, 0)
        values = [points[at][1] for at in range(first, last + 1)]
        known[row] = float(any(value is not None for value in values))
        pressed[row] = float(any(value is True for value in values))
    return pressed, known


def engaged_inputs(series: dict, ns: np.ndarray) -> np.ndarray:
    """The engaged group (frames, size): the car's EngagedInputs given the changes of gas, brake and sd_enabled in their
    order, the pedals before openpilot at the same moment as on the car, and asked at each frame. Unknown at a frame with no
    time."""
    events = sorted([(t, 0, name, value) for name in ('gas', 'brake') for t, value in series.get(name) or []] +
                    [(t, 1, 'enabled', value) for t, value in series.get('sd_enabled') or []], key=lambda event: event[:2])
    engaged, pedals = BRAIN_INPUTS.EngagedInputs(), {'gas': False, 'brake': False}
    rows = np.zeros((len(ns), INPUT_SIZES['engaged']), dtype=np.float32)
    at = 0
    for row in sorted(np.flatnonzero(ns >= 0), key=lambda row: ns[row]):
        while at < len(events) and events[at][0] <= ns[row]:
            t, _, name, value = events[at]
            if name == 'enabled':
                engaged.add_enabled(t / NS, value is True)
            else:
                pedals[name] = value is True
                engaged.add_pedals(t / NS, pedals['gas'], pedals['brake'])
            at += 1
        rows[row] = engaged.row(ns[row] / NS)
    return rows


MOTION_CAR = ('blinker_l', 'blinker_r', 'steer', 'yaw_car')


def motion_inputs(series: dict, ns: np.ndarray) -> np.ndarray:
    """The motion group (frames, size): the car's MotionInputs given the changes of the turn signals, the steering wheel
    and the yaw rates in their order, and the values the log still holds again at each frame (the log keeps a change only,
    the car hears every message; where carState stops, its last message is GAP_SECONDS before the None). Unknown at a
    frame with no time."""
    events = sorted([(t, name, value) for name in (*MOTION_CAR, 'yaw_dev') for t, value in series.get(name) or []],
                    key=lambda event: event[0])
    motion = BRAIN_INPUTS.MotionInputs()
    held = dict.fromkeys((*MOTION_CAR, 'yaw_dev'))

    def give(t: float, car: bool, device: bool) -> None:
        if car and held['steer'] is not None:
            motion.add_car_state(t, held['steer'], held['blinker_l'] is True, held['blinker_r'] is True,
                                 math.radians(held['yaw_car'] or 0.0))
        if device and held['yaw_dev'] is not None:
            motion.add_device_motion(t, -math.radians(held['yaw_dev']), True)

    rows = np.zeros((len(ns), INPUT_SIZES['motion']), dtype=np.float32)
    at = 0
    for row in sorted(np.flatnonzero(ns >= 0), key=lambda row: ns[row]):
        while at < len(events) and events[at][0] <= ns[row]:
            t, name, value = events[at]
            held[name] = value
            if value is None and name in MOTION_CAR:  # drive_extract's None: GAP_SECONDS after the last carState
                motion.car_at = min(motion.car_at, t / NS - GAP_SECONDS['carState'])
            give(t / NS, name in MOTION_CAR, name == 'yaw_dev')
            at += 1
        give(ns[row] / NS, True, True)
        rows[row] = motion.row(ns[row] / NS)
    return rows


class _SpeedFeed:
    """vEgo (a step series [t_ns, km/h]) to a vision.SignDistance on a grid of SPEED_GRID_SECONDS from its first point."""

    def __init__(self, points: list):
        self.points, self.at = points, 0
        self.t = int(points[0][0]) if points else None

    def to(self, distance, until_ns: int) -> None:
        step = int(SPEED_GRID_SECONDS * NS)
        while self.t is not None and self.t <= until_ns:
            while self.at + 1 < len(self.points) and self.points[self.at + 1][0] <= self.t:
                self.at += 1
            kph = self.points[self.at][1]
            if kph is not None:
                distance.add_speed(self.t / NS, kph / 3.6)
            self.t += step


def sign_distance_inputs(ns: np.ndarray, merged: dict, readings: list | None, boxes: list | None) -> np.ndarray:
    """The sign_distance group (frames, size): see the module's text. Readings [[seconds, [[class, confidence], ...]], ...]
    and their boxes in the same order; at a frame with no time, or with no boxes, unknown."""
    rows = np.zeros((len(ns), INPUT_SIZES['sign_distance']), dtype=np.float32)
    if not readings or boxes is None or len(boxes) != len(readings):
        return rows
    exposures, seen = _exposures(merged), {}
    for (seconds, read), places in zip(readings, boxes):
        segment = int(seconds // SEGMENT_SECONDS)
        at = exposures.get(segment, {}).get(int(round((seconds - segment * SEGMENT_SECONDS) * VIDEO_FPS)))
        if at is not None:
            seen[int(at)] = [{'class': name, 'confidence': confidence, 'box': tuple(box)}
                             for (name, confidence), box in zip(read, places)]
    asked: dict[int, list[int]] = {}
    for row in np.flatnonzero(ns >= 0):
        asked.setdefault(int(ns[row]), []).append(int(row))
    distance = VISION.SignDistance(VISION.FOCAL_LENGTHS.get(FRAME_SIZE, 0.0))
    speeds = _SpeedFeed(sorted((merged.get('series') or {}).get('v_ego') or [], key=lambda point: point[0]))
    for t in sorted(set(seen) | set(asked)):
        speeds.to(distance, t)
        metres = distance.update(t / NS, seen.get(t, []), None)
        for row in asked.get(t, ()):
            rows[row] = BRAIN_INPUTS.sign_distance_row(metres)
    return rows


def metres_driven(points: list):
    """ns -> metres driven since the first point of vEgo (a step series [t_ns, km/h]), the speed held between points."""
    points = [point for point in sorted(points, key=lambda point: point[0]) if point[1] is not None]
    if not points:
        return lambda ns: np.zeros(np.shape(ns))
    times = np.array([point[0] for point in points], dtype=np.int64)
    speed = np.array([max(point[1], 0.0) / 3.6 for point in points], dtype=np.float64)
    before = np.concatenate([[0.0], np.cumsum(speed[:-1] * np.diff(times) / NS)])

    def at(ns):
        ns = np.asarray(ns, dtype=np.int64)
        last = np.maximum(np.searchsorted(times, ns, side='right') - 1, 0)
        return before[last] + speed[last] * np.maximum(ns - times[last], 0) / NS

    return at


def _sign_review(root: Path, route: str) -> tuple[str | None, list[dict]]:
    """(the drive's current run, its frames confirmed in Sign review that wait for no person), read only: no run without
    a report, nothing without the database."""
    path = sign_review_path(root)
    if not path.is_file():
        return None, []
    from nnslr_tools.review import latest_report  # the review's server: loaded only here
    try:
        run = latest_report(root, route).name
    except (ValueError, OSError):
        return None, []
    connection = _connect_read_only(path)
    try:
        rows = connection.execute('SELECT frame_key, output_index, payload, revision, updated_at FROM frame_reviews '
                                  'WHERE route_id = ? AND run_id = ? AND confirmed = 1' + NOT_WAITING_SQL,
                                  (route, run)).fetchall()
    finally:
        connection.close()
    return run, [dict(row) for row in rows]


def _zero_marks(root: Path, route: str, merged: dict) -> list[tuple[int, dict | None, dict]]:
    """[(t_ns, the vision's run it is on or None, line)] of the zero marks in effect of Sign distance on a drive, in time
    order, by Sign distance's own reading of its file (the runs with their windows, as its page has them)."""
    try:
        lines = sign_distance.load_marks(root, route)
    except sign_distance.SignDistanceError:
        return []
    if not lines:
        return []
    factor = sign_distance.FEET if (merged.get('meta') or {}).get('is_metric') is False else 1.0
    margin = int(sign_distance.WINDOW_S * NS)
    runs = [{**run, 'window': [(run['last_seen_ns'] or run['start_ns']) - margin,
                               (run['zero_ns'] or run['end_ns']) + margin]}
            for run in sign_distance.find_signs(merged, factor)]
    of_key = {run['key']: run for run in runs}
    current = sign_distance._current_marks(lines, sign_distance._resolve(lines, runs))
    return sorted(((line['t_ns'], of_key.get(key), line) for key, line in current.items()
                   if line['mark'] == 'zero' and type(line.get('t_ns')) is int), key=lambda item: item[0])


def _groups(boxes: list[tuple]) -> list[list[tuple]]:
    """Boxes (t_ns first) in time order, cut where SIGN_GAP_SECONDS pass without one: one sign each."""
    found: list[list[tuple]] = []
    for box in sorted(boxes, key=lambda box: box[0]):
        if found and box[0] - found[-1][-1][0] <= SIGN_GAP_SECONDS * NS:
            found[-1].append(box)
        else:
            found.append([box])
    return found


def _passings(groups: list[tuple[int, int]], zeros: list) -> list[int]:
    """When each sign of boxes from first to last was passed: see the module's text. A zero mark can be the passing of a
    sign from its first box to MARK_SECONDS after its last, on a run of the vision's that held it then; it goes to the
    sign whose last box + PASSAGE_SECONDS is nearest, and each sign takes the nearest of its own."""
    aims = [last + int(PASSAGE_SECONDS * NS) for _first, last in groups]
    taken: dict[int, list[int]] = {}
    for t, run, _line in zeros:
        could = [at for at, (first, last) in enumerate(groups)
                 if first <= t <= last + MARK_SECONDS * NS and
                 (run is None or run['start_ns'] <= aims[at] and
                  (run['zero_ns'] or run['end_ns']) + PASSAGE_SECONDS * NS >= first)]
        if could:
            taken.setdefault(min(could, key=lambda at: abs(t - aims[at])), []).append(t)
    return [min(taken[at], key=lambda t: abs(t - aim)) if at in taken else aim for at, aim in enumerate(aims)]


def _covered(times: list[int]) -> list[tuple[int, int]]:
    """(start_ns, end_ns) of the runs of confirmed 1 Hz frames with no gap over COVER_GAP_SECONDS."""
    spans: list[list[int]] = []
    for t in sorted(set(times)):
        if spans and t - spans[-1][1] <= COVER_GAP_SECONDS * NS:
            spans[-1][1] = t
        else:
            spans.append([t, t])
    return [(first, last) for first, last in spans]


def _kph(value) -> int | None:
    return value if type(value) is int and 1 <= value <= BRAIN_MAX_KPH else None


def _kind(box: dict):
    """What a box is when his signs and the networks' are told apart: the two families with a number are one."""
    family = box.get('family')
    if family in LIMIT_FAMILIES:
        return 'limit'
    return (family, box.get('sign_type')) if family == 'other_sign' else family


def reviewed_signs(root: Path, route: str, merged: dict, stamp: Any = IN_USE) -> dict:
    """{'signs': [(pass_ns, kph, None for an end)], 'covered': [(start_ns, end_ns)], 'masked': [(start_ns, end_ns)],
    'fingerprint', 'read_by'} of a drive from Sign review, the networks' reading (brain_signs) and Sign distance, read
    only: see the module's text. fingerprint: a hash of the rows, the reading and the marks it came from, None with
    neither; read_by: the networks that read the drive, None when they never did. stamp: the networks whose reading
    counts (brain_signs.networks_stamp; IN_USE: those in use, brain_signs.stamp_in_use)."""
    run, rows = _sign_review(root, route)
    network = brain_signs.load(root, route)
    if stamp is IN_USE:
        stamp = brain_signs.stamp_in_use(root)
    if network is not None and network['read_by'].get('networks') != stamp:
        # Read by other networks than those chosen, or they are not whole now: Sign review alone.
        network = None
    found = {'signs': [], 'covered': [], 'masked': [], 'fingerprint': None, 'read_by': None}
    if run is None and network is None:
        return found
    exposures = _exposures(merged)

    def when(payload: dict) -> int | None:
        try:
            return exposures.get(int(payload['segment_index']), {}).get(int(payload['decoded_frame_index']))
        except (TypeError, ValueError, KeyError):
            return None

    pictures, limits, ends, hidden = [], {}, [], {}

    def take(t: int, box: dict, unsure: bool) -> None:
        family = box.get('family')
        hide = family in MASKED_FAMILIES or (family == 'other_sign' and box.get('sign_type') in MASKED_SIGN_TYPES)
        if unsure:
            # A sign the applicability network could not give a road: its approach is out of the loss.
            if family in LIMIT_FAMILIES or family == 'cancellation' or hide:
                hidden.setdefault(('unsure', _kind(box)), []).append((t,))
        elif family in LIMIT_FAMILIES:
            limits.setdefault(family, []).append((t, _kph(box.get('value_kph')), box.get('supplementary_panel')))
        elif family == 'cancellation':
            ends.append((t,))
        elif hide:
            hidden.setdefault((family, box.get('sign_type')), []).append((t,))

    confirmed, his = set(), {}
    for row in rows:
        try:
            payload = json.loads(row['payload'])
        except (TypeError, ValueError):
            continue
        t = when(payload)
        if t is None:
            continue
        confirmed.add(t)
        if row['output_index'] < READ_INDEX_BASE:
            pictures.append(t)
        for box in payload.get('detections') or []:
            if not isinstance(box, dict):
                continue
            his.setdefault(_kind(box), []).append((t, box))
            if box.get('applicability') == 'own_road':
                take(t, box, False)
    # The networks' reading on the pictures he did not confirm, sign by sign (_groups). Their other signs say what they
    # are in their class (model_label "other_sign:town_entry"), not in a sign_type.
    theirs = {}
    for row in (network or {}).get('rows', ()):
        t = when(row)
        if t is None or t in confirmed:
            continue
        pictures.append(t)
        for box in row.get('detections') or []:
            if not isinstance(box, dict) or box.get('applicability') not in ('own_road', 'unresolved'):
                continue
            if box.get('family') == 'other_sign' and not box.get('sign_type'):
                box = {**box, 'sign_type': str(box.get('model_label') or '').partition(':')[2] or None}
            theirs.setdefault(_kind(box), []).append((t, box))
    gap = int(SIGN_GAP_SECONDS * NS)
    # Where a number was read, by them or by him on any road.
    numbers = np.array(sorted(t for t, _box in theirs.get('limit', []) + his.get('limit', [])), dtype=np.int64)

    def near(times: np.ndarray, group: list[tuple]) -> np.ndarray:
        return np.flatnonzero((times >= group[0][0] - gap) & (times <= group[-1][0] + gap))

    for kind, boxes in theirs.items():
        mine = sorted(his.get(kind, ()), key=lambda item: item[0])
        mine_t = np.array([t for t, _box in mine], dtype=np.int64)
        for group in _groups(boxes):
            close = near(mine_t, group)
            if len(close):
                # A sign he boxed within SIGN_GAP_SECONDS is his, whatever road he gave it. On his road their pictures
                # of it make it longer, with his reading: for a number, where they read his number.
                _t, his_box = mine[min(close, key=lambda at: max(group[0][0] - mine_t[at], 0, mine_t[at] - group[-1][0]))]
                if his_box.get('applicability') == 'own_road':
                    for t, box in group:
                        if kind != 'limit' or box.get('value_kph') == his_box.get('value_kph'):
                            take(t, his_box, False)
                continue
            if kind == 'unreadable' and len(near(numbers, group)):
                # A number too far to read, then read: the sign is the number's, its approach is not masked.
                continue
            # A sign the applicability network gave our road on any picture is ours; one it never could give is unsure.
            unsure = not any(box.get('applicability') == 'own_road' for _t, box in group)
            for t, box in group:
                take(t, box, unsure)
    zeros = _zero_marks(root, route, merged)
    signs, masked = [], []  # [(first, last, kph)], [(first, last)]
    for boxes in limits.values():
        for group in _groups(boxes):
            values = {kph for _t, kph, _panel in group if kph is not None}
            if not values or any(panel in MASKED_PANELS for _t, _kph, panel in group):
                masked.append((group[0][0], group[-1][0]))
                continue
            for kph in sorted(values):
                times = [t for t, value, _panel in group if value == kph]
                signs.append((times[0], times[-1], kph))
    signs += [(group[0][0], group[-1][0], None) for group in _groups(ends)]
    masked += [(group[0][0], group[-1][0]) for boxes in hidden.values() for group in _groups(boxes)]
    passes = _passings([(first, last) for first, last, _kph in signs] + masked, zeros)
    found['signs'] = sorted(((at, kph) for at, (_first, _last, kph) in zip(passes, signs)), key=lambda sign: sign[0])
    found['masked'] = sorted((first, at) for at, (first, _last) in zip(passes[len(signs):], masked))
    found['covered'] = _covered(pictures)
    read_by = (network or {}).get('read_by')
    found['read_by'] = {part: entry['id'] for part, entry in read_by['networks'].items()} if read_by else None
    seen = {'run': run, 'rows': sorted([row['frame_key'], row['revision'], row['updated_at']] for row in rows),
            'marks': [[line['_line'], line['sign'], line['mark'], t] for t, _run, line in zeros]}
    if read_by:
        seen['networks'] = [read_by['networks'], read_by['pictures']]
    found['fingerprint'] = hashlib.sha256(json.dumps(seen, sort_keys=True).encode()).hexdigest()[:16]
    return found


def ahead_from_signs(ns: np.ndarray, reviewed: dict, metres_at, asked: np.ndarray) -> tuple[np.ndarray, ...]:
    """(next class per frame, its weight, the metres to it over METRES_SCALE, their weight) at the times ns from the signs
    of reviewed_signs, on the frames asked (those with an answer of the speed): see the module's text."""
    ns = np.asarray(ns, dtype=np.int64)
    following = np.full(len(ns), NONE_AHEAD, dtype=np.int64)
    next_weight, metres, metres_weight = (np.zeros(len(ns), dtype=np.float32) for _ in range(3))
    rows = np.flatnonzero(np.asarray(asked, dtype=bool) & (ns >= 0))
    if not len(rows):
        return following, next_weight, metres, metres_weight
    t = ns[rows]
    here = np.asarray(metres_at(t), dtype=np.float64)

    def first_after(times: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(which, there is one, the metres to it) of the first of times (sorted) after each frame."""
        times = np.array(times, dtype=np.int64)
        which = np.searchsorted(times, t, side='right')
        there = which < len(times)
        which = np.minimum(which, max(len(times) - 1, 0))
        left = np.full(len(t), np.inf)
        if len(times):
            left[there] = np.asarray(metres_at(times[which[there]]), dtype=np.float64) - here[there]
        return which, there, left

    signs = sorted(reviewed['signs'], key=lambda sign: sign[0])
    times = np.array([at for at, _kph in signs], dtype=np.int64)
    values = np.array([-1 if value is None else value for _at, value in signs], dtype=np.int64)
    masked = sorted(reviewed['masked'], key=lambda span: span[1])
    ends = np.array([end for _start, end in masked], dtype=np.int64)
    hidden, hidden_has, hidden_left = first_after(ends.tolist())
    hidden_end = np.append(ends, 0)[hidden]
    inside = np.zeros(len(t), dtype=bool)
    for start, end in masked:
        inside |= (t >= start) & (t < end)
    # The limit in force: the number of the last sign passed, none at the start, after an end or after a masked sign.
    last = np.searchsorted(times, t, side='right') - 1
    last_masked = np.searchsorted(ends, t, side='right') - 1
    last_passed = np.where(last >= 0, np.append(times, 0)[last], -1)
    masked_passed = np.where(last_masked >= 0, np.append(ends, 0)[last_masked], -1)
    in_force = np.where((last >= 0) & (masked_passed < last_passed), np.append(values, -1)[last], -1)
    # The next: the first sign whose number is not the one in force (an end always is), the first of all with none.
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    changed = np.append(changes, len(times))[np.searchsorted(changes, np.maximum(last, 0), side='right')]
    nearest = np.searchsorted(times, t, side='right')
    spans = sorted(reviewed['covered'])
    reach, held, stop = np.zeros(len(t)), np.zeros(len(t), dtype=bool), np.zeros(len(t), dtype=np.int64)
    certain = np.maximum(last_passed, masked_passed) < 0
    if spans:
        starts, stops = np.array([span[0] for span in spans], dtype=np.int64), np.array([span[1] for span in spans],
                                                                                        dtype=np.int64)
        span = np.maximum(np.searchsorted(starts, t, side='right') - 1, 0)
        held = (t >= starts[span]) & (t <= stops[span])
        # A passing is a little after the box it comes from: as near the span as its frames are to one another.
        stop = stops[span] + int(COVER_GAP_SECONDS * NS)
        reach[held] = np.asarray(metres_at(stops[span[held]]), dtype=np.float64) - here[held]
        # What is in force is known only where everything since the sign that put it there was read.
        certain |= held & (np.maximum(last_passed, masked_passed) >= starts[span] - int(COVER_GAP_SECONDS * NS))
    # Where it is not known a repeat cannot be told from a change: no number, and none only with no sign at all ahead.
    upcoming = np.where(certain & (in_force > 0), changed, nearest)
    has = upcoming < len(times)
    kph = np.append(values, -1)[upcoming]
    passed = np.append(times, 0)[upcoming]
    left = np.full(len(t), np.inf)
    if has.any():
        left[has] = np.asarray(metres_at(passed[has]), dtype=np.float64) - here[has]
    numbered = (certain & has & (left <= AHEAD_METRES) & (kph > 0) & ~(hidden_has & (hidden_end < passed)) &
                held & (passed <= stop))
    none = (left > AHEAD_MARGIN_METRES) & (hidden_left > AHEAD_MARGIN_METRES) & ~inside & (reach >= AHEAD_MARGIN_METRES)
    following[rows[numbered]] = speed_class(kph[numbered])
    next_weight[rows[numbered]] = 1
    metres[rows[numbered]], metres_weight[rows[numbered]] = left[numbered] / METRES_SCALE, 1
    next_weight[rows[none]] = 1
    return following, next_weight, metres, metres_weight


def sign_inputs(seconds: np.ndarray, readings: list | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The signs, plate, plate_hours and plate_wet groups (frames, size) from the sure readings [[seconds, [[class, confidence], ...]], ...] of a
    drive, by the car's own file. Only readings at or before the frame count; None: the drive was not read, all unknown."""
    signs = np.zeros((len(seconds), INPUT_SIZES['signs']), dtype=np.float32)
    plates = np.zeros((len(seconds), INPUT_SIZES['plate']), dtype=np.float32)
    hours = np.zeros((len(seconds), INPUT_SIZES['plate_hours']), dtype=np.float32)
    wet = np.zeros((len(seconds), INPUT_SIZES['plate_wet']), dtype=np.float32)
    ordered = sorted(readings or [], key=lambda item: item[0])
    memory, pointer = BRAIN_INPUTS.SignInputs(VISION), 0
    for row, second in enumerate(seconds):
        while pointer < len(ordered) and ordered[pointer][0] <= second:
            now, read = ordered[pointer]
            memory.add(now, [(name, confidence, VISION.class_plate(name), VISION.class_hours(name))
                             for name, confidence in read])
            pointer += 1
        signs[row], plates[row], hours[row] = memory.features(second), memory.plate_row(), memory.plate_hours_row()
        wet[row] = memory.plate_wet_row()
    return signs, plates, hours, wet


def road_sign_inputs(seconds: np.ndarray, readings: list | None) -> np.ndarray:
    """The road_signs group (frames, size): every other sign of the sure readings, by the car's own SignInputs as on the
    car. Only readings at or before the frame count; None: the drive was not read, all unknown."""
    rows = np.zeros((len(seconds), INPUT_SIZES['road_signs']), dtype=np.float32)
    ordered = sorted(readings or [], key=lambda item: item[0])
    memory, pointer = BRAIN_INPUTS.SignInputs(VISION), 0
    for row, second in enumerate(seconds):
        while pointer < len(ordered) and ordered[pointer][0] <= second:
            now, read = ordered[pointer]
            memory.add(now, [(name, confidence) for name, confidence in read])
            pointer += 1
        rows[row] = memory.road_signs_row(second)
    return rows


def countries_so_far(gps: list, ns: np.ndarray, root: Path | None) -> list[str | None]:
    """The country at each time as the car has it: that of the last fix at or before it [t_ns, latitude, longitude,
    unix_ms], none before the first."""
    times, of_fix, found = [row[0] for row in gps], {}, []
    for t in ns:
        at = bisect.bisect_right(times, int(t)) - 1
        if at >= 0 and at not in of_fix:
            of_fix[at] = frame_country(gps[at][1], gps[at][2], root)
        found.append(of_fix.get(at))
    return found


def unix_at(gps: list, ns: np.ndarray) -> list[float | None]:
    """The GPS clock at each time, seconds: its last fix at or before it [t_ns, latitude, longitude, unix_ms] and the time
    since. None before the first."""
    fixes = [row for row in gps if row[3]]
    times = [row[0] for row in fixes]
    found = []
    for t in ns:
        at = bisect.bisect_right(times, int(t)) - 1
        found.append(None if at < 0 or t < 0 else fixes[at][3] / 1000 + (int(t) - times[at]) / NS)
    return found


def drive_inputs(frames: list[dict], ns: np.ndarray, merged: dict, readings: list | None,
                 root: Path | None, boxes: list | None = None) -> dict[str, np.ndarray]:
    """{group: (frames, size) float32}: see the module's text. Nothing after a frame reaches it."""
    series, gps = merged.get('series') or {}, merged.get('gps') or []
    seconds = np.array([frame.get('seconds', frame['segment'] * SEGMENT_SECONDS + frame['index']) for frame in frames],
                       dtype=np.float64)
    distance, lead_speed = _last_at(series.get('lead_d') or [], ns), _last_at(series.get('lead_v') or [], ns)
    lead = np.array([BRAIN_INPUTS.lead_row(metres, kph) for metres, kph in zip(distance, lead_speed)], dtype=np.float32)
    gas, gas_known = _pressed(series.get('gas') or [], ns)
    brake, brake_known = _pressed(series.get('brake') or [], ns)
    speed = [BRAIN_INPUTS.speed_row(None if kph is None else kph / 3.6) for kph in _last_at(series.get('v_ego') or [], ns)]
    countries = countries_so_far(gps, ns, root)
    country = np.array([BRAIN_INPUTS.country_row(name) for name in countries], dtype=np.float32)
    hour = np.array([BRAIN_INPUTS.hour_row(BRAIN_INPUTS.local_hour(unix, name))
                     for unix, name in zip(unix_at(gps, ns), countries)], dtype=np.float32)
    signs, plate, plate_hours, plate_wet = sign_inputs(seconds, readings)
    return {'signs': signs, 'plate': plate, 'plate_hours': plate_hours, 'plate_wet': plate_wet,
            'hour': hour.reshape(len(ns), INPUT_SIZES['hour']),
            'lead': lead.reshape(len(ns), INPUT_SIZES['lead']), 'country': country.reshape(len(ns), INPUT_SIZES['country']),
            'pedals': np.stack([gas, brake, gas_known * brake_known], axis=1),
            'speed': np.array(speed, dtype=np.float32).reshape(len(ns), INPUT_SIZES['speed']),
            'engaged': engaged_inputs(series, ns), 'sign_distance': sign_distance_inputs(ns, merged, readings, boxes),
            'motion': motion_inputs(series, ns), 'road_signs': road_sign_inputs(seconds, readings)}


def inputs_sha256() -> str:
    """The hash of the car's file of the inputs, for the checkpoint: the car refuses a network made with another."""
    return hashlib.sha256(BRAIN_INPUTS_PATH.read_bytes()).hexdigest()


def memory_cuts(ns: np.ndarray) -> set[int]:
    """The frames the memory starts again at: after MEMORY_GAP_SECONDS without a frame."""
    known = np.flatnonzero(ns >= 0)
    return {int(at) for at in known[1:][np.diff(ns[known]) > BRAIN_INPUTS.MEMORY_GAP_SECONDS * NS]}


def drive_targets(root: Path, route: str, labels: dict, merged: dict, split: dict, *, rate: float = 1.0,
                  need_pictures: bool = True, inputs: bool = True, kinds: tuple[str, ...] = (), types: bool = False,
                  models: Path | None = None, stamp: Any = IN_USE) -> dict:
    """A drive for the training at rate frames a second (1 or RATE_HZ): frames, ns, partitions, cells, cuts (memory_cuts),
    speed (class per frame), speed_out (why a frame is out of the loss, 0 when in), speed_weight, blocks (Speed review's),
    blocked (a block with a speed holds there), works, works_weight, road, road_weight (road_answers, a column for each of
    kinds: road_kinds), road_type, type_weight, typed_by_person (type_answers), types (the frames without a road type
    answer nothing: see the module's text), next, next_weight, metres, metres_weight
    (ahead_from_signs), sign_review (reviewed_signs, with its fingerprint), measured (the frames a measure counts: one a
    second, so that it reads the same at both rates), inputs, own_kph (the car's speed as the dashboard shows it, in the
    labels' units: drive_logs.dashboard_speed; NaN unknown; for the measure, no input), signs_read, reader (what read
    them), pictures (at RATE_HZ, the brain's pictures are there), no_picture_s (seconds of the drive with an answer of the
    speed and no frame with its picture: prepare keeps only the slow ones at the start and the end of a drive).
    labels: read_labels' of the drive, NO_LABELS without a block or a mark. need_pictures=False (a dry run): a frame
    without its picture keeps its answers. inputs=False (only what the answers are, for summary): inputs is None, since
    the country of every frame is most of the time a drive takes. models: the release its signs were read with
    (road_type_signs.cache_path; None: the car's of the sunnypilot clone); stamp: reviewed_signs'."""
    frames = drive_frames(root, route, merged)
    partitions = frame_partitions(route, frames, split)
    cells = frame_cells(frames)
    record = load_signs(root, route, models)
    if rate == RATE_HZ:
        frames, partitions, cells = brain_frames(root, frames, partitions, cells, merged, record)
    elif rate != 1.0:
        raise ValueError(f'the brain runs at 1 or {RATE_HZ:g} frames a second, not {rate}')
    ns = np.array([-1 if frame['ns'] is None else frame['ns'] for frame in frames], dtype=np.int64)
    usable = np.array([frame['ns'] is not None and (frame['present'] or not need_pictures) for frame in frames], dtype=bool)
    blocks = labels['blocks']
    series = merged.get('series') or {}
    speed, speed_out = speed_answers(ns, driver_speed(merged, ns), blocks, machine_held(merged, ns))
    pictured = (speed_out == 0) & np.array([frame['ns'] is not None and frame['present'] for frame in frames], dtype=bool)
    speed_out[~usable] = NO_FRAME
    speed[~usable] = -1
    blocked = np.zeros(len(ns), dtype=bool)
    for block in blocks:
        if block['kph'] is not None:
            blocked |= (ns >= block['start_ns']) & (ns < block['end_ns'])
    works, works_weight = works_answers(ns, labels['works'], labels['checked_ns'])
    works_weight[~usable] = 0
    road, road_weight = road_answers(ns, labels.get('road', []), labels.get('road_checked', {}), kinds)
    road_weight[~usable] = 0
    road_type, typed_by_person = type_answers(root, route, frames)
    typed = road_type >= 0
    if types:
        speed_out[(speed_out == 0) & ~typed] = NO_TYPE
        speed[~typed] = -1
        works_weight[~typed] = 0
        road_weight[~typed] = 0
    readings = record['readings'] if record else None
    v_ego = series.get('v_ego') or []
    own = _last_at(drive_logs.dashboard_speed(merged), ns)
    reviewed = reviewed_signs(root, route, merged, stamp)
    # Without vEgo the car has driven no metres: no limit is ahead of any frame. The signs ahead are there whoever held
    # the speed.
    asked = np.isin(speed_out, (0, CRUISE)) & any(point[1] is not None for point in v_ego)
    following, next_weight, metres, metres_weight = ahead_from_signs(ns, reviewed, metres_driven(v_ego), asked)
    if types:
        next_weight[~typed] = 0
        metres_weight[~typed] = 0
    span = merged.get('span') or [None, None]
    taught = 0
    if span[0] is not None and span[1] is not None:
        grid = np.arange(int(span[0]), int(span[1]), int(NS / rate), dtype=np.int64)
        taught = int((speed_answers(grid, driver_speed(merged, grid), blocks, machine_held(merged, grid))[1] == 0).sum())
    return {'route': route, 'rate': rate, 'frames': frames, 'ns': ns, 'partitions': partitions,
            'own_kph': np.array([np.nan if kph is None else kph for kph in own], dtype=np.float32),
            'cells': cells, 'cuts': memory_cuts(ns), 'speed': speed, 'speed_out': speed_out,
            'speed_weight': (speed_out == 0).astype(np.float32), 'blocks': blocks, 'blocked': blocked,
            'works': works, 'works_weight': works_weight, 'road': road, 'road_weight': road_weight, 'road_kinds': list(kinds),
            'road_type': road_type, 'type_weight': (typed & usable).astype(np.float32), 'typed_by_person': typed_by_person,
            'types': types,
            'next': following, 'next_weight': next_weight, 'metres': metres, 'metres_weight': metres_weight,
            'sign_review': reviewed,
            'measured': np.array([not frame.get('step') for frame in frames], dtype=bool),
            'marks': [mark for mark in labels['works'] if mark['end_ns'] is not None],
            'inputs': drive_inputs(frames, ns, merged, readings, root, record.get('boxes') if record else None)
            if inputs else None,
            'signs_read': readings is not None,
            'reader': record.get('fingerprint') if record else None,
            'pictures': rate == 1.0 or pictures_ready(root, record),
            'no_picture_s': max(round((taught - int(pictured.sum())) / rate), 0)}


def keep_sites_whole(drives: list[dict]) -> None:
    """Train frames of a held-out works site, and of any site on its areas, become buffer: see the module's text."""
    sites = []
    for drive in drives:
        for mark in drive['marks']:
            near = np.flatnonzero((drive['ns'] >= mark['start_ns'] - SITE_MARGIN_S * NS) &
                                  (drive['ns'] < mark['end_ns'] + SITE_MARGIN_S * NS))
            sites.append((drive, near, {drive['cells'][i] for i in near} - {None}))
    held = set()
    for drive, near, cells in sites:
        if any(drive['partitions'][i] in HELD_OUT for i in near):
            held |= cells
    for drive, near, cells in sites:
        if cells & held:
            for i in near:
                if drive['partitions'][i] == 'train':
                    drive['partitions'][i] = 'buffer'


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def dataset(root: Path, merged_of, routes: list[str] | None = None, log=_say, *, rate: float = 1.0,
            need_pictures: bool = True, inputs: bool = True, types: bool = False, networks: dict | None = None) -> list[dict]:
    """Every drive of training_routes (or only routes), as drive_targets at rate with its labels (inputs and types as
    there), each with reader_current: read by the car's reader of today (networks: by the networks chosen for the
    brain, sign_networks.load's: their release for the inputs, the three for the labels; None: the car's release of
    the sunnypilot clone and Our networks in use). merged_of(route): the drive's joined logs, None
    before a scan. A drive that is no dashcam drive, one whose logs show openpilot driving (engaged_series) and one of the frozen split
    whose frames changed are left out with a line in log. Sites stay whole only across the drives asked for. Every drive
    has the kinds of road of today (road_kinds), in the same columns."""
    split = read_split(root)
    kinds = road_kinds(root)
    models = networks['models'] if networks else None
    reader = fingerprint(package_path(), models=models)
    stamp = brain_signs.stamp_in_use(root, {part: networks[part] for part in SIGN_PARTS}) if networks else IN_USE
    left_out: dict[str, str] = {}
    labels, trainable = read_labels(root), training_routes(root, left_out)
    for route in sorted(set(labels) | set(routes or ()) | set(left_out)):
        if route not in trainable and (routes is None or route in routes):
            log(f"{route}: {left_out.get(route, 'no finished scan or no frames')}, left out")
    drives = []
    for route in trainable:
        if routes is not None and route not in routes:
            continue
        merged = merged_of(route)
        if merged is None:
            log(f'{route}: no scan, left out')
            continue
        engaged = engaged_series(merged)
        if engaged:
            log(f"{route}: openpilot drove ({', '.join(engaged)} on), left out")
            continue
        try:
            drive = drive_targets(root, route, labels.get(route, NO_LABELS), merged, split, rate=rate,
                                  need_pictures=need_pictures, inputs=inputs, kinds=kinds, types=types, models=models,
                                  stamp=stamp)
        except FramesChanged as error:
            log(f'{error}, left out')
            continue
        drives.append({**drive, 'reader_current': drive['reader'] == reader})
    keep_sites_whole(drives)
    return drives


def check_drives(drives: list[dict]) -> None:
    """Raises on what would train wrong without a word: see the module's text."""
    empty = [drive['route'] for drive in drives if not len(drive['frames'])]
    if empty:
        raise ValueError(f'drives with no frame: {", ".join(empty)}')
    unread = [drive['route'] for drive in drives if not drive['signs_read']]
    if unread:
        raise ValueError(f'drives with no reading of the signs: {", ".join(unread)}. Run {READ_COMMAND}')
    other = [drive['route'] for drive in drives if not drive['reader_current']]
    if other:
        raise ValueError(f"drives read by another reader than the car's of today: {', '.join(other)}. Run {READ_COMMAND}")
    bare = [drive['route'] for drive in drives if not drive.get('pictures', True)]
    if bare:
        raise ValueError(f"drives without the brain's pictures: {', '.join(bare)}. Run {READ_COMMAND}")


def loss_weights(drive: dict, partition: str) -> tuple[np.ndarray, np.ndarray]:
    """(speed weight, works weight) of a drive in one partition: the rest of the drive still runs through the memory."""
    inside = np.array([name == partition for name in drive['partitions']], dtype=np.float32)
    return drive['speed_weight'] * inside, drive['works_weight'] * inside


def road_weights(drive: dict, partition: str) -> np.ndarray | None:
    """The road weight (frames, kinds) of a drive in one partition; None for a drive without the kinds of road."""
    if 'road_weight' not in drive:
        return None
    inside = np.array([name == partition for name in drive['partitions']], dtype=np.float32)
    return drive['road_weight'] * inside[:, None]


def type_weights(drive: dict, partition: str) -> np.ndarray | None:
    """The road type's weight of a drive in one partition; None for a drive without it."""
    if 'type_weight' not in drive:
        return None
    inside = np.array([name == partition for name in drive['partitions']], dtype=np.float32)
    return drive['type_weight'] * inside


def ahead_weights(drive: dict, partition: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(next weight, metres weight) of a drive in one partition; None for a drive without them."""
    if 'next_weight' not in drive:
        return None
    inside = np.array([name == partition for name in drive['partitions']], dtype=np.float32)
    return drive['next_weight'] * inside, drive['metres_weight'] * inside


def other_hour(hour: np.ndarray, hours: float) -> np.ndarray:
    """The hour group with the clock moved by hours on every frame. An unknown hour stays unknown."""
    sin, cos, angle = hour[:, 0], hour[:, 1], 2 * np.pi * hours / 24
    return np.stack([sin * np.cos(angle) + cos * np.sin(angle), cos * np.cos(angle) - sin * np.sin(angle), hour[:, 2]],
                    axis=1).astype(np.float32)


def _clock(hour: np.ndarray) -> np.ndarray:
    return np.arctan2(hour[:, 0], hour[:, 1]) / (2 * np.pi) * 24 % 24


def _plate_window(drive: dict) -> tuple[np.ndarray, np.ndarray]:
    """(from, how many hours) of the plate on each frame: the hours read, the night when none were."""
    hours = drive['inputs']['plate_hours']
    start = np.where(hours[:, 4] > 0, np.rint(_clock(hours[:, :2])), NIGHT_FROM)
    end = np.where(hours[:, 4] > 0, np.rint(_clock(hours[:, 2:4])), NIGHT_TO)
    return start % 24, (end - start) % 24


def _inside(clock: np.ndarray, window: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return (clock - window[0]) % 24 < window[1]


def _hours_plate(drive: dict) -> np.ndarray:
    return drive['inputs']['plate'][:, BRAIN_INPUTS.PLATES.index('time')] > 0


def _wet_plate(drive: dict) -> np.ndarray:
    return drive['inputs']['plate_wet'][:, 0] > 0


def plates_in_train(drives: list[dict]) -> bool:
    """Whether a brain trained on these drives learnt the plate: a plate of hours on a frame of train with a speed to answer.
    The car gives a brain its plates only then (contract step.plates), since weights never trained are no reading."""
    return any(bool((_hours_plate(drive) & (loss_weights(drive, 'train')[0] > 0)).any()) for drive in drives)


def hours_in_train(drives: list[dict]) -> bool:
    """Whether a brain trained on these drives learnt the hours of a plate: as plates_in_train, with hours read."""
    return any(bool(((drive['inputs']['plate_hours'][:, 4] > 0) & (loss_weights(drive, 'train')[0] > 0)).any())
               for drive in drives)


def wet_in_train(drives: list[dict]) -> bool:
    """Whether a brain trained on these drives learnt the plate for wet roads: as plates_in_train."""
    return any(bool((_wet_plate(drive) & (loss_weights(drive, 'train')[0] > 0)).any()) for drive in drives)


def _with_hour(drive: dict, groups: tuple[str, ...], hours: float) -> dict:
    inputs = {**drive['inputs'], 'hour': other_hour(drive['inputs']['hour'], hours)}
    return {**drive, 'inputs': inputs, 'extra': pack_inputs(inputs, groups)}


def sign_kph(signs: np.ndarray) -> np.ndarray:
    """The km/h of the number confirmed in the rows of the signs input."""
    return np.rint(signs[:, 0] * BRAIN_INPUTS.SIGN_KPH_SCALE).astype(np.int64)


def night_frames(drive: dict) -> tuple[np.ndarray, np.ndarray]:
    """(the frames a night copy can answer, the class of the number on each): see the module's text."""
    signs, hour = drive['inputs']['signs'], drive['inputs']['hour']
    kph = sign_kph(signs)
    a_class = (kph >= SIGN_STEP_KPH) & (kph <= BRAIN_MAX_KPH) & (kph % SIGN_STEP_KPH == 0) & (signs[:, 6] == 0)
    plated, label = _hours_plate(drive) & a_class, class_kph(drive['speed'])
    answers = (hour[:, 2] > 0) & ~_inside(_clock(hour), _plate_window(drive)) & (loss_weights(drive, 'train')[0] > 0)
    frames = plated & answers & (label > kph)
    for passage in np.split(np.arange(len(kph)), np.flatnonzero((np.diff(kph) != 0) | (np.diff(plated) != 0)) + 1):
        if (answers[passage] & (np.abs(label[passage] - kph[passage]) <= SWAP_TOLERANCE_KPH)).any():
            frames[passage] = False
    return frames, speed_class(kph)


def night_copy(drive: dict, groups: tuple[str, ...], rng) -> dict | None:
    """The drive with its first frame to answer at a random hour of its plate's, the number the answer of the frames that
    fall in their plate's hours: see the module's text. None with nothing to answer."""
    frames, number = night_frames(drive)
    if not frames.any():
        return None
    clock = _clock(drive['inputs']['hour'])
    window, first = _plate_window(drive), np.flatnonzero(frames)[0]
    hours = window[0][first] + rng.uniform(0, window[1][first]) - clock[first]
    answered = frames & _inside((clock + hours) % 24, window)
    copy = {**_with_hour(drive, groups, hours), 'speed': np.where(answered, number, drive['speed']),
            'speed_weight': answered.astype(np.float32), 'works_weight': np.zeros_like(drive['works_weight'])}
    if 'next_weight' in drive:
        copy['next_weight'], copy['metres_weight'] = np.zeros_like(drive['next_weight']), np.zeros_like(drive['metres_weight'])
    if 'road_weight' in drive:
        copy['road_weight'] = np.zeros_like(drive['road_weight'])
    if 'type_weight' in drive:
        copy['type_weight'] = np.zeros_like(drive['type_weight'])
    return copy


def swap_passages(drive: dict, partition: str = 'train') -> list[np.ndarray]:
    """The passages of a number of SWAPS with an answer of this partition, in force, all such answers within
    SWAP_TOLERANCE_KPH of the number and no block of Speed review in them: see the module's text."""
    signs = drive['inputs']['signs']
    kph, label = sign_kph(signs), class_kph(drive['speed'])
    answers = (loss_weights(drive, partition)[0] > 0) & (signs[:, 6] == 0)
    blocked = drive.get('blocked')
    blocked = np.zeros(len(kph), dtype=bool) if blocked is None else blocked
    found = []
    for passage in np.split(np.arange(len(kph)), np.flatnonzero(np.diff(kph) != 0) + 1):
        number, asked = int(kph[passage[0]]), passage[answers[passage]]
        near = np.abs(label[asked] - number) <= SWAP_TOLERANCE_KPH
        said = asked[near]
        # Before the sign the speed may still be the one before it: the number is read some way ahead of it.
        if number in SWAPS and len(said) and near[asked >= said[0]].all() and not blocked[passage].any():
            found.append(passage)
    return found


def swapped(drive: dict, passages: list[np.ndarray], groups: tuple[str, ...]) -> dict:
    """The drive with the number of these passages, and what says it, as the other of its family: see the module's text."""
    signs, speed, label = drive['inputs']['signs'].copy(), drive['speed'].copy(), class_kph(drive['speed'])
    following = drive['next'].copy() if 'next' in drive else None
    weights = (drive['next_weight'].copy(), drive['metres_weight'].copy()) if 'next_weight' in drive else None
    scale = BRAIN_INPUTS.SIGN_KPH_SCALE
    for passage in passages:
        number = int(sign_kph(signs[passage[:1]])[0])
        said = passage[(np.abs(label[passage] - number) <= SWAP_TOLERANCE_KPH) & (signs[passage, 6] == 0) &
                       (drive['speed'][passage] >= 0)]
        single = passage[np.rint(signs[passage, 8] * scale) == number]
        signs[passage, 0] = signs[single, 8] = SWAPS[number] / scale
        # His speed off the number by as much as before: 52 past a 50 is 62 past a 60.
        speed[said] = speed_class(np.clip(label[said] + SWAPS[number] - number, 1, BRAIN_MAX_KPH))
        if following is not None:
            ahead = passage[following[passage] == speed_class(number)]
            # A next of the other number is now a repeat of the one in force, which is no next: what is after it unknown.
            repeat = passage[following[passage] == speed_class(SWAPS[number])]
            following[ahead] = speed_class(SWAPS[number])
            if weights is not None:
                weights[0][repeat] = weights[1][repeat] = 0
    inputs = {**drive['inputs'], 'signs': signs}
    found = {**drive, 'inputs': inputs, 'speed': speed, 'extra': pack_inputs(inputs, groups)}
    if following is not None:
        found['next'] = following
    if weights is not None:
        found['next_weight'], found['metres_weight'] = weights
    return found


def _half_swapped(drive: dict, groups: tuple[str, ...], rng) -> dict:
    passages = [passage for passage in swap_passages(drive) if rng.random() < .5]
    return swapped(drive, passages, groups) if passages else drive


def epoch_drives(drives: list[dict], groups: tuple[str, ...], rng) -> list[dict]:
    """What an epoch of training goes through, new every epoch: the drives with a moved clock and half of their numbers
    swapped, and the night copies. See the module's text. Never for a measure: drives stays as it is."""
    found = []
    for drive in drives:
        if (_hours_plate(drive) & (np.array(drive['partitions']) == 'train')).any():
            copy = night_copy(drive, groups, rng)
            seen = _half_swapped(drive, groups, rng)
            found += [seen] if copy is None else [seen, copy]
        else:
            found.append(_half_swapped(_with_hour(drive, groups, rng.uniform(0, 24)), groups, rng))
    return found


def drive_chunks(length: int, chunk: int, offset: int = 0) -> list[tuple[int, int]]:
    """(from, to) of the pieces of a drive in order, covering every frame once: the first one offset frames long (a
    different offset every epoch moves the cuts), then chunk frames each. The state goes from one to the next."""
    edges = [0, *range(offset or chunk, length, chunk), length]
    return [(first, last) for first, last in zip(edges, edges[1:]) if first < last]


def learning_pieces(partitions: list[str], chunk: int, offset: int = 0, cuts: set[int] = frozenset()) -> list[tuple[int, int, bool, bool]]:
    """(from, to, learn, fresh) of a drive in order, fresh: the memory starts again here. The pieces never mix train with the
    rest, and frames of test and validation are in none: their pictures run through nothing that moves a weight, not even
    through the memory, which starts again after them. cuts: frames it starts again at anyway."""
    def kind(name: str) -> str:
        return 'held' if name in HELD_OUT else name == 'train'

    pieces, first, fresh = [], 0, True
    for last in range(1, len(partitions) + 1):
        if last == len(partitions) or kind(partitions[last]) != kind(partitions[first]) or last in cuts:
            learn = partitions[first] == 'train'
            if kind(partitions[first]) == 'held':
                fresh = True
            else:
                for a, b in drive_chunks(last - first, chunk, offset if learn else 0):
                    pieces.append((first + a, first + b, learn, fresh))
                    fresh = False
            first = last
            fresh = fresh or last in cuts
    return pieces


def _per_second(drive: dict) -> dict:
    """The answers of a drive on its measured frames only, one a second."""
    if 'measured' not in drive:
        return {}
    kept = drive['measured']
    found = {'frames': [frame for frame, measured in zip(drive['frames'], kept) if measured],
             'speed_weight': drive['speed_weight'] * kept, 'works_weight': drive['works_weight'] * kept,
             'speed_out': np.where(kept, drive['speed_out'], -1)}
    if 'next_weight' in drive:
        found['next_weight'] = drive['next_weight'] * kept
    if 'road_weight' in drive:
        found['road_weight'] = drive['road_weight'] * kept[:, None]
    if 'type_weight' in drive:
        found['type_weight'] = drive['type_weight'] * kept
    return found


def summary(drives: list[dict]) -> dict:
    """What the labels give the training today: per partition the frames with a speed, the works frames, the frames of
    each kind of road (road: {kind: {in, out}}) and the frames with an answer of the next limit, per drive the same with
    why the rest is out, its blocks, its signs (Sign review's and our networks') and the seconds read there. Frames one a
    second (the measured ones) at either rate: seconds of the drive."""
    kinds = next((drive['road_kinds'] for drive in drives if 'road_kinds' in drive), [])

    def road_of(drive: dict, here) -> dict:
        if 'road_weight' not in drive:
            return {}
        weight, road = drive['road_weight'][here], drive['road'][here]
        return {kind: {'in': int((weight[:, column] * road[:, column]).sum()),
                       'out': int((weight[:, column] * (1 - road[:, column])).sum())} for column, kind in enumerate(kinds)}

    parts = {name: {'speed': 0, 'works_in': 0, 'works_out': 0, 'sites': 0, 'next': 0,
                    'road': {kind: {'in': 0, 'out': 0} for kind in kinds}, 'types': {kind: 0 for kind in ROAD_TYPES}}
             for name in ('train', 'validation', 'test', 'buffer')}
    rows = []
    for drive in drives:
        names = np.array(drive['partitions'])
        drive = {**drive, **_per_second(drive)}
        following = drive.get('next_weight', np.zeros(len(names), dtype=np.float32))
        for name, part in parts.items():
            here = names == name
            part['speed'] += int(drive['speed_weight'][here].sum())
            part['works_in'] += int((drive['works_weight'][here] * drive['works'][here]).sum())
            part['works_out'] += int((drive['works_weight'][here] * (1 - drive['works'][here])).sum())
            part['next'] += int(following[here].sum())
            for kind, count in road_of(drive, here).items():
                part['road'][kind]['in'] += count['in']
                part['road'][kind]['out'] += count['out']
            if 'type_weight' in drive:
                for column, kind in enumerate(ROAD_TYPES):
                    part['types'][kind] += int((drive['type_weight'][here] * (drive['road_type'][here] == column)).sum())
        for mark in drive['marks']:
            inside = np.flatnonzero((drive['ns'] >= mark['start_ns']) & (drive['ns'] < mark['end_ns']))
            if len(inside):
                parts[names[inside[len(inside) // 2]]]['sites'] += 1
        reviewed = drive.get('sign_review') or {}
        rows.append({'route': drive['route'], 'frames': len(drive['frames']), 'speed': int(drive['speed_weight'].sum()),
                     'out': {name: int((drive['speed_out'] == reason).sum()) for reason, name in OUT_REASONS.items()},
                     'works_in': int((drive['works_weight'] * drive['works']).sum()),
                     'works_out': int((drive['works_weight'] * (1 - drive['works'])).sum()), 'marks': len(drive['marks']),
                     'road': {kind: count for kind, count in road_of(drive, slice(None)).items() if count['in'] or count['out']},
                     'blocks': len(drive.get('blocks') or ()), 'signs': len(reviewed.get('signs') or ()),
                     'covered_s': round(sum(end - start for start, end in reviewed.get('covered') or ()) / NS),
                     'networks_read': bool(reviewed.get('read_by')),
                     'next': int(following.sum()),
                     'typed_s': int(drive['type_weight'].sum()) if 'type_weight' in drive else 0,
                     'typed_by_person_s': int((drive['type_weight'] * drive['typed_by_person']).sum())
                     if 'type_weight' in drive else 0,
                     'signs_read': drive['signs_read'], 'reader_current': drive.get('reader_current', False),
                     'pictures': drive.get('pictures', True), 'no_picture_s': drive['no_picture_s']})
    return {'partitions': parts, 'drives': rows, 'road_kinds': list(kinds), 'road_types': list(ROAD_TYPES),
            'types': any(drive.get('types') for drive in drives)}


def summary_text(found: dict) -> str:
    lines = ['Per partition (frames, one a second):',
             f"  {'':<11}{'speed':>7}{'works in':>10}{'works out':>11}{'sites':>7}{'next':>7}"]
    for name, part in found['partitions'].items():
        lines.append(f"  {name:<11}{part['speed']:>7}{part['works_in']:>10}{part['works_out']:>11}{part['sites']:>7}"
                     f"{part.get('next', 0):>7}")
    for kind in found.get('road_kinds', []):
        lines.append(f"  {kind}: " + ', '.join(f"{name} {part['road'][kind]['in']} in, {part['road'][kind]['out']} out"
                                               for name, part in found['partitions'].items()))
    for name, part in found['partitions'].items():
        typed = ', '.join(f'{kind} {count}' for kind, count in part.get('types', {}).items() if count)
        lines.append(f"  road types in {name}: {typed or 'none'}")
    lines.append('Per drive:')
    for row in found['drives']:
        out = ', '.join(f'{name} {count}' for name, count in row['out'].items() if count)
        signs = 'NOT read' if not row['signs_read'] else 'read' if row['reader_current'] else 'read by ANOTHER reader'
        signs += '' if row.get('pictures', True) else ", the brain's pictures NOT made"
        lines.append(f"  {row['route']}: {row['speed']} of {row['frames']} with a speed ({out}), {row.get('blocks', 0)} "
                     f"blocks; works {row['works_in']} in, {row['works_out']} out, {row['marks']} marks; "
                     + ''.join(f"{kind} {count['in']} in, {count['out']} out; " for kind, count in row.get('road', {}).items()) +
                     f"{row.get('signs', 0)} signs ("
                     f"{'Sign review and our networks' if row.get('networks_read') else 'Sign review only, NOT read by our networks'}"
                     f"), {row.get('covered_s', 0)} s read, the next limit on "
                     f"{row.get('next', 0)}; a road type on {row.get('typed_s', 0)} s ({row.get('typed_by_person_s', 0)} "
                     f"set by a person); signs {signs}; {row['no_picture_s']} seconds with a speed and no picture")
    return '\n'.join(lines)
