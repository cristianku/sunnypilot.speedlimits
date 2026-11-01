"""Tracks: the same physical sign followed from one confirmed frame to the next.

``nnslr tracks --route <id>`` (or ``--all``) reads the confirmed reviews of a route's current report and links
every box to the box of the frame just before that is the same sign, with the matching of "Apply previous"
(nnslr_tools.carry_over.match_boxes: signs grow and move away from the point the road runs to, the view
slides in turns). A box that matches nothing starts a new track. It writes
``derived/tracks/<route>/tracks.jsonl``: one line per box,
``{"frame_key", "detection_id", "track_id", "index", "length", "family", "value_kph", "applicability"}``,
where index is the box's place in its track (0: first seen) and length the frames of the track.

Two frames are consecutive when they follow each other in the timeline of the report, in the same segment or
at the end of the segment before (as "Apply previous"), and both are confirmed. Boxes marked "not a sign" are
tracked too: a logo seen in five frames is one object, not five.

The tracks are for what comes after the reading: the same sign over time, whether it was passed, and keeping
a limit. Nothing here changes a review or the dataset.
"""
from __future__ import annotations

from pathlib import Path

from nnslr_tools.annotations import atomic_write, jsonl_bytes, load_jsonl
from nnslr_tools.carry_over import match_boxes
from nnslr_tools.manifest import parse_route_id
from nnslr_tools.sign_reads import frame_order

TRACKS_RELATIVE_DIR = Path('derived') / 'tracks'


def tracks_path(root: Path, route: str) -> Path:
    parse_route_id(route)
    return Path(root) / TRACKS_RELATIVE_DIR / route / 'tracks.jsonl'


def link_tracks(frames: list[dict], *, width: float, height: float) -> list[dict]:
    """The track of every box of ``frames``: [{'frame_key', 'segment_index', 'output_index', 'detections'}] in
    timeline order, each with its confirmed boxes. Returns one line per box, without index and length."""
    lines = []
    previous = None
    previous_tracks: list[str] = []
    next_track = 0
    for frame in frames:
        boxes = [d for d in frame['detections'] if isinstance(d, dict)]
        tracks: list[str | None] = [None] * len(boxes)
        consecutive = (previous is not None and previous['position'] == frame['position'] - 1
                       and previous['segment_index'] >= frame['segment_index'] - 1)
        if consecutive and previous['detections'] and boxes:
            for old, (new, _scale, _shift) in match_boxes(previous['detections'], boxes, width=width,
                                                          height=height).items():
                tracks[new] = previous_tracks[old]
        for index, box in enumerate(boxes):
            if tracks[index] is None:
                next_track = next_track + 1
                tracks[index] = f"{frame['frame_key'].replace('/', '-')}-t{next_track}"
            lines.append({'frame_key': frame['frame_key'], 'detection_id': box.get('detection_id'),
                          'track_id': tracks[index], 'family': box.get('family'), 'value_kph': box.get('value_kph'),
                          'applicability': box.get('applicability')})
        previous = {**frame, 'detections': boxes}
        previous_tracks = [track for track in tracks if track is not None]
    return lines


def build_tracks(root: Path, route: str, rows: list[dict], reviews: list[dict]) -> dict:
    """Write tracks.jsonl of a route from its report ``rows`` and its saved ``reviews``; returns counts."""
    order = sorted((*frame_order(row), row['frame_key']) for row in rows
                   if isinstance(row.get('segment_index'), int) and isinstance(row.get('output_index'), int))
    position = {key: index for index, (_segment, _second, _output, key) in enumerate(order)}
    sizes = {row['frame_key']: (row.get('width') or 1344, row.get('height') or 760) for row in rows}
    confirmed = {review['frame_key']: review for review in reviews if review['confirmed']}
    frames = []
    for segment, _second, output, key in order:
        review = confirmed.get(key)
        if review is None:
            continue
        frames.append({'frame_key': key, 'segment_index': segment, 'output_index': output,
                       'position': position[key], 'detections': review['payload'].get('detections', [])})
    width, height = next(iter(sizes.values()), (1344, 760))
    lines = link_tracks(frames, width=width, height=height)
    lengths: dict[str, int] = {}
    for line in lines:
        lengths[line['track_id']] = lengths.get(line['track_id'], 0) + 1
    seen: dict[str, int] = {}
    for line in lines:
        line['index'] = seen.get(line['track_id'], 0)
        seen[line['track_id']] = line['index'] + 1
        line['length'] = lengths[line['track_id']]
    target = tracks_path(root, route)
    atomic_write(target, jsonl_bytes(lines))
    long_tracks = sum(1 for length in lengths.values() if length >= 2)
    return {'route_id': route, 'frames': len(frames), 'boxes': len(lines), 'tracks': len(lengths),
            'tracks_over_two_frames': long_tracks, 'output': str(target)}


def load_tracks(root: Path, route: str) -> dict:
    """{(frame_key, detection_id): track line} of a route, {} when ``nnslr tracks`` has not run for it."""
    path = tracks_path(root, route)
    if not path.is_file():
        return {}
    return {(line['frame_key'], line['detection_id']): line for line in load_jsonl(path) if isinstance(line, dict)}
