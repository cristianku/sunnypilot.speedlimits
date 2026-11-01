"""Second-opinion hints and "suspicious" frames for the web review.

Hints are read-only. They come from ``<data root>/derived/hints/<route>/*.jsonl``,
one line per frame::

    {"route_id": "...", "frame_key": "<segment>/<output_index>", "segment_index": 2,
     "output_index": 24, "source": "...", "passes": ["A", "B"],
     "signs": [{"pass": "A", "type": "maximum_speed", "value_kph": 60,
                "box": [x1, y1, x2, y2], "confidence": "high", "side": "right", "note": "..."}]}

``box`` is in native image pixels. Unknown fields are ignored; a line that does
not follow this form is skipped and counted, and so is a single sign that does
not. The page shows hints with textContent only and never saves them: a hint
becomes a box of the review only when the reviewer presses "Usa".

A frame is "suspicious" when it is not confirmed yet and the model proposals,
the hints and the confirmed neighbours disagree (rules a-d in
:func:`suspicious_frames`). The server computes the flags and sends them with
the route page; the page only shows them.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from nnslr_tools.annotations import strict_json_loads
from nnslr_tools.manifest import parse_route_id
from nnslr_tools.sign_reads import frame_order
from speed_vision_core.types import MAX_VALUE_KPH, MIN_VALUE_KPH

HINTS_RELATIVE_DIR = Path('derived') / 'hints'
HINT_TYPES = ('maximum_speed', 'cancellation', 'zone', 'zone_end', 'variable_display', 'road_marking', 'unsure')
CONFIDENCES = ('high', 'medium', 'low')
# Families of a model proposal (or of a confirmed box) that carry a speed limit.
SPEED_FAMILIES = ('maximum_speed', 'cancellation', 'zone', 'variable_display')

MAX_HINT_FILE_BYTES = 16 * 1024 * 1024
MAX_SIGNS_PER_LINE = 50
MAX_NOTE_LENGTH = 300
MAX_SHORT_TEXT_LENGTH = 40
FRAME_KEY_PATTERN = re.compile(r'([0-9]{1,9})/([0-9]{1,9})')


def empty_stats() -> dict:
    return {'files': 0, 'lines': 0, 'frames': 0, 'signs': 0, 'skipped_files': 0, 'skipped_lines': 0,
            'skipped_signs': 0, 'unknown_frames': 0}


def hints_directory(root: Path, route: str) -> Path | None:
    """The hints directory of a route, resolved, or None when it is missing or leaves the data root."""
    parse_route_id(route)
    root_resolved = Path(root).resolve()
    directory = Path(root) / HINTS_RELATIVE_DIR / route
    if not directory.is_dir():
        return None
    resolved = directory.resolve()
    # A symlink anywhere on the way (derived, hints or the route folder) may not lead outside the data root.
    if not resolved.is_relative_to(root_resolved):
        return None
    return resolved


def _number(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _clean_text(value, limit: int):
    """Printable text of at most ``limit`` characters, or None."""
    if not isinstance(value, str):
        return None
    characters = []
    for character in value:
        if character.isprintable():
            characters.append(character)
        else:
            characters.append(' ')
    text = ''.join(characters).strip()
    if not text:
        return None
    if len(text) > limit:
        text = text[:limit] + '…'
    return text


def _clean_box(box, width, height):
    """[x1, y1, x2, y2] clipped to the image, or None when it is not a box with area inside the image."""
    if not isinstance(box, list) or len(box) != 4:
        return None
    for value in box:
        if not _number(value):
            return None
    if not _number(width) or not _number(height) or width <= 0 or height <= 0:
        return None
    x1 = min(max(box[0], 0), width)
    y1 = min(max(box[1], 0), height)
    x2 = min(max(box[2], 0), width)
    y2 = min(max(box[3], 0), height)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [x1, y1, x2, y2]


def parse_sign(sign, width, height, source: str) -> dict | None:
    """One validated hint, or None when the sign is malformed."""
    if not isinstance(sign, dict):
        return None
    kind = sign.get('type')
    if kind not in HINT_TYPES:
        return None
    confidence = sign.get('confidence')
    if confidence not in CONFIDENCES:
        return None
    value = sign.get('value_kph')
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        if value < MIN_VALUE_KPH or value > MAX_VALUE_KPH:
            return None
    box = _clean_box(sign.get('box'), width, height)
    if box is None:
        return None
    return {
        'type': kind,
        'value_kph': value,
        'box': box,
        'confidence': confidence,
        'pass': _clean_text(sign.get('pass'), MAX_SHORT_TEXT_LENGTH),
        'side': _clean_text(sign.get('side'), MAX_SHORT_TEXT_LENGTH),
        'note': _clean_text(sign.get('note'), MAX_NOTE_LENGTH),
        'source': source,
    }


def parse_line(text: str, route: str):
    """(frame_key, raw signs, source) of one hint line, or None when the line is malformed."""
    try:
        row = strict_json_loads(text)
    except (ValueError, TypeError, RecursionError):
        return None
    if not isinstance(row, dict):
        return None
    if row.get('route_id') != route:
        return None
    raw_key = row.get('frame_key')
    if not isinstance(raw_key, str):
        return None
    match = FRAME_KEY_PATTERN.fullmatch(raw_key)
    if match is None:
        return None
    segment = int(match.group(1))
    output = int(match.group(2))
    # The explicit indices, when present, must agree with the frame key.
    if 'segment_index' in row:
        value = row['segment_index']
        if isinstance(value, bool) or value != segment:
            return None
    if 'output_index' in row:
        value = row['output_index']
        if isinstance(value, bool) or value != output:
            return None
    signs = row.get('signs')
    if not isinstance(signs, list) or len(signs) > MAX_SIGNS_PER_LINE:
        return None
    source = _clean_text(row.get('source'), MAX_SHORT_TEXT_LENGTH)
    if source is None:
        source = 'second_opinion'
    return str(segment) + '/' + str(output), signs, source


def load_hints(root: Path, route: str, frame_sizes: dict) -> tuple[dict, dict]:
    """Hints of one route: ({frame_key: [hint, ...]}, statistics).

    ``frame_sizes`` maps the frame keys of the current report to (width, height);
    hints of other frames are not kept (counted as ``unknown_frames``).
    """
    by_frame = {}
    stats = empty_stats()
    directory = hints_directory(root, route)
    if directory is None:
        outside = Path(root) / HINTS_RELATIVE_DIR / route
        if outside.is_dir():
            stats['skipped_files'] = stats['skipped_files'] + 1
        return by_frame, stats
    for path in sorted(directory.glob('*.jsonl')):
        resolved = path.resolve()
        # Each file (or the file a symlink points to) must be inside the hints directory.
        if not resolved.is_relative_to(directory) or not resolved.is_file():
            stats['skipped_files'] = stats['skipped_files'] + 1
            continue
        try:
            if resolved.stat().st_size > MAX_HINT_FILE_BYTES:
                stats['skipped_files'] = stats['skipped_files'] + 1
                continue
            content = resolved.read_bytes()
        except OSError:
            stats['skipped_files'] = stats['skipped_files'] + 1
            continue
        stats['files'] = stats['files'] + 1
        for raw_line in content.split(b'\n'):
            if not raw_line.strip():
                continue
            stats['lines'] = stats['lines'] + 1
            try:
                text = raw_line.decode('utf-8')
            except UnicodeDecodeError:
                stats['skipped_lines'] = stats['skipped_lines'] + 1
                continue
            parsed = parse_line(text, route)
            if parsed is None:
                stats['skipped_lines'] = stats['skipped_lines'] + 1
                continue
            frame_key, signs, source = parsed
            if frame_key not in frame_sizes:
                stats['unknown_frames'] = stats['unknown_frames'] + 1
                continue
            width, height = frame_sizes[frame_key]
            for sign in signs:
                hint = parse_sign(sign, width, height, source)
                if hint is None:
                    stats['skipped_signs'] = stats['skipped_signs'] + 1
                    continue
                if frame_key not in by_frame:
                    by_frame[frame_key] = []
                by_frame[frame_key].append(hint)
                stats['signs'] = stats['signs'] + 1
    stats['frames'] = len(by_frame)
    return by_frame, stats


def frame_sizes_of(rows: list[dict]) -> dict:
    sizes = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get('frame_key'), str):
            sizes[row['frame_key']] = (row.get('width'), row.get('height'))
    return sizes


# ---------- suspicious frames ----------

def _join_words(words: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    if not words:
        return ''
    if len(words) == 1:
        return words[0]
    return ', '.join(words[:-1]) + ' and ' + words[-1]


UNREADABLE_LIMIT = 'an unreadable limit'


def _unique(words: list[str]) -> list[str]:
    """Words without repetitions; "an unreadable limit" only when nothing better was read."""
    result = []
    for word in words:
        if word not in result:
            result.append(word)
    if len(result) > 1 and UNREADABLE_LIMIT in result:
        result.remove(UNREADABLE_LIMIT)
    return result


def _with_value(word: str, value) -> str:
    if value is None:
        return word
    return word + ' ' + str(value)


def hint_words(hint: dict) -> str:
    """Short English description of a hint: '60', 'end of limit 60', 'zone 30', ..."""
    kind = hint['type']
    value = hint['value_kph']
    if kind == 'maximum_speed':
        if value is None:
            return UNREADABLE_LIMIT
        return str(value)
    elif kind == 'cancellation':
        return _with_value('end of limit', value)
    elif kind == 'zone':
        return _with_value('zone', value)
    elif kind == 'zone_end':
        return _with_value('end of zone', value)
    elif kind == 'variable_display':
        return _with_value('variable display', value)
    elif kind == 'road_marking':
        return _with_value('number on the road', value)
    else:
        return 'an uncertain sign'


def box_words(detection: dict) -> str:
    """Short English description of a speed box (model proposal or confirmed box)."""
    family = detection.get('family')
    value = detection.get('value_kph')
    if family == 'maximum_speed':
        if value is None:
            return UNREADABLE_LIMIT
        return str(value)
    elif family == 'cancellation':
        return _with_value('end of limit', value)
    elif family == 'zone':
        return _with_value('zone', value)
    else:
        return _with_value('variable display', value)


def speed_boxes(detections) -> list[dict]:
    """The boxes of a speed family in a list of detections (anything malformed is ignored)."""
    found = []
    if not isinstance(detections, list):
        return found
    for detection in detections:
        if isinstance(detection, dict) and detection.get('family') in SPEED_FAMILIES:
            found.append(detection)
    return found


def strong_hints(hints: list[dict]) -> list[dict]:
    """Hints the second reading is fairly sure of: confidence high or medium, type not 'unsure'."""
    found = []
    for hint in hints:
        if hint['confidence'] in ('high', 'medium') and hint['type'] != 'unsure':
            found.append(hint)
    return found


def _values(items: list[dict]) -> list[int]:
    values = []
    for item in items:
        value = item.get('value_kph')
        if isinstance(value, int) and not isinstance(value, bool) and value not in values:
            values.append(value)
    values.sort()
    return values


def _value_words(values: list[int]) -> str:
    words = []
    for value in values:
        words.append(str(value))
    return _join_words(words)


def _ordered(rows: list[dict]) -> list[dict]:
    """Report frames sorted by segment and image, with their key and model proposals."""
    frames = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get('frame_key')
        segment = row.get('segment_index')
        output = row.get('output_index')
        if not isinstance(key, str) or not isinstance(segment, int) or not isinstance(output, int):
            continue
        frames.append({'key': key, 'segment': segment, 'output': output, 'detections': row.get('detections'),
                       'order': frame_order(row)})
    frames.sort(key=lambda frame: frame['order'])
    return frames


def _neighbours(frames: list[dict], position: int) -> list[tuple[str, dict]]:
    """Previous and next frame: same segment, or the last/first frame of the adjacent segment."""
    current = frames[position]
    found = []
    if position > 0:
        previous = frames[position - 1]
        if previous['segment'] == current['segment'] or previous['segment'] == current['segment'] - 1:
            found.append(('previous', previous))
    if position + 1 < len(frames):
        following = frames[position + 1]
        if following['segment'] == current['segment'] or following['segment'] == current['segment'] + 1:
            found.append(('next', following))
    return found


def suspicious_frames(rows: list[dict], hints_by_frame: dict, confirmed: dict, *,
                      hints_available: bool) -> dict:
    """{frame_key: [{'rule': 'a'..'d', 'text': English reason}, ...]} of the suspicious frames.

    ``confirmed`` maps the confirmed frames to their reviewed detections.
    Confirmed frames are never suspicious. ``hints_available`` tells whether
    the route has a second reading at all: without one, rule b would flag
    every model proposal.
    """
    frames = _ordered(rows)
    result = {}
    for position in range(len(frames)):
        frame = frames[position]
        key = frame['key']
        if key in confirmed:
            continue
        hints = hints_by_frame.get(key, [])
        sure = strong_hints(hints)
        proposals = speed_boxes(frame['detections'])
        reasons = []

        # a) The second reading is fairly sure of a sign, the model proposes no speed sign.
        if sure and not proposals:
            words = []
            for hint in sure:
                words.append(hint_words(hint))
            reasons.append({'rule': 'a', 'text': 'second reading sees ' + _join_words(_unique(words))
                            + ', the model nothing'})

        # b) The model proposes a speed sign, the second reading sees nothing at all.
        if proposals and not hints and hints_available:
            words = []
            for proposal in proposals:
                words.append(box_words(proposal))
            reasons.append({'rule': 'b', 'text': 'the model sees ' + _join_words(_unique(words))
                            + ', the second reading nothing'})

        # c) Both read a value and no value is the same.
        hint_values = _values(hints)
        model_values = _values(proposals)
        if hint_values and model_values:
            common = False
            for value in hint_values:
                if value in model_values:
                    common = True
            if not common:
                reasons.append({'rule': 'c', 'text': 'second reading sees ' + _value_words(hint_values)
                                + ', the model ' + _value_words(model_values)})

        # d) A neighbour has a confirmed speed box or a fairly sure hint, this frame has neither proposal nor hint.
        if not proposals and not hints:
            for side, neighbour in _neighbours(frames, position):
                if neighbour['key'] in confirmed:
                    boxes = speed_boxes(confirmed[neighbour['key']])
                    if boxes:
                        words = []
                        for box in boxes:
                            words.append(box_words(box))
                        reasons.append({'rule': 'd', 'text': 'the ' + side + ' frame has a confirmed sign ('
                                        + _join_words(_unique(words)) + '), here nothing'})
                    # A reviewed neighbour takes precedence over unreviewed hints,
                    # even when its confirmed box list is intentionally empty.
                    continue
                neighbour_sure = strong_hints(hints_by_frame.get(neighbour['key'], []))
                if neighbour_sure:
                    words = []
                    for hint in neighbour_sure:
                        words.append(hint_words(hint))
                    reasons.append({'rule': 'd', 'text': 'in the ' + side + ' frame the second reading sees '
                                    + _join_words(_unique(words)) + ', here nothing'})

        if reasons:
            result[key] = reasons
    return result
