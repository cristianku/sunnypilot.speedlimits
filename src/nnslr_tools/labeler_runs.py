"""The runs of the road labeler for the portal (the Road labeler card at the bottom of Models,
GET /api/panel/road-labeler): every data/runs/<run>/ with the result.json of nnslr road-labeler-train, newest first,
with what it measured on validation and which of its heads write into Speed review. It reads files only (no torch, no
GPU): the heads that write are road_labeler.held_back of the result's own train sides and gate, the same rule as nnslr
road-labeler-predict.

What a run writes from the portal's "Prepare Speed review": the marks of the kinds of road whose head writes (never over a
person's mark, removal or check). The roadworks and the road types write only with --write, from a terminal.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from nnslr_tools import road_labeler

# The runs shown on the card, newest first.
SHOWN = 8
RUN_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}')


def _result(run_dir: Path) -> dict | None:
    """A run's result.json when it is the road labeler's (its train sides and kinds), else None."""
    try:
        result = json.loads((run_dir / road_labeler.RESULT).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict) or not isinstance(result.get('train_sides'), dict) \
            or not isinstance(result.get('kinds'), list):
        return None
    return result


def trained(root: Path, name: str) -> bool:
    """Whether data/runs/<name> is a road labeler run with its checkpoint, the run "Prepare Speed review" can use."""
    if not isinstance(name, str) or not RUN_NAME.fullmatch(name):
        return False
    run_dir = Path(root) / 'runs' / name
    return (run_dir / road_labeler.CHECKPOINT).is_file() and _result(run_dir) is not None


def trained_runs(root: Path) -> list[str]:
    """The names of the road labeler runs with their checkpoint, in order."""
    return sorted(path.parent.name for path in (Path(root) / 'runs').glob('*/' + road_labeler.CHECKPOINT)
                  if trained(root, path.parent.name))


def _heads(result: dict) -> list[dict]:
    """The kinds of road and the roadworks of a run: their precision and recall on validation, and whether they write
    (or why not)."""
    validation = result.get('validation') if isinstance(result.get('validation'), dict) else {}
    measured = {**(validation.get('kinds') or {}), 'works': validation.get('works')}
    gate = result.get('gate') if isinstance(result.get('gate'), dict) else {}
    held = road_labeler.held_back({'kinds': result['kinds'], 'trained': road_labeler.trained_heads(result['train_sides']),
                                   'earned': gate, 'road_types': result.get('road_types') or ()})
    heads = []
    for head in (*result['kinds'], 'works'):
        item = measured.get(head) if isinstance(measured.get(head), dict) else {}
        why = 'a dry run never writes' if result.get('dry_run') else held.get(head)
        heads.append({'head': head, 'inside': item.get('inside'), 'outside': item.get('outside'),
                      'precision': item.get('precision'), 'recall': item.get('recall'),
                      'writes': why is None, 'why': why,
                      # The roadworks write only with --write (never from the portal); the kinds by default.
                      'with_write_only': head == 'works'})
    return heads


def _types(result: dict) -> dict:
    """The road type on validation: the share right on every step with one and on the steps a person set, beside the
    map's own share there, and the road types that would write with --write."""
    validation = result.get('validation') if isinstance(result.get('validation'), dict) else {}
    types = validation.get('types') if isinstance(validation.get('types'), dict) else {}
    gate = result.get('gate') if isinstance(result.get('gate'), dict) else {}
    writing = [] if result.get('dry_run') else [
        name for name in result.get('road_types') or () if isinstance(gate.get(f'type:{name}'), dict)
        and gate[f'type:{name}'].get('why') is None]
    return {key: types.get(key) for key in ('steps', 'right', 'person_steps', 'right_on_person', 'map_on_person')} \
        | {'writing_with_write': writing}


def summary(root: Path) -> dict:
    """{runs: the newest SHOWN road labeler runs, total: how many there are, drives: how many drives it can read}.
    A run: {run, dry_run, finished (ISO time of its result), drives, steps, epochs, checkpoint, types, heads}."""
    found = []
    for path in (Path(root) / 'runs').glob('*/' + road_labeler.RESULT):
        run_dir = path.parent
        if not RUN_NAME.fullmatch(run_dir.name):
            continue
        result = _result(run_dir)
        if result is None:
            continue
        try:
            stamp = path.stat().st_mtime
        except OSError:
            continue
        drives = result.get('drives') if isinstance(result.get('drives'), dict) else {}
        found.append((stamp, {
            'run': run_dir.name, 'dry_run': bool(result.get('dry_run')),
            'finished': datetime.fromtimestamp(stamp).astimezone().isoformat(timespec='seconds'),
            'drives': len(drives), 'steps': sum(value for value in drives.values() if isinstance(value, int)),
            'epochs': len(result.get('history') or ()),
            'checkpoint': (run_dir / road_labeler.CHECKPOINT).is_file(),
            'types': _types(result), 'heads': _heads(result)}))
    found.sort(key=lambda item: item[0], reverse=True)
    from nnslr_tools.road_labeler_data import labeler_routes
    try:
        drives_ready = len(labeler_routes(Path(root)))
    except Exception:  # a store being written: the card says nothing of it
        drives_ready = None
    return {'runs': [run for _, run in found[:SHOWN]], 'total': len(found), 'drives': drives_ready}
