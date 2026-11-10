"""How far a long command is, for the panel of the review site.

A command reports ``report(task, done, total, text)`` while it works; the last report is written to the file of
NNSLR_PROGRESS_FILE, which the panel gives each of its jobs (``<data root>/logs/progress-<job id>.json``: two jobs at
once, one on each GPU, never read each other's), else to ``<data root>/logs/progress.json`` (the data root of
NNSLR_DATA_ROOT, as for every nnslr command), at most once a second and always at the end. The panel
(nnslr_tools.panel) shows it as the progress of the step of the running job that the task belongs to. Reporting never
fails a command: a write error is ignored.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from nnslr_tools.sync import DEFAULT_DATA_ROOT

INTERVAL_SECONDS = 1.0
_last = {'time': 0.0}


def progress_path() -> Path:
    """The file of the reports: the job's own (NNSLR_PROGRESS_FILE, set by the panel), else the data root's."""
    own = os.environ.get('NNSLR_PROGRESS_FILE')
    if own:
        return Path(own).expanduser()
    root = Path(os.environ.get('NNSLR_DATA_ROOT') or DEFAULT_DATA_ROOT).expanduser()
    return root / 'logs' / 'progress.json'


def report(task: str, done: int, total: int, text: str = '') -> None:
    """The task (import, validate, split, detector, reader, evaluate, hard_examples, export) is at done of total."""
    now = time.monotonic()
    if done < total and now - _last['time'] < INTERVAL_SECONDS:
        return
    _last['time'] = now
    path = progress_path()
    record = {'task': task, 'done': int(done), 'total': int(total), 'text': text, 'pid': os.getpid(),
              'at': time.time()}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
        temporary.write_text(json.dumps(record) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    except OSError:
        pass
