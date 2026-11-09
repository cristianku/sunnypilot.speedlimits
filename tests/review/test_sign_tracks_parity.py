"""The signs on every frame match Sign review's boxes on the frames both read (nnslr_tools.sign_tracks.parity): the real
networks in use on the first 41 frames of a real drive at 1344 x 760, on the CPU, into a temporary folder. Skipped
without torch (the site's .venv) or without that drive, its video and its Sign review report by the networks in use."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROUTE = '00000128--edfa65432f'
ROOT = Path(os.environ.get('NNSLR_DATA_ROOT') or Path(__file__).resolve().parents[2] / 'data')


def test_the_first_frames_of_a_drive_match_sign_review(tmp_path):
    pytest.importorskip('torch')
    from nnslr_tools import drive_video, sign_tracks
    networks = sign_tracks.networks_in_use(ROOT)
    if networks is None or drive_video.index(ROOT, ROUTE) is None:
        pytest.skip('no networks in use or no video of the drive')
    try:
        rows = sign_tracks.report_rows(ROOT, ROUTE)
    except (OSError, ValueError):
        pytest.skip('no Sign review report of the drive')
    finder = sign_tracks.load_finder(ROOT, 'cpu')
    result = sign_tracks.run_drive(ROOT, ROUTE, finder, limit_frames=41, out_dir=tmp_path, log=lambda text: None)
    assert result['path'].parent == tmp_path
    track = json.loads(result['path'].read_text(encoding='utf-8'))
    found = sign_tracks.parity(ROOT, track, rows)
    if not found['frames']:
        pytest.skip('Sign review has no frame of these networks in the first 41')
    assert found['matching'] == found['frames'], found
