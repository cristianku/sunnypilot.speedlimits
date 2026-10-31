"""The video of a drive for the Drive view of Results (nnslr_tools.drive_video): the comma's minutes of road video joined
in order into one MP4 with an index of where each minute starts, made once and served with byte ranges."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from nnslr_tools import drive_video
from review_helpers import ROUTE, RunningServer, make_report

needs_ffmpeg = pytest.mark.skipif(shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None,
                                  reason='ffmpeg is not installed')


def _minute(root: Path, segment: int, seconds: int = 1) -> Path:
    """A raw HEVC "minute" as the comma writes it (here a second or two of a test picture at 20 frames a second)."""
    folder = root / 'raw' / 'routes' / ROUTE / str(segment)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'fcamera.hevc'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                    f'testsrc=size=320x180:rate=20:duration={seconds}', '-c:v', 'libx265', '-x265-params', 'log-level=error',
                    '-f', 'hevc', str(path)], check=True, timeout=120)
    return path


def test_minutes_in_order_and_nothing_to_make_without_them(tmp_path):
    for segment in (10, 2, 9):
        folder = tmp_path / 'raw' / 'routes' / ROUTE / str(segment)
        folder.mkdir(parents=True)
        (folder / 'fcamera.hevc').write_bytes(b'x')
    (tmp_path / 'raw' / 'routes' / ROUTE / 'notes').mkdir()
    assert [segment for segment, _ in drive_video.minutes(tmp_path, ROUTE)] == [2, 9, 10]
    assert drive_video.state(tmp_path, '00000002--abcdef1234')['state'] == 'missing'
    assert drive_video.state(tmp_path, '../etc')['state'] == 'missing'


@needs_ffmpeg
def test_a_drive_becomes_one_video_with_where_each_minute_starts(tmp_path):
    _minute(tmp_path, 3, 1)
    _minute(tmp_path, 5, 2)
    shares = []
    made = drive_video.make(tmp_path, ROUTE, shares.append)
    assert [(item['segment'], item['start'], item['frames']) for item in made['segments']] == [(3, 0.0, 20), (5, 1.0, 40)]
    assert made['duration'] == 3.0 and shares and shares[-1] <= 1
    video, index = drive_video.paths(tmp_path, ROUTE)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,width',
                            '-of', 'json', str(video)], capture_output=True, text=True, check=True)
    assert json.loads(probe.stdout)['streams'][0] == {'codec_name': 'h264', 'width': drive_video.WIDTH}
    assert not list(video.parent.glob('*.part.mp4')) and not list(video.parent.glob('*.parts'))
    # Encoded minute by minute and joined: every frame is there, in order.
    frames = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                             'stream=nb_read_packets', '-of', 'csv=p=0', str(video)], capture_output=True, text=True, check=True)
    assert int(frames.stdout.strip()) == 60
    assert drive_video.state(tmp_path, ROUTE)['state'] == 'ready'
    # A minute changed (copied again from the comma): the video must be made again.
    _minute(tmp_path, 5, 1)
    assert drive_video.index(tmp_path, ROUTE) is None
    assert json.loads(index.read_text())['route'] == ROUTE


@needs_ffmpeg
def test_the_site_serves_the_video_with_byte_ranges(tmp_path):
    make_report(tmp_path)
    _minute(tmp_path, 0, 1)
    drive_video.make(tmp_path, ROUTE)
    size = drive_video.paths(tmp_path, ROUTE)[0].stat().st_size
    server = RunningServer(tmp_path)
    try:
        status, _, data = server.request('GET', '/api/panel/drive-video/' + ROUTE)
        answer = json.loads(data)
        assert status == 200 and answer['state'] == 'ready' and answer['url'] == f'/route/{ROUTE}/video.mp4'
        status, response, data = server.request('GET', f'/route/{ROUTE}/video.mp4', headers={'Range': 'bytes=10-109'})
        assert status == 206 and len(data) == 100 and response.getheader('Content-Range') == f'bytes 10-109/{size}'
        assert response.getheader('Content-Type') == 'video/mp4' and response.getheader('Accept-Ranges') == 'bytes'
        status, _, data = server.request('GET', f'/route/{ROUTE}/video.mp4')
        assert status == 200 and len(data) == size
        status, _, _ = server.request('GET', f'/route/{ROUTE}/video.mp4', headers={'Range': f'bytes={size}-'})
        assert status == 416
        # The control pages may play it (media-src).
        status, response, _ = server.request('GET', '/evaluate')
        assert "media-src 'self'" in response.getheader('Content-Security-Policy')
    finally:
        server.close()
