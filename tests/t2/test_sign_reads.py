"""The frames of the signs the comma read (nnslr_tools.sign_reads)."""
import json
import shutil
import subprocess

import pytest

from nnslr_tools.media import extract_frames
from nnslr_tools.preannotate import discover_frames
from nnslr_tools.sign_reads import (READ_INDEX_BASE, frame_order, frame_second, is_read_frame, read_frames,
                                    wanted_reads)

ROUTE = '00000089--0ac1c0fdec'
MS = 1_000_000


def merged(events):
    # Two segments of 40 frames at 20 a second; segment 1 starts 2 s after segment 0.
    frames = {seg: {'ids': list(range(40)), 'sof': [1000 * MS + seg * 2000 * MS + i * 50 * MS for i in range(40)]}
              for seg in (0, 1)}
    return {'events': [[t * MS, 'sign', kph] for t, kph in events], 'frames': frames}


def test_one_frame_per_run_of_reads_the_last_exposed_before_it():
    found = read_frames(merged([(1120, 50.0), (1320, 50.0), (1500, 80.0), (1700, 80.0), (3000, 80.0), (3010, 80.0)]))
    # 50 at 1.12 s: frame 2 (exposed at 1.10 s); its repeat 0.2 s later is the same run; 80 at 1.5 s: a new value;
    # 80 at 3.0 s (segment 1, frame 0): 1.3 s after the last 80, still the same run.
    assert [(r['decoded'], r['kph']) for r in found[0]] == [(2, 50.0), (10, 80.0)]
    assert 1 not in found


def test_a_gap_starts_a_new_run_and_a_read_outside_the_frames_is_dropped():
    found = read_frames(merged([(1120, 50.0), (3500, 50.0), (500, 30.0), (9000, 30.0)]))
    assert [(r['decoded'], r['kph']) for r in found[1]] == [(10, 50.0)]
    assert [(r['decoded'], r['kph']) for r in found[0]] == [(2, 50.0)]


def test_wanted_reads_skip_the_pictures_at_one_a_second():
    reads = [{'decoded': 20, 'kph': 50.0}, {'decoded': 25, 'kph': 50.0}, {'decoded': 65, 'kph': 30.0}]
    pictures = [{'output_index': 1, 'decoded_frame_index': 20}, {'output_index': 3, 'decoded_frame_index': 60}]
    assert [r['decoded'] for r in wanted_reads(reads, pictures)] == [25, 65]


def test_frame_order_puts_reads_between_the_seconds():
    rows = [{'segment_index': 0, 'output_index': 3}, {'segment_index': 0, 'output_index': READ_INDEX_BASE + 50,
                                                      'media_time_s': 2.5},
            {'segment_index': 0, 'output_index': 2}, {'segment_index': 1, 'output_index': 0}]
    assert [r['output_index'] for r in sorted(rows, key=frame_order)] == [2, READ_INDEX_BASE + 50, 3, 0]
    assert frame_second({'output_index': READ_INDEX_BASE + 70}) == 3.5
    assert is_read_frame(rows[1]) and not is_read_frame(rows[0])


@pytest.fixture
def raw_hevc(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe required for local media integration')
    video = tmp_path / 'fcamera.hevc'
    p = subprocess.run([
        'ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=red:s=64x48:r=20:d=2',
        '-c:v', 'libx265', '-x265-params', 'pools=1:frame-threads=1:log-level=error:bframes=0', str(video),
    ], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return video


def test_read_pictures_survive_a_new_extraction_and_never_collide(raw_hevc, tmp_path):
    output = tmp_path / 'frames'
    extract_frames(raw_hevc, output, fps=1)
    reads = extract_frames(raw_hevc, output, decoded=[13, 27], prefix='read', index_base=READ_INDEX_BASE)
    assert [(r.output_index, r.decoded_frame_index, r.media_time_s) for r in reads] == [
        (READ_INDEX_BASE + 13, 13, pytest.approx(0.65)), (READ_INDEX_BASE + 27, 27, pytest.approx(1.35))]
    assert sorted(p.name for p in output.glob('read_*.png')) == ['read_00001013.png', 'read_00001027.png']
    # The pictures at 1 a second taken out again leave the reads alone, and the reads again replace only theirs.
    extract_frames(raw_hevc, output, fps=1, overwrite=True)
    assert len(list(output.glob('read_*.png'))) == 2
    extract_frames(raw_hevc, output, decoded=[27], prefix='read', index_base=READ_INDEX_BASE, overwrite=True)
    assert [p.name for p in output.glob('read_*.png')] == ['read_00001027.png']
    assert len(list(output.glob('frame_*.png'))) == 2


def test_discovery_adds_the_reads_in_time_order(tmp_path):
    directory = tmp_path / 'derived/frames' / ROUTE / '0'
    directory.mkdir(parents=True)

    def row(name, index, time, **extra):
        image = directory / name
        image.write_bytes(b'fake image')
        return {'image_path': str(image), 'route_id': ROUTE, 'segment_index': 0, 'output_index': index,
                'media_time_s': time, **extra}
    pictures = [row(f'frame_{i:08}.png', i, float(i), requested_sample_fps=1) for i in range(3)]
    reads = [row('read_00001030.png', 1030, 1.5, comma_read_kph=20.0)]
    (directory / 'frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pictures))
    (directory / 'sign_reads.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in reads))
    found = discover_frames(tmp_path, ROUTE)
    assert [f['frame_key'] for f in found] == ['0/0', '0/1', '0/1030', '0/2']
    assert found[2]['comma_read_kph'] == 20.0 and 'comma_read_kph' not in found[0]
