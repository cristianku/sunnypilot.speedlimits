"""Every frame of the road video in Sign review (nnslr_tools.video_frames): GOP previews, a frame between the seconds
saved when it is confirmed (with boxes or as no sign), as a real picture of the report that Propose again and the
import keep."""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from nnslr_tools import cli, video_frames
from nnslr_tools.media import extract_frames, extracted_frame_rows
from nnslr_tools.preannotate import discover_frames
from nnslr_tools.review import write_review
from nnslr_tools.review_db import ReviewStore, db_path
from review_helpers import ROUTE, RUN_ID, RunningServer, json_of, proposal

NEW_RUN = '20261022T100000Z-0123abce'
BOX = {'bbox_xyxy': [4, 4, 20, 20], 'family': 'maximum_speed', 'value_kph': 50, 'value_state': 'reviewed',
       'score': None, 'model_label': None, 'class_id': None, 'review_state': 'pending', 'applicability': 'unresolved',
       'supplementary_panel': 'unknown', 'detection_id': 'manual-1'}


def report_of(root, rows, run_id):
    """A pre-annotation report of these frame rows: a proposal on the first one."""
    lines = []
    directory = root / 'derived/preannotations' / ROUTE / run_id
    (directory / 'images').mkdir(parents=True)
    for ordinal, row in enumerate(rows):
        (directory / 'images' / f'{ordinal:08d}.jpg').write_bytes(b'jpeg preview')
        lines.append({**row, 'schema_version': 1, 'kind': 'preannotated_frame', 'run_id': run_id, 'width': 64,
                      'height': 48, 'preview_path': f'images/{ordinal:08d}.jpg',
                      'detections': [proposal(row['output_index'])] if ordinal == 0 else [],
                      'review_state': 'pending', 'ground_truth': False, 'model': {'repository': 'synthetic'}})
    (directory / 'preannotations.jsonl').write_text(''.join(json.dumps(line) + '\n' for line in lines))
    (directory / 'run.json').write_text(json.dumps({'run_id': run_id, 'route_id': ROUTE}))
    write_review(directory, lines, run_id)
    (root / 'derived/preannotations' / ROUTE / 'latest.json').write_text(json.dumps({'run_id': run_id}))
    return lines


@pytest.fixture
def drive(tmp_path):
    """A 3-second road video at 20 frames a second, a keyframe every 20, its pictures at 1 a second and a report."""
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe required for local media integration')
    video = tmp_path / 'raw/routes' / ROUTE / '0' / 'fcamera.hevc'
    video.parent.mkdir(parents=True)
    made = subprocess.run([
        'ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=s=64x48:r=20:d=3', '-pix_fmt', 'yuv420p',
        '-c:v', 'libx265', '-x265-params', 'pools=1:frame-threads=1:log-level=error:bframes=0:keyint=20:min-keyint=20',
        str(video)], capture_output=True, text=True)
    assert made.returncode == 0, made.stderr
    folder = tmp_path / 'derived/frames' / ROUTE / '0'
    pictures = extracted_frame_rows(extract_frames(video, folder, fps=1), route_id=ROUTE, segment_index=0)
    (folder / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in pictures))
    rows = report_of(tmp_path, discover_frames(tmp_path, ROUTE), RUN_ID)
    server = RunningServer(tmp_path)
    yield tmp_path, video, rows, server
    server.close()


def body(detections, run_id=RUN_ID, confirmed=True):
    return {'run_id': run_id, 'reviewer': 'Cristian', 'confirmed': confirmed, 'detections': detections,
            'site_group': None, 'encounter_id': None}


def test_a_gop_has_the_pixels_of_a_decode_of_the_whole_video(drive, tmp_path):
    root, video, rows, server = drive
    assert video_frames.gop_of(video, 25) == (20, 39) and video_frames.gop_of(video, 59) == (40, 59)
    reference = tmp_path / 'reference.jpg'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-vf', 'select=eq(n\\,25)', '-vsync', '0',
                    '-q:v', video_frames.JPEG_QUALITY, str(reference)], check=True)
    assert video_frames.preview(root, ROUTE, 0, 25).read_bytes() == reference.read_bytes()
    assert len(list((root / video_frames.CACHE_DIR / ROUTE / '0').glob('*.jpg'))) == 20
    # The page: the frames of the GOP, the picture at 1 a second under its own key.
    status, _response, data = server.request('GET', f'/api/route/{ROUTE}/video/0/25')
    assert status == 200, data
    gop = json_of(data)
    assert gop['frame_count'] == 60 and gop['gop'] == [20, 39] and gop['seconds'] is None
    keys = {frame['decoded']: frame['frame_key'] for frame in gop['frames']}
    assert keys[20] == '0/1' and keys[25] == '0/1025' and len(keys) == 20
    status, response, data = server.request('GET', f'/route/{ROUTE}/video/0/25.jpg')
    assert status == 200 and response.getheader('Content-Type') == 'image/jpeg' and data == reference.read_bytes()
    assert server.request('GET', f'/api/route/{ROUTE}/video/0/60')[0] == 404
    # The cache stays within its budget, the least recently shown first.
    video_frames.prune(root, budget=0)
    assert not list((root / video_frames.CACHE_DIR).glob('*/*/*.jpg'))


def wide_video(root, frames_s, source='smptebars'):
    """A wide road video of segment 0 (ecamera.hevc), another picture from the road video's."""
    video = root / 'raw/routes' / ROUTE / '0' / video_frames.WIDE_VIDEO
    made = subprocess.run([
        'ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', f'{source}=s=64x48:r=20:d={frames_s}', '-pix_fmt',
        'yuv420p', '-c:v', 'libx265', '-x265-params',
        'pools=1:frame-threads=1:log-level=error:bframes=0:keyint=20:min-keyint=20', '-f', 'hevc', str(video)],
        capture_output=True, text=True)
    assert made.returncode == 0, made.stderr
    return video


def test_the_wide_camera_gives_the_same_frame_only_with_the_frames_of_the_road_video(drive, tmp_path):
    root, video, _rows, _server = drive
    with pytest.raises(video_frames.VideoFrameError, match='no wide road video'):
        video_frames.preview(root, ROUTE, 0, 25, wide=True)
    wide = wide_video(root, 3)
    reference = tmp_path / 'wide-reference.jpg'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'hevc', '-i', str(wide), '-vf', 'select=eq(n\\,25)', '-vsync',
                    '0', '-q:v', video_frames.JPEG_QUALITY, str(reference)], check=True)
    shown = video_frames.preview(root, ROUTE, 0, 25, wide=True)
    assert shown.read_bytes() == reference.read_bytes() and shown.parent.name == '0-wide'
    assert shown.read_bytes() != video_frames.preview(root, ROUTE, 0, 25).read_bytes()
    # One frame short of the road video: its frame n may not be the road's, so it is not shown.
    wide_video(root, 2.95)
    with pytest.raises(video_frames.VideoFrameError, match='not the frames of the road video'):
        video_frames.wide_video_of(root, ROUTE, 0)
    video_frames.prune(root, budget=0)
    assert not list((root / video_frames.CACHE_DIR).glob('*/*/*.jpg'))


def test_a_frame_between_the_seconds_is_saved_with_a_box_and_kept_by_propose_again(drive, capsys):
    root, video, rows, server = drive
    manifest = root / 'derived/frames' / ROUTE / '0' / video_frames.VIDEO_MANIFEST
    # Not confirmed: nothing taken out of the video.
    assert server.post_frame(ROUTE, '0/1025', body([BOX], confirmed=False))[0] == 404 and not manifest.exists()
    # The picture at 1 a second under a key of the frames between: never a second key for the same image.
    assert server.post_frame(ROUTE, '0/1020', body([BOX]))[0] == 404 and not manifest.exists()
    # A save refused after the picture was taken out leaves nothing behind.
    status, _response, data = server.post_frame(ROUTE, '0/1025', body([{**BOX, 'family': 'no-such-family'}]))
    assert status == 400 and not manifest.exists() and not list(manifest.parent.glob('video_*.png'))
    status, _response, data = server.post_frame(ROUTE, '0/1025', body([BOX]))
    assert status == 200, data
    picture = manifest.parent / 'video_00001025.png'
    assert picture.is_file() and [row['decoded_frame_index'] for row in map(json.loads, manifest.open())] == [25]
    added = server.app.report(ROUTE)[2]['0/1025']
    assert added['image_path'] == str(picture.relative_to(root)) and added['preview_path'] == 'video/0/25.jpg'
    assert [row['frame_key'] for row in server.app.report(ROUTE)[1]] == ['0/0', '0/1', '0/1025', '0/2']
    # The page made again (a reload) has it among its frames.
    status, _response, data = server.request('GET', f'/route/{ROUTE}/')
    assert status == 200 and b'"frame_key": "0/1025"' in data
    store = ReviewStore(db_path(root))
    saved = store.review(ROUTE, RUN_ID, '0/1025')
    assert saved['confirmed'] and saved['payload']['image_path'] == added['image_path']
    # Its picture is the same as the decode of the whole video.
    same = extract_frames(video, root / 'check', decoded=[25])
    assert (root / 'check' / 'frame_00000000.png').read_bytes() == picture.read_bytes() and same
    # Saved again: the same picture, no second row.
    status, _response, data = server.post_frame(ROUTE, '0/1025', {**body([BOX]), 'base_revision': 1})
    assert status == 200, data
    assert len(manifest.read_text().splitlines()) == 1
    # Propose again: the frame is in the new report and its confirmation is carried onto it.
    found = discover_frames(root, ROUTE)
    assert [row['frame_key'] for row in found] == ['0/0', '0/1', '0/1025', '0/2']
    report_of(root, found, NEW_RUN)
    keys = [row['frame_key'] for row in found]
    assert store.carry_confirmed(route_id=ROUTE, from_run=RUN_ID, to_run=NEW_RUN, frame_keys=keys, client='t') == 1
    assert server.app.report(ROUTE)[2]['0/1025']['run_id'] == NEW_RUN
    assert server.app.report(ROUTE)[2]['0/1025']['preview_path'] == 'images/00000002.jpg'
    # The import: the frame between the seconds with its box, on its own picture.
    try:
        code = cli.main(['import-annotations', '--from-db', '--data-root', str(root)])
    except SystemExit as exc:
        code = exc.code
    out = capsys.readouterr()
    assert code == 0, out.out + out.err
    objects = [json.loads(line) for line in (root / 'annotations/objects.jsonl').read_text().splitlines()]
    imported = [row for row in objects if row['output_index'] == 1025]
    assert imported and all(row['image_path'] == added['image_path'] for row in imported)


def test_a_frame_between_the_seconds_with_no_sign_is_saved_too(drive):
    root, video, rows, server = drive
    status, _response, data = server.post_frame(ROUTE, '0/1045', body([]))
    assert status == 200, data
    manifest = root / 'derived/frames' / ROUTE / '0' / video_frames.VIDEO_MANIFEST
    assert [row['decoded_frame_index'] for row in map(json.loads, manifest.open())] == [45]
    assert [row['frame_key'] for row in server.app.report(ROUTE)[1]] == ['0/0', '0/1', '0/2', '0/1045']
    saved = ReviewStore(db_path(root)).review(ROUTE, RUN_ID, '0/1045')
    assert saved['confirmed'] and saved['payload']['detections'] == []


def test_a_later_read_of_the_same_frame_keeps_one_row(drive):
    root, video, rows, server = drive
    status, _response, data = server.post_frame(ROUTE, '0/1025', body([BOX]))
    assert status == 200, data
    folder = root / 'derived/frames' / ROUTE / '0'
    read = json.loads((folder / video_frames.VIDEO_MANIFEST).read_text())
    shutil.copy(read['image_path'], folder / 'read_00001025.png')
    read.update(image_path=str(folder / 'read_00001025.png'), comma_read_kph=50.0)
    (folder / 'sign_reads.jsonl').write_text(json.dumps(read) + '\n')
    found = [row for row in discover_frames(root, ROUTE) if row['frame_key'] == '0/1025']
    assert len(found) == 1 and found[0]['comma_read_kph'] == 50.0


def test_apply_previous_and_fit_box_work_on_a_frame_of_the_video_not_saved(drive):
    """A frame between the seconds not saved yet: Apply previous carries the boxes of the frame of the report just
    before it, each to check (it has no proposals), and Fit box fits on its picture. Neither saves anything."""
    root, video, rows, server = drive
    manifest = root / 'derived/frames' / ROUTE / '0' / video_frames.VIDEO_MANIFEST

    def apply(key):
        status, _response, data = server.request('POST', f'/api/route/{ROUTE}/frame/{key}/apply-previous',
                                                 body={'run_id': RUN_ID, 'detections': []})
        return status, json_of(data)

    status, answer = apply('0/1025')
    assert status == 409 and answer['error'] == 'previous_not_confirmed' and '0/1' in answer['message']
    assert server.post_frame(ROUTE, '0/1', body([BOX]))[0] == 200
    status, answer = apply('0/1025')
    assert status == 200, answer
    assert answer['source_frame_key'] == '0/1' and answer['save'] == {'run_id': RUN_ID, 'base_revision': 0}
    carried = answer['detections']
    assert [box['bbox_xyxy'] for box in carried] == [BOX['bbox_xyxy']] and carried[0]['value_kph'] == 50
    assert carried[0]['check_position'] is True and answer['to_check']
    # The frame before is the one of the report before it in time: the next second starts at its own picture.
    status, answer = apply('0/1041')
    assert status == 409 and answer['error'] == 'previous_not_confirmed' and '0/2' in answer['message']
    # Not such a frame: the picture at 1 a second under another key, past the end of the video.
    assert apply('0/1020')[0] == 404 and apply('0/1060')[0] == 404
    fit = f'/api/route/{ROUTE}/frame/0/1025/fit'
    status, _response, data = server.request('POST', fit, body={'run_id': RUN_ID, 'boxes': [BOX['bbox_xyxy']]})
    assert status == 200, data
    assert len(json_of(data)['boxes']) == 1
    status, _response, data = server.request('POST', fit, body={'run_id': NEW_RUN, 'boxes': [BOX['bbox_xyxy']]})
    assert status == 409 and json_of(data)['error'] == 'stale_report'
    assert not manifest.exists() and not list(manifest.parent.glob('video_*.png'))
    # The page knows the segments with their video, and how long it is once a frame of it is saved.
    status, _response, data = server.request('GET', f'/route/{ROUTE}/')
    assert status == 200 and b'"videos": {"0": null}' in data
    # A box still to check is refused, as on any frame; checked ("Here", or Fit box on its sign) it is saved.
    assert server.post_frame(ROUTE, '0/1025', body(carried))[0] == 400 and not manifest.exists()
    checked = [{name: value for name, value in box.items() if name != 'check_position'} for box in carried]
    assert server.post_frame(ROUTE, '0/1025', body(checked))[0] == 200
    status, answer = apply('0/1026')
    assert status == 200 and answer['source_frame_key'] == '0/1025'
    status, _response, data = server.request('GET', f'/route/{ROUTE}/')
    assert status == 200 and b'"videos": {"0": 60}' in data
