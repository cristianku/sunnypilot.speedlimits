"""A drive recorded at full definition (2688 x 1520, the comma's switch Record road at full resolution) beside the old
ones: rows, pictures and boxes stay 1344 x 760, the frame as recorded stays beside each picture."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nnslr_tools import video_frames
from nnslr_tools.crop_cache import EXACT, WINDOW, build_crop_cache
from nnslr_tools.media import (FRAME_SIZE, FULL_SIZE, extract_frames, extracted_frame_rows, full_picture,
                               remove_picture, save_preview)
from review_helpers import ROUTE

X265 = 'pools=1:frame-threads=1:log-level=error:bframes=0:keyint=20:min-keyint=20'


def recorded(path, size):
    """A 2-second raw HEVC road video of this size, 20 frames a second, a keyframe every 20."""
    path.parent.mkdir(parents=True, exist_ok=True)
    made = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'testsrc2=s={size[0]}x{size[1]}:r=20:d=2',
                           '-pix_fmt', 'yuv420p', '-c:v', 'libx265', '-preset', 'ultrafast', '-x265-params', X265,
                           '-f', 'hevc', str(path)], capture_output=True, text=True)
    assert made.returncode == 0, made.stderr
    return path


@pytest.fixture(scope='module')
def full_video(tmp_path_factory):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe required for local media integration')
    return recorded(tmp_path_factory.mktemp('full') / 'fcamera.hevc', FULL_SIZE)


@pytest.fixture
def segment(tmp_path, full_video):
    """(data root, its video) of a drive at full definition, one segment, pictures at 1 a second."""
    video = tmp_path / 'raw/routes' / ROUTE / '0' / 'fcamera.hevc'
    video.parent.mkdir(parents=True)
    shutil.copyfile(full_video, video)
    folder = tmp_path / 'derived/frames' / ROUTE / '0'
    pictures = extracted_frame_rows(extract_frames(video, folder, fps=1), route_id=ROUTE, segment_index=0)
    (folder / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in pictures))
    return tmp_path, video


def test_the_picture_is_reduced_and_the_frame_stays_beside_it(tmp_path, full_video):
    rows = extract_frames(full_video, tmp_path / 'frames', fps=1)
    assert len(rows) == 2
    for row in rows:
        assert (row.width, row.height) == FRAME_SIZE
        with Image.open(row.image_path) as picture, Image.open(full_picture(row.image_path)) as full:
            assert picture.size == FRAME_SIZE and full.size == FULL_SIZE
            reduced = np.asarray(full.convert('RGB').resize(FRAME_SIZE, Image.Resampling.BOX), dtype=np.int16)
            # The same frame: only the JPEG and the two scalers tell them apart.
            assert np.abs(reduced - np.asarray(picture.convert('RGB'), dtype=np.int16)).mean() < 6


def test_a_video_of_another_size_is_not_touched(tmp_path):
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required')
    video = recorded(tmp_path / 'fcamera.hevc', (1928, 1208))
    rows = extract_frames(video, tmp_path / 'frames', fps=1)
    assert (rows[0].width, rows[0].height) == (1928, 1208)
    assert not list((tmp_path / 'frames').glob('*.jpg'))


def test_extracting_again_leaves_no_stale_frame(tmp_path, full_video):
    folder = tmp_path / 'frames'
    extract_frames(full_video, folder, fps=2)
    assert len(list(folder.glob('*.full.jpg'))) == 4
    extract_frames(full_video, folder, fps=1, overwrite=True)
    assert len(list(folder.glob('*.png'))) == len(list(folder.glob('*.full.jpg'))) == 2


def test_a_frame_between_the_seconds_keeps_its_full_frame_and_loses_it_with_its_picture(segment):
    root, video = segment
    made, new = video_frames.materialize(root, ROUTE, 0, 25)
    assert new and (made['width'], made['height']) == FRAME_SIZE
    full = full_picture(made['image_path'])
    with Image.open(full) as picture:
        assert picture.size == FULL_SIZE
    video_frames.forget(root, ROUTE, 0, 25)
    assert not full.exists()


def test_a_gop_gives_the_frame_of_the_whole_video(segment, tmp_path):
    root, video = segment
    whole = extract_frames(video, tmp_path / 'whole', decoded=[25])[0]
    made, _new = video_frames.materialize(root, ROUTE, 0, 25)
    assert made['image_sha256'] == whole.image_sha256


def test_nv12_comes_at_the_size_the_cutter_reads_or_as_recorded(segment):
    _root, video = segment
    data, width, height = video_frames.nv12_frame(video, 5)
    assert (width, height) == FRAME_SIZE and len(data) == width * height * 3 // 2
    data, width, height = video_frames.nv12_frame(video, 5, native=True)
    assert (width, height) == FULL_SIZE and len(data) == width * height * 3 // 2


def test_the_preview_is_the_full_frame_when_there_is_one(tmp_path, full_video):
    row = extract_frames(full_video, tmp_path / 'frames', fps=1)[0]
    with Image.open(row.image_path) as picture:
        save_preview(picture.convert('RGB'), row.image_path, tmp_path / 'full.jpg')
    with Image.open(tmp_path / 'full.jpg') as preview:
        assert preview.size == FULL_SIZE
    remove_picture(row.image_path)
    Image.new('RGB', FRAME_SIZE).save(tmp_path / 'plain.png')
    save_preview(Image.new('RGB', FRAME_SIZE), tmp_path / 'plain.png', tmp_path / 'plain.jpg')
    with Image.open(tmp_path / 'plain.jpg') as preview:
        assert preview.size == FRAME_SIZE


def stored_crops(root, path, box, kind):
    records = [{'image_path': path, 'image_sha256': 'a', 'bbox_xyxy': box}]
    return build_crop_cache(root, records, [kind], size=160, scheme='v2', content={})


def test_a_training_crop_comes_from_the_full_frame(tmp_path, full_video):
    row = extract_frames(full_video, tmp_path / 'derived/frames', fps=1)[0]
    path = str(Path(row.image_path).relative_to(tmp_path))
    box = [300.0, 100.0, 340.0, 141.0]
    cache, summary = stored_crops(tmp_path, path, box, EXACT)
    assert summary['full_frames'] == 1
    with Image.open(full_picture(row.image_path)) as source:
        wanted = source.convert('RGB').crop((600, 200, 680, 282)).resize((160, 160), Image.BILINEAR)
    assert np.array_equal(np.asarray(cache.crop(0)), np.asarray(wanted))
    # A jittered crop is cut from the same frame: a box of the picture, the pixels of the frame.
    cache, _summary = stored_crops(tmp_path, path, box, WINDOW)
    crop = cache.crop(0, box).resize((160, 160), Image.BILINEAR)
    assert np.abs(np.asarray(crop, dtype=np.int16) - np.asarray(wanted, dtype=np.int16)).mean() < 2
    assert cache.index[0, 7:9].tolist() == list(FRAME_SIZE)


def test_a_frame_without_a_full_one_keeps_its_crops_and_its_cache(tmp_path):
    folder = tmp_path / 'derived/frames'
    folder.mkdir(parents=True)
    Image.effect_noise(FRAME_SIZE, 60).convert('RGB').save(folder / 'frame_00000000.png')
    path = 'derived/frames/frame_00000000.png'
    box = [300.0, 100.0, 340.0, 141.0]
    cache, summary = stored_crops(tmp_path, path, box, EXACT)
    assert summary['full_frames'] == 0
    with Image.open(folder / 'frame_00000000.png') as source:
        wanted = source.convert('RGB').crop((300, 100, 340, 141)).resize((160, 160), Image.BILINEAR)
    assert np.array_equal(np.asarray(cache.crop(0)), np.asarray(wanted))
    # A frame at full definition arriving later is another cache.
    Image.new('RGB', FULL_SIZE).save(full_picture(folder / 'frame_00000000.png'))
    assert stored_crops(tmp_path, path, box, EXACT)[1]['built']
