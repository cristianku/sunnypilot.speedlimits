"""Speed review's boxes on every frame of the video (web/src/control/sign-track.ts): the track of nnslr sign-tracks read
by minute and frame k, only on the minutes whose frames are the video's; a frame of Sign review confirmed (or saved by the
AI reviewer) wins on its very frame and not on the frames next to it; without the track only Sign review's own frames,
one a second, have boxes; and the time of a frame in the drive's video. Node runs the TypeScript with its types stripped
(uvx nodejs-wheel, as helpers/build_web.sh); skipped where uvx is not installed."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / 'web' / 'src' / 'control' / 'sign-track.ts'

SCRIPT = """
import {boxesAt, exactFrames, frameVideoTime, reviewedFrames, trackFrames} from %(module)s;
const index = {fps: 20, duration: 180, segments: [
  {segment: 0, start: 0, frames: 1200}, {segment: 1, start: 60.02, frames: 1200}, {segment: 2, start: 120, frames: 1199}]};
const track = {format: 1, route: 'r', fps: 20, width: 1344, height: 760, networks: {}, min_score: 0.1, made_at: '',
  labels: [{model_label: 'maximum_speed:60', family: 'maximum_speed', value_kph: 60, supplementary_panel: 'unknown'},
           {model_label: 'other_sign:pedestrian_crossing', family: 'other_sign', value_kph: null,
            supplementary_panel: 'unknown', sign_type: 'pedestrian_crossing'}],
  segments: [
    {segment: 0, frames: 1200, source: [1, 2], boxes: [[19, 0, 10, 20, 30, 40, 97], [20, 0, 11, 21, 31, 41, 96],
                                                       [21, 1, 12, 22, 32, 42, 55], [21, 0, 13, 23, 33, 43, 50]]},
    {segment: 2, frames: 1200, source: [1, 2], boxes: [[5, 0, 1, 2, 3, 4, 90]]}]};
const box = (x) => ({bbox_xyxy: [x, x, x + 9, x + 9], family: 'maximum_speed', value_kph: 80});
const signs = [
  {frame_key: '0/1', segment: 0, second: 1, width: 1344, height: 760, confirmed: true, reviewer: 'cz', reviewer_kind: 'human', boxes: [box(500)]},
  {frame_key: '0/2', segment: 0, second: 2, width: 1344, height: 760, confirmed: false, reviewer: null, reviewer_kind: null, boxes: [box(600)]},
  {frame_key: '1/3', segment: 1, second: 3, width: 1344, height: 760, confirmed: false, reviewer: null, reviewer_kind: null, boxes: [box(700)]},
  {frame_key: '2/4', segment: 2, second: 4, width: 1344, height: 760, confirmed: false, needs_person: true, reviewer: 'ai/x', reviewer_kind: 'ai', boxes: [box(800)]}];
const frames = trackFrames(track, index);
const reviewed = reviewedFrames(signs);
const exact = exactFrames(signs);
const at = (segment, frame) => {
  const drawn = boxesAt({segment, frame}, track, frames, reviewed, exact);
  return drawn && {boxes: drawn.boxes, width: drawn.width, height: drawn.height, confirmed: drawn.confirmed,
    needsPerson: drawn.needsPerson, key: drawn.frame ? drawn.frame.frame_key : null};
};
const out = {
  minutes: [...frames.keys()],
  k19: at(0, 19), k20: at(0, 20), k21: at(0, 21), k22: at(0, 22),
  k40: at(0, 40), k60: at(1, 60), k61: at(1, 61), k80: at(2, 80), k5: at(2, 5), none: boxesAt(null, track, frames, reviewed, exact),
  noTrack: boxesAt({segment: 0, frame: 40}, null, null, reviewed, exact),
  times: [frameVideoTime(index, 0, 0), frameVideoTime(index, 1, 3), frameVideoTime(index, 1, 1200), frameVideoTime(index, 9, 0)]};
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope='module')
def result(tmp_path_factory):
    uvx = shutil.which('uvx')
    if uvx is None:
        pytest.skip('uvx is not installed: no node to run the TypeScript')
    folder = tmp_path_factory.mktemp('sign-track')
    # Node strips the types of a .ts it imports by its file URL; the copy keeps the test away from web/.
    copy = folder / 'sign-track.ts'
    shutil.copyfile(MODULE, copy)
    script = folder / 'check.mjs'
    script.write_text(SCRIPT % {'module': json.dumps(copy.as_uri())}, encoding='utf-8')
    done = subprocess.run([uvx, '--quiet', '--from', 'nodejs-wheel==24.*', 'node', '--no-warnings', str(script)],
                          capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_only_the_minutes_whose_frames_are_the_video_s(result):
    # Minute 2 of the track has 1200 frames, the video's 1199: another video of the minute, its boxes left out.
    assert result['minutes'] == [0]


def test_the_track_s_boxes_on_their_frame_with_their_class(result):
    assert result['k19'] == {'boxes': [{'bbox_xyxy': [10, 20, 30, 40], 'family': 'maximum_speed', 'value_kph': 60,
                                        'supplementary_panel': 'unknown', 'model_label': 'maximum_speed:60',
                                        'score': 0.97}],
                             'width': 1344, 'height': 760, 'confirmed': False, 'needsPerson': False, 'key': None}
    assert [box['bbox_xyxy'][0] for box in result['k21']['boxes']] == [12, 13]
    assert result['k21']['boxes'][0]['sign_type'] == 'pedestrian_crossing'
    # A frame where the networks found nothing has no box, not Sign review's.
    assert result['k22']['boxes'] == [] and result['k40']['boxes'] == []


def test_a_confirmed_frame_wins_on_its_very_frame_only(result):
    assert result['k20']['key'] == '0/1' and result['k20']['confirmed'] is True
    assert result['k20']['boxes'][0]['bbox_xyxy'][0] == 500
    assert result['k19']['key'] is None and result['k21']['key'] is None


def test_a_proposed_frame_of_sign_review_does_not_win_over_the_track(result):
    assert result['k40']['key'] is None


def test_without_the_track_sign_review_s_frames_one_a_second(result):
    # Minute 1 has no track: second 3 is frame 60, and only that frame.
    assert result['k60']['key'] == '1/3' and result['k60']['confirmed'] is False
    assert result['k61'] is None
    # Minute 2's track does not match the video: the AI reviewer's frame on frame 80, nothing on the track's frame 5.
    assert result['k80']['key'] == '2/4' and result['k80']['needsPerson'] is True
    assert result['k5'] is None
    assert result['none'] is None
    assert result['noTrack']['boxes'][0]['bbox_xyxy'][0] == 600


def test_the_time_of_a_frame_in_the_video(result):
    # The middle of the frame; a minute starts on its frame of the video (60.02 s is frame 1200).
    assert result['times'] == [0.025, 60.175, None, None]
