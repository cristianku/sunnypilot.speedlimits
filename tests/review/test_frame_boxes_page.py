"""Speed review's boxes over the video (web/src/control/frame-boxes.ts): a frame of Sign review drawn on its very frame of
the video (second N of a minute is frame 20N) and on no other. Node runs the TypeScript with its types stripped (uvx
nodejs-wheel, as helpers/build_web.sh); skipped where uvx is not installed."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / 'web' / 'src' / 'control' / 'frame-boxes.ts'

SCRIPT = """
import {boxesAt, exactFrames} from %(module)s;
const box = (x) => ({bbox_xyxy: [x, x, x + 9, x + 9], family: 'maximum_speed', value_kph: 80});
const signs = [
  {frame_key: '0/1', segment: 0, second: 1, width: 1344, height: 760, confirmed: true, reviewer: 'cz', reviewer_kind: 'human', boxes: [box(500)]},
  {frame_key: '1/3', segment: 1, second: 3, confirmed: false, reviewer: null, reviewer_kind: null, boxes: [box(700)]},
  {frame_key: '2/4', segment: 2, second: 4, width: 1344, height: 760, confirmed: false, needs_person: true, reviewer: 'ai/x', reviewer_kind: 'ai', boxes: [box(800)]}];
const exact = exactFrames(signs);
const at = (segment, frame) => {
  const drawn = boxesAt({segment, frame}, exact);
  return drawn && {x: drawn.boxes[0].bbox_xyxy[0], width: drawn.width, height: drawn.height, confirmed: drawn.confirmed,
    needsPerson: drawn.needsPerson, key: drawn.frame.frame_key};
};
console.log(JSON.stringify({k20: at(0, 20), k19: at(0, 19), k21: at(0, 21), k60: at(1, 60), k80: at(2, 80),
  none: boxesAt(null, exact)}));
"""


@pytest.fixture(scope='module')
def result(tmp_path_factory):
    uvx = shutil.which('uvx')
    if uvx is None:
        pytest.skip('uvx is not installed: no node to run the TypeScript')
    folder = tmp_path_factory.mktemp('frame-boxes')
    # Node strips the types of a .ts it imports by its file URL; the copy keeps the test away from web/.
    copy = folder / 'frame-boxes.ts'
    shutil.copyfile(MODULE, copy)
    script = folder / 'check.mjs'
    script.write_text(SCRIPT % {'module': json.dumps(copy.as_uri())}, encoding='utf-8')
    done = subprocess.run([uvx, '--quiet', '--from', 'nodejs-wheel==24.*', 'node', '--no-warnings', str(script)],
                          capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_a_frame_of_sign_review_on_its_very_frame_only(result):
    assert result['k20'] == {'x': 500, 'width': 1344, 'height': 760, 'confirmed': True, 'needsPerson': False, 'key': '0/1'}
    assert result['k19'] is None and result['k21'] is None
    assert result['none'] is None


def test_a_proposed_frame_and_the_ai_reviewer_s(result):
    # A frame without its size takes the road camera's.
    assert result['k60'] == {'x': 700, 'width': 1344, 'height': 760, 'confirmed': False, 'needsPerson': False, 'key': '1/3'}
    assert result['k80']['needsPerson'] is True and result['k80']['key'] == '2/4'
