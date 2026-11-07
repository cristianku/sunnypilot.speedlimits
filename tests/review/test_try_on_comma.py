"""Try on the comma now (Models › a SignNet version): nnslr_tools.comma_trial, scripts/try_on_comma.sh and the panel
actions try_on_comma and comma_restore.

The comma is a fake: ssh and scp on PATH run here, on a folder laid out as the comma's (params, the openpilot build with
its speed-limits package and its models submodule, hw.py, the log). The networks are fakes too: the export writes
contracts as the real one does, and onnxruntime is replaced by networks that answer a class. Nothing leaves this machine.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import date
from pathlib import Path

import pytest
import test_panel
from test_panel import finished, full_log
from test_panel_runs import write
from test_send_to_car import IDENTITY, git, ready_payload, servers_runs, sha256

from nnslr_tools import car_release, car_settings, comma_trial
from nnslr_tools.comma_trial import CommaError
from nnslr_tools.job_progress import progress
from nnslr_tools.panel import Panel, PanelError

# The panel tests' fixtures (pytest finds them by name in this module; only_this_checkout is autouse).
checkout, only_this_checkout, root = test_panel.checkout, test_panel.only_this_checkout, test_panel.root
REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / comma_trial.TRY_SCRIPT
TODAY = date(2026, 10, 28)
SPEED = 'maximum_speed:50'
CLASSES = ['cancellation', 'maximum_speed:30', SPEED, 'maximum_speed:80', 'other_sign:give_way']
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

FAKE_SSH = r'''#!/usr/bin/env bash
# ssh [-o option]... host command: the command runs here; the host "offline" does not answer (255, as ssh).
while [ "$#" -gt 0 ]; do
  case "$1" in -o) shift 2 ;; -G) echo "hostname 127.0.0.1"; echo "port 9"; exit 0 ;; -*) shift ;; *) break ;; esac
done
host="$1"; shift
if [ "$host" = offline ]; then echo "ssh: connect to host offline port 22: No route to host" >&2; exit 255; fi
printf '%s\n' "$*" >> "$FAKE_COMMA_LOG"
exec bash -c "$*"
'''
FAKE_SCP = r'''#!/usr/bin/env bash
# scp [-q] [-o option]... source target, "host:" dropped; FAKE_SCP_CORRUPT spoils what arrives as .tmp.
files=()
while [ "$#" -gt 0 ]; do
  case "$1" in -o) shift 2 ;; -q) shift ;; *) files+=("${1#*:}"); shift ;; esac
done
cp -- "${files[0]}" "${files[1]}" || exit 1
printf 'scp %s %s\n' "${files[0]}" "${files[1]}" >> "$FAKE_COMMA_LOG"
if [ -n "${FAKE_SCP_CORRUPT:-}" ] && [ "${files[1]%.tmp}" != "${files[1]}" ]; then printf 'x' >> "${files[1]}"; fi
'''
# The comma's vision.py, as the build's: the constants, the reading of a class, read_signs (the networks' answers).
VISION = '''import numpy as np
SIGNNET_WIDTH = 896
SIGNNET_HEIGHT = 288
SIGNNET_STRIDE = 8
CLASSIFIER_SIZE = 160
CLASSIFIER_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
CLASSIFIER_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SPEED_KINDS = ("maximum_speed", "variable_display", "zone", "road_marking")
CANCELLATION = "cancellation"


def class_speed(name):
    if name == CANCELLATION:
        return 0.0
    kind, _, value = name.partition(":")
    return float(value) if kind in SPEED_KINDS and value.isdigit() else None


class SignNetInput:
    def __init__(self, width, height):
        self.width, self.height = width, height


def read_signs(y, uv, band, finder, classifier, classes):
    finder(np.zeros((1, 3, SIGNNET_HEIGHT, SIGNNET_WIDTH), dtype=np.float32))
    logits = classifier(np.zeros((1, 3, CLASSIFIER_SIZE, CLASSIFIER_SIZE), dtype=np.float32))
    name = classes[int(np.argmax(logits[0]))]
    return [{"speed": class_speed(name), "name": name}]
'''
HARDWARE = '''class Paths:
  @staticmethod
  def model_root() -> str:
    if PC:
      return str(Path(Paths.comma_home()) / "media" / "0" / "models")
    else:
      return "{root}"

  @staticmethod
  def config_root() -> str:
    return "/tmp/.comma"
'''


@pytest.fixture(autouse=True)
def git_identity(monkeypatch):
    for key, value in IDENTITY.items():
        monkeypatch.setenv(key, value)


class FakeOnnx:
    """A network in place of onnxruntime: its shapes from the contract next to it, and an answer: the classifier
    gives the class FakeOnnx.answer (an exception: raised)."""
    answer: object = SPEED

    def __init__(self, path: Path):
        contract = json.loads(Path(str(path) + '.contract.json').read_text())
        self.classes = (contract.get('output') or {}).get('classes')
        self.inputs = [('images', contract['input']['shape'])]
        self.outputs = [('out', contract['output']['shape'])] * contract.get('fake_outputs', 1)

    def __call__(self, x):
        import numpy as np
        if isinstance(FakeOnnx.answer, Exception):
            raise FakeOnnx.answer
        if self.classes is None:
            return np.zeros(self.outputs[0][1], dtype=np.float32)
        logits = np.zeros((1, len(self.classes)), dtype=np.float32)
        logits[0, self.classes.index(FakeOnnx.answer)] = 5
        return logits


def contracts(signnet: dict | None = None, classifier: dict | None = None) -> dict:
    """The contracts of an export as the real one writes them, with changes."""
    finder = {'task': 'signnet', 'stride': 8, 'input': {'name': 'images', 'shape': [1, 3, 288, 896]},
              'output': {'name': 'out', 'shape': [1, 5, 36, 112],
                         'channels': ['heat', 'log_width', 'log_height', 'offset_x', 'offset_y']}}
    reader = {'task': 'reader', 'input': {'name': 'images', 'shape': [1, 3, 160, 160],
                                          'normalization': {'mean': MEAN, 'std': STD}},
              'output': {'name': 'logits', 'shape': [1, len(CLASSES)], 'classes': CLASSES}}
    for contract, changes in ((finder, signnet or {}), (reader, classifier or {})):
        for key, value in changes.items():
            part, _, field = key.partition('.')
            if field:
                contract[part] = {**contract[part], field: value}
            else:
                contract[key] = value
    return {'finder': finder, 'classifier': reader}


def exporter_of(content: bytes = b'v1', **changes):
    """An exporter that writes both ONNX and their contracts as the real one does (changes: contracts(...))."""
    def exporter(chosen: dict, folder: Path) -> dict:
        written = contracts(**changes)
        for network, name in car_release.FILES.items():
            onnx = content + b' ' + name.encode()
            (folder / name).write_bytes(onnx)
            write(folder / (name + '.contract.json'), {
                **written[network], 'checkpoint': str(chosen[network]['weights']), 'onnx_sha256': sha256(onnx)})
        return car_release.finish_contracts(chosen, folder)
    return exporter


def frames_of(tmp_path: Path, speeds=(50, 50, 50, 50, 50)) -> list[dict]:
    from PIL import Image
    frames = []
    for index, speed in enumerate(speeds):
        image = tmp_path / 'frames' / f'f{index}.png'
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (64, 36), (90, 120, 160)).save(image)
        frames.append({'image': image, 'name': image.name, 'speeds': [float(speed)], 'route': f'r{index}'})
    return frames


@pytest.fixture
def comma(tmp_path, monkeypatch):
    """A comma that answers, parked, running a build with speedvisiond, Show Vision Speed Limit on."""
    home = tmp_path / 'comma'
    params, openpilot, log, model_root = home / 'params', home / 'openpilot', home / 'log', home / 'models-root'
    package = openpilot / comma_trial.PACKAGE
    params.mkdir(parents=True)
    for name, value in (('IsOnroad', '0'), ('SpeedLimitVisionDisplay', '1'), ('SpeedLimitPolicy', '1'),
                        ('GitCommit', '41a4975099a9cdf0f57a45510b0bf2057f671481')):
        (params / name).write_text(value)
    package.mkdir(parents=True)
    (package / 'runner.py').write_text('MANIFEST_FORMAT = 2\n')
    (package / 'warmup.py').write_text('')
    (package / 'vision.py').write_text(VISION)
    (openpilot / comma_trial.HARDWARE).parent.mkdir(parents=True)
    (openpilot / comma_trial.HARDWARE).write_text(HARDWARE.replace('{root}', str(model_root)))
    models = openpilot / comma_trial.MODELS_DIR
    models.mkdir()
    git(models, 'init', '-q')
    for name in car_release.FILES.values():
        (models / name).write_bytes(b'build ' + name.encode())
        (models / (name + '.contract.json')).write_text('{}\n')
    (models / car_release.MANIFEST).write_text(json.dumps(car_release.manifest_of('2026-10-27')) + '\n')
    git(models, 'add', '-A')
    git(models, 'commit', '-q', '-m', 'build networks')
    log.mkdir()
    (log / 'swaglog.2026-10-28--18-56-01').write_text(
        '{"ctx": {"daemon": "speed_limits_warmup"}, "created": 1793213877.26, '
        '"msg$s": "speed limits warmup: signnet did not compile: out of memory"}\n'
        'speed limits warmup: not JSON\nother line\n')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name, text in (('ssh', FAKE_SSH), ('scp', FAKE_SCP)):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setenv('FAKE_COMMA_LOG', str(tmp_path / 'ssh.log'))
    monkeypatch.setenv(comma_trial.HOST_ENV, 'fakecomma')
    monkeypatch.setenv(comma_trial.OPENPILOT_ENV, str(openpilot))
    monkeypatch.setenv(comma_trial.PARAMS_ENV, str(params))
    monkeypatch.setenv(comma_trial.LOG_ENV, str(log))
    monkeypatch.setattr(comma_trial, 'Onnx', FakeOnnx)
    monkeypatch.setattr(FakeOnnx, 'answer', SPEED)
    data = tmp_path / 'data'
    servers_runs(data)
    return {'home': home, 'params': params, 'package': package, 'models': models, 'model_root': model_root,
            'data': data, 'log': tmp_path / 'ssh.log', 'frames': frames_of(tmp_path)}


def run_trial(comma: dict, *, exporter=None, wait: float = 5, sleep=None, said: list | None = None, **kwargs):
    said = [] if said is None else said
    return comma_trial.try_on_comma(comma['data'], 's1', today=TODAY, exporter=exporter or exporter_of(),
                                    frames=comma['frames'], poll=0, wait=wait, sleep=sleep or (lambda _: None),
                                    say=said.append, **kwargs)


def compile_all(comma: dict):
    """The comma's warmup: both compiled files appear (at the first wait)."""
    def sleep(_):
        folder = comma['model_root'] / 'speed_limits'
        folder.mkdir(parents=True, exist_ok=True)
        for name in car_release.FILES.values():
            digest = sha256(b'v1 ' + name.encode())
            (folder / comma_trial.compiled_name(Path(name).stem, digest)).write_bytes(b'pkl')
    return sleep


def models_state(comma: dict) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(comma['models'].iterdir()) if path.is_file()}


# -- the copy and the wait ------------------------------------------------------------------------------------
def test_copied_checked_moved_manifest_last_then_ready(comma):
    said: list[str] = []
    result = run_trial(comma, sleep=compile_all(comma), said=said)
    assert result['outcome'] == 'ready'
    models = models_state(comma)
    assert models['signnet.onnx'] == b'v1 signnet.onnx' and models['classifier.onnx'] == b'v1 classifier.onnx'
    assert json.loads(models['speed_limits.json']) == car_release.manifest_of('2026-10-28-try',
                                                                             settings=car_settings.DEFAULTS)
    assert not [name for name in models if name.endswith('.tmp')]
    contract = json.loads(models['signnet.onnx.contract.json'])
    assert contract['checkpoint'] == 'cutter:s1/signnet'
    # Every file arrives as .tmp; one move puts them in place, speed_limits.json last.
    commands = comma['log'].read_text()
    move = next(line for line in commands.splitlines() if 'mv -f' in line)
    assert move.rstrip().endswith('mv -f -- speed_limits.json.tmp speed_limits.json')
    assert not any(word in commands for word in ('git checkout', 'git -C', 'reboot', 'kill', 'systemctl'))
    trial = comma_trial.read_trial(comma['data'])
    assert trial['state'] == 'ready' and trial['signnet']['run'] == 's1' and trial['version'] == '2026-10-28-try'
    assert trial['classifier']['id'] == 'classifier:r1/reader' and trial['comma_build'] == '41a497509'
    assert trial['signnet']['onnx_sha256'] == sha256(b'v1 signnet.onnx')
    assert said[0] == '== 1/5 Check the comma' and "== 3/5 Check them with the comma's code" in said
    assert any(line.startswith('compiled on the comma: signnet (signnet_') for line in said)
    assert said[-1].startswith('ready: used from the next drive')


def test_the_builds_brain_stays_in_use_during_a_trial(comma):
    (comma['models'] / car_release.BRAIN_FILE).write_bytes(b'build brain')
    (comma['models'] / (car_release.BRAIN_FILE + '.contract.json')).write_text('{}\n')
    assert run_trial(comma, sleep=compile_all(comma))['outcome'] == 'ready'
    models = models_state(comma)
    assert json.loads(models['speed_limits.json']) == car_release.manifest_of('2026-10-28-try', brain=True,
                                                                                 settings=car_settings.DEFAULTS)
    assert models[car_release.BRAIN_FILE] == b'build brain'


def test_a_dry_run_reads_the_comma_and_copies_nothing(comma):
    before = models_state(comma)
    said: list[str] = []
    assert run_trial(comma, said=said, dry_run=True)['outcome'] == 'dry_run'
    assert models_state(comma) == before and comma_trial.read_trial(comma['data']) is None
    assert '== 4/5 Copy them onto the comma' not in said
    assert 'mv ' not in comma['log'].read_text() and 'scp ' + str(comma['package']) + '/vision.py' in comma['log'].read_text()


def test_not_compiled_in_time_puts_the_build_back_with_the_comma_log(comma):
    build = models_state(comma)
    said: list[str] = []
    with pytest.raises(CommaError, match="did not prepare them within .* the build's networks are back "
                                         r'\(speed-limits 2026-10-27\)'):
        run_trial(comma, wait=0, said=said)
    assert models_state(comma) == build
    assert comma_trial.read_trial(comma['data']) is None
    assert 'comma log: 2026-10-28 18:57:57 UTC speed limits warmup: signnet did not compile: out of memory' in said
    assert 'comma log: speed limits warmup: not JSON' in said and 'comma log: other line' not in said


def test_a_drive_started_while_waiting_leaves_them_for_speedvisiond(comma):
    def drive(_):
        (comma['params'] / 'IsOnroad').write_text('1')
    with pytest.raises(CommaError, match='started a drive before preparing them'):
        run_trial(comma, sleep=drive)
    assert models_state(comma)['signnet.onnx'] == b'v1 signnet.onnx'
    assert comma_trial.read_trial(comma['data'])['state'] == 'preparing'


def test_a_copy_that_arrives_wrong_is_removed_and_nothing_moved(comma, monkeypatch):
    build = models_state(comma)
    monkeypatch.setenv('FAKE_SCP_CORRUPT', '1')
    with pytest.raises(CommaError, match='the copy arrived wrong on the comma: signnet.onnx'):
        run_trial(comma)
    assert models_state(comma) == build and comma_trial.read_trial(comma['data']) is None


def test_back_to_the_build(comma):
    run_trial(comma, sleep=compile_all(comma))
    (comma['models'] / 'signnet.onnx.tmp').write_bytes(b'half')
    said: list[str] = []
    assert comma_trial.back_to_build(comma['data'], say=said.append) == '2026-10-27'
    assert models_state(comma)['signnet.onnx'] == b'build signnet.onnx'
    assert not (comma['models'] / 'signnet.onnx.tmp').exists()
    assert comma_trial.read_trial(comma['data']) is None
    assert said == ['== 1/2 Check the comma', 'the comma is parked; its networks now: speed-limits 2026-10-28-try',
                    "== 2/2 Put back the build's networks",
                    ("back: the comma has the build's networks again (speed-limits 2026-10-27); "
                     'models/on-comma-trial.json cleared')]


# -- the refusals: nothing copied -----------------------------------------------------------------------------
def refused(comma: dict, message: str, **kwargs) -> None:
    build = models_state(comma)
    with pytest.raises(CommaError, match=message):
        run_trial(comma, **kwargs)
    assert models_state(comma) == build and comma_trial.read_trial(comma['data']) is None
    assert 'mv ' not in (comma['log'].read_text() if comma['log'].exists() else '')


def test_refused_when_the_comma_does_not_answer(comma, monkeypatch):
    monkeypatch.setenv(comma_trial.HOST_ENV, 'offline')
    refused(comma, r'does not answer over ssh \(offline\).*No route to host')


def test_refused_without_openpilot(comma):
    import shutil
    shutil.rmtree(comma['params'])
    refused(comma, 'no openpilot params on the comma')


def test_refused_while_driving(comma):
    (comma['params'] / 'IsOnroad').write_text('1')
    refused(comma, 'the comma is driving: try again when the car is parked')


@pytest.mark.parametrize('change', ['runner', 'format', 'warmup', 'vision'])
def test_refused_on_a_build_without_speedvisiond(comma, change):
    if change == 'format':
        (comma['package'] / 'runner.py').write_text('MANIFEST_FORMAT = 1\n')
    else:
        (comma['package'] / f'{change}.py').unlink()
    refused(comma, r'the comma \(build 41a497509\) cannot run these networks .*install the testing build first')


def test_refused_without_the_models_folder(comma):
    import shutil
    shutil.rmtree(comma['models'] / '.git')
    refused(comma, 'has no speed-limits models folder')


def test_refused_while_the_comma_does_not_prepare_them(comma):
    (comma['params'] / 'SpeedLimitVisionDisplay').write_text('0')
    refused(comma, 'turn on Show Vision Speed Limit, or choose Speed Limit Source Smartspeed')
    # Smartspeed alone is enough, and a 5 (the old Map Data + Vision) on a comma not updated yet.
    for policy in ('5', '6'):
        (comma['params'] / 'SpeedLimitPolicy').write_text(policy)
        assert run_trial(comma, dry_run=True)['outcome'] == 'dry_run'


def test_refused_when_the_compiled_folder_is_unknown(comma):
    (comma['home'] / 'openpilot' / comma_trial.HARDWARE).write_text('class Paths:\n  pass\n')
    refused(comma, 'where the comma keeps compiled networks is not readable')


@pytest.mark.parametrize('changes, message', [
    ({'signnet': {'input.shape': [1, 3, 256, 768]}}, r'SignNet takes \[1, 3, 256, 768\], the comma gives it \[1, 3, 288, 896\]'),
    ({'signnet': {'stride': 4}}, 'SignNet has a grid step of 4 px, the comma decodes 8 px'),
    ({'signnet': {'output.shape': [1, 4, 36, 112]}}, r'SignNet gives \[1, 4, 36, 112\], the comma reads \[1, 5, 36, 112\]'),
    ({'signnet': {'fake_outputs': 2}}, 'SignNet has 2 outputs, the comma reads one'),
    ({'signnet': {'output.channels': ['log_width', 'heat', 'log_height', 'offset_x', 'offset_y']}},
     'SignNet gives the channels .* the comma reads heat, then 4 shape channels'),
    ({'classifier': {'input.shape': [1, 3, 128, 128]}}, r'the classifier takes \[1, 3, 128, 128\], the comma gives it \[1, 3, 160, 160\]'),
    ({'classifier': {'input.normalization': {'mean': [0.5, 0.5, 0.5], 'std': STD}}},
     r'the classifier was trained with the mean \[0.5, 0.5, 0.5\]'),
    ({'classifier': {'output.shape': [1, 4]}}, r'the classifier gives \[1, 4\], not one score for each of its 5 classes'),
    ({'classifier': {'output.classes': ['other_sign:a', 'other_sign:b', 'other_sign:c', 'other_sign:d', 'other_sign:e']}},
     'no class of the classifier gives a speed'),
])
def test_refused_when_the_contracts_do_not_fit_the_comma(comma, changes, message):
    refused(comma, 'these networks do not fit the build on the comma: .*' + message + '.*a build is needed: use Publish',
            exporter=exporter_of(**changes))


def test_refused_when_the_comma_reads_a_class_otherwise(comma):
    (comma['package'] / 'vision.py').write_text(VISION.replace(', "road_marking")', ')'))
    refused(comma, 'these networks do not fit the build on the comma: the comma reads the classes road_marking:50 as None',
            exporter=exporter_of(classifier={'output.classes': [*CLASSES[:-1], 'road_marking:50']}))


def test_refused_when_the_comma_vision_lacks_what_it_reads_with(comma):
    (comma['package'] / 'vision.py').write_text(VISION.replace('def class_speed', 'def class_speed_old'))
    refused(comma, "the comma's vision.py has no class_speed")


def test_refused_when_the_comma_code_fails_on_them(comma, monkeypatch):
    monkeypatch.setattr(FakeOnnx, 'answer', RuntimeError('bad input'))
    refused(comma, "the comma's read_signs fails with these networks: RuntimeError: bad input; a build is needed")


def test_refused_when_the_comma_code_reads_them_wrong(comma, tmp_path):
    comma['frames'] = frames_of(tmp_path, speeds=(50, 80, 80, 80, 50))
    said: list[str] = []
    refused(comma, "reads the confirmed speed on 2 of 5 frames only \\(at least 3 wanted\\): nothing copied", said=said)
    assert '  f1.png: confirmed 80 km/h, read 50 km/h (wrong)' in said


def test_refused_with_too_few_reviewed_frames(comma):
    comma['frames'] = comma['frames'][:4]
    refused(comma, 'fewer than 5 reviewed frames with a confirmed speed sign')


# -- the frames of the reading check ---------------------------------------------------------------------------
def test_the_frames_are_whole_single_speed_signs_of_a_usual_size(tmp_path):
    from PIL import Image
    data = tmp_path / 'data'
    rows, assignments = [], {}

    def frame(name, boxes, route='r', state='accepted', test=False, basis='human_review'):
        image = data / 'frames' / f'{name}.png'
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (8, 8)).save(image)
        for index, (box, value) in enumerate(boxes):
            annotation = f'{name}-{index}'
            rows.append({'annotation_id': annotation, 'image_path': f'frames/{name}.png', 'bbox_xyxy': box,
                         'sign_family': 'maximum_speed', 'value_kph': value, 'review_state': state, 'width': 1344,
                         'height': 760, 'route_id': route, 'provenance': {'label_basis': basis}})
            if test:
                assignments[annotation] = 'test'

    frame('usual', [([600, 100, 650, 150], 50)], route='a')
    frame('usual_test', [([600, 100, 652, 152], 80)], route='b', test=True)
    frame('ai', [([600, 100, 650, 150], 60)], route='c', basis='ai_review')
    frame('far', [([600, 100, 620, 120], 50)], route='d')
    frame('close', [([600, 100, 700, 200], 50)], route='e')
    frame('two_speeds', [([600, 100, 650, 150], 50), ([700, 100, 750, 150], 80)], route='f')
    frame('below_band', [([600, 400, 650, 450], 50)], route='g')
    frame('at_edge', [([0, 100, 50, 150], 50)], route='h')
    frame('no_value', [([600, 100, 650, 150], None)], route='i')
    frame('pending', [([600, 100, 650, 150], 50)], route='j', state='pending')
    frame('zz_same_drive', [([600, 100, 650, 150], 50)], route='a')
    (data / 'annotations').mkdir(parents=True)
    (data / 'annotations' / 'objects.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    write(data / 'splits' / 'split-1.json', {'assignments': assignments})
    write(data / 'splits' / 'latest.json', {'path': 'splits/split-1.json', 'sha256': 'x'})
    chosen = comma_trial.speed_frames(data, count=4)
    assert [frame['name'] for frame in chosen] == ['frames/usual_test.png', 'frames/usual.png', 'frames/ai.png',
                                                   'frames/zz_same_drive.png']
    assert chosen[0]['speeds'] == [80.0] and chosen[0]['image'] == data / 'frames' / 'usual_test.png'


def test_nv12_of_a_grey_frame():
    import numpy as np
    y, uv = comma_trial.nv12_of(np.full((4, 6, 3), 128, dtype=np.uint8))
    assert y.shape == (4, 6) and uv.shape == (2, 6)
    assert int(y[0, 0]) == 126 and set(uv.reshape(-1).tolist()) == {128}


# -- scripts/try_on_comma.sh and nnslr try-on-comma ----------------------------------------------------------
def script(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    venv = tmp_path / 'venv'
    (venv / 'bin').mkdir(parents=True, exist_ok=True)
    (venv / 'bin' / 'nnslr').write_text('#!/usr/bin/env bash\necho "nnslr $* CUDA=[$CUDA_VISIBLE_DEVICES]"\n')
    (venv / 'bin' / 'nnslr').chmod(0o755)
    return subprocess.run(['bash', str(SCRIPT), *args], capture_output=True, text=True, check=False, timeout=30,
                          env={**os.environ, 'NNSLR_GPU_VENV': str(venv), 'NNSLR_DATA_ROOT': str(tmp_path / 'data')})


def test_the_script_checks_its_arguments(tmp_path):
    assert 'Usage:' in script(tmp_path, '--help').stdout
    for args, message in ((('--bogus',), 'unknown argument: --bogus'), ((), 'which SignNet?'),
                          (('--signnet', '../x'), 'invalid run name: ../x'),
                          (('--signnet=s1', '--classifier=cutter:x/y'), 'invalid classifier id'),
                          (('--restore', '--signnet=s1'), '--restore takes no SignNet'),
                          (('--restore', '--dry-run'), '--restore takes no SignNet')):
        done = script(tmp_path, *args)
        assert done.returncode == 1 and message in done.stderr, (args, done.stderr)
    done = script(tmp_path, '--signnet=s1', '--dry-run')
    assert done.returncode == 0 and done.stdout.splitlines() == [
        '== Try on the comma: SignNet s1',
        f'nnslr try-on-comma --data-root {tmp_path / "data"} --signnet s1 --dry-run CUDA=[]']
    assert script(tmp_path, '--restore').stdout.splitlines() == [
        "== Back to the build's networks on the comma", f'nnslr try-on-comma --data-root {tmp_path / "data"} --restore CUDA=[]']


def test_nnslr_try_on_comma(comma, monkeypatch, capsys):
    from nnslr_tools.cli import main
    data = str(comma['data'])
    assert main(['try-on-comma', '--data-root', data]) == 2
    assert main(['try-on-comma', '--restore', '--signnet', 's1', '--data-root', data]) == 2
    assert main(['try-on-comma', '--signnet', '../x', '--data-root', data]) == 2
    (comma['params'] / 'IsOnroad').write_text('1')
    assert main(['try-on-comma', '--restore', '--data-root', data]) == 2
    assert 'error: the comma is driving' in capsys.readouterr().err
    (comma['params'] / 'IsOnroad').write_text('0')
    assert main(['try-on-comma', '--restore', '--data-root', data]) == 0
    assert "back: the comma has the build's networks again (speed-limits 2026-10-27)" in capsys.readouterr().out


# -- the panel ------------------------------------------------------------------------------------------------
def trial_installed(tmp_path: Path, repo: Path, monkeypatch) -> None:
    """What the actions need on the server: the script (a fake that prints its arguments) and the GPU venv."""
    path = repo / comma_trial.TRY_SCRIPT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/usr/bin/env bash\necho "== Try on the comma: SignNet ${1#--signnet=}"\necho "args: $*"\n'
                    'echo "== 1/5 Check the comma"\necho "== 3/5 Check them"\n')
    path.chmod(0o755)
    write(tmp_path / 'venv' / 'bin' / 'nnslr', '')
    monkeypatch.setenv('NNSLR_GPU_VENV', str(tmp_path / 'venv'))


def test_the_actions_say_why_they_cannot_run(root, checkout, tmp_path, monkeypatch):
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel.comma_probe, 'state', lambda: None)
    states = panel._action_states(ready_payload(), [])
    assert states['try_on_comma']['reason'] == 'scripts/try_on_comma.sh is missing in this installation'
    assert states['comma_restore']['reason'] == 'scripts/try_on_comma.sh is missing in this installation'
    trial_installed(tmp_path, checkout, monkeypatch)
    assert panel._action_states(ready_payload(), [])['try_on_comma']['reason'] == \
        'no classifier in use: choose one on the Models tab'
    servers_runs(root)
    states = panel._action_states(ready_payload(), [])
    assert states['try_on_comma'] == {'label': 'Try on the comma now', 'allowed': True, 'reason': ''}
    assert states['comma_restore'] == {'label': "Back to the build's networks", 'allowed': True, 'reason': ''}
    # The probe of a minute ago says the comma does not answer: both wait for it.
    monkeypatch.setattr(panel.comma_probe, 'state', lambda: False)
    states = panel._action_states(ready_payload(), [])
    for name in ('try_on_comma', 'comma_restore'):
        assert states[name]['reason'] == 'the comma does not answer on the network: it answers while the car is on or just parked'
    assert panel.models()['comma_trial'] is None
    comma_trial.write_trial(root, {'signnet': {'id': 'cutter:s1/signnet', 'run': 's1', 'onnx_sha256': 'a'},
                                   'classifier': {'id': 'classifier:r1/reader', 'run': 'r1', 'onnx_sha256': 'b'},
                                   'version': '2026-10-28-try', 'state': 'ready', 'copied_at': 'x', 'ready_at': 'y',
                                   'comma_build': '41a497509'})
    assert panel.models()['comma_trial']['signnet']['run'] == 's1'


def test_try_on_comma_takes_only_a_finished_signnet_and_restore_takes_nothing(root, checkout, tmp_path, monkeypatch):
    trial_installed(tmp_path, checkout, monkeypatch)
    servers_runs(root)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', list)
    monkeypatch.setattr(panel.comma_probe, 'state', lambda: True)
    for options in (None, {'signnet': 'nothing'}, {'signnet': 's3'}, {'signnet': '../s1'}, {'signnet': '--restore'}):
        with pytest.raises(PanelError) as refusal:
            panel.start('try_on_comma', options)
        assert (refusal.value.status, refusal.value.reason) == (400, 'bad_option'), options
    with pytest.raises(PanelError) as refusal:
        panel.start('comma_restore', {'signnet': 's1'})
    assert refusal.value.reason == 'bad_option'
    job = panel.start('try_on_comma', {'signnet': 's1'})
    done = finished(panel, job['id'])
    assert done['state'] == 'ok' and done['run'] == 's1' and done['label'] == 'Try on the comma now'
    assert 'args: --signnet=s1\n' in full_log(panel, job['id'])
    assert [step['key'] for step in panel.job_progress(done, None)['steps']] == ['check', 'export', 'verify', 'copy', 'prepare']
    job = panel.start('comma_restore', None)
    assert 'args: --restore\n' in full_log(panel, finished(panel, job['id'])['id'])


def test_the_progress_follows_the_steps():
    log = '== Try on the comma: SignNet s1\n== 1/5 Check the comma\n== 2/5 Export\n== 3/5 Check them\n== 4/5 Copy\n'
    running = progress('try_on_comma', 'running', log)
    assert [step['state'] for step in running['steps']] == ['done', 'done', 'done', 'current', 'todo']
    assert running['percent'] == 33.0  # check 3, export 15, verify 15 of 100
    back = progress('comma_restore', 'running', "== Back\n== 1/2 Check the comma\n== 2/2 Put back the build's networks\n")
    assert [step['state'] for step in back['steps']] == ['done', 'current']


def test_send_to_car_clears_the_trial_once_its_build_is_pushed():
    text = (REPO / car_release.CAR_SCRIPT).read_text()
    assert text.index('rm -f "$TRIAL"') > text.index('NO_PUSH=true: $TESTING_BRANCH built')
