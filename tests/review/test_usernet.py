"""UserNet's labels and plan without a network (nnslr_tools.usernet): BDD100K's labels as ours, our road-user boxes
as UserNet's, the window and its row offset, what a measure judges, the stages' frames, the licence along the
starts, and the UserNet of a run on the Models tab. Nothing here trains or reads a picture."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import nnslr_tools.usernet as usernet
from nnslr_tools.model_roles import ModelChoiceError, ModelIndex, role_of, usernet_digest
from nnslr_tools.runs import RunsIndex
from nnslr_tools.usernet import (COMPLETE_HEIGHT, INPUT_WIDTH, KINDS, WINDOW_HEIGHT, WINDOW_TOP, bdd_frames, bdd_split,
                                 bdd_users, digest, drive_frame, measured_truth, prepare_usernet, ticked, to_frame,
                                 uses_bdd, window_boxes)

BDD_W, BDD_H = 1280, 720


def detection(label: str, x1: float, y1: float, x2: float, y2: float) -> dict:
    """A BDD label as samples.json has it: bounding_box [x, y, w, h] in fractions of the 1280 x 720 picture."""
    return {'label': label, 'bounding_box': [x1 / BDD_W, y1 / BDD_H, (x2 - x1) / BDD_W, (y2 - y1) / BDD_H]}


def near(box, expected, tolerance=1e-6):
    return box == pytest.approx(expected, abs=tolerance)


# -- BDD100K's labels as ours ------------------------------------------------------------------------------------
def test_bdd_labels_become_persons_two_wheelers_vehicles_or_ignore_boxes():
    users, ignore = bdd_users([
        detection('pedestrian', 100, 300, 130, 380),
        # A rider on a bicycle: one two-wheeler, the box around both.
        detection('rider', 400, 280, 440, 360), detection('bicycle', 395, 320, 450, 400),
        # A motorcycle alone is a two-wheeler too.
        detection('motorcycle', 700, 350, 760, 410),
        detection('car', 800, 300, 950, 420), detection('truck', 1000, 250, 1200, 450), detection('bus', 10, 200, 90, 300),
        # Signs and lights are no road users: background, as on our frames.
        detection('traffic sign', 600, 100, 640, 140), detection('traffic light', 650, 100, 670, 150),
        # Neither user nor background: ignore boxes.
        detection('other person', 300, 300, 320, 340), detection('train', 0, 0, 100, 100),
        detection('trailer', 960, 300, 1000, 400), detection('other vehicle', 500, 500, 600, 600),
        # A rider far from any two-wheeler: an ignore box.
        detection('rider', 1200, 600, 1220, 650),
    ], BDD_W, BDD_H)
    by_kind = {kind: [user['box'] for user in users if user['kind'] == kind] for kind in KINDS}
    assert len(by_kind['person']) == 1 and near(by_kind['person'][0], [100, 300, 130, 380])
    assert len(by_kind['two_wheeler']) == 2
    assert near(by_kind['two_wheeler'][0], [395, 280, 450, 400])  # bicycle and rider together
    assert near(by_kind['two_wheeler'][1], [700, 350, 760, 410])
    assert len(by_kind['vehicle']) == 3
    assert len(ignore) == 5
    assert near(ignore[-1], [1200, 600, 1220, 650])
    # The signs and lights are neither users nor ignore boxes.
    assert not any(near(box, [600, 100, 640, 140]) for box in ignore + [user['box'] for user in users])


def test_each_rider_goes_with_one_two_wheeler_the_best_overlap_first():
    users, ignore = bdd_users([
        detection('bicycle', 100, 300, 160, 400), detection('bicycle', 150, 300, 210, 400),
        detection('rider', 155, 260, 200, 360),  # mostly on the second bicycle
        detection('rider', 300, 260, 340, 360),  # on nothing
    ], BDD_W, BDD_H)
    boxes = [user['box'] for user in users]
    assert [user['kind'] for user in users] == ['two_wheeler', 'two_wheeler']
    assert near(boxes[0], [100, 300, 160, 400])  # the bicycle without its rider stays as it is
    assert near(boxes[1], [150, 260, 210, 400])
    assert len(ignore) == 1 and near(ignore[0], [300, 260, 340, 360])


def test_a_picture_with_bad_labels_keeps_the_good_ones():
    users, ignore = bdd_users([{'label': 'car'}, {'bounding_box': [0, 0, 1, 1]}, {'label': 'car', 'bounding_box': [0, 0]},
                               detection('car', 10, 300, 100, 380)], BDD_W, BDD_H)
    assert len(users) == 1 and ignore == []


def names_by_split() -> dict[str, str]:
    """A picture name for each partition."""
    found: dict[str, str] = {}
    index = 0
    while len(found) < 3:
        name = f'b{index:05d}-picture'
        found.setdefault(bdd_split(name), name)
        index += 1
    return found


def bdd_folder(tmp_path: Path, samples: list[dict]) -> Path:
    folder = tmp_path / 'bdd100k'
    (folder / 'data').mkdir(parents=True)
    (folder / 'samples.json').write_text(json.dumps({'samples': samples}))
    return folder


def sample(name: str, detections: list[dict] | None) -> dict:
    item = {'filepath': f'data/{name}.jpg', 'metadata': {'width': BDD_W, 'height': BDD_H}}
    if detections is not None:
        item['detections'] = {'_cls': 'Detections', 'detections': detections}
    return item


def test_the_bdd_split_is_frozen_by_the_picture_name():
    names = [f'picture-{index}' for index in range(3000)]
    first = [bdd_split(name) for name in names]
    assert first == [bdd_split(name) for name in names]
    shares = {split: first.count(split) / len(names) for split in ('train', 'validation', 'test')}
    assert 0.76 < shares['train'] < 0.84 and 0.07 < shares['validation'] < 0.13 and 0.07 < shares['test'] < 0.13


def test_bdd_frames_from_a_samples_json_alone(tmp_path):
    names = names_by_split()
    folder = bdd_folder(tmp_path, [
        sample(names['train'], [detection('car', 100, 300, 300, 450)]),
        sample(names['validation'], [detection('pedestrian', 600, 300, 640, 400)]),
        sample(names['test'], []),  # labelled, nobody there
        sample('no-labels', None),
        'not a sample',
    ])
    read = bdd_frames(folder)
    assert read['unlabelled'] == 2 and len(read['frames']) == 3
    frames = {frame['id']: frame for frame in read['frames']}
    assert {frame['split'] for frame in frames.values()} == {'train', 'validation', 'test'}
    train = frames[names['train']]
    assert train['image_path'] == str((folder / 'data' / f"{names['train']}.jpg").resolve())
    assert (train['source'], train['width'], train['height'], train['complete']) == ('bdd', BDD_W, BDD_H, True)
    assert train['users'][0]['kind'] == 'vehicle' and frames[names['test']]['users'] == []
    # The same pictures, the same split's sha256.
    assert bdd_frames(folder)['split_sha256'] == read['split_sha256'] and len(read['samples_sha256']) == 64


def test_bdd_frames_without_samples_json_says_so(tmp_path):
    with pytest.raises(ValueError, match='usernet_bdd_missing'):
        bdd_frames(tmp_path / 'nowhere')
    (tmp_path / 'samples.json').write_text('{"samples": 3}')
    with pytest.raises(ValueError, match='usernet_bdd_unreadable'):
        bdd_frames(tmp_path)


# -- our drives --------------------------------------------------------------------------------------------------
def road_user(kind: str, box: list[float], **extra) -> dict:
    return {'sign_family': 'other_sign', 'sign_type': kind, 'bbox_xyxy': box, **extra}


def reviewed(kept: list[dict], *, split: str = 'train', **extra) -> dict:
    return {'image_path': 'frames/r1/000100.jpg', 'image_sha256': 'a' * 64, 'width': 1344, 'height': 760,
            'split': split, 'route_id': 'r1', 'label_basis': 'human_review', 'kept': kept, **extra}


def test_our_road_user_boxes_become_usernets_kinds():
    frame = drive_frame(reviewed([
        road_user('road_user_person', [10, 300, 40, 380]),
        road_user('road_user_bicycle', [100, 300, 150, 400]), road_user('road_user_motorcycle', [200, 300, 260, 400]),
        road_user('road_user_car', [300, 300, 500, 420]), road_user('road_user_truck', [500, 250, 700, 450]),
        road_user('road_user_bus', [700, 250, 900, 450]), road_user('road_user_tram', [900, 250, 1100, 450]),
        road_user('road_user_other', [1100, 400, 1200, 450]),
        # A sign is background, never a user nor an ignore box.
        {'sign_family': 'maximum_speed', 'sign_type': 'maximum_speed', 'value_kph': 50, 'bbox_xyxy': [600, 100, 640, 140]},
        road_user('road_user_car', None),
    ]))
    assert [user['kind'] for user in frame['users']] == ['person', 'two_wheeler', 'two_wheeler', 'vehicle', 'vehicle',
                                                         'vehicle', 'vehicle']
    assert frame['ignore'] == [[1100.0, 400.0, 1200.0, 450.0]]
    assert (frame['source'], frame['complete'], frame['split'], frame['width']) == ('drive', False, 'train', 1344)


def test_a_frame_is_ticked_on_itself_or_on_any_of_its_rows():
    assert not ticked(reviewed([road_user('road_user_car', [0, 300, 100, 400])]))
    assert ticked(reviewed([], users_complete=True))
    assert ticked(reviewed([road_user('road_user_car', [0, 300, 100, 400], users_complete=True)]))
    assert drive_frame(reviewed([road_user('road_user_car', [0, 300, 100, 400], users_complete=True)]))['complete']


# -- the window --------------------------------------------------------------------------------------------------
def test_the_window_scales_the_frame_and_takes_the_row_offset_off():
    # A 1344-wide frame: half the size in the window, rows 144 to 656 of the frame are the window's 0 to 256.
    users = [{'kind': 'person', 'box': [400, 300, 440, 380]},  # 20 x 40 in the window: a target
             {'kind': 'vehicle', 'box': [600, 400, 640, 420]},  # 10 px tall in the window: too small, ignore
             {'kind': 'vehicle', 'box': [100, 0, 300, 100]},  # above the window: left out
             {'kind': 'two_wheeler', 'box': [700, 60, 760, 160]},  # centre above, reaching in: ignore
             {'kind': 'vehicle', 'box': [1300, 400, 1360, 500]}]  # centre inside, cut at the right edge
    targets, ignored = window_boxes(users, [[0, 600, 50, 700]], 1344)
    assert [(KINDS[kind], box) for kind, box in targets] == [
        ('person', [200.0, 150.0 - WINDOW_TOP, 220.0, 190.0 - WINDOW_TOP]),
        ('vehicle', [650.0, 200.0 - WINDOW_TOP, float(INPUT_WIDTH), 250.0 - WINDOW_TOP])]
    assert ignored == [[300.0, 200.0 - WINDOW_TOP, 320.0, 210.0 - WINDOW_TOP], [350.0, 0.0, 380.0, 80.0 - WINDOW_TOP],
                       [0.0, 300.0 - WINDOW_TOP, 25.0, float(WINDOW_HEIGHT)]]
    # Back in the frame: the row offset added back, the scale undone.
    assert to_frame(targets[0][1], 1344) == [400.0, 300.0, 440.0, 380.0]
    # Mirrored, the boxes mirror too.
    flipped, flipped_ignored = window_boxes(users, [[0, 600, 50, 700]], 1344, flip=True)
    assert flipped[0][1] == [INPUT_WIDTH - 220.0, 78.0, INPUT_WIDTH - 200.0, 118.0]
    assert flipped_ignored[-1] == [INPUT_WIDTH - 25.0, 228.0, float(INPUT_WIDTH), float(WINDOW_HEIGHT)]
    assert all(0 <= box[1] < box[3] <= WINDOW_HEIGHT for _, box in targets + flipped)


def test_a_measure_judges_only_the_users_of_the_size_rule():
    frame = {'width': INPUT_WIDTH, 'users': [
        {'kind': 'person', 'box': [100, 150, 115, 180]},  # 30 px tall: judged
        {'kind': 'person', 'box': [200, 150, 210, 170]},  # 20 px: in the window, below the rule: neither
        {'kind': 'vehicle', 'box': [300, 150, 360, 186]},  # 36 px: below the vehicles' 40: neither
        {'kind': 'vehicle', 'box': [400, 150, 480, 200]},  # 50 px: judged
        {'kind': 'two_wheeler', 'box': [500, 150, 505, 158]}],  # 8 px: an ignore box already in the window
        'ignore': [[600, 150, 650, 200]]}
    truth, dont_care = measured_truth(frame)
    assert truth == {'person': [[100.0, 150.0, 115.0, 180.0]], 'two_wheeler': [],
                     'vehicle': [[400.0, 150.0, 480.0, 200.0]]}
    assert len(dont_care) == 4
    assert COMPLETE_HEIGHT == {'person': 24, 'two_wheeler': 24, 'vehicle': 40}


# -- the stages --------------------------------------------------------------------------------------------------
def ours(split: str, *, complete: bool, users: bool) -> dict:
    return {'source': 'drive', 'image_path': f'frames/{split}-{complete}-{users}.jpg', 'width': 1344, 'height': 760,
            'split': split, 'complete': complete, 'ignore': [],
            'users': [{'kind': 'person', 'box': [400, 300, 440, 380]}] if users else []}


def fake_drives(monkeypatch, frames=None, error=None):
    def drive_frames(*args, **kwargs):
        if error:
            raise error
        return {'frames': frames, 'dataset_sha256': 'd' * 64, 'split_sha256': 's' * 64}
    monkeypatch.setattr(usernet, 'drive_frames', drive_frames)


def test_the_drives_stage_learns_ticked_or_boxed_frames_and_is_judged_on_ticked_ones(monkeypatch, tmp_path):
    fake_drives(monkeypatch, [ours('train', complete=True, users=False), ours('train', complete=False, users=True),
                              ours('train', complete=False, users=False), ours('validation', complete=True, users=True),
                              ours('validation', complete=False, users=True), ours('test', complete=True, users=True),
                              ours('test', complete=False, users=True)])
    plan = prepare_usernet('drives', tmp_path, bdd_dir=tmp_path / 'nowhere')
    assert plan['trainable'] and plan['not_trainable'] is None
    by_split = {split: [f for f in plan['frames'] if f['split'] == split] for split in ('train', 'validation', 'test')}
    # Train: ticked frames (the negatives) and frames with a user box; nothing from a frame with neither.
    assert [(f['complete'], bool(f['users'])) for f in by_split['train']] == [(True, False), (False, True)]
    # Validation and test: the ticked frames only.
    assert all(f['complete'] for f in by_split['validation'] + by_split['test'])
    assert len(by_split['validation']) == len(by_split['test']) == 1
    assert (plan['stage'], plan['uses_bdd'], plan['validation_source'], plan['bdd']) == ('drives', False, 'drives', None)
    assert plan['counts_by_split']['validation']['measured_users'] == 1
    assert plan['drives'] == {'frames': 7, 'ticked': 3, 'with_users': 5}


def test_without_ticked_validation_frames_the_drives_stage_says_so(monkeypatch, tmp_path):
    fake_drives(monkeypatch, [ours('train', complete=False, users=True), ours('validation', complete=False, users=True)])
    plan = prepare_usernet('drives', tmp_path)
    assert not plan['trainable'] and 'key J' in plan['not_trainable']
    fake_drives(monkeypatch, [ours('validation', complete=True, users=False)])
    plan = prepare_usernet('drives', tmp_path)
    assert 'no train frame with a road user box' in plan['not_trainable']
    assert 'no road user to find' in plan['not_trainable']


def test_the_bdd_stage_trains_on_bdd_and_validates_on_our_ticked_frames_when_there_are(monkeypatch, tmp_path):
    names = names_by_split()
    folder = bdd_folder(tmp_path, [sample(names[split], [detection('car', 100, 300, 300, 450)])
                                   for split in ('train', 'validation', 'test')])
    # No dataset of ours yet: BDD's own validation and test.
    fake_drives(monkeypatch, error=ValueError('dataset_missing'))
    plan = prepare_usernet('bdd', tmp_path, bdd_dir=folder)
    assert plan['trainable'] and (plan['validation_source'], plan['uses_bdd']) == ('bdd', True)
    assert [f['source'] for f in plan['frames']] == ['bdd'] * 3
    assert plan['drives'] == {'not_read': 'dataset_missing'} and plan['bdd']['pictures'] == 3
    # With ticked validation frames of ours: those are validation and test, BDD only trains.
    fake_drives(monkeypatch, [ours('train', complete=True, users=True), ours('validation', complete=True, users=True),
                              ours('test', complete=True, users=True)])
    plan = prepare_usernet('bdd', tmp_path, bdd_dir=folder)
    assert plan['validation_source'] == 'drives'
    assert [(f['split'], f['source']) for f in plan['frames']] == [('train', 'bdd'), ('validation', 'drive'),
                                                                   ('test', 'drive')]
    with pytest.raises(ValueError, match='usernet_stage'):
        prepare_usernet('mapillary', tmp_path)


def test_the_licence_goes_along_the_starts():
    assert uses_bdd({'stage': 'bdd'}, None)
    assert uses_bdd({'stage': 'drives', 'uses_bdd': False}, {'uses_bdd': True})
    assert not uses_bdd({'stage': 'drives', 'uses_bdd': False}, {'uses_bdd': False})
    assert not uses_bdd({'stage': 'drives', 'uses_bdd': False}, None)


# -- the Models tab ----------------------------------------------------------------------------------------------
def result(stage: str = 'drives', **extra) -> dict:
    found = {'recall': 0.8, 'precision': 0.9, 'false_per_frame': 0.2,
             'by_kind': {kind: {'recall': 0.7} for kind in KINDS}}
    return {'task': 'usernet', 'stage': stage, 'best_epoch': 12, 'epochs': 30, 'history': [{}] * 30,
            'parameters': 123456, 'split_sha256': 's' * 64, 'uses_bdd': stage == 'bdd',
            'validation_source': 'drives', 'validation': {'frames': 40, 'at_min_score': found},
            'test': {'frames': 50, 'at_min_score': found}, **extra}


def test_the_digest_of_a_finished_usernet():
    found = digest(result('bdd'))
    assert (found['stage'], found['best_epoch'], found['epochs_run'], found['uses_bdd']) == ('bdd', 12, 30, True)
    assert found['test'] == {'frames': 50, 'recall': 0.8, 'precision': 0.9, 'false_per_frame': 0.2,
                             'recall_by_kind': {'person': 0.7, 'two_wheeler': 0.7, 'vehicle': 0.7}}
    assert digest({'task': 'brain'}) is None and digest(None) is None
    # The start without its path; a start that learnt our drives on another split says so.
    start = {'path': '/secret/u0/usernet-best.pt', 'run': 'u0', 'epoch': 3, 'stage': 'drives', 'split_sha256': 'o' * 64,
             'uses_bdd': True}
    on_models = usernet_digest(result(start=start))
    assert on_models['start'] == {'run': 'u0', 'epoch': 3, 'stage': 'drives', 'uses_bdd': True, 'another_split': True}
    assert not usernet_digest(result(start={**start, 'stage': 'bdd'}))['start']['another_split']
    with pytest.raises(ValueError):
        usernet_digest({'task': 'brain'})


def test_the_ids_of_usernet():
    assert role_of('users:2026-11-12_1200-users/usernet') == 'users'
    assert role_of('users:u1/brain') is None and role_of('brain:u1/usernet') is None
    assert role_of('cutter:u1/usernet') is None


def test_a_usernet_run_is_a_version_of_its_own_and_can_be_put_in_use(tmp_path):
    root = tmp_path / 'data'
    run = root / 'runs' / 'u1'
    run.mkdir(parents=True)
    (run / 'usernet-best.pt').write_text('weights')
    (run / 'result.json').write_text(json.dumps(result('bdd')))
    half = root / 'runs' / 'u2'
    half.mkdir()
    (half / 'usernet-best.pt').write_text('weights of a training stopped half way')
    runs = RunsIndex(root)
    runs.running_runs = lambda: {}
    models = ModelIndex(runs)
    roles = {role['role']: role for role in models.index()['roles']}
    versions = {version['run']: version for version in roles['users']['versions']}
    assert set(versions) == {'u1', 'u2'}
    done = versions['u1']
    assert (done['id'], done['kind'], done['stage'], done['uses_bdd'], done['finished']) == (
        'users:u1/usernet', 'usernet', 'bdd', True, True)
    assert done['detail'] == 'learnt on BDD100K' and done['test']['recall'] == 0.8 and done['selectable']
    assert not versions['u2']['finished'] and not versions['u2']['selectable']
    # No other role lists it.
    assert all(version['run'] not in ('u1', 'u2') for role in ('cutter', 'classifier', 'decider', 'brain')
               for version in roles[role]['versions'])
    entry = models.set_active('users', 'users:u1/usernet', who='admin', via='sign-in')
    saved = json.loads((root / 'models' / 'active.json').read_text())
    assert saved['users'] == entry and saved['cutter'] is None
    with pytest.raises(ModelChoiceError) as refused:
        models.set_active('users', 'users:u2/usernet', who='admin', via='sign-in')
    assert refused.value.reason == 'not_finished'
