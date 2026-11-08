"""The signs our networks in use read on the drives the brain learns from (nnslr_tools.brain_signs), with the worker
replaced: what is read again, what a failure leaves, and a training that goes on without the networks."""
import pytest

ROUTE = '00000001--abcdef0123'


def _signs():
    from nnslr_tools import brain_signs
    return brain_signs


def setup_of(tmp_path, size=1):
    model = {}
    for part in ('cutter', 'classifier', 'decider'):
        weights = tmp_path / 'models' / f'{part}.pt'
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b'x' * size)
        model.update({part: f'{part}:run/x', f'{part}_path': str(weights)})
    return {'model': model, 'confidence': 0.1, 'request': {'model': model, 'previews': False}}


FRAMES = [{'frame_key': f'0/{second}', 'segment_index': 0, 'decoded_frame_index': second * 20, 'output_index': second,
           'frame_sha256': str(second)} for second in range(3)]


def worker(calls, fail=False):
    def run(setup, root, frames, work, run_id, route):
        calls.append(route)
        (work / 'request.json').write_text('{}', encoding='utf-8')
        if fail:
            from nnslr_tools.preannotate import PreannotationError
            raise PreannotationError(f'preannotation failed; diagnostic request: {work / "request.json"}')
        return [{**frame, 'detections': [{'family': 'maximum_speed', 'value_kph': 60, 'applicability': 'own_road'}]}
                for frame in frames]
    return run


def test_a_drive_is_read_once_by_the_networks_in_use(tmp_path, monkeypatch):
    signs = _signs()
    from nnslr_tools import preannotate
    calls = []
    monkeypatch.setattr(signs, 'pictures', lambda root, route: FRAMES)
    monkeypatch.setattr(preannotate, 'run_proposer_worker', worker(calls))
    setup = setup_of(tmp_path)
    assert signs.read_drives(tmp_path, [ROUTE], setup=setup, log=lambda line: None) == {ROUTE: 'read'}
    found = signs.load(tmp_path, ROUTE)
    assert found['read_by']['networks'] == signs.networks_stamp(setup['model']) and len(found['rows']) == 3
    assert set(found['rows'][0]) == {'frame_key', 'segment_index', 'decoded_frame_index', 'output_index', 'detections'}
    assert signs.read_drives(tmp_path, [ROUTE], setup=setup, log=lambda line: None) == {ROUTE: 'current'}
    assert calls == [ROUTE]
    # A classifier trained again in place reads the drive again.
    again = setup_of(tmp_path, size=2)
    monkeypatch.setattr(signs, 'stamp_in_use', lambda root: signs.networks_stamp(again['model']))
    assert signs.unread(tmp_path, [ROUTE]) == 1
    assert signs.read_drives(tmp_path, [ROUTE], setup=again, log=lambda line: None) == {ROUTE: 'read'}
    assert signs.unread(tmp_path, [ROUTE]) == 0 and calls == [ROUTE, ROUTE]
    assert not [path for path in signs.cache_path(tmp_path, ROUTE).parent.iterdir() if path.is_dir()]


def test_a_failed_reading_keeps_the_last_one_and_its_work_folder(tmp_path, monkeypatch):
    signs = _signs()
    from nnslr_tools import preannotate
    monkeypatch.setattr(signs, 'pictures', lambda root, route: FRAMES)
    monkeypatch.setattr(preannotate, 'run_proposer_worker', worker([]))
    signs.read_drives(tmp_path, [ROUTE], setup=setup_of(tmp_path), log=lambda line: None)
    before = signs.cache_path(tmp_path, ROUTE).read_bytes()
    monkeypatch.setattr(preannotate, 'run_proposer_worker', worker([], fail=True))
    with pytest.raises(preannotate.PreannotationError, match='diagnostic request'):
        signs.read_drives(tmp_path, [ROUTE], setup=setup_of(tmp_path, size=2), log=lambda line: None)
    assert signs.cache_path(tmp_path, ROUTE).read_bytes() == before
    kept = [path for path in signs.cache_path(tmp_path, ROUTE).parent.iterdir() if path.is_dir()]
    assert len(kept) == 1 and (kept[0] / 'request.json').is_file()


def test_the_reader_runs_in_the_gpu_environment_of_the_training(tmp_path, monkeypatch):
    signs = _signs()
    from nnslr_tools import preannotate
    seen = {}

    def proposer_setup(proposer, root, *, confidence=None, python_executable=None):
        seen.update(proposer=proposer, python=python_executable)
        return {'model': {}, 'confidence': 0.1, 'request': {'model': {}}}

    monkeypatch.setattr(preannotate, 'proposer_setup', proposer_setup)
    monkeypatch.setenv('NNSLR_GPU_VENV', '/opt/gpu')
    assert signs.reader_setup(tmp_path)['request'] == {'model': {}, 'previews': False}
    assert seen == {'proposer': 'ours', 'python': '/opt/gpu/bin/python'}
    monkeypatch.delenv('NNSLR_GPU_VENV')
    signs.reader_setup(tmp_path)
    assert seen['python'] is None


def test_without_our_networks_the_training_goes_on_with_sign_review_alone(tmp_path, monkeypatch, capsys):
    signs = _signs()
    from nnslr_tools import cli, preannotate

    def missing(root):
        raise preannotate.PreannotationError('Our networks propose only with all three parts in use')

    monkeypatch.setattr(signs, 'reader_setup', missing)
    assert cli.main(['brain-signs', '--route', ROUTE, '--data-root', str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert 'WARNING' in out and 'Sign review alone' in out
    assert signs.stamp_in_use(tmp_path) is None and signs.unread(tmp_path, [ROUTE]) is None
