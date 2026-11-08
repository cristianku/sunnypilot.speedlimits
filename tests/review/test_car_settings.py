"""The settings of the comma's vision limit (nnslr_tools.car_settings): kept in the data root, read and saved by
/api/panel/car-settings (no longer shown on Models), sent in speed_limits.json; the same as the constants of vision.py."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from nnslr_tools import car_release, car_settings

VISION = Path('/root/sunnypilot/openpilot/sunnypilot/speed_limits/vision.py')


def test_the_defaults_until_one_is_saved(tmp_path):
    assert car_settings.values(tmp_path) == car_settings.DEFAULTS
    view = car_settings.view(tmp_path)
    assert [setting['key'] for setting in view['settings']] == list(car_settings.DEFAULTS)
    assert view['by'] is None and not view['sent'] and view['settings'][0]['on_car'] is None
    assert 'constant' not in view['settings'][0]


def test_only_the_changed_ones_are_kept_and_null_is_the_default_again(tmp_path):
    record = car_settings.save(tmp_path, {'values': {'zone_confidence': 0.95, 'confirm_frames': 3.0,
                                                     'min_confidence': 0.7}}, 'cristian')
    assert record['values'] == {'confirm_frames': 3, 'zone_confidence': 0.95} and record['by'] == 'cristian'
    values = car_settings.values(tmp_path)
    assert values['zone_confidence'] == 0.95 and values['confirm_frames'] == 3 and isinstance(values['confirm_frames'], int)
    # A key left out keeps its value; null goes back to the default.
    car_settings.save(tmp_path, {'values': {'zone_confidence': None}}, 'cristian')
    assert car_settings.values(tmp_path) == {**car_settings.DEFAULTS, 'confirm_frames': 3}


@pytest.mark.parametrize('body', [
    {'values': {'zone_confidence': 2}}, {'values': {'zone_confidence': '0.95'}}, {'values': {'confirm_frames': True}},
    {'values': {'unknown': 1}}, {'values': {'merge_seconds': float('nan')}}, {'zone_confidence': 0.95}, [],
])
def test_a_wrong_value_is_refused_and_nothing_is_written(tmp_path, body):
    with pytest.raises(car_settings.SettingsError):
        car_settings.save(tmp_path, body, 'cristian')
    assert not (tmp_path / car_settings.SETTINGS_FILE).exists()


def test_a_value_spoiled_in_the_file_is_the_default(tmp_path):
    path = tmp_path / car_settings.SETTINGS_FILE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'values': {'zone_confidence': 5, 'confirm_frames': 4}}))
    assert car_settings.values(tmp_path) == {**car_settings.DEFAULTS, 'confirm_frames': 4}


def test_on_the_car_and_sent(tmp_path):
    car_settings.save(tmp_path, {'values': {'parking_limit': 15}}, 'cristian')
    # The published manifest without settings: the comma's defaults.
    view = car_settings.view(tmp_path, {})
    on_car = {item['key']: item['on_car'] for item in view['settings']}
    assert not view['sent'] and on_car['parking_limit'] == 20.0 and on_car['garage_limit'] == 10.0
    assert car_settings.view(tmp_path, car_settings.values(tmp_path))['sent']


def test_the_manifest_carries_them_and_the_published_ones_are_read(tmp_path):
    manifest = car_release.manifest_of('2026-10-30', settings={'zone_confidence': 0.95})
    assert manifest['settings'] == {'zone_confidence': 0.95}
    assert 'settings' not in car_release.manifest_of('2026-10-30')
    assert car_release.published_settings(tmp_path) is None
    (tmp_path / car_release.MANIFEST).write_text(json.dumps(car_release.manifest_of('2026-10-30')))
    assert car_release.published_settings(tmp_path) == {}
    (tmp_path / car_release.MANIFEST).write_text(json.dumps(manifest))
    assert car_release.published_settings(tmp_path) == {'zone_confidence': 0.95}


@pytest.mark.skipif(not VISION.is_file(), reason='no sunnypilot clone')
def test_the_same_as_vision_py():
    spec = importlib.util.spec_from_file_location('speed_limits_vision', VISION)
    vision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vision)
    for setting in car_settings.SETTINGS:
        constant, low, high = vision.SETTINGS[setting['key']]
        assert (constant, low, high) == (setting['constant'], setting['min'], setting['max'])
        assert getattr(vision, constant) == setting['default'] and type(getattr(vision, constant)) is type(setting['default'])
    assert set(vision.SETTINGS) == set(car_settings.DEFAULTS)


def test_over_http(tmp_path, monkeypatch):
    from review_helpers import RunningServer, json_of

    monkeypatch.setenv(car_release.MODELS_CLONE_ENV, str(tmp_path / 'models'))
    server = RunningServer(tmp_path)
    try:
        status, _, data = server.request('GET', '/api/panel/car-settings')
        assert status == 200 and len(json_of(data)['settings']) == len(car_settings.SETTINGS)
        status, _, data = server.request('POST', '/api/panel/car-settings', body={'values': {'zone_confidence': 0.95}})
        view = json_of(data)
        assert status == 200 and view['ok'] and car_settings.values(tmp_path)['zone_confidence'] == 0.95
        assert {item['key']: item['value'] for item in view['settings']}['zone_confidence'] == 0.95
        status, _, data = server.request('POST', '/api/panel/car-settings', body={'values': {'zone_confidence': 9}})
        assert status == 400 and json_of(data)['error'] == 'invalid_settings'
        server.app.panel.actions_enabled = False
        status, _, data = server.request('POST', '/api/panel/car-settings', body={'values': {'zone_confidence': None}})
        assert status == 403 and json_of(data)['error'] == 'actions_disabled'
        assert car_settings.values(tmp_path)['zone_confidence'] == 0.95
    finally:
        server.close()
