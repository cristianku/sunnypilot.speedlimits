from types import SimpleNamespace

import pytest


def confirm(memory, value, t=0, **kw):
    memory.update(t, [value], **kw)
    return memory.update(t + .2, [value], **kw)


@pytest.mark.parametrize('cc,number,want', [('FR',30,8.3333333333), ('GB',30,13.4112),
                                         ('US',30,13.4112), ('US',25,11.176), ('GB',50,22.352)])
def test_normalize_before_memory(vision, cc, number, want):
    memory = vision.SpeedLimitMemory(cc)
    assert confirm(memory, number) == pytest.approx(want)
    assert memory.speed_limit == pytest.approx(want)


def test_unknown_unit_holds_physical_limit_and_accepts_end(vision):
    memory = vision.SpeedLimitMemory('GB')
    confirm(memory, 50)
    memory.set_country(None)
    assert confirm(memory, 30, t=2) == pytest.approx(22.352)
    assert confirm(memory, 0, t=4) == 0
    assert memory.limit_state == 'unknown'
    assert confirm(vision.SpeedLimitMemory('ZZ'), 30) == 0
    assert confirm(vision.SpeedLimitMemory(), 30) == 0


def test_confirmed_country_change_invalidates_memory_and_candidates(vision):
    memory = vision.SpeedLimitMemory('FR')
    confirm(memory, 30)
    memory.update(1, [50])
    memory.set_country('GB')
    assert memory.speed_limit == 0
    assert memory.update(1.2, [50]) == 0
    assert memory.update(1.4, [50]) == pytest.approx(22.352)
    memory.set_country(None)  # loss does not erase the last confirmed jurisdiction
    assert memory.speed_limit == pytest.approx(22.352)
    memory.set_country('FR')
    assert memory.speed_limit == 0


def test_implicit_country_change_and_unverified_rules(vision):
    memory = vision.SpeedLimitMemory('CH')
    assert memory.update(0, [], road_type='rural') == pytest.approx(22.2222222222)
    memory.set_country('FR')
    assert memory.speed_limit == pytest.approx(22.2222222222)
    memory.set_country('GB')
    assert memory.update(1, [], road_type='rural') == 0
    assert memory.limit_state == 'unknown'
    de = vision.SpeedLimitMemory('DE')
    assert de.update(0, [], road_type='motorway') == 0
    assert de.limit_state == 'none'


@pytest.mark.parametrize('cc', ['CH', 'GB', 'US', None])
def test_legacy_parking_physical_values(vision, cc):
    memory = vision.SpeedLimitMemory(cc)
    memory.garage_known = True
    memory.update(0, [], road_type='parking', v_ego=1)
    assert memory.update(6, [], road_type='parking', v_ego=1) == pytest.approx(5.5555555556)
    memory.update(7, [], road_type='garage', v_ego=1)
    assert memory.update(13, [], road_type='garage', v_ego=1) == pytest.approx(2.7777777778)


def test_explicit_town_number_is_not_implicit_default(vision):
    memory = vision.SpeedLimitMemory('CH')
    confirm(memory, 50, road_type='town')
    memory.update(2, [], road_type='zone30')
    assert memory.update(20, [], road_type='zone30') == pytest.approx(13.8888888889)


def test_raw_legacy_kph_plus_original_unit_and_country(vision):
    memory = vision.SpeedLimitMemory('US')
    raw = {'class': 'maximum_speed:25', 'confidence': .95}
    confirm(memory, 25, raw=raw)
    message = SimpleNamespace()
    memory.fill_log(message)
    assert message.rawSignSpeed == pytest.approx(40.2336)
    assert (message.rawSignValue, message.rawSignUnit, message.rawSignCountry) == (25, 'mph', 'US')
    assert message.rawSignTaken
    memory.set_country('FR')
    memory.fill_log(message)
    assert (message.rawSignSpeed, message.rawSignCountry) == (pytest.approx(40.2336), 'US')
    memory.set_country(None)
    memory.update(3, [30], raw={'class': 'maximum_speed:30', 'confidence': .99})
    memory.fill_log(message)
    assert message.rawSignSpeed == 0
    assert message.rawSignValue == 30 and message.rawSignUnit == 'unknown' and not message.rawSignTaken


def test_invalid_numeric_readings_never_enter_memory(vision):
    memory = vision.SpeedLimitMemory('FR')
    for value in [float('nan'), float('inf'), -30, True]:
        assert confirm(memory, value) == 0


def test_unit_table_is_independent_of_implicit_legal_rules(vision,monkeypatch):
    monkeypatch.setattr(vision,'IMPLICIT_LIMITS',{})
    assert confirm(vision.SpeedLimitMemory('FR'),30)==pytest.approx(8.3333333333)
