"""The front wipers of a drive from the raw CAN of its rlogs: what a segment says, what a drive keeps, which cars count."""
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from nnslr_tools import drive_extract, drive_logs, drive_wipers


def can(t, *frames):
    return NS(which=lambda: 'can', logMonoTime=t, can=[NS(address=address, src=src, dat=bytes(dat)) for address, src, dat in frames])


def params(fingerprint):
    return NS(which=lambda: 'carParams', logMonoTime=0, carParams=NS(carFingerprint=fingerprint))


def wiper(state, src=1):
    return (0x452, src, [state << 6 | 5, 0])


def read(monkeypatch, *messages):
    monkeypatch.setattr(drive_extract, '_messages', lambda path, cereal, car: iter(messages))
    return drive_wipers.read_segment(Path('rlog.zst'), Path('schema'))


def test_a_segment_gives_each_change_and_ends_unknown(monkeypatch):
    found = read(monkeypatch, params('PSA_PEUGEOT_3008'), can(10, wiper(0)), can(20, wiper(0)), can(30, wiper(2)), can(40, wiper(2)))
    assert found == {'car': 'PSA_PEUGEOT_3008', 'series': [[10, 0], [30, 2], [40, None]]}


def test_only_the_two_top_bits_of_the_message_on_the_car_count(monkeypatch):
    found = read(monkeypatch, can(10, (0x453, 1, [0xC0]), wiper(1, src=128), wiper(3)))
    assert found['series'] == [[10, 3], [10, None]]


def test_a_segment_without_the_message_says_nothing(monkeypatch):
    assert read(monkeypatch, can(10, (0x453, 1, [0xC0])))['series'] == []


def test_the_overlap_of_two_minutes_never_hides_the_next():
    parts = [{'series': [[10, 0], [50, None]]}, {'series': [[40, 1], [60, None]]}]
    assert drive_wipers.joined(parts) == [[10, 0], [50, None], [50, 1], [60, None]]


@pytest.fixture
def logs(tmp_path, monkeypatch):
    """Two segments of a drive; their reader counts its calls."""
    folder = tmp_path / 'data/raw/routes/00000001--0123456789'
    for segment in (0, 1):
        (folder / str(segment)).mkdir(parents=True)
        (folder / str(segment) / 'rlog.zst').write_bytes(b'a log')
    monkeypatch.setattr(drive_logs, 'schema_dir', lambda fetch=None: tmp_path)
    monkeypatch.setattr(drive_logs, 'videos_dir', lambda: tmp_path / 'none')
    return tmp_path / 'data'


class Pool:
    """In this process, so that the fake reader is seen."""
    calls = 0

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def map(self, function, jobs):
        Pool.calls += 1
        return [function(job) for job in jobs]


def build(monkeypatch, logs, part):
    monkeypatch.setattr(drive_wipers, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(drive_wipers, 'read_segment', lambda path, schema: part)
    return drive_wipers.build_wipers(logs, ['00000001--0123456789'])['00000001--0123456789']


def test_a_drive_of_our_car_is_kept_and_read_once(monkeypatch, logs):
    Pool.calls = 0
    part = {'car': 'PSA_PEUGEOT_3008', 'series': [[10, 1], [20, None]]}
    assert build(monkeypatch, logs, part)['series'] == [[10, 1], [20, None], [20, 1], [20, None]]
    assert drive_wipers.load_wipers(logs, '00000001--0123456789')[0] == [10, 1]
    build(monkeypatch, logs, part)
    assert Pool.calls == 1


@pytest.mark.parametrize('car', ['SUBARU_FORESTER', None])
def test_another_car_may_use_the_address_for_something_else(monkeypatch, logs, car):
    assert build(monkeypatch, logs, {'car': car, 'series': [[10, 1], [20, None]]})['series'] == []
    assert drive_wipers.load_wipers(logs, '00000001--0123456789') == []


def test_a_drive_not_read_has_no_wipers(tmp_path):
    assert drive_wipers.load_wipers(tmp_path, '00000001--0123456789') == []
