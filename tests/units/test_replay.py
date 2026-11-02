from pathlib import Path
import importlib.util


def test_country_gap_does_not_invent_observed_crossing():
    path=Path(__file__).resolve().parents[2]/'scripts/replay_speed_vision_country.py'
    spec=importlib.util.spec_from_file_location('country_replay_script',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[dict(segment_index=1,output_index=1,country='CH'),
          dict(segment_index=1,output_index=2,country=None),
          dict(segment_index=2,output_index=0,country='DE')]
    assert module.transition_segments(rows)==set()
    rows.pop(1)
    assert module.transition_segments(rows)=={1,2}
