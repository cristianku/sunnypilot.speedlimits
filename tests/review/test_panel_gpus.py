"""The two V100s of the panel (issue #29): each card read on its own, by nvidia-smi's index."""
from nnslr_tools import panel as panel_module
from nnslr_tools.panel import Panel

CARDS = ('0, 00000000:81:00.0, Tesla V100-PCIE-32GB, 0, 0, 32768, 31, 25.1\n'
         '1, 00000000:C1:00.0, Tesla V100-PCIE-32GB, 87, 9120, 32768, 54, 160.3\n')
APPS = '00000000:C1:00.0, 429246, /root/.venv-gpu/bin/python, 9100\n00000000:C1:00.0, 429300, ffmpeg, 300\n'


def fake_nvidia(cards=CARDS, apps=APPS):
    def run(arguments):
        return cards if arguments[0].startswith('--query-gpu=') else apps
    return run


def test_each_card_with_its_own_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia())
    system = Panel(tmp_path).system()
    first, second = system['gpus']
    assert (first['index'], first['bus_id'], first['processes'], first['memory_used_mb']) == (0, '81:00.0', 0, 0)
    assert (second['index'], second['bus_id'], second['processes'], second['utilization']) == (1, 'c1:00.0', 2, 87)
    assert [process['name'] for process in second['process_list']] == ['/root/.venv-gpu/bin/python', 'ffmpeg']
    assert second['process_list'][0] == {'pid': 429246, 'name': '/root/.venv-gpu/bin/python', 'memory_mb': 9100}
    # What a page loaded before the update reads: the first card's numbers, the processes of both.
    assert system['gpu']['name'] == 'Tesla V100-PCIE-32GB'
    assert (system['gpu']['utilization'], system['gpu']['processes']) == (0, 2)


def test_no_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(panel_module, '_nvidia', lambda arguments: None)
    system = Panel(tmp_path).system()
    assert (system['gpu'], system['gpus']) == (None, [])


def test_bus_as_nvidia_smi_and_the_driver_write_it():
    assert panel_module._bus('00000000:81:00.0') == '81:00.0'
    assert panel_module._bus('0000:c1:00.0 ') == 'c1:00.0'
