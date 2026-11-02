"""Re-run the migrated upstream vision tests against the unapplied patch."""
import subprocess
import sys


def test_migrated_native_vision_suite(candidate):
    # Native msgq/params are unavailable here. These tests need only vision and
    # custom.capnp, so expose precisely that namespace in the subprocess.
    (candidate/'conftest.py').write_text('''
import importlib.util, sys, types
from pathlib import Path
root=Path(__file__).parent
for name in ('openpilot','openpilot.sunnypilot','openpilot.sunnypilot.speed_limits'):
    module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
spec=importlib.util.spec_from_file_location('openpilot.sunnypilot.speed_limits.vision',
    root/'openpilot/sunnypilot/speed_limits/vision.py')
vision=importlib.util.module_from_spec(spec);sys.modules[spec.name]=vision;spec.loader.exec_module(vision)
''')
    result=subprocess.run([sys.executable,'-m','pytest','openpilot/sunnypilot/speed_limits/tests','-q',
                           '--rootdir',str(candidate)],cwd=candidate,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
