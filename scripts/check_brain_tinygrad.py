#!/usr/bin/env python3
"""The brain's ONNX through tinygrad, as the comma compiles it (compile_onnx.py, the --out-of-band pickle runner.StepNetwork
loads), along a whole real drive against what training measured: the check of the network between onnxruntime and the car.

  NNSLR_TINYGRAD=/root/sunnypilot/tinygrad_repo .venv-gpu/bin/python scripts/check_brain_tinygrad.py <run> [--route R] [--float16]

On the server's GPU (DEV=CUDA), with FLOAT16=1 like the comma when --float16 is given. Not the Adreno: timing and the IMAGE
kernels show only on the comma. Writes nothing but the compiled file, in a folder of its own that is removed at the end.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', help='the folder of the run, with export/brain.onnx')
parser.add_argument('--route')
parser.add_argument('--float16', action='store_true')
parser.add_argument('--data-root')
args = parser.parse_args()

tinygrad = Path(os.environ.get('NNSLR_TINYGRAD', '/root/sunnypilot/tinygrad_repo'))
flags = {'DEV': 'CUDA', **({'FLOAT16': '1'} if args.float16 else {})}
os.environ.update(flags)
sys.path.insert(0, str(tinygrad))

import numpy as np  # noqa: E402

from nnslr_tools import brain_export  # noqa: E402
from nnslr_tools.drive_review import DriveReviewStore  # noqa: E402
from nnslr_tools.preannotate import resolve_data_root  # noqa: E402


def compiled_step(path: Path):
    """step(image, extra, hidden) -> (logits, hidden_out) on the compiled network, as runner.StepNetwork runs it."""
    from examples.openpilot.helpers import load_pickle
    from tinygrad.tensor import Tensor

    jits = load_pickle(path, out_of_band=True)
    specs = jits['input_specs'].items()
    hosts = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype, _) in specs}
    buffers = {name: Tensor(host, device='NPY')._buffer() for name, host in hosts.items()}
    inputs = {name: Tensor(np.zeros(shape, dtype=dtype), device=device).realize() for name, (shape, dtype, device) in specs}
    outputs = {name: Tensor(np.zeros(shape, dtype=dtype), device=device).realize()
               for name, (shape, dtype, device) in jits['output_specs'].items()}

    def step(image, extra, hidden):
        for name, value in (('image', image), ('extra', extra), ('hidden', hidden)):
            hosts[name][...] = value
            inputs[name]._buffer().copy_from(buffers[name])
        jits['run'](output_buffers=outputs, **inputs)
        return outputs['speed'].numpy().astype(np.float32), outputs['hidden_out'].numpy().astype(np.float32)

    return step


def main() -> int:
    run = Path(args.run)
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    store = DriveReviewStore(root)
    model, saved = brain_export.load_brain(run / brain_export.CHECKPOINT)
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / 'brain.pkl'
        subprocess.run([sys.executable, str(tinygrad / 'examples' / 'openpilot' / 'compile_onnx.py'),
                        str((run / 'export' / brain_export.BRAIN_FILE).resolve()), str(target), '--out-of-band',
                        '--benchmark-runs', '1'], cwd=folder, check=True, capture_output=True, text=True,
                       env={**os.environ, 'PYTHONPATH': str(tinygrad), 'CACHELEVEL': '0'})
        drive, frames = brain_export.rollout_drive(root, lambda route: store.merged(route, keep=False), tuple(saved['groups']),
                                                   args.route, 'cuda')
        print(json.dumps({'flags': flags, **brain_export.drive_rollout(model, compiled_step(target), drive, frames, 'cuda')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
