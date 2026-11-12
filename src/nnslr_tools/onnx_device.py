# [preannotation] - START
"""Where the pre-annotation runs its ONNX models, and with how many CPU threads.

The GPU when the installed onnxruntime can use it (onnxruntime-gpu, extra "preannotate-gpu"
in pyproject.toml), otherwise the CPU. NNSLR_PREANNOTATE_DEVICE=cpu forces the CPU.
NNSLR_PREANNOTATE_THREADS sets the threads of each model; helpers/1_prepare.sh sets it to
the cores divided by the routes prepared at the same time. Without it: half of the cores
(the panel prepares 2 routes at a time).
"""
from __future__ import annotations

import os
import sys

THREADS_ENV = 'NNSLR_PREANNOTATE_THREADS'
DEVICE_ENV = 'NNSLR_PREANNOTATE_DEVICE'
CUDA = 'CUDAExecutionProvider'
CPU = 'CPUExecutionProvider'
# Set after the GPU failed once in this process: the next models go straight to the CPU.
_gpu_failed = False


def threads() -> int:
    value = os.environ.get(THREADS_ENV, '')
    if value.isdigit() and int(value) > 0:
        return int(value)
    cores = os.cpu_count() or 4
    if cores // 2 < 2:
        return 2
    return cores // 2


def providers(ort) -> list[str]:
    """[CUDA, CPU] when onnxruntime-gpu is installed and not turned off, otherwise [CPU]."""
    if os.environ.get(DEVICE_ENV, '') == 'cpu':
        return [CPU]
    if CUDA not in ort.get_available_providers():
        return [CPU]
    # onnxruntime-gpu[cuda,cudnn] brings the CUDA and cuDNN libraries as nvidia-* wheels:
    # they are not on the library path, preload_dlls loads them from site-packages.
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()
    return [CUDA, CPU]


def session_options(ort):
    # Explicit thread counts: with the default onnxruntime tries to pin its threads to the
    # cores, which an LXC container refuses (pthread_setaffinity_np errors).
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads()
    options.inter_op_num_threads = 1
    return options


def create_session(ort, path, probe_shape=None):
    """An InferenceSession on the GPU when it works there, otherwise on the CPU.

    A GPU that the installed CUDA/cuDNN does not support does not fall back by itself: with
    CUDA 13 (no Volta) the session fails to start, with a cuDNN without V100 kernels the first
    inference fails. So with probe_shape one inference on zeros of that shape is run first;
    on any failure the session is made again on the CPU, the reason is printed and the next
    models of this process go straight to the CPU.
    """
    global _gpu_failed
    chosen = providers(ort)
    if CUDA in chosen and not _gpu_failed:
        try:
            session = ort.InferenceSession(str(path), sess_options=session_options(ort), providers=chosen)
            if probe_shape is not None:
                import numpy as np
                session.run(None, {session.get_inputs()[0].name: np.zeros(probe_shape, dtype=np.float32)})
            return session
        except Exception as exc:
            _gpu_failed = True
            reason = str(exc).strip().splitlines()[-1][:300]
            print(f'GPU not usable, using the CPU: {reason}', file=sys.stderr, flush=True)
    return ort.InferenceSession(str(path), sess_options=session_options(ort), providers=[CPU])


def on_gpu(session) -> bool:
    return session.get_providers()[0] == CUDA
# [preannotation] - END


def check_providers(ort) -> list:
    """onnxruntime for the checks of an export (Publish, the networks' ONNX against PyTorch): the GPU, never the CPU on
    its own. No onnxruntime-gpu: it stops and says so. NNSLR_CHECK_DEVICE=cpu: the tests, a machine with no card."""
    if os.environ.get('NNSLR_CHECK_DEVICE') == 'cpu':
        return [CPU]
    if CUDA not in ort.get_available_providers():
        raise RuntimeError('this environment has no onnxruntime for the GPU (onnxruntime-gpu): the checks of an export '
                           'run on a V100, never on the CPU')
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()
    # The networks are small: the memory they ask for and no more, beside whatever else holds the card.
    return [(CUDA, {'device_id': 0, 'arena_extend_strategy': 'kSameAsRequested', 'cudnn_conv_algo_search': 'HEURISTIC'})]
