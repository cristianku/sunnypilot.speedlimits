# [preannotation] - START
"""Route-based pretrained proposals. Parent process and helpers are stdlib-only."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import ssl
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from nnslr_tools.manifest import parse_route_id, sha256_of
from nnslr_tools.sign_reads import READS_MANIFEST, frame_order
from nnslr_tools.video_frames import VIDEO_MANIFEST
from nnslr_tools.sync import DEFAULT_DATA_ROOT
from nnslr_tools.road_text import ROAD_MODEL_INFO, OCR_INFO

MODEL_ID = 'star092304/traffic-sign-detection-vietnam-yolo'
MODEL_REVISION = '58719e5c03bf5fa2817b59f9c7c3c385d3e92c9d'
MODEL_SHA256 = '9c171487bc80e00ba6a8dadaaec0914b0760b92511bc58f13328ef7f7f7a8063'
MODEL_URL = f'https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/best.onnx'
MODEL_INFO = {
    'repository': MODEL_ID, 'revision': MODEL_REVISION, 'sha256': MODEL_SHA256,
    'url': MODEL_URL, 'model_card_license': 'apache-2.0',
    'embedded_model_license': 'AGPL-3.0', 'training_domain': 'Vietnam traffic signs',
    'swiss_validation': 'not_established', 'input_size': [640, 640],
}


class PreannotationError(ValueError):
    pass


def resolve_data_root(explicit: Path | None = None) -> Path:
    return Path(explicit or os.environ.get('NNSLR_DATA_ROOT') or DEFAULT_DATA_ROOT).expanduser().resolve()


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise PreannotationError(f'path outside data root: {path}')
    return resolved


def frame_row(root: Path, route: str, segment: int, raw, manifest: Path) -> dict:
    """The frame row of a line of a segment's manifest (frames.jsonl, sign_reads.jsonl, video_frames.jsonl): its
    identity checked, its picture inside the route's frames and with the hash it was extracted with."""
    directory = _inside(root, root / 'derived/frames' / route)
    if not isinstance(raw, dict) or raw.get('route_id') != route or raw.get('segment_index') != segment:
        raise PreannotationError(f'frame identity mismatch: {manifest}')
    index = raw.get('output_index')
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise PreannotationError(f'invalid/duplicate output_index: {manifest}')
    name = raw.get('image_path')
    if not isinstance(name, str) or not name:
        raise PreannotationError(f'missing image_path: {manifest}')
    path = Path(name)
    path = _inside(directory, path if path.is_absolute() else root / path)
    if not path.is_file():
        raise PreannotationError(f'frame not found: {path}')
    # [frame-provenance] - START
    image_hash = sha256_of(path)
    if raw.get('image_sha256') is not None and raw['image_sha256'] != image_hash:
        raise PreannotationError(f'frame hash mismatch since extraction: {path}')
    # [frame-provenance] - END
    return {
        'frame_key': f'{segment}/{index}', 'route_id': route, 'segment_index': segment,
        'output_index': index, 'image_path': str(path.relative_to(root)),
        'frame_sha256': image_hash, 'camera_stream': raw.get('camera_stream', 'narrow_road'),
        'requested_sample_fps': raw.get('requested_sample_fps'), 'start_s': raw.get('start_s'),
        # [frame-provenance] - START
        # Preserve explicit source evidence; legacy output_index is never promoted.
        'decoded_frame_index': raw.get('decoded_frame_index'),
        'capture_mono_ns': raw.get('capture_mono_ns'),
        'alignment_status': raw.get('alignment_status', 'unresolved'),
        'alignment_reason': raw.get('alignment_reason', 'legacy_manifest'),
        'capture_time_provenance': raw.get('capture_time_provenance', 'unknown'),
        'media_time_s': raw.get('media_time_s'),
        'media_time_provenance': raw.get('media_time_provenance', 'unknown'),
        'source_video_sha256': raw.get('source_video_sha256'),
        'source_log_sha256': raw.get('source_log_sha256'),
        # [frame-provenance] - END
        **({'comma_read_kph': raw.get('comma_read_kph')} if 'comma_read_kph' in raw else {}),
    }


def discover_frames(root: Path, route: str) -> list[dict]:
    parse_route_id(route)
    directory = _inside(root, root / 'derived/frames' / route)
    manifests = sorted(directory.glob('*/frames.jsonl'), key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else -1)
    if not manifests:
        raise PreannotationError(f'no frame manifests; run: nnslr extract-frames --route {route}')
    rows = []
    seen = set()
    for manifest in manifests:
        _inside(directory, manifest)
        if not manifest.parent.name.isdigit():
            raise PreannotationError(f'non-numeric segment: {manifest.parent.name}')
        segment = int(manifest.parent.name)
        # The pictures at 1 a second, the frames of the comma's reads (nnslr_tools.sign_reads), then the frames
        # between the seconds saved in Sign review (nnslr_tools.video_frames): one of those that a later read took
        # too is the same picture, and its row is the read's.
        lines = [(line, False) for line in manifest.read_text().splitlines()]
        for name, added in ((READS_MANIFEST, False), (VIDEO_MANIFEST, True)):
            extra = manifest.parent / name
            if extra.is_file():
                lines += [(line, added) for line in extra.read_text().splitlines()]
        for line, added in lines:
            if not line.strip():
                continue
            raw = json.loads(line)
            if added and isinstance(raw, dict) and (segment, raw.get('output_index')) in seen:
                continue
            row = frame_row(root, route, segment, raw, manifest)
            if (segment, row['output_index']) in seen:
                raise PreannotationError(f'invalid/duplicate output_index: {manifest}')
            seen.add((segment, row['output_index']))
            rows.append(row)
    if not rows:
        raise PreannotationError('frame manifests contain no frames')
    return sorted(rows, key=frame_order)


def model_path(root: Path) -> Path:
    return _inside(root, root / 'models/preannotation' / MODEL_SHA256 / 'model.onnx')


def _open_download(url=MODEL_URL):
    request = urllib.request.Request(url, headers={'User-Agent': 'nnslr-tools'})
    try:
        return urllib.request.urlopen(request, timeout=60)
    except urllib.error.URLError as exc:
        # Some macOS Python distributions omit the OS CA bundle. Still verify
        # TLS using that bundle; never disable certificate verification.
        ca = Path('/etc/ssl/cert.pem')
        if isinstance(exc.reason, ssl.SSLCertVerificationError) and ca.is_file():
            return urllib.request.urlopen(request, context=ssl.create_default_context(cafile=str(ca)), timeout=60)
        raise


def ensure_model(root: Path) -> Path:
    return _ensure_asset(model_path(root), MODEL_SHA256, MODEL_ID, _open_download)


def ensure_road_model(root: Path) -> Path:
    info = ROAD_MODEL_INFO
    path = _inside(root, root / "models/preannotation" / info["sha256"] / "model.onnx")
    return _ensure_asset(path, info["sha256"], info["repository"], lambda: _open_download(info["url"]))


def _ensure_asset(path, checksum, repository, opener):
    if path.exists():
        if sha256_of(path) != checksum:
            raise PreannotationError(f'model checksum mismatch: {path}; remove only this cached model and retry')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading pretrained ONNX model: {repository}', file=sys.stderr, flush=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.download-', delete=False) as target:
            temporary = Path(target.name)
            with opener() as source:
                while chunk := source.read(1024*1024):
                    target.write(chunk)
        if sha256_of(temporary) != checksum:
            raise PreannotationError('downloaded model checksum mismatch')
        temporary.replace(path)
    except (OSError, urllib.error.URLError) as exc:
        raise PreannotationError(f'model download failed: {exc}') from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def semantic_label(label: str) -> tuple[str, int | None]:
    match = re.fullmatch(r'Speed limit ([1-9][0-9]*)km/h', label)
    if match:
        return 'maximum_speed', int(match[1])
    if label in {'End of all prohibition', 'End of 50km/h speed limit'}:
        return 'cancellation', None
    return 'other_sign', None


def unletterbox(xywh, original_size, resized_size, padding, offset=(0,0)):
    """Map network pixels to native pixels using actual rounded resize sizes."""
    cx, cy, w, h = map(float, xywh)
    if not all(math.isfinite(x) for x in (cx,cy,w,h)) or w <= 0 or h <= 0:
        return None
    ow,oh = original_size; rw,rh = resized_size; px,py = padding; ox,oy = offset
    x1=max(0,min(ow,(cx-w/2-px)*ow/rw)); x2=max(0,min(ow,(cx+w/2-px)*ow/rw))
    y1=max(0,min(oh,(cy-h/2-py)*oh/rh)); y2=max(0,min(oh,(cy+h/2-py)*oh/rh))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1+ox,y1+oy,x2+ox,y2+oy]


def _iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/area if area else 0


def suppress_overlaps(detections: list[dict], threshold: float = .5) -> list[dict]:
    for d in detections:
        b=d['bbox_xyxy']; score=d['score']
        if len(b)!=4 or not all(math.isfinite(v) for v in [*b,score]) or not 0<=score<=1 or b[2]<=b[0] or b[3]<=b[1]:
            raise PreannotationError('invalid model detection')
    ordered=sorted(detections,key=lambda d:(-d['score'],d['class_id'],d['bbox_xyxy']))
    keep=[]
    for item in ordered:
        if not any(item['class_id']==other['class_id'] and _iou(item['bbox_xyxy'],other['bbox_xyxy'])>threshold for other in keep):
            keep.append(item)
    return keep


# Who proposes the signs of a route (nnslr preannotate --proposer, the choice of the Prepare tab).
PROPOSERS = {'vietnam': 'Vietnamese traffic-sign YOLO (Hugging Face)',
             'ours': 'Our cutter, classifier and decider',
             'qwen': 'Qwen3-VL-8B, a model that sees images'}
# Short names of the proposers, as the site shows them.
PROPOSER_NAMES = {'vietnam': 'Vietnamese YOLO', 'ours': 'Our networks', 'qwen': 'Qwen3-VL'}


def proposer_of(run: dict) -> str:
    """The proposer of a report (its run.json): its "proposer", or the Vietnamese YOLO of the reports made before
    there was a choice."""
    proposer = run.get('proposer') or (run.get('model') or {}).get('proposer')
    return proposer if proposer in PROPOSERS else 'vietnam'


# Qwen3-VL-8B-Instruct as scripts/setup_vlm.sh installs it (its four weight files), inside the data root.
VLM_REPOSITORY = 'Qwen/Qwen3-VL-8B-Instruct'
VLM_REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
VLM_SHARDS = {'model-00001-of-00004.safetensors': 4902275944, 'model-00002-of-00004.safetensors': 4915962496,
              'model-00003-of-00004.safetensors': 4999831048, 'model-00004-of-00004.safetensors': 2716270024}


# The mark scripts/setup_vlm.sh leaves in vlm_dir after a successful try on the GPU.
VLM_INSTALLED = '.installed'


def vlm_dir(root: Path) -> Path:
    return Path(root) / 'models' / 'vlm' / 'Qwen3-VL-8B-Instruct'


def vlm_missing(root: Path) -> str | None:
    """Why Qwen3-VL cannot propose on this server, or None: its weight files with their sizes, and the packages of
    the GPU environment (scripts/setup_vlm.sh installs both)."""
    folder = vlm_dir(root)
    for name, size in VLM_SHARDS.items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != size:
            return (f'Qwen3-VL is not installed yet: {name} is missing or incomplete in data/models/vlm/'
                    'Qwen3-VL-8B-Instruct; copy its four weight files there, then run scripts/setup_vlm.sh')
    if not any((GPU_PYTHON.parent.parent / 'lib').glob('python*/site-packages/transformers')):
        return 'Qwen3-VL needs transformers in .venv-gpu (scripts/setup_vlm.sh)'
    # Written by scripts/setup_vlm.sh once the sha256 of the weights are right and the model answered on the GPU.
    if not (folder / VLM_INSTALLED).is_file():
        return 'Qwen3-VL is copied here but not checked yet: Install checks its files and tries it on the GPU'
    return None


# Our networks need torch: the GPU environment of this repository.
GPU_PYTHON = Path(__file__).resolve().parents[2] / '.venv-gpu' / 'bin' / 'python'


# Our networks propose only whole (Cristian, 25 October 2026): the cutter finds the boxes, the classifier says what they
# are, the decider whom they apply to (_chain_proposer). Each must be in use (Models tab).
OUR_PARTS = ('cutter', 'classifier', 'decider')


def ours_missing(root: Path) -> str | None:
    """Why Our networks cannot propose on this server, or None: the parts not in use."""
    try:
        active = json.loads((Path(root) / 'models' / 'active.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        active = {}
    missing = [part for part in OUR_PARTS if not (isinstance(active, dict) and active.get(part))]
    if missing:
        return ('Our networks propose only with all three parts in use (Models tab): ' + ', '.join(missing)
                + (' is' if len(missing) == 1 else ' are') + ' not in use yet')
    return None


def networks_in_use(root: Path) -> dict[str, str]:
    """The cutter, classifier and decider in use (models/active.json) by id: what Our networks propose with."""
    try:
        active = json.loads((Path(root) / 'models' / 'active.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return {part: active[part]['id'] for part in OUR_PARTS if isinstance(active, dict) and isinstance(active.get(part), dict)
            and isinstance(active[part].get('id'), str)}


def proposed_by(root: Path, run: dict) -> dict:
    """Who proposed the boxes of a report (its run.json): the proposer, Our networks' parts by id, and whether they are
    the networks in use (in_use), those Propose again proposes with (issue #42). A report of another proposer is not."""
    proposer = proposer_of(run)
    model = run.get('model') if isinstance(run.get('model'), dict) else {}
    networks = {part: model[part] for part in OUR_PARTS if isinstance(model.get(part), str)} if proposer == 'ours' else {}
    in_use = networks_in_use(root)
    return {'proposer': proposer, 'networks': networks, 'in_use': proposer == 'ours' and bool(in_use) and networks == in_use}


def our_networks(root: Path, chosen: dict | None = None) -> dict:
    """The cutter, the classifier and the decider that propose with --proposer ours: those in use (Models tab), only
    when Our networks are whole (ours_missing); chosen: these three instead ({part: id of the Models tab}, the networks
    the brain's signs are read with: sign_networks)."""
    from nnslr_tools.chain import resolve_model
    if chosen is None:
        missing = ours_missing(root)
        if missing:
            raise PreannotationError(missing)
    try:
        cutter = resolve_model(chosen['cutter'] if chosen else 'active', 'cutter', root)
        classifier = resolve_model(chosen['classifier'] if chosen else 'active', 'classifier', root)
        decider = resolve_model(chosen['decider'] if chosen else 'active', 'decider', root)
    except ValueError as exc:
        raise PreannotationError(f"Our networks cannot load a part {'chosen' if chosen else 'in use'}: {exc}") from exc
    return {'repository': 'ours', 'proposer': 'ours', 'training_domain': 'our reviewed Swiss drives',
            'cutter': cutter['id'] or str(cutter['path']), 'cutter_path': str(cutter['path']),
            'classifier': classifier['id'] or str(classifier['path']), 'classifier_path': str(classifier['path']),
            'classifier_chosen': 'chosen' if chosen else 'in use',
            'decider': decider['id'] or str(decider['path']), 'decider_path': str(decider['path'])}


def proposer_setup(proposer: str, root: Path, *, confidence: float | None = None,
                   python_executable: str | None = None, networks: dict | None = None) -> dict:
    """What runs a proposer: its python, worker module, lowest score, the model as the rows record it, and the
    fields of its request. Shared by Prepare (run_preannotation) and the measure of the proposers
    (proposer_eval)."""
    if proposer not in PROPOSERS:
        raise PreannotationError('proposer must be one of ' + ', '.join(PROPOSERS))
    if confidence is None and proposer != 'qwen':
        # Our cutter must miss nothing (its false boxes are dropped by the classifier); the YOLO keeps its own.
        confidence = .1 if proposer == 'ours' else .25
    if proposer != 'qwen' and (not math.isfinite(confidence) or not 0 < confidence <= 1):
        raise PreannotationError('confidence must be in (0, 1]')
    if proposer == 'qwen':
        missing = vlm_missing(root)
        if missing:
            raise PreannotationError(missing)
        model={'repository': VLM_REPOSITORY, 'revision': VLM_REVISION, 'proposer': 'qwen', 'path': str(vlm_dir(root)),
               'model_card_license': 'apache-2.0', 'training_domain': 'general images and text (Alibaba)',
               'swiss_validation': 'not_established'}
        return {'python': python_executable or str(GPU_PYTHON), 'worker': 'nnslr_tools._vlm_proposer',
                'confidence': confidence, 'model': model, 'road_model': None, 'ocr': None, 'request': {'model': model}}
    if proposer == 'ours':
        python=python_executable or str(GPU_PYTHON)
        check=subprocess.run([python,'-c','import torch, torchvision'],capture_output=True,text=True)
        if check.returncode:
            raise PreannotationError('our networks need the GPU environment (.venv-gpu with torch): see docs/INSTALL.md')
        model=our_networks(root, networks)
        # Our networks run on a card only: a server without one stops here and says so, never on the CPU (AGENTS.md).
        return {'python': python, 'worker': 'nnslr_tools._chain_proposer', 'confidence': confidence, 'model': model,
                'road_model': None, 'ocr': None, 'request': {'model': model, 'require_gpu': True}}
    python=python_executable or sys.executable
    check=subprocess.run([python,'-c','import numpy, PIL, onnxruntime, rapidocr'],capture_output=True,text=True)
    if check.returncode:
        raise PreannotationError('preannotation dependencies missing; in the nnslr environment run: python -m pip install ".[preannotate]"')
    model=ensure_model(root)
    road_model=ensure_road_model(root)
    return {'python': python, 'worker': 'nnslr_tools._preannotate_worker', 'confidence': confidence,
            'model': MODEL_INFO, 'road_model': ROAD_MODEL_INFO, 'ocr': OCR_INFO,
            'request': {'model_path': str(model), 'model': MODEL_INFO, 'road_model_path': str(road_model),
                        'road_model': ROAD_MODEL_INFO, 'ocr': OCR_INFO}}


def run_proposer_worker(setup: dict, root: Path, frames: list[dict], output: Path, run_id: str, route: str, *,
                        gpu: str | None = None, log_path: Path | None = None) -> list[dict]:
    """Runs the worker of a proposer_setup on frames, writing into output (images/ and preannotations.jsonl);
    returns its rows, one per frame in order. gpu: the card it runs on, by nvidia-smi's number (None: the cards this
    process sees); log_path: the file its lines go to (None: this process's own output)."""
    request={'schema_version':1,'run_id':run_id,'route_id':route,'data_root':str(root),
             'confidence':setup['confidence'],'frames':frames,'output':str(output),**setup['request']}
    request_path=output/'request.json'; request_path.write_text(json.dumps(request,allow_nan=False))
    env=dict(os.environ)
    # The worker imports nnslr_tools from this checkout, also from the GPU environment.
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    if gpu is not None:
        # CUDA counts the cards by PCI bus, as nvidia-smi does.
        env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
    command=[setup['python'],'-m',setup['worker'],str(request_path)]
    if log_path is None:
        proc=subprocess.run(command,capture_output=False,env=env,check=False)
    else:
        with open(log_path,'w',encoding='utf-8') as handle:
            proc=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,env=env,check=False)
    if proc.returncode:
        raise PreannotationError(f'preannotation failed; previous results preserved; diagnostic request: {request_path}')
    rows=[json.loads(line) for line in (output/'preannotations.jsonl').read_text().splitlines()]
    if len(rows)!=len(frames) or [r['frame_key'] for r in rows]!=[r['frame_key'] for r in frames]:
        raise PreannotationError('worker returned incomplete or reordered frames')
    return rows


def run_preannotation(route: str, *, data_root: Path | None = None, confidence: float | None = None,
                      python_executable: str | None = None, proposer: str = 'vietnam',
                      keep_reviews: bool = False) -> dict:
    """A new report of a route's proposals, made the current one. keep_reviews (Propose again on a drive in review):
    the frames confirmed on the current report stay confirmed on the new one (ReviewStore.carry_confirmed), copied
    before the switch: when the copy fails the current report stays."""
    root=resolve_data_root(data_root)
    # The frames first: a route without frames downloads no model.
    frames=discover_frames(root,route)
    setup=proposer_setup(proposer, root, confidence=confidence, python_executable=python_executable)
    confidence=setup['confidence']
    model_info, road_info, ocr_info = setup['model'], setup['road_model'], setup['ocr']
    base=_inside(root,root/'derived/preannotations'/route)
    base.mkdir(parents=True,exist_ok=True)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    pending=base/('.pending-'+run_id); pending.mkdir()
    rows=run_proposer_worker(setup, root, frames, pending, run_id, route)
    from nnslr_tools.review import write_review
    write_review(pending, rows, run_id)
    device='cpu'
    if rows and rows[0].get('inference',{}).get('provider')=='CUDAExecutionProvider':
        device='cuda'
    summary={'schema_version':1,'run_id':run_id,'route_id':route,'proposer':proposer,'model':model_info,
             'road_model':road_info,'ocr':ocr_info,
             'confidence':confidence,'device':device,'frame_count':len(rows),
             'frames_with_proposals':sum(bool(r['detections']) for r in rows),
             'proposal_count':sum(len(r['detections']) for r in rows),
             'speed_proposal_count':sum(d['family']=='maximum_speed' for r in rows for d in r['detections']),
             'road_marking_proposal_count':sum(d['family']=='road_marking_candidate' for r in rows for d in r['detections']),
             'review_state':'pending','ground_truth':False,'capture_time_established':False}
    (pending/'run.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    destination=base/run_id; pending.rename(destination)
    carried=0
    if keep_reviews and (base/'latest.json').is_file():
        from nnslr_tools.review import latest_report
        from nnslr_tools.review_db import ReviewStore, db_path
        if db_path(root).is_file():
            carried=ReviewStore(db_path(root)).carry_confirmed(
                route_id=route, from_run=latest_report(root,route).name, to_run=run_id,
                frame_keys=[row['frame_key'] for row in rows], client='propose-again')
    summary['carried_reviews']=carried
    (destination/'run.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    pointer=base/('.latest-'+uuid.uuid4().hex+'.json')
    pointer.write_text(json.dumps({'run_id':run_id})); pointer.replace(base/'latest.json')
    return {**summary,'output':str(destination),'report':str(destination/'index.html'),
            'review_command':f'nnslr review --route {route}'}
# [preannotation] - END
