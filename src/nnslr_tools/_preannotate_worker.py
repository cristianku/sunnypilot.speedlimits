# [preannotation] - START
"""Optional ONNX worker (GPU when onnxruntime-gpu can use it, else CPU), invoked only in its own interpreter.

A frame is mostly work for one CPU core in Python (the PNG, the tiles, the road mask, the OCR, the JPEG): frames_at_once()
processes propose frames side by side, each with its own models (one GPU context each: two threads on one onnxruntime
CUDA session crash it with an illegal memory access), and the rows are written in the order of the frames."""
from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
import sys
from pathlib import Path

from nnslr_tools.preannotate import MODEL_SHA256, semantic_label, suppress_overlaps, unletterbox
from nnslr_tools.manifest import sha256_of
from nnslr_tools.media import save_preview
from nnslr_tools import onnx_device

FRAMES_ENV = 'NNSLR_PREANNOTATE_FRAMES'


def tile_boxes(width, height, size=640, overlap=128):
    def starts(length):
        if length <= size:
            return [0]
        return sorted(set([*range(0,length-size+1,size-overlap),length-size]))
    full=(0,0,width,height)
    return [full]+[b for y in starts(height) for x in starts(width)
                  if (b:=(x,y,min(width,x+size),min(height,y+size))) != full]


def frames_at_once() -> int:
    """NNSLR_PREANNOTATE_FRAMES, else a quarter of the cores, 1 to 8."""
    value = os.environ.get(FRAMES_ENV, '')
    if value.isdigit() and int(value) > 0:
        return int(value)
    return max(1, min(8, (os.cpu_count() or 4) // 4))


class Proposer:
    """The models of one process and the proposals of one frame."""

    def __init__(self, request):
        import onnxruntime as ort

        if sha256_of(Path(request['model_path'])) != MODEL_SHA256:
            raise ValueError('model checksum mismatch in worker')
        # First model of the worker: its probe decides GPU or CPU also for the road models and the OCR.
        self.session=onnx_device.create_session(ort,request['model_path'],probe_shape=(1,3,640,640))
        # The provider really in use: the GPU can fall back to the CPU.
        self.provider=self.session.get_providers()[0]
        if self.session.get_inputs()[0].shape != [1,3,640,640] or self.session.get_outputs()[0].shape != [1,86,8400]:
            raise ValueError('unexpected model input/output contract')
        self.names=ast.literal_eval(self.session.get_modelmeta().custom_metadata_map['names'])
        if set(self.names)!=set(range(82)) or not all(isinstance(v,str) for v in self.names.values()):
            raise ValueError('invalid class metadata')
        from nnslr_tools._road_text_worker import RoadTextDetector
        self.road_text=RoadTextDetector(request['road_model_path'])
        self.request=request
        self.root=Path(request['data_root'])
        self.output=Path(request['output'])

    def propose(self, ordinal, frame):
        import numpy as np
        from PIL import Image

        request=self.request
        source=self.root/frame['image_path']
        if sha256_of(source)!=frame['frame_sha256']:
            raise ValueError(f'frame changed during inference: {source}')
        with Image.open(source) as raw:
            image=raw.convert('RGB')
        width,height=image.size; detections=[]
        for tile in tile_boxes(width,height):
            crop=image.crop(tile); w,h=crop.size
            scale=min(640/w,640/h); rw,rh=round(w*scale),round(h*scale)
            px,py=(640-rw)//2,(640-rh)//2
            padded=Image.new('RGB',(640,640),(114,114,114))
            padded.paste(crop.resize((rw,rh),Image.Resampling.BILINEAR),(px,py))
            tensor=np.asarray(padded,dtype=np.float32).transpose(2,0,1)[None]/255.0
            predictions=self.session.run(None,{'images':tensor})[0][0].T
            if predictions.shape != (8400,86) or not np.isfinite(predictions).all():
                raise ValueError('invalid model output')
            class_ids=predictions[:,4:].argmax(1)
            scores=predictions[np.arange(len(predictions)),class_ids+4]
            for i in np.flatnonzero(scores>=request['confidence']):
                box=unletterbox(predictions[i,:4],(w,h),(rw,rh),(px,py),tile[:2])
                if box is None:
                    continue
                class_id=int(class_ids[i]); family,value=semantic_label(self.names[class_id])
                detections.append({'bbox_xyxy':box,'class_id':class_id,'model_label':self.names[class_id],
                                   'score':float(scores[i]),'family':family,'value_kph':value,
                                   'value_state':'proposed' if value is not None else 'unknown',
                                   'review_state':'pending','applicability':'unresolved',
                                   'supplementary_panel':'unknown'})
        detections=suppress_overlaps(detections)
        detections.extend(self.road_text.detect(image))
        for i,item in enumerate(detections):
            item['detection_id']=f"{frame['frame_key']}/{i}"
        preview=f'images/{ordinal:08}.jpg'
        save_preview(image,source,self.output/preview)
        return {**frame,'schema_version':1,'kind':'preannotated_frame','run_id':request['run_id'],
                'width':width,'height':height,'preview_path':preview,'detections':detections,
                'review_state':'pending','ground_truth':False,'model':request['model'],
                'road_model':request['road_model'],'ocr':request['ocr'],
                'inference':{'provider':self.provider,'confidence':request['confidence'],
                             'nms_iou':.5,'tiles':len(tile_boxes(width,height))}}


_proposer = None


def _start(request, threads):
    """In each process: its share of the CPU threads for the models, then the models."""
    global _proposer
    os.environ[onnx_device.THREADS_ENV]=str(threads)
    _proposer=Proposer(request)


def _propose(ordinal, frame):
    return _proposer.propose(ordinal, frame)


def run(request):
    frames=request['frames']
    together=min(frames_at_once(),max(1,len(frames)))
    # The models of this process first: a model that does not load stops the worker before any frame.
    first=Proposer(request)
    print(f"Preannotation on {first.provider}, {onnx_device.threads()} CPU threads, {together} frames at a time",
          file=sys.stderr,flush=True)
    output=Path(request['output']); (output/'images').mkdir()
    slim={key:value for key,value in request.items() if key!='frames'}
    pool=None
    if together>1:
        pool=ProcessPoolExecutor(max_workers=together,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_start,initargs=(slim,max(1,onnx_device.threads()//together)))
        rows=pool.map(_propose,range(len(frames)),frames)
    else:
        rows=(first.propose(ordinal,frame) for ordinal,frame in enumerate(frames))
    try:
        with (output/'preannotations.jsonl').open('w') as manifest:
            for ordinal,row in enumerate(rows):
                manifest.write(json.dumps(row,allow_nan=False)+'\n'); manifest.flush()
                print(f"Preannotating {ordinal+1}/{len(frames)}: segment {row['segment_index']}, frame {row['output_index']}, {len(row['detections'])} proposals",file=sys.stderr,flush=True)
    finally:
        # A failed frame: the frames not started are not proposed.
        if pool is not None:
            pool.shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':
    try:
        run(json.loads(Path(sys.argv[1]).read_text()))
    except Exception as exc:
        print(f'preannotation worker error: {exc}',file=sys.stderr)
        sys.exit(1)
# [preannotation] - END
