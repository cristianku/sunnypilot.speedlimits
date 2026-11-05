"""The proposals of a route made by Qwen3-VL-8B-Instruct, a model that sees images (nnslr preannotate --proposer
qwen; installed by scripts/setup_vlm.sh). Each frame is shown whole with one question: every traffic sign, one line
each with its box, its kind and the number on it. The answer becomes proposals in the same rows as
_preannotate_worker writes. Run with the GPU environment (.venv-gpu) on the request.json written by
preannotate.run_preannotation. The model gives no score: the proposals carry none.

Speed: the model writes its answer a token at a time, so the answer is short (one line per sign, not JSON) and
frames are asked together, as many as the memory of the GPU left by the model holds (batch_for; NNSLR_QWEN_BATCH
chooses; halved when the GPU runs out of memory all the same). A group takes as long as its longest answer: an
answer stops at MAX_TOKENS."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

PROMPT = ('This is a frame of a dashcam in a car driving in Switzerland. Find every traffic sign in it: speed '
          'limits, end of a limit, zone signs, electronic signs over the road, and any other road sign, also small '
          'and far ones. Answer with one line per sign and nothing else: x1 y1 x2 y2 kind speed. Coordinates are '
          'relative, from 0 to 1000. kind is one of limit, end, zone, display, other. speed is the number on the '
          'sign, or - when it has none. Example: 512 431 530 462 limit 80. With no sign, answer none.')
# The kinds of the answer (the short ones of the prompt, and the long ones of earlier answers), as the families of
# the review page.
KINDS = {'limit': 'maximum_speed', 'end': 'cancellation', 'zone': 'zone', 'display': 'variable_display',
         'other': 'other_sign', 'speed_limit': 'maximum_speed', 'end_of_limit': 'cancellation',
         'variable_display': 'variable_display', 'other_sign': 'other_sign'}
LINE = re.compile(r'^\W*(\d+(?:\.\d+)?)[ ,]+(\d+(?:\.\d+)?)[ ,]+(\d+(?:\.\d+)?)[ ,]+(\d+(?:\.\d+)?)[ ,]+([a-z_]+)'
                  r'(?:[ ,]+(\d+|-|null|none))?', re.I)
# Qwen3-VL gives boxes relative to the image, from 0 to 1000.
SCALE = 1000.0
# About 12 lines of signs (a line is some 17 tokens). On Basel → Mumpf a real answer had at most 13 lines; 4 answers in
# 10 went on in a loop to the 400 tokens of before, and a group of frames waits for its longest answer.
MAX_TOKENS = 200
# A frame of a group takes FRAME_MIB of GPU memory (on the V100 the model holds 17 GB, a group of 4 took 1.75 GB more;
# the attention of the images is one image at a time, so it grows with the frames, not faster). SPARE_MIB stays free.
FRAME_MIB = 450
SPARE_MIB = 1024
MAX_BATCH = 32
# An answer in a loop (_in_a_loop): this many boxes stacked in one column, edges within NEAR of each other (0-1000).
STACKED = 3
NEAR = 6


def _items(text: str) -> list:
    """The signs of an answer: its lines "x1 y1 x2 y2 kind speed", or else a JSON list (inside ``` fences or not)."""
    lines = []
    for line in text.splitlines():
        match = LINE.match(line.strip())
        if match:
            speed = match.group(6)
            lines.append({'bbox_2d': [float(match.group(i)) for i in range(1, 5)], 'kind': match.group(5).lower(),
                          'speed': int(speed) if speed and speed.isdigit() else None})
    if lines:
        return lines
    match = re.search(r'\[.*\]', text, re.S)
    if match is None:
        return []
    try:
        items = json.loads(match.group(0))
    except ValueError:
        return []
    return items if isinstance(items, list) else []


def _box_of(item) -> tuple | None:
    box = item.get('bbox_2d') or item.get('bbox') if isinstance(item, dict) else None
    if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
        return None
    return tuple(float(v) for v in box)


def _in_a_loop(boxes: list[tuple]) -> bool:
    """Does the answer repeat itself? Qwen3-VL sometimes writes a box, then the one just under it (or beside it), and
    so on to the edge of the picture and past it: STACKED or more boxes of one column (or row), each starting where
    the one before ends; or the same line STACKED or more times. No sign looks like that."""
    if any(boxes.count(box) >= STACKED for box in boxes):
        return True
    for along in (1, 0):
        across = 1 - along
        lines: dict[tuple, list[tuple]] = {}
        for box in boxes:
            lines.setdefault((round(box[across] / NEAR), round(box[across + 2] / NEAR)), []).append(box)
        for line in lines.values():
            line.sort(key=lambda box: box[along])
            run = 1
            for first, second in zip(line, line[1:]):
                run = run + 1 if abs(second[along] - first[along + 2]) <= NEAR else 1
                if run >= STACKED:
                    return True
    return False


def parse_answer(text: str, width: int, height: int) -> list[dict]:
    """The proposals of an answer (_items), boxes in frame pixels; items without a usable box are dropped, an unknown
    kind is another sign, a speed only when plausible. A line written twice counts once; an answer in a loop
    (_in_a_loop) keeps only its speed signs (limits, ends, zones, displays), its other signs are the loop."""
    every = [item for item in _items(text) if _box_of(item) is not None]
    items, seen = [], set()
    for item in every:
        key = (_box_of(item), item.get('kind'))
        if key not in seen:
            seen.add(key)
            items.append(item)
    if _in_a_loop([_box_of(item) for item in every]):
        items = [item for item in items if KINDS.get(item.get('kind'), 'other_sign') != 'other_sign']
    found = []
    for item in items:
        if not isinstance(item, dict):
            continue
        box = item.get('bbox_2d') or item.get('bbox')
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        x1, x2 = sorted((max(0.0, min(SCALE, x1)), max(0.0, min(SCALE, x2))))
        y1, y2 = sorted((max(0.0, min(SCALE, y1)), max(0.0, min(SCALE, y2))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        kind = item.get('kind') if item.get('kind') in KINDS else 'other_sign'
        family = KINDS[kind]
        speed = item.get('speed')
        value = int(speed) if isinstance(speed, (int, float)) and not isinstance(speed, bool) and 5 <= speed <= 150 \
            and family in ('maximum_speed', 'zone', 'variable_display') else None
        found.append({'bbox_xyxy': [round(x1 * width / SCALE, 1), round(y1 * height / SCALE, 1),
                                    round(x2 * width / SCALE, 1), round(y2 * height / SCALE, 1)],
                      'class_id': None, 'model_label': 'qwen3-vl:' + kind, 'score': None, 'family': family,
                      'value_kph': value, 'value_state': 'proposed' if value is not None else 'unknown',
                      'review_state': 'pending', 'applicability': 'unresolved', 'supplementary_panel': 'unknown'})
    return found


def batch_for(free_bytes: int) -> int:
    """The frames asked together with free_bytes of GPU memory free once the model is loaded: 1 to MAX_BATCH."""
    return max(1, min(MAX_BATCH, (free_bytes // 2**20 - SPARE_MIB) // FRAME_MIB))


def run(request: dict) -> None:
    import os
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    root = Path(request['data_root'])
    model_dir = request['model']['path']
    # The V100 has no bfloat16; sdpa attention (no flash-attention on sm_70).
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.float16, device_map='cuda',
                                                        attn_implementation='sdpa')
    processor = AutoProcessor.from_pretrained(model_dir)
    # Several frames in one generate: the shorter prompts are padded on the left, where generation does not look.
    processor.tokenizer.padding_side = 'left'
    chosen = os.environ.get('NNSLR_QWEN_BATCH')
    batch = max(1, int(chosen)) if chosen else batch_for(torch.cuda.mem_get_info()[0])
    print(f"Proposals with {request['model']['repository']} on the GPU, {batch} frames at a time", file=sys.stderr, flush=True)
    output = Path(request['output'])
    (output / 'images').mkdir()
    frames = request['frames']
    done = 0
    with (output / 'preannotations.jsonl').open('w') as manifest:
        while done < len(frames):
            group = frames[done:done + batch]
            images = []
            for frame in group:
                with Image.open(root / frame['image_path']) as raw:
                    images.append(raw.convert('RGB'))
            conversations = [[{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': PROMPT}]}]
                             for image in images]
            try:
                inputs = processor.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True,
                                                       return_dict=True, return_tensors='pt', padding=True).to(model.device)
                with torch.no_grad():
                    generated = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                batch = batch // 2
                torch.cuda.empty_cache()
                print(f"Out of GPU memory: {batch} frames at a time from now", file=sys.stderr, flush=True)
                continue
            answers = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            for frame, image, answer in zip(group, images, answers):
                ordinal = done
                width, height = image.size
                detections = parse_answer(answer, width, height)
                for i, item in enumerate(detections):
                    item['detection_id'] = f"{frame['frame_key']}/{i}"
                preview = f'images/{ordinal:08}.jpg'
                save_preview(image, root / frame['image_path'], output / preview)
                row = {**frame, 'schema_version': 1, 'kind': 'preannotated_frame', 'run_id': request['run_id'],
                       'width': width, 'height': height, 'preview_path': preview, 'detections': detections,
                       'review_state': 'pending', 'ground_truth': False, 'model': request['model'], 'road_model': None,
                       'ocr': None, 'vlm_answer': answer[:4000],
                       'inference': {'provider': 'CUDAExecutionProvider', 'confidence': None, 'nms_iou': None, 'tiles': 1}}
                manifest.write(json.dumps(row, allow_nan=False) + '\n')
                done += 1
                print(f"Preannotating {done}/{len(frames)}: segment {frame['segment_index']}, frame "
                      f"{frame['output_index']}, {len(detections)} proposals", file=sys.stderr, flush=True)
            manifest.flush()


if __name__ == '__main__':
    try:
        run(json.loads(Path(sys.argv[1]).read_text()))
    except Exception as exc:
        print(f'preannotation worker error: {exc}', file=sys.stderr)
        sys.exit(1)
