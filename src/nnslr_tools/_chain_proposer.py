"""The proposals of a route made by our own networks (nnslr preannotate --proposer ours): the cutter in use finds the
boxes (a SignNet: its best 5 a frame in the frame's top band, from the request's confidence, the car keeping those of
0.3 or more; an old Detector: on the whole frame), the classifier reads each box cut from the full-resolution frame,
and every box it does not call
"not a sign" becomes a proposal, in the same rows as _preannotate_worker writes; the decider in use then says whom
each speed sign applies to (its applicability: own_road, other_road, or unresolved when it abstains), with the car's
speed from the route's positions (nnslr gps) when they exist. Run with the GPU environment (.venv-gpu: torch,
torchvision) on the request.json written by preannotate.run_preannotation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from nnslr_tools.sign_types import class_parts


def proposal_of(label: str) -> tuple[str, int | None] | None:
    """(family, value) of a classifier label ("maximum_speed:60", "maximum_speed_hours:30", "variable_display:unknown",
    "other_sign", "road_marking:30"); None for not_a_sign. Its panel: proposal_panel."""
    if label == 'not_a_sign':
        return None
    family, value, _ = class_parts(label)
    family = {'road_marking': 'road_marking_candidate'}.get(family, family)
    return family, int(value) if value.isdigit() else None


def proposal_panel(label: str) -> str:
    """The supplementary_panel of a proposal: "time" for a number with hours, else "unknown"."""
    return class_parts(label)[2] or 'unknown'


def dataset_family(family: str) -> str:
    """The family of a proposal as the dataset (and the decider) calls it: road_marking for road_marking_candidate."""
    return {'road_marking_candidate': 'road_marking'}.get(family, family)


def car_speed(root: Path, positions: dict, frame: dict, route: str) -> float | None:
    """The speed (m/s) of the car at a frame from its route's positions (nnslr gps), None without them. positions
    keeps each route's, read once; the frames of a measure of the proposers come from several routes."""
    from nnslr_tools.gps import load_positions
    from nnslr_tools.manifest import NnslerManifestError
    route = frame.get('route_id') or route
    if route not in positions:
        try:
            positions[route] = load_positions(root, route)
        except NnslerManifestError:
            positions[route] = {}
    speed = (positions[route].get(frame['frame_key']) or {}).get('speed_mps')
    return float(speed) if type(speed) in (int, float) and speed >= 0 else None


def run(request: dict) -> None:
    import torch
    import torchvision.transforms.functional as TF
    from nnslr_tools import chain
    from nnslr_tools.chain import (CLASSIFIER_BATCH, _device, _load_classifier, _load_cutter, crop_cutter,
                                   cutter_boxes, cutter_kind)
    from nnslr_tools.context_experiment import Decider
    from nnslr_tools.media import save_preview
    from nnslr_tools.preannotate import suppress_overlaps
    from nnslr_tools.signnet import CAR_MAX_BOXES
    from nnslr_tools.training import crop_rectangle

    root = Path(request['data_root'])
    # The progress of a training lives in logs/progress.json: this worker reports in its log lines only.
    chain.report_progress = lambda *args, **kwargs: None
    device = _device(torch, 'cuda' if torch.cuda.is_available() else 'cpu')
    proposer = request['model']
    cutter, _, input_size = _load_cutter({'path': Path(proposer['cutter_path']), 'id': proposer['cutter']}, device)
    kind = cutter_kind(cutter)
    # What the rows say of the cutter: a SignNet's band, stride and box cap; a Detector's own non-maximum suppression.
    found_by = ({'cutter_kind': kind, 'band': list(cutter.band), 'stride': cutter.stride, 'max_boxes': CAR_MAX_BOXES}
                if kind == 'signnet' else {'cutter_kind': kind, 'input_size': input_size, 'nms_iou': .5})
    classifier, _, classes, transform = _load_classifier({'path': Path(proposer['classifier_path']),
                                                          'id': proposer['classifier']}, device)
    decider = Decider(Path(proposer['decider_path']), device)
    # The speed of the car at each frame, when nnslr gps has run for its route (Propose again); else the decider is
    # told it is missing, as for the training rows without one.
    positions: dict[str, dict] = {}
    print(f"Proposals with our networks on {device}: cutter {proposer['cutter']} ({kind}), classifier {proposer['classifier']}, "
          f"decider {proposer['decider']}", file=sys.stderr, flush=True)
    output = Path(request['output'])
    (output / 'images').mkdir()
    frames = request['frames']
    with (output / 'preannotations.jsonl').open('w') as manifest:
        for ordinal, (frame, pixels, boxes) in enumerate(cutter_boxes(cutter, device, frames, root,
                                                                      min_score=request['confidence'],
                                                                      batch_size=4, num_workers=2, task='prepare')):
            height, width = int(pixels.shape[1]), int(pixels.shape[2])
            image = TF.to_pil_image(pixels)
            cut = crop_cutter(root, frame, pixels)
            detections = []
            for start in range(0, len(boxes), CLASSIFIER_BATCH):
                chunk = boxes[start:start + CLASSIFIER_BATCH]
                crops = []
                for item in chunk:
                    crops.append(transform(cut(crop_rectangle(item['box'], width, height))))
                with torch.no_grad():
                    probabilities = torch.softmax(classifier(torch.stack(crops).to(device)).float(), dim=1)
                for item, index, confidence in zip(chunk, probabilities.argmax(1).tolist(),
                                                   probabilities.max(1).values.tolist()):
                    proposal = proposal_of(classes[index])
                    if proposal is None:
                        continue
                    family, value = proposal
                    detections.append({'bbox_xyxy': [round(float(v), 1) for v in item['box']], 'class_id': index,
                                       'model_label': classes[index], 'score': round(float(confidence), 4),
                                       'cutter_score': round(float(item['score']), 4), 'family': family,
                                       'value_kph': value, 'value_state': 'proposed' if value is not None else 'unknown',
                                       'review_state': 'pending', 'applicability': 'unresolved',
                                       'supplementary_panel': proposal_panel(classes[index])})
            detections = suppress_overlaps(detections)
            # Whom each speed sign applies to: all the boxes of the frame in one call. The other families keep
            # unresolved: the decider learnt ownership only on the speed families.
            ruled = [item for item in detections if dataset_family(item['family']) in decider.families]
            if ruled:
                speed = car_speed(root, positions, frame, request['route_id'])
                for item, (applicability, probability) in zip(ruled, decider.decide(
                        image, [item['bbox_xyxy'] for item in ruled], speed)):
                    item['applicability'] = applicability
                    item['applicability_score'] = round(probability, 4)
            for i, item in enumerate(detections):
                item['detection_id'] = f"{frame['frame_key']}/{i}"
            preview = f'images/{ordinal:08}.jpg'
            save_preview(image, root / frame['image_path'], output / preview)
            row = {**frame, 'schema_version': 1, 'kind': 'preannotated_frame', 'run_id': request['run_id'],
                   'width': width, 'height': height, 'preview_path': preview, 'detections': detections,
                   'review_state': 'pending', 'ground_truth': False, 'model': proposer, 'road_model': None, 'ocr': None,
                   'inference': {'provider': 'CUDAExecutionProvider' if device.type == 'cuda' else 'CPUExecutionProvider',
                                 'confidence': request['confidence'], 'tiles': 1, **found_by}}
            manifest.write(json.dumps(row, allow_nan=False) + '\n')
            manifest.flush()
            print(f"Preannotating {ordinal + 1}/{len(frames)}: segment {frame['segment_index']}, frame "
                  f"{frame['output_index']}, {len(detections)} proposals", file=sys.stderr, flush=True)


if __name__ == '__main__':
    try:
        run(json.loads(Path(sys.argv[1]).read_text()))
    except Exception as exc:
        print(f'preannotation worker error: {exc}', file=sys.stderr)
        sys.exit(1)
