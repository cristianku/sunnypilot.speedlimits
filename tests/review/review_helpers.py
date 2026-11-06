"""Synthetic pre-annotation reports and an in-thread review server for the web review tests."""
from __future__ import annotations

import copy
import hashlib
import http.client
import json
import struct
import threading
import zlib
from pathlib import Path

from nnslr_tools.review import create_review_server, write_review

ROUTE = '00000001--abcdef1234'
OTHER_ROUTE = '00000002--abcdef1234'
RUN_ID = '20261021T100000Z-0123abcd'
TOKEN = 'test-token-0123456789abcdef'
WIDTH = 32
HEIGHT = 24


def png(path: Path, value: int = 30) -> None:
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack('>IIBBBBB', WIDTH, HEIGHT, 8, 2, 0, 0, 0)
    pixels = zlib.compress((b'\0' + bytes([value, 0, 0]) * WIDTH) * HEIGHT)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', pixels) + chunk(b'IEND', b''))


def proposal(index: int, segment: int = 0) -> dict:
    return {'bbox_xyxy': [1, 2, 15, 20], 'class_id': 3, 'model_label': 'Speed limit 30km/h', 'score': 0.8,
            'family': 'maximum_speed', 'value_kph': 30, 'value_state': 'proposed', 'review_state': 'pending',
            'applicability': 'unresolved', 'supplementary_panel': 'unknown', 'detection_id': f'{segment}/{index}/0'}


def real_previews(root: Path, route: str = ROUTE) -> None:
    """make_report's previews as real pictures, for what opens them."""
    from PIL import Image
    for preview in (root / 'derived/preannotations' / route).glob('*/images/*.jpg'):
        Image.new('RGB', (64, 36)).save(preview, 'JPEG')


def make_report(root: Path, route: str = ROUTE, run_id: str = RUN_ID, count: int = 3, keys=None) -> list[dict]:
    """Frames 0 and 2 have one speed proposal, frame 1 none. Returns the report rows.

    ``keys`` [(segment, image), ...] replaces the frames 0/0 .. 0/<count-1>; the
    frames with an even position in that list have a proposal.
    """
    if keys is None:
        keys = []
        for index in range(count):
            keys.append((0, index))
    count = len(keys)
    rows = []
    for ordinal in range(count):
        segment, index = keys[ordinal]
        image = root / 'derived/frames' / route / str(segment) / f'frame_{index:08d}.png'
        png(image, (ordinal * 40 + segment + int(route[:8], 16)) % 256)
        detections = []
        if ordinal % 2 == 0:
            detections.append(proposal(index, segment))
        rows.append({
            'frame_key': f'{segment}/{index}', 'route_id': route, 'segment_index': segment, 'output_index': index,
            'image_path': str(image.relative_to(root)), 'frame_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
            'camera_stream': 'narrow_road', 'requested_sample_fps': 1, 'start_s': None,
            'decoded_frame_index': None, 'capture_mono_ns': None, 'alignment_status': 'unresolved',
            'alignment_reason': 'legacy_manifest', 'capture_time_provenance': 'unknown', 'media_time_s': None,
            'media_time_provenance': 'unknown', 'source_video_sha256': None, 'source_log_sha256': None,
            'schema_version': 1, 'kind': 'preannotated_frame', 'run_id': run_id, 'width': WIDTH, 'height': HEIGHT,
            'preview_path': f'images/{ordinal:08d}.jpg', 'detections': detections, 'review_state': 'pending',
            'ground_truth': False, 'model': {'repository': 'synthetic'},
        })
    base = root / 'derived/preannotations' / route
    directory = base / run_id
    (directory / 'images').mkdir(parents=True)
    for ordinal in range(count):
        (directory / 'images' / f'{ordinal:08d}.jpg').write_bytes(b'jpeg preview')
    (directory / 'preannotations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (directory / 'run.json').write_text(json.dumps({'run_id': run_id, 'route_id': route}))
    write_review(directory, rows, run_id)
    (base / 'latest.json').write_text(json.dumps({'run_id': run_id}))
    return rows


def page_body(row: dict, *, reviewer: str = 'Cristian', value_kph=None, drop_proposals: bool = False,
              manual_box=None, base_revision=None, run_id: str = RUN_ID, confirmed: bool = True) -> dict:
    """The POST body the page sends for one frame: its edited copy of the proposals."""
    detections = []
    if not drop_proposals:
        detections = copy.deepcopy(row['detections'])
    if value_kph is not None:
        detections[0]['value_kph'] = value_kph
        detections[0]['value_state'] = 'reviewed'
    if manual_box is not None:
        detections.append({'bbox_xyxy': manual_box, 'family': 'unreadable', 'value_kph': None,
                           'value_state': 'unknown', 'score': None, 'model_label': None, 'class_id': None,
                           'review_state': 'pending', 'applicability': 'unresolved',
                           'supplementary_panel': 'unknown', 'detection_id': 'manual-1'})
    body = {'run_id': run_id, 'reviewer': reviewer, 'confirmed': confirmed, 'detections': detections,
            'site_group': None, 'encounter_id': None}
    if base_revision is not None:
        body['base_revision'] = base_revision
    return body


def page_export_row(row: dict, body: dict, review_timestamp: str, *, reviewer_kind: str = 'human',
                    channel: str | None = None) -> dict:
    """What "Scarica revisioni JSONL" writes for the same frame (mirrors the export in review.html).

    A frame an AI confirmed keeps its provenance: label_basis ai_review, reviewer_kind and review_channel.
    """
    detections = []
    for item in body['detections']:
        detection = copy.deepcopy(item)
        detection['review_state'] = 'reviewed'
        if detection.get('value_kph') is None:
            detection['value_state'] = 'unknown'
        else:
            detection['value_state'] = 'reviewed'
        detections.append(detection)
    exported = copy.deepcopy(row)
    # The country of the frame: the one sent, else the one its GPS position gives.
    exported['country'] = body['country'] if 'country' in body else row.get('country_proposed')
    exported.update(reviewer=body['reviewer'], review_timestamp=review_timestamp,
                    site_group=body['site_group'] or row.get('site_group'),
                    encounter_id=body['encounter_id'] or row.get('encounter_id'),
                    kind='reviewed_frame', ground_truth=False, training_ready=False, review_state='reviewed',
                    label_basis='human_review', source_proposals=copy.deepcopy(row['detections']),
                    detections=detections)
    if reviewer_kind == 'ai':
        exported.update(label_basis='ai_review', reviewer_kind='ai', review_channel=channel)
    return exported


class RunningServer:
    def __init__(self, root: Path, **options):
        self.server, self.app = create_review_server(root, host='127.0.0.1', port=0, **options)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.app.close()

    def request(self, method: str, path: str, *, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            sent = {}
            if headers:
                sent.update(headers)
            payload = None
            if body is not None:
                if isinstance(body, bytes):
                    payload = body
                else:
                    payload = json.dumps(body).encode()
                if 'Content-Type' not in sent:
                    sent['Content-Type'] = 'application/json'
            connection.request(method, path, body=payload, headers=sent)
            response = connection.getresponse()
            data = response.read()
            return response.status, response, data
        finally:
            connection.close()

    def post_frame(self, route: str, frame_key: str, body, headers=None):
        return self.request('POST', f'/api/route/{route}/frame/{frame_key}', body=body, headers=headers)


# [site-login] - START
ADMIN = 'admin'
PASSWORD = 'correct horse battery'


def sign_in(server) -> dict:
    """Creates the administrator of a server on the LAN (setup code = TOKEN) and returns the Cookie header
    of its session."""
    token = server.app.auth.setup(TOKEN, ADMIN, PASSWORD, 'test')
    return {'Cookie': f'{server.app.cookie_name}={token}'}
# [site-login] - END


def json_of(data: bytes):
    return json.loads(data.decode('utf-8'))
