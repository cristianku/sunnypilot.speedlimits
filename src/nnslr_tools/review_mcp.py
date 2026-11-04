"""MCP server of the web review: the operations of ``/api/v1`` as MCP tools.

Two transports, one set of tools:

- stdio: ``nnslr mcp`` (or ``helpers/mcp.sh``), for a client started on the
  machine that has the data root, e.g. Claude Code through ssh;
- Streamable HTTP: ``/mcp`` on the review server (``speedlimits-web``), behind
  the same token as ``/api/v1``. :class:`HttpBridge` hands each request of the
  threaded HTTP server to the SDK's session manager (stateless, JSON responses)
  running in its own event loop.

The tools only call :class:`nnslr_tools.review_service.ReviewService`: same
rules and same SQLite database as the page and the REST API. Every save made
here is an AI review (``reviewer_kind = "ai"``). The panel actions (prepare,
import, train, update) are not exposed.

The official MCP SDK (``pip install -e ".[mcp]"``) is optional and imported
only by :func:`build_server` and :class:`HttpBridge`; :class:`ReviewTools`
works without it.

No ``from __future__ import annotations`` here: the SDK reads the tool
signatures, whose types are defined inside :func:`build_server`.
"""
import asyncio
import base64
import json
import sys
import threading
from pathlib import Path

from nnslr_tools.review_db import ReviewRejected
from nnslr_tools.review_service import DETECTION_FAMILIES, review_schema
from nnslr_tools.sign_types import NOT_A_SIGN_TYPES, SIGN_TYPES

SERVER_NAME = 'nnslr-review'
# Neighbour images that get_frame_context attaches at most, to keep answers small.
MAX_CONTEXT_IMAGES = 4
BRIDGE_TIMEOUT_SECONDS = 120

INSTRUCTIONS = """Review of Swiss speed-limit signs proposed by a model on dashcam frames.
Typical flow: list_routes -> next_frame(route_id) -> get_frame_image(...) -> look at the image ->
get_frame_context(...) when the sign is far, covered, blurred or ambiguous -> save_frame_review(...)
-> next_frame(route_id, after=<frame_key>) until it returns frame null.
When the previous frame is confirmed, apply_previous_frame(...) carries its labels onto the same signs here:
check each carried box on this image; boxes with check_position true must be fixed or dropped first.
Model proposals and the second reading are hints, not truth. Never invent a speed: unreadable number
-> value_kph null. Other traffic signs (direction, town entry, give way, no entry, danger...) get a box too:
family other_sign with their sign_type; logos and advertising stay out. Call get_review_schema for the allowed values and the rules.
Save only when the owner has authorized saving for this route; otherwise only read. Your saves are
recorded as an AI review (reviewer stored as "ai/<name>"), never as a person, and are final: no person
approves them again, the next dataset import takes them as labels. Send run_id and
base_revision exactly as read; on a 409 conflict read the frame again instead of overwriting."""


def sdk_available() -> bool:
    try:
        import mcp.server.mcpserver  # noqa: F401
    except ImportError:
        return False
    return True


class ReviewTools:
    """The MCP tools as plain methods returning JSON-ready dicts (and image bytes), without the SDK."""

    def __init__(self, service, *, client: str, read_only: bool = False):
        self.service = service
        self.client = client
        self.read_only = read_only

    def list_routes(self) -> dict:
        return {'routes': self.service.list_routes()}

    def list_frames(self, route_id, frame_filter='unreviewed', cursor=None, limit=50) -> dict:
        return self.service.list_frames(route_id, frame_filter=frame_filter, cursor=cursor, limit=limit)

    def next_frame(self, route_id, frame_filter='unreviewed', after=None) -> dict:
        return self.service.next_frame(route_id, frame_filter=frame_filter, after=after)

    def get_frame(self, route_id, frame_key) -> dict:
        return self.service.get_frame(route_id, frame_key)

    def get_frame_image(self, route_id, frame_key, variant='preview', crop=None, long_side=None) -> tuple[dict, dict]:
        """(description, image) where image has 'data' (bytes) and 'mime_type'."""
        image = self.service.image(route_id, frame_key, variant=variant, crop=crop, long_side=long_side)
        info = {'route_id': route_id, 'frame_key': image['frame_key'], 'variant': image['variant'],
                'crop': image['crop'], 'mime_type': image['mime_type'], 'width': image['width'],
                'height': image['height'], 'bytes': len(image['data']),
                'coordinates': 'bbox_xyxy and crop are in pixels of the original image'}
        return info, image

    def get_frame_context(self, route_id, frame_key, before=2, after=2, include_images=False,
                          long_side=None) -> tuple[dict, list[dict]]:
        """(neighbours, images): images of the nearest neighbours only when include_images is true."""
        context = self.service.frame_context(route_id, frame_key, before=before, after=after)
        images = []
        if include_images:
            others = []
            for frame in context['frames']:
                if frame['offset'] != 0:
                    others.append(frame)
            others.sort(key=lambda frame: (abs(frame['offset']), frame['offset']))
            for frame in others[:MAX_CONTEXT_IMAGES]:
                image = self.service.image(route_id, frame['frame_key'], long_side=long_side)
                images.append({'frame_key': frame['frame_key'], 'offset': frame['offset'], 'image': image})
            context['images_included'] = []
            for item in images:
                context['images_included'].append(item['frame_key'])
        return context, images

    def get_review_schema(self) -> dict:
        return review_schema()

    def get_review_progress(self, route_id, suspicious_limit=50) -> dict:
        return self.service.progress(route_id, suspicious_limit=suspicious_limit)

    def get_review_history(self, route_id, frame_key) -> dict:
        return self.service.history(route_id, frame_key)

    def apply_previous_frame(self, route_id, frame_key, detections=None) -> dict:
        body = {}
        if detections is not None:
            body['detections'] = detections
        return self.service.apply_previous(route_id, frame_key, body)

    def save_frame_review(self, route_id, frame_key, *, run_id, base_revision, reviewer, detections,
                          confirmed=True, site_group=None, encounter_id=None, override_human_review=False,
                          country=None) -> dict:
        if self.read_only:
            raise ReviewRejected(403, 'read_only', 'Read-only MCP server: saving is disabled.')
        body = {'run_id': run_id, 'base_revision': base_revision, 'reviewer': reviewer, 'reviewer_kind': 'ai',
                'confirmed': confirmed, 'detections': detections, 'site_group': site_group,
                'encounter_id': encounter_id, 'override_human_review': override_human_review}
        # Omitted: the country the GPS position of the frame gives.
        if country is not None:
            body['country'] = country
        return self.service.save_review(route_id, frame_key, body, client=self.client, channel='mcp')


def build_server(service, *, client: str, read_only: bool = False):
    """The MCP server (official SDK) with the review tools. Needs the optional "mcp" dependency."""
    from typing import Annotated, Any, Literal

    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp_types import CallToolResult, ImageContent, TextContent, ToolAnnotations
    from pydantic import Field
    from typing_extensions import NotRequired, TypedDict

    tools = ReviewTools(service, client=client, read_only=read_only)
    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS, log_level='WARNING')
    reading = ToolAnnotations(read_only_hint=True, open_world_hint=False)

    def call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ReviewRejected as exc:
            raise ToolError(json.dumps({'error': exc.reason, 'status': exc.status, 'message': exc.message},
                                       ensure_ascii=False)) from exc

    def image_content(image: dict):
        return ImageContent(type='image', data=base64.b64encode(image['data']).decode('ascii'),
                            mime_type=image['mime_type'])

    RouteId = Annotated[str, Field(description='Route id, e.g. "0000011e--0c8825628a" (from list_routes).')]
    FrameKey = Annotated[str, Field(description='Frame key "<segment>/<image>", e.g. "3/28".')]
    Filter = Annotated[Literal['unreviewed', 'reviewed', 'suspicious', 'with_proposals', 'all'],
                       Field(description='unreviewed: not confirmed yet (withdrawn included); reviewed: confirmed; '
                                         'suspicious: model and second reading disagree, or a neighbour has a sign '
                                         'and this frame nothing; with_proposals: at least one model box.')]
    LongSide = Annotated[int | None, Field(ge=32, le=2048, description='Resize so that the longer side has this '
                                                                       'many pixels (enlarges small crops). '
                                                                       'Needs Pillow on the server.')]

    class RouteSummary(TypedDict):
        route_id: str
        run_id: str
        frame_count: int
        confirmed_count: int
        confirmed_by_human: int
        confirmed_by_ai: int
        missing_count: int
        withdrawn_count: int
        suspicious_count: int
        frames_with_proposals: int
        hint_frames: int
        # title ("Herznach → Zürich"), when ("Fri 23 Oct 2026 · 08:01"), minutes, start, from, to; None if unknown.
        drive: dict[str, Any]

    class RouteList(TypedDict):
        routes: list[RouteSummary]

    class FrameList(TypedDict):
        route_id: str
        run_id: str
        filter: str
        total: int
        frames: list[dict[str, Any]]
        next_cursor: str | None

    class FrameDetail(TypedDict):
        route_id: str
        run_id: str
        frame_key: str
        segment_index: int
        output_index: int
        position: int
        frame_count: int
        status: str
        revision: int
        reviewer: str | None
        reviewer_kind: str | None
        width: int | None
        height: int | None
        proposal_count: int
        second_reading_count: int
        suspicious: bool
        suspicious_reasons: list[str]
        saved_box_count: NotRequired[int]
        proposals: list[dict[str, Any]]
        gps: dict[str, Any] | None
        country_proposed: str | None
        second_reading: list[dict[str, Any]]
        review: dict[str, Any] | None
        previous_frame_key: str | None
        next_frame_key: str | None
        image: dict[str, Any]
        metadata: dict[str, Any]
        save: dict[str, Any]

    class NextFrame(TypedDict):
        route_id: str
        run_id: str
        filter: str
        remaining: int
        wrapped: bool
        frame: FrameDetail | None

    class Progress(RouteSummary):
        suspicious: list[dict[str, Any]]
        suspicious_truncated: bool
        confirmed_with_doubts: dict[str, list[str]]
        first_unreviewed: str | None

    class History(TypedDict):
        route_id: str
        run_id: str
        frame_key: str
        history: list[dict[str, Any]]

    class Detection(TypedDict, total=False):
        detection_id: str
        bbox_xyxy: list[float]
        family: Literal[DETECTION_FAMILIES]
        value_kph: int | None
        applicability: Literal['own_road', 'other_road', 'unresolved']
        visibility: Literal['unknown', 'clear', 'partial', 'poor']
        occlusion: Literal['unknown', 'none', 'partial', 'heavy']
        supplementary_panel: Literal['unknown', 'none', 'present', 'time', 'unresolved']
        temporary: bool | None
        variable_display: bool | None
        sign_type: Literal[SIGN_TYPES + NOT_A_SIGN_TYPES] | None
        painted: bool | None
        notes: str
        # From apply_previous_frame: a save refuses check_position true (look, fix the box, drop the flag).
        check_position: bool
        carried_from: str

    class SaveResult(TypedDict):
        ok: bool
        review: dict[str, Any]
        status: str
        confirmed_count: int
        missing_count: int
        next_unreviewed_frame_key: str | None

    @server.tool(annotations=reading)
    def list_routes() -> RouteList:
        """Routes with a pre-annotation report and their progress: current run_id, frames, confirmed
        (by humans and by AI), missing, suspicious, and the drive (title "Herznach → Zürich", when, minutes)
        to tell the routes apart as the site names them."""
        return call(tools.list_routes)

    @server.tool(annotations=reading)
    def list_frames(route_id: RouteId, filter: Filter = 'unreviewed',
                    cursor: Annotated[str | None, Field(description='next_cursor of the previous page.')] = None,
                    limit: Annotated[int, Field(ge=1, le=200)] = 50) -> FrameList:
        """Frames of a route in timeline order (segment, image), one page at a time, without images."""
        return call(tools.list_frames, route_id, filter, cursor, limit)

    @server.tool(annotations=reading)
    def next_frame(route_id: RouteId, filter: Filter = 'unreviewed',
                   after: Annotated[str | None, Field(description='Frame key to continue after (wraps to the '
                                                                  'start once). Omit to resume from the first '
                                                                  'unconfirmed frame.')] = None) -> NextFrame:
        """The next frame to review with its proposals, second reading, suspicious reasons and saved review;
        frame is null when nothing matches. Confirmed frames are skipped, so this resumes after an
        interruption. Get the picture with get_frame_image."""
        return call(tools.next_frame, route_id, filter, after)

    @server.tool(annotations=reading)
    def get_frame(route_id: RouteId, frame_key: FrameKey) -> FrameDetail:
        """One frame: original metadata, image size, model proposals, second reading, suspicious reasons,
        saved review, run_id and revision (send both back when saving)."""
        return call(tools.get_frame, route_id, frame_key)

    @server.tool(annotations=reading)
    def get_frame_image(route_id: RouteId, frame_key: FrameKey,
                        variant: Annotated[Literal['preview', 'original'],
                                           Field(description='preview: the report JPEG; original: the '
                                                             'lossless extracted PNG.')] = 'preview',
                        crop: Annotated[list[float] | None,
                                        Field(min_length=4, max_length=4,
                                              description='[x1, y1, x2, y2] in original pixels, to zoom on a '
                                                          'sign (needs Pillow on the server).')] = None,
                        long_side: LongSide = None) -> CallToolResult:
        """The frame as an MCP image you can look at, plus a short JSON description. Ask for crops of
        small signs (e.g. a box enlarged by 50%) with long_side 512 instead of guessing a number."""
        info, image = call(tools.get_frame_image, route_id, frame_key, variant, crop, long_side)
        return CallToolResult(content=[TextContent(type='text', text=json.dumps(info)), image_content(image)],
                              structured_content=info)

    @server.tool(annotations=reading)
    def get_frame_context(route_id: RouteId, frame_key: FrameKey,
                          before: Annotated[int, Field(ge=0, le=10)] = 2,
                          after: Annotated[int, Field(ge=0, le=10)] = 2,
                          include_images: Annotated[bool, Field(description=f'Attach the images of the '
                                                                            f'{MAX_CONTEXT_IMAGES} nearest '
                                                                            f'neighbours.')] = False,
                          long_side: LongSide = None) -> CallToolResult:
        """Frames before and after one frame in the real report order, across segment boundaries; indices
        need not be consecutive. Use it when a sign is far, covered, blurred or ambiguous. Never copy a
        value from a neighbour: it only helps to read this frame."""
        context, images = call(tools.get_frame_context, route_id, frame_key, before, after, include_images,
                               long_side)
        content = [TextContent(type='text', text=json.dumps(context))]
        for item in images:
            content.append(TextContent(type='text', text=f'Frame {item["frame_key"]} '
                                                         f'(offset {item["offset"]}):'))
            content.append(image_content(item['image']))
        return CallToolResult(content=content, structured_content=context)

    @server.tool(annotations=reading)
    def get_review_schema() -> dict[str, Any]:
        """Fields and allowed values of save_frame_review, the review rules and the error codes."""
        return call(tools.get_review_schema)

    @server.tool(annotations=reading)
    def get_review_progress(route_id: RouteId,
                            suspicious_limit: Annotated[int, Field(ge=0, le=1000)] = 50) -> Progress:
        """Progress of a route: counts, suspicious frames with their reasons, confirmed frames with doubts
        (unknown value, unreadable, road unresolved) and the first unconfirmed frame."""
        return call(tools.get_review_progress, route_id, suspicious_limit)

    class Carried(TypedDict):
        route_id: str
        run_id: str
        frame_key: str
        source_frame_key: str
        source_revision: int
        source_reviewer_kind: str
        detections: list[dict[str, Any]]
        matched: int
        estimated: int
        dropped: int
        kept: int
        duplicates: int
        to_check: list[int]
        message: str
        save: dict[str, Any]

    @server.tool(annotations=reading)
    def apply_previous_frame(route_id: RouteId, frame_key: FrameKey,
                             detections: Annotated[list[Detection] | None, Field(max_length=100, description=(
                                 'The boxes of this frame as you have them now; omit to start from the model '
                                 'proposals.'))] = None) -> Carried:
        """The boxes confirmed on the previous frame carried onto this one (saves nothing). A box of this
        frame that is the same sign (moved and grown) takes the previous labels; a sign not found here gets a
        box where it should be, with check_position true: look at the image, fix its bbox_xyxy or drop it if
        the sign is gone, and remove check_position before save_frame_review (which refuses it otherwise).
        Check every carried value in this image too. 409 previous_not_confirmed: confirm the previous frame
        first."""
        return call(tools.apply_previous_frame, route_id, frame_key, detections)

    @server.tool(annotations=reading)
    def get_review_history(route_id: RouteId, frame_key: FrameKey) -> History:
        """Every saved revision of a frame: reviewer, human or AI, channel, time and boxes."""
        return call(tools.get_review_history, route_id, frame_key)

    if not read_only:
        @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                                 idempotent_hint=False, open_world_hint=False))
        def save_frame_review(
                route_id: RouteId, frame_key: FrameKey,
                run_id: Annotated[str, Field(description='run_id of the frame as read.')],
                base_revision: Annotated[int, Field(ge=0, description='revision of the frame as read '
                                                                      '(0 if never saved).')],
                reviewer: Annotated[str, Field(min_length=1, max_length=97,
                                               description='Name of the AI reviewer, e.g. "Claude Code"; stored '
                                                           'as "ai/<name>". Never a person\'s name.')],
                detections: Annotated[list[Detection], Field(max_length=100, description=(
                    'The boxes to keep, all of them. [] = no relevant sign in the frame. To keep or correct a '
                    'proposal send its detection_id and only the fields that change (applicability at least); '
                    'a proposal left out is rejected. A new box has no detection_id and needs bbox_xyxy, '
                    'family and value_kph.'))],
                confirmed: Annotated[bool, Field(description='false withdraws a confirmation (only if asked).')] = True,
                site_group: str | None = None,
                encounter_id: str | None = None,
                country: Annotated[str | None, Field(pattern='^[A-Z]{2}$', description=(
                    'ISO code of the country of the frame ("CH", "IT", "PL"); omit to keep country_proposed, '
                    'from the GPS position of the frame.'))] = None,
                override_human_review: Annotated[bool, Field(description='Change a frame confirmed by a human: '
                                                                         'only when the owner asked for it.')] = False,
        ) -> SaveResult:
            """Confirm or correct one frame, after looking at its image. Only when the owner authorized saving
            for this route. The confirmation is final (no human approval follows) and becomes a training label
            at the next dataset import. Same validation as the review page; 409 means the frame changed or the
            report was rebuilt: read it again."""
            return call(tools.save_frame_review, route_id, frame_key, run_id=run_id, base_revision=base_revision,
                        reviewer=reviewer, detections=detections, confirmed=confirmed, site_group=site_group,
                        encounter_id=encounter_id, override_human_review=override_human_review, country=country)

    return server


def _stderr(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def run_stdio(root: Path, *, read_only: bool = False) -> None:
    """``nnslr mcp``: the MCP server on stdin/stdout for a client started on this machine."""
    from nnslr_tools.review_db import ReviewStore, db_path
    from nnslr_tools.review_service import ReviewService
    if not sdk_available():
        raise RuntimeError('the MCP SDK is not installed: pip install -e ".[mcp]"')
    service = ReviewService(root, ReviewStore(db_path(root)), log=_stderr)
    server = build_server(service, client='mcp-stdio', read_only=read_only)
    _stderr(f'nnslr MCP (stdio) sul data root {root}' + (' in sola lettura' if read_only else ''))
    server.run('stdio')


class HttpBridge:
    """Serve /mcp of the threaded review server with the SDK's Streamable HTTP session manager.

    Stateless with JSON responses: every POST is a complete exchange, so no
    session lives between requests and GET (server stream) is refused by the SDK.
    Authentication and Origin checks are done by the review server first.
    """

    def __init__(self, service, *, read_only: bool = False):
        from mcp.server.transport_security import TransportSecuritySettings
        from nnslr_tools.review import MAX_BODY_BYTES
        server = build_server(service, client='mcp-http', read_only=read_only)
        # Host and Origin are checked by the review server (token required, Origin = this server).
        server.streamable_http_app(json_response=True, stateless_http=True, max_request_body_size=MAX_BODY_BYTES,
                                   transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
        self._manager = server.session_manager
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stop = None
        self._failure = None
        self._thread = threading.Thread(target=self._run, name='nnslr-mcp', daemon=True)
        self._thread.start()
        self._ready.wait(10)
        if self._failure is not None:
            raise self._failure

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except BaseException as exc:  # noqa: BLE001 - reported to the constructor
            self._failure = exc
            self._ready.set()

    async def _main(self) -> None:
        self._stop = asyncio.Event()
        async with self._manager.run():
            self._ready.set()
            await self._stop.wait()

    def handle(self, method: str, headers: list, body: bytes, client, server=('127.0.0.1', 0)):
        """(status, headers, body) of one HTTP request to /mcp."""
        future = asyncio.run_coroutine_threadsafe(self._exchange(method, headers, body, client, server), self._loop)
        try:
            return future.result(BRIDGE_TIMEOUT_SECONDS)
        except TimeoutError:
            future.cancel()
            error = {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32603, 'message': 'timeout'}}
            return 504, [('Content-Type', 'application/json')], json.dumps(error).encode('utf-8')

    async def _exchange(self, method, headers, body, client, server):
        raw_headers = []
        for name, value in headers:
            raw_headers.append((name.lower().encode('latin-1'), value.encode('latin-1')))
        scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': method,
                 'scheme': 'http', 'path': '/mcp', 'raw_path': b'/mcp', 'query_string': b'', 'root_path': '',
                 'headers': raw_headers, 'client': tuple(client[:2]), 'server': tuple(server[:2])}
        response = {'status': 500, 'headers': [], 'body': []}
        finished = asyncio.Event()
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            await finished.wait()
            return {'type': 'http.disconnect'}

        async def send(message):
            if message['type'] == 'http.response.start':
                response['status'] = message['status']
                response['headers'] = message.get('headers', [])
            elif message['type'] == 'http.response.body':
                response['body'].append(message.get('body', b''))

        try:
            await self._manager.handle_request(scope, receive, send)
        finally:
            finished.set()
        decoded = []
        for name, value in response['headers']:
            decoded.append((name.decode('latin-1'), value.decode('latin-1')))
        return response['status'], decoded, b''.join(response['body'])

    def close(self) -> None:
        if self._stop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=5)
