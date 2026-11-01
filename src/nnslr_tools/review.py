# [preannotation] - START
"""Review page and review server.

The server reads the pre-annotation reports (read-only) and writes the reviewed
frames into the review database (:mod:`nnslr_tools.review_db`). It listens on
127.0.0.1 by default; on any other address a browser signs in as the administrator
(:mod:`nnslr_tools.review_auth`, name and password; the first visit creates it with
the access token as setup code), and agents send the access token in a header.

Besides the page it serves the REST API ``/api/v1`` and, when the optional MCP
SDK is installed, the MCP Streamable HTTP endpoint ``/mcp``
(:mod:`nnslr_tools.review_mcp`). Both use the operations of
:mod:`nnslr_tools.review_service`, as the page does, and always need the token
in a header (``Authorization: Bearer`` or ``X-Review-Token``), also on 127.0.0.1.

The AI reviewer of Settings (:mod:`nnslr_tools.llm_providers`, :mod:`nnslr_tools.ai_review`)
belongs to the site, not to /api/v1: only a signed-in browser, or the site on 127.0.0.1,
changes its providers, tests them, asks for a frame's suggestion or starts an AI job; the
token only reads. No answer or log line ever holds a provider's key.

"Fit box" of the page (``/frame/<key>/fit``) fits the reviewer's boxes to the signs under them on
this server's CPU (:mod:`nnslr_tools.box_fit`), with no AI provider; like the AI it needs a
signed-in browser or the site on 127.0.0.1, and nothing is saved.
"""
from __future__ import annotations

import ipaddress
import json
import mimetypes
import os
import re
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from nnslr_tools.drives import drive_of
from nnslr_tools.manifest import parse_route_id
from nnslr_tools.preannotate import PreannotationError, _inside, resolve_data_root
# [site-login] - START
from nnslr_tools.review_auth import SESSION_MAX_SECONDS, AdminAuth, auth_dir
# [site-login] - END
from nnslr_tools.pages import (CONTROL_TABS, OLD_ADDRESSES, render_account, render_control, render_index, render_login,
                               render_problem, render_review)
from nnslr_tools.review_service import ReviewService

TOKEN_ENV = 'NNSLR_REVIEW_TOKEN'
MAX_BODY_BYTES = 512 * 1024
# A connection that sends nothing for this many seconds is closed, and at most
# MAX_CONNECTIONS are open at once, so idle clients cannot use up threads.
REQUEST_TIMEOUT_SECONDS = 30
MAX_CONNECTIONS = 64
# [site-login] - START
LOGIN_MESSAGE = 'Sign-in required: reload the page and sign in with your name and password.'
# Sign-in calls that work without a session, and all of them.
PUBLIC_AUTH_ROUTES = ('/api/auth/login', '/api/auth/setup/check', '/api/auth/setup')
AUTH_ROUTES = PUBLIC_AUTH_ROUTES + ('/api/auth/logout', '/api/auth/logout-all', '/api/auth/password')
# The pages a browser opens: without a session they show the sign-in page instead.
PAGE_RE = re.compile(r'/(index\.html|routes|prepare|dataset|roads|train|evaluate|models|publish|panoramica|prepara|allena|pannello|stato|account|'
                     r'settings|review|route/[^/]+(/(index\.html)?)?)?')
# [site-login] - END
# A drive page whose route has no report (or an id that is no route), in words.
UNKNOWN_DRIVE_MESSAGE = 'This drive is not on the server, or it is not prepared yet.'
API_TOKEN_MESSAGE = ('Missing or wrong token: send it in the header "Authorization: Bearer <token>" '
                     '(or "X-Review-Token"), never in the address.')
DEFAULT_CSP = "default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
# [ai-reviewer] - START
# The AI reviewer (nnslr_tools.llm_providers, nnslr_tools.ai_review): only a signed-in browser, or the site on
# 127.0.0.1, changes the providers, tests them or asks the AI; the access token (an agent) only reads the settings,
# the AI state and the measures, none of which holds a key.
SESSION_REQUIRED_MESSAGE = 'Sign in on the site to use the AI reviewer.'
# At most this many suggestions (and Tests) wait for the model at once: each holds a request thread for minutes.
MAX_AI_SUGGESTIONS = 2
# [box-fit] - START
# Fit box (nnslr_tools.box_fit): at most this many fits run at once (each one CPU core for a few tenths of a second),
# a request waits FIT_WAIT_SECONDS for its turn, and fits at most MAX_FIT_BOXES boxes of one frame.
MAX_LOCAL_FITS = 2
FIT_WAIT_SECONDS = 5.0
MAX_FIT_BOXES = 64
FIT_SESSION_MESSAGE = 'Sign in on the site to fit boxes.'
# [box-fit] - END
AI_SETTINGS_POSTS = ('/api/settings/ai/providers', '/api/settings/ai/reviewer', '/api/settings/ai/models',
                     '/api/settings/ai/test')
AI_DELETE_RE = re.compile(r'/api/settings/ai/providers/([a-z][a-z0-9-]{0,31})/delete')
# Settings > Hugging Face: the token the networks are published with (nnslr_tools.hf_account).
HF_POSTS = ('/api/settings/huggingface', '/api/settings/huggingface/remove')
# [ai-reviewer] - END


def write_review(directory: Path, rows: list[dict], run_id: str) -> None:
    route_id=''
    if rows and isinstance(rows[0],dict) and isinstance(rows[0].get('route_id'),str):
        route_id=rows[0]['route_id']
    (directory/'index.html').write_text(render_review(rows,run_id,route_id=route_id),encoding='utf-8')


def public_file(directory: Path, url: str) -> Path | None:
    name=unquote(urlsplit(url).path)
    if name in {'/','/index.html'}:
        relative=Path('index.html')
    elif re.fullmatch(r'/images/[0-9]{8}\.jpg',name):
        relative=Path(name.lstrip('/'))
    else:
        return None
    path=directory/relative
    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
        return None
    return path


def latest_report(root: Path, route: str) -> Path:
    parse_route_id(route)
    base=_inside(root,root/'derived/preannotations'/route)
    pointer=_inside(base,base/'latest.json')
    if not pointer.is_file():
        raise PreannotationError(f'no report; run: nnslr preannotate --route {route}')
    run_id=json.loads(pointer.read_text()).get('run_id','')
    if not isinstance(run_id,str) or not re.fullmatch(r'[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}',run_id):
        raise PreannotationError('invalid latest report pointer')
    directory=_inside(base,base/run_id)
    if public_file(directory,'/') is None:
        raise PreannotationError('report index is missing')
    return directory


# [reviewed-dataset] - START
def report_rows(directory: Path) -> list[dict]:
    from nnslr_tools.annotations import load_jsonl
    source = _inside(directory, directory / 'preannotations.jsonl')
    return load_jsonl(source)


def refresh_review(directory: Path) -> None:
    """Upgrade the presentation of an existing run, preserving proposals and run identity."""
    metadata = _inside(directory, directory / 'run.json')
    run_id = json.loads(metadata.read_text())['run_id']
    write_review(directory, report_rows(directory), run_id)
# [reviewed-dataset] - END


def list_reports(root: Path) -> list[tuple[str, Path]]:
    """(route, current report directory) for every route with a valid pre-annotation report."""
    base = root / 'derived' / 'preannotations'
    found = []
    if not base.is_dir():
        return found
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        try:
            directory = latest_report(root, entry.name)
        except (PreannotationError, ValueError, OSError):
            continue
        found.append((entry.name, directory))
    return found


def terminal_text(value) -> str:
    """Text from a request made safe for the server terminal: no control characters, bounded length."""
    text = str(value)
    if len(text) > 100:
        text = text[:100] + '…'
    result = []
    for character in text:
        if character.isprintable():
            result.append(character)
        else:
            result.append('?')
    return ''.join(result)


def is_loopback(host: str) -> bool:
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host.strip('[]')).is_loopback
    except ValueError:
        return False


def review_token() -> str:
    """The access token: NNSLR_REVIEW_TOKEN when set, otherwise a new random one."""
    configured = os.environ.get(TOKEN_ENV, '')
    if configured:
        if not re.fullmatch(r'[A-Za-z0-9_-]{16,}', configured):
            raise PreannotationError(f'{TOKEN_ENV} must have at least 16 characters among letters, digits, "-" and "_"')
        return configured
    return secrets.token_urlsafe(24)


def lan_addresses() -> list[str]:
    """IPv4 addresses of this machine that other computers can use (best effort, no traffic sent)."""
    found = []
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        infos = []
    for info in infos:
        address = info[4][0]
        if not is_loopback(address) and address not in found:
            found.append(address)
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # A UDP connect only selects the outgoing interface; nothing is sent.
        probe.connect(('10.255.255.255', 1))
        address = probe.getsockname()[0]
        if not is_loopback(address) and address not in found:
            found.append(address)
    except OSError:
        pass
    finally:
        probe.close()
    return found


def _print_flush(text: str) -> None:
    print(text, flush=True)


class ReviewApp(ReviewService):
    """State shared by the request threads: the review operations plus access token and panel.

    ``api_token`` protects /api/v1 and /mcp. It is the page token when there is
    one; on 127.0.0.1 without a page token it is NNSLR_REVIEW_TOKEN or a random one.
    """

    def __init__(self, root: Path, store, *, token: str | None, require_token: bool, api_token: str | None = None):
        from nnslr_tools.panel import Panel
        super().__init__(root, store, log=_print_flush)
        self.token = token
        self.require_token = require_token
        self.api_token = api_token or token
        self.cookie_name = 'nnslr_review'
        # [site-login] - START
        # On the LAN a browser signs in; the access token is the setup code of the first visit.
        self.auth = AdminAuth(auth_dir(root), token) if require_token else None
        # [site-login] - END
        self._mcp = None
        self._mcp_failed = False
        self._mcp_lock = threading.Lock()
        # [control-panel] - START
        self.panel = Panel(root)
        # [control-panel] - END
        # [ai-reviewer] - START
        self.ai_slots = threading.BoundedSemaphore(MAX_AI_SUGGESTIONS)
        # [ai-reviewer] - END
        # [box-fit] - START
        self.fit_slots = threading.BoundedSemaphore(MAX_LOCAL_FITS)
        # [box-fit] - END

    def token_matches(self, supplied) -> bool:
        if self.token is None or not isinstance(supplied, str):
            return False
        return secrets.compare_digest(supplied.encode('utf-8'), self.token.encode('utf-8'))

    def api_token_matches(self, supplied) -> bool:
        if self.api_token is None or not isinstance(supplied, str):
            return False
        return secrets.compare_digest(supplied.encode('utf-8'), self.api_token.encode('utf-8'))

    def mcp_bridge(self):
        """The MCP Streamable HTTP bridge, started at the first /mcp request; None without the MCP SDK."""
        with self._mcp_lock:
            if self._mcp is None:
                try:
                    from nnslr_tools.review_mcp import HttpBridge, sdk_available
                except ImportError:
                    return None
                if self._mcp_failed or not sdk_available():
                    return None
                try:
                    self._mcp = HttpBridge(self, read_only=os.environ.get('NNSLR_MCP_READ_ONLY', '') == '1')
                except Exception as exc:
                    # An incompatible SDK must not take the site and /api/v1 down: /mcp answers 501.
                    self._mcp_failed = True
                    self.log(f'MCP non disponibile: {type(exc).__name__}: {terminal_text(exc)}')
                    return None
            return self._mcp

    def close(self) -> None:
        with self._mcp_lock:
            if self._mcp is not None:
                self._mcp.close()
                self._mcp = None

    def host_allowed(self, host_header) -> bool:
        # With a token the cookie protects every request. Without one (loopback
        # only) the Host header must be a loopback name, against DNS rebinding.
        if self.require_token:
            return True
        if not isinstance(host_header, str) or not host_header:
            return False
        name = host_header.strip().lower()
        if name.startswith('['):
            end = name.find(']')
            if end < 0:
                return False
            name = name[1:end]
        elif ':' in name:
            name = name.rsplit(':', 1)[0]
        return is_loopback(name)

    def view(self, route: str) -> tuple[Path, str, list[dict], dict | None, str]:
        """What the review page shows of a drive: (folder of its pictures, run id, rows, busy, proposer). Its report,
        or, while Propose again works on it (nnslr_tools.reproposing), the new proposals made so far, read only; busy
        is then the state of the drive for the page, else None. Raises as report() for an unknown drive."""
        from nnslr_tools import reproposing
        directory, rows, by_key = self.report(route)
        item = reproposing.of(self.root, route)
        if item is not None and item['phase'] == 'proposing again':
            fresh = located(reproposing.at_work_rows(item), by_key)
            item = dict(item, done=max(item['done'], len(fresh)))
            return (item['pending'], item['pending'].name, fresh, reproposing.public(item, len(rows)),
                    item.get('proposer_id') or _report_proposer(directory))
        return directory, directory.name, rows, reproposing.public(item, len(rows)), _report_proposer(directory)

    def frame_rows(self, route: str) -> dict:
        """The rows of the current pre-annotation report of a route by frame key ({} without a report): the Watch
        panel of the Evaluate tab shows a frame with its preview."""
        try:
            return self.report(route)[2]
        except (PreannotationError, ValueError, OSError):
            return {}

    def route_links(self) -> list[dict]:
        """Every route with a report, for the route menu of the review page, named as the drive it was
        ("Herznach → Zürich · Fri 23 Oct 2026 · 08:01", nnslr_tools.drives); its title and when apart, for the
        two lines of the menu."""
        links = []
        for route, directory in list_reports(self.root):
            drive = drive_of(self.root, route)
            name = ' · '.join(part for part in (drive['title'], drive['when']) if part) or None
            links.append({'route_id': route, 'url': '/route/' + route + '/', 'name': name, 'title': drive['title'],
                          'when': drive['when'], 'minutes': drive['minutes']})
        return links

    def route_summaries(self) -> list[dict]:
        summaries = []
        for route, directory in list_reports(self.root):
            try:
                directory, rows, by_key = self.report(route)
            except (PreannotationError, ValueError, OSError):
                continue
            with_proposals = 0
            for row in rows:
                detections = row.get('detections')
                if isinstance(detections, list) and detections:
                    with_proposals = with_proposals + 1
            hints_by_frame, hint_stats = self.hints(route, rows)
            suspicious = self.suspicious(route, directory.name, rows, hints_by_frame)
            summaries.append({
                'route_id': route,
                'run_id': directory.name,
                'frame_count': len(rows),
                'frames_with_proposals': with_proposals,
                'confirmed_count': self.store.confirmed_count(route, directory.name),
                # Of those, the frames an AI reviewer confirmed (REST API, MCP, Review with AI).
                'confirmed_by_ai': self.store.confirmed_count(route, directory.name, reviewer_kind='ai'),
                # Saved by the site's AI reviewer, for a person to confirm (review_db.SITE_AI_REVIEWER).
                'waiting_for_person': self.store.waiting_count(route, directory.name),
                'suspicious_count': len(suspicious),
                'hint_frames': hint_stats['frames'],
                'url': '/route/' + route + '/',
                # The route as the drive it was: when, how long, from where to where (nnslr_tools.drives).
                'drive': drive_of(self.root, route),
                # Who proposed the signs of this report: vietnam, ours or qwen (run.json).
                'proposer': _report_proposer(directory),
            })
        return summaries


def located(fresh: list[dict], by_key: dict) -> list[dict]:
    """The new proposals of the drive at work with the GPS position and country of their frame, from its current
    report (ReviewService.report adds them; the frames are the same)."""
    for row in fresh:
        known = by_key.get(row.get('frame_key')) or {}
        for name in ('gps', 'country_proposed', 'country_from'):
            row[name] = known.get(name)
    return fresh


def _report_proposer(directory: Path) -> str:
    from nnslr_tools.preannotate import proposer_of
    try:
        return proposer_of(json.loads((directory / 'run.json').read_text(encoding='utf-8')))
    except (OSError, ValueError):
        return 'vietnam'


def make_handler(app: ReviewApp):
    from nnslr_tools.annotations import strict_json_loads
    from nnslr_tools.panel import PanelError
    from nnslr_tools.review_db import ReviewRejected, db_path, public_review, save_frame_review
    from nnslr_tools.review_service import review_schema

    class Handler(BaseHTTPRequestHandler):
        server_version = 'speedlimits-web'
        sys_version = ''
        # Socket timeout: a client that stops sending is disconnected, even before the token check.
        timeout = REQUEST_TIMEOUT_SECONDS

        def do_GET(self):
            self._dispatch('GET')

        def do_POST(self):
            self._dispatch('POST')

        def do_DELETE(self):
            self._dispatch('DELETE')

        def do_OPTIONS(self):
            self._dispatch('OPTIONS')

        def log_message(self, *args):
            pass

        def _dispatch(self, method):
            try:
                self._handle(method)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
            except ReviewRejected as exc:
                self._problem(exc.status, exc.reason, exc.message, True)
            except PanelError as exc:
                self._problem(exc.status, exc.reason, exc.message, True)
            except Exception as exc:
                self._problem(500, 'server_error', 'Internal server error: ' + type(exc).__name__ + '.',
                              self.path.startswith('/api/') or self.path.startswith('/mcp'))

        def _handle(self, method):
            parts = urlsplit(self.path)
            path = unquote(parts.path)
            api = path.startswith('/api/')
            if not app.host_allowed(self.headers.get('Host')):
                self._problem(403, 'host_not_allowed', 'Host name not allowed for this server.', api)
                return
            query = parse_qs(parts.query, keep_blank_values=True)
            if path == '/mcp' or path == '/api/v1' or path.startswith('/api/v1/'):
                self._external(method, path, query)
                return
            if method not in ('GET', 'POST'):
                self._problem(405, 'method_not_allowed', 'Method not allowed.', api)
                return
            # [site-login] - START
            if 'token' in query:
                # An old link with ?token=: the token no longer opens the site in a browser (it is the
                # setup code of the first visit); drop it from the address, the sign-in page follows.
                self._redirect(self._clean_target(path), None)
                return
            if not self._authorized():
                self._public(method, path, api)
                return
            # [site-login] - END
            if method == 'POST':
                self._post(path)
            else:
                self._get(path)

        # [site-login] - START
        @staticmethod
        def _clean_target(path):
            # Our own paths only, never "//host" or a backslash.
            if not re.fullmatch(r'/[A-Za-z0-9._/-]*', path) or path.startswith('//'):
                return '/'
            return path

        def _authorized(self):
            """A browser signed in as the administrator, or an agent with the access token in a header
            (X-Review-Token, as for /api/v1: a page cannot make another site send it)."""
            if not app.require_token:
                return True
            if app.token_matches(self.headers.get('X-Review-Token')):
                return True
            return self._session() is not None

        def _session(self):
            """The session token of this request when it is signed in, otherwise None."""
            if app.auth is None:
                return None
            # Plain "name=value; name=value" split: the browser also sends the
            # cookies of other web apps on this host (any port), and a value our
            # parser does not like must not hide the review cookie.
            for raw in self.headers.get_all('Cookie', []):
                for part in raw.split(';'):
                    name, separator, value = part.strip().partition('=')
                    if separator and name.strip() == app.cookie_name and app.auth.session_valid(value.strip()):
                        return value.strip()
            return None

        def _session_cookie(self, token):
            parts = [app.cookie_name + '=' + token, 'Path=/', 'HttpOnly', 'SameSite=Strict',
                     'Max-Age=' + str(SESSION_MAX_SECONDS)]
            origin = (self.headers.get('Origin') or '').strip().lower()
            proto = (self.headers.get('X-Forwarded-Proto') or '').strip().lower()
            if origin.startswith('https://') or proto == 'https':
                parts.append('Secure')
            return '; '.join(parts)

        def _expired_cookie(self):
            return app.cookie_name + '=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0'

        def _who(self):
            """Who makes this request, for the record of a choice: (name, how it was recognised)."""
            if app.auth is None:
                return None, 'local'
            if self._session() is not None:
                return app.auth.username(), 'sign-in'
            return None, 'token'

        def _auth_state(self):
            if app.auth is None:
                return {'configured': False, 'signed_in': True, 'local': True}
            signed_in = self._session() is not None
            state = {'configured': app.auth.configured(), 'signed_in': signed_in, 'local': False}
            if signed_in:
                state['user'] = app.auth.username()
                state['sessions'] = app.auth.sessions_count()
            return state

        def _public(self, method, path, api):
            """A request without a session: the sign-in page (in place of any page), and the sign-in calls."""
            if method == 'GET' and not api and PAGE_RE.fullmatch(path):
                nonce = secrets.token_urlsafe(18)
                csp = ("default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'unsafe-inline'; "
                       "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                self._send(200 if path in ('/', '/index.html') else 401, 'text/html; charset=utf-8',
                           render_login(nonce).encode('utf-8'), csp)
                return
            if method == 'GET' and path == '/api/auth/state':
                self._json(200, self._auth_state())
                return
            if method == 'POST' and path in PUBLIC_AUTH_ROUTES:
                self._post(path)
                return
            self._problem(401, 'login_required', LOGIN_MESSAGE, api)

        def _auth_post(self, path, body):
            if app.auth is None:
                self._problem(404, 'not_found', 'On 127.0.0.1 the site does not ask you to sign in.', True)
                return
            if not isinstance(body, dict):
                raise ReviewRejected(400, 'invalid_json', 'A JSON object is required.')
            address = self.client_address[0]
            if path == '/api/auth/setup/check':
                app.auth.check_setup_code(body.get('code'), address)
                self._json(200, {'ok': True})
            elif path == '/api/auth/setup':
                token = app.auth.setup(body.get('code'), body.get('user'), body.get('password'), address)
                self._json(200, {'ok': True}, [('Set-Cookie', self._session_cookie(token))])
            elif path == '/api/auth/login':
                token = app.auth.login(body.get('user'), body.get('password'), address)
                print('Sito: amministratore entrato', flush=True)
                self._json(200, {'ok': True}, [('Set-Cookie', self._session_cookie(token))])
            elif path == '/api/auth/password':
                session = self._session()
                if session is None:
                    self._problem(401, 'login_required', LOGIN_MESSAGE, True)
                    return
                token = app.auth.change_password(session, body.get('current'), body.get('password'), address)
                # Every other browser is signed out: this one, with its new session, is the only one.
                state = {'configured': True, 'signed_in': True, 'local': False, 'user': app.auth.username(),
                         'sessions': 1}
                self._json(200, {'ok': True, 'state': state}, [('Set-Cookie', self._session_cookie(token))])
            else:
                app.auth.logout(self._session(), everywhere=path == '/api/auth/logout-all')
                self._json(200, {'ok': True}, [('Set-Cookie', self._expired_cookie())])
        # [site-login] - END

        def _get(self, path):
            # [site-login] - START
            if path == '/api/auth/state':
                self._json(200, self._auth_state())
                return
            if path == '/account':
                nonce = secrets.token_urlsafe(18)
                csp = ("default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'unsafe-inline'; "
                       "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                self._send(200, 'text/html; charset=utf-8', render_account(nonce).encode('utf-8'), csp)
                return
            if path in AUTH_ROUTES:
                self._problem(405, 'method_not_allowed', 'Use POST.', True)
                return
            # [site-login] - END
            if path in ('/routes', '/index.html'):
                nonce = secrets.token_urlsafe(18)
                from nnslr_tools import llm_providers
                # No AI reviewer chosen in Settings: one link to set it up instead of a button on every drive.
                page = render_index(app.route_summaries(), db_path(app.root), nonce,
                                    ai_reviewer=llm_providers.ai_state(app.root)['provider'] is not None)
                self._send(200, 'text/html; charset=utf-8', page.encode('utf-8'),
                           "default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'unsafe-inline'; "
                           "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                return
            if path == '/api/routes':
                self._json(200, {'routes': app.route_summaries()})
                return
            # [control-panel] - START
            # The control pages: one Vue app, a tab each (web/src/control); the old addresses lead there.
            if path in CONTROL_TABS:
                nonce = secrets.token_urlsafe(18)
                # img-src: the frames of the Watch panel of the Evaluate tab (the previews of /route/<id>/images/);
                # media-src: the video of a drive in the Drive view of Results (/route/<id>/video.mp4).
                csp = ("default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'unsafe-inline'; "
                       "img-src 'self'; media-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; "
                       "frame-ancestors 'none'")
                self._send(200, 'text/html; charset=utf-8', render_control(nonce, CONTROL_TABS[path]).encode('utf-8'),
                           csp)
                return
            if path in OLD_ADDRESSES:
                self._redirect(OLD_ADDRESSES[path], None)
                return
            if path == '/api/panel/status':
                self._json(200, app.panel.status())
                return
            # The machine, for the server pill of the pages outside the panel (every 4 s), and what runs: the dots of
            # the busy tabs.
            if path == '/api/panel/system':
                self._json(200, {'system': app.panel.system(), 'running': app.panel.running()})
                return
            if path == '/api/panel/comma':
                self._json(200, {'comma': app.panel.comma()})
                return
            if path == '/api/panel/runs':
                self._json(200, app.panel.runs())
                return
            match = re.fullmatch(r'/api/panel/runs/([A-Za-z0-9][A-Za-z0-9._-]{0,99})', path)
            if match:
                self._json(200, app.panel.run(match.group(1)))
                return
            # The Watch panel: the frames of one route of an evaluation, with the boxes to draw over each.
            match = re.fullmatch(r'/api/panel/runs/([A-Za-z0-9][A-Za-z0-9._-]{0,99})/frames', path)
            if match:
                query = parse_qs(urlsplit(self.path).query)
                value = lambda name: query.get(name, [None])[0]  # noqa: E731
                self._json(200, app.panel.frames(match.group(1), value('model'), value('partition') or 'test',
                                                 value('route') or None, app.frame_rows))
                return
            # The video of a drive for the Drive view of Results (nnslr_tools.drive_video): ready, or being made;
            # ?start=1 begins making it when it is missing.
            match = re.fullmatch(r'/api/panel/drive-video/([0-9a-f]{8}--[0-9a-f]{10})', path)
            if match:
                from nnslr_tools import drive_video
                start = parse_qs(urlsplit(self.path).query).get('start', [''])[0] == '1'
                self._json(200, drive_video.state(app.root, match.group(1), start=start))
                return
            # [drive-review] - START
            # Data › Drive review (nnslr_tools.drive_review): the drives with their logs and scans, the queue of
            # episodes, a drive's timeline.
            if path.startswith('/api/drive-review/'):
                self._drive_review_get(path)
                return
            # [drive-review] - END
            # The networks by role (cutter, classifier, decider) and the one in use, for the Models tab.
            if path == '/api/panel/models':
                self._json(200, app.panel.models())
                return
            # The proposers of Prepare as measured on the test frames, for the Evaluate tab.
            if path == '/api/panel/proposers':
                self._json(200, app.panel.proposers())
                return
            # The road type of every frame of the drives, from the map (nnslr road-types), for Data › Roads.
            if path == '/api/panel/road-types':
                from nnslr_tools import road_types
                self._json(200, road_types.summary(app.root))
                return
            # One drive of Data › Roads opened: every frame with its picture and its road type, the map's or a person's.
            match = re.fullmatch(r'/api/panel/road-types/([^/]+)', path)
            if match:
                from nnslr_tools import road_types
                by_key = self._prepared_drive(match.group(1))
                if by_key is not None:
                    self._json(200, road_types.drive_view(app.root, match.group(1), by_key))
                return
            if re.fullmatch(r'/api/panel/road-types/[^/]+(/junction)?/undo|/api/panel/road-types/[^/]+/junction', path):
                self._problem(405, 'method_not_allowed', 'Use POST.', True)
                return
            if path == '/api/panel/models/active':
                self._problem(405, 'method_not_allowed', 'Use POST, with {"role", "id"}.', True)
                return
            # The settings of the comma's vision limit for Models › On the car (nnslr_tools.car_settings): the ones
            # set here and the ones the published speed_limits.json carries to the car.
            if path == '/api/panel/car-settings':
                self._json(200, self._car_settings_view())
                return
            # The settings of the trainings of Train, for their next run (nnslr_tools.train_settings).
            if path == '/api/panel/train-settings':
                self._json(200, app.panel.train_settings())
                return
            match = re.fullmatch(r'/api/panel/job/([0-9a-z_-]+)/log', path)
            if match:
                offset = parse_qs(urlsplit(self.path).query).get('offset', ['0'])[0]
                self._json(200, app.panel.log(match.group(1), int(offset) if offset.isdigit() else 0))
                return
            # [control-panel] - END
            # [ai-reviewer] - START
            # The AI providers of Settings (never a key: has_key only) and who may change them.
            if path == '/api/settings/ai':
                self._json(200, self._ai_settings())
                return
            # Whose Hugging Face token the portal keeps, never the token.
            if path == '/api/settings/huggingface':
                from nnslr_tools import hf_account
                self._json(200, {**hf_account.public_state(app.root), 'can_edit': self._can_use_ai()})
                return
            # Whether the review page can ask the AI, and why not.
            if path == '/api/ai/state':
                self._json(200, self._ai_state())
                return
            # The providers with the latest measure of each, for the AI reviewers view of the Evaluate tab.
            if path == '/api/panel/ai-reviewers':
                self._json(200, app.panel.ai_reviewers())
                return
            # The newest "Try without saving" of the route, on the frames still to review: no key, nothing saved.
            match = re.fullmatch(r'/api/route/([^/]+)/ai-trial', path)
            if match:
                from nnslr_tools import ai_review
                self._json(200, {'ok': True, 'trial': ai_review.latest_trial(app, match.group(1))})
                return
            if (path in AI_SETTINGS_POSTS or AI_DELETE_RE.fullmatch(path) or path == HF_POSTS[1]
                    or re.fullmatch(r'/api/route/[^/]+/frame/[^/]+/[^/]+/ai-(?:suggest|fit)', path)):
                self._problem(405, 'method_not_allowed', 'Use POST.', True)
                return
            # [ai-reviewer] - END
            match = re.fullmatch(r'/api/route/([^/]+)/reviews', path)
            if match:
                self._send_reviews(match.group(1))
                return
            # While Review with AI works on a drive, its page asks every few seconds: how far the AI is, and the frames
            # saved since the newest saved time the page has seen (?since=, an updated_at).
            match = re.fullmatch(r'/api/route/([^/]+)/live', path)
            if match:
                self._send_live(match.group(1), parse_qs(urlsplit(self.path).query).get('since', [None])[0])
                return
            # Is the drive read only now (Propose again works on it)? The review page asks while it is. With
            # /<after>, the drive at work also sends its new proposals after the first <after> frames (run: theirs).
            match = re.fullmatch(r'/api/route/([^/]+)/busy(?:/([0-9]{1,7}))?', path)
            if match:
                route = match.group(1)
                try:
                    parse_route_id(route)
                    _directory, rows, by_key = app.report(route)
                    total = len(rows)
                except (PreannotationError, ValueError, OSError):
                    self._problem(404, 'unknown_route', UNKNOWN_DRIVE_MESSAGE, True)
                    return
                from nnslr_tools import reproposing
                item = reproposing.of(app.root, route)
                fresh = None
                if match.group(2) is not None and item is not None and item['phase'] == 'proposing again':
                    every = reproposing.at_work_rows(item)
                    # Frames written since the count: the page has them, so does its count.
                    item = dict(item, done=max(item['done'], len(every)))
                    fresh = every[int(match.group(2)):]
                answer = {'ok': True, 'busy': reproposing.public(item, total)}
                if fresh is not None:
                    answer.update(run_id=item['pending'].name, rows=located(fresh, by_key))
                self._json(200, answer)
                return
            if re.fullmatch(r'/api/route/[^/]+/frame/[^/]+/[^/]+', path):
                self._problem(405, 'method_not_allowed', 'Use POST to save a frame.', True)
                return
            match = re.fullmatch(r'/route/([^/]+)', path)
            if match:
                route = self._route_or_none(match.group(1))
                if route is not None:
                    self._redirect('/route/' + quote(route) + '/', None)
                return
            match = re.fullmatch(r'/route/([^/]+)/(index\.html)?', path)
            if match:
                self._send_route_page(match.group(1))
                return
            match = re.fullmatch(r'/route/([^/]+)(/images/[0-9]{8}\.jpg)', path)
            if match:
                self._send_image(match.group(1), match.group(2))
                return
            match = re.fullmatch(r'/route/([0-9a-f]{8}--[0-9a-f]{10})/video\.mp4', path)
            if match:
                self._send_video(match.group(1))
                return
            self._problem(404, 'not_found', 'Page not found.', path.startswith('/api/'))

        # ---------- REST API /api/v1 and MCP /mcp (token in a header, never a cookie) ----------

        def _external(self, method, path, query):
            if 'token' in query:
                self._problem(400, 'token_in_url', API_TOKEN_MESSAGE, True)
                return
            # MCP transport rule: an Origin header that is not this server is refused (DNS rebinding).
            if not self._same_origin():
                self._problem(403, 'cross_origin', 'Request from another origin refused.', True)
                return
            if not self._api_authorized():
                self.close_connection = True
                body = json.dumps({'ok': False, 'error': 'unauthorized', 'message': API_TOKEN_MESSAGE},
                                  ensure_ascii=False).encode('utf-8')
                self._send(401, 'application/json; charset=utf-8', body, DEFAULT_CSP,
                           [('WWW-Authenticate', 'Bearer realm="nnslr-review"')])
                return
            if path == '/mcp':
                self._mcp(method)
                return
            if method == 'POST':
                self._api_post(path)
            elif method == 'GET':
                self._api_get(path, query)
            else:
                self._problem(405, 'method_not_allowed', 'Method not allowed.', True)

        def _api_authorized(self):
            supplied = self.headers.get('Authorization')
            if isinstance(supplied, str) and supplied[:7].lower() == 'bearer ':
                return app.api_token_matches(supplied[7:].strip())
            return app.api_token_matches(self.headers.get('X-Review-Token'))

        def _api_get(self, path, query):
            values = {}
            for name, items in query.items():
                values[name] = items[0]
            if path in ('/api/v1', '/api/v1/'):
                self._json(200, {'ok': True, 'api': 'nnslr-review', 'version': 1,
                                 'mcp': app.mcp_bridge() is not None, 'documentation': 'docs/API_MCP.md'})
                return
            if path == '/api/v1/schema':
                self._json(200, review_schema())
                return
            if path == '/api/v1/routes':
                self._json(200, {'routes': app.list_routes()})
                return
            match = re.fullmatch(r'/api/v1/routes/([^/]+)', path)
            if match:
                self._json(200, app.progress(match.group(1), suspicious_limit=values.get('suspicious_limit', 100)))
                return
            match = re.fullmatch(r'/api/v1/routes/([^/]+)/frames', path)
            if match:
                self._json(200, app.list_frames(match.group(1), frame_filter=values.get('filter'),
                                                cursor=values.get('cursor'), limit=values.get('limit')))
                return
            match = re.fullmatch(r'/api/v1/routes/([^/]+)/next', path)
            if match:
                self._json(200, app.next_frame(match.group(1), frame_filter=values.get('filter'),
                                               after=values.get('after')))
                return
            match = re.fullmatch(r'/api/v1/routes/([^/]+)/frames/([0-9]{1,9})/([0-9]{1,9})(/[a-z-]+)?', path)
            if match is None:
                self._problem(404, 'not_found', 'Invalid API address.', True)
                return
            route = match.group(1)
            key = match.group(2) + '/' + match.group(3)
            action = match.group(4)
            if action is None:
                self._json(200, app.get_frame(route, key))
            elif action == '/context':
                self._json(200, app.frame_context(route, key, before=values.get('before'), after=values.get('after')))
            elif action == '/history':
                self._json(200, app.history(route, key))
            elif action == '/image':
                image = app.image(route, key, variant=values.get('variant'), crop=values.get('crop'),
                                  long_side=values.get('long_side'))
                self._send(200, image['mime_type'], image['data'])
            elif action == '/review':
                self._problem(405, 'method_not_allowed', 'Use POST to save a frame.', True)
            elif action == '/apply-previous':
                self._problem(405, 'method_not_allowed', 'Use POST, with the boxes of this frame or {}.', True)
            else:
                self._problem(404, 'not_found', 'Invalid API address.', True)

        def _api_post(self, path):
            match = re.fullmatch(r'/api/v1/routes/([^/]+)/frames/([0-9]{1,9})/([0-9]{1,9})/(review|apply-previous)',
                                 path)
            if match is None:
                self._problem(404, 'not_found', 'Invalid API address.', True)
                return
            content_type = self.headers.get('Content-Type', '')
            if content_type.split(';')[0].strip().lower() != 'application/json':
                self._problem(415, 'unsupported_media_type', 'Only JSON data is accepted.', True)
                return
            body = self._read_json()
            key = match.group(2) + '/' + match.group(3)
            if match.group(4) == 'apply-previous':
                # Saves nothing: the boxes of the previous frame carried onto this one, to check and save.
                self._json(200, app.apply_previous(match.group(1), key, body))
                return
            self._json(200, app.save_review(match.group(1), key, body, client=self.client_address[0],
                                            channel='rest'))

        def _mcp(self, method):
            bridge = app.mcp_bridge()
            if bridge is None:
                self._problem(501, 'mcp_unavailable', 'MCP server not available: install the optional '
                                                      'dependencies with  pip install -e ".[mcp]"  and restart the server.',
                              True)
                return
            if method != 'POST':
                # Stateless JSON server: no server stream (GET) and no session to end (DELETE). The MCP
                # transport lets a server answer both with 405.
                body = json.dumps({'ok': False, 'error': 'method_not_allowed',
                                   'message': 'This MCP server only accepts POST.'}).encode('utf-8')
                self._send(405, 'application/json; charset=utf-8', body, DEFAULT_CSP, [('Allow', 'POST')])
                return
            if self.headers.get('Transfer-Encoding') is not None:
                raise ReviewRejected(411, 'length_required', 'Request without Content-Length.')
            raw_length = self.headers.get('Content-Length')
            if raw_length is None or not raw_length.strip().isdigit():
                raise ReviewRejected(411, 'length_required', 'Request without Content-Length.')
            length = int(raw_length)
            if length > MAX_BODY_BYTES:
                self.close_connection = True
                raise ReviewRejected(413, 'too_large', 'Request too large (at most 512 KiB).')
            body = self.rfile.read(length)
            headers = []
            for name, value in self.headers.items():
                # The token stays in this process: the MCP layer never sees it.
                if name.lower() in ('authorization', 'x-review-token', 'cookie'):
                    continue
                headers.append((name, value))
            status, response_headers, data = bridge.handle(method, headers, body, self.client_address,
                                                           self.server.server_address)
            extra = []
            content_type = 'application/json'
            for name, value in response_headers:
                lowered = name.lower()
                if lowered == 'content-type':
                    content_type = value
                elif lowered in ('mcp-session-id', 'mcp-protocol-version', 'allow'):
                    extra.append((name, value))
            self._send(status, content_type, data, DEFAULT_CSP, extra)

        def _route_or_none(self, route):
            # Unknown routes, malformed ids and paths outside the data root all end as 404.
            try:
                parse_route_id(route)
                app.report(route)
            except (PreannotationError, ValueError, OSError):
                self._problem(404, 'unknown_route', UNKNOWN_DRIVE_MESSAGE, False)
                return None
            return route

        def _prepared_drive(self, route):
            """The rows of a prepared drive's report by frame key; None (answered 404) for a drive that is not prepared,
            a malformed id or a path outside the data root."""
            try:
                parse_route_id(route)
                return app.report(route)[2]
            except (PreannotationError, ValueError, OSError):
                self._problem(404, 'unknown_route', UNKNOWN_DRIVE_MESSAGE, True)
                return None

        def _road_type_post(self, route, undo, body):
            """Data › Roads: a person's road type for the frames from/to of a drive ({"from", "to", "road_type"}, null
            when they cannot tell), or Undo of the last one; the drive again. A read-only panel refuses it, as it refuses
            the actions."""
            from nnslr_tools import road_types
            if not app.panel.actions_enabled:
                raise PanelError(403, 'actions_disabled', 'The panel is read-only on this server.')
            if not isinstance(body, dict):
                raise ReviewRejected(400, 'invalid_json', 'A JSON object is required.')
            by_key = self._prepared_drive(route)
            if by_key is None:
                return
            who, via = self._who()
            if undo:
                if not road_types.undo_correction(app.root, route):
                    raise ReviewRejected(409, 'nothing_to_undo', 'This drive has no correction to undo.')
                print(f'Roads: {route} last correction undone ({terminal_text(who or via)})', flush=True)
            else:
                if set(body) != {'from', 'to', 'road_type'}:
                    raise ReviewRejected(400, 'invalid_correction', 'Send {"from", "to", "road_type"}: two frame keys '
                                                                    'of the drive and the type (null: cannot tell).')
                try:
                    line = road_types.correct(app.root, route, body['from'], body['to'], body['road_type'], who or via)
                except road_types.CorrectionError as exc:
                    raise ReviewRejected(400, 'invalid_correction', str(exc)) from None
                print(f'Roads: {route} {line["from"]} to {line["to"]} is {line["road_type"] or "cannot tell"} '
                      f'({terminal_text(who or via)})', flush=True)
            self._json(200, {'ok': True, **road_types.drive_view(app.root, route, by_key)})

        def _car_settings_view(self):
            from nnslr_tools import car_release, car_settings
            return car_settings.view(app.root, car_release.published_settings())

        def _car_settings_post(self, body):
            """Models › On the car: {"values": {setting: number or null}} saved for the next Publish to the car; a
            read-only panel refuses it, as it refuses the actions."""
            from nnslr_tools import car_settings
            if not app.panel.actions_enabled:
                raise PanelError(403, 'actions_disabled', 'The panel is read-only on this server.')
            who, via = self._who()
            try:
                record = car_settings.save(app.root, body, who or via)
            except car_settings.SettingsError as exc:
                raise ReviewRejected(400, 'invalid_settings', str(exc)) from None
            print(f'Panel: car settings {record["values"] or "all default"} ({terminal_text(who or via)})', flush=True)
            self._json(200, {'ok': True, **self._car_settings_view()})

        def _junction_post(self, route, undo, body):
            """Data › Roads: a person's "Junction here" ({"frame_key", "junction": "here"}) or "Not a junction"
            ("none") for a frame of a drive, or Undo of the last one; the drive again. Kept apart from the map's
            junctions (annotations/junctions/<route>.jsonl), as the road types are; a read-only panel refuses it."""
            from nnslr_tools import junctions, road_types
            if not app.panel.actions_enabled:
                raise PanelError(403, 'actions_disabled', 'The panel is read-only on this server.')
            if not isinstance(body, dict):
                raise ReviewRejected(400, 'invalid_json', 'A JSON object is required.')
            by_key = self._prepared_drive(route)
            if by_key is None:
                return
            who, via = self._who()
            if undo:
                if not junctions.undo_correction(app.root, route):
                    raise ReviewRejected(409, 'nothing_to_undo', 'This drive has no junction correction to undo.')
                print(f'Roads: {route} last junction correction undone ({terminal_text(who or via)})', flush=True)
            else:
                if set(body) != {'frame_key', 'junction'}:
                    raise ReviewRejected(400, 'invalid_correction', 'Send {"frame_key", "junction"}: a frame key of '
                                                                    'the drive and "here" or "none".')
                try:
                    line = junctions.correct(app.root, route, body['frame_key'], body['junction'], who or via)
                except junctions.CorrectionError as exc:
                    raise ReviewRejected(400, 'invalid_correction', str(exc)) from None
                print(f'Roads: {route} {line["frame_key"]} is '
                      f'{"a junction" if line["junction"] == "here" else "not a junction"} ({terminal_text(who or via)})',
                      flush=True)
            self._json(200, {'ok': True, **road_types.drive_view(app.root, route, by_key)})

        def _send_route_page(self, route):
            try:
                directory, run_id, rows, busy, proposer = app.view(route)
            except (PreannotationError, ValueError, OSError):
                self._problem(404, 'unknown_route', UNKNOWN_DRIVE_MESSAGE, False)
                return
            hints_by_frame, hint_stats = app.hints(route, rows)
            suspicious = app.suspicious(route, run_id, rows, hints_by_frame)
            nonce = secrets.token_urlsafe(18)
            page = render_review(rows, run_id, route_id=route, server=True, nonce=nonce,
                                 hints=hints_by_frame, hint_stats=hint_stats, suspicious=suspicious,
                                 routes=app.route_links(), proposer=proposer, busy=busy)
            csp = ("default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'unsafe-inline'; "
                   "img-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            self._send(200, 'text/html; charset=utf-8', page.encode('utf-8'), csp)

        def _send_image(self, route, name):
            try:
                directory = app.view(route)[0]
            except (PreannotationError, ValueError, OSError):
                self._problem(404, 'unknown_route', 'Unknown route, or route without a pre-annotation report.', False)
                return
            path = public_file(directory, name)
            if path is None:
                self._problem(404, 'not_found', 'Image not found.', False)
                return
            self.send_response(200)
            self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(path.stat().st_size))
            self._security_headers(DEFAULT_CSP)
            self.end_headers()
            with path.open('rb') as source:
                while block := source.read(65536):
                    self.wfile.write(block)

        def _send_video(self, route):
            """The video of a drive (nnslr_tools.drive_video), with byte ranges: the player seeks in it."""
            from nnslr_tools import drive_video
            path = drive_video.paths(app.root, route)[0]
            try:
                size = path.stat().st_size
            except OSError:
                self._problem(404, 'not_found', 'This drive has no video yet.', False)
                return
            first, last, status = 0, size - 1, 200
            wanted = re.fullmatch(r'bytes=([0-9]*)-([0-9]*)', self.headers.get('Range', '').strip())
            if wanted and (wanted.group(1) or wanted.group(2)):
                if wanted.group(1):
                    first = int(wanted.group(1))
                    last = min(int(wanted.group(2)), size - 1) if wanted.group(2) else size - 1
                else:
                    first = max(0, size - int(wanted.group(2)))
                if first > last or first >= size:
                    self._send(416, 'text/plain', b'', DEFAULT_CSP, [('Content-Range', f'bytes */{size}')])
                    return
                status = 206
            self.send_response(status)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Content-Length', str(last - first + 1))
            self.send_header('Accept-Ranges', 'bytes')
            if status == 206:
                self.send_header('Content-Range', f'bytes {first}-{last}/{size}')
            # Kept by this browser only (private) for a day: its address changes when the video is made again.
            self._security_headers(DEFAULT_CSP, 'private, max-age=86400')
            self.end_headers()
            try:
                with path.open('rb') as source:
                    source.seek(first)
                    left = last - first + 1
                    while left > 0 and (block := source.read(min(1 << 18, left))):
                        self.wfile.write(block)
                        left -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                # The player asked for another range and closed this one.
                pass

        # [drive-review] - START
        def _drive_review_get(self, path):
            from nnslr_tools import drive_review, drive_video
            from nnslr_tools.drive_logs import DriveLogError
            store = drive_review.store_for(app.root)
            query = parse_qs(urlsplit(self.path).query)
            value = lambda name: query.get(name, [''])[0] or None  # noqa: E731
            try:
                if path == '/api/drive-review/drives':
                    self._json(200, {'drives': drive_review.drives_view(store), 'review_from': store.review_from()})
                    return
                if path == '/api/drive-review/episodes':
                    route, kind, data = value('route'), value('kind'), value('data')
                    if ((route and not re.fullmatch(r'[0-9a-f]{8}--[0-9a-f]{10}', route))
                            or (kind and kind not in drive_review.QUEUE_KINDS)
                            or (data and data not in drive_review.QUEUE_DATA)):
                        self._problem(400, 'bad_filter', 'Unknown filter.', True)
                        return
                    self._json(200, {'episodes': drive_review.queue_view(store, route=route, kind=kind, data=data)})
                    return
                match = re.fullmatch(r'/api/drive-review/drives/([0-9a-f]{8}--[0-9a-f]{10})(/timeline)?', path)
                if match and match.group(2):
                    route = match.group(1)
                    video = drive_video.state(app.root, route)
                    view = drive_review.timeline(store, route, video)
                    drive = drive_of(app.root, route)
                    when, certain = drive_review._when(view.get('civil_ms'), drive)
                    view.update(video=video, title=drive.get('title'), when=when, when_certain=certain,
                                inference=drive_review.inference_lane(app.root, route),
                                map_today=drive_review.map_today_lane(app.root, route),
                                sign_review=any(name == route for name, _directory in list_reports(app.root)))
                    self._json(200, view)
                    return
                if match:
                    self._json(200, {'scan': store.scan_state(match.group(1)), 'files': store.log_files(match.group(1))})
                    return
            except (drive_review.DriveReviewError, DriveLogError) as exc:
                self._problem(exc.status, exc.reason, exc.message, True)
                return
            self._problem(404, 'not_found', 'Not found.', True)

        def _drive_review_post(self, path):
            """The drive review's actions (logs to read and scan, the first day shown; nothing is judged here): same
            origin and a session, as every change of the portal. A log is sent as it is (application/octet-stream,
            ?name=<its path or name>&segment=N); everything else is JSON."""
            from nnslr_tools import drive_logs, drive_review
            store = drive_review.store_for(app.root)
            if not self._same_origin():
                self._problem(403, 'cross_origin', 'Request from another origin refused.', True)
                return
            content_type = self.headers.get('Content-Type', '').split(';')[0].strip().lower()
            upload = re.fullmatch(r'/api/drive-review/drives/([0-9a-f]{8}--[0-9a-f]{10})/logs', path)
            try:
                if upload:
                    if content_type != 'application/octet-stream':
                        self._problem(415, 'unsupported_media_type', 'Send the log file as it is.', True)
                        return
                    self._require_ai_session()
                    raw_length = self.headers.get('Content-Length', '')
                    if self.headers.get('Transfer-Encoding') is not None or not raw_length.strip().isdigit():
                        self._problem(411, 'length_required', 'Request without Content-Length.', True)
                        return
                    query = parse_qs(urlsplit(self.path).query)
                    segment, kind, ext = drive_logs.parse_upload_name(upload.group(1), query.get('name', [''])[0],
                                                                      query.get('segment', [None])[0])
                    if int(raw_length) > drive_logs.MAX_UPLOAD_BYTES:
                        self.close_connection = True
                    added = store.add_upload(upload.group(1), segment, kind, ext, self.rfile, int(raw_length))
                    self._json(200, {'ok': True, **added, 'scan': store.scan_state(upload.group(1))})
                    return
                if content_type != 'application/json':
                    self._problem(415, 'unsupported_media_type', 'Only JSON data is accepted.', True)
                    return
                self._require_ai_session()
                body = self._read_json()
                who, via = self._who()
                author = who or 'administrator'
                match = re.fullmatch(r'/api/drive-review/drives/([0-9a-f]{8}--[0-9a-f]{10})/scan', path)
                if match:
                    self._json(200, {'ok': True, 'scan': store.queue_scan(match.group(1))})
                    return
                if path == '/api/drive-review/scan-all':
                    queued = []
                    early = drive_review.early_routes(store)
                    for route in drive_review.all_routes(app.root):
                        if route in early:
                            continue
                        state = store.scan_state(route)['state']
                        if state in ('not_scanned', 'outdated') or (
                                state == 'no_logs' and drive_logs.discover(app.root, route)):
                            store.queue_scan(route)
                            queued.append(route)
                    self._json(200, {'ok': True, 'queued': queued})
                    return
                if not isinstance(body, dict):
                    self._problem(400, 'invalid_json', 'An object is expected.', True)
                    return
                if path == '/api/drive-review/settings':
                    review_from = store.set_review_from(body.get('review_from'))
                    print(f'Drive review: drives reviewed from {review_from or "the first"} ({terminal_text(author)})',
                          flush=True)
                    self._json(200, {'ok': True, 'review_from': review_from})
                    return
            except (drive_review.DriveReviewError, drive_logs.DriveLogError) as exc:
                self._problem(exc.status, exc.reason, exc.message, True)
                return
            self._problem(404, 'not_found', 'Not found.', True)
        # [drive-review] - END

        def _send_reviews(self, route):
            try:
                directory, run_id, rows, busy, proposer = app.view(route)
            except (PreannotationError, ValueError, OSError):
                self._problem(404, 'unknown_route', 'Unknown route, or route without a pre-annotation report.', True)
                return
            reviews = []
            confirmed = 0
            for review in app.store.reviews(route, run_id):
                reviews.append(public_review(review))
                if review['confirmed']:
                    confirmed = confirmed + 1
            hints_by_frame, hint_stats = app.hints(route, rows)
            suspicious = app.suspicious(route, run_id, rows, hints_by_frame)
            self._json(200, {'route_id': route, 'run_id': run_id, 'frame_count': len(rows),
                             'confirmed_count': confirmed, 'reviews': reviews, 'suspicious': suspicious,
                             'busy': busy})

        def _send_live(self, route, since):
            try:
                directory, run_id, rows, busy, proposer = app.view(route)
            except (PreannotationError, ValueError, OSError):
                self._problem(404, 'unknown_route', 'Unknown route, or route without a pre-annotation report.', True)
                return
            if since is not None and not re.fullmatch(r'[0-9T:.+Z-]{10,40}', since):
                self._problem(400, 'invalid_since', 'since must be the updated_at of a review.', True)
                return
            live = app.panel.ai_live()
            reviews = [public_review(review) for review in app.store.reviews(route, run_id, since=since)]
            self._json(200, {'route_id': route, 'run_id': run_id, 'reviews': reviews,
                             'ai': live if live is not None and live['route'] == route else None})

        def _post(self, path):
            # [site-login] - START
            if path in AUTH_ROUTES:
                # Same protections as saving a frame: same origin, JSON only (no form can send it).
                if not self._same_origin():
                    self._problem(403, 'cross_origin', 'Request from another origin refused.', True)
                    return
                content_type = self.headers.get('Content-Type', '')
                if content_type.split(';')[0].strip().lower() != 'application/json':
                    self._problem(415, 'unsupported_media_type', 'Only JSON data is accepted.', True)
                    return
                self._auth_post(path, self._read_json())
                return
            # [site-login] - END
            if path == '/api/panel/comma/test':
                # The comma checked now: a TCP connection and an ssh login running true; a session, as its address.
                if self._ai_post_refused():
                    return
                self._json(200, {'comma': app.panel.test_comma()})
                return
            if path == '/api/panel/comma':
                # The address of the comma: same protections as the AI settings (same origin, JSON, a session).
                if self._ai_post_refused():
                    return
                try:
                    self._json(200, {'comma': app.panel.set_comma_address(self._read_json())})
                except PanelError as error:
                    self._problem(error.status, error.reason, error.message, True)
                return
            # [ai-reviewer] - START
            if path in AI_SETTINGS_POSTS or AI_DELETE_RE.fullmatch(path):
                if self._ai_post_refused():
                    return
                self._ai_settings_post(path, self._read_json())
                return
            if path in HF_POSTS:
                # The same protections as the AI settings: same origin, JSON, a signed-in browser.
                if self._ai_post_refused():
                    return
                self._hf_post(path, self._read_json())
                return
            # [ai-reviewer] - END
            # [drive-review] - START
            if path.startswith('/api/drive-review/'):
                self._drive_review_post(path)
                return
            # [drive-review] - END
            # [control-panel] - START
            run = re.fullmatch(r'/api/panel/run/([a-z_]+)', path)
            stop = re.fullmatch(r'/api/panel/job/([0-9a-z_-]+)/stop', path)
            active = path == '/api/panel/models/active'
            # Who proposes the signs in Prepare ({"proposer"}), kept for every browser.
            chosen = path == '/api/panel/proposer'
            # A person's road type for a stretch of a drive of Data › Roads ({"from", "to", "road_type"}), or /undo;
            # /junction: a person's "Junction here" or "Not a junction" for a frame ({"frame_key", "junction"}), or
            # /junction/undo.
            road = re.fullmatch(r'/api/panel/road-types/([^/]+)(/junction)?(/undo)?', path)
            # The settings of the comma's vision limit ({"values": {setting: number or null}}).
            settings = path == '/api/panel/car-settings'
            # The settings of a training of Train ({"training": action, "values": {setting: number or null}}).
            trainings = path == '/api/panel/train-settings'
            # Train › Pipeline: our networks one after the other ({"steps": [{"action", "options"}]}), or /stop: the
            # next ones do not start.
            pipeline = re.fullmatch(r'/api/panel/pipeline(/stop)?', path)
            if run or stop or active or chosen or road or settings or trainings or pipeline:
                # Same protections as saving a frame: same origin, JSON only (no form can send it).
                if not self._same_origin():
                    self._problem(403, 'cross_origin', 'Request from another origin refused.', True)
                    return
                content_type = self.headers.get('Content-Type', '')
                if content_type.split(';')[0].strip().lower() != 'application/json':
                    self._problem(415, 'unsupported_media_type', 'Only JSON data is accepted.', True)
                    return
                # [ai-reviewer] - START
                # An AI review saves final reviews and both AI actions spend the provider's time: a session only.
                if run and run.group(1) in ('ai_review', 'measure_ai_reviewer'):
                    self._require_ai_session()
                # [ai-reviewer] - END
                body = self._read_json()
                if run:
                    # The options: where a training starts ({"start": key}), the run to resume ({"run": name}), the
                    # cutter's size and signs ({"size", "families"}), the SignNet sent to the car ({"signnet": run}),
                    # the classifier's Mapillary signs ({"mapillary": "on", "region": "europe", "share": "5"}), the
                    # route, provider, save and limit of an AI action, all checked by Panel.start.
                    options = {key: body[key] for key in ('start', 'run', 'size', 'families', 'proposer', 'package',
                                                          'signnet', 'mapillary', 'region', 'share', 'cutter')
                               if isinstance(body, dict) and isinstance(body.get(key), str)}
                    # [ai-reviewer] - START
                    # Passed as they are, also when not strings: Panel.start refuses them, never drops them (a limit
                    # sent as a number must not become a whole route).
                    options.update((key, body[key]) for key in ('route', 'provider', 'save', 'limit')
                                   if isinstance(body, dict) and body.get(key) is not None)
                    options = options or None
                    # [ai-reviewer] - END
                    job = app.panel.start(run.group(1), options)
                    print(f'Pannello: avviata {job["label"]} ({job["id"]})', flush=True)
                    self._json(200, {'ok': True, **job})
                elif active:
                    # The model in use for a role: only an id of the index is written, never a path.
                    who, via = self._who()
                    entry = app.panel.set_active_model(body, who=who, via=via)
                    role = body['role']
                    print(f'Panel: {role} in use {entry["id"]} ({terminal_text(who or via)})', flush=True)
                    self._json(200, {'ok': True, 'role': role, 'active': entry})
                elif chosen:
                    self._json(200, {'ok': True, **app.panel.set_proposer(body)})
                elif settings:
                    self._car_settings_post(body)
                elif trainings:
                    who, via = self._who()
                    self._json(200, {'ok': True, **app.panel.save_train_settings(body, who or via)})
                elif pipeline and pipeline.group(1):
                    self._json(200, {'ok': True, 'pipeline': app.panel.stop_pipeline()})
                elif pipeline:
                    who, via = self._who()
                    self._json(200, {'ok': True, 'pipeline': app.panel.start_pipeline(body, who or via)})
                elif road and road.group(2):
                    self._junction_post(road.group(1), road.group(3) is not None, body)
                elif road:
                    self._road_type_post(road.group(1), road.group(3) is not None, body)
                else:
                    self._json(200, {'ok': True, **app.panel.stop(stop.group(1))})
                return
            # [control-panel] - END
            match = re.fullmatch(
                r'/api/route/([^/]+)/frame/([0-9]{1,9})/([0-9]{1,9})(/apply-previous|/ai-suggest|/ai-fit|/fit)?', path)
            if match is None:
                self._problem(404, 'not_found', 'Invalid save address.', True)
                return
            if not self._same_origin():
                self._problem(403, 'cross_origin', 'Request from another origin refused.', True)
                return
            content_type = self.headers.get('Content-Type', '')
            if content_type.split(';')[0].strip().lower() != 'application/json':
                self._problem(415, 'unsupported_media_type', 'Only JSON data is accepted.', True)
                return
            # [ai-reviewer] - START
            if match.group(4) in ('/ai-suggest', '/ai-fit'):
                self._require_ai_session()
            # [ai-reviewer] - END
            # [box-fit] - START
            if match.group(4) == '/fit' and not self._can_use_ai():
                raise ReviewRejected(403, 'session_required', FIT_SESSION_MESSAGE)
            # [box-fit] - END
            body = self._read_json()
            route = match.group(1)
            frame_key = str(int(match.group(2))) + '/' + str(int(match.group(3)))
            # [ai-reviewer] - START
            if match.group(4) == '/ai-suggest':
                # The AI's suggestion for this frame: nothing is saved, the page shows it to apply and confirm.
                self._ai_suggest(route, frame_key, body)
                return
            if match.group(4) == '/ai-fit':
                # The rough box of the reviewer fitted by the AI: nothing is saved, the page moves the box.
                self._ai_fit(route, frame_key, body)
                return
            # [ai-reviewer] - END
            # [box-fit] - START
            if match.group(4) == '/fit':
                # The reviewer's boxes fitted on this server (Fit box): nothing is saved, the page moves the boxes.
                self._json(200, self._local_fit(route, frame_key, body))
                return
            # [box-fit] - END
            if match.group(4):
                # "Apply previous" of the page: nothing is saved, the page shows the boxes to check.
                self._json(200, app.apply_previous(route, frame_key, body))
                return
            try:
                directory, rows, by_key = app.report(route)
            except (PreannotationError, ValueError, OSError):
                self._problem(404, 'unknown_route', 'Unknown route, or route without a pre-annotation report.', True)
                return
            report_row = by_key.get(frame_key)
            if report_row is None:
                self._problem(404, 'unknown_frame', 'Frame not in the report of this route.', True)
                return
            review = save_frame_review(app.store, app.root, route_id=route, run_id=directory.name,
                                       report_row=report_row, body=body, client=self.client_address[0])
            reviewer = terminal_text(review['reviewer'])
            if review['confirmed']:
                print(f'Salvato: {route} fotogramma {frame_key}, revisione {review["revision"]} '
                      f'({reviewer})', flush=True)
            else:
                print(f'Conferma annullata: {route} fotogramma {frame_key} ({reviewer})', flush=True)
            # A confirmation can change which neighbours are suspicious (rule d): send the new flags.
            hints_by_frame, hint_stats = app.hints(route, rows)
            suspicious = app.suspicious(route, directory.name, rows, hints_by_frame)
            self._json(200, {'ok': True, 'review': public_review(review),
                             'confirmed_count': app.store.confirmed_count(route, directory.name),
                             'suspicious': suspicious})

        # [ai-reviewer] - START
        def _can_use_ai(self):
            """A browser signed in as the administrator, or the site on 127.0.0.1 (nobody signs in there); never the
            access token alone."""
            return self._who()[1] in ('sign-in', 'local')

        def _require_ai_session(self):
            if not self._can_use_ai():
                raise ReviewRejected(403, 'session_required', SESSION_REQUIRED_MESSAGE)

        def _ai_post_refused(self):
            """The protections of every AI POST: same origin, JSON only, a session. True when refused (answered)."""
            if not self._same_origin():
                self._problem(403, 'cross_origin', 'Request from another origin refused.', True)
                return True
            content_type = self.headers.get('Content-Type', '')
            if content_type.split(';')[0].strip().lower() != 'application/json':
                self._problem(415, 'unsupported_media_type', 'Only JSON data is accepted.', True)
                return True
            self._require_ai_session()
            return False

        def _ai_settings(self):
            from nnslr_tools import llm_providers
            return {**llm_providers.public_settings(app.root), 'can_edit': self._can_use_ai()}

        def _ai_state(self):
            """llm_providers.ai_state with can_ask; reason is the first thing that stops the page asking, or ''."""
            from nnslr_tools import llm_providers
            state = dict(llm_providers.ai_state(app.root))
            state['can_ask'] = self._can_use_ai()
            if not state.get('available'):
                state['reason'] = state.get('reason') or 'Choose an AI reviewer in Settings.'
            elif not state['can_ask']:
                state['reason'] = SESSION_REQUIRED_MESSAGE
            else:
                state['reason'] = ''
            return state

        @staticmethod
        def _stored_key(provider_id):
            """The key saved for a provider, only to mask it in an error message; None when there is none."""
            from nnslr_tools import llm_providers
            if not isinstance(provider_id, str):
                return None
            try:
                return llm_providers.provider(app.root, provider_id).get('key')
            except ReviewRejected:
                return None

        @staticmethod
        def _masked(exc, *secrets):
            """The same refusal with every copy of a key replaced by ***: a key never goes back to the browser."""
            message = str(exc.message)
            for secret in secrets:
                if isinstance(secret, str) and secret:
                    message = message.replace(secret, '***')
            return ReviewRejected(exc.status, exc.reason, message)

        def _ai_settings_post(self, path, body):
            """Settings > AI reviewer: save or delete a provider, choose the reviewer, list a server's models, Test."""
            from nnslr_tools import ai_review, llm_providers
            if not isinstance(body, dict):
                raise ReviewRejected(400, 'invalid_json', 'A JSON object is required.')
            who = terminal_text(self._who()[0] or self._who()[1])
            typed = body.get('key')
            deleted = AI_DELETE_RE.fullmatch(path)
            try:
                if deleted:
                    llm_providers.delete_provider(app.root, deleted.group(1))
                    print(f'AI settings: provider {deleted.group(1)} removed ({who})', flush=True)
                    self._json(200, {'ok': True, **self._ai_settings()})
                elif path == '/api/settings/ai/providers':
                    provider = llm_providers.save_provider(app.root, body)
                    print(f'AI settings: provider {terminal_text(provider["id"])} saved ({who})', flush=True)
                    self._json(200, {'ok': True, 'provider': provider, **self._ai_settings()})
                elif path == '/api/settings/ai/reviewer':
                    chosen = body.get('id')
                    if chosen is not None and not isinstance(chosen, str):
                        raise ReviewRejected(400, 'invalid_provider', 'id is the id of a provider, or null.')
                    llm_providers.set_reviewer(app.root, chosen)
                    print(f'AI settings: reviewer {terminal_text(chosen or "none")} ({who})', flush=True)
                    self._json(200, {'ok': True, **self._ai_settings()})
                elif path == '/api/settings/ai/models':
                    models = llm_providers.list_models(app.root, body)
                    self._json(200, {'ok': True, 'models': models})
                else:
                    chosen = body.get('id')
                    if not isinstance(chosen, str):
                        raise ReviewRejected(400, 'invalid_provider', 'id is the id of a provider.')
                    llm_providers.provider(app.root, chosen)
                    if not app.ai_slots.acquire(blocking=False):
                        raise ReviewRejected(429, 'ai_busy', 'The AI is already answering two frames: '
                                                             'try again in a moment.')
                    try:
                        result = ai_review.test_provider(app, chosen)
                    finally:
                        app.ai_slots.release()
                    print(f'AI settings: test of {terminal_text(chosen)}: {terminal_text(result.get("message", ""))}',
                          flush=True)
                    provider = next(item for item in llm_providers.public_settings(app.root)['providers']
                                    if item['id'] == chosen)
                    # The whole list too: a Test that passes may have made its provider the reviewer.
                    self._json(200, {'ok': True, 'result': result, 'provider': provider, **self._ai_settings()})
            except ReviewRejected as exc:
                raise self._masked(exc, typed, self._stored_key(body.get('id'))) from None

        def _hf_post(self, path, body):
            """Settings > Hugging Face: keep a token once Hugging Face says it can write, or forget it."""
            from nnslr_tools import hf_account
            if not isinstance(body, dict):
                raise ReviewRejected(400, 'invalid_json', 'A JSON object is required.')
            who = terminal_text(self._who()[0] or self._who()[1])
            if path == HF_POSTS[1]:
                state = hf_account.remove(app.root)
                print(f'Hugging Face: token removed ({who})', flush=True)
            else:
                typed = body.get('token')
                try:
                    state = hf_account.save(app.root, typed)
                except ReviewRejected as exc:
                    raise self._masked(exc, typed) from None
                print(f'Hugging Face: token of {terminal_text(state.get("user") or "?")} saved ({who})', flush=True)
            self._json(200, {'ok': True, **state, 'can_edit': self._can_use_ai()})

        def _ai_suggest(self, route, frame_key, body):
            """One frame reviewed by the AI reviewer of Settings, in pixels of the full frame; nothing is saved."""
            from nnslr_tools import ai_review, llm_providers
            if not isinstance(body, dict):
                raise ReviewRejected(400, 'invalid_json', 'A JSON object is required.')
            detections = body.get('detections')
            if detections is not None and not isinstance(detections, list):
                raise ReviewRejected(400, 'invalid_detections', 'detections is a list of boxes.')
            run_id = body.get('run_id')
            if run_id is not None and not isinstance(run_id, str):
                raise ReviewRejected(400, 'invalid_json', 'run_id is the run of the report, a string.')
            provider = llm_providers.reviewer_provider(app.root)
            if provider is None:
                raise ReviewRejected(409, 'no_ai_reviewer', 'Choose an AI reviewer in Settings.')
            llm_providers.check_can_review(provider)
            if not app.ai_slots.acquire(blocking=False):
                raise ReviewRejected(429, 'ai_busy', 'The AI is already answering two frames: try again in a moment.')
            try:
                suggestion = ai_review.suggest_frame(app, route, frame_key, provider, detections=detections,
                                                     run_id=run_id)
            except ReviewRejected as exc:
                print(f'AI: {terminal_text(route)} {frame_key} failed: {terminal_text(exc.reason)}', flush=True)
                raise self._masked(exc, provider.get('key')) from None
            finally:
                app.ai_slots.release()
            print(f'AI: {terminal_text(route)} {frame_key} ({terminal_text(suggestion.get("reviewer", ""))}) '
                  f'{float(suggestion.get("seconds") or 0):.1f} s', flush=True)
            self._json(200, {**suggestion, 'ok': True})

        def _ai_fit(self, route, frame_key, body):
            """The reviewer's rough box fitted tight by the AI reviewer of Settings, shown only the crop around it."""
            from nnslr_tools import ai_review, llm_providers
            if not isinstance(body, dict):
                raise ReviewRejected(400, 'invalid_json', 'A JSON object is required.')
            run_id = body.get('run_id')
            if run_id is not None and not isinstance(run_id, str):
                raise ReviewRejected(400, 'invalid_json', 'run_id is the run of the report, a string.')
            provider = llm_providers.reviewer_provider(app.root)
            if provider is None:
                raise ReviewRejected(409, 'no_ai_reviewer', 'Choose an AI reviewer in Settings.')
            llm_providers.check_can_review(provider)
            if not app.ai_slots.acquire(blocking=False):
                raise ReviewRejected(429, 'ai_busy', 'The AI is already answering two frames: try again in a moment.')
            try:
                fitted = ai_review.fit_box(app, route, frame_key, body.get('box'), provider, run_id=run_id)
            except ReviewRejected as exc:
                print(f'AI fit: {terminal_text(route)} {frame_key} failed: {terminal_text(exc.reason)}: '
                      f'{terminal_text(llm_providers._masked(exc.message, provider.get("key")))}', flush=True)
                raise self._masked(exc, provider.get('key')) from None
            finally:
                app.ai_slots.release()
            print(f'AI fit: {terminal_text(route)} {frame_key} {terminal_text(fitted["message"])}', flush=True)
            self._json(200, fitted)
        # [ai-reviewer] - END

        # [box-fit] - START
        def _local_fit(self, route, frame_key, body):
            """{boxes: [box or None for each box sent], ms}: the reviewer's boxes fitted to the signs under them by
            :mod:`nnslr_tools.box_fit`, on the original frame decoded once. None: no better box, keep it."""
            import math
            import time
            if not isinstance(body, dict):
                raise ReviewRejected(400, 'invalid_json', 'A JSON object is required.')
            run_id = body.get('run_id')
            if run_id is not None and not isinstance(run_id, str):
                raise ReviewRejected(400, 'invalid_json', 'run_id is the run of the report, a string.')
            sent = body.get('boxes')
            if not isinstance(sent, list) or not 1 <= len(sent) <= MAX_FIT_BOXES:
                raise ReviewRejected(400, 'invalid_boxes',
                                     f'boxes is a list of 1 to {MAX_FIT_BOXES} boxes [x1, y1, x2, y2] in pixels.')
            frame = app.get_frame(route, frame_key)
            if run_id is not None and run_id != frame['run_id']:
                raise ReviewRejected(409, 'stale_report', 'The report of this route was rebuilt: reload the page.')
            width, height = frame.get('width'), frame.get('height')
            if type(width) is not int or type(height) is not int or width < 1 or height < 1:
                raise ReviewRejected(409, 'invalid_frame', 'The frame has no size in the report: rebuild the report.')
            boxes = []
            for box in sent:
                if not isinstance(box, list) or len(box) != 4 \
                        or not all(type(v) in (int, float) and math.isfinite(v) for v in box):
                    raise ReviewRejected(400, 'invalid_boxes', 'Each box is [x1, y1, x2, y2]: four finite numbers.')
                x1, y1, x2, y2 = (float(v) for v in box)
                # The page's boxes stay inside the frame; half a pixel of rounding is let through.
                if not (-0.5 <= x1 < x2 <= width + 0.5 and -0.5 <= y1 < y2 <= height + 0.5):
                    raise ReviewRejected(400, 'invalid_boxes', 'A box is outside the frame or not x1 < x2, y1 < y2.')
                boxes.append([x1, y1, x2, y2])
            if not app.fit_slots.acquire(timeout=FIT_WAIT_SECONDS):
                raise ReviewRejected(429, 'fit_busy', 'The server is fitting other boxes: try again in a moment.')
            try:
                started = time.perf_counter()
                from nnslr_tools import box_fit
                try:
                    path = app.image_file(route, frame_key, 'original')[0]
                except (ReviewRejected, OSError):
                    path = app.image_file(route, frame_key, 'preview')[0]
                try:
                    img = box_fit.read_image(path)
                except OSError:
                    raise ReviewRejected(409, 'invalid_image', 'The frame image cannot be read.') from None
                # A preview smaller than the frame: the boxes go to its pixels and back.
                sx, sy = img.shape[1] / width, img.shape[0] / height
                fitted = []
                for x1, y1, x2, y2 in boxes:
                    box = box_fit.fit_image(img, [x1 * sx, y1 * sy, x2 * sx, y2 * sy])
                    fitted.append(None if box is None else
                                  [round(box[0] / sx, 1), round(box[1] / sy, 1), round(box[2] / sx, 1),
                                   round(box[3] / sy, 1)])
                ms = round((time.perf_counter() - started) * 1000)
            finally:
                app.fit_slots.release()
            print(f'Fit box: {terminal_text(route)} {frame_key} {sum(b is not None for b in fitted)}/{len(fitted)} '
                  f'fitted in {ms} ms', flush=True)
            return {'ok': True, 'boxes': fitted, 'ms': ms}
        # [box-fit] - END

        def _same_origin(self):
            fetch_site = self.headers.get('Sec-Fetch-Site')
            if fetch_site is not None and fetch_site not in ('same-origin', 'none'):
                return False
            origin = self.headers.get('Origin')
            if origin is None:
                return True
            host = self.headers.get('Host')
            if not host:
                return False
            # Same host name and port as the request; https as well, for a reverse proxy
            # (Traefik) that terminates TLS and keeps the Host header.
            supplied = origin.strip().lower()
            if supplied == ('http://' + host.strip()).lower():
                return True
            if supplied == ('https://' + host.strip()).lower():
                return True
            return False

        def _read_json(self):
            if self.headers.get('Transfer-Encoding') is not None:
                raise ReviewRejected(411, 'length_required', 'Request without Content-Length.')
            raw_length = self.headers.get('Content-Length')
            if raw_length is None or not raw_length.strip().isdigit():
                raise ReviewRejected(411, 'length_required', 'Request without Content-Length.')
            length = int(raw_length)
            if length > MAX_BODY_BYTES:
                self.close_connection = True
                raise ReviewRejected(413, 'too_large', 'Review too large (at most 512 KiB).')
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ReviewRejected(400, 'invalid_json', 'Incomplete request.')
            try:
                return strict_json_loads(raw.decode('utf-8'))
            except (UnicodeDecodeError, ValueError, TypeError) as exc:
                raise ReviewRejected(400, 'invalid_json', 'Invalid JSON data.') from exc

        def _security_headers(self, csp, cache='no-store'):
            self.send_header('Cache-Control', cache)
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
            self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
            self.send_header('Content-Security-Policy', csp)

        def _send(self, status, content_type, body, csp=DEFAULT_CSP, extra=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self._security_headers(csp)
            if extra is not None:
                for name, value in extra:
                    self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, payload, extra=None):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self._send(status, 'application/json; charset=utf-8', body, DEFAULT_CSP, extra)

        def _problem(self, status, reason, message, api):
            if api:
                self._json(status, {'ok': False, 'error': reason, 'message': message})
            else:
                nonce = secrets.token_urlsafe(18)
                page = render_problem(status, message, nonce, drive=reason == 'unknown_route')
                self._send(status, 'text/html; charset=utf-8', page.encode('utf-8'),
                           "default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'unsafe-inline'; "
                           "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

        def _redirect(self, location, cookie):
            extra = [('Location', location)]
            if cookie is not None:
                extra.append(('Set-Cookie', cookie))
            self._send(303, 'text/plain; charset=utf-8', b'', DEFAULT_CSP, extra)

    return Handler


class _ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class):
        self._connection_slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(server_address, handler_class)

    def process_request(self, request, client_address):
        if not self._connection_slots.acquire(blocking=False):
            # Every slot is busy: close this connection now instead of starting another thread.
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            # The thread did not start, so it will never release its slot.
            self._connection_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connection_slots.release()


class _ReviewServerIPv6(_ReviewServer):
    address_family = socket.AF_INET6


def create_review_server(root: Path, *, host: str = '127.0.0.1', port: int = 8765, token: str | None = None,
                         require_token: bool | None = None):
    """Bind the server (not started). The token is required whenever the address is not loopback."""
    from nnslr_tools.review_db import ReviewStore, db_path
    if require_token is None:
        require_token = not is_loopback(host)
    if require_token and token is None:
        token = review_token()
    store = ReviewStore(db_path(root))
    # /api/v1 and /mcp always need a token, also when the page on 127.0.0.1 does not.
    api_token = token
    if api_token is None:
        api_token = review_token()
    app = ReviewApp(root, store, token=token, require_token=require_token, api_token=api_token)
    if ':' in host:
        server = _ReviewServerIPv6((host, port), make_handler(app))
    else:
        server = _ReviewServer((host, port), make_handler(app))
    # One cookie per port, so two review servers on one machine do not overwrite each other.
    app.cookie_name = 'nnslr_review_' + str(server.server_port)
    return server, app


def _base_urls(host: str, port: int) -> list[str]:
    if host in ('0.0.0.0', '', '::'):
        names = lan_addresses()
        if not names:
            names = [socket.gethostname()]
    elif ':' in host:
        names = ['[' + host + ']']
    else:
        names = [host]
    urls = []
    for name in names:
        urls.append(f'http://{name}:{port}')
    return urls


def serve_review(route: str | None = None, *, data_root: Path | None=None, host: str='127.0.0.1',
                 port: int=8765) -> None:
    root=resolve_data_root(data_root)
    if route is not None:
        latest_report(root,route)
    import sqlite3
    try:
        server, app = create_review_server(root, host=host, port=port)
    except (OSError, sqlite3.Error) as exc:
        raise PreannotationError(f'cannot start review server: {exc}') from exc
    from nnslr_tools.review_db import db_path
    # A token fixed in NNSLR_REVIEW_TOKEN (the systemd service) is never printed:
    # the output may end in a log. A random token must be printed, or nobody could use it.
    token_from_env = bool(os.environ.get(TOKEN_ENV))
    with server:
        port = server.server_port
        # [site-login] - START
        # The address alone: on the LAN the browser signs in (the token is the setup code of the first visit).
        suffix = ''
        # [site-login] - END
        path = '/'
        if route is not None:
            path = '/route/' + route + '/'
        print(f'Revisione NNSLR in ascolto su {host}:{port}', flush=True)
        print('Apri nel browser:', flush=True)
        for base in _base_urls(host, port):
            print(f'  {base}{path}{suffix}', flush=True)
        if route is not None:
            print(f'Elenco di tutte le route: {_base_urls(host, port)[0]}/{suffix}', flush=True)
        # [site-login] - START
        if app.require_token and app.auth.configured():
            print(f'Accedi come amministratore ({app.auth.username()}).', flush=True)
        elif app.require_token and token_from_env:
            print(f'Prima visita: crea l’amministratore con il codice di configurazione, cioè il token di '
                  f'{TOKEN_ENV} (per il servizio: /etc/speedlimits-web.env), che non viene stampato.', flush=True)
        elif app.require_token:
            print(f'Prima visita: crea l’amministratore con questo codice di configurazione: {app.token} '
                  f'(vale finché il server è acceso; fissalo con {TOKEN_ENV}).', flush=True)
        # [site-login] - END
        else:
            print(f'Solo da questo computer. Da un altro: ssh -L {port}:127.0.0.1:{port} <server>, '
                  'oppure riavvia con --host 0.0.0.0', flush=True)
        print(f'Database revisioni: {db_path(root)}', flush=True)
        if app.token is None and not token_from_env:
            # Loopback without a page token: the API token is random and must be shown to be usable.
            print(f'API REST /api/v1 e MCP /mcp: header "Authorization: Bearer {app.api_token}" '
                  f'(fissalo con {TOKEN_ENV}).', flush=True)
        else:
            print(f'API REST /api/v1 e MCP /mcp: header "Authorization: Bearer <{TOKEN_ENV}>" (docs/API_MCP.md).',
                  flush=True)
        print('Ogni “Conferma” viene salvata subito nel database. Ctrl+C per fermare.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            app.close()
# [preannotation] - END
