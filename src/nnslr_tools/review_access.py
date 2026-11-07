# [guest-reviewers] - START
"""The reviewers from outside: who they are, what they may open, and the work given to them.

A guest never signs in here. The proxy in front of the site (Traefik with Authentik, the only way in from outside) has
already checked the person and sends the name and the groups in X-authentik-* headers. Those headers count only on a
request that proves it came through the proxy: from one of NNSLR_PROXY_IPS and with NNSLR_PROXY_SECRET in the header
PROXY_SECRET_HEADER, which the proxy adds and overwrites. Any other request is the site as always (the administrator's
sign-in, the access token of the agents), with those headers ignored. Without both settings nobody is a guest.

The way in from outside never gives more than a reviewer's work, whatever the groups say: the administrator signs in
from the LAN only. On that way the administrator's cookie and the access token count for nothing, /api/v1 and /mcp are
closed, and a request the proxy sends with no valid reviewer is refused, never shown the local sign-in. NNSLR_PUBLIC_HOSTS
names the addresses of that way, so a proxy that lost its settings refuses instead of opening the local site. Their / is
the public page of the project, on a machine of its own in front of the site (docs/smartspeed), so a guest's home is
/reviews, and a / that reaches the site leads there.

What a guest may open is the table GUEST_ROUTES, and nothing else: a page added to the site tomorrow stays closed until
someone adds it here. A row with kinds holds a drive: the server checks, on every request, that this work on this drive
is given to this person (Assignments). The answers lose the places of the drive (scrubbed): the positions, the names of
roads and towns, the paths of this server.

Files (<data root>/web/auth/, next to the administrator's): assignments.json, 600, written all at once. Standard library
only.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from nnslr_tools.review_auth import USERNAME_RE, read_private, write_private
from nnslr_tools.review_db import ReviewRejected

PROXY_IPS_ENV = 'NNSLR_PROXY_IPS'
PROXY_SECRET_ENV = 'NNSLR_PROXY_SECRET'
PUBLIC_HOSTS_ENV = 'NNSLR_PUBLIC_HOSTS'
# Set on the machine of the outside reviewers: only there a person can say a review is completed.
REVIEW_BOX_ENV = 'NNSLR_REVIEW_BOX'
REVIEWER_GROUP_ENV = 'NNSLR_REVIEWER_GROUP'
DEFAULT_REVIEWER_GROUP = 'reviewers'
PROXY_SECRET_HEADER = 'X-Speedlimits-Proxy-Secret'
USERNAME_HEADER = 'X-authentik-username'
GROUPS_HEADER = 'X-authentik-groups'
# Authentik's own address for ending a session, answered by the proxy before the request reaches the site.
SIGN_OUT_ADDRESS = '/outpost.goauthentik.io/sign_out'
# How a reviewer's name is stored on the labels: never a bare name that reads as the administrator, never "ai/".
GUEST_PREFIX = 'guest/'

NOT_A_REVIEWER = 'Your account has no access to this site: ask who invited you.'
NOT_THROUGH_PROXY = 'This address opens only through the sign-in of the site.'
NOT_ASSIGNED = 'This is not part of the work given to you.'

# The kinds of work, in the order of the pages: (key, what the person does).
WORK_KINDS = (('signs', 'Sign review'), ('distance', 'Sign distance'), ('speed', 'Speed review'))
KIND_KEYS = tuple(key for key, _ in WORK_KINDS)
# The page of each kind for a drive.
WORK_LINKS = {'signs': '/route/{route}/', 'distance': '/sign-distance?drive={route}', 'speed': '/speed-review?drive={route}'}

DRIVE = r'([0-9a-f]{8}--[0-9a-f]{10})'
FRAME = r'[0-9]{1,4}/[0-9]{1,6}'
SAVED_FRAME = r'[0-9]{1,9}/[0-9]{1,9}'
# All a guest can ask: (method, address, the kinds of work that open it). None: no drive in the address, the same for
# every reviewer. The first group of a row with kinds is the drive.
GUEST_ROUTES = tuple((method, re.compile(address), kinds) for method, address, kinds in (
    # The pages of the guest's app (pages.render_guest), which hold no data, and their own list.
    ('GET', r'/(reviews|speed-review|sign-distance)?', None),
    ('GET', r'/api/auth/state', None),
    ('GET', r'/api/reviewer/work', None),
    # "Review completed" of the list: a drive of the guest's, checked on their work (Assignments.complete).
    ('POST', r'/api/reviewer/done', None),
    # The list behind "Next drive" of Sign distance: the server keeps only the guest's drives.
    ('GET', r'/api/sign-distance/drives', None),
    # Sign review.
    ('GET', r'/route/' + DRIVE + r'(/(index\.html)?)?', ('signs',)),
    ('GET', r'/route/' + DRIVE + r'/images/[0-9]{8}\.jpg', ('signs',)),
    ('GET', r'/route/' + DRIVE + r'/video/' + FRAME + r'\.jpg', ('signs',)),
    ('GET', r'/api/route/' + DRIVE + r'/(reviews|live|busy(/[0-9]{1,7})?)', ('signs',)),
    ('GET', r'/api/route/' + DRIVE + r'/video/' + FRAME, ('signs',)),
    ('POST', r'/api/route/' + DRIVE + r'/frame/' + SAVED_FRAME + r'(/apply-previous|/fit)?', ('signs',)),
    # Sign distance.
    ('GET', r'/api/sign-distance/drives/' + DRIVE, ('distance',)),
    ('GET', r'/api/sign-distance/drives/' + DRIVE + r'/signs/[0-9]{1,20}', ('distance',)),
    ('GET', r'/api/sign-distance/drives/' + DRIVE + r'/frames/' + FRAME + r'(-wide)?\.jpg', ('distance',)),
    ('POST', r'/api/sign-distance/drives/' + DRIVE + r'/(marks|undo)', ('distance',)),
    # Speed review, on the lanes and the video of the drive.
    ('GET', r'/api/speed-review/drives/' + DRIVE, ('speed',)),
    ('GET', r'/api/speed-review/drives/' + DRIVE + r'/frames/' + FRAME + r'\.jpg', ('speed',)),
    ('POST', r'/api/speed-review/drives/' + DRIVE + r'/(stretches|works)', ('speed',)),
    ('GET', r'/api/drive-review/drives/' + DRIVE + r'(/timeline)?', ('speed',)),
    ('GET', r'/api/panel/drive-video/' + DRIVE, ('speed',)),
    ('GET', r'/route/' + DRIVE + r'/video\.mp4', ('speed',)),
))

# What a guest never receives, at any depth of an answer: where the drive was.
PLACE_FIELDS = frozenset({'gps', 'latitude', 'longitude', 'lat', 'lon', 'lng', 'road_name', 'title', 'way_id'})
# The last time a person was seen is written at most this often.
SEEN_SAVE_SECONDS = 3600


def guest_rule(method: str, path: str) -> tuple[str | None, tuple[str, ...]] | None:
    """What opens an address for a guest: (drive, kinds), (None, ()) for one without a drive; None when it stays closed."""
    for allowed, address, kinds in GUEST_ROUTES:
        match = address.fullmatch(path) if allowed == method else None
        if match:
            return (None, ()) if kinds is None else (match.group(1), kinds)
    return None


def scrubbed(value):
    """An answer without the places of the drive and the paths of this server, at any depth."""
    if isinstance(value, dict):
        return {key: scrubbed(item) for key, item in value.items()
                if key not in PLACE_FIELDS and not (key.endswith('_path') and str(item).startswith('/'))}
    if isinstance(value, list):
        return [scrubbed(item) for item in value]
    return value


def host_name(header) -> str:
    """The name of a Host header, lower case, without its port."""
    name = header.strip().lower() if isinstance(header, str) else ''
    return name.rsplit(':', 1)[0] if ':' in name and not name.endswith(']') else name


def _listed(text: str | None) -> frozenset[str]:
    return frozenset(part.strip().lower() for part in (text or '').split(',') if part.strip())


class GuestAccess:
    """Which requests come from a reviewer outside. Nobody does until the proxy's addresses and secret are set."""

    def __init__(self, *, proxy_ips: frozenset[str] = frozenset(), secret: str = '', group: str = DEFAULT_REVIEWER_GROUP,
                 public_hosts: frozenset[str] = frozenset()):
        self.proxy_ips = proxy_ips
        self.secret = secret
        self.group = group
        self.public_hosts = public_hosts

    @classmethod
    def from_env(cls) -> GuestAccess:
        return cls(proxy_ips=_listed(os.environ.get(PROXY_IPS_ENV)), secret=os.environ.get(PROXY_SECRET_ENV, ''),
                   group=os.environ.get(REVIEWER_GROUP_ENV, '').strip() or DEFAULT_REVIEWER_GROUP,
                   public_hosts=_listed(os.environ.get(PUBLIC_HOSTS_ENV)))

    @property
    def enabled(self) -> bool:
        return bool(self.proxy_ips) and bool(self.secret)

    def _through_proxy(self, peer: str, headers) -> bool:
        """Both proofs, the address and the secret, on one copy of the header."""
        if not self.enabled or peer.removeprefix('::ffff:') not in self.proxy_ips:
            return False
        sent = headers.get_all(PROXY_SECRET_HEADER, [])
        return len(sent) == 1 and hmac.compare_digest(sent[0].encode('utf-8'), self.secret.encode('utf-8'))

    def guest_of(self, peer: str, headers) -> str | None:
        """The reviewer behind a request; None for a request that did not come the outside way. Raises for one that
        came that way, or to its address, without a reviewer: it must not fall through to the local site."""
        if not self._through_proxy(peer, headers):
            if host_name(headers.get('Host')) in self.public_hosts:
                raise ReviewRejected(403, 'not_through_proxy', NOT_THROUGH_PROXY)
            return None
        names = headers.get_all(USERNAME_HEADER, [])
        groups = headers.get_all(GROUPS_HEADER, [])
        if len(names) != 1 or len(groups) != 1 or not USERNAME_RE.fullmatch(names[0]) \
                or self.group not in groups[0].split('|'):
            raise ReviewRejected(403, 'not_a_reviewer', NOT_A_REVIEWER)
        return names[0]


def _now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


class Assignments:
    """The people the administrator gave work to, and the work: which kinds of which drives."""

    def __init__(self, directory: Path, clock=time.monotonic):
        self.path = Path(directory) / 'assignments.json'
        self._lock = threading.Lock()
        self._clock = clock
        self._seen: dict[str, float] = {}

    def _read(self) -> dict:
        text = read_private(self.path)
        if not text:
            return {'people': {}, 'work': {}}
        try:
            data = json.loads(text)
        except ValueError:
            raise ReviewRejected(500, 'unreadable', 'The assignments file of this server is damaged.') from None
        if not isinstance(data, dict) or not isinstance(data.get('people'), dict) or not isinstance(data.get('work'), dict) \
                or not all(isinstance(drives, dict) and all(isinstance(kinds, list) for kinds in drives.values())
                           for drives in data['work'].values()):
            raise ReviewRejected(500, 'unreadable', 'The assignments file of this server is damaged.')
        # A kind of work this server no longer gives (Roads, until 7 Nov 2026) is left out, and a drive with only that.
        work = {name: {route: [kind for kind in kinds if kind in KIND_KEYS] for route, kinds in drives.items()}
                for name, drives in data['work'].items()}
        data['work'] = {name: {route: kinds for route, kinds in drives.items() if kinds} for name, drives in work.items()}
        return data

    def _write(self, data: dict) -> None:
        write_private(self.path, json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + '\n')

    def view(self) -> dict:
        """Everything, for the page of the administrator: {'people': [{'name', 'added', 'last_seen'}], 'work'}."""
        data = self._read()
        people = [{'name': name, 'added': item.get('added'), 'last_seen': item.get('last_seen')}
                  for name, item in sorted(data['people'].items(), key=lambda pair: pair[0].lower())]
        return {'people': people, 'work': data['work']}

    def add_person(self, name) -> None:
        check_name(name)
        with self._lock:
            data = self._read()
            data['people'].setdefault(name, {'added': _now(), 'last_seen': None})
            self._write(data)

    def remove_person(self, name) -> None:
        """A person and all their work. Their account stays in Authentik: with no work they see an empty list."""
        check_name(name)
        with self._lock:
            data = self._read()
            data['people'].pop(name, None)
            data['work'].pop(name, None)
            self._write(data)
            self._seen.pop(name, None)

    def set_work(self, name, route, kind, on) -> None:
        check_name(name)
        if not isinstance(route, str) or not re.fullmatch(DRIVE, route):
            raise ReviewRejected(400, 'invalid_assignment', 'route is the id of a drive.')
        if kind not in KIND_KEYS:
            raise ReviewRejected(400, 'invalid_assignment', 'kind is one of ' + ', '.join(KIND_KEYS) + '.')
        if type(on) is not bool:
            raise ReviewRejected(400, 'invalid_assignment', 'on is true or false.')
        with self._lock:
            data = self._read()
            if name not in data['people']:
                raise ReviewRejected(404, 'unknown_person', 'Add this person first.')
            drives = data['work'].setdefault(name, {})
            kinds = set(drives.get(route, ()))
            if on:
                kinds.add(kind)
            else:
                kinds.discard(kind)
            if kinds:
                drives[route] = [key for key in KIND_KEYS if key in kinds]
            else:
                drives.pop(route, None)
            self._write(data)

    def complete(self, name, route) -> None:
        """A person says their work on a drive is finished. It stays theirs until a copy from the review box has taken the
        results (nnslr_tools.review_box), which then takes it back."""
        with self._lock:
            data = self._read()
            kinds = data['work'].get(name, {}).get(route) if isinstance(route, str) else None
            if not kinds:
                raise ReviewRejected(403, 'not_assigned', NOT_ASSIGNED)
            done = data.setdefault('done', {}).setdefault(name, {}).setdefault(route, {})
            for kind in kinds:
                done.setdefault(kind, _now())
            self._write(data)

    def done_of(self, name: str) -> dict:
        """By drive, by kind, when a person called it finished."""
        return self._read().get('done', {}).get(name, {})

    def visits(self) -> dict:
        """What the review box tells this server: who came, when, and what they called finished."""
        data = self._read()
        return {'people': data['people'], 'done': data.get('done', {})}

    def replace_work(self, names: list[str], work: dict) -> None:
        """On the review box: the people and the work this server gives. A visit keeps its time, and a finished work
        stays said only while it is still given."""
        with self._lock:
            data = self._read()
            people = data['people']
            data['people'] = {name: people.get(name, {'added': _now(), 'last_seen': None}) for name in names}
            data['work'] = work
            data['done'] = {name: {route: {kind: at for kind, at in kinds.items() if kind in work.get(name, {}).get(route, ())}
                                   for route, kinds in drives.items()}
                            for name, drives in data.get('done', {}).items()}
            self._write(data)

    def take_visits(self, visits: dict, known: list[str]) -> None:
        """On this server: the last visits seen on the review box, and the people new there. Known: the people the box
        was last told, so that someone removed here is not brought back by the box."""
        with self._lock:
            data = self._read()
            changed = False
            for name, item in visits.items():
                last = item.get('last_seen') if isinstance(item, dict) else None
                if not isinstance(last, str) or not re.fullmatch(r'[0-9TZ:+.-]{10,35}', last):
                    last = None
                if name in data['people']:
                    if last and last > (data['people'][name].get('last_seen') or ''):
                        data['people'][name]['last_seen'] = last
                        changed = True
                elif name not in known and USERNAME_RE.fullmatch(name):
                    data['people'][name] = {'added': _now(), 'last_seen': last}
                    changed = True
            if changed:
                self._write(data)

    def work_of(self, name: str) -> dict:
        """The drives of a person, with the kinds of each."""
        return self._read()['work'].get(name, {})

    def may(self, name: str, route: str, kinds: tuple[str, ...]) -> bool:
        return any(kind in kinds for kind in self.work_of(name).get(route, ()))

    def seen(self, name: str) -> None:
        """A person just came: the first time they join the list of the administrator, then their last visit, at most
        once in SEEN_SAVE_SECONDS."""
        now = self._clock()
        last = self._seen.get(name)
        if last is not None and now - last < SEEN_SAVE_SECONDS:
            return
        with self._lock:
            self._seen[name] = now
            data = self._read()
            data['people'].setdefault(name, {'added': _now()})['last_seen'] = _now()
            self._write(data)


def check_name(name) -> None:
    if not isinstance(name, str) or not USERNAME_RE.fullmatch(name):
        raise ReviewRejected(400, 'invalid_name', 'The name is the person\'s user name in Authentik: letters, digits and '
                                                  '. _ @ -, at most 64.')
# [guest-reviewers] - END
