"""Download a drive someone shared in comma connect: the road camera, the wide camera and the rlog of every segment.

The owner of the comma uploads all the files of the drive in connect (by default a comma uploads only the small
qcamera and qlog), switches the drive to public and sends its link. The files land in the shared folder of the comma
files (COMMA_VIDEOS_DIR) as nnslr_tools.sync lays them out, <route>--<segment>/{fcamera.hevc,ecamera.hevc,rlog.zst},
and sync.link_videos links them into raw/routes, so the drive is prepared like one fetched from our comma. An rlog
connect keeps as rlog.bz2 is re-packed as rlog.zst, the only log the readers take. The driver camera stays out.

The record of where a drive comes from, raw/connect/<route>.json (dongle, car, software, link, time), is written before
the first file and marks it as a drive of another car (outside_drive). It goes where a drive of our comma goes: Sign
review, and the brain and Speed review when it is a dashcam drive (sunnypilot off the whole way; Cristian, 12 Nov
2026). Nothing of the owner's account, VIN or position is kept.

Network: api.commadotai.com and the signed blob links its /files answer gives. The query of a blob link is a signature:
it is never printed, logged or stored (_safe). A link that has expired is asked again once.
"""
from __future__ import annotations

import argparse
import bz2
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

from nnslr_tools import sync

API = 'https://api.commadotai.com'
CONNECT = 'https://connect.comma.ai'
CONNECT_RELATIVE = Path('raw') / 'connect'
# The files taken from each segment; the rlog under whichever name connect keeps it.
CAMERAS = ('fcamera.hevc', 'ecamera.hevc')
LOGS = ('rlog.zst', 'rlog.bz2')
LOG_NAME = 'rlog.zst'
CHUNK = 1 << 20
TIMEOUT = 60
# How the owner shares a drive, for the message of a drive that is not public or not uploaded.
HOW_TO_SHARE = ('In comma connect (connect.comma.ai), open the drive, open its Files menu, press upload next to "All files" '
                'and wait until the upload is done with the comma on Wi-Fi, then switch the drive to public and send its link.')

_DONGLE = r'[0-9a-f]{16}'
_ROUTE = r'[0-9a-f]{8}--[0-9a-f]{10}'
_LINK_RE = re.compile(rf'({_DONGLE})(?:\||%7c|/)({_ROUTE})(?![0-9a-z])', re.IGNORECASE)
_OLD_ROUTE_RE = re.compile(rf'({_DONGLE})(?:\||%7c|/)(\d{{4}}-\d\d-\d\d--\d\d-\d\d-\d\d)', re.IGNORECASE)
_FILE_PATH_RE = re.compile(rf'/[^/]+/({_DONGLE})/({_ROUTE})/(0|[1-9][0-9]*)/([^/]+)$')


class ConnectError(ValueError):
    """A drive that cannot be downloaded, with what to do about it: safe to show in the panel and the CLI."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f'{code}: {message}')
        self.code = code


class _Expired(Exception):
    """A signed blob link the store no longer accepts."""


def _safe(url: str) -> str:
    """A link without its query, the signature of a blob link: the only form printed or kept."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))


def parse_link(text: str) -> tuple[str, str]:
    """(dongle, route) of a connect link (connect.comma.ai/<dongle>/<route>[/...]) or of '<dongle>|<route>'."""
    text = (text or '').strip()
    found = _LINK_RE.search(text)
    if found:
        return found.group(1).lower(), found.group(2).lower()
    if _OLD_ROUTE_RE.search(text):
        raise ConnectError('old_route_name', 'this drive has a date for a name, from an older openpilot: '
                           'only drives named <counter>--<id> (e.g. 0000012a--3f9c1e0b2d) are read here.')
    raise ConnectError('no_link', 'paste the link of the drive from comma connect '
                       '(https://connect.comma.ai/<dongle>/<route>) or <dongle>|<route>.')


def _open(url: str):
    request = urllib.request.Request(url, headers={'User-Agent': 'sunnypilot.speedlimits connect_fetch'})
    return urllib.request.urlopen(request, timeout=TIMEOUT)


def _api(path: str) -> Any:
    try:
        with _open(API + path) as answer:
            return json.load(answer)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403, 404):
            raise ConnectError('not_public', f'comma connect does not show this drive (HTTP {error.code}): it is not '
                               f'public, or the link is wrong. {HOW_TO_SHARE}') from None
        raise ConnectError('connect_error', f'comma connect answered HTTP {error.code} for {path}') from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ConnectError('connect_unreachable', f'api.commadotai.com cannot be reached: {error}') from None


def _route_path(dongle: str, route: str) -> str:
    return '/v1/route/' + urllib.parse.quote(f'{dongle}|{route}', safe='')


def route_meta(dongle: str, route: str, api: Callable[[str], Any] = _api) -> dict:
    """The drive as connect describes it; a drive that is not public stops here."""
    meta = api(_route_path(dongle, route) + '/')
    if not isinstance(meta, dict) or meta.get('is_public') is not True:
        raise ConnectError('not_public', f'the drive is not public. {HOW_TO_SHARE}')
    return meta


def route_files(dongle: str, route: str, api: Callable[[str], Any] = _api) -> dict[int, dict[str, str]]:
    """{segment: {name: signed link}} of the road and wide cameras and the rlog connect holds for the drive. The segment
    is read from each link's path, since the lists skip the segments whose files were never uploaded."""
    answer = api(_route_path(dongle, route) + '/files')
    if not isinstance(answer, dict):
        raise ConnectError('connect_error', 'comma connect gave no list of files')
    found: dict[int, dict[str, str]] = {}
    for kind in ('cameras', 'ecameras', 'logs'):
        for url in answer.get(kind) or ():
            match = _FILE_PATH_RE.search(urllib.parse.urlsplit(str(url)).path)
            if not match or match.group(1).lower() != dongle or match.group(2).lower() != route:
                continue
            name = match.group(4)
            if name in CAMERAS or name in LOGS:
                found.setdefault(int(match.group(3)), {})[name] = str(url)
    return found


def outside_drive(root: str | Path, route: str) -> dict | None:
    """The record of a drive downloaded from comma connect, a drive of another car (raw/connect/<route>.json), or None
    for a drive of our comma. The brain (brain_targets.training_routes) and Speed review leave those out."""
    path = Path(root) / CONNECT_RELATIVE / f'{route}.json'
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        # A record that cannot be read still says the drive came from outside.
        return {'route': route, 'unreadable': str(path)}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    part.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    os.replace(part, path)


def _origin(dongle: str, route: str, meta: dict, old: dict | None) -> dict:
    keep = ('platform', 'make', 'version', 'git_branch', 'git_remote', 'git_commit', 'start_time', 'end_time')
    return {'source': 'comma connect', 'link': f'{CONNECT}/{dongle}/{route}', 'dongle_id': dongle, 'route': route,
            **{key: meta.get(key) for key in keep},
            'first_download': (old or {}).get('first_download') or datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'segments': (old or {}).get('segments') or {}}


def _download(url: str, target: Path, repack: bool) -> int:
    """The file at url into target, through target.part renamed at the end, so a file in place is always whole; with
    repack, an rlog.bz2 is written as zstd. The size written, in bytes."""
    part = target.with_name(target.name + '.part')
    try:
        answer = _open(url)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise _Expired() from None
        raise ConnectError('download_error', f'HTTP {error.code} for {_safe(url)}') from None
    written = 0
    try:
        with answer, open(part, 'wb') as out:
            if repack:
                try:
                    import zstandard
                except ImportError:
                    raise ConnectError('no_zstandard', 'zstandard is missing: run with the .venv python '
                                       '(helpers/fetch_connect_route.sh)') from None
                unpack = bz2.BZ2Decompressor()
                with zstandard.ZstdCompressor(level=10).stream_writer(out, closefd=False) as pack:
                    while chunk := answer.read(CHUNK):
                        pack.write(unpack.decompress(chunk))
                if not unpack.eof:
                    raise ConnectError('download_error', f'{_safe(url)} ended before the end of its bz2 stream')
            else:
                while chunk := answer.read(CHUNK):
                    out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
            written = out.tell()
        os.replace(part, target)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return written


def fetch(link: str, *, segments: str = 'all', videos_dir: str | Path | None = None,
          data_root: str | Path | None = None, dry_run: bool = False, say: Callable[[str], None] = print,
          api: Callable[[str], Any] = _api, download: Callable[[str, Path, bool], int] = _download) -> dict:
    """Download the shared drive at link and link it into raw/routes. Files already in the shared folder are kept.
    Dry-run downloads and writes nothing, and says what it would take and what connect lacks."""
    dongle, route = parse_link(link)
    root = sync._resolve_data_root(data_root)
    videos = sync.resolve_videos_dir(videos_dir)
    sync._check_roots(root, videos)
    if not videos.is_dir():
        raise ConnectError('videos_dir_not_found', f'{videos} (set COMMA_VIDEOS_DIR)')
    wanted = sync._parse_segments(segments)
    old = outside_drive(root, route)
    if old is None and any(videos.glob(f'{route}--*')):
        raise ConnectError('route_of_our_comma', f'{route} is already in {videos} from our comma: nothing downloaded.')
    if old is not None and old.get('dongle_id') not in (None, dongle):
        raise ConnectError('route_taken', f'{route} was downloaded from another dongle ({old.get("dongle_id")}).')

    meta = route_meta(dongle, route, api)
    files = route_files(dongle, route, api)
    if wanted is not None:
        files = {index: names for index, names in files.items() if index in wanted}
    report: dict[str, Any] = {'route': route, 'dongle_id': dongle, 'link': f'{CONNECT}/{dongle}/{route}',
                              'platform': meta.get('platform'), 'videos_dir': str(videos), 'dry_run': dry_run,
                              'segments': sorted(files), 'downloaded': [], 'kept': [], 'missing': [], 'bytes': 0,
                              'link_report': None, 'ok': True}
    for index in sorted(files):
        names = files[index]
        lacking = [name for name in CAMERAS if name not in names]
        if not any(log in names for log in LOGS):
            lacking.append('rlog')
        report['missing'] += [f'{index}/{name}' for name in lacking]
    if not files:
        raise ConnectError('nothing_uploaded', f'connect holds no road camera or rlog of this drive. {HOW_TO_SHARE}')
    say(f'{route} of {dongle} ({meta.get("platform") or "car unknown"}): {len(files)} segments in connect')
    if report['missing']:
        say(f'not uploaded to connect: {", ".join(report["missing"])}. {HOW_TO_SHARE}')
    if dry_run:
        return report

    origin = _origin(dongle, route, meta, old)
    record = root / CONNECT_RELATIVE / f'{route}.json'
    _write_json(record, origin)
    refreshed = False
    for position, index in enumerate(sorted(files), 1):
        folder = videos / f'{route}--{index}'
        sync._check_destination(videos, folder)
        folder.mkdir(exist_ok=True)
        names = files[index]
        jobs = [(name, name, False) for name in CAMERAS if name in names]
        log = next((name for name in LOGS if name in names), None)
        if log is not None:
            jobs.append((log, LOG_NAME, log != LOG_NAME))
        for source, name, repack in jobs:
            target = folder / name
            if target.is_file():
                report['kept'].append(f'{index}/{name}')
                continue
            say(f'segment {index} ({position}/{len(files)}): {name}')
            try:
                size = download(files[index][source], target, repack)
            except _Expired:
                if refreshed:
                    raise ConnectError('link_expired', 'connect refused its own links twice; run it again') from None
                refreshed = True
                fresh = route_files(dongle, route, api)
                files.update({key: value for key, value in fresh.items() if key in files})
                if source not in files[index]:
                    raise ConnectError('file_gone', f'connect no longer holds {index}/{source}') from None
                size = download(files[index][source], target, repack)
            report['downloaded'].append(f'{index}/{name}')
            report['bytes'] += size
        origin['segments'][str(index)] = sorted(path.name for path in folder.iterdir() if not path.name.endswith('.part'))
        _write_json(record, origin)
    origin['last_download'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    _write_json(record, origin)
    linked = sync.link_videos(route=route, videos_dir=videos, data_root=root)
    report['link_report'] = {key: linked.get(key) for key in ('counts', 'missing_files', 'log_only', 'ok')}
    report['ok'] = bool(linked.get('ok'))
    say(f'done: {len(report["downloaded"])} files downloaded ({report["bytes"] / 1e9:.2f} GB), '
        f'{len(report["kept"])} already there; linked into {root / "raw" / "routes" / route}')
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m nnslr_tools.connect_fetch', description=__doc__.split('\n\n')[0])
    parser.add_argument('link', help='the drive in comma connect: https://connect.comma.ai/<dongle>/<route> or <dongle>|<route>')
    parser.add_argument('--segments', default='all', help='all (default), 0-3 or 0,2,4-6')
    parser.add_argument('--videos-dir', help='shared folder of the comma files (default COMMA_VIDEOS_DIR)')
    parser.add_argument('--data-root', help='data root (default NNSLR_DATA_ROOT or data/ of the clone)')
    parser.add_argument('--dry-run', action='store_true', help='say what would be downloaded, write nothing')
    parser.add_argument('--json', action='store_true', help='print the report as JSON at the end')
    args = parser.parse_args(argv)
    try:
        report = fetch(args.link, segments=args.segments, videos_dir=args.videos_dir, data_root=args.data_root,
                       dry_run=args.dry_run, say=lambda line: print(line, flush=True))
    except (ConnectError, sync.SyncError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
