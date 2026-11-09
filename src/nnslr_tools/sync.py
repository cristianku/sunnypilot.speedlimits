# [nnslr-sync] - START
"""Explicit, read-only remote acquisition of one comma route via SSH/rsync.

Only calling sync_route accesses the network. Complete local files are never
overwritten; rsync's separate partial directory makes interrupted copies resumable.

The raw comma files are stored once in a shared folder outside the clone
(COMMA_VIDEOS_DIR), with the same layout as on the comma:
<videos dir>/<route>--<segment>/<file>. Other projects (openpilot.distill) use
the same folder. <data root>/raw/routes/<route>/<segment>/<file> only holds
absolute symlinks to those files. In this project sync_route is the only
writer of the shared folder (openpilot.distill's fetch script writes it too);
link_videos only reads it. sync_route and link_videos are the only writers of
those symlinks.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any

from nnslr_tools.manifest import NnslerManifestError, RouteIdentity, parse_route_id, sha256_of

DEFAULT_HOST = "comma"
# Datasets, frames and runs live in data/ inside the clone (gitignored).
DEFAULT_DATA_ROOT = str(Path(__file__).resolve().parents[2] / "data")
# Raw videos and logs live in the folder comma-videos next to the clone.
DEFAULT_VIDEOS_DIR = str(Path(__file__).resolve().parents[3] / "comma-videos")
REMOTE_ROOT = "/data/media/0/realdata"
_CAMERAS = {"narrow": ("fcamera.hevc",), "wide": ("ecamera.hevc",),
            "both": ("fcamera.hevc", "ecamera.hevc")}
_LOG_NAMES = ("rlog.zst", "rlog", "rlog.bz2")
# Raw files that link_videos links when it finds them in a shared segment folder.
_LINKABLE_NAMES = ("fcamera.hevc", "ecamera.hevc", "dcamera.hevc", "qcamera.ts",
                   "rlog.zst", "rlog", "rlog.bz2", "qlog.zst", "qlog", "qlog.bz2")
_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
_HOST_RE = re.compile(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9_][A-Za-z0-9_.-]*")


class SyncError(ValueError):
    """A local preflight or remote transport failure, safe to show in the CLI."""


def resolve_videos_dir(explicit: str | Path | None = None) -> Path:
    """Shared folder of the raw comma files: explicit, COMMA_VIDEOS_DIR or the default."""
    if explicit:
        raw = str(explicit)
    else:
        raw = os.environ.get("COMMA_VIDEOS_DIR", "").strip()
        if not raw:
            raw = DEFAULT_VIDEOS_DIR
    return Path(raw).expanduser().resolve()


def _resolve_data_root(explicit: str | Path | None) -> Path:
    if explicit:
        raw = str(explicit)
    else:
        raw = os.environ.get("NNSLR_DATA_ROOT", "").strip()
        if not raw:
            raw = DEFAULT_DATA_ROOT
    return Path(raw).expanduser().resolve()


def _videos_dir_warning(videos: Path) -> str | None:
    # Readers (route-manifest, extract-frames, ...) take the shared folder from
    # COMMA_VIDEOS_DIR or the default, never from --videos-dir.
    readers = resolve_videos_dir()
    if videos == readers:
        return None
    return (f"readers only accept links into COMMA_VIDEOS_DIR ({readers}); "
            f"export COMMA_VIDEOS_DIR={shlex.quote(str(videos))}")


def _check_roots(root: Path, videos: Path) -> None:
    # The links in raw/routes point into the shared folder, so the shared folder
    # must never be (or be inside) raw/routes itself.
    routes = root / "raw" / "routes"
    if videos == routes or videos.is_relative_to(routes):
        raise SyncError(f"COMMA_VIDEOS_DIR must be outside {routes}: {videos}")
    if videos.exists() and not videos.is_dir():
        raise SyncError(f"COMMA_VIDEOS_DIR is not a directory: {videos}")


def _parse_segments(value: str) -> list[int] | None:
    if value == "all":
        return None
    selected: set[int] = set()
    for part in value.split(","):
        match = re.fullmatch(r"(0|[1-9][0-9]*)(?:-(0|[1-9][0-9]*))?", part.strip())
        if match is None:
            raise SyncError("invalid segments: use all, 0-3, or 0,2,4-6")
        start = int(match[1])
        end = int(match[2]) if match[2] is not None else start
        if end < start:
            raise SyncError("invalid segments: range end precedes start")
        selected.update(range(start, end + 1))
    return sorted(selected)


def _check_destination(root: Path, path: Path) -> None:
    # Check every component, including partial files. A symlink must not redirect
    # rsync into another dataset (even one under the same configured root).
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise SyncError(f"destination_symlink: {current}")
    if root.exists() and not root.is_dir():
        raise SyncError(f"data root is not a directory: {root}")


def _inventory(host: str, route: str, camera_files: tuple[str, ...]) -> dict[int, set[str]]:
    # Route and host have already been validated. The remote command uses only
    # POSIX shell builtins; no remote Python, GNU find or software installation.
    names = " ".join(shlex.quote(name) for name in (*camera_files, *_LOG_NAMES))
    command = (
        f"test -d {shlex.quote(REMOTE_ROOT)} || exit 1; "
        f"for d in {shlex.quote(REMOTE_ROOT + '/' + route + '--')}*; do "
        '[ -d "$d" ] && [ ! -L "$d" ] || continue; '
        'printf "segment\\t%s\\n" "${d##*/}"; '
        f"for name in {names}; do "
        '[ -f "$d/$name" ] && [ ! -L "$d/$name" ] || continue; '
        'printf "file\\t%s\\t%s\\n" "${d##*/}" "$name"; '
        "done; done"
    )
    try:
        proc = subprocess.run([*_SSH, host, command], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise SyncError(f"ssh_inventory_timeout: {host}") from exc
    if proc.returncode:
        raise SyncError(f"ssh_inventory_failed: {host} (exit {proc.returncode}): {proc.stderr.strip()}")
    inventory: dict[int, set[str]] = {}
    allowed = set((*camera_files, *_LOG_NAMES))
    for line in proc.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) not in (2, 3):
            raise SyncError(f"invalid_remote_inventory: {line!r}")
        match = re.fullmatch(re.escape(route) + r"--(0|[1-9][0-9]*)", fields[1])
        if match is None:
            raise SyncError(f"invalid_remote_segment: {fields[1]!r}")
        index = int(match[1])
        if fields[0] == "segment" and len(fields) == 2 and index not in inventory:
            inventory[index] = set()
        elif fields[0] == "file" and len(fields) == 3 and index in inventory and fields[2] in allowed:
            inventory[index].add(fields[2])
        else:
            raise SyncError(f"invalid_remote_inventory: {line!r}")
    if not inventory:
        raise SyncError(f"route_not_found: {route} on {host}:{REMOTE_ROOT}")
    return inventory


def _rsync_source(host: str, route: str, segment: int, name: str) -> str:
    return f"{host}:{REMOTE_ROOT}/{route}--{segment}/{name}"


def _copy_file(source: str, destination: Path) -> tuple[str, str | None]:
    transport = ["-e", shlex.join(_SSH)]
    if destination.exists():
        if not destination.is_file():
            return "failed", f"destination_conflict: not a regular file: {destination}"
        # Size/mtime alone cannot establish that an existing recording is the
        # same file. Check content without ever overwriting it or its metadata.
        proc = subprocess.run(
            ["rsync", "--checksum", "--dry-run", "--itemize-changes", *transport,
             "--", source, str(destination)], capture_output=True, text=True,
        )
        if proc.returncode:
            return "failed", f"rsync_check_failed (exit {proc.returncode}): {proc.stderr.strip()}"
        if proc.stdout.strip():
            return "failed", f"destination_conflict: content differs: {destination}"
        return "existing", None

    partial = destination.parent / ".nnslr-partial" / destination.name
    resuming = partial.is_file()
    destination.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["rsync", "--times", "--ignore-existing", "--partial-dir=.nnslr-partial",
         "--progress", *transport, "--", source, str(destination)],
        stdout=sys.stderr,  # progress remains visible; stdout is the JSON report
    )
    if proc.returncode:
        return "failed", f"rsync_transfer_failed (exit {proc.returncode}); rerun the same command to resume"
    if not destination.is_file():
        return "failed", f"rsync_destination_missing: {destination}"
    return ("resumed" if resuming else "copied"), None


def _move_file(source: Path, target: Path) -> None:
    # The caller has checked that target does not exist. On the same disk this
    # is one rename. On another disk the copy gets a temporary name first, so an
    # interrupted move never leaves a truncated file under the final name.
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(source, target)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    temporary = target.parent / (".nnslr-move-" + target.name)
    if temporary.is_symlink() or temporary.exists():
        # Left over by an interrupted move, or a planted link. copy2 would
        # follow a link and overwrite its target; unlinking removes only the link.
        temporary.unlink()
    shutil.copy2(source, temporary)
    os.replace(temporary, target)
    source.unlink()


def _symlink_state(link: Path) -> str:
    """'live', 'dangling' or 'loop' for a path the caller knows is a symlink."""
    # os.stat follows the link; Path.resolve() would raise RuntimeError on a
    # loop (Python 3.11), so the loop is detected here first.
    try:
        os.stat(link)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            return "loop"
        if exc.errno == errno.ENOENT or exc.errno == errno.ENOTDIR:
            return "dangling"
        raise
    return "live"


def _adopt_local_file(link: Path, shared: Path) -> tuple[str | None, str | None]:
    """Turn an old regular file in raw/routes into a symlink to the shared copy.

    The shared copy must already exist. Returns (link_status, error).
    link_status is None when there is no old regular file at the link path
    (nothing to do). The old file is never moved here: only sync_route moves
    one into the shared folder, after checking it against the comma.
    """
    if link.is_symlink() or not link.exists():
        return None, None
    if not link.is_file():
        return "failed", f"destination_conflict: not a regular file: {link}"
    if shared.is_symlink() or not shared.is_file():
        return "failed", f"destination_conflict: shared copy is missing or not a regular file: {shared}"
    local_size = link.stat().st_size
    shared_size = shared.stat().st_size
    if local_size != shared_size:
        return "failed", (f"destination_conflict: size differs: {link} ({local_size} bytes) "
                          f"and shared copy {shared} ({shared_size} bytes); nothing changed, "
                          "check both files and remove the wrong one by hand")
    if sha256_of(link) != sha256_of(shared):
        return "failed", (f"destination_conflict: content differs: {link} and shared copy {shared}; "
                          "nothing changed, check both files and remove the wrong one by hand")
    # Identical content: the old file is only a duplicate of the shared copy.
    link.unlink()
    link.symlink_to(shared)
    return "replaced_duplicate", None


def _ensure_link(link: Path, shared: Path) -> tuple[str, str | None]:
    """Create the absolute symlink link -> shared, or accept an identical one."""
    if link.is_symlink():
        state = _symlink_state(link)
        if state == "loop":
            return "failed", f"destination_conflict: symlink loop: {link}"
        if state == "dangling":
            # A link to nothing holds no data (e.g. the shared folder was moved
            # to a new COMMA_VIDEOS_DIR), so it is pointed to the shared copy.
            link.unlink()
            link.symlink_to(shared)
            return "relinked", None
        if link.resolve() == shared.resolve():
            return "existing", None
        return "failed", f"destination_conflict: {link} points to {os.readlink(link)}, expected {shared}"
    if link.exists():
        return "failed", f"destination_conflict: not a symlink to the shared folder: {link}"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(shared)
    return "created", None


def _plan_link(link: Path, shared: Path) -> tuple[str, str | None]:
    """Predict what link_videos will do at `link`, without changing anything.

    Returns (status, error): planned, existing, would_relink,
    would_replace_duplicate, or conflict with the reason.
    """
    if link.is_symlink():
        state = _symlink_state(link)
        if state == "loop":
            return "conflict", f"destination_conflict: symlink loop: {link}"
        if state == "dangling":
            return "would_relink", None
        if link.resolve() == shared.resolve():
            return "existing", None
        return "conflict", f"destination_conflict: {link} points to {os.readlink(link)}, expected {shared}"
    if not link.exists():
        return "planned", None
    if not link.is_file():
        return "conflict", f"destination_conflict: not a regular file: {link}"
    local_size = link.stat().st_size
    shared_size = shared.stat().st_size
    if local_size != shared_size:
        return "conflict", (f"destination_conflict: size differs: {link} ({local_size} bytes) "
                            f"and shared copy {shared} ({shared_size} bytes)")
    # Same size: the real run also compares the content (sha256) first.
    return "would_replace_duplicate", None


def _plan_log_only_cleanup(report: dict[str, Any], route_id: str, index: int,
                           segment_dir: Path, shared_dir: Path) -> None:
    """Plan the removal of the links an earlier run made for a log-only segment.

    Only a folder that holds nothing but links into shared_dir is removed. Anything
    else (an old copy, a link elsewhere) is left alone and reported in
    missing_files, because without fcamera.hevc it would stop extract-frames.
    """
    if not segment_dir.is_dir():
        return
    links = []
    for entry in sorted(segment_dir.iterdir()):
        own_link = False
        if entry.is_symlink() and os.readlink(entry) == str(shared_dir / entry.name):
            own_link = True
        if not own_link:
            report["missing_files"].append({"route": route_id, "segment": index, "name": "fcamera.hevc"})
            return
        links.append(str(entry))
    report["removed"].append({"route": route_id, "segment": index, "path": str(segment_dir),
                              "links": links, "status": "would_remove"})


def sync_route(
    *, route: str, host: str | None = None, segments: str = "all",
    camera: str = "narrow", data_root: str | Path | None = None,
    videos_dir: str | Path | None = None, dry_run: bool = False,
) -> dict[str, Any]:
    """Inventory and copy selected camera files plus one full log per segment.

    Files are copied into the shared videos folder and linked into
    raw/routes. The route is always explicit. Omitted segments mean only the
    available segments of that route. Dry-run contacts SSH for inventory but
    writes nothing.
    """
    parse_route_id(route)
    host = host if host is not None else os.environ.get("NNSLR_COMMA_HOST", DEFAULT_HOST)
    if _HOST_RE.fullmatch(host) is None:
        raise SyncError("invalid SSH host: use an alias, hostname, IPv4 address or user@host")
    selected = _parse_segments(segments)
    if camera not in _CAMERAS:
        raise SyncError(f"invalid camera: {camera}")
    root = _resolve_data_root(data_root)
    videos = resolve_videos_dir(videos_dir)
    _check_roots(root, videos)
    destination_root = root / "raw" / "routes" / route
    _check_destination(root, destination_root)
    for executable in ("ssh", "rsync"):
        if shutil.which(executable) is None:
            raise SyncError(f"sync_tool_not_found: install {executable} before using sync-routes")

    inventory = _inventory(host, route, _CAMERAS[camera])
    selected = sorted(inventory) if selected is None else selected
    report: dict[str, Any] = {
        "host": host, "route": route, "segments": selected, "camera": camera,
        "data_root": str(root), "videos_dir": str(videos), "destination": str(destination_root),
        "dry_run": dry_run, "missing_segments": [], "missing_files": [], "files": [], "ok": True,
        "warning": _videos_dir_warning(videos),
    }
    for index in selected:
        if index not in inventory:
            report["missing_segments"].append(index)
            continue
        available = inventory[index]
        log = next((name for name in _LOG_NAMES if name in available), None)
        for name in (*_CAMERAS[camera], log or "rlog"):
            if name not in available:
                report["missing_files"].append({"segment": index, "name": name})
                continue
            destination = destination_root / str(index) / name
            shared = videos / f"{route}--{index}" / name
            # Only the file itself may be a symlink in raw/routes (checked by
            # _ensure_link); in the shared folder nothing may be a symlink.
            _check_destination(root, destination.parent)
            _check_destination(videos, shared)
            _check_destination(videos, shared.parent / ".nnslr-partial" / name)
            report["files"].append({
                "segment": index, "name": name, "destination": str(destination),
                "shared_path": str(shared), "source": _rsync_source(host, route, index, name),
                "status": "planned", "link_status": "planned", "size_bytes": None, "sha256": None,
            })

    report["ok"] = not (report["missing_segments"] or report["missing_files"])
    if dry_run:
        return report
    for row in report["files"]:
        destination = Path(row["destination"])
        shared = Path(row["shared_path"])
        try:
            old_file = False
            if not destination.is_symlink() and destination.exists():
                old_file = True
            if old_file and not shared.exists():
                # An old regular file in raw/routes and no shared copy yet. It is
                # checked against the comma first: only a verified copy becomes
                # the shared copy; a different one stays untouched in raw/routes.
                row["status"], error = _copy_file(row["source"], destination)
                if error is None:
                    _move_file(destination, shared)
                    destination.symlink_to(shared)
                    row["link_status"] = "moved_to_shared"
            else:
                # Download into the shared folder, or check the copy already
                # there against the comma, before any link is changed.
                row["status"], error = _copy_file(row["source"], shared)
                if error is None:
                    link_status, error = _adopt_local_file(destination, shared)
                    if link_status is None:
                        link_status, error = _ensure_link(destination, shared)
                    row["link_status"] = link_status
            if error:
                row["status"] = "failed"
                row["error"] = error
                report["ok"] = False
                break
            row["size_bytes"] = shared.stat().st_size
            row["sha256"] = sha256_of(shared)
        except OSError as exc:
            row.update(status="failed", error=str(exc))
            report["ok"] = False
            break
    return report


def link_videos(
    *, route: str | None = None, videos_dir: str | Path | None = None,
    data_root: str | Path | None = None, dry_run: bool = False,
) -> dict[str, Any]:
    """Link segments already in the shared videos folder into raw/routes.

    No network access, and nothing is ever written into the shared folder.
    Every route in the shared folder is linked, or only `route`. A segment
    with only logs and no fcamera.hevc (the fetch keeps just the log of the
    segments it trims) is not linked and is listed in log_only, not an error;
    links an earlier run made for it are removed, because extract-frames
    stops on a segment folder without video. A segment with fcamera.hevc but
    without a full log is reported in missing_files; its other files are
    still linked. Dry-run writes nothing but already reports the conflicts
    the real run would hit.
    """
    if route is not None:
        parse_route_id(route)
    root = _resolve_data_root(data_root)
    videos = resolve_videos_dir(videos_dir)
    _check_roots(root, videos)
    if not videos.is_dir():
        raise SyncError(f"videos_dir_not_found: {videos} (set COMMA_VIDEOS_DIR or use --videos-dir)")

    # Find the segment folders: <route>--<segment> with the segment in decimal,
    # exactly as the comma names them.
    found: list[tuple[str, int, Path]] = []
    ignored: list[str] = []
    for entry in sorted(videos.iterdir()):
        if entry.name.startswith("."):
            continue
        try:
            identity = RouteIdentity.from_segment_dir(entry.name)
        except NnslerManifestError:
            ignored.append(entry.name)
            continue
        if identity.segment_dir != entry.name or entry.is_symlink() or not entry.is_dir():
            ignored.append(entry.name)
            continue
        if route is not None and identity.route_id != route:
            continue
        found.append((identity.route_id, identity.segment_index, entry))
    if route is not None and not found:
        raise SyncError(f"route_not_found: {route} in {videos}")
    found.sort(key=lambda item: (item[0], item[1]))

    report: dict[str, Any] = {
        "videos_dir": str(videos), "data_root": str(root), "route": route, "dry_run": dry_run,
        "segments": [], "files": [], "missing_files": [], "log_only": [], "removed": [],
        "ignored": ignored, "counts": {}, "ok": True, "warning": _videos_dir_warning(videos),
    }
    # Plan and check every link before writing anything.
    for route_id, index, directory in found:
        report["segments"].append(directory.name)
        present = []
        for name in _LINKABLE_NAMES:
            shared = directory / name
            if shared.is_file() and not shared.is_symlink():
                present.append(name)
        segment_dir = root / "raw" / "routes" / route_id / str(index)
        _check_destination(root, segment_dir)
        if "fcamera.hevc" not in present and not (segment_dir / "fcamera.hevc").is_file():
            report["log_only"].append(directory.name)
            _plan_log_only_cleanup(report, route_id, index, segment_dir, directory)
            continue
        # (An old copy of the video in raw/routes keeps the segment usable: its logs are linked.)
        has_full_log = False
        for name in _LOG_NAMES:
            if name in present:
                has_full_log = True
        if not has_full_log:
            report["missing_files"].append({"route": route_id, "segment": index, "name": "rlog"})
        for name in present:
            destination = segment_dir / name
            shared = directory / name
            status, error = _plan_link(destination, shared)
            row = {
                "route": route_id, "segment": index, "name": name,
                "destination": str(destination), "shared_path": str(shared), "status": status,
            }
            if error:
                row["error"] = error
            report["files"].append(row)

    if not dry_run:
        for row in report["files"]:
            destination = Path(row["destination"])
            shared = Path(row["shared_path"])
            try:
                status, error = _adopt_local_file(destination, shared)
                if status is None:
                    status, error = _ensure_link(destination, shared)
            except OSError as exc:
                status = "failed"
                error = str(exc)
            # A conflict on one file must not stop the other segments.
            row["status"] = status
            if error:
                row["error"] = error
            elif "error" in row:
                del row["error"]
        for row in report["removed"]:
            try:
                for link in row["links"]:
                    Path(link).unlink()
                Path(row["path"]).rmdir()
                row["status"] = "removed"
            except OSError as exc:
                row["status"] = "failed"
                row["error"] = str(exc)

    if report["missing_files"]:
        report["ok"] = False
    for row in report["files"]:
        if row["status"] == "failed" or row["status"] == "conflict":
            report["ok"] = False
    for row in report["removed"]:
        if row["status"] == "failed":
            report["ok"] = False

    for row in report["files"]:
        status = row["status"]
        if status in report["counts"]:
            report["counts"][status] = report["counts"][status] + 1
        else:
            report["counts"][status] = 1
    return report
# [nnslr-sync] - END
