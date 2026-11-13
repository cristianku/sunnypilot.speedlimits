"""State of every route, from the shared comma videos to the dataset, and the next thing to do.

Used by helpers/status.sh (table) and helpers/1_prepare.sh (which routes still need work).
Standard library only, read-only: it never writes in the data root or in the review database.

  python -m helpers.route_status                    (table + next step)
  python -m helpers.route_status --todo=extract     (one route per line: still to extract)
  python -m helpers.route_status --todo=preannotate (extracted, without a pre-annotation report)
  python -m helpers.route_status --todo=repropose   (with a report and frames still to confirm: Propose again gives
                                                    those new proposals, the confirmed ones stay)
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3

REPO = Path(__file__).resolve().parents[1]
# <route>--<segment> as on the comma, e.g. 0000011e--0c8825628a--14
SEGMENT_DIR = re.compile(r"^([0-9a-f]{8}--[0-9a-f]{10})--(\d+)$")
ROUTE_ID = re.compile(r"^[0-9a-f]{8}--[0-9a-f]{10}$")
RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")
# nnslr build-splits makes three disjoint partitions (train, validation, test) of independent
# route/site groups: fewer reviewed routes cannot be split.
MIN_REVIEWED_ROUTES = 3

# State of a route, in pipeline order, with the helper that moves it forward.
# The keys stay as they are (the JSON of the site and of --json uses them); the words are English.
STATES = {
    "collegare": ("videos to link", "helpers/1_prepare.sh"),
    "estrarre": ("pictures to take out", "helpers/1_prepare.sh"),
    "preannotare": ("signs to propose", "helpers/1_prepare.sh"),
    "revisionare": ("to review", "helpers/2_review.sh"),
    "importare": ("new reviews", "helpers/3_train.sh"),
    "pronta": ("in the dataset", ""),
}


def data_root() -> Path:
    return Path(os.environ.get("NNSLR_DATA_ROOT") or REPO / "data").expanduser().resolve()


def videos_dir() -> Path:
    return Path(os.environ.get("COMMA_VIDEOS_DIR") or REPO.parent / "comma-videos").expanduser().resolve()


def shared_segments(videos: Path) -> dict[str, set[int]]:
    """Segments with a front camera video of every route in the shared folder (only real
    folders, as nnslr link-videos; a folder still holding a *.part file, a download running
    or cut off, is left out as link-videos leaves it out)."""
    found: dict[str, set[int]] = {}
    if not videos.is_dir():
        return found
    for entry in videos.iterdir():
        match = SEGMENT_DIR.match(entry.name)
        if (match and entry.is_dir() and not entry.is_symlink() and (entry / "fcamera.hevc").is_file()
                and not any(item.name.endswith(".part") for item in entry.iterdir())):
            found.setdefault(match[1], set()).add(int(match[2]))
    return found


def _numbered_dirs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [entry for entry in directory.iterdir() if entry.is_dir() and entry.name.isdigit()]


def latest_run(root: Path, route: str) -> str | None:
    pointer = root / "derived/preannotations" / route / "latest.json"
    try:
        run_id = json.loads(pointer.read_text()).get("run_id")
    except (OSError, ValueError, AttributeError):
        return None
    return run_id if isinstance(run_id, str) and RUN_ID.fullmatch(run_id) else None


def _line_count(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def reviewed_routes(root: Path) -> set[str] | None:
    """The routes with any review row, on any of their runs (confirmed or not); None when the database cannot be
    read, so that nothing is proposed again on a guess."""
    database = root / "annotations/review.sqlite"
    if not database.is_file():
        return set()
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            return {route for (route,) in connection.execute("SELECT DISTINCT route_id FROM frame_reviews")}
        finally:
            connection.close()
    except sqlite3.Error:
        return None


def review_counts(root: Path, since: str | None) -> tuple[Counter, Counter]:
    """Confirmed frames per (route, run), and those confirmed or changed after `since` (ISO UTC)."""
    confirmed: Counter = Counter()
    newer: Counter = Counter()
    database = root / "annotations/review.sqlite"
    if not database.is_file():
        return confirmed, newer
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            # The frames the site's AI reviewer saved wait for a person (nnslr_tools.review_db.SITE_AI_REVIEWER).
            "SELECT route_id, run_id, updated_at FROM frame_reviews WHERE confirmed = 1 "
            "AND NOT (reviewer_kind = 'ai' AND reviewer GLOB 'ai/*/*')").fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        connection.close()
    for route, run, updated_at in rows:
        confirmed[(route, run)] += 1
        # updated_at is ISO UTC with milliseconds and "Z": comparable as text.
        if since is None or updated_at > since:
            newer[(route, run)] += 1
    return confirmed, newer


def dataset_objects(root: Path) -> tuple[Counter, str | None]:
    """Objects per route in the gold dataset, and the time of its last change (ISO UTC)."""
    objects: Counter = Counter()
    dataset = root / "annotations/objects.jsonl"
    if not dataset.is_file():
        return objects, None
    with dataset.open() as handle:
        for line in handle:
            try:
                objects[json.loads(line).get("route_id")] += 1
            except (ValueError, AttributeError):
                continue
    changed = datetime.fromtimestamp(dataset.stat().st_mtime, timezone.utc)
    return objects, changed.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def route_rows(root: Path, videos: Path) -> list[dict]:
    shared = shared_segments(videos)
    routes = set(shared)
    for folder in ("raw/routes", "derived/frames", "derived/preannotations"):
        base = root / folder
        if base.is_dir():
            routes.update(entry.name for entry in base.iterdir() if ROUTE_ID.fullmatch(entry.name))
    objects, dataset_changed = dataset_objects(root)
    confirmed, newer = review_counts(root, dataset_changed)

    rows = []
    for route in sorted(routes):
        linked = {int(d.name) for d in _numbered_dirs(root / "raw/routes" / route) if (d / "fcamera.hevc").exists()}
        frame_dirs = [d for d in _numbered_dirs(root / "derived/frames" / route) if (d / "frames.jsonl").is_file()]
        run = latest_run(root, route)
        report_frames = _line_count(root / "derived/preannotations" / route / run / "preannotations.jsonl") if run else 0
        available = shared.get(route, set()) | linked
        row = {
            "route": route,
            "segments": len(available),
            "unlinked": len(shared.get(route, set()) - linked),
            "linked": len(linked),
            "extracted": len(frame_dirs),
            "frames": sum(_line_count(d / "frames.jsonl") for d in frame_dirs),
            "run_id": run,
            "report_frames": report_frames,
            "confirmed": confirmed.get((route, run), 0) if run else 0,
            "new_reviews": newer.get((route, run), 0) if run else 0,
            "objects": objects.get(route, 0),
        }
        row["state"] = route_state(row)
        rows.append(row)
    return rows


def route_state(row: dict) -> str:
    # A pre-annotated route is prepared again only by Propose again (helpers/1_prepare.sh --again), which copies
    # its confirmed frames onto the new report. Segments that show up later stay unused (see print_table).
    if row["run_id"] is not None:
        if row["confirmed"] < row["report_frames"]:
            return "revisionare"
        if row["new_reviews"]:
            return "importare"
        return "pronta"
    if row["unlinked"] or row["linked"] == 0:
        return "collegare"
    if row["extracted"] < row["linked"]:
        return "estrarre"
    return "preannotare"


def next_step(rows: list[dict]) -> str:
    states = {row["state"] for row in rows}
    if not rows:
        return "no drives yet: download the videos with helpers/fetch_comma_segments.sh"
    for state in ("collegare", "estrarre", "preannotare", "revisionare"):
        if state in states:
            return STATES[state][1]
    reviewed = sum(1 for row in rows if row["confirmed"])
    if reviewed < MIN_REVIEWED_ROUTES:
        return (f"training needs at least {MIN_REVIEWED_ROUTES} reviewed routes (now {reviewed}): "
                "download more drives with helpers/fetch_comma_segments.sh, then helpers/1_prepare.sh")
    if "importare" in states:
        return STATES["importare"][1]
    return "all ready: helpers/3_train.sh to train on the current dataset"


def print_table(rows: list[dict]) -> None:
    print(f"{'route':22} {'segments':>9} {'frames':>6} {'reviewed':>12} {'objects':>8}  state")
    for row in rows:
        segments = f"{row['extracted']}/{row['segments']}"
        reviewed = f"{row['confirmed']}/{row['report_frames']}" if row["run_id"] else "-"
        state = STATES[row["state"]][0]
        if row["state"] == "importare":
            state += f" ({row['new_reviews']} to import)"
        if row["run_id"] is not None and row["segments"] > row["extracted"]:
            state += f", {row['segments'] - row['extracted']} segments that came later, not used"
        print(f"{row['route']:22} {segments:>9} {row['frames']:>6} {reviewed:>12} {row['objects']:>8}  {state}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--todo", choices=("extract", "preannotate", "repropose"),
                        help="print only the routes that still need this step, one per line")
    parser.add_argument("--json", action="store_true", help="every route as JSON")
    args = parser.parse_args()
    rows = route_rows(data_root(), videos_dir())
    if args.todo == "repropose":
        # A drive with frames still to confirm (its confirmed frames stay, nnslr preannotate --keep-reviews); none
        # when the review database cannot be read, so that nothing is proposed again on a guess.
        reviewed = reviewed_routes(data_root())
        for row in rows:
            if reviewed is not None and row["run_id"] and row["confirmed"] < row["report_frames"]:
                print(row["route"])
    elif args.todo:
        wanted = "estrarre" if args.todo == "extract" else "preannotare"
        for row in rows:
            if row["state"] == wanted:
                print(row["route"])
    elif args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows)
        print()
        print(f"Next step: {next_step(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
