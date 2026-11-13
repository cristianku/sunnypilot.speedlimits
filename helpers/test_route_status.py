"""Offline tests for the route states used by the easy helpers (temporary folders only)."""
from __future__ import annotations

import calendar
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

from helpers import route_status as status

ROUTE = "0000011e--0c8825628a"
RUN = "20261021T080000Z-0123abcd"


class RouteStatusTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        base = Path(self.directory.name)
        self.videos = base / "comma-videos"
        self.root = base / "data"
        self.videos.mkdir()
        self.root.mkdir()

    def tearDown(self):
        self.directory.cleanup()

    def shared(self, *segments):
        for segment in segments:
            folder = self.videos / f"{ROUTE}--{segment}"
            folder.mkdir()
            (folder / "fcamera.hevc").write_bytes(b"video")

    def linked(self, *segments):
        for segment in segments:
            folder = self.root / "raw/routes" / ROUTE / str(segment)
            folder.mkdir(parents=True)
            (folder / "fcamera.hevc").symlink_to(self.videos / f"{ROUTE}--{segment}" / "fcamera.hevc")

    def extracted(self, *segments, frames=60):
        for segment in segments:
            folder = self.root / "derived/frames" / ROUTE / str(segment)
            folder.mkdir(parents=True)
            (folder / "frames.jsonl").write_text("{}\n" * frames)

    def report(self, frames):
        base = self.root / "derived/preannotations" / ROUTE
        (base / RUN).mkdir(parents=True)
        (base / RUN / "preannotations.jsonl").write_text("{}\n" * frames)
        (base / "latest.json").write_text(json.dumps({"run_id": RUN}))

    def reviews(self, count, updated_at="2026-10-21T09:00:00.000Z"):
        database = self.root / "annotations/review.sqlite"
        database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database)
        connection.execute("CREATE TABLE IF NOT EXISTS frame_reviews (route_id TEXT, run_id TEXT, frame_key TEXT, "
                           "confirmed INTEGER, updated_at TEXT)")
        connection.executemany("INSERT INTO frame_reviews VALUES (?, ?, ?, 1, ?)",
                               [(ROUTE, RUN, f"frame-{i}-{updated_at}", updated_at) for i in range(count)])
        connection.commit()
        connection.close()

    def dataset(self, objects, changed_at):
        dataset = self.root / "annotations/objects.jsonl"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_text("".join(json.dumps({"route_id": ROUTE}) + "\n" for _ in range(objects)))
        stamp = calendar.timegm(time.strptime(changed_at, "%Y-%m-%dT%H:%M:%S"))
        os.utime(dataset, (stamp, stamp))

    def row(self):
        rows = status.route_rows(self.root, self.videos)
        self.assertEqual([r["route"] for r in rows], [ROUTE])
        return rows[0]

    def test_pipeline_states(self):
        self.shared(1, 2)
        self.assertEqual(self.row()["state"], "collegare")
        self.linked(1, 2)
        self.assertEqual(self.row()["state"], "estrarre")
        self.extracted(1)
        self.assertEqual(self.row()["state"], "estrarre")
        self.extracted(2)
        self.assertEqual(self.row()["state"], "preannotare")
        self.report(120)
        self.assertEqual(self.row()["state"], "revisionare")
        self.reviews(120)
        row = self.row()
        self.assertEqual((row["state"], row["confirmed"], row["new_reviews"]), ("importare", 120, 120))

    def test_a_cut_off_download_does_not_hold_the_drive(self):
        self.shared(1, 2)
        (self.videos / f"{ROUTE}--2" / "ecamera.hevc.part").write_bytes(b"")
        self.linked(1)
        row = self.row()
        self.assertEqual((row["state"], row["unlinked"], row["segments"]), ("estrarre", 0, 1))

    def test_reviews_after_the_import_are_new(self):
        self.shared(1)
        self.linked(1)
        self.extracted(1)
        self.report(2)
        self.reviews(2, "2026-10-21T09:00:00.000Z")
        self.dataset(3, "2026-10-21T10:00:00")
        self.assertEqual(self.row()["state"], "pronta")
        self.reviews(1, "2026-10-21T11:00:00.000Z")
        row = self.row()
        self.assertEqual((row["state"], row["new_reviews"], row["objects"]), ("importare", 1, 3))

    def test_reported_route_is_never_reprocessed(self):
        self.shared(1, 2, 3)
        self.linked(1)
        self.extracted(1)
        self.report(60)
        row = self.row()
        self.assertEqual((row["state"], row["segments"], row["extracted"]), ("revisionare", 3, 1))
        rows = status.route_rows(self.root, self.videos)
        self.assertNotIn(ROUTE, [r["route"] for r in rows if r["state"] in ("estrarre", "preannotare")])

    def test_shared_segment_without_video_is_ignored(self):
        (self.videos / f"{ROUTE}--1").mkdir()
        (self.videos / "not-a-segment").mkdir()
        self.assertEqual(status.route_rows(self.root, self.videos), [])

    def test_next_step(self):
        self.assertIn("fetch_comma_segments", status.next_step([]))
        self.assertEqual(status.next_step([{"state": "revisionare"}, {"state": "estrarre"}]), "helpers/1_prepare.sh")
        self.assertEqual(status.next_step([{"state": "revisionare"}]), "helpers/2_review.sh")
        self.assertIn(f"at least {status.MIN_REVIEWED_ROUTES} reviewed routes (now 1)",
                      status.next_step([{"state": "importare", "confirmed": 3}]))
        five = [{"state": "pronta", "confirmed": 1}] * 4
        self.assertIn("all ready", status.next_step(five + [{"state": "pronta", "confirmed": 1}]))
        self.assertEqual(status.next_step(five + [{"state": "importare", "confirmed": 1}]), "helpers/3_train.sh")


if __name__ == "__main__":
    unittest.main()
