"""connect_fetch: a drive shared in comma connect, downloaded from a recorded API answer (no network)."""
from __future__ import annotations

import bz2
import io
import json
import random
from pathlib import Path

import pytest

from nnslr_tools import connect_fetch
from nnslr_tools.connect_fetch import ConnectError, fetch, outside_drive, parse_link, route_files
from nnslr_tools.speed_review import _brain_drive

DONGLE = "0123456789abcdef"
ROUTE = "0000012a--3f9c1e0b2d"
SIG = "?sv=2020-08-04&se=2026-11-13&sig=SECRET"
BLOB = f"https://commadata2.blob.core.windows.net/commadata2/{DONGLE}/{ROUTE}"
META = {"fullname": f"{DONGLE}|{ROUTE}", "dongle_id": DONGLE, "is_public": True, "platform": "VOLKSWAGEN_GOLF_MK7",
        "make": "volkswagen", "version": "0.10.1", "git_branch": "smartspeed", "vin": "WVWZZZ", "user_id": "u1",
        "start_lat": 47.1, "start_lng": 8.5}
# Segment 1 was never uploaded in full: only its qcamera and qlog are in connect.
FILES = {
    "cameras": [f"{BLOB}/0/fcamera.hevc{SIG}", f"{BLOB}/2/fcamera.hevc{SIG}"],
    "ecameras": [f"{BLOB}/0/ecamera.hevc{SIG}", f"{BLOB}/2/ecamera.hevc{SIG}"],
    "dcameras": [f"{BLOB}/0/dcamera.hevc{SIG}", f"{BLOB}/2/dcamera.hevc{SIG}"],
    "logs": [f"{BLOB}/0/rlog.bz2{SIG}", f"{BLOB}/2/rlog.zst{SIG}",
             f"https://commadata2.blob.core.windows.net/commadata2/fedcba9876543210/{ROUTE}/3/rlog.zst{SIG}"],
    "qcameras": [f"{BLOB}/{n}/qcamera.ts{SIG}" for n in range(3)],
    "qlogs": [f"{BLOB}/{n}/qlog.zst{SIG}" for n in range(3)],
}


def fake_api(meta=META, files=FILES):
    calls = []

    def api(path):
        calls.append(path)
        assert path.startswith(f"/v1/route/{DONGLE}%7C{ROUTE}/")
        return files if path.endswith("/files") else meta
    api.calls = calls
    return api


def fake_download(log):
    def download(url, target, repack):
        assert "sig=" in url
        log.append((url, target.name, repack))
        target.write_bytes(f"{Path(url.split('?')[0]).name}:{repack}".encode())
        return 10
    return download


@pytest.fixture
def roots(tmp_path):
    videos = tmp_path / "comma videos"
    videos.mkdir()
    return videos, tmp_path / "data"


def test_parse_link():
    assert parse_link(f"https://connect.comma.ai/{DONGLE}/{ROUTE}") == (DONGLE, ROUTE)
    assert parse_link(f"https://connect.comma.ai/{DONGLE.upper()}/{ROUTE}/120/240") == (DONGLE, ROUTE)
    assert parse_link(f"{DONGLE}|{ROUTE}") == (DONGLE, ROUTE)
    assert parse_link(f" {DONGLE}%7C{ROUTE} ") == (DONGLE, ROUTE)
    with pytest.raises(ConnectError, match="old_route_name"):
        parse_link(f"{DONGLE}|2023-07-27--13-01-19")
    with pytest.raises(ConnectError, match="no_link"):
        parse_link("https://connect.comma.ai/")


def test_route_files_reads_the_segment_from_the_path():
    found = route_files(DONGLE, ROUTE, fake_api())
    # No segment 1 (not uploaded), no other dongle's segment 3, no driver camera, qcamera or qlog.
    assert sorted(found) == [0, 2]
    assert sorted(found[0]) == ["ecamera.hevc", "fcamera.hevc", "rlog.bz2"]
    assert sorted(found[2]) == ["ecamera.hevc", "fcamera.hevc", "rlog.zst"]


def test_fetch_downloads_links_and_marks_the_drive(roots):
    videos, root = roots
    log, lines = [], []
    report = fetch(f"https://connect.comma.ai/{DONGLE}/{ROUTE}", videos_dir=videos, data_root=root, say=lines.append,
                   api=fake_api(), download=fake_download(log))
    assert report["ok"] and report["segments"] == [0, 2] and report["missing"] == []
    assert sorted(path.name for path in (videos / f"{ROUTE}--0").iterdir()) == ["ecamera.hevc", "fcamera.hevc", "rlog.zst"]
    # The bz2 rlog is re-packed, the zst one copied as it is; no driver camera.
    assert (videos / f"{ROUTE}--0" / "rlog.zst").read_bytes() == b"rlog.bz2:True"
    assert (videos / f"{ROUTE}--2" / "rlog.zst").read_bytes() == b"rlog.zst:False"
    assert not any(name == "dcamera.hevc" for _, name, _ in log)
    for segment in (0, 2):
        link = root / "raw" / "routes" / ROUTE / str(segment) / "fcamera.hevc"
        assert link.is_symlink() and link.resolve() == (videos / f"{ROUTE}--{segment}" / "fcamera.hevc").resolve()
    origin = outside_drive(root, ROUTE)
    assert origin["dongle_id"] == DONGLE and origin["platform"] == "VOLKSWAGEN_GOLF_MK7"
    assert origin["segments"] == {"0": ["ecamera.hevc", "fcamera.hevc", "rlog.zst"],
                                  "2": ["ecamera.hevc", "fcamera.hevc", "rlog.zst"]}
    # Nothing of the owner's account, car or position, and no signature anywhere.
    kept = json.dumps(origin) + json.dumps(report) + "\n".join(lines)
    for secret in ("vin", "WVWZZZ", "user_id", "47.1", "sig=", "SECRET"):
        assert secret not in kept

    # A second run downloads nothing and keeps the record.
    again = fetch(f"{DONGLE}|{ROUTE}", videos_dir=videos, data_root=root, say=lines.append,
                  api=fake_api(), download=fake_download(log2 := []))
    assert log2 == [] and len(again["kept"]) == 6
    assert outside_drive(root, ROUTE)["first_download"] == origin["first_download"]


def test_missing_files_are_named(roots):
    videos, root = roots
    files = {**FILES, "ecameras": FILES["ecameras"][:1], "logs": FILES["logs"][:1]}
    lines = []
    report = fetch(f"{DONGLE}|{ROUTE}", videos_dir=videos, data_root=root, dry_run=True, say=lines.append,
                   api=fake_api(files=files), download=fake_download([]))
    assert report["missing"] == ["2/ecamera.hevc", "2/rlog"]
    assert "Upload all files" in lines[-1]
    assert not (root / "raw").exists() and list(videos.iterdir()) == []


def test_a_drive_that_is_not_public_stops(roots):
    videos, root = roots
    with pytest.raises(ConnectError, match="not_public"):
        fetch(f"{DONGLE}|{ROUTE}", videos_dir=videos, data_root=root, say=print,
              api=fake_api(meta={**META, "is_public": False}), download=fake_download([]))
    with pytest.raises(ConnectError, match="nothing_uploaded"):
        fetch(f"{DONGLE}|{ROUTE}", videos_dir=videos, data_root=root, say=print,
              api=fake_api(files={"qlogs": FILES["qlogs"]}), download=fake_download([]))


def test_a_route_of_our_comma_is_never_touched(roots):
    videos, root = roots
    (videos / f"{ROUTE}--0").mkdir()
    with pytest.raises(ConnectError, match="route_of_our_comma"):
        fetch(f"{DONGLE}|{ROUTE}", videos_dir=videos, data_root=root, say=print, api=fake_api(),
              download=fake_download([]))


def test_an_expired_link_is_asked_again_once(roots):
    videos, root = roots
    api, seen = fake_api(), []

    def download(url, target, repack):
        seen.append(url)
        if len(seen) == 1:
            raise connect_fetch._Expired()
        target.write_bytes(b"x")
        return 1
    report = fetch(f"{DONGLE}|{ROUTE}", videos_dir=videos, data_root=root, say=print, api=api, download=download)
    assert report["ok"] and sum(path.endswith("/files") for path in api.calls) == 2


def test_download_repacks_bz2_as_zstd(tmp_path, monkeypatch):
    zstandard = pytest.importorskip("zstandard")
    payload = random.Random(0).randbytes(200_000)
    monkeypatch.setattr(connect_fetch, "_open", lambda url: io.BytesIO(bz2.compress(payload)))
    target = tmp_path / "rlog.zst"
    connect_fetch._download(f"{BLOB}/0/rlog.bz2{SIG}", target, True)
    assert zstandard.ZstdDecompressor().stream_reader(io.BytesIO(target.read_bytes())).read() == payload
    assert not (tmp_path / "rlog.zst.part").exists()

    # A cut bz2 stream leaves no file behind.
    monkeypatch.setattr(connect_fetch, "_open", lambda url: io.BytesIO(bz2.compress(payload)[:2000]))
    with pytest.raises(ConnectError, match="ended before"):
        connect_fetch._download(f"{BLOB}/0/rlog.bz2{SIG}", tmp_path / "other.zst", True)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["rlog.zst"]


def test_outside_drives_are_not_brain_drives(tmp_path):
    class Drives:
        root = tmp_path

        def summary(self, route):
            return {"meta": {"car": {"passive": True}}, "coverage": {"engaged": False}}
    assert _brain_drive(Drives(), ROUTE) is True
    (tmp_path / "raw" / "connect").mkdir(parents=True)
    (tmp_path / "raw" / "connect" / f"{ROUTE}.json").write_text(json.dumps({"route": ROUTE}))
    assert _brain_drive(Drives(), ROUTE) is False
