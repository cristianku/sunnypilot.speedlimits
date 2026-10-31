"""Keep every test away from the real folders next to the clone."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def private_data_folders(tmp_path, monkeypatch):
    # The defaults are <repo>/data and <repo>/../comma-videos, which on the
    # training container is /root/comma-videos with the real drives. Tests that
    # check the defaults remove these variables themselves (monkeypatch.delenv).
    monkeypatch.setenv("COMMA_VIDEOS_DIR", str(tmp_path / "comma-videos"))
    monkeypatch.setenv("NNSLR_DATA_ROOT", str(tmp_path / "data"))
    # openpilot.distill next to the clone has real localizer tracks and the Python that runs the localizer.
    monkeypatch.setenv("NNSLR_DISTILL_ROOT", str(tmp_path / "openpilot.distill"))
