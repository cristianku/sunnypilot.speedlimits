"""A frame manifest written on another machine (#77): the GPU runner reads on CT 131 manifests whose pictures were
extracted on the site's machine, with absolute paths of its data root."""
from pathlib import Path

from nnslr_tools.preannotate import _in_this_root


def test_a_picture_of_another_data_root_is_read_in_this_one(tmp_path):
    root = tmp_path / 'data'
    there = Path('/root/smartspeed/data/derived/frames/00000003--effdc23b0d/0/frame_00000000.png')
    assert _in_this_root(root, there) == root / 'derived/frames/00000003--effdc23b0d/0/frame_00000000.png'
    assert _in_this_root(root, Path('derived/frames/r/0/f.png')) == root / 'derived/frames/r/0/f.png'
    here = root / 'derived/frames/r/0/f.png'
    assert _in_this_root(root, here) == here
    # Not a frame of derived/frames: left as it is, and _inside refuses it.
    assert _in_this_root(root, Path('/etc/passwd')) == Path('/etc/passwd')
