"""One segment through the localizer of openpilot.distill, for nnslr_tools.localized.

    <distill python> localize_child.py <openpilot.distill> <segment folder> <output folder>

Run by the Python of openpilot.distill (its .venv: laika, numpy, safetensors, openpilot's LogReader), never
imported by nnslr_tools. The segment folder holds the comma's rlog.zst, fcamera.hevc and ecamera.hevc (read only);
the output folder gets localizer.safetensors, frame_info.safetensors and localizer.ionex, as distill's
localizer.run writes them next to its own segments. Nothing is written into openpilot.distill.

Exit 3 (distill's UNUSABLE_SEGMENT_EXIT_CODE): the segment cannot be localized because of its data (car never
above 18 km/h, no GPS fix, too few satellites, no calibration yet, incomplete); trying again does not help.
"""
import sys
import tempfile
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print('usage: localize_child.py <openpilot.distill> <segment folder> <output folder>', file=sys.stderr)
        return 2
    distill, segment, output = Path(argv[1]), Path(argv[2]), Path(argv[3])
    sys.path.insert(0, str(distill))
    from openpilot.tools.lib.logreader import LogReader
    from safetensors.numpy import save_file

    from localizer.errors import UNUSABLE_SEGMENT_EXIT_CODE, UnusableSegment
    from localizer.ionex import write_ionex_file
    from localizer.localizer import localize
    from localizer.logs import FRAME_INFO_METADATA, read_log

    try:
        localization_input, frame_info = read_log(
            LogReader(str(segment / 'rlog.zst'), sort_by_time=True, only_union_types=True), segment)
        localization, ionex_used = localize(localization_input)
    except UnusableSegment as error:
        print(f'UnusableSegment: {error}', file=sys.stderr)
        return UNUSABLE_SEGMENT_EXIT_CODE
    output.mkdir(parents=True, exist_ok=True)
    # Written beside, then renamed: a folder holds the three files of one run, or none.
    with tempfile.TemporaryDirectory(dir=output, prefix='.run-') as scratch:
        scratch = Path(scratch)
        save_file(localization, scratch / 'localizer.safetensors', metadata={'schema_version': '1'})
        save_file(frame_info, scratch / 'frame_info.safetensors', metadata=FRAME_INFO_METADATA)
        write_ionex_file(scratch, ionex_used)
        for name in ('frame_info.safetensors', 'localizer.ionex', 'localizer.safetensors'):
            (scratch / name).replace(output / name)
    print(f'frames={localization["frame_states"].shape[0]} ionex={ionex_used}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
