#!/usr/bin/env python3
"""Make the comma's time zones, openpilot/sunnypilot/speed_limits/time-zones.json.gz in the sunnypilot clone (default
$NNSLR_SUNNYPILOT_CLONE or /root/sunnypilot), read by brain_inputs.zone_hour for the brain's hour_tz (#81).

The comma may have no time-zone library, so the table carries everything: every zone of tzdata's zone1970.tab with its
countries, the position of its main city, and its UTC offsets (minutes) with the moment each starts, from FIRST_YEAR to
LAST_YEAR, summer time included. Source: this machine's tzdata (public domain), read with Python's zoneinfo. Run it
again when tzdata changes a country's rules, and before LAST_YEAR.

  python scripts/make_time_zones.py [<clone>]
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

TZDATA = Path('/usr/share/zoneinfo')
FIRST_YEAR, LAST_YEAR = 2000, 2050
DAY = 86400
# Countries tzdata does not list, with the country whose zones they keep: Kosovo, on Serbia's time.
ALSO = {'XK': 'RS'}


def degrees(text: str, digits: int) -> float:
    """zone1970.tab's +DDMM[SS] / +DDDMM[SS]."""
    sign = -1 if text[0] == '-' else 1
    whole, rest = int(text[1:1 + digits]), text[1 + digits:]
    minutes = int(rest[:2])
    seconds = int(rest[2:4]) if len(rest) >= 4 else 0
    return sign * (whole + minutes / 60 + seconds / 3600)


def position(text: str) -> tuple[float, float]:
    split = max(text.rfind('+'), text.rfind('-'))
    return round(degrees(text[:split], 2), 4), round(degrees(text[split:], 3), 4)


def offsets(name: str) -> list[list[int]]:
    """[unix second it starts, UTC offset in minutes], the first at FIRST_YEAR."""
    zone = ZoneInfo(name)

    def minutes(t: int) -> int:
        return int(datetime.fromtimestamp(t, zone).utcoffset().total_seconds() // 60)

    start = int(datetime(FIRST_YEAR, 1, 1, tzinfo=timezone.utc).timestamp())
    end = int(datetime(LAST_YEAR + 1, 1, 1, tzinfo=timezone.utc).timestamp())
    changes = [[start, minutes(start)]]
    t = start
    while t < end:
        if minutes(t + DAY) != changes[-1][1]:
            low, high = t, t + DAY  # the change is after low, at or before high
            while high - low > 1:
                middle = (low + high) // 2
                if minutes(middle) == changes[-1][1]:
                    low = middle
                else:
                    high = middle
            changes.append([high, minutes(high)])
        t += DAY
    return changes


def main() -> int:
    clone = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('NNSLR_SUNNYPILOT_CLONE', '/root/sunnypilot'))
    output = clone / 'openpilot/sunnypilot/speed_limits/time-zones.json.gz'
    zones = []
    for line in (TZDATA / 'zone1970.tab').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        codes, coordinates, name = line.split('\t')[:3]
        latitude, longitude = position(coordinates)
        codes = codes + ''.join(f',{extra}' for extra, of in ALSO.items() if of in codes.split(','))
        zones.append({'zone': name, 'countries': codes.split(','), 'position': [latitude, longitude],
                      'offsets': offsets(name)})
    version = (TZDATA / 'tzdata.zi').read_text().split('\n', 1)[0].removeprefix('# version ').strip() \
        if (TZDATA / 'tzdata.zi').exists() else None
    payload = {'source': 'tzdata zone1970.tab', 'tzdata': version, 'license': 'public domain',
               'years': [FIRST_YEAR, LAST_YEAR], 'zones': zones}
    output.write_bytes(gzip.compress(json.dumps(payload, separators=(',', ':')).encode(), mtime=0))
    print(f'{len(zones)} zones of {len({c for z in zones for c in z["countries"]})} countries, tzdata {version} -> '
          f'{output} ({output.stat().st_size // 1024} KiB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
