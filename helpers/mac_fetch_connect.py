#!/usr/bin/env python3
"""A PUBLIC comma connect drive downloaded on a computer with a free internet line (Cristian's Mac), into
~/Downloads/connect/<route>--<segment>/ with fcamera.hevc, ecamera.hevc and rlog.bz2: the layout of the shared folder
of the comma files. Standard library only, no account. Files already there are kept (run it again after a cut).
The folders reach the server by USB stick, by hand; on the server, helpers/fetch_connect_route.sh is the same download.
usage: python3 helpers/mac_fetch_connect.py <dongle> <route>
   e.g. python3 helpers/mac_fetch_connect.py 35edd31af7e86e74 00000003--effdc23b0d
"""
import json, os, re, sys, urllib.parse, urllib.request
from pathlib import Path

dongle, route = sys.argv[1], sys.argv[2]
api = 'https://api.commadotai.com/v1/route/' + urllib.parse.quote(f'{dongle}|{route}', safe='') + '/files'
files = json.load(urllib.request.urlopen(api, timeout=60))
out = Path.home() / 'Downloads' / 'connect'
names = {'fcamera.hevc', 'ecamera.hevc', 'rlog.bz2'}
jobs = []
for kind in ('cameras', 'ecameras', 'logs'):
    for url in files.get(kind) or ():
        m = re.search(r'/(\d+)/([^/]+)$', urllib.parse.urlsplit(url).path)
        if m and m.group(2) in names:
            jobs.append((int(m.group(1)), m.group(2), url))
jobs.sort()
for i, (seg, name, url) in enumerate(jobs, 1):
    target = out / f'{route}--{seg}' / name
    if target.is_file():
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(name + '.part')
    with urllib.request.urlopen(url, timeout=120) as answer, open(part, 'wb') as f:
        while chunk := answer.read(1 << 20):
            f.write(chunk)
    os.replace(part, target)
    print(f'{i}/{len(jobs)} {route}--{seg}/{name}', flush=True)
print('done:', out)
