#!/usr/bin/env python3
"""Read-only country replay of local logs around observed cached GPS transitions.

No downloads, inference, annotations, imports into the dataset, or device access.
Requires an explicit experimental policy and an explicit offline border source.
Emits aggregate statistics only; it does not export positions or private route IDs.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
BASE = '8d37cf946'
COUNTRY_PATH = 'openpilot/sunnypilot/speed_limits/country.py'


def transition_segments(rows):
    """Candidate transitions, not independently validated border crossings."""
    result = set()
    previous = None
    for row in sorted(rows, key=lambda r: (r['segment_index'], r['output_index'])):
        cc = row.get('country')
        if cc is None:
            previous = None
            continue
        if previous is not None and previous['country'] != cc:
            result.update((previous['segment_index'], row['segment_index']))
        previous = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--runtime-repo', type=Path, default=Path('/root/sunnypilot'))
    parser.add_argument('--border-source', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True, help='explicit experimental CountryPolicy JSON')
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    patch = REPO/'patches/speed-limits-unit-normalization/runtime.patch'
    # All materialization is ordinary temporary files, never a git checkout/worktree.
    with tempfile.TemporaryDirectory(prefix='country-replay-') as directory:
        target = Path(directory)
        for name in ('openpilot/sunnypilot/speed_limits/vision.py',
                     'openpilot/sunnypilot/speed_limits/speedvisiond.py',
                     'openpilot/cereal/custom.capnp',
                     'openpilot/sunnypilot/speed_limits/tests/test_vision.py',
                     'openpilot/sunnypilot/speed_limits/tests/test_speed_vision_sp.py',
                     'openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_resolver.py'):
            p=target/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(subprocess.check_output(['git','-C',str(args.runtime_repo),'show',f'{BASE}:{name}']))
        subprocess.run(['patch','--batch','-p1','-d',str(target),'-i',str(patch)],check=True,capture_output=True)
        spec=importlib.util.spec_from_file_location('replay_country',target/COUNTRY_PATH)
        country=importlib.util.module_from_spec(spec);sys.modules[spec.name]=country;spec.loader.exec_module(country)
        borders=country.OfflineBorders.load(args.border_source)
        if not borders.countries:
            parser.error('border source unavailable or invalid')
        country.CountryPolicy(**policy)  # validate before opening logs
        sys.path.insert(0,str(REPO/'helpers'))
        import comma_fetch_log
        def no_network(url):
            raise RuntimeError('pinned local log schema unavailable; downloads are disabled')
        comma_fetch_log._pinned_schema(fetch=no_network)
        # Freeze selection to the verified pinned local schema; never fallback to downloading.
        schema_folder=comma_fetch_log._pinned_schema(fetch=no_network)
        comma_fetch_log._schema_dirs=lambda:(schema_folder,schema_folder)
        totals=collections.Counter(); sources=collections.Counter(); results=[]
        accuracy=[]; gaps=[]; available_countries=collections.Counter()
        for path in sorted((args.data_root/'derived/gps').glob('*/positions.jsonl')):
            rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            available_countries.update(r.get('country') or 'unknown' for r in rows)
            segments=transition_segments(rows)
            if not segments:continue
            selected=sorted({s+d for s in segments for d in (-1,0,1) if s+d>=0})
            resolver=country.CountryResolver(borders,country.CountryPolicy(**policy))
            states=collections.Counter(); transitions=collections.Counter(); previous=None; last_time=None
            for segment in selected:
                folder=args.data_root/'raw/routes'/path.parent.name/str(segment)
                log=next((folder/n for n in ('rlog.zst','rlog.bz2','rlog','qlog.zst','qlog.bz2','qlog') if (folder/n).is_file()),None)
                if log is None:
                    totals['missing_logs']+=1;continue
                totals['logs']+=1
                for event in comma_fetch_log._messages(log):
                    if event.which() not in ('gpsLocationExternal','gpsLocation'):continue
                    service=event.which();sources[service]+=1
                    g=getattr(event,service);t=event.logMonoTime*1e-9
                    fix=country.GpsFix(t,g.latitude,g.longitude,g.horizontalAccuracy,bool(event.valid and g.hasFix))
                    if last_time is not None and t<=last_time:
                        totals['out_of_order_or_duplicate']+=1
                    if last_time is not None and 0<t-last_time<60:gaps.append(t-last_time)
                    last_time=t
                    if fix.valid:accuracy.append(fix.accuracy)
                    state=resolver.update(t,fix)
                    states[state or 'unknown']+=1
                    if state and previous and state!=previous:transitions[previous+'>'+state]+=1
                    if state:previous=state
            results.append({'country_states':dict(states),'confirmed_transitions':dict(transitions)})
            print(f'country replay: {len(results)} candidate routes read',file=sys.stderr,flush=True)
        def quantiles(values):
            values=sorted(values)
            return {str(q):values[min(len(values)-1,int((len(values)-1)*q))] for q in (.5,.95,.99)} if values else {}
        print(json.dumps({'base':BASE,'patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest(),
              'border_sha256':hashlib.sha256(args.border_source.read_bytes()).hexdigest(),
              'policy':policy,'status':'experimental; not validated driving thresholds',
              'cached_frame_countries':dict(available_countries),'gps_services':dict(sources),
              'totals':dict(totals),'routes':results,'gps_accuracy_m_quantiles':quantiles(accuracy),
              'gps_sample_gap_s_quantiles':quantiles(gaps),
              'limitations':['selected crossing windows only','no GB/US route or km/h-mph crossing',
                             'border country labels are proposals, not independent ground truth',
                             'no frame arrival latency or inference replay']},indent=2))


if __name__=='__main__':
    main()
