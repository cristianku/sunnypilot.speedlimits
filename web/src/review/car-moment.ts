// What the comma had at the moment of a frame, from its own logs as Drive review read them: the map's limit (and the
// one ahead), the limit the vision memory held (speedVisionSP) and the one the car used (the resolver). Shown only:
// the two sources stay separate on the car, only the resolver weighs them. Read once per drive, on the server.
import {shallowReactive} from 'vue';
import {
  type Episode, type Series, type SyncSegment, type Timeline, limitOf, secondOf, valueAt,
} from '../control/drive-review';
import {segmentSecond} from './frames';
import {call} from './server';
import {report} from './store';
import type {FrameRow} from './types';

const SERIES = ['map_valid', 'map_limit', 'map_ahead', 'map_ahead_dist', 'vis_limit', 'res_limit', 'res_source'];

/** The drive's logs: 'loading', 'none' (not scanned, or not reachable: the line is not shown) or 'ready'. */
export const carLogs = shallowReactive<{phase: 'idle' | 'loading' | 'none' | 'ready'; segments: SyncSegment[];
  series: Record<string, Series>; disagreements: Episode[]}>({phase: 'idle', segments: [], series: {}, disagreements: []});

export async function loadCarLogs(): Promise<void> {
  if (!report.apiBase || carLogs.phase !== 'idle') return;
  carLogs.phase = 'loading';
  try {
    const view = await call<Timeline>('GET', '/api/drive-review/drives/' + encodeURIComponent(report.routeId) + '/timeline');
    if (view.t0_ns === null || !view.segments.length) {
      carLogs.phase = 'none';
      return;
    }
    const series: Record<string, Series> = {};
    for (const name of SERIES) if (view.series[name]) series[name] = view.series[name]!;
    carLogs.segments = view.segments;
    carLogs.series = series;
    carLogs.disagreements = view.episodes.filter((item) => item.kinds.includes('disagreement'));
    carLogs.phase = 'ready';
  } catch {
    carLogs.phase = 'none';
  }
}

export interface CarMoment {
  map: string;
  ahead: string;
  vision: string;
  car: string;
  /** Inside a "Vision ≠ map" episode of Drive review. */
  differ: boolean;
}

const kph = (value: number | null | undefined, none: string): string =>
  value === undefined ? 'not in the log' : value === null ? none : value + ' km/h';

/** The comma's limits at the frame, or null when the drive's logs are not read or do not cover it. */
export function carMomentOf(row: FrameRow): CarMoment | null {
  if (carLogs.phase !== 'ready') return null;
  const second = segmentSecond(row);
  const s = second === null ? null : secondOf(carLogs.segments, row.segment_index, second);
  if (s === null) return null;
  const at = (name: string) => valueAt(carLogs.series[name], s);
  const valid = at('map_valid');
  const ahead = limitOf(at('map_ahead'));
  const distance = at('map_ahead_dist');
  const source = at('res_source');
  return {
    map: valid === undefined || valid === null ? 'not in the log' : valid === false ? 'none' : kph(limitOf(at('map_limit')), 'none'),
    ahead: typeof ahead === 'number' && typeof distance === 'number' ? ahead + ' km/h in ' + Math.round(distance) + ' m' : '',
    vision: kph(limitOf(at('vis_limit')), 'none held'),
    car: kph(limitOf(at('res_limit')), 'none') + (typeof source === 'string' && source ? ' · ' + source : ''),
    differ: carLogs.disagreements.some((item) => item.start_s <= s && s <= item.end_s),
  };
}
