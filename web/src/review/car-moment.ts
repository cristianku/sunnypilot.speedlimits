// What the comma had at the moment of a frame, from its own logs as Drive review read them: the map's limit (and the
// one ahead), the limit the vision memory held (speedVisionSP) and the one the car used (the resolver), on the photo,
// and the frames where map and vision differ, underlined in the timeline. Shown only: the two sources stay separate
// on the car, only the resolver weighs them. Read once per drive, on the server.
import {shallowReactive} from 'vue';
import {type Series, type SyncSegment, type Timeline, limitOf, secondOf, valueAt} from '../control/drive-review';
import {segmentSecond} from './frames';
import {call} from './server';
import {report} from './store';
import type {FrameRow} from './types';

const SERIES = ['map_valid', 'map_limit', 'map_ahead', 'map_ahead_dist', 'vis_limit', 'res_limit', 'res_source'];

/** Map and vision differ at a frame when their limits are this far apart (drive_episodes.DISAGREE_KPH). */
const DIFFER_KPH = 5;

/** The drive's logs: 'loading', 'none' (not scanned, or not reachable: nothing is shown) or 'ready'. */
export const carLogs = shallowReactive<{phase: 'idle' | 'loading' | 'none' | 'ready'; segments: SyncSegment[];
  series: Record<string, Series>}>({phase: 'idle', segments: [], series: {}});

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
  /** Both limits known and DIFFER_KPH or more apart. */
  differ: boolean;
}

const kph = (value: number | null | undefined, none: string): string =>
  value === undefined ? 'not in the log' : value === null ? none : String(value);

/** The time of the logs of a frame, or null when the drive's logs are not read or do not cover it. */
function logTime(row: FrameRow): number | null {
  if (carLogs.phase !== 'ready') return null;
  const second = segmentSecond(row);
  return second === null ? null : secondOf(carLogs.segments, row.segment_index, second);
}

function limitsAt(s: number): {map: number | null | undefined; vision: number | null | undefined} {
  const valid = valueAt(carLogs.series['map_valid'], s);
  const map = valid === undefined || valid === null ? undefined : valid === false ? null
    : limitOf(valueAt(carLogs.series['map_limit'], s));
  return {map, vision: limitOf(valueAt(carLogs.series['vis_limit'], s))};
}

const apart = (map: number | null | undefined, vision: number | null | undefined): boolean =>
  typeof map === 'number' && typeof vision === 'number' && Math.abs(map - vision) >= DIFFER_KPH;

/** Map and vision differ at the frame (the red underline of its tick). */
export function carDiffers(row: FrameRow): boolean {
  const s = logTime(row);
  if (s === null) return false;
  const {map, vision} = limitsAt(s);
  return apart(map, vision);
}

/** The comma's limits at the frame, in km/h, or null when the drive's logs are not read or do not cover it. */
export function carMomentOf(row: FrameRow): CarMoment | null {
  const s = logTime(row);
  if (s === null) return null;
  const {map, vision} = limitsAt(s);
  const ahead = limitOf(valueAt(carLogs.series['map_ahead'], s));
  const distance = valueAt(carLogs.series['map_ahead_dist'], s);
  const source = valueAt(carLogs.series['res_source'], s);
  return {
    map: kph(map, 'none'),
    ahead: typeof ahead === 'number' && typeof distance === 'number' ? ahead + ' in ' + Math.round(distance) + ' m' : '',
    vision: kph(vision, 'none'),
    car: kph(limitOf(valueAt(carLogs.series['res_limit'], s)), 'none') + (typeof source === 'string' && source ? ' · ' + source : ''),
    differ: apart(map, vision),
  };
}
