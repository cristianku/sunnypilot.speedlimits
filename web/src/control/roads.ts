// Data › Roads: the kinds of road with their words, and a drive opened (RoadDrive.vue) as runs of frames, the stretches
// still to review, the stretch a correction sets, why a frame has its type in a driver's words and the time of a frame.
import type {RoadFrame, RoadType} from '../common/panel-types';

/** The kinds of road in the order of the legend, each with its colour (control.html .rt-town ...). */
export const ROAD_TYPES: [key: RoadType | 'unknown', label: string][] = [['town', 'Town'], ['zone30', 'Zone 30'],
  ['rural', 'Rural'], ['expressway', 'Expressway'], ['motorway', 'Motorway'], ['ramp', 'Ramp'], ['merge', 'Merge lane'],
  ['parking', 'Parking'], ['garage', 'Garage'], ['unknown', 'Unknown']];
export const ROAD_LABEL = Object.fromEntries(ROAD_TYPES) as Record<RoadType | 'unknown', string>;

/** A run of frames of one kind, the map's or a person's: what the strip draws. */
export interface RoadRun {
  first: number;
  count: number;
  kind: RoadType | 'unknown';
  reviewed: boolean;
}
export function roadRuns(frames: readonly RoadFrame[]): RoadRun[] {
  const runs: RoadRun[] = [];
  frames.forEach((frame, index) => {
    const kind = frame.road_type ?? 'unknown';
    const last = runs[runs.length - 1];
    if (last && last.kind === kind && last.reviewed === frame.reviewed) last.count += 1;
    else runs.push({first: index, count: 1, kind, reviewed: frame.reviewed});
  });
  return runs;
}

/** A frame still to review: no person confirmed or corrected it. Every proposal of the map is, the sure ones too: only
 * a person's types are labels of the training (nnslr_tools.road_type_data). */
export function toCheck(frame: RoadFrame): boolean {
  return !frame.reviewed;
}

/** The stretches to review, in the order of the drive: the frames to review one after the other, of one kind and as
 * sure (an unknown gap then an unsure ramp are two stretches, as stretchAt sets them one at a time: one Confirm clears
 * one). */
export function checkStretches(frames: readonly RoadFrame[]): {first: number; count: number}[] {
  const stretches: {first: number; count: number}[] = [];
  frames.forEach((frame, index) => {
    if (!toCheck(frame)) return;
    const last = stretches[stretches.length - 1];
    const before = frames[index - 1];
    if (last && last.first + last.count === index && before?.road_type === frame.road_type && before.sure === frame.sure) {
      last.count += 1;
    } else stretches.push({first: index, count: 1});
  });
  return stretches;
}

/** The first frame of the next stretch to review after the one at `from` (from the start again after the last); null
 * when none is left. */
export function nextToCheck(stretches: readonly {first: number}[], from: number): number | null {
  const next = stretches.find((stretch) => stretch.first > from) ?? stretches[0];
  return next && next.first !== from ? next.first : null;
}

/** The stretch a correction sets at a frame, [first, last]: the frames around it of the same kind, as sure and set by a
 * person or not. A tap on an unsure ramp changes only its unsure frames, never the sure motorway around it nor a stretch
 * a person already set. */
export function stretchAt(frames: readonly RoadFrame[], index: number): [number, number] | null {
  const here = frames[index];
  if (!here) return null;
  const same = (frame: RoadFrame | undefined): boolean => frame !== undefined && frame.road_type === here.road_type
    && frame.sure === here.sure && frame.reviewed === here.reviewed;
  let first = index;
  let last = index;
  while (same(frames[first - 1])) first -= 1;
  while (same(frames[last + 1])) last += 1;
  return [first, last];
}

/** The kinds of road of the map (highway=...) that name the type of road alone. */
const HIGHWAY_WORDS: Record<string, string> = {
  motorway: 'A motorway on the map', motorway_link: 'A motorway ramp on the map: check it',
  trunk: 'An expressway on the map', trunk_link: 'An expressway ramp on the map: check it',
  residential: 'A residential street on the map', living_street: 'A residential street on the map',
};

/** Why a frame has its type, in a driver's words, from the reason of the map (nnslr_tools.road_types: "maxspeed 50",
 * "maxspeed 35 mph", "highway motorway_link", "filled (no gps)"...); '' for a reason without words here, rather than the tags of the map. */
export function reasonOf(frame: RoadFrame): string {
  if (frame.reviewed) return frame.road_type === null ? 'Set by you: can\'t tell' : 'Set by you';
  const basis = frame.basis ?? '';
  if (basis.startsWith('acceleration lane')) return 'The acceleration lane after an entry ramp: the carriageway\'s limit from here';
  if (basis.startsWith('filled')) return 'Between two stretches of the same kind (a tunnel or a gap)';
  if (basis.startsWith('smoothed')) return 'A short jump of the GPS';
  if (basis === 'no gps') return 'No GPS here (garage, tunnel or the start of the drive)';
  if (basis === 'no road') return 'No road on the map here';
  if (basis === 'garage on the map') return 'An underground or indoor car park on the map';
  if (basis === 'slow without gps') return 'Slow, no GPS, by a garage or at the start or end of the drive: check it';
  if (basis === 'parking aisle' || basis.includes('car park')) return 'A car park on the map';
  const limit = /^maxspeed (\d+)$/.exec(basis)?.[1];
  if (limit === '30') return 'The map says 30 km/h: a 30 zone or a 30 street, check it';
  if (limit) return 'The map says ' + limit + ' km/h' + (frame.sure ? ' here' : ': often a town edge, check it');
  // The US: up to 45 mph a town, 50 mph and more rural (nnslr_tools.road_types.US_TOWN_MPH).
  const miles = /^maxspeed (\d+) mph$/.exec(basis)?.[1];
  if (miles) return 'The map says ' + miles + ' mph' + (frame.sure ? ' here' : ': often a town edge, check it');
  const zone = /^limit \S*zone ?(\d+)/i.exec(basis)?.[1];
  if (zone) return 'A ' + zone + ' zone on the map';
  if (/^limit \S*urban/i.test(basis)) return 'In town on the map';
  if (/^limit \S*rural/i.test(basis)) return 'Out of town on the map';
  if (basis.endsWith(' without a limit')) {
    return /^highway (primary|secondary|tertiary)/.test(basis) ? 'A road with no limit on the map' : 'A small road with no limit on the map';
  }
  const road = /^highway (\w+)( in town)?$/.exec(basis);
  if (!road) return '';
  return road[2] ? 'An expressway through a town on the map: check it' : HIGHWAY_WORDS[road[1] ?? ''] ?? '';
}

/** Seconds from the start of the drive of a frame "segment/image": a segment a minute, a frame a second. */
export function frameSeconds(key: string): number | null {
  const [segment, image] = key.split('/').map(Number);
  return segment === undefined || image === undefined || Number.isNaN(segment) || Number.isNaN(image) ? null : segment * 60 + image;
}

/** The time of a frame: the clock ("08:14:32", the browser's time zone) when the start of the drive is known, else how
 * far into the drive ("12:05", "1:02:05"). */
export function frameTime(key: string, start: string | null): string {
  const seconds = frameSeconds(key);
  if (seconds === null) return '';
  const pad = (value: number): string => String(value).padStart(2, '0');
  const begun = start ? Date.parse(start) : NaN;
  if (!Number.isNaN(begun)) {
    const date = new Date(begun + seconds * 1000);
    return pad(date.getHours()) + ':' + pad(date.getMinutes()) + ':' + pad(date.getSeconds());
  }
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor(seconds / 60) % 60;
  return (hours ? hours + ':' + pad(minutes) : String(minutes)) + ':' + pad(seconds % 60);
}
