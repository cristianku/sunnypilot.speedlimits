// Data › Speed review: what the page knows of a drive (GET /api/speed-review/drives/<route>, nnslr_tools.speed_review) and
// the rules of its commands. Everything is on the comma's clock, in ns. The driver's dashboard speed is the answer the brain
// learns, at every moment of the drive: nothing needs a confirmation. A block is the only correction, from a time to a time:
// its kph replaces his speed there, and a block with kph null leaves those moments out of the training.
import type {DriveRow} from './drive-review';

/** A point of the driver's speed, where it changes: [t_ns, kph], kph null where the logs have no speed. */
export type SpeedPoint = [number, number | null];

/** A correction a person wrote: from start_ns to end_ns the answer is kph, or nothing (null: left out). */
export interface Block {
  start_ns: number;
  end_ns: number;
  kph: number | null;
  note: string;
  reviewer: string;
  revision: number;
}

/** Roadworks a person marked, two taps on the frame shown: they say nothing of the speed and change no block. */
export interface Works {
  start_ns: number;
  /** Null until the second tap: they run to the end of the drive. */
  end_ns: number | null;
  reviewer: string;
}

export type WorksTap = 'start' | 'end' | 'remove' | 'checked';

/** A kind of road a person can mark as the roadworks (narrow, country, gravel, potholes, and those added from the page):
 * the brain learns them aside, as the roadworks, to understand the road better; they change no speed. */
export interface RoadKind {
  kind: string;
  label: string;
}

/** A mark of a kind of road: as the roadworks, each kind with its own marks; a road can be of several kinds at once. */
export interface RoadMark extends Works {
  kind: string;
}

/** What the marks on the page are of: the roadworks, or a kind of road by its kind. */
export const ROADWORKS = 'works';

/** Whose a stretch of road type is: the map's, the road labeler's (its layer over the map, Prepare Speed review), or a
 * person's correction over both. */
export type RoadTypeSource = 'map' | 'net' | 'person';
/** A stretch of one road type (speed_review.road_type_stretches), as the brain takes it: the map's, one a second (Read
 * the map in Data › Prepare), the road labeler's over it where it is sure, or a person's correction over both.
 * road_type null: the map has none, or (a person's) they cannot tell. */
export interface RoadTypeStretch {
  road_type: string | null;
  /** Missing from a server of before 9 Nov 2026: then a reviewer means a person, none the map (typeSource). */
  source?: RoadTypeSource;
  /** Who corrected it; null where it is not a person's. */
  reviewer: string | null;
  /** The road labeler's run, on its stretches. */
  run?: string | null;
  /** The frame keys of its first and last frame. */
  from: string;
  to: string;
  from_ns: number;
  to_ns: number;
}

/** The road types as the page names them, in the order of road_types.ROAD_TYPES. */
export const ROAD_TYPE_LABELS: Record<string, {label: string; short: string}> = {
  town: {label: 'Town', short: 'town'}, rural: {label: 'Rural', short: 'rural'},
  expressway: {label: 'Expressway', short: 'expr'}, motorway: {label: 'Motorway', short: 'mway'},
  ramp: {label: 'Ramp', short: 'ramp'}, parking: {label: 'Parking', short: 'park'}, zone30: {label: 'Zone 30', short: '30'},
  garage: {label: 'Garage', short: 'gar'}, merge: {label: 'Merge', short: 'merge'},
};
export const roadTypeLabel = (type: string | null): string => type === null ? 'Can’t tell' : ROAD_TYPE_LABELS[type]?.label ?? type;

export interface SpeedDrive {
  route: string;
  /** The drive's last write: a save sends it back, and a stale one is refused. */
  revision: number;
  /** The dashboard speed along the drive at the brain's steps, in whole km/h, where it changes (speed_curve). */
  speed: SpeedPoint[];
  /** [from_ns, to_ns] where openpilot or the car's cruise held the speed, the gas not pressed (held_spans): not his foot,
   * so the brain learns no speed there unless a block says one. */
  held: [number, number][];
  blocks: Block[];
  works: Works[];
  /** From the start of the drive up to here a person looked for roadworks: outside the marks there are none. */
  works_checked_ns: number | null;
  /** The kinds of road in their order, their marks on the drive, and how far it was gone through for each (by kind;
   * a kind nobody went through for is not there). */
  road_kinds: RoadKind[];
  road: RoadMark[];
  road_checked_ns: Record<string, number>;
  /** The road types along the drive, the map's and a person's corrections, and the types there are; missing from a
   * server older than them. */
  road_types?: RoadTypeStretch[];
  road_type_names?: string[];
  /** The drive's first and last ns, where a block may go; null before its logs are scanned. */
  span_ns: [number, number] | null;
  total_s: number;
  /** The highest speed a block may say. */
  max_kph: number;
  /** Whether the brain learns from the drive: sunnypilot off in all of it (speed_review.dashcam_drive); null before its
   * logs are read. */
  dashcam: boolean | null;
}

export interface SpeedRow extends DriveRow {
  /** How many blocks, roadworks marks and marks of a kind of road the drive has. */
  blocks: number;
  works: number;
  road: number;
  total_s: number;
  /** As SpeedDrive's: the list shows only the drives where it is not false. */
  dashcam: boolean | null;
}

/** A block is at least this long (MIN_BLOCK_SECONDS of speed_review.py). */
export const MIN_BLOCK_NS = 1e9;

/** The block at a time, null outside the blocks. */
export function blockAt(blocks: readonly Block[], ns: number): Block | null {
  return blocks.find((item) => item.start_ns <= ns && ns < item.end_ns) ?? null;
}

/** The index of the last point of the speed at or before a time, -1 before the first. */
export function speedIndex(speed: readonly SpeedPoint[], ns: number): number {
  let low = 0;
  let high = speed.length - 1;
  let found = -1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (speed[middle]![0] <= ns) {
      found = middle;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return found;
}

/** The driver's dashboard speed at a time, null before the first point or where the logs have none. */
export function speedAt(speed: readonly SpeedPoint[], ns: number): number | null {
  const index = speedIndex(speed, ns);
  return index < 0 ? null : speed[index]![1];
}

/** Whether openpilot or the car's cruise held the speed at a time. */
export function heldAt(held: readonly [number, number][], ns: number): boolean {
  return held.some(([from, to]) => from <= ns && ns < to);
}

/** The answer at a time: the block's kph inside a block (null for one left out), else his speed, null where a machine
 * held it. */
export function labelAt(blocks: readonly Block[], speed: readonly SpeedPoint[], held: readonly [number, number][],
                        ns: number): number | null {
  const block = blockAt(blocks, ns);
  return block ? block.kph : heldAt(held, ns) ? null : speedAt(speed, ns);
}

/** A kph a block may say: whole, 1..max (below 1 km/h the brain has no class: leave it out instead). */
export function clampKph(kph: number, max: number): number {
  return Math.max(1, Math.min(max, Math.round(kph)));
}

/** One end of a block dragged to a time: kept inside the span given (the drive in the time window shown), off the blocks
 * next to it and at least MIN_BLOCK_NS from its other end, and pulled onto the nearest edge within reach (where the map's
 * limit changes, the ends of the other blocks, the time shown), as a magnet. With a block rolled (the one whose start
 * touches the end of item, dragged with it as one edge) the end goes up to MIN_BLOCK_NS before the rolled block's end.
 * snapped is the edge it was pulled onto, null when none was near. */
export function draggedEdge(item: Block, side: 'start' | 'end', ns: number, blocks: readonly Block[], span: readonly [number, number],
                            edges: readonly number[], reach: number, rolled: Block | null = null): {ns: number; snapped: number | null} {
  const others = blocks.filter((other) => other.start_ns !== item.start_ns && other.start_ns !== rolled?.start_ns);
  const low = Math.max(span[0], ...side === 'end' ? [item.start_ns + MIN_BLOCK_NS]
    : others.filter((other) => other.end_ns <= item.start_ns).map((other) => other.end_ns));
  const high = Math.min(span[1], ...side === 'start' ? [item.end_ns - MIN_BLOCK_NS] : rolled ? [rolled.end_ns - MIN_BLOCK_NS]
    : others.filter((other) => other.start_ns >= item.end_ns).map((other) => other.start_ns));
  // A window too short for the block to change in: it stays as it is.
  if (low > high) return {ns: side === 'start' ? item.start_ns : item.end_ns, snapped: null};
  let snapped: number | null = null;
  for (const edge of [...edges, ...others.flatMap((other) => [other.start_ns, other.end_ns])]) {
    if (edge >= low && edge <= high && Math.abs(edge - ns) <= reach && (snapped === null || Math.abs(edge - ns) < Math.abs(snapped - ns))) {
      snapped = edge;
    }
  }
  return {ns: Math.round(snapped ?? Math.min(high, Math.max(low, ns))), snapped};
}

/** The roadworks on a frame, null outside them. */
export function worksAt(works: readonly Works[], ns: number): Works | null {
  return works.find((item) => item.start_ns <= ns && (item.end_ns === null || ns < item.end_ns)) ?? null;
}

/** The stretch of road type at a time, null where the map has no frame. */
/** Whose a stretch is, also from a server that does not say it yet. */
export const typeSource = (item: RoadTypeStretch): RoadTypeSource => item.source ?? (item.reviewer !== null ? 'person' : 'map');

export function roadTypeAt(stretches: readonly RoadTypeStretch[], ns: number): RoadTypeStretch | null {
  return stretches.find((item) => item.from_ns <= ns && ns < item.to_ns) ?? null;
}

/** True when roadworks with no end yet may end at this time: after their start, not past the next ones. */
export function canEndWorks(works: readonly Works[], open: Works, ns: number): boolean {
  const next = works.find((item) => item.start_ns > open.start_ns);
  return ns > open.start_ns && (!next || ns <= next.start_ns);
}
