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
  /** The drive's first and last ns, where a block may go; null before its logs are scanned. */
  span_ns: [number, number] | null;
  total_s: number;
  /** The highest speed a block may say. */
  max_kph: number;
}

export interface SpeedRow extends DriveRow {
  /** How many blocks and roadworks marks the drive has. */
  blocks: number;
  works: number;
  total_s: number;
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

/** The roadworks on a frame, null outside them. */
export function worksAt(works: readonly Works[], ns: number): Works | null {
  return works.find((item) => item.start_ns <= ns && (item.end_ns === null || ns < item.end_ns)) ?? null;
}

/** True when roadworks with no end yet may end at this time: after their start, not past the next ones. */
export function canEndWorks(works: readonly Works[], open: Works, ns: number): boolean {
  const next = works.find((item) => item.start_ns > open.start_ns);
  return ns > open.start_ns && (!next || ns <= next.start_ns);
}
