// Data › Speed review: what the page knows of a drive (GET /api/speed-review/drives/<route>, nnslr_tools.speed_review) and
// the rules of its commands. A stretch is on the comma's clock, in ns. A stretch not examined has no row on the server:
// its kph is the proposal: the OpenStreetMap limit (source 'map'), and where the map has no number the limit of the signs
// the comma read (source 'signs'). Next to it the page shows what the map and the signs said, and where the two differ. A split lives here, as a cut in the list of the page, until one of its halves is saved: the server then clips
// what stays outside, and the cut is a boundary of its own.
import type {DriveRow} from './drive-review';

export type StretchState = 'not_examined' | 'confirmed' | 'corrected' | 'undetermined';

export interface Stretch {
  start_ns: number;
  end_ns: number;
  kph: number | null;
  state: StretchState;
  /** From when the past is enough to work the limit out, null when it is known from the start. */
  known_ns: number | null;
  note: string;
  reviewer: string | null;
  /** On a stretch not examined: where the proposal comes from, the map or the signs read where the map has no number. */
  source?: 'map' | 'signs';
}

/** A limit along the drive: kph null where there is none. */
export interface LimitRun {
  start_ns: number;
  end_ns: number;
  kph: number | null;
}

/** Where the signs read said another limit than the map's, for long enough to be no sign and way end at two places. */
export interface Differ {
  start_ns: number;
  end_ns: number;
  map_kph: number;
  sign_kph: number;
}

/** Roadworks a person marked, two taps on the frame shown: they say nothing of the speed and cut no stretch. */
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
  stretches: Stretch[];
  works: Works[];
  /** From the start of the drive up to here a person looked for roadworks: outside the marks there are none. */
  works_checked_ns: number | null;
  examined_s: number;
  total_s: number;
  /** Whether the map was read for this drive at all (Overview › Find the road types). */
  map_read: boolean;
  /** The OpenStreetMap limit along the drive, as read on the server. */
  map: LimitRun[];
  /** The limit of the signs the comma read: a number holds to the next sign, an end of a limit or a minute unseen. */
  signs: LimitRun[];
  differ: Differ[];
}

export interface SpeedRow extends DriveRow {
  first: boolean;
  examined_s: number;
  total_s: number;
}

/** What a save sends for a stretch; state null (Clear) puts it back to not examined. */
export interface Label {
  kph?: number | null;
  state: Exclude<StretchState, 'not_examined'> | null;
  known_ns?: number | null;
  note?: string;
}

/** The speeds a tap sets; the others are a step of 5 away. */
export const SPEEDS = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130] as const;
export const STEP_KPH = 5;
const MAX_KPH = 200;
/** A cut leaves at least this on both sides: less is no stretch of road. */
const MIN_PIECE_NS = 1e9;
/** A limit shorter than this on a stretch is a flicker (MERGE_SECONDS of speed_review.py). */
const MERGE_NS = 5e9;

export const STATE_LABEL: Record<StretchState, string> = {not_examined: 'Not examined', confirmed: 'Confirmed', corrected: 'Corrected',
  undetermined: 'Can\'t tell'};

/** The stretches of the server with the cuts of the page: the pieces of a stretch keep what it says. */
export function withCuts(stretches: readonly Stretch[], cuts: readonly number[]): Stretch[] {
  return stretches.flatMap((item) => {
    const edges = [item.start_ns, ...cuts.filter((cut) => item.start_ns < cut && cut < item.end_ns).sort((a, b) => a - b), item.end_ns];
    return edges.slice(1).map((end, index) => {
      const start = edges[index]!;
      // A piece before the time stays not known to its end, as the server clips it.
      const known = item.known_ns === null || item.known_ns < start ? null : Math.min(item.known_ns, end - 1);
      return {...item, start_ns: start, end_ns: end, known_ns: known};
    });
  });
}

/** The cuts still only on the page, after an answer of the server. */
export function openCuts(cuts: readonly number[], stretches: readonly Stretch[]): number[] {
  const edges = new Set(stretches.flatMap((item) => [item.start_ns, item.end_ns]));
  return cuts.filter((cut) => !edges.has(cut));
}

/** The index of the piece at a time; the last one at the very end of the drive, -1 with no pieces. */
export function pieceAt(pieces: readonly Stretch[], ns: number): number {
  let found = pieces.length ? 0 : -1;
  for (let index = 0; index < pieces.length && pieces[index]!.start_ns <= ns; index++) found = index;
  return found;
}

/** The state a value saves with: the proposal stands or a person gave another, and what a person corrected stays so. */
export function stateFor(item: Stretch, kph: number): 'confirmed' | 'corrected' {
  return kph !== item.kph || item.state === 'corrected' ? 'corrected' : 'confirmed';
}

/** The label of a stretch with this value, what else it says kept. A piece clipped before its time was not known for the old
 * value: another value starts with no time, or the whole piece would stay out of training. */
export function labelOf(item: Stretch, kph: number): Label {
  const clipped = kph !== item.kph && item.known_ns === item.end_ns - 1;
  return {kph, state: stateFor(item, kph), known_ns: clipped ? null : item.known_ns, note: item.note};
}

/** A value a step away from the stretch's, null past the grid's ends or with no value to step from. */
export function stepped(item: Stretch, delta: number): number | null {
  if (item.kph === null) return null;
  const kph = item.kph + delta;
  return kph >= STEP_KPH && kph <= MAX_KPH ? kph : null;
}

/** The next stretch nobody looked at after a time, from the start of the drive when none is left after it. */
export function nextOpen(pieces: readonly Stretch[], ns: number): Stretch | null {
  const open = pieces.filter((item) => item.state === 'not_examined');
  return open.find((item) => item.start_ns >= ns) ?? open[0] ?? null;
}

/** True when a cut at this time is inside a piece, far enough from its ends. */
export function canCut(piece: Stretch | null, ns: number): boolean {
  return piece !== null && ns - piece.start_ns >= MIN_PIECE_NS && piece.end_ns - ns >= MIN_PIECE_NS;
}

/** The cut of the page at an end of a piece (the one nearest to the time when it has two), null when both ends are the
 * server's. */
export function cutOf(piece: Stretch | null, cuts: readonly number[], ns: number): number | null {
  if (!piece) return null;
  const ends = [piece.start_ns, piece.end_ns].filter((edge) => cuts.includes(edge));
  return ends.sort((a, b) => Math.abs(a - ns) - Math.abs(b - ns))[0] ?? null;
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

/** The limits along a stretch in time order, each once in a row, where there is one; one that touches the stretch for less
 * than MERGE_NS (or than the stretch, when it is shorter) is left out. */
export function limitsOver(runs: readonly LimitRun[], start: number, end: number): number[] {
  const least = Math.min(MERGE_NS, end - start);
  const out: number[] = [];
  for (const item of runs) {
    if (item.kph !== null && Math.min(end, item.end_ns) - Math.max(start, item.start_ns) >= least && out[out.length - 1] !== item.kph) out.push(item.kph);
  }
  return out;
}

/** The limits of limitsOver as a line: "120 → 80", "none" without one. */
export function limitsText(limits: readonly number[]): string {
  return limits.length ? limits.join(' → ') : 'none';
}
