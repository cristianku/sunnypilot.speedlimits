// Data › Speed review: what the page knows of a drive (GET /api/speed-review/drives/<route>, nnslr_tools.speed_review) and
// the rules of its commands. A stretch is on the comma's clock, in ns. A stretch not examined has no row on the server:
// its kph is the proposal, the limit the comma's vision held. A split lives here, as a cut in the list of the page, until
// one of its halves is saved: the server then clips what stays outside, and the cut is a boundary of its own.
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
}

/** Roadworks a person marked, two taps on the frame shown: they say nothing of the speed and cut no stretch. */
export interface Works {
  start_ns: number;
  /** Null until the second tap: they run to the end of the drive. */
  end_ns: number | null;
  reviewer: string;
}

export type WorksTap = 'start' | 'end' | 'remove';

export interface SpeedDrive {
  route: string;
  /** The drive's last write: a save sends it back, and a stale one is refused. */
  revision: number;
  stretches: Stretch[];
  works: Works[];
  examined_s: number;
  total_s: number;
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

export const STATE_LABEL: Record<StretchState, string> = {not_examined: 'Not examined', confirmed: 'Confirmed', corrected: 'Corrected',
  undetermined: 'Can\'t tell'};

/** The stretches of the server with the cuts of the page: the pieces of a stretch keep what it says. */
export function withCuts(stretches: readonly Stretch[], cuts: readonly number[]): Stretch[] {
  return stretches.flatMap((item) => {
    const edges = [item.start_ns, ...cuts.filter((cut) => item.start_ns < cut && cut < item.end_ns).sort((a, b) => a - b), item.end_ns];
    return edges.slice(1).map((end, index) => {
      const start = edges[index]!;
      const known = item.known_ns !== null && start <= item.known_ns && item.known_ns < end ? item.known_ns : null;
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

/** The label of a stretch with this value, what else it says kept. */
export function labelOf(item: Stretch, kph: number): Label {
  return {kph, state: stateFor(item, kph), known_ns: item.known_ns, note: item.note};
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
