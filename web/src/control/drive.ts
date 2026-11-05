// The Drive view of Results (DriveView.vue): a drive of the test (or validation) replayed as in the car. Frame by
// frame, in the order of the drive, the speed limit the networks would show — the last speed sign they read, carried on
// until the next one or an end of limit — next to the limit of the signs a person reviewed, carried on the same way.
// Only what the chain report keeps is used: the cutter's boxes from a score and what the classifier called each, and the
// reviewed signs with their labels. Which road a sign applies to (the decider) is not in the report: every speed sign
// read counts, as if it were ours.
import type {PixelBox, WatchFrame} from '../common/panel-types';
import {NOT_A_SIGN} from './watch';

/** A limit in km/h; 'end' after an end of limit; null before any sign. */
export type Limit = number | 'end' | null;
/** A frame against the signs: the limit right, wrong, or nothing to judge yet (neither has a limit). */
export type DriveState = 'right' | 'wrong' | 'none';

/** A box drawn on the frame: a sign the networks read (right, wrong, or on nothing), or a reviewed sign they missed. */
export interface DriveMark {
  box: PixelBox;
  text: string;
  tone: 'right' | 'wrong' | 'false' | 'missed';
}

export interface DriveFrame {
  model: Limit;
  truth: Limit;
  state: DriveState;
  /** A speed sign read on this frame (the networks), or reviewed on it (the truth). */
  modelSign: boolean;
  truthSign: boolean;
  marks: DriveMark[];
}

export interface DriveSummary {
  /** Frames with a limit to judge, and those where the networks show the right one. */
  judged: number;
  right: number;
  /** Speed signs reviewed on the drive: read right, read as another value, missed; and speed signs read on nothing. */
  signs: number;
  read: number;
  misread: number;
  missed: number;
  invented: number;
}

/** The limit a label sets: 80 for maximum_speed:80 (also a number with hours, a zone or a display showing a number),
 * 'end' for an end of limit; null for any other label (another sign, a speed sign whose number was not read, "not a sign"). */
export function limitOf(label: string | null | undefined): Limit {
  if (!label || label === NOT_A_SIGN) return null;
  const [family = '', value = ''] = label.split(':');
  if (family === 'cancellation') return 'end';
  if (family !== 'maximum_speed' && family !== 'maximum_speed_hours' && family !== 'zone'
    && family !== 'variable_display') return null;
  const number = Number(value);
  return Number.isInteger(number) && number > 0 ? number : null;
}

/** "80", "END", "–". */
export function limitText(limit: Limit): string {
  return limit === null ? '–' : limit === 'end' ? 'END' : String(limit);
}

const area = (box: PixelBox): number => Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
/** How much of the smaller of two boxes the other covers (0-1). */
function cover(a: PixelBox, b: PixelBox): number {
  const width = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const height = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  const small = Math.min(area(a), area(b));
  return width <= 0 || height <= 0 || small <= 0 ? 0 : width * height / small;
}

/** The limit the largest (nearest) of the signs of a frame sets, or null when none sets one. */
function nearest(items: {box: PixelBox; limit: Limit}[]): Limit {
  let best: {box: PixelBox; limit: Limit} | null = null;
  for (const item of items) if (item.limit !== null && (best === null || area(item.box) > area(best.box))) best = item;
  return best?.limit ?? null;
}

/** The drive replayed at a score of the cutter: every frame's limits and marks, and the drive at a glance. */
export function replayDrive(frames: readonly WatchFrame[], score: number): {frames: DriveFrame[]; summary: DriveSummary} {
  let model: Limit = null;
  let truth: Limit = null;
  const summary: DriveSummary = {judged: 0, right: 0, signs: 0, read: 0, misread: 0, missed: 0, invented: 0};
  const out = frames.map((frame): DriveFrame => {
    const kept = frame.boxes.filter((item) => item.score >= score && item.label && item.label !== NOT_A_SIGN);
    const marks: DriveMark[] = [];
    const read: {box: PixelBox; limit: Limit}[] = [];
    for (const item of kept) {
      const limit = limitOf(item.label);
      // A second box of the cutter on a reviewed sign (the report matches one box per sign) reads that sign too.
      const on = item.match >= 0 ? item.match : frame.truth.findIndex((sign) => cover(sign.box, item.box) >= 0.5);
      if (item.match < 0 && on >= 0) {
        if (limit !== null && limit !== limitOf(frame.truth[on]?.label)) {
          marks.push({box: item.box, text: limitText(limit), tone: 'wrong'});
          read.push({box: item.box, limit});
        }
        continue;
      }
      // Only what changes the limit is drawn: a speed sign or an end of limit read (another sign never does).
      if (limit === null) continue;
      const truthLabel = item.match >= 0 ? frame.truth[item.match]?.label ?? null : null;
      const tone = item.match < 0 ? 'false' : limitOf(truthLabel) === limit ? 'right' : 'wrong';
      marks.push({box: item.box, text: limitText(limit), tone});
      read.push({box: item.box, limit});
      if (item.match < 0) summary.invented += 1;
    }
    const reviewed: {box: PixelBox; limit: Limit}[] = [];
    frame.truth.forEach((item, index) => {
      const limit = limitOf(item.label);
      if (limit === null) return;
      reviewed.push({box: item.box, limit});
      summary.signs += 1;
      const answer = kept.find((box) => box.match === index);
      if (!answer) {
        summary.missed += 1;
        marks.push({box: item.box, text: limitText(limit), tone: 'missed'});
      } else if (limitOf(answer.label) === limit) {
        summary.read += 1;
      } else {
        summary.misread += 1;
      }
    });
    const modelHere = nearest(read);
    const truthHere = nearest(reviewed);
    if (modelHere !== null) model = modelHere;
    if (truthHere !== null) truth = truthHere;
    const state: DriveState = model === null && truth === null ? 'none' : model === truth ? 'right' : 'wrong';
    if (state !== 'none') {
      summary.judged += 1;
      if (state === 'right') summary.right += 1;
    }
    return {model, truth, state, modelSign: modelHere !== null, truthSign: truthHere !== null, marks};
  });
  return {frames: out, summary};
}

/** The timeline: runs of frames in the same state, as [first, count, state]. */
export function stateRuns(frames: readonly DriveFrame[]): [number, number, DriveState][] {
  const runs: [number, number, DriveState][] = [];
  frames.forEach((frame, index) => {
    const last = runs[runs.length - 1];
    if (last && last[2] === frame.state) last[1] += 1;
    else runs.push([index, 1, frame.state]);
  });
  return runs;
}
