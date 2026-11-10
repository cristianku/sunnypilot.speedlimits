// The signs of Sign review on Speed review's drive (SpeedReviewDrive.vue): the boxes of every frame of Sign review (GET
// /api/speed-review/drives/<route>/signs, drive_review.sign_frames), the confirmed ones where a frame is confirmed, else
// the model's proposals, each frame placed at its time of the drive with the drive player's own mapping (drive-review.ts
// secondOf, as the Server lane places our networks' signs). Only shown here: the boxes are changed in Sign review.
import {signsHref} from '../common/sign-link';
import {isSpeedFamily} from '../review/catalog';
import type {Detection} from '../review/types';

/** One frame of Sign review with its boxes, as the server sends it. */
export interface SignFrame {
  frame_key: string;
  segment: number;
  /** The second of its segment's video (its minute). */
  second: number;
  /** The size of the frame the boxes are in, pixels. */
  width: number | null;
  height: number | null;
  /** Confirmed by a person, or by an agent through the API; not a frame of the site's AI reviewer waiting for a person. */
  confirmed: boolean;
  /** Saved by the site's AI reviewer, waiting for a person to confirm it (its boxes, not the model's). Older servers do not
   * send it. */
  needs_person?: boolean;
  reviewer: string | null;
  reviewer_kind: 'human' | 'ai' | null;
  boxes: Detection[];
}
/** Who proposed the boxes of a drive's report: the proposer, Our networks' parts by id, and whether those are the
 * networks in use (issue #42). */
export interface ProposedBy {
  proposer: 'vietnam' | 'ours' | 'qwen';
  networks: Record<string, string>;
  in_use: boolean;
}
export interface SignFrames {
  run_id: string;
  frames: SignFrame[];
  /** null while Propose again works on the drive; older servers do not send it. */
  proposed_by?: ProposedBy | null;
}

/** A frame at its time of the drive, in seconds of the logs. */
export interface PlacedFrame extends SignFrame {
  at: number;
}

/** A frame shown on the picture is the nearest one within this much of the time shown. */
export const NEAR_S = 0.5;

/** The frames that can be placed on the drive, in time order. */
export function placeFrames(frames: readonly SignFrame[], timeOf: (segment: number, second: number) => number | null): PlacedFrame[] {
  return frames.flatMap((frame) => {
    const at = timeOf(frame.segment, frame.second);
    return at === null || !Number.isFinite(at) ? [] : [{...frame, at}];
  }).sort((a, b) => a.at - b.at);
}

/** The frame nearest to s, if it is within `within` seconds of it. */
export function nearestFrame(placed: readonly PlacedFrame[], s: number, within = NEAR_S): PlacedFrame | null {
  if (!placed.length) return null;
  let low = 0;
  let high = placed.length - 1;
  // The first frame at or after s.
  while (low < high) {
    const middle = (low + high) >> 1;
    if (placed[middle]!.at < s) low = middle + 1;
    else high = middle;
  }
  const after = placed[low]!;
  const before = low > 0 ? placed[low - 1]! : null;
  const best = before && Math.abs(before.at - s) <= Math.abs(after.at - s) ? before : after;
  return Math.abs(best.at - s) <= within ? best : null;
}

/** A sign about the speed: a limit with its number (limit, zone, variable display) or an end of limit. */
export const isSpeedSign = (box: Detection): boolean => isSpeedFamily(box.family) || box.family === 'cancellation'
  || box.family === 'zone_end';

/** A sign confirmed for another road (applicability other_road, "Another" in Sign review): grey on the picture, never on
 * the lane of our road's speed signs; the brain leaves it out too (brain_targets). */
export const isOtherRoad = (box: Detection): boolean => box.applicability === 'other_road';

/** The text of a speed sign on the lane: its number, "Z30" for a zone, "end" for an end of limit, "Z end" of a zone. */
export function signText(box: Detection): string {
  if (box.family === 'cancellation') return 'end';
  if (box.family === 'zone_end') return 'Z end';
  if (box.value_kph === null || box.value_kph === undefined) return '?';
  return (box.family === 'zone' ? 'Z' : '') + box.value_kph;
}

/** A confirmed speed sign of the lane: the frames in a row that show it (a second or less apart, the same text), from its
 * first frame to its last. */
export interface SignRun {
  from: number;
  to: number;
  text: string;
  ticks: number[];
}

/** The confirmed speed signs of the drive, frame by frame, joined into runs. */
export function signRuns(placed: readonly PlacedFrame[]): SignRun[] {
  const runs: SignRun[] = [];
  for (const frame of placed) {
    if (!frame.confirmed) continue;
    const texts = [...new Set(frame.boxes.filter((box) => isSpeedSign(box) && !isOtherRoad(box)).map(signText))];
    if (!texts.length) continue;
    const text = texts.join(' ');
    const last = runs[runs.length - 1];
    if (last && last.text === text && frame.at - last.to <= 1.5) {
      last.to = frame.at;
      last.ticks.push(frame.at);
    } else {
      runs.push({from: frame.at, to: frame.at, text, ticks: [frame.at]});
    }
  }
  return runs;
}

/** The Signs side of the drive opened on this very frame (common/sign-link.ts). */
export function frameHref(route: string, frame: SignFrame): string {
  return signsHref(route, {segment: frame.segment, second: frame.second}, frame.frame_key);
}
