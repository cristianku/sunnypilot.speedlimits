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

/** A stretch of frames where a speed limit read for our road is far from the driver's speed: its first and last
 * frame, how many, and at its worst the number read, his speed and when. */
export interface OffRead {from: number; to: number; frames: number; read: number; kph: number; at: number}

/** Where the speed limit read on the signs (a frame's speed signs with their number that are not another road's: the
 * confirmed boxes of a confirmed frame, else the proposals) is more than `off` km/h from the driver's speed there
 * (speedOf: null where the logs have none): a sign read wrong, or a sign of a side road taken for ours, to correct
 * in Sign review. Frames less than `join` seconds apart are one stretch. */
export function offReads(placed: readonly PlacedFrame[], speedOf: (at: number) => number | null, off: number,
                         join: number): OffRead[] {
  const out: OffRead[] = [];
  for (const frame of placed) {
    const kph = speedOf(frame.at);
    if (kph === null) continue;
    let worst: number | null = null;
    for (const box of frame.boxes) {
      const value = box.value_kph;
      if (!isSpeedFamily(box.family) || isOtherRoad(box) || typeof value !== 'number' || value <= 0) continue;
      if (Math.abs(value - kph) > off && (worst === null || Math.abs(value - kph) > Math.abs(worst - kph))) worst = value;
    }
    if (worst === null) continue;
    const last = out[out.length - 1];
    if (last && frame.at - last.to < join) {
      last.to = frame.at;
      last.frames++;
      if (Math.abs(worst - kph) > Math.abs(last.read - last.kph)) Object.assign(last, {read: worst, kph, at: frame.at});
    } else {
      out.push({from: frame.at, to: frame.at, frames: 1, read: worst, kph, at: frame.at});
    }
  }
  return out;
}

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

/** A stretch of the drive where the confirmed frames show one kind of thing (a red light, a pedestrian): from its first
 * frame to its last, frames a second and a half apart at most. */
export interface MarkRun {
  from: number;
  to: number;
  kind: string;
  ticks: number[];
}

/** The confirmed frames joined into runs of the kind each one shows (kindOf: null where it shows none). */
export function markRuns(placed: readonly PlacedFrame[], kindOf: (frame: PlacedFrame) => string | null, gap = 1.5): MarkRun[] {
  const runs: MarkRun[] = [];
  for (const frame of placed) {
    if (!frame.confirmed) continue;
    const kind = kindOf(frame);
    if (kind === null) continue;
    const last = runs[runs.length - 1];
    if (last && last.kind === kind && frame.at - last.to <= gap) {
      last.to = frame.at;
      last.ticks.push(frame.at);
    } else runs.push({from: frame.at, to: frame.at, kind, ticks: [frame.at]});
  }
  return runs;
}

/** The traffic lights, the one that matters most first: a red stops the car before an amber, a green, a dark one. */
export const LIGHTS = ['traffic_light_red', 'traffic_light_amber', 'traffic_light_green', 'traffic_light_off'] as const;
/** The road users of their lane: people and two-wheelers, the ones the car must see; cars, trucks, buses stay out. */
export const ROAD_USERS = ['road_user_person', 'road_user_bicycle', 'road_user_motorcycle', 'road_user_other'] as const;

/** The first of the kinds (in their order) that the frame has a box of; boxes of another road left out. */
function firstKind(kinds: readonly string[], frame: PlacedFrame, otherRoadToo: boolean): string | null {
  const here = new Set(frame.boxes.filter((box) => otherRoadToo || !isOtherRoad(box)).map((box) => String(box.sign_type ?? '')));
  return kinds.find((kind) => here.has(kind)) ?? null;
}
/** The light of our road on the frame (the red first), else null. */
export const lightOf = (frame: PlacedFrame): string | null => firstKind(LIGHTS, frame, false);
/** The road user on the frame (a pedestrian first), on any road: a person crossing is on no road's lane. */
export const roadUserOf = (frame: PlacedFrame): string | null => firstKind(ROAD_USERS, frame, true);

/** The Signs side of the drive opened on this very frame (common/sign-link.ts). */
export function frameHref(route: string, frame: SignFrame): string {
  return signsHref(route, {segment: frame.segment, second: frame.second}, frame.frame_key);
}

/** A frame the site's AI reviewer saved (reviewer ai/<provider>/<model>), final or waiting for a person. */
export const isAiFrame = (frame: SignFrame): boolean => frame.reviewer_kind === 'ai' && (frame.confirmed || !!frame.needs_person);

/** A stretch of the drive the AI reviewed: its frames a few seconds apart at most, for the Signs lane's violet bar. */
export interface AiRun {
  from: number;
  to: number;
  count: number;
}
export function aiRuns(frames: readonly PlacedFrame[], gap = 2.5): AiRun[] {
  const runs: AiRun[] = [];
  for (const frame of frames) {
    const last = runs[runs.length - 1];
    if (last && frame.at - last.to <= gap) {
      last.to = frame.at;
      last.count += 1;
    } else runs.push({from: frame.at, to: frame.at, count: 1});
  }
  return runs;
}

/** The index of the last frame before s (step -1) or the first after it (step 1), more than a twentieth of a second
 * away; -1 when there is none. The frames are in time order. */
export function stepFrame(frames: readonly PlacedFrame[], s: number, step: -1 | 1): number {
  let low = 0;
  let high = frames.length;
  const edge = step > 0 ? s + 0.05 : s - 0.05;
  // The first frame after edge.
  while (low < high) {
    const middle = (low + high) >> 1;
    if (frames[middle]!.at <= edge) low = middle + 1;
    else high = middle;
  }
  if (step > 0) return low < frames.length ? low : -1;
  // The last frame before edge.
  let index = low - 1;
  while (index >= 0 && frames[index]!.at >= edge) index -= 1;
  return index;
}
