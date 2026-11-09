// The signs on every frame of a drive's video, for the boxes over Speed review's picture (SpeedReviewDrive.vue) at 20
// frames a second: the file of nnslr_tools.sign_tracks (GET /api/speed-review/drives/<route>/sign-track), the boxes of the
// same networks as Sign review's proposals on frame k of every minute's fcamera.hevc, which is video frame
// round(start*20)+k of the drive's video. A frame of Sign review confirmed (or saved by the AI reviewer) wins on its very
// frame; without the track only Sign review's own frames, one a second, have boxes. Type imports only: node runs this file
// with its types stripped (tests/review/test_sign_track_page.py).
import type {Detection} from '../review/types';
import type {VideoState} from './drive-review';
import type {SignFrame} from './sign-frames';

/** The drive's video index (drive_video.index): where each minute starts in the video and its frames. */
export type VideoIndex = NonNullable<VideoState['index']>;

/** A class of the classifier seen on the drive, as the boxes' label index names it. */
export interface TrackLabel {
  model_label: string;
  family: string;
  value_kph: number | null;
  supplementary_panel: string;
  plate_hours?: string;
  sign_type?: string;
}
/** A minute of the drive: its decoded frames and a box per row [k, label, x0, y0, x1, y1, score %], sorted by k. */
export interface TrackSegment {
  segment: number;
  frames: number;
  source: [number, number];
  boxes: [number, number, number, number, number, number, number][];
}
export interface SignTrack {
  format: number;
  route: string;
  fps: number;
  width: number;
  height: number;
  networks: Record<string, {id: string; size: number; mtime_ns: number}>;
  min_score: number;
  made_at: string;
  labels: TrackLabel[];
  segments: TrackSegment[];
  /** Made by the networks in use, on the drive's minutes as they are now. */
  current?: boolean;
}

/** A frame of the comma's road video: its minute and its frame k in that minute. */
export interface VideoFrame {
  segment: number;
  frame: number;
}

/** The boxes of the track by minute and frame k. Only the minutes whose frames are the video's: another count means
 * another video of the minute, and its boxes would land on the wrong frames. */
export function trackFrames(track: SignTrack, index: VideoIndex): Map<number, Map<number, Detection[]>> {
  const found = new Map<number, Map<number, Detection[]>>();
  for (const item of track.segments) {
    const minute = index.segments.find((entry) => entry.segment === item.segment);
    if (!minute || minute.frames !== item.frames) continue;
    const byFrame = new Map<number, Detection[]>();
    for (const [k, label, x0, y0, x1, y1, score] of item.boxes) {
      const kind = track.labels[label];
      if (!kind) continue;
      const box: Detection = {bbox_xyxy: [x0, y0, x1, y1], family: kind.family, value_kph: kind.value_kph,
        supplementary_panel: kind.supplementary_panel, model_label: kind.model_label, score: score / 100};
      if (kind.sign_type !== undefined) box.sign_type = kind.sign_type;
      if (kind.plate_hours !== undefined) box.plate_hours = kind.plate_hours;
      const list = byFrame.get(k);
      if (list) list.push(box);
      else byFrame.set(k, [box]);
    }
    found.set(item.segment, byFrame);
  }
  return found;
}

/** The key of a frame of Sign review: its minute and its frame k (second N is frame 20N). */
const keyOf = (segment: number, frame: number): string => segment + '/' + frame;
const frameOfSecond = (second: number): number => Math.round(second * 20);

/** The frames of Sign review confirmed, or saved by the AI reviewer, by their minute and frame k: their boxes win. */
export function reviewedFrames(frames: readonly SignFrame[]): Map<string, SignFrame> {
  return new Map(frames.filter((frame) => frame.confirmed || frame.needs_person)
    .map((frame) => [keyOf(frame.segment, frameOfSecond(frame.second)), frame]));
}

/** Every frame of Sign review by its minute and frame k: the boxes where the drive has no track. */
export function exactFrames(frames: readonly SignFrame[]): Map<string, SignFrame> {
  return new Map(frames.map((frame) => [keyOf(frame.segment, frameOfSecond(frame.second)), frame]));
}

/** What the picture draws on a frame: its boxes, in a frame of width×height, and where they come from. */
export interface Drawn {
  boxes: Detection[];
  width: number;
  height: number;
  confirmed: boolean;
  needsPerson: boolean;
  /** The frame of Sign review drawn; absent for the track's boxes. */
  frame?: SignFrame;
}

const fromFrame = (frame: SignFrame, track: SignTrack | null): Drawn => ({boxes: frame.boxes,
  width: frame.width ?? track?.width ?? 1344, height: frame.height ?? track?.height ?? 760, confirmed: frame.confirmed,
  needsPerson: !!frame.needs_person, frame});

/** The boxes of a frame of the video: a frame of Sign review confirmed (or the AI reviewer's) on it, else the track's (none
 * on a frame where the networks found nothing), else, on a minute without the track, Sign review's own frame on it;
 * null where nothing says. */
export function boxesAt(at: VideoFrame | null, track: SignTrack | null, frames: Map<number, Map<number, Detection[]>> | null,
  reviewed: Map<string, SignFrame>, exact: Map<string, SignFrame>): Drawn | null {
  if (!at) return null;
  const key = keyOf(at.segment, at.frame);
  const mine = reviewed.get(key);
  if (mine) return fromFrame(mine, track);
  const minute = frames?.get(at.segment);
  if (track && minute) {
    return {boxes: minute.get(at.frame) ?? [], width: track.width, height: track.height, confirmed: false, needsPerson: false};
  }
  const own = exact.get(key);
  return own ? fromFrame(own, track) : null;
}

/** The time in the drive's video of frame k of a minute (the middle of it, as a step does), or null where the video has
 * no such frame. */
export function frameVideoTime(index: VideoIndex | null | undefined, segment: number, frame: number): number | null {
  const item = index?.segments.find((entry) => entry.segment === segment);
  if (!item || frame < 0 || frame >= item.frames) return null;
  return Math.round(item.start * 20) / 20 + (frame + 0.5) / 20;
}
