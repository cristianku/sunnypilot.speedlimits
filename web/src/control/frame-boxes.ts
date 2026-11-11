// The boxes over Speed review's picture (SpeedReviewDrive.vue): those of a frame of Sign review, on its very frame of the
// drive's video (second N of a minute is its frame 20N) and on no other, since the car moves the sign within a fraction
// of a second. Type imports only: node runs this file with its types stripped (tests/review/test_frame_boxes_page.py).
import type {Detection} from '../review/types';
import type {VideoState} from './drive-review';
import type {SignFrame} from './sign-frames';

/** The drive's video index (drive_video.index): where each minute starts in the video and its frames. */
export type VideoIndex = NonNullable<VideoState['index']>;

/** A frame of the comma's road video: its minute and its frame k in that minute. */
export interface VideoFrame {
  segment: number;
  frame: number;
}

/** The key of a frame of Sign review: its minute and its frame k (second N is frame 20N). */
const keyOf = (segment: number, frame: number): string => segment + '/' + frame;
const frameOfSecond = (second: number): number => Math.round(second * 20);

/** Every frame of Sign review by its minute and frame k. */
export function exactFrames(frames: readonly SignFrame[]): Map<string, SignFrame> {
  return new Map(frames.map((frame) => [keyOf(frame.segment, frameOfSecond(frame.second)), frame]));
}

/** What the picture draws on a frame: its boxes, in a frame of width×height, and the frame of Sign review they are of. */
export interface Drawn {
  boxes: Detection[];
  width: number;
  height: number;
  confirmed: boolean;
  needsPerson: boolean;
  frame: SignFrame;
}

/** The boxes of a frame of the video: Sign review's frame on it, or null where there is none. */
export function boxesAt(at: VideoFrame | null, exact: Map<string, SignFrame>): Drawn | null {
  const frame = at ? exact.get(keyOf(at.segment, at.frame)) : undefined;
  return frame ? {boxes: frame.boxes, width: frame.width ?? 1344, height: frame.height ?? 760, confirmed: frame.confirmed,
    needsPerson: !!frame.needs_person, frame} : null;
}
