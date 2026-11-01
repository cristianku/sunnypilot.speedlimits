// The order of the frames of a drive, the same as the server's (nnslr_tools.sign_reads.frame_order): segment, second,
// output index. The pictures at 1 a second have output index = their second; a frame between the seconds (a sign the
// comma read, or a frame of the video saved, video.ts) has READ_INDEX_BASE + its frame in the video.
import type {FrameRow} from './types';

export const READ_INDEX_BASE = 1000;
/** Frames a second of the road camera's video. */
export const CAMERA_FPS = 20;

/** The second of its segment a frame was taken at, as the server sorts it. */
export function frameSecond(row: FrameRow): number {
  const index = row.output_index;
  if (index < READ_INDEX_BASE) return index;
  return typeof row.media_time_s === 'number' ? row.media_time_s : (index - READ_INDEX_BASE) / CAMERA_FPS;
}

export function compareFrames(a: FrameRow, b: FrameRow): number {
  return a.segment_index - b.segment_index || frameSecond(a) - frameSecond(b) || a.output_index - b.output_index;
}
