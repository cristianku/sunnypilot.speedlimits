// A second of the drive. Where the server has the segment's road video (report.videos) it is all the frames of that
// second, 20 of them: the picture at 1 a second and the frames of the video up to the next one (a sign the comma read,
// a frame saved, a frame not saved yet). Confirm with To review or All goes on through them frame by frame, after a
// frame with a sign as after one with no sign (nextInVideo). Offline, or in a segment with no video, the second is its
// picture at 1 a second alone.
import {computed} from 'vue';
import {frameStatus, isBetweenFrame} from './frames';
import {CAMERA_FPS, READ_INDEX_BASE} from './order';
import {anchorIndex, indexOf, report, rowOf, state} from './store';
import type {FrameRow} from './types';

// The rows of the report by frame of their segment's video ("segment/decoded").
const byFrame = computed(() => {
  const map = new Map<string, FrameRow>();
  for (const row of report.rows) {
    if (typeof row.decoded_frame_index === 'number') map.set(row.segment_index + '/' + row.decoded_frame_index, row);
  }
  return map;
});

/** The segment has its road video on the server. */
export function hasVideo(segment: number): boolean {
  return report.videos !== null && segment in report.videos;
}

/** The key of a frame of a segment's video: its row of the report, else the frame not saved (store.ts virtualRowOf). */
function keyAt(segment: number, decoded: number): string {
  return byFrame.value.get(segment + '/' + decoded)?.frame_key ?? segment + '/' + (READ_INDEX_BASE + decoded);
}

/** The walk of Confirm reaches the frame: a row of the report, or a frame of the video right after one confirmed, with a
 * box or with no sign (the video is walked frame by frame, never a second at a time). */
function walked(segment: number, decoded: number): boolean {
  return byFrame.value.has(segment + '/' + decoded) || (decoded > 0 && frameStatus(keyAt(segment, decoded - 1)) === 'done');
}

/** The frames of the video in the cell of a picture at 1 a second, [first, end) in its segment's video; null: the cell
 * is the picture alone (offline, a segment with no video, a report with no frame indices). The end is the video's
 * last frame when its count is known, else a whole second. */
function secondSpan(row: FrameRow): [number, number] | null {
  const decoded = row.decoded_frame_index;
  if (isBetweenFrame(row) || typeof decoded !== 'number' || !hasVideo(row.segment_index)) return null;
  const first = Math.floor(decoded / CAMERA_FPS) * CAMERA_FPS;
  const count = report.videos?.[row.segment_index] ?? null;
  return [first, Math.max(decoded + 1, Math.min(first + CAMERA_FPS, count ?? Infinity))];
}

/** Which frame of its second's cell the frame is, [from 1, of how many]: 20 a second, fewer in the last second of a
 * video whose count is known; null where the cell is its picture alone. */
export function frameOfSecond(row: FrameRow): [number, number] | null {
  const decoded = row.decoded_frame_index;
  if (typeof decoded !== 'number' || !hasVideo(row.segment_index)) return null;
  const first = Math.floor(decoded / CAMERA_FPS) * CAMERA_FPS;
  const count = report.videos?.[row.segment_index] ?? null;
  const place = decoded - first + 1;
  return [place, Math.max(place, Math.min(CAMERA_FPS, (count ?? Infinity) - first))];
}

/** A frame between the seconds that its cell already counts (the walk goes through it there). */
function inCell(row: FrameRow): boolean {
  const decoded = row.decoded_frame_index;
  if (typeof decoded !== 'number' || !hasVideo(row.segment_index)) return false;
  const picture = byFrame.value.get(row.segment_index + '/' + Math.floor(decoded / CAMERA_FPS) * CAMERA_FPS);
  return picture !== undefined && !isBetweenFrame(picture);
}

/** After Confirm with To review (``skipDone``) or All: the frame after this one, through the frames of the cells with
 * their video that the walk reaches (walked: a frame not saved only right after one confirmed) and from picture to
 * picture elsewhere; with To review the first one not confirmed. Null: none left after it. */
export function nextInVideo(key: string, skipDone: boolean): string | null {
  const row = rowOf(key);
  if (row === null) return null;
  const rows = report.rows;
  const index = indexOf(key);
  // A frame of the video not saved: after the frame of the report just before it.
  const start = index >= 0 ? index : anchorIndex(key);
  const here = row.decoded_frame_index;
  // From the cell of this frame: the picture at 1 a second at or before it.
  let cell = Math.max(start, 0);
  while (cell > 0 && isBetweenFrame(rows[cell] as FrameRow)) cell -= 1;
  const wanted = (candidate: string): boolean => candidate !== key && (!skipDone || frameStatus(candidate) !== 'done');
  for (let i = cell; i < rows.length; i += 1) {
    const item = rows[i] as FrameRow;
    const span = secondSpan(item);
    if (span === null) {
      // A picture alone, or a frame between the seconds its cell does not count: in the order of the report.
      if (i > start && (isBetweenFrame(item) ? !inCell(item) : true) && wanted(item.frame_key)) return item.frame_key;
      continue;
    }
    for (let decoded = span[0]; decoded < span[1]; decoded += 1) {
      if (item.segment_index === row.segment_index && typeof here === 'number' && decoded <= here) continue;
      const candidate = keyAt(item.segment_index, decoded);
      if (walked(item.segment_index, decoded) && wanted(candidate)) return candidate;
    }
  }
  return null;
}
