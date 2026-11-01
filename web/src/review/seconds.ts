// A cell of the timeline is a second of the drive. Where the server has the segment's road video (report.videos)
// the cell is all the frames of that second, 20 of them: the picture at 1 a second and the frames of the video up to
// the next one (a sign the comma read, a frame saved, a frame not saved yet). It is confirmed when they all are, and
// Confirm with To review or All goes on through them, frame by frame (nextInVideo). Offline, or in a segment with
// no video, the cell is its picture at 1 a second alone, as before.
import {computed} from 'vue';
import {frameStatus, isBetweenFrame} from './frames';
import {CAMERA_FPS, READ_INDEX_BASE} from './order';
import {anchorIndex, indexOf, report, rowOf} from './store';
import type {FrameRow} from './types';

/** The frames of a cell confirmed, of all its frames. */
export interface SecondProgress {
  done: number;
  total: number;
}

function secondId(segment: number, decoded: number): string {
  return segment + '/' + Math.floor(decoded / CAMERA_FPS);
}

// The rows of the report by frame of their segment's video ("segment/decoded").
const byFrame = computed(() => {
  const map = new Map<string, FrameRow>();
  for (const row of report.rows) {
    if (typeof row.decoded_frame_index === 'number') map.set(row.segment_index + '/' + row.decoded_frame_index, row);
  }
  return map;
});

// The frames between the seconds of the report (comma reads, frames of the video saved) by second (secondId).
const betweenBySecond = computed(() => {
  const map = new Map<string, FrameRow[]>();
  for (const row of report.rows) {
    const decoded = row.decoded_frame_index;
    if (!isBetweenFrame(row) || typeof decoded !== 'number') continue;
    const id = secondId(row.segment_index, decoded);
    const list = map.get(id);
    if (list === undefined) map.set(id, [row]);
    else list.push(row);
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

/** The frames between the seconds of the report in the cell of a picture at 1 a second (empty: none, or a cell of its
 * picture alone). */
export function cellFrames(row: FrameRow): FrameRow[] {
  const span = secondSpan(row);
  if (span === null) return [];
  const list = betweenBySecond.value.get(secondId(row.segment_index, span[0])) ?? [];
  return list.filter((other) => (other.decoded_frame_index ?? Infinity) < span[1]);
}

/** How many frames of the cell of a picture at 1 a second are confirmed; null for a cell of its picture alone. A frame
 * of the video not saved is not confirmed: confirmed, it is a row of the report. */
export function secondProgress(row: FrameRow): SecondProgress | null {
  const span = secondSpan(row);
  if (span === null) return null;
  const done = [row, ...cellFrames(row)].filter((frame) => frameStatus(frame.frame_key) === 'done').length;
  return {done, total: span[1] - span[0]};
}

/** A frame between the seconds that its cell already counts (the walk goes through it there). */
function inCell(row: FrameRow): boolean {
  const decoded = row.decoded_frame_index;
  if (typeof decoded !== 'number' || !hasVideo(row.segment_index)) return false;
  const picture = byFrame.value.get(row.segment_index + '/' + Math.floor(decoded / CAMERA_FPS) * CAMERA_FPS);
  return picture !== undefined && !isBetweenFrame(picture);
}

/** After Confirm with To review (``skipDone``) or All: the frame after this one, through every frame of the cells
 * with their video and from picture to picture elsewhere; with To review the first one not confirmed. Null: none left
 * after it. */
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
      if (wanted(candidate)) return candidate;
    }
  }
  return null;
}
