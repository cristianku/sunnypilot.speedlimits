// Every frame of the road video, between the pictures at 1 a second: ← → and ‹ › go through them, 20 a second.
// The server decodes them on demand, a GOP at a time (GET <api>/video/<segment>/<frame>, review_service.video_gop),
// and gives the key of each: a picture of the report where there is one, else READ_INDEX_BASE + the frame. A frame
// of the video is not a picture of the report (store.ts virtualRowOf) until it is confirmed. Offline (no
// server) the arrows go from picture to picture (navigation.ts step).
import {currentRow, isBetweenFrame} from './frames';
import {advanceAfterConfirm, openFrame, secondTarget, step} from './navigation';
import {CAMERA_FPS} from './order';
import {hasVideo, nextInVideo} from './seconds';
import {call} from './server';
import {isVirtual, report, rowOf, state, videoCounted} from './store';
import type {FrameRow} from './types';

/** A frame of the GOP, by its index in the segment's video. */
interface VideoFrame {
  decoded: number;
  media_time_s: number;
  frame_key: string;
}

/** The GOP of a frame (its keyframe and the frames up to the next one) and the video of its segment. */
interface Gop {
  segment_index: number;
  frame_count: number;
  gop: [number, number];
  frames: VideoFrame[];
}

const gops = new Map<number, Gop[]>();
const asked = new Map<string, Promise<Gop>>();
// At most a second of steps waits while a GOP is decoded (an arrow held down).
const MAX_QUEUED = CAMERA_FPS;
// The frames loaded ahead in the direction of travel.
const AHEAD = 4;
const preloaded = new Set<string>();
let queued = 0;
let walking = false;

function cachedGop(segment: number, decoded: number): Gop | null {
  return gops.get(segment)?.find((gop) => gop.gop[0] <= decoded && decoded <= gop.gop[1]) ?? null;
}

function gopOf(segment: number, decoded: number): Promise<Gop> {
  const known = cachedGop(segment, decoded);
  if (known !== null) return Promise.resolve(known);
  const id = segment + '/' + decoded;
  let pending = asked.get(id);
  if (pending === undefined) {
    pending = call<Gop>('GET', report.apiBase + '/video/' + id).then((gop) => {
      gops.set(segment, [...(gops.get(segment) ?? []), gop]);
      videoCounted(segment, gop.frame_count);
      return gop;
    }).finally(() => asked.delete(id));
    asked.set(id, pending);
  }
  return pending;
}

function decodedOf(row: FrameRow): number | null {
  return typeof row.decoded_frame_index === 'number' ? row.decoded_frame_index : null;
}

/** The frame delta (±1) away in the segment's video; null past its first or last frame. */
function nextInSegment(gop: Gop, decoded: number, delta: number): number | null {
  const target = decoded + delta;
  return target < 0 || target >= gop.frame_count ? null : target;
}

async function keyOf(segment: number, decoded: number): Promise<string | null> {
  const gop = await gopOf(segment, decoded);
  return gop.frames.find((frame) => frame.decoded === decoded)?.frame_key ?? null;
}

/** Past the end of a segment: forward the first picture of the next one, back the last frame of the one before. */
async function across(segment: number, delta: number): Promise<string | null> {
  if (delta > 0) return report.rows.find((row) => row.segment_index > segment)?.frame_key ?? null;
  let before: FrameRow | undefined;
  for (const row of report.rows) {
    if (row.segment_index >= segment) break;
    before = row;
  }
  if (before === undefined) return null;
  const decoded = decodedOf(before);
  if (decoded === null) return before.frame_key;
  const gop = await gopOf(before.segment_index, decoded);
  const last = nextInSegment(gop, gop.frame_count, -1);
  return last === null ? before.frame_key : keyOf(before.segment_index, last);
}

/** The next frames in the direction of travel, in this browser's cache before the reviewer gets there; near the
 * end of the GOP the next one, decoded by the server. */
function preload(segment: number, decoded: number, delta: number): void {
  const gop = cachedGop(segment, decoded);
  if (gop === null) return;
  for (let ahead = 1; ahead <= AHEAD; ahead += 1) {
    const target = decoded + delta * ahead;
    const frame = gop.frames.find((item) => item.decoded === target);
    if (frame === undefined) {
      if (target >= 0 && target < gop.frame_count) void gopOf(segment, target).catch(() => undefined);
      return;
    }
    // Only the frames of the video may be kept by the browser (the pictures of the report are never cached).
    const source = rowOf(frame.frame_key)?.preview_path;
    if (source === undefined || !source.startsWith(report.pictureBase + 'video/') || preloaded.has(source)) continue;
    preloaded.add(source);
    new Image().src = source;
  }
}

async function moveOne(delta: number): Promise<boolean> {
  const row = currentRow();
  if (row === null) return false;
  const decoded = decodedOf(row);
  if (decoded === null) {
    step(delta);
    return true;
  }
  let key: string | null;
  try {
    const target = nextInSegment(await gopOf(row.segment_index, decoded), decoded, delta);
    key = target === null ? await across(row.segment_index, delta) : await keyOf(row.segment_index, target);
  } catch {
    // No video for this frame (or the server did not answer): from picture to picture.
    step(delta);
    return false;
  }
  // The reviewer opened another frame meanwhile.
  if (state.currentKey !== row.frame_key) return false;
  if (key === null) {
    state.notice = {key: row.frame_key, text: delta > 0 ? 'The last frame of the drive.' : 'The first frame of the drive.',
      error: false};
    return false;
  }
  openFrame(key);
  const opened = rowOf(key);
  const at = opened === null ? null : decodedOf(opened);
  if (opened !== null && at !== null) preload(opened.segment_index, at, delta);
  return true;
}

/** After Confirm on a frame of the video between the seconds: the next frame of the video, with what was saved; saved
 * with no sign (``noSign``), the picture of the next second. */
async function nextVideoFrame(savedText: string, noSign: boolean): Promise<void> {
  if (noSign) {
    const target = walking ? null : secondTarget(state.currentKey, 1);
    if (target !== null) openFrame(target);
    if (state.currentKey !== null) {
      state.notice = {key: state.currentKey, text: savedText + (target !== null ? ' Here is the next second.' : ''),
        error: false};
    }
    return;
  }
  const moved = !walking && await moveOne(1);
  if (state.currentKey === null) return;
  state.notice = {key: state.currentKey, text: savedText + (moved ? ' Here is the next frame of the video.' : ''),
    error: false};
}

/** Confirm goes on frame by frame through the video (seconds.ts): with To review and All, in a segment whose video the
 * server has. */
function walksVideo(row: FrameRow): boolean {
  return Boolean(report.apiBase) && (state.filter === 'todo' || state.filter === 'all')
    && typeof row.decoded_frame_index === 'number' && hasVideo(row.segment_index);
}

/** nextInVideo, once the server has said how many frames the video of the frame found has when a frame not saved is in
 * the last second of a segment not counted yet (the frames of its GOP give it). */
async function nextToConfirm(key: string): Promise<string | null> {
  let found = nextInVideo(key, false);
  const row = found === null || !isVirtual(found) ? null : rowOf(found);
  if (row === null || typeof row.decoded_frame_index !== 'number' || report.videos?.[row.segment_index] != null) {
    return found;
  }
  // The picture at 1 a second of its cell is in the video: its GOP says how long the video is.
  const first = Math.floor(row.decoded_frame_index / CAMERA_FPS) * CAMERA_FPS;
  try {
    await gopOf(row.segment_index, first);
    found = nextInVideo(key, false);
  } catch {
    // Kept as found: the frame shows that it cannot be read.
  }
  return found;
}

/** After Confirm (and Confirm on a frame already confirmed, unchanged: ``savedText`` "Already confirmed."): the next
 * frame, confirmed or not, with To review as with All (going back over confirmed frames, Enter goes through each of
 * them, it never jumps to the next one not confirmed), through the frames of the video (seconds.ts) where
 * the server has it, frame by frame while they have a sign and on to the next second after one with no sign;
 * elsewhere a frame of the video that was not saved (``fromVideo``) goes on to the next frame of the video (with no
 * sign to the next second), a frame of the report to the next one of the filter (navigation.ts). */
export async function afterConfirm(key: string, savedText: string, fromVideo = false): Promise<void> {
  const row = rowOf(key);
  const noSign = (state.edits[key]?.detections.length ?? 0) === 0;
  if (row === null || !walksVideo(row)) {
    if (fromVideo) await nextVideoFrame(savedText, noSign);
    else advanceAfterConfirm(key, savedText);
    return;
  }
  if (walking) {
    state.notice = {key, text: savedText, error: false};
    return;
  }
  const next = await nextToConfirm(key);
  // The reviewer opened another frame meanwhile.
  if (state.currentKey !== key) return;
  if (next === null) {
    advanceAfterConfirm(key, savedText);
    return;
  }
  openFrame(next);
  const opened = rowOf(next);
  const following = opened !== null && opened.segment_index === row.segment_index
    && opened.decoded_frame_index === (row.decoded_frame_index ?? NaN) + 1;
  const where = following ? ' Here is the next frame of the video.'
    : opened !== null && !isBetweenFrame(opened) ? ' Here is the next second.' : ' Here is the next frame to review.';
  state.notice = {key: next, text: savedText + where, error: false};
  const at = opened === null ? null : decodedOf(opened);
  if (opened !== null && at !== null) {
    void gopOf(opened.segment_index, at).then(() => preload(opened.segment_index, at, 1)).catch(() => undefined);
  }
}

/** ← → ‹ ›: the frame of the video just before or after (20 a second), with the review server; the picture just
 * before or after offline. Steps pressed while a GOP is decoded wait their turn. */
export function stepFrame(delta: number): void {
  if (!report.apiBase) {
    step(delta);
    return;
  }
  queued = Math.max(-MAX_QUEUED, Math.min(MAX_QUEUED, queued + delta));
  if (walking) return;
  walking = true;
  void (async () => {
    try {
      while (queued !== 0) {
        const move = Math.sign(queued);
        queued -= move;
        if (!await moveOne(move)) queued = 0;
      }
    } finally {
      walking = false;
    }
  })();
}

/** Another drive on the same page (embed.ts): the GOPs and the steps of the one before are forgotten. */
export function resetVideo(): void {
  gops.clear();
  asked.clear();
  preloaded.clear();
  queued = 0;
}
