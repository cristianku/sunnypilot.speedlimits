// Sign review's editor on another page: the Signs side of Speed review (control/SpeedReviewDrive.vue) edits the frames
// of Sign review of the drive it plays with the same modules and components as the page of Sign review (store, server,
// navigation, video, Stage, StageBar, Inspector), saved through the same endpoint (POST /api/route/<drive>/frame/...,
// run_id and base_revision, a 409 read again). Its data: GET /api/route/<drive>/page, the payload the page of Sign
// review embeds without the menu of the drives. The modules hold one drive at a time: a new drive resets them first.
// The frame of the player is the frame edited: frameKeyAt gives the key of Sign review of a frame of the road video
// exactly as the server does (review_service.video_gop), never the nearest second. As on the page of Sign review the
// drive is followed while Propose again works on it (busy.ts) and while the AI reviews it (ai-live.ts), and the
// administrator has Ask AI, Fit with AI, the car's networks and the last "Try without saving" (ai.ts, car.ts): all
// forgotten when another drive opens, since the keys of the frames are the same on every drive.
import {api} from '../common/api';
import type {AuthState} from '../common/auth-types';
import {isGuest} from '../common/guest';
import {loadAiState, resetAi} from './ai';
import {resetAiLive, watchAiLive} from './ai-live';
import {resetBusy, watchBusy} from './busy';
import {resetCar} from './car';
import {openFrame} from './navigation';
import {READ_INDEX_BASE} from './order';
import {hasVideo} from './seconds';
import {loadServer, retryUnsaved, SaveError, call} from './server';
import {initStore, report, resetStore, restore, rowOf, state} from './store';
import type {FrameRow, Payload} from './types';
import {resetVideo} from './video';

/** The drive whose editor the modules hold now ('' none), and a counter that tells a load from a later one. */
let loadedRoute = '';
let generation = 0;

/** Where the pictures of the drive are: its page of Sign review (preview_path is relative to it). */
export function pictureBaseOf(route: string): string {
  return '/route/' + encodeURIComponent(route) + '/';
}

/** Reads the drive's frames of Sign review and what the server has saved of them. 'ready': the editor can open;
 * 'none': this server or this drive has none (an older server answers 404: the page stays read only, quietly);
 * 'stale': another load started meanwhile. Throws on any other failure. */
export async function loadSignEditor(route: string): Promise<'ready' | 'none' | 'stale'> {
  const mine = ++generation;
  let payload: Partial<Payload>;
  try {
    payload = await call<Partial<Payload>>('GET', '/api/route/' + encodeURIComponent(route) + '/page');
  } catch (problem) {
    if (problem instanceof SaveError && [403, 404, 405].includes(problem.status)) return mine === generation ? 'none' : 'stale';
    throw problem;
  }
  if (mine !== generation) return 'stale';
  if (!payload.server || !Array.isArray(payload.frames) || payload.frames.length === 0) return 'none';
  forget();
  initStore(payload, pictureBaseOf(route));
  // What this browser kept of this drive's run (its unsaved changes), and the filter of Sign review: with To review,
  // Confirm goes on to the next frame not confirmed.
  restore();
  loadedRoute = route;
  // The confirmations carry the signed-in person, never a name typed in another page (nor an AI's).
  if (state.reviewer.startsWith('ai/')) state.reviewer = '';
  try {
    const auth = await api<AuthState>('/api/auth/state');
    if (auth.user && mine === generation) state.reviewer = auth.user;
  } catch {
    // ReviewToolbar asks again; Confirm asks for a name while there is none.
  }
  if (mine !== generation) return 'stale';
  state.server = 'loading';
  // A drive Propose again works on is read only: followed until it is done (the banner of the inspector says so).
  watchBusy();
  // The reviewer chosen in Settings and the drive's last "Try without saving" (its filter AI trial): the administrator's.
  if (!isGuest()) void loadAiState();
  await loadServer();
  if (mine !== generation) return 'stale';
  // The AI's saves on this drive as they come (asked once the saved reviews are read).
  watchAiLive();
  return 'ready';
}

/** The drive again, on the frame open (the button of the banner once Propose again is done with it): its new
 * proposals, to review. The whole page is read again where the editor has no drive. */
export async function reloadSignEditor(): Promise<void> {
  const route = loadedRoute;
  const key = state.currentKey;
  if (route === '') {
    window.location.reload();
    return;
  }
  try {
    if (await loadSignEditor(route) === 'ready' && key !== null && rowOf(key) !== null) openFrame(key, false);
  } catch {
    window.location.reload();
  }
}

/** Forgets the drive: what was on screen, its frames of the video, its GOPs. Its unsaved changes stay in this
 * browser (store.ts store) and come back with it. */
export function stopSignEditor(): void {
  // A load still on its way finds itself stale.
  generation += 1;
  forget();
}

function forget(): void {
  loadedRoute = '';
  resetBusy();
  resetAiLive();
  resetAi();
  resetCar();
  resetStore();
  resetVideo();
}

/** Retries the confirmations that did not reach the server, every 30 s and when the network comes back, as the page of
 * Sign review does; the returned function stops it. */
export function keepSaving(): () => void {
  const retryLater = (): void => void retryUnsaved(false);
  const timer = window.setInterval(retryLater, 30000);
  window.addEventListener('online', retryLater);
  return () => {
    window.clearInterval(timer);
    window.removeEventListener('online', retryLater);
  };
}

/** The frame of its segment's road video a row of Sign review is (its frame in that minute's fcamera.hevc), or null. */
export function videoFrameOf(row: FrameRow): number | null {
  if (typeof row.decoded_frame_index === 'number') return row.decoded_frame_index;
  if (typeof row.media_time_s === 'number' && Number.isFinite(row.media_time_s)) return Math.round(row.media_time_s * 20);
  return row.output_index < READ_INDEX_BASE ? row.output_index * 20 : null;
}

/** The key of Sign review of frame `frame` of a segment's road video, as the server gives it (video_gop): the report's
 * row with that frame, else a picture at 1 a second without its frame number taken at that frame, else the frame of the
 * video not saved yet (READ_INDEX_BASE + frame; never frame 0, which the server does not take; only where the server has
 * the segment's video). Null where Sign review has nothing of that frame. */
export function frameKeyAt(segment: number, frame: number): string | null {
  if (report.apiBase === null) return null;
  let inSegment = false;
  let loose: string | null = null;
  for (const row of report.rows) {
    if (row.segment_index !== segment) continue;
    inSegment = true;
    if (row.decoded_frame_index === frame) return row.frame_key;
    if (loose === null && typeof row.decoded_frame_index !== 'number' && row.output_index < READ_INDEX_BASE
      && videoFrameOf(row) === frame) loose = row.frame_key;
  }
  if (loose !== null) return loose;
  // A frame not saved yet only where the server has the segment's road video (seconds.ts hasVideo).
  return inSegment && frame > 0 && hasVideo(segment) ? segment + '/' + (READ_INDEX_BASE + frame) : null;
}
