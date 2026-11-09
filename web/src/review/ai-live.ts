// Review with AI at work on this drive (panel.Panel.ai_live): the page asks every LIVE_MS how far the AI is and for
// the frames saved since the newest it has (GET /api/route/<id>/live?since=), and shows each as it comes, in the
// timeline and on the photo; a frame being changed here keeps the reviewer's changes. Following, it opens each frame
// as the AI does it; another frame opened stops that. With no AI at work on the drive it asks every IDLE_MS, so a
// review started later shows up too.
import {ref} from 'vue';
import type {AiLive} from '../common/panel-types';
import {fromServer} from './frames';
import {openFrame} from './navigation';
import {call} from './server';
import {report, state, store} from './store';
import type {ServerReview} from './types';

const LIVE_MS = 4000;
const IDLE_MS = 20000;

interface LiveAnswer {
  run_id: string;
  reviews: ServerReview[];
  ai: AiLive | null;
}

/** The AI at work on this drive, or null. */
export const aiLive = ref<AiLive | null>(null);
/** The page opens each frame as the AI does it. */
export const following = ref(false);
let followed: string | null = null;
let timer: number | undefined;
// Tells the drive asked about from a later one (resetAiLive): an answer for the drive before is dropped.
let generation = 0;

function follow(key: string | null | undefined): void {
  if (!key) return;
  followed = key;
  if (state.currentKey !== key) openFrame(key, false);
}

/** Follow the AI, or stop. */
export function toggleFollow(): void {
  following.value = !following.value;
  if (following.value) follow(aiLive.value?.frame_key);
}

async function ask(): Promise<void> {
  const apiBase = report.apiBase;
  const mine = generation;
  if (!apiBase) return;
  let next = IDLE_MS;
  try {
    const since = state.newestSave;
    const answer = await call<LiveAnswer>('GET', apiBase + '/live' + (since ? '?since=' + encodeURIComponent(since) : ''));
    if (mine !== generation) return;
    if (answer.run_id !== report.runId) return;
    for (const item of answer.reviews) {
      if (state.newestSave === null || item.updated_at > state.newestSave) state.newestSave = item.updated_at;
      // Changed here and not saved yet: the reviewer's (their save then meets the new revision and says so).
      if (state.edits[item.frame_key]?.saved === false) continue;
      state.edits[item.frame_key] = fromServer(item);
    }
    if (answer.reviews.length > 0) store();
    aiLive.value = answer.ai;
    if (answer.ai !== null) next = LIVE_MS;
    if (following.value) {
      // Another frame opened by the reviewer: they look elsewhere now.
      if (answer.ai === null || (followed !== null && state.currentKey !== followed)) following.value = false;
      else follow(answer.ai.frame_key);
    }
  } catch {
    // The network or a restart of the site: asked again later.
  }
  if (mine === generation) timer = window.setTimeout(() => void ask(), next);
}

/** Stops asking and forgets the drive (the editor on the Signs side of Speed review opens another one,
 * review/embed.ts); watchAiLive starts again for the next. */
export function resetAiLive(): void {
  generation += 1;
  window.clearTimeout(timer);
  timer = undefined;
  aiLive.value = null;
  following.value = false;
  followed = null;
}

/** Starts asking, once the saved reviews are read (state.newestSave). */
export function watchAiLive(): void {
  if (!report.apiBase || report.readOnly || timer !== undefined) return;
  timer = window.setTimeout(() => void ask(), 0);
}
