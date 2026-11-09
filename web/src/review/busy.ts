// A drive Propose again works on (nnslr_tools.reproposing) is read only: nothing on it is confirmed or changed until
// its new proposals are in (the server refuses a save too). The drive at work shows the new proposals made so far,
// live: every LIVE_MS the page asks for the frames proposed since and adds them to the timeline; on the newest frame
// it follows, each new one shown as it comes (another frame opened stops it, the newest again resumes it). The banner
// shows how far the drive is and, when its frames come in slow steps (Qwen3-VL answers 4 at a time), how far the
// step at work is: an estimate from the steps before, ticking every TICK_MS. A drive waiting its turn shows its
// current report and asks every INTERVAL_MS. When the drive is done the page says so: a reload shows the new
// proposals, to review.
import {ref} from 'vue';
import {thousands} from '../common/format';
import {durationText} from '../control/jobs';
import {currentRow} from './frames';
import {openFrame} from './navigation';
import {call} from './server';
import {appendRows, report, state} from './store';
import type {Busy, FrameRow} from './types';

const INTERVAL_MS = 15000;
const LIVE_MS = 4000;
const TICK_MS = 250;
const READ_ONLY_TEXT = 'Read only while Propose again works on this drive.';
let timer: number | undefined;
let ticker: number | undefined;
// The clock of the step at work: when the server said how long ago the last step was written, and now.
let answeredAt = performance.now();
const now = ref(answeredAt);

/** True (and says why under the photo) when the drive is read only: the change asked for is not made. */
export function refuseChange(): boolean {
  if (!report.readOnly) return false;
  const row = currentRow();
  if (row) state.notice = {key: row.frame_key, text: READ_ONLY_TEXT, error: true};
  return true;
}

function plural(count: number, word: string): string {
  return thousands(count) + ' ' + word + (count === 1 ? '' : 's');
}

/** The banner of a read-only page. */
export interface Banner {
  title: string;
  text: string;
  /** The label of the button reloading the page, '' for none. */
  reload: string;
  /** At work: how far the drive is, 0-100; null otherwise. */
  percent: number | null;
  /** The frames being proposed now ("Qwen3-VL looks at 4 frames at once: now", "441–444") and how far, 0-99 (an
   * estimate; over: it takes longer than the steps before); null when not shown. */
  step: {text: string; range: string; percent: number; over: boolean} | null;
}

/** "440 of 2,160 frames · about 2 h 50 min left". */
function howFar(busy: Busy): string {
  const done = busy.done ?? 0;
  return thousands(done) + (busy.total ? ' of ' + thousands(busy.total) : '') + ' frames'
    + (busy.left_seconds ? ' · ' + durationText(busy.left_seconds) + ' left' : '');
}

function percentOf(busy: Busy): number | null {
  return busy.total ? Math.min(100, Math.floor((busy.done ?? 0) * 100 / busy.total)) : null;
}

function stepOf(busy: Busy, who: string): Banner['step'] {
  const step = busy.step;
  if (!step || step.seconds <= 0 || step.frames < 1) return null;
  const elapsed = step.since + (now.value - answeredAt) / 1000;
  const first = (busy.done ?? 0) + 1;
  const text = step.frames > 1 ? who + ' looks at ' + step.frames + ' frames at once: now' : 'Now frame';
  const range = thousands(first) + (step.frames > 1 ? '–' + thousands(first + step.frames - 1) : '');
  return {text, range, percent: Math.max(0, Math.min(99, Math.floor(elapsed * 100 / step.seconds))),
    over: elapsed > step.seconds};
}

export function busyBanner(): Banner | null {
  if (!report.readOnly) return null;
  const busy = state.busy;
  if (busy === null) {
    return {title: 'Propose again is done', text: 'This drive has its new proposals now.', reload: 'Review them',
      percent: null, step: null};
  }
  const who = busy.proposer ?? 'Propose again';
  if (busy.phase === 'waiting') {
    const ahead = busy.ahead ?? 0;
    return {title: 'Read only · waiting for ' + who,
      text: who + ' will propose the signs of this drive again' + (ahead > 0 ? ' after ' + plural(ahead, 'drive')
        + ' before it' : '') + '. These proposals will be replaced, so nothing can be confirmed until that is done.',
      reload: '', percent: null, step: null};
  }
  const started = (busy.done ?? 0) > 0;
  return {title: (report.atWork ? 'Live · ' : 'Read only · ') + who + ' at work',
    text: started ? howFar(busy) : 'Starting: the first frames come in a minute.',
    reload: report.atWork ? '' : 'Show its proposals', percent: started ? percentOf(busy) : null, step: stepOf(busy, who)};
}

/** The newest frame proposed on the drive at work, null before the first. */
function newestKey(): string | null {
  return report.rows[report.rows.length - 1]?.frame_key ?? null;
}

/** On the drive at work, the page shows a frame older than the newest: it does not follow. */
export function behind(): boolean {
  return report.atWork && newestKey() !== null && state.currentKey !== newestKey();
}

export function showNewest(): void {
  const key = newestKey();
  if (key !== null) openFrame(key);
}

let asking = false;
// Tells the drive asked about from a later one (resetBusy): an answer for the drive before is dropped.
let generation = 0;
async function ask(): Promise<void> {
  if (!report.apiBase || asking) return;
  asking = true;
  const mine = generation;
  try {
    const url = report.apiBase + '/busy' + (report.atWork ? '/' + report.rows.length : '');
    const answer = await call<{busy: Busy | null; run_id?: string; rows?: FrameRow[]}>('GET', url);
    if (mine !== generation) return;
    answeredAt = performance.now();
    now.value = answeredAt;
    state.busy = answer.busy;
    if (answer.busy === null) stop();
    if (answer.rows && answer.run_id === report.runId && answer.rows.length > 0) {
      const follow = state.currentKey === newestKey();
      appendRows(answer.rows);
      const key = newestKey();
      // Not on a phone's photo: the page stays where it was scrolled to.
      if (follow && key !== null && key !== state.currentKey) openFrame(key, false);
    }
  } catch {
    // The network or a restart of the site: asked again at the next round.
  } finally {
    asking = false;
  }
}

function stop(): void {
  window.clearInterval(timer);
  window.clearInterval(ticker);
}

/** Stops following and forgets the drive (the editor on the Signs side of Speed review opens another one,
 * review/embed.ts); watchBusy starts again for the next. */
export function resetBusy(): void {
  generation += 1;
  stop();
  timer = undefined;
  ticker = undefined;
}

/** Follows a read-only drive until Propose again is done with it. */
export function watchBusy(): void {
  if (!report.readOnly || !report.apiBase || timer !== undefined) return;
  timer = window.setInterval(() => void ask(), report.atWork ? LIVE_MS : INTERVAL_MS);
  if (report.atWork) ticker = window.setInterval(() => { now.value = performance.now(); }, TICK_MS);
}
