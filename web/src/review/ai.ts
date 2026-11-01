// The AI reviewer on the review page: "Ask AI" (I) sends the boxes of the frame on screen to the server, which
// asks the model chosen in Settings (POST …/ai-suggest, nnslr_tools.ai_review.suggest_frame) and answers with a
// suggestion for each box. Nothing is saved: Apply makes the suggestion the frame's unsaved edits, and the
// reviewer's Confirm saves them as their own review. The suggestions live in this page only, by frame: an answer
// that comes after the reviewer moved on is kept for its frame. The route's last "Try without saving" (GET
// …/ai-trial) comes in the same way, one ready suggestion per frame it tried, and fills the "AI trial" filter.
// Not offline (the static report has no server).
import {refuseChange} from './busy';
import {reactive} from 'vue';
import {messageOf} from '../common/api';
import type {AiAction, AiState, AiSuggestion, AiTrial} from '../common/ai-types';
import type {Box, Detection, FrameRow} from './types';
import {askConfirm} from '../composables/confirm';
import {clone} from './dom';
import {changed} from './edits';
import {currentRow, startingDetections, workOf} from './frames';
import {call} from './server';
import {report, rowOf, state} from './store';
import {trialKeys} from './trial';

/** What the page has for a frame: nothing, a question on its way, a suggestion, or the error of the last one. */
export type AiPhase = 'none' | 'asking' | 'ready' | 'failed';

export interface AiAsk {
  phase: Exclude<AiPhase, 'none'>;
  /** Tells this question from a later one for the same frame. */
  ticket: number;
  /** When it was asked (ms since the epoch), for the seconds shown while waiting. */
  since: number;
  /** The boxes of the frame when it was asked (JSON): Apply asks first when they changed since. */
  sent: string;
  suggestion: AiSuggestion | null;
  error: string;
  /** From the route's "Try without saving", not asked on this page: the boxes sent were the proposals. */
  trial?: {at: string | null};
}

interface AiPage {
  /** GET /api/ai/state: the reviewer chosen in Settings and whether this browser may ask it; null until read. */
  state: AiState | null;
  /** Why the state could not be read. */
  error: string;
  /** By frame key; never kept in localStorage. */
  asks: Record<string, AiAsk>;
}

export const ai: AiPage = reactive({state: null, error: '', asks: {}});

let tickets = 0;

/** The pill of each verdict in the list, with its colour (pill classes of shared.css). */
export const VERDICT: Record<AiAction | 'delete', [label: string, kind: string]> = {
  keep: ['Keep', 'ok'], correct: ['Correct', 'rev'], add: ['Add', 'imp'], delete: ['Delete', 'bad'],
  unjudged: ['Not judged', 'muted'],
};

let trialRead = false;

/** Reads (again) which AI reviewer the server has and whether this browser may ask it; the first time, the trial. */
export async function loadAiState(): Promise<void> {
  if (!report.apiBase) return;
  if (!trialRead) {
    trialRead = true;
    void loadAiTrial();
  }
  try {
    ai.state = await call<AiState>('GET', '/api/ai/state');
    ai.error = '';
  } catch (error) {
    ai.error = 'The AI reviewer cannot be read: ' + messageOf(error);
  }
}

/** The route's last "Try without saving": a ready suggestion for each frame it tried that nobody confirmed, unless
 * this page already asked the AI about it. Nothing is saved; without a trial nothing changes. */
async function loadAiTrial(): Promise<void> {
  const apiBase = report.apiBase;
  if (!apiBase) return;
  let answer: {trial: AiTrial | null};
  try {
    answer = await call<{trial: AiTrial | null}>('GET', apiBase + '/ai-trial');
  } catch {
    // An older server, or no trial to read: the page works as before.
    return;
  }
  const trial = answer.trial;
  if (trial === null || trial.run_id !== report.runId) return;
  for (const frame of trial.frames) {
    const row = rowOf(frame.frame_key);
    if (!row) continue;
    trialKeys.add(frame.frame_key);
    if (ai.asks[frame.frame_key]) continue;
    ai.asks[frame.frame_key] = {
      phase: 'ready', ticket: ++tickets, since: Date.now(), sent: JSON.stringify(startingDetections(row)), error: '',
      suggestion: {...frame, ok: true, route_id: trial.route_id, run_id: trial.run_id, provider: trial.provider,
        reviewer: trial.reviewer},
      trial: {at: trial.finished_at ?? trial.started_at},
    };
  }
}

/** Why "Ask AI" is off for every frame ('' when it is on). */
export function aiBlocker(): string {
  if (!report.apiBase) return 'Only with the review server.';
  if (state.server === 'stale') return 'The report of this route was made again: reload the page.';
  if (ai.state === null) return ai.error || 'Reading the AI reviewer…';
  if (!ai.state.available) return ai.state.reason || 'Choose an AI reviewer in Settings.';
  if (ai.state.can_ask === false) return ai.state.reason || 'Sign in on the site to use the AI reviewer.';
  return '';
}

export function aiPhase(key: string): AiPhase {
  return ai.asks[key]?.phase ?? 'none';
}

/** Asks the AI about the frame on screen, with its boxes as they are now. */
export async function askAi(): Promise<void> {
  const row = currentRow();
  const apiBase = report.apiBase;
  if (!row || !apiBase || aiBlocker() !== '') return;
  const key = row.frame_key;
  if (aiPhase(key) === 'asking') return;
  const detections = clone(workOf(row).detections);
  const ticket = ++tickets;
  ai.asks[key] = {phase: 'asking', ticket, since: Date.now(), sent: JSON.stringify(detections), suggestion: null, error: ''};
  try {
    const answer = await call<AiSuggestion>('POST', apiBase + '/frame/' + key + '/ai-suggest',
      {run_id: report.runId, detections});
    const ask = ai.asks[key];
    // Dismissed, or asked again, while the model was thinking: that decides.
    if (ask?.ticket !== ticket) return;
    ask.suggestion = answer;
    ask.phase = 'ready';
  } catch (error) {
    const ask = ai.asks[key];
    if (ask?.ticket !== ticket) return;
    ask.error = 'AI: ' + messageOf(error);
    ask.phase = 'failed';
    // The reviewer may have been changed in Settings meanwhile.
    void loadAiState();
  }
}

/** What became of a box sent to a fit: its answer applied ('fitted'); left as it was, with no answer or one within
 * ``within`` of it on every side ('same'); or left alone, since it was moved ('changed') or removed ('removed') while
 * the fit ran. */
type FitOutcome = 'fitted' | 'same' | 'changed' | 'removed';

function settleFit(row: FrameRow, detection: Detection, sent: string, box: Box | null, within = 0): FitOutcome {
  if (!workOf(row).detections.includes(detection)) return 'removed';
  if (JSON.stringify(detection.bbox_xyxy) !== sent) return 'changed';
  const was = detection.bbox_xyxy;
  if (!box || box.every((value, side) => Math.abs(value - (was[side] ?? Infinity)) < within)) {
    // A sign found right under a box carried from the previous frame (a frame of the video, 1/20 s later): it is
    // on its sign, as after "Here".
    if (box && detection.check_position) {
      delete detection.check_position;
      changed(row);
    }
    return 'same';
  }
  detection.bbox_xyxy = box;
  // A box carried from the previous frame is now on its sign: as "Here" does, its check goes.
  delete detection.check_position;
  changed(row);
  return 'fitted';
}

/** How many boxes of a fit had each outcome. */
function countFits(outcomes: FitOutcome[]): Record<FitOutcome, number> {
  const count = {fitted: 0, same: 0, changed: 0, removed: 0};
  for (const outcome of outcomes) count[outcome] += 1;
  return count;
}

/** ", 2 changed meanwhile and left as they are, 1 removed meanwhile": the boxes a fit left alone. */
function meanwhile(count: Record<FitOutcome, number>): string {
  const moved = count.changed ? ', ' + count.changed + ' changed meanwhile and left as '
    + (count.changed === 1 ? 'it is' : 'they are') : '';
  return moved + (count.removed ? ', ' + count.removed + ' removed meanwhile' : '');
}

/** The box being fitted (Fit with AI), by frame key and index; null when none. */
export const fitting = reactive<{key: string | null; index: number}>({key: null, index: -1});

/** Fit with AI (Shift+C): the box the reviewer picked (``selected``: the one selected, as the button of its
 * editor), else every box of the frame (the page selects the first one by itself), sent with only the picture
 * around it (…/ai-fit, ai_review.fit_box), one after the other; each comes back tight, as an unsaved change to
 * confirm. A box moved or removed meanwhile is left as it is. */
export async function fitBox(selected = false): Promise<void> {
  const row = currentRow();
  const apiBase = report.apiBase;
  if (!row || !apiBase || fitting.key !== null || aiBlocker() !== '' || refuseChange()) return;
  const boxes = workOf(row).detections;
  const chosen = selected || state.selectedByUser ? boxes[state.selected] : undefined;
  const targets = chosen ? [chosen] : [...boxes];
  if (targets.length === 0) return;
  const key = row.frame_key;
  fitting.key = key;
  fitting.index = chosen ? state.selected : -1;
  const outcomes: FitOutcome[] = [];
  let message = '';
  try {
    for (const [number, detection] of targets.entries()) {
      if (targets.length > 1) state.notice = {key, text: 'Fitting box ' + (number + 1) + ' of ' + targets.length + '…', error: false};
      const sent = JSON.stringify(detection.bbox_xyxy);
      const answer = await call<{box: Box | null; message: string}>('POST', apiBase + '/frame/' + key + '/ai-fit',
        {run_id: report.runId, box: detection.bbox_xyxy});
      message = answer.message;
      outcomes.push(settleFit(row, detection, sent, answer.box));
    }
    const count = countFits(outcomes);
    const one = count.changed ? ': the box changed meanwhile, left as it is.'
      : count.removed ? ': the box was removed meanwhile.' : '.';
    const text = targets.length === 1 ? message + one
      : count.fitted + ' of ' + targets.length + ' boxes fitted by AI'
        + (count.same ? ', ' + count.same + ' with no sign seen' : '') + meanwhile(count) + '.';
    state.notice = {key, text, error: false, warn: count.fitted < targets.length};
  } catch (error) {
    state.notice = {key, text: 'Fit with AI: ' + messageOf(error), error: true};
    void loadAiState();
  } finally {
    fitting.key = null;
    fitting.index = -1;
  }
}

/** The frame whose boxes the server is fitting (Fit box), and the box's index (-1: all of them); null when none. */
export const localFitting = reactive<{key: string | null; index: number}>({key: null, index: -1});

/** Whether Fit box can be used: only with the review server, and a report that is still the one on the server. */
export function canFitLocally(): boolean {
  return Boolean(report.apiBase) && state.server !== 'stale';
}

/** Fit box (C): the box the reviewer picked (``selected``: the one selected, as the button of its editor), else every
 * box of the frame, fitted to the sign under it on the server's CPU (…/fit, nnslr_tools.box_fit), all in one request
 * and in a fraction of a second, with no AI. Each fitted box is an unsaved change to confirm, as with Fit with AI; a
 * box with no better box found (no sign there, or already right), moved less than a pixel, or moved or removed
 * meanwhile, is left as it is. */
export async function fitLocal(selected = false): Promise<void> {
  const row = currentRow();
  const apiBase = report.apiBase;
  if (!row || !apiBase || !canFitLocally() || localFitting.key !== null || refuseChange()) return;
  const boxes = workOf(row).detections;
  const chosen = selected || state.selectedByUser ? boxes[state.selected] : undefined;
  const targets = chosen ? [chosen] : [...boxes];
  if (targets.length === 0) return;
  const key = row.frame_key;
  localFitting.key = key;
  localFitting.index = chosen ? state.selected : -1;
  const sent = targets.map((detection) => ({detection, box: JSON.stringify(detection.bbox_xyxy)}));
  try {
    const answer = await call<{boxes: (Box | null)[]; ms: number}>('POST', apiBase + '/frame/' + key + '/fit',
      {run_id: report.runId, boxes: targets.map((detection) => detection.bbox_xyxy)});
    // An answer that moves no side by a pixel or more is no change: a box already right (or a confirmed frame)
    // stays as it is, not an edit to confirm again.
    const count = countFits(sent.map(({detection, box}, number) =>
      settleFit(row, detection, box, answer.boxes[number] ?? null, 1)));
    const ms = ' · ' + answer.ms + ' ms.';
    const text = targets.length > 1
      ? count.fitted + ' of ' + targets.length + ' boxes fitted'
        + (count.same ? ', ' + count.same + ' kept as ' + (count.same === 1 ? 'it was' : 'they were') : '')
        + meanwhile(count) + ms
      : count.fitted ? 'Box fitted' + ms : count.same ? 'No better box found: kept as it was' + ms
        : count.changed ? 'The box changed meanwhile: left as it is.' : 'The box was removed meanwhile.';
    // Only a box the fit could not reach is worth a warning; one already right is not.
    state.notice = {key, text, error: false, warn: count.changed + count.removed > 0};
  } catch (error) {
    state.notice = {key, text: 'Fit box: ' + messageOf(error), error: true};
  } finally {
    localFitting.key = null;
    localFitting.index = -1;
  }
}

/** Apply: the suggestion becomes the boxes of the frame on screen, to check and then confirm (nothing is saved). */
export async function applyAi(): Promise<void> {
  const row = currentRow();
  if (!row || refuseChange()) return;
  const key = row.frame_key;
  const ask = ai.asks[key];
  const suggestion = ask?.suggestion;
  if (!ask || ask.phase !== 'ready' || !suggestion) return;
  const work = workOf(row);
  if (JSON.stringify(work.detections) !== ask.sent) {
    const ok = await askConfirm({title: 'Apply the AI suggestion?', action: 'Apply',
      text: 'The boxes of this frame changed after you asked the AI: applying replaces them with its suggestion.'});
    if (!ok || state.currentKey !== key || ai.asks[key]?.ticket !== ask.ticket) return;
  }
  work.detections = clone(suggestion.detections);
  delete ai.asks[key];
  state.selected = work.detections.length > 0 ? 0 : -1;
  state.selectedByUser = false;
  state.drawing = false;
  changed(row);
  state.notice = {key, text: 'AI suggestion applied: check each box, then Confirm.', error: false, warn: true};
}

/** Dismiss: the suggestion (or the error) of the frame on screen goes away; the boxes stay as they are. */
export function dismissAi(): void {
  const key = state.currentKey;
  if (key !== null) delete ai.asks[key];
}
