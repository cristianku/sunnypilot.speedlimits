// The review server: reading the saved reviews of the route, sending each confirmation, retrying the ones that
// did not reach it. Not used offline (the static report, apiBase null).
import {messageOf} from '../common/api';
import {dropStaleKind} from './catalog';
import {clone} from './dom';
import {countryOf, fromServer, unsavedKeys} from './frames';
import {insertRow, isVirtual, report, rowOf, state, store} from './store';
import type {FrameRow, Problem, ReviewsAnswer, SaveAnswer, Suspicious} from './types';

/** A failed call; retry is true when trying again later can help (no network, server error); status and code: the
 * server's answer (0 and '' when none came). */
export class SaveError extends Error {
  constructor(message: string, readonly retry: boolean, readonly status = 0, readonly code = '') {
    super(message);
  }
}

/** What the page says when a frame was saved elsewhere meanwhile (another tab, another person): it shows that save. */
export const CONFLICT_TEXT = 'Changed elsewhere meanwhile: this frame now shows what is saved. Check it, then confirm again.';

// Not common/api.ts: a signed-out page must keep its unsaved confirmations on screen (no jump to the sign-in
// page), and the messages say whether the save is retried.
export async function call<T>(method: 'GET' | 'POST', url: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {'Accept': 'application/json'};
  const options: RequestInit = {method, headers, credentials: 'same-origin', cache: 'no-store'};
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  let response: Response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new SaveError('server unreachable.', true);
  }
  let result: unknown = null;
  try {
    result = await response.json();
  } catch {
    result = null;
  }
  if (!response.ok) {
    const problem = result as Problem | null;
    throw new SaveError(problem && problem.message ? problem.message : 'HTTP error ' + response.status + '.',
      response.status >= 500, response.status, problem && typeof problem.error === 'string' ? problem.error : '');
  }
  return result as T;
}

/** The suspicious flags change with the confirmations (rule d): the server sends them again. */
export function takeSuspicious(result: {suspicious?: unknown} | null): void {
  const flags = result?.suspicious;
  if (flags && typeof flags === 'object' && !Array.isArray(flags)) state.suspicious = flags as Suspicious;
}

/** The reviewer name in the top bar. */
export function reviewerName(): string {
  return state.reviewer.trim();
}

// POST one confirmed frame. True when the server stored it. Confirm goes on at once, without waiting for it
// (confirm.ts), and Apply previous sends the boxes of a frame still on its way (carry.ts).
export async function send(key: string): Promise<boolean> {
  const s = state.edits[key];
  const apiBase = report.apiBase;
  if (!apiBase || !s || state.inFlight.has(key)) return false;
  const version = s.version || 0;
  // The drive whose frame this is: a page that changes drive (Speed review's editor, embed.ts) never takes the answer
  // into another drive's frame of the same key.
  const runId = report.runId;
  state.inFlight.add(key);
  state.lastSave = {key, result: 'saving', revision: s.revision || 0};
  let ok = false;
  let conflict = false;
  // A frame edited before the kind was dropped with the family (it would be refused at every retry).
  s.detections.forEach(dropStaleKind);
  // A frame of the video between the seconds is a picture of the report from now on, as it will be once saved
  // (review_service.video_frame_row): the frame after it has it as the frame before (carry.ts previousRow), and the
  // walk of Confirm goes on from it (seconds.ts nextInVideo).
  const added = isVirtual(key) ? rowOf(key) : null;
  if (added !== null) insertRow(added);
  try {
    const result = await call<SaveAnswer>('POST', apiBase + '/frame/' + key, {
      run_id: report.runId,
      reviewer: s.reviewer || reviewerName(),
      confirmed: true,
      detections: s.detections,
      site_group: s.site_group === undefined ? null : s.site_group,
      encounter_id: s.encounter_id === undefined ? null : s.encounter_id,
      country: countryOf(rowOf(key) ?? {country_proposed: null} as FrameRow, s),
      users_complete: s.users_complete === true,
      base_revision: s.revision || 0,
    });
    if (report.runId !== runId) return true;
    if (state.edits[key] === s && (s.version || 0) === version) {
      state.edits[key] = fromServer(result.review);
    } else {
      // Edited again while saving: keep the new edits on top of the saved revision.
      s.revision = result.review.revision;
      s.server_confirmed = result.review.confirmed;
      s.base = clone(result.review);
    }
    takeSuspicious(result);
    ok = true;
  } catch (error) {
    if (report.runId !== runId) return false;
    s.error = messageOf(error);
    s.retry = error instanceof SaveError && error.retry;
    conflict = error instanceof SaveError && error.status === 409 && error.code === 'revision_conflict';
  } finally {
    if (report.runId !== runId) return ok;
    state.inFlight.delete(key);
    state.saveCount += 1;
    const saved = ok ? state.edits[key] : s;
    state.lastSave = {key, result: ok ? 'ok' : 'error', revision: saved?.revision || 0};
    store();
  }
  // Saved elsewhere on a newer revision: never sent again on top of it; the frame is read again (loadServer replaces
  // the working copy made on the older one) and the reviewer is told.
  if (conflict) {
    await loadServer();
    state.notice = {key, text: CONFLICT_TEXT, error: true};
  }
  return ok;
}

/** Sends the frames confirmed and not saved; all: also those whose last error will not go away by itself. */
export async function retryUnsaved(all: boolean): Promise<void> {
  if (!report.apiBase || state.retrying) return;
  state.retrying = true;
  try {
    for (const key of unsavedKeys()) {
      if (all || state.edits[key]?.retry !== false) await send(key);
    }
  } finally {
    state.retrying = false;
  }
}

/** The reviews saved on the server, merged with what this browser kept. */
export async function loadServer(): Promise<void> {
  const apiBase = report.apiBase;
  if (!apiBase) return;
  const runId = report.runId;
  try {
    const body = await call<ReviewsAnswer>('GET', apiBase + '/reviews');
    // Another drive meanwhile (Speed review's editor, embed.ts): this answer is not its own.
    if (report.runId !== runId || report.apiBase !== apiBase) return;
    if (body.run_id !== report.runId) {
      state.serverWarning = 'The report of this route was made again: reload the page.';
      state.server = 'stale';
      return;
    }
    const onServer = new Set<string>();
    for (const item of body.reviews) {
      onServer.add(item.frame_key);
      if (state.newestSave === null || item.updated_at > state.newestSave) state.newestSave = item.updated_at;
      const local = state.edits[item.frame_key];
      // Unsaved local edits survive only if they were made on top of the version the server still has.
      if (local && local.saved === false && (local.revision || 0) === item.revision) {
        local.server_confirmed = item.confirmed;
        local.base = clone(item);
      } else {
        state.edits[item.frame_key] = fromServer(item);
      }
    }
    for (const row of report.rows) {
      const key = row.frame_key;
      const s = state.edits[key];
      if (!s || onServer.has(key)) continue;
      s.server_confirmed = false;
      s.base = null;
      // Confirmed here but unknown to the server (older offline reviews): send them.
      if (s.review_state === 'reviewed' && s.saved !== false) {
        s.saved = false;
        s.revision = 0;
      } else if (s.saved === undefined) {
        s.saved = true;
      }
    }
    takeSuspicious(body);
    state.serverWarning = '';
    state.server = 'ready';
    store();
    void retryUnsaved(true);
  } catch (error) {
    if (report.runId !== runId) return;
    state.serverWarning = 'Reviews not read from the server: ' + messageOf(error)
      + ' The changes stay in this browser.';
    state.server = 'error';
  }
}
