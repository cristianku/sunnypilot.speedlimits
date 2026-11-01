// Confirm, Withdraw confirmation, Discard changes: what the reviewer decides about the frame on screen. With the
// server a confirmation is saved at once; offline it stays in this browser until the JSONL export.
import {refuseChange} from './busy';
import {nextTick} from 'vue';
import {messageOf} from '../common/api';
import {hourMinute} from '../common/format';
import {duplicatesOf} from './boxes';
import {clone, writeLocal} from './dom';
import {countryOf, currentRow, frameStatus, fromServer, sameAsServer, workOf} from './frames';
import {blockUnchecked} from './carry';
import {advanceAfterConfirm, allConfirmed, allConfirmedText, go} from './navigation';
import {revealEditor} from './scroll';
import {call, reviewerName, send, takeSuspicious} from './server';
import {isVirtual, report, state, store} from './store';
import type {FrameRow, SaveAnswer} from './types';

function needReviewer(key: string): void {
  state.reviewerNeeded = true;
  state.reviewerFocus += 1;
  state.notice = {key, text: 'Type the reviewer’s name or initials, then confirm.', error: true};
  state.lastSave = {key, result: 'need-reviewer', revision: 0};
}

function timeNow(): string {
  return hourMinute(new Date());
}

/** A box that repeats an earlier one (the same sign drawn twice, "same sign as 1" in the list): the first Confirm
 * selects it and asks; Confirm again on the same boxes saves both. True: asked, nothing saved. */
function askAboutDuplicate(row: FrameRow): boolean {
  const work = workOf(row);
  const duplicates = duplicatesOf(work);
  const index = duplicates.findIndex((earlier) => earlier !== null);
  const earlier = duplicates[index];
  if (index < 0 || earlier === null || earlier === undefined) return false;
  // Asked once for these boxes: any change (a box deleted or moved) asks again.
  const asked = row.frame_key + '#' + (work.version || 0);
  if (state.duplicateAsked === asked) return false;
  state.duplicateAsked = asked;
  state.selected = index;
  state.selectedByUser = true;
  state.drawing = false;
  state.notice = {key: row.frame_key, error: true, text: 'Box ' + (index + 1) + ' looks like the same sign as box '
    + (earlier + 1) + ': delete it (✕ or Del), or press Confirm again to keep both.'};
  state.lastSave = {key: row.frame_key, result: 'need-duplicate-check', revision: work.revision || 0};
  void nextTick(revealEditor);
  return true;
}

export async function confirmFrame(): Promise<void> {
  const row = currentRow();
  if (!row || refuseChange()) return;
  const key = row.frame_key;
  if (state.inFlight.has(key)) return;
  // A frame of the video between the seconds becomes a picture of the report only with a sign on it: with no box
  // nothing is saved (the server refuses it too, review_service.video_frame_row).
  if (isVirtual(key) && workOf(row).detections.length === 0) {
    state.notice = {key, error: true, text: 'A frame between the seconds is saved only with a box: draw the sign '
      + 'first (B), or go to the pictures at 1 a second with ⇧ ← →.'};
    return;
  }
  // Already on the server and not changed since: Confirm moves on, it does not sign the frame again
  // (a frame an AI confirmed too: that confirmation is final).
  const current = state.edits[key];
  if (report.apiBase && frameStatus(key) === 'done' && current && sameAsServer(current)) {
    state.lastSave = {key, result: 'unchanged', revision: current.revision || 0};
    go(1);
    if (state.currentKey === key) {
      const text = allConfirmed() ? allConfirmedText() : 'Already confirmed: there are no more frames with this filter.';
      state.notice = {key, text, error: false};
    }
    return;
  }
  // Boxes "Apply previous" placed where a sign should be: the reviewer looks at each before saving.
  if (blockUnchecked(row)) return;
  if (askAboutDuplicate(row)) return;
  const reviewer = reviewerName();
  if (!reviewer) {
    needReviewer(key);
    return;
  }
  writeLocal('nnslr-reviewer', reviewer);
  const s = workOf(row);
  s.review_state = 'reviewed';
  s.reviewer = reviewer;
  s.reviewer_kind = 'human';
  s.review_timestamp = new Date().toISOString();
  s.saved = false;
  s.error = null;
  s.retry = true;
  s.version = (s.version || 0) + 1;
  state.drawing = false;
  if (!report.apiBase) {
    s.saved = true;
    s.base = {
      detections: clone(s.detections),
      site_group: s.site_group === undefined ? null : s.site_group,
      encounter_id: s.encounter_id === undefined ? null : s.encounter_id,
      country: countryOf(row, s),
      reviewer: s.reviewer,
      review_timestamp: s.review_timestamp,
    };
    store();
    advanceAfterConfirm(key, 'Confirmed ' + key + ' in this browser.');
    return;
  }
  state.notice = null;
  store();
  const ok = await send(key);
  if (ok && state.currentKey === key) advanceAfterConfirm(key, 'Saved ' + key + ' at ' + timeNow() + '.');
}

export async function withdrawFrame(): Promise<void> {
  const row = currentRow();
  const apiBase = report.apiBase;
  if (!row || !apiBase || refuseChange()) return;
  const key = row.frame_key;
  if (state.inFlight.has(key)) return;
  const reviewer = reviewerName();
  if (!reviewer) {
    needReviewer(key);
    return;
  }
  // Two presses instead of a browser dialog: the second one, on the same frame, withdraws.
  if (state.withdrawArmed !== key) {
    state.withdrawArmed = key;
    const text = 'Press “Yes, withdraw” to take the confirmation away: the frame will not be imported.';
    state.notice = {key, text, error: false, warn: true};
    return;
  }
  state.withdrawArmed = null;
  const s = workOf(row);
  state.inFlight.add(key);
  try {
    const result = await call<SaveAnswer>('POST', apiBase + '/frame/' + key,
      {run_id: report.runId, reviewer, confirmed: false, base_revision: s.revision || 0});
    const saved = fromServer(result.review);
    state.edits[key] = saved;
    takeSuspicious(result);
    state.notice = {key, text: 'Confirmation withdrawn: the frame will not be imported.', error: false};
    state.lastSave = {key, result: 'withdrawn', revision: saved.revision || 0};
  } catch (error) {
    state.notice = {key, text: 'Withdrawing failed: ' + messageOf(error), error: true};
    state.lastSave = {key, result: 'error', revision: s.revision || 0};
  } finally {
    state.inFlight.delete(key);
    state.saveCount += 1;
    store();
  }
}

// Back to the last confirmed version of the frame (on the server, or in this browser offline), else to the proposals.
export function discardChanges(): void {
  const row = currentRow();
  if (!row) return;
  const key = row.frame_key;
  const s = state.edits[key];
  if (!s || state.inFlight.has(key)) return;
  if (s.base) {
    if (report.apiBase) {
      state.edits[key] = fromServer(s.base);
    } else {
      state.edits[key] = {
        review_state: 'reviewed',
        detections: clone(s.base.detections || []),
        site_group: s.base.site_group === undefined ? null : s.base.site_group,
        encounter_id: s.base.encounter_id === undefined ? null : s.base.encounter_id,
        country: s.base.country ?? null,
        reviewer: s.base.reviewer,
        review_timestamp: s.base.review_timestamp,
        saved: true,
        revision: 0,
        base: s.base,
      };
    }
  } else {
    // Never confirmed: back to the proposals.
    delete state.edits[key];
  }
  state.selected = -1;
  state.notice = null;
  state.withdrawArmed = null;
  state.valueError = false;
  store();
}
