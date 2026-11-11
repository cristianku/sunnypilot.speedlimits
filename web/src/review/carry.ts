// "Apply previous": the boxes confirmed on the frame before, carried onto the frame on screen by the server
// (review_service.apply_previous). A box of this frame that is the same sign takes the previous labels; a sign
// not found here gets a box where it should be, marked "check": Confirm refuses the frame until the reviewer
// has looked at each one (moved it onto the sign, pressed "Here", or deleted it). Pressing A is that look: once the fit
// has run on the carried boxes without an error, the frame is saved, as Enter does, and stays on screen to be looked at
// (→ then A for the next frame of the sign).
import {fitLocal} from './ai';
import {confirmAndStay} from './confirm';
import {refuseChange} from './busy';
import {nextTick} from 'vue';
import {messageOf} from '../common/api';
import {changed, isPlacedByHand, markPlacedByHand} from './edits';
import {currentRow, isConfirmed, workOf} from './frames';
import {revealEditor} from './scroll';
import {call} from './server';
import {anchorIndex, indexOf, isVirtual, report, state} from './store';
import type {CarryAnswer, FrameEdit, FrameRow} from './types';

/** The frame just before in the timeline (this segment or the end of the one before), or null. For a frame of the
 * video not saved (video.ts): the frame of the report just before it. */
export function previousRow(row: FrameRow): FrameRow | null {
  const key = row.frame_key;
  const before = report.rows[isVirtual(key) ? anchorIndex(key) : indexOf(key) - 1];
  if (!before || before.segment_index < row.segment_index - 1) return null;
  return before;
}

export interface CarryState {
  /** Shown at all: only with the review server. */
  shown: boolean;
  enabled: boolean;
  /** Why it is off, or what it does (the button's tooltip). */
  title: string;
}

export function carryState(row: FrameRow): CarryState {
  if (!report.apiBase) return {shown: false, enabled: false, title: ''};
  const before = previousRow(row);
  if (before === null) return {shown: true, enabled: false, title: 'No frame just before this one.'};
  if (!isConfirmed(before.frame_key)) {
    return {shown: true, enabled: false, title: 'The previous frame (' + before.frame_key + ') is not confirmed yet.'};
  }
  if (state.inFlight.has(row.frame_key)) return {shown: true, enabled: false, title: 'Saving…'};
  return {
    shown: true,
    enabled: true,
    title: 'The signs confirmed on ' + before.frame_key + ', carried onto this frame, fitted and saved; the frame stays on screen (A)',
  };
}

/** The boxes waiting for the reviewer to check their position, by index. */
export function uncheckedIndices(work: FrameEdit): number[] {
  const found: number[] = [];
  work.detections.forEach((d, i) => {
    if (d.check_position) found.push(i);
  });
  return found;
}

export async function applyPrevious(): Promise<void> {
  const row = currentRow();
  const apiBase = report.apiBase;
  if (!row || !apiBase || refuseChange()) return;
  const carry = carryState(row);
  if (!carry.enabled) {
    // Why not, said where it is seen (a tooltip never shows on a touch screen).
    state.notice = {key: row.frame_key, text: 'Apply previous: ' + carry.title, error: false, warn: true};
    return;
  }
  const key = row.frame_key;
  const work = workOf(row);
  const byHand = new Set(work.detections.filter(isPlacedByHand).map((d) => JSON.stringify(d.bbox_xyxy)));
  try {
    const answer = await call<CarryAnswer>('POST', apiBase + '/frame/' + key + '/apply-previous',
      {run_id: report.runId, detections: work.detections});
    // The reviewer may have moved on while the server answered.
    if (state.currentKey !== key) return;
    // Every box comes back anew: one placed by hand that the carry kept where it was is still placed by hand.
    for (const d of answer.detections) if (byHand.has(JSON.stringify(d.bbox_xyxy))) markPlacedByHand(d, true);
    work.detections = answer.detections;
    const first = uncheckedIndices(work)[0];
    state.selected = first === undefined ? -1 : first;
    state.selectedByUser = first !== undefined;
    state.drawing = false;
    changed(row);
    if (first !== undefined) void nextTick(revealEditor);
    const carried = {key, text: answer.message, error: false, warn: answer.to_check.length > 0};
    state.notice = carried;
    // The carried boxes go straight onto their signs, as C does, but those placed by hand stay where they were put;
    // its message after the one of the carry.
    await fitLocal('carried');
    const fit = state.notice;
    if (fit && fit !== carried && fit.key === key) {
      state.notice = {...fit, text: carried.text + ' ' + fit.text, warn: carried.warn || Boolean(fit.warn)};
    }
    // The fit without error: every carried box counts as checked (a box the fit kept stays where the carry put it),
    // and the frame is saved as Enter does, but stays on screen with no message: a wrong box is moved and saved again
    // with Enter.
    if (state.currentKey !== key || state.notice?.error) return;
    for (const d of work.detections) delete d.check_position;
    await confirmAndStay();
  } catch (error) {
    state.notice = {key, text: 'Apply previous: ' + messageOf(error), error: true};
  }
}

/** "Here": the reviewer has looked, the carried box is on its sign. */
export function positionChecked(index: number): void {
  const row = currentRow();
  if (!row || refuseChange()) return;
  const d = workOf(row).detections[index];
  if (!d || !d.check_position) return;
  delete d.check_position;
  changed(row);
}

/** Confirm with boxes still to check: the first one is selected and the reason shown. False: none left. */
export function blockUnchecked(row: FrameRow): boolean {
  const work = workOf(row);
  const left = uncheckedIndices(work);
  const first = left[0];
  if (first === undefined) return false;
  const numbers = left.map((i) => i + 1).join(', ');
  const text = (left.length === 1 ? 'Check box ' + numbers : 'Check boxes ' + numbers)
    + ', carried from the previous frame: move ' + (left.length === 1 ? 'it' : 'each')
    + ' onto the sign and press “Here”, or delete it if the sign is gone.';
  state.selected = first;
  state.selectedByUser = true;
  state.notice = {key: row.frame_key, text, error: true};
  state.lastSave = {key: row.frame_key, result: 'need-check', revision: work.revision || 0};
  void nextTick(revealEditor);
  return true;
}

