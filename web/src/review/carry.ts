// "Apply previous": the boxes confirmed on the frame before, carried onto the frame on screen by the server
// (review_service.apply_previous). A box of this frame that is the same sign takes the previous labels; a sign
// not found here gets a box where it should be, marked "check": Confirm refuses the frame until the reviewer
// has looked at each one (moved it onto the sign, pressed "Here", or deleted it). Pressing A is that look: once the fit
// has run on the carried boxes without an error, they count as checked and the frame stays on screen, not saved, to be
// looked at, its first sign selected so that its zoom opens inside the photo; A again on it saves it and goes on, as
// Enter does. The boxes of the frame before show at once, the first one zoomed, while the server carries them.
import {fitLocal} from './ai';
import {confirmFrame} from './confirm';
import {refuseChange} from './busy';
import {nextTick, ref} from 'vue';
import {messageOf} from '../common/api';
import {clone} from './dom';
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

/** The frame on which A carried the boxes: A again on it saves it and goes on (confirmFrame). */
export const carried = ref<string | null>(null);

export function carryState(row: FrameRow): CarryState {
  if (!report.apiBase) return {shown: false, enabled: false, title: ''};
  if (state.inFlight.has(row.frame_key)) return {shown: true, enabled: false, title: 'Saving…'};
  const before = previousRow(row);
  if (before === null) return {shown: true, enabled: false, title: 'No frame just before this one.'};
  if (!isConfirmed(before.frame_key) && !savingConfirmed(before.frame_key)) {
    return {shown: true, enabled: false, title: 'The previous frame (' + before.frame_key + ') is not confirmed yet.'};
  }
  return {
    shown: true,
    enabled: true,
    title: carried.value === row.frame_key ? 'Save this frame and go on, as Enter (A)'
      : 'The signs confirmed on ' + before.frame_key + ', carried onto this frame and fitted; A again saves and goes on (A)',
  };
}

/** Confirmed on this page and still on its way to the server: Apply previous sends its boxes along (applyPrevious). */
function savingConfirmed(key: string): boolean {
  return state.inFlight.has(key) && state.edits[key]?.review_state === 'reviewed';
}

// The frame whose boxes Apply previous is carrying, and A pressed again meanwhile: it saves once they are there.
let carrying: string | null = null;
let saveAfterCarry = false;

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
  if (!row || !apiBase) return;
  // A, A pressed quickly: the second one saves once the boxes are carried.
  if (carrying === row.frame_key) {
    saveAfterCarry = true;
    return;
  }
  if (refuseChange()) return;
  const carry = carryState(row);
  if (!carry.enabled) {
    // Why not, said where it is seen (a tooltip never shows on a touch screen).
    state.notice = {key: row.frame_key, text: 'Apply previous: ' + carry.title, error: false, warn: true};
    return;
  }
  const key = row.frame_key;
  if (carried.value === key) {
    carried.value = null;
    await confirmFrame();
    return;
  }
  const work = workOf(row);
  const byHand = new Set(work.detections.filter(isPlacedByHand).map((d) => JSON.stringify(d.bbox_xyxy)));
  // The frame before, confirmed and still on its way to the server: its boxes go with the request, nothing waits.
  const before = previousRow(row);
  const previous = before !== null && savingConfirmed(before.frame_key)
    ? {frame_key: before.frame_key, detections: workOf(before).detections} : undefined;
  carrying = key;
  saveAfterCarry = false;
  const sent = work.detections;
  let answered = false;
  // At once, before the server answers: the boxes of the frame before shown here, the first selected so that its zoom
  // opens; the server's carried boxes take their place.
  const shown = before !== null ? workOf(before).detections : [];
  if (shown.length > 0) {
    work.detections = shown.map((d) => ({...clone(d), check_position: true}));
    state.selected = 0;
    state.selectedByUser = false;
    state.drawing = false;
  }
  try {
    const answer = await call<CarryAnswer>('POST', apiBase + '/frame/' + key + '/apply-previous',
      {run_id: report.runId, detections: sent, previous});
    // The reviewer may have moved on while the server answered.
    if (state.currentKey !== key) {
      work.detections = sent;
      return;
    }
    // Every box comes back anew: one placed by hand that the carry kept where it was is still placed by hand.
    for (const d of answer.detections) if (byHand.has(JSON.stringify(d.bbox_xyxy))) markPlacedByHand(d, true);
    work.detections = answer.detections;
    answered = true;
    // The first box to check is selected, else the first sign carried: its zoom opens inside the photo.
    const first = uncheckedIndices(work)[0];
    state.selected = first ?? (work.detections.length > 0 ? 0 : -1);
    state.selectedByUser = first !== undefined;
    state.drawing = false;
    changed(row);
    if (first !== undefined) void nextTick(revealEditor);
    const message = {key, text: answer.message, error: false, warn: answer.to_check.length > 0};
    state.notice = message;
    // The carried boxes go straight onto their signs, as C does, but those placed by hand stay where they were put;
    // its message after the one of the carry.
    await fitLocal('carried');
    const fit = state.notice;
    if (fit && fit !== message && fit.key === key) {
      state.notice = {...fit, text: message.text + ' ' + fit.text, warn: message.warn || Boolean(fit.warn)};
    }
    // The fit without error: every carried box counts as checked (a box the fit kept stays where the carry put it) and
    // the frame stays on screen, not saved, with no message: A again (or Enter) saves it and goes on.
    if (state.currentKey !== key || state.notice?.error) return;
    for (const d of work.detections) delete d.check_position;
    if (state.selected < 0 || state.selected >= work.detections.length) state.selected = work.detections.length > 0 ? 0 : -1;
    state.notice = null;
    carried.value = key;
    if (saveAfterCarry) {
      carrying = null;
      carried.value = null;
      await confirmFrame();
    }
  } catch (error) {
    if (!answered) work.detections = sent;
    state.notice = {key, text: 'Apply previous: ' + messageOf(error), error: true};
  } finally {
    if (carrying === key) carrying = null;
    saveAfterCarry = false;
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

