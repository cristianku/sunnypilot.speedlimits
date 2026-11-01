// Moving between frames: arrows, the frame after a confirmation, the filters, the frame to start from.
import {nextTick} from 'vue';
import {thousands} from '../common/format';
import {readLocal, writeLocal} from './dom';
import {
  aiConfirmed, confirmedFilter, currentRow, frameStatus, isQualityFilter, matchesFilter, segmentSecond, visibleRows, workOf,
} from './frames';
import {revealStage, scrollTimelineToCurrent} from './scroll';
import {anchorIndex, indexOf, report, rowOf, state} from './store';
import type {Filter} from './types';

/** Shift ← → (and ‹ › ← → offline, video.ts): the picture just before or after in the route (a second, or a frame
 * between the seconds of the report), whatever the filter (the filter decides where Confirm goes next). From a frame
 * of the video not saved: the picture just before it, or just after. */
export function step(delta: number): void {
  if (state.currentKey === null) return;
  const index = indexOf(state.currentKey);
  const anchor = anchorIndex(state.currentKey);
  const target = report.rows[index >= 0 ? index + delta : delta < 0 ? anchor : anchor + 1];
  if (target === undefined) return;
  openFrame(target.frame_key);
}

/** The frame before or after in the filter ("Already confirmed · next"). */
export function go(step: number): void {
  const list = visibleRows();
  const index = list.findIndex((row) => row.frame_key === state.currentKey);
  const target = list[index + step];
  if (index < 0 || index + step < 0 || target === undefined) return;
  openFrame(target.frame_key);
}

/** Opens a frame; reveal: on a phone, the photo back in view (not when the frame changes by itself). */
export function openFrame(key: string, reveal = true): void {
  state.currentKey = key;
  state.selected = -1;
  state.drawing = false;
  state.drag = null;
  state.draft = null;
  state.filterNote = '';
  state.withdrawArmed = null;
  const row = rowOf(key);
  // The first box is selected: the keys (speed, road) work at once.
  if (row !== null && workOf(row).detections.length > 0) state.selected = 0;
  // Opened on it, not chosen: Del does not delete it.
  state.selectedByUser = false;
  state.typing = '';
  writeLocal(report.lastKeyStorage, key);
  // Once the page shows the frame: its tick in sight, and on a phone the photo.
  void nextTick(() => {
    scrollTimelineToCurrent();
    if (reveal) revealStage();
  });
}

// The frame to show after a confirmation: the next one still to do under the current filter.
function nextAfterConfirm(key: string): string | null {
  for (const row of report.rows.slice(anchorIndex(key) + 1)) {
    if (state.filter === 'all') return row.frame_key;
    if (confirmedFilter(state.filter)) {
      if (matchesFilter(row, state.filter)) return row.frame_key;
      continue;
    }
    if (frameStatus(row.frame_key) === 'done') continue;
    if (matchesFilter(row, state.filter)) return row.frame_key;
  }
  return null;
}

/** Every frame of the drive is confirmed (on the server, or in this browser offline). */
export function allConfirmed(): boolean {
  return report.rows.length > 0 && report.rows.every((row) => frameStatus(row.frame_key) === 'done');
}

/** "All 780 frames are confirmed (780 by AI)." */
export function allConfirmedText(): string {
  const byAi = report.rows.filter((row) => aiConfirmed(row.frame_key)).length;
  return 'All ' + thousands(report.rows.length) + ' frames are confirmed'
    + (byAi > 0 ? ' (' + thousands(byAi) + ' by AI).' : '.');
}

/** Once the server has said what is confirmed: with the "To review" filter, a frame already confirmed (the last one
 * open here, or the first of the route) gives way to the next frame still to review. A drive with nothing left
 * opens on All (this time only: the filter kept in this browser stays To review). */
export function startAtWork(): void {
  const key = state.currentKey;
  if (state.filter !== 'todo' || key === null || frameStatus(key) !== 'done') return;
  if (allConfirmed()) {
    state.filter = 'all';
    state.notice = {key, text: allConfirmedText(), error: false};
    return;
  }
  const next = nextAfterConfirm(key)
    ?? report.rows.find((row) => frameStatus(row.frame_key) !== 'done' && matchesFilter(row, 'todo'))?.frame_key ?? null;
  if (next === null || next === key) return;
  openFrame(next);
  state.notice = {key: next, text: 'Opened at the next frame still to review.', error: false};
}

export function advanceAfterConfirm(key: string, savedText: string): void {
  const next = nextAfterConfirm(key);
  if (next === null) {
    state.notice = {key, text: savedText + ' No frames left to review with this filter.', error: false};
    return;
  }
  state.notice = {key: next, text: savedText + ' Here is the next one.', error: false};
  openFrame(next);
}

export function setFilter(filter: Filter): void {
  state.filter = filter;
  writeLocal('nnslr-review-filter', filter);
  const row = currentRow();
  if (row === null || matchesFilter(row, filter)) {
    state.filterNote = '';
    return;
  }
  // The next frame of the filter after this one, else the first one before it.
  const start = anchorIndex(row.frame_key);
  const target = report.rows.slice(start + 1).find((item) => matchesFilter(item, filter))
    ?? report.rows.slice(0, Math.max(start, 0)).find((item) => matchesFilter(item, filter));
  if (target === undefined) {
    if (filter === 'suspicious') state.filterNote = 'No suspicious frame to review.';
    else if (isQualityFilter(filter)) state.filterNote = 'No confirmed frame with this doubt.';
    else if (filter === 'ai') state.filterNote = 'No frame confirmed by an AI.';
    else if (filter === 'trial') state.filterNote = 'No frame of the AI trial left to review.';
    else state.filterNote = 'No frame matches this filter.';
    return;
  }
  openFrame(target.frame_key);
}

/** A click on a filter of the top bar or on a doubt counter of the timeline card. */
export function pickFilter(filter: Filter): void {
  // A second click on a pressed doubt counter goes back to the frames to review.
  if (isQualityFilter(filter) && state.filter === filter) setFilter('todo');
  else setFilter(filter);
}

/** Opened from Drive review ("?segment=3&second=12.5"): the frame nearest that moment, else null. */
function linkedFrame(): string | null {
  const query = new URLSearchParams(window.location.search);
  const segment = Number(query.get('segment'));
  const second = Number(query.get('second') ?? 0);
  if (query.get('segment') === null || !Number.isInteger(segment) || !Number.isFinite(second)) return null;
  const target = segment * 60 + second;
  let best: string | null = null;
  let gap = Infinity;
  for (const row of report.rows) {
    const distance = Math.abs(row.segment_index * 60 + (segmentSecond(row) ?? 0) - target);
    if (distance < gap) {
      gap = distance;
      best = row.frame_key;
    }
  }
  return best;
}

/** The frame Drive review asked for, else the frame last open in this browser, else the first of the filter, else
 * the first one. */
export function pickStartFrame(): string | null {
  // The drive at work opens on its newest frame, and follows (review/busy.ts).
  if (report.atWork) return report.rows[report.rows.length - 1]?.frame_key ?? null;
  // From Drive review every frame is shown (not saved as the filter): the moment asked for may be confirmed already.
  const linked = linkedFrame();
  if (linked !== null) {
    state.filter = 'all';
    return linked;
  }
  const last = readLocal(report.lastKeyStorage, null);
  if (last !== null && rowOf(last)) return last;
  const first = report.rows.find((row) => matchesFilter(row, state.filter)) ?? report.rows[0];
  return first ? first.frame_key : null;
}
