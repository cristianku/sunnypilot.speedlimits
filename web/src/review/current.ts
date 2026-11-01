// The frame on screen, as computed values the components share: its row, its working copy, its state, the
// box selected in it, its place in the route.
import {computed, watchEffect} from 'vue';
import {aiConfirmed, currentRow, doubtsOf, frameStatus, isConfirmed, reasonsOf, workOf} from './frames';
import {anchorIndex, indexOf, report, state} from './store';
import type {Detection, FrameEdit, FrameRow, FrameStatus, QualityFilter} from './types';

export const row = computed<FrameRow | null>(() => currentRow());
/** The working copy of the frame on screen (ensureCurrentFrame makes sure there is one). */
export const work = computed<FrameEdit | null>(() => (row.value ? state.edits[row.value.frame_key] ?? null : null));
export const status = computed<FrameStatus>(() => (row.value ? frameStatus(row.value.frame_key) : 'todo'));
export const selectedBox = computed<Detection | null>(() => work.value?.detections[state.selected] ?? null);

/** Position of the frame in the route (0 based; a frame of the video between the seconds: of the picture before
 * it) and the number of frames; whether ‹ › can go back and on (with the server ‹ › go through every frame of the
 * video, video.ts). */
export const position = computed(() => {
  const key = row.value?.frame_key;
  const index = key === undefined ? -1 : indexOf(key);
  const anchor = key === undefined ? -1 : anchorIndex(key);
  const length = report.rows.length;
  return {index: Math.max(0, anchor), length, canPrev: index !== 0 && anchor >= 0,
    canNext: index < 0 || index < length - 1 || report.apiBase !== null};
});

/** Confirmed, suspicious and AI-confirmed frames of the route. */
export const routeCounts = computed(() => {
  let done = 0;
  let suspicious = 0;
  let byAi = 0;
  for (const item of report.rows) {
    if (isConfirmed(item.frame_key)) done += 1;
    if (reasonsOf(item.frame_key).length > 0) suspicious += 1;
    if (aiConfirmed(item.frame_key)) byAi += 1;
  }
  return {done, suspicious, byAi, total: report.rows.length};
});

/** What each doubt filter shows, after its count: "26 without a number". */
export const DOUBT_LABELS: Record<QualityFilter, string> = {
  unknown: 'without a number', unreadable: 'with unreadable signs', road: 'road not set',
};

/** The confirmed frames that still carry each doubt (the counters of the timeline card, each one a filter). */
export const doubtCounts = computed<Record<QualityFilter, number>>(() => {
  const counts = {unknown: 0, unreadable: 0, road: 0};
  for (const item of report.rows) {
    const doubts = doubtsOf(item.frame_key);
    if (doubts.unknown > 0) counts.unknown += 1;
    if (doubts.unreadable > 0) counts.unreadable += 1;
    if (doubts.road > 0) counts.road += 1;
  }
  return counts;
});

/** Before each drawing: a frame is on screen when the report has frames, it has its working copy (the
 * proposals until the reviewer changes it), and the selection points at one of its boxes. */
export function ensureCurrentFrame(): void {
  watchEffect(() => {
    let shown = currentRow();
    if (shown === null) {
      shown = report.rows[0] ?? null;
      if (shown === null) return;
      state.currentKey = shown.frame_key;
    }
    const copy = workOf(shown);
    if (state.selected >= copy.detections.length) state.selected = -1;
  });
}
