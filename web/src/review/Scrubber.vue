<script setup lang="ts">
// The drive as one bar under the photo, as in Speed review: dragged, it opens the picture of that second; the buttons
// go a minute or 10 s back and on. One step per picture at 1 a second; a frame between the seconds (a sign the comma
// read, a frame of the video) is on the bar where its second is. Under it, only when there are some, the confirmed
// frames that still carry a doubt: each counter is a filter.
import {computed} from 'vue';
import {DOUBT_LABELS, doubtCounts} from './current';
import {driveTime, framePlace, isBetweenFrame} from './frames';
import {openFrame, pickFilter} from './navigation';
import {report, rowOf, state} from './store';
import type {FrameRow, QualityFilter} from './types';

/** The pictures at 1 a second, in the order of the drive. */
const seconds = computed((): FrameRow[] => report.rows.filter((row) => !isBetweenFrame(row)));

/** The step of the frame on screen: its own, or the second before it for a frame between the seconds. */
const index = computed((): number => {
  const key = state.currentKey;
  if (key === null) return 0;
  const at = report.rows.findIndex((row) => row.frame_key === key);
  if (at < 0) return 0;
  let count = -1;
  for (let i = 0; i <= at; i++) if (!isBetweenFrame(report.rows[i]!)) count++;
  return Math.max(0, count);
});

const timeOf = (row: FrameRow | undefined): string => row === undefined ? '–' : driveTime(row) ?? framePlace(row);
const shown = computed(() => {
  const key = state.currentKey;
  const row = key === null ? null : rowOf(key);
  return row === null ? '–' : timeOf(row);
});
const total = computed(() => timeOf(seconds.value[seconds.value.length - 1]));

const DOUBTS: {id: string; filter: QualityFilter; warn: boolean}[] = [
  {id: 'qUnknown', filter: 'unknown', warn: true},
  {id: 'qUnreadable', filter: 'unreadable', warn: true},
  {id: 'qRoad', filter: 'road', warn: false},
];
const doubts = computed(() => DOUBTS.map((doubt) => ({...doubt, count: doubtCounts.value[doubt.filter]}))
  .filter((doubt) => doubt.count > 0 || state.filter === doubt.filter));

let pending: number | null = null;
let frame = 0;
/** Dragged: the picture of that step, once per screen refresh while the bar moves. */
function onInput(event: Event): void {
  pending = Number((event.target as HTMLInputElement).value);
  if (frame !== 0) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    if (pending !== null) go(pending);
    pending = null;
  });
}

/** The keys that move the bar stay on it: the page's own (→ the next frame to review) would move it twice. */
function onKey(event: KeyboardEvent): void {
  if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown'].includes(event.key)) {
    event.stopPropagation();
  }
}

function go(step: number): void {
  const list = seconds.value;
  const row = list[Math.max(0, Math.min(list.length - 1, step))];
  if (row !== undefined && row.frame_key !== state.currentKey) openFrame(row.frame_key);
}
</script>

<template>
  <section v-if="seconds.length > 1" class="sc-card" aria-label="The drive">
    <div class="sc-row">
      <button type="button" class="sc-step" title="A minute back" aria-label="A minute back" @click="go(index - 60)">−1 min</button>
      <button type="button" class="sc-step" title="10 s back" aria-label="10 seconds back" @click="go(index - 10)">−10 s</button>
      <input id="scrubber" class="sc-bar" type="range" min="0" :max="seconds.length - 1" step="1" :value="index"
             :aria-valuetext="shown" aria-label="Where in the drive" @input="onInput" @keydown="onKey"/>
      <button type="button" class="sc-step" title="10 s on" aria-label="10 seconds on" @click="go(index + 10)">+10 s</button>
      <button type="button" class="sc-step" title="A minute on" aria-label="A minute on" @click="go(index + 60)">+1 min</button>
    </div>
    <p class="sc-time"><b>{{ shown }}</b> / {{ total }}</p>
    <div v-if="doubts.length > 0" class="sc-doubts" role="group" aria-label="Doubts in the confirmed frames">
      <span>Confirmed with doubts:</span>
      <button v-for="doubt in doubts" :id="doubt.id" :key="doubt.id" type="button" class="qchip" :class="{some: doubt.warn}"
              :data-filter="doubt.filter" :aria-pressed="state.filter === doubt.filter"
              @click="pickFilter(doubt.filter)">{{ doubt.count }} {{ DOUBT_LABELS[doubt.filter] }}</button>
    </div>
  </section>
</template>
