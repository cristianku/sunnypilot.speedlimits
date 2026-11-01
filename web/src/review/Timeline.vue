<script setup lang="ts">
// The timeline card: one row of ticks per segment, coloured by the state of each frame, and the counters of
// the confirmed frames that still carry a doubt (each one a filter). On a phone the card folds under its title.
// Where the server has the road video, a cell is a whole second (seconds.ts): confirmed when its 20 frames are, filled
// from the bottom as they are confirmed.
import {computed, nextTick} from 'vue';
import {thousands} from '../common/format';
import {carDiffers, carLogs} from './car-moment';
import {writeLocal} from './dom';
import {DOUBT_LABELS, doubtCounts, routeCounts} from './current';
import {aiConfirmed, commaReadText, doubtsOf, driveTime, framePlace, frameStatus, isBetweenFrame, isReadFrame, proposalsOf,
  reasonsOf, statusLabel} from './frames';
import {openFrame, pickFilter} from './navigation';
import {scrollTimelineToCurrent} from './scroll';
import {cellFrames, hasVideo, secondProgress} from './seconds';
import {frameSecond} from './order';
import {hintsOf, isVirtual, report, rowOf, state} from './store';
import type {FrameRow, QualityFilter} from './types';

interface Between {
  row: FrameRow;
  /** Where its bar is, in cells: the cell of the picture before it and how far into its second it is. */
  at: number;
}

interface Segment {
  segment: number;
  rows: FrameRow[];
  between: Between[];
}

/** Where the thin bar of a frame between the seconds goes, its cells the pictures of its segment before it. */
function placeOf(row: FrameRow, before: number): number {
  const second = frameSecond(row);
  return Math.max(0, before - 1 + second - Math.floor(second));
}

/** A frame between the seconds with a thin bar of its own: a sign the comma read, and a frame of the video saved
 * where its cell does not count it (offline). */
function hasBar(row: FrameRow): boolean {
  return isBetweenFrame(row) && !isVirtual(row.frame_key) && (isReadFrame(row) || !hasVideo(row.segment_index));
}

/** One row of ticks per segment, with its label: a cell per picture at 1 a second, and a thin bar across the cells
 * for each frame between the seconds that has one (hasBar). */
const segments = computed(() => {
  const list: Segment[] = [];
  for (const row of report.rows) {
    let last = list[list.length - 1];
    if (last === undefined || last.segment !== row.segment_index) {
      last = {segment: row.segment_index, rows: [], between: []};
      list.push(last);
    }
    if (hasBar(row)) last.between.push({row, at: placeOf(row, last.rows.length)});
    else if (!isBetweenFrame(row)) last.rows.push(row);
  }
  return list;
});
// The cells fill the width: one column per picture of the longest segment.
const columns = computed(() => String(Math.max(1, ...segments.value.map((segment) => segment.rows.length))));
const hasReads = computed(() => report.rows.some(isReadFrame));

/** The frame of the video on screen when it has no bar of its own (not saved, or counted by its cell): its bar in its
 * segment. */
const walking = computed<{segment: number; bar: Between} | null>(() => {
  const key = state.currentKey;
  const found = key === null ? null : rowOf(key);
  const row = found === null || !isBetweenFrame(found) || hasBar(found) ? null : found;
  const segment = row === null ? undefined : segments.value.find((item) => item.segment === row.segment_index);
  if (row === null || segment === undefined) return null;
  const before = segment.rows.filter((item) => frameSecond(item) <= frameSecond(row)).length;
  return {segment: row.segment_index, bar: {row, at: placeOf(row, before)}};
});

function tickClass(row: FrameRow): string {
  const key = row.frame_key;
  const status = frameStatus(key);
  const part = secondProgress(row);
  let cls = '';
  if (status === 'done' && (part === null || part.complete)) {
    const unsure = [row, ...cellFrames(row)].some((frame) => {
      const doubts = doubtsOf(frame.frame_key);
      return doubts.unknown + doubts.unreadable > 0;
    });
    cls = unsure ? 'done unsure' : 'done';
    if (aiConfirmed(key)) cls += ' ai';
  } else if (status === 'error') {
    cls = 'error';
  } else if (status === 'dirty' || status === 'saving') {
    cls = 'dirty';
  } else if (reasonsOf(key).length > 0) {
    cls = 'sus';
  } else if (proposalsOf(row).length > 0 || hintsOf(key).length > 0) {
    cls = 'has';
  }
  if (part !== null && part.done > 0 && !part.complete) cls += ' part';
  if (carDiffers(row)) cls += ' differ';
  if (isBetweenFrame(row)) cls += ' between';
  if (isReadFrame(row)) cls += ' read';
  if (isVirtual(key)) cls += ' video';
  if (key === state.currentKey) cls += ' cur';
  return cls;
}

// "2:06 into the drive (segment 2, image 6) · confirmed": the tooltip and the name of a tick.
function tickLabel(row: FrameRow): string {
  const place = driveTime(row) === null ? framePlace(row)
    : framePlace(row) + ' (segment ' + row.segment_index + ', image ' + row.output_index + ')';
  if (isVirtual(row.frame_key)) return place + ' · a frame of the video, not saved yet';
  const part = secondProgress(row);
  return place + ' · ' + statusLabel(row.frame_key).toLowerCase()
    + (part === null ? '' : ' · ' + part.done + '/' + part.total + ' frames of the second confirmed')
    + (carDiffers(row) ? ' · map ≠ vision' : '')
    + (isReadFrame(row) ? ' · ' + commaReadText(row) : isBetweenFrame(row) ? ' · a frame between the seconds' : '');
}

/** How much of a cell's second is confirmed, for its fill (class part). */
function fillOf(row: FrameRow): Record<string, string> | undefined {
  const part = secondProgress(row);
  return part === null || part.done === 0 || part.complete ? undefined : {'--fill': String(part.done / part.total)};
}

/** "120/720 confirmed · 3 AI · 2 suspicious": the title of the folded card on a phone. */
const summary = computed(() => {
  const {done, suspicious, byAi, total} = routeCounts.value;
  let text = thousands(done) + '/' + thousands(total) + ' confirmed';
  if (byAi > 0) text += ' · ' + thousands(byAi) + ' AI';
  if (suspicious > 0) text += ' · ' + thousands(suspicious) + ' suspicious';
  return text;
});

interface Chip {
  id: string;
  filter: QualityFilter;
  count: number;
  label: string;
  /** A count above zero is a warning (a number or a sign not read). */
  warn: boolean;
}

// Confirmed is not the same as certain: how many confirmed frames still carry a doubt. Each counter is a filter.
const chips = computed<Chip[]>(() => {
  const counts = doubtCounts.value;
  return [
    {id: 'qUnknown', filter: 'unknown', count: counts.unknown, label: DOUBT_LABELS.unknown, warn: true},
    {id: 'qUnreadable', filter: 'unreadable', count: counts.unreadable, label: DOUBT_LABELS.unreadable, warn: true},
    {id: 'qRoad', filter: 'road', count: counts.road, label: DOUBT_LABELS.road, warn: false},
  ];
});

function chipClass(chip: Chip): string {
  if (chip.count > 0 && chip.warn) return 'qchip some';
  return chip.count > 0 ? 'qchip' : 'qchip zero';
}

/** The fold of the card on a phone, kept in this browser. */
function toggle(): void {
  state.timelineOpen = !state.timelineOpen;
  writeLocal('nnslr-timeline-open', state.timelineOpen ? '1' : '0');
  if (state.timelineOpen) void nextTick(scrollTimelineToCurrent);
}

// One listener for all the ticks.
function tickClicked(event: MouseEvent): void {
  const target = event.target;
  const key = target instanceof HTMLElement ? target.dataset['key'] : undefined;
  if (key) openFrame(key);
}
</script>

<template>
  <section class="tl-card" aria-labelledby="tlHeading">
    <h2 class="tl-heading" id="tlHeading">Timeline</h2>
    <button type="button" class="tl-toggle" id="tlToggle" :aria-expanded="state.timelineOpen" aria-controls="timeline"
            @click="toggle"><span class="tl-title">Timeline</span><span class="tl-sub" id="tlSub">{{ summary }}</span></button>
    <div class="legend">
      <span><i class="l-done"></i>{{ report.videos === null ? 'confirmed' : 'confirmed, every frame of the second' }}</span>
      <span v-if="report.videos !== null"><i class="l-part"></i>partly confirmed</span>
      <span><i class="l-ai"></i>by AI</span>
      <span><i class="l-unsure"></i>confirmed with doubts</span>
      <span><i class="l-sus"></i>suspicious</span>
      <span><i class="l-has"></i>with proposals or 2nd reading</span>
      <span><i class="l-none"></i>nothing proposed</span>
      <span><i class="l-dirty"></i>unsaved changes</span>
      <span><i class="l-error"></i>not saved</span>
      <span v-if="carLogs.phase === 'ready'"><i class="l-differ"></i>map ≠ vision in the car</span>
      <span v-if="hasReads"><i class="l-read"></i>a sign the comma read</span>
    </div>
    <div class="timeline" id="timeline" :class="{'car-logs': carLogs.phase === 'ready'}" aria-label="All the frames of the route" :style="{'--tl-cols': columns}"
         @click="tickClicked">
      <div v-for="(segment, i) in segments" :key="i" class="tl-seg">
        <span class="tl-label">seg {{ segment.segment }}</span>
        <div class="tl-ticks">
          <button v-for="row in segment.rows" :key="row.frame_key" type="button" tabindex="-1" :data-key="row.frame_key"
                  :class="tickClass(row)" :style="fillOf(row)" :title="tickLabel(row)" :aria-label="tickLabel(row)"></button>
          <button v-for="bar in segment.between" :key="bar.row.frame_key" type="button" tabindex="-1"
                  :data-key="bar.row.frame_key" :class="tickClass(bar.row)" :style="{'--at': bar.at}"
                  :title="tickLabel(bar.row)" :aria-label="tickLabel(bar.row)"></button>
          <button v-if="walking !== null && walking.segment === segment.segment" type="button" tabindex="-1"
                  :data-key="walking.bar.row.frame_key" :class="tickClass(walking.bar.row)" :style="{'--at': walking.bar.at}"
                  :title="tickLabel(walking.bar.row)" :aria-label="tickLabel(walking.bar.row)"></button>
        </div>
      </div>
    </div>
    <div class="quality" role="group" aria-label="Doubts in the confirmed frames">
      <span class="quality-label">Confirmed with doubts:</span>
      <button v-for="chip in chips" :key="chip.id" type="button" :class="chipClass(chip)" :id="chip.id"
              :data-filter="chip.filter" :data-count="chip.count" :aria-pressed="state.filter === chip.filter"
              @click="pickFilter(chip.filter)">{{ chip.count }} {{ chip.label }}</button>
    </div>
  </section>
</template>
