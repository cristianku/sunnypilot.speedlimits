<script setup lang="ts">
// The grip under the photo on a computer: dragged up, the photo gets lower and the timeline under it gets the room;
// dragged down, it grows back up to the width of the column. ↑ ↓ move it too when it has the focus (Shift: in big
// steps), a double click goes back to the page's own height. A tablet or phone scrolls the page instead: hidden there.
import {computed, ref, useTemplateRef} from 'vue';
import {PHOTO_MIN, photoHeight, setPhotoHeight} from './photo-size';
import type {FrameRow} from './types';

const props = defineProps<{row: FrameRow; size: {width: number; height: number}}>();
/** Kept under the photo for its bar and the start of the timeline (the same room as in review.html). */
const ROOM = 180;

const grip = useTemplateRef<HTMLElement>('grip');
const drag = ref<{pointer: number; y: number; height: number} | null>(null);

/** The photo's height on screen now. */
const shown = computed(() => Math.round(Math.min(props.size.height, props.size.width * props.row.height / (props.row.width || 1))));

/** The highest the photo goes: the column's width, and the window with room for the bar under it. */
function highest(): number {
  const width = grip.value?.getBoundingClientRect().width ?? 0;
  const top = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--topbar-h')) || 57;
  return Math.max(PHOTO_MIN, Math.min(width * props.row.height / (props.row.width || 1), window.innerHeight - top - ROOM));
}

function clamp(height: number): number {
  return Math.min(highest(), Math.max(PHOTO_MIN, height));
}

function onDown(event: PointerEvent): void {
  if (event.button !== 0 || grip.value === null) return;
  event.preventDefault();
  grip.value.setPointerCapture(event.pointerId);
  drag.value = {pointer: event.pointerId, y: event.clientY, height: shown.value};
}

function onMove(event: PointerEvent): void {
  const started = drag.value;
  if (started === null || started.pointer !== event.pointerId) return;
  photoHeight.value = Math.round(clamp(started.height + event.clientY - started.y));
}

function onUp(event: PointerEvent): void {
  if (drag.value?.pointer !== event.pointerId) return;
  drag.value = null;
  if (photoHeight.value !== null) setPhotoHeight(photoHeight.value);
}

function onKey(event: KeyboardEvent): void {
  const step = event.shiftKey ? 80 : 20;
  let height: number;
  switch (event.key) {
    case 'ArrowUp': height = shown.value - step; break;
    case 'ArrowDown': height = shown.value + step; break;
    case 'Home': height = PHOTO_MIN; break;
    case 'End': height = highest(); break;
    default: return;
  }
  // Not a box or a frame step of the page's keys.
  event.preventDefault();
  event.stopPropagation();
  setPhotoHeight(clamp(height));
}
</script>

<template>
  <div ref="grip" class="photo-grip" :class="{dragging: drag !== null}" role="separator" aria-orientation="horizontal"
       tabindex="0" aria-label="Photo height" :aria-valuemin="PHOTO_MIN" :aria-valuenow="shown"
       title="Drag up for a lower photo and more timeline · double click: back to the default"
       @pointerdown="onDown" @pointermove="onMove" @pointerup="onUp" @pointercancel="onUp"
       @dblclick="setPhotoHeight(null)" @keydown="onKey"></div>
</template>
