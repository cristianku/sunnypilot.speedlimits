<script setup lang="ts">
// The grip under the drive's picture on a computer, as under Sign review's photo, only the other way: the page's own
// height is the lowest, and dragged down the picture grows past its column, up to the width of the window and its height
// under the top bar (stage-size.ts lays the page out around it); dragged back up to the lowest it is the page's own again.
// ↑ ↓ move it too when it has the focus (Shift: in big steps), Home or a double click go back to the page's own height.
// A tablet or phone has one column and scrolls the page instead: hidden there.
import {ref, useTemplateRef} from 'vue';
import {highestHeight, lowestHeight, setStageHeight, stageHeight} from './stage-size';

const props = defineProps<{stage: HTMLElement | null}>();

const grip = useTemplateRef<HTMLElement>('grip');
const drag = ref<{pointer: number; y: number; height: number} | null>(null);

/** The picture's height on screen now. */
const shown = (): number => Math.round(props.stage?.getBoundingClientRect().height ?? lowestHeight());
const clamp = (height: number): number => Math.min(highestHeight(), Math.max(lowestHeight(), height));
/** Keeps a height, or none at the lowest: the page's own height follows the window. */
const keep = (height: number): void => setStageHeight(height <= lowestHeight() + 1 ? null : clamp(height));

function onDown(event: PointerEvent): void {
  if (event.button !== 0 || grip.value === null) return;
  event.preventDefault();
  grip.value.setPointerCapture(event.pointerId);
  drag.value = {pointer: event.pointerId, y: event.clientY, height: shown()};
}
function onMove(event: PointerEvent): void {
  const started = drag.value;
  if (started === null || started.pointer !== event.pointerId) return;
  stageHeight.value = Math.round(clamp(started.height + event.clientY - started.y));
}
function onUp(event: PointerEvent): void {
  if (drag.value?.pointer !== event.pointerId) return;
  drag.value = null;
  if (stageHeight.value !== null) keep(stageHeight.value);
}
function onKey(event: KeyboardEvent): void {
  const step = event.shiftKey ? 80 : 20;
  let height: number;
  switch (event.key) {
    case 'ArrowUp': height = shown() - step; break;
    case 'ArrowDown': height = shown() + step; break;
    case 'Home': height = lowestHeight(); break;
    case 'End': height = highestHeight(); break;
    default: return;
  }
  // Not a second or a frame of the player's keys.
  event.preventDefault();
  event.stopPropagation();
  keep(height);
}
</script>

<template>
  <div ref="grip" class="dr-grip" :class="{dragging: drag !== null}" role="separator" aria-orientation="horizontal"
       tabindex="0" aria-label="Picture height" :aria-valuemin="Math.round(lowestHeight())" :aria-valuenow="stageHeight ?? undefined"
       title="Drag down for a bigger picture · double click: back to the page’s own size"
       @pointerdown="onDown" @pointermove="onMove" @pointerup="onUp" @pointercancel="onUp"
       @dblclick="setStageHeight(null)" @keydown="onKey"></div>
</template>
