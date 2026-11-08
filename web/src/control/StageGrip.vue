<script setup lang="ts">
// The grip under the drive's picture on a computer, as under Sign review's photo: dragged down, the picture grows up to
// the width of its column; dragged up, it gets lower and the lanes under it get the room. ↑ ↓ move it too when it has
// the focus (Shift: in big steps), a double click goes back to the page's own height. A tablet or phone has one column
// and scrolls the page instead: hidden there.
import {ref, useTemplateRef} from 'vue';
import {setStageHeight, STAGE_MIN, stageHeight} from './stage-size';

const props = defineProps<{stage: HTMLElement | null}>();
/** The picture's width over its height: the comma's road camera. */
const ASPECT = 1928 / 1208;

const grip = useTemplateRef<HTMLElement>('grip');
const drag = ref<{pointer: number; y: number; height: number} | null>(null);

/** The picture's height on screen now. */
const shown = (): number => Math.round(props.stage?.getBoundingClientRect().height ?? STAGE_MIN);
/** The highest the picture goes: as wide as its column, the grip's own width. */
const highest = (): number => Math.max(STAGE_MIN, (grip.value?.getBoundingClientRect().width ?? 0) / ASPECT);
const clamp = (height: number): number => Math.min(highest(), Math.max(STAGE_MIN, height));

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
  if (stageHeight.value !== null) setStageHeight(stageHeight.value);
}
function onKey(event: KeyboardEvent): void {
  const step = event.shiftKey ? 80 : 20;
  let height: number;
  switch (event.key) {
    case 'ArrowUp': height = shown() - step; break;
    case 'ArrowDown': height = shown() + step; break;
    case 'Home': height = STAGE_MIN; break;
    case 'End': height = highest(); break;
    default: return;
  }
  // Not a second or a frame of the player's keys.
  event.preventDefault();
  event.stopPropagation();
  setStageHeight(clamp(height));
}
</script>

<template>
  <div ref="grip" class="dr-grip" :class="{dragging: drag !== null}" role="separator" aria-orientation="horizontal"
       tabindex="0" aria-label="Picture height" :aria-valuemin="STAGE_MIN" :aria-valuenow="stageHeight ?? undefined"
       title="Drag down for a bigger picture, up for more lanes · double click: back to the default"
       @pointerdown="onDown" @pointermove="onMove" @pointerup="onUp" @pointercancel="onUp"
       @dblclick="setStageHeight(null)" @keydown="onKey"></div>
</template>
