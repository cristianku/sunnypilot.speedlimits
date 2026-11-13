<script setup lang="ts">
// The photo and what is drawn on it: the boxes with their dark under-stroke and labels, the corner handles of
// the selected box, the boxes of the second reading, the AI suggestion not applied yet and what the car's networks
// read (car.ts; neither takes the pointer) and the box being drawn. The frame's country and what the comma had at that moment are in the bar above
// it (CarMoment.vue). Under it, the grip that makes it lower on a computer (PhotoGrip.vue). Inside it, the zoom of the
// selected box (ZoomInset.vue), always in the bottom right corner.
// The boxes of a frame wait for its photo: they are never drawn on the previous one.
import {computed, ref, useTemplateRef, watch} from 'vue';
import {ai} from './ai';
import {carAskOf, carDetection, carSignLeft} from './car';
import {carried} from './carry';
import {signKindOf} from './catalog';
import {selectedBox} from './current';
import {coarsePointer} from './dom';
import {drawAi, drawCar, drawStage, sceneScale, WET_TAG} from './overlay';
import {finishDrag, onPointerDown, onPointerMove, onSceneClick, onTouchStart} from './pointer';
import PhotoGrip from './PhotoGrip.vue';
import {hintsOf, state} from './store';
import type {FrameEdit, FrameRow} from './types';
import {useScreenSize} from './use-layout';
import ZoomInset from './ZoomInset.vue';

const props = defineProps<{row: FrameRow; work: FrameEdit}>();

const scene = useTemplateRef<SVGSVGElement>('scene');
const size = useScreenSize(scene);
const stage = useTemplateRef<HTMLDivElement>('stage');

/** The photo that has loaded (or failed): its address and the frame it was loaded for. */
const loaded = ref<{href: string; key: string} | null>(null);
watch(() => props.row.preview_path, () => { loaded.value = null; });
const layersVisible = computed(() => loaded.value !== null && loaded.value.href === props.row.preview_path);
const visibility = computed(() => (layersVisible.value ? undefined : {visibility: 'hidden' as const}));

/** The photo has loaded (or failed): show its boxes, and say which frame it is (stage[data-photo-key]). */
function photoDone(event: Event, failed: boolean): void {
  const photo = event.currentTarget;
  if (!(photo instanceof SVGImageElement) || photo.getAttribute('href') !== props.row.preview_path) return;
  loaded.value = {href: props.row.preview_path, key: (failed ? 'error:' : '') + props.row.frame_key};
}

const drawing = computed(() => drawStage(props.work.detections, state.selected,
  state.showHints ? hintsOf(props.row.frame_key) : [], props.row.height,
  sceneScale(size.value, props.row.width, props.row.height), coarsePointer(), props.row.width));
const aiBoxes = computed(() => {
  const ask = ai.asks[props.row.frame_key];
  if (ask?.phase !== 'ready' || !ask.suggestion) return [];
  return drawAi(ask.suggestion, props.row.height, sceneScale(size.value, props.row.width, props.row.height),
    props.row.width);
});
const carBoxes = computed(() => {
  const reading = carAskOf(props.row.frame_key)?.reading;
  if (!reading) return [];
  const left = reading.signs.map((sign) => carSignLeft(props.work, sign));
  return drawCar(reading.signs, reading.signs.map(carDetection), reading.surest, props.row.height,
    sceneScale(size.value, props.row.width, props.row.height), props.row.width).filter((_, i) => left[i]);
});
const resizing = computed(() => (state.drag?.kind === 'resize' || state.drag?.kind === 'move') && state.drag.svg === 'scene');

// The zoom of the selected box: inside the photo (the picture shown in the picture's frame, which may leave bands
// around it), always in its bottom right corner, so it never jumps while a box is worked on; about a third of the
// photo wide, half on a phone.
const INSET_GAP = 8;
const inset = computed(() => {
  const d = selectedBox.value;
  const svg = scene.value;
  const frame = stage.value;
  const {width, height} = size.value;
  if (d === null || svg === null || frame === null || !width || !height || state.drawing || !layersVisible.value) return null;
  const k = Math.min(width / props.row.width, height / props.row.height);
  const photoWidth = props.row.width * k;
  const photoHeight = props.row.height * k;
  // The picture's frame inside the stage (absolute positions start inside its border).
  const outer = frame.getBoundingClientRect();
  const inner = svg.getBoundingClientRect();
  const left = inner.left - outer.left - frame.clientLeft + (width - photoWidth) / 2;
  const top = inner.top - outer.top - frame.clientTop + (height - photoHeight) / 2;
  // Between what the page puts over the top of the photo (Speed review: the comma's limits, --zoom-inset-top) and
  // over its bottom (Speed review: the video bar, --zoom-inset-bottom).
  const style = getComputedStyle(frame);
  const minTop = Math.max(top + INSET_GAP, parseFloat(style.getPropertyValue('--zoom-inset-top')) || 0);
  const bottom = top + photoHeight - Math.max(INSET_GAP, parseFloat(style.getPropertyValue('--zoom-inset-bottom')) || 0);
  const w = Math.min(Math.max(photoWidth * (photoWidth < 640 ? 0.5 : 0.34), 170), 440, photoWidth - 2 * INSET_GAP);
  const h = Math.max(Math.min(w * 0.6, photoHeight * 0.6, bottom - minTop), 60);
  const y = bottom - h;
  const x = left + photoWidth - INSET_GAP - w;
  return {left: x + 'px', top: y + 'px', width: w + 'px', height: h + 'px'};
});
</script>

<template>
  <div ref="stage" class="stage" id="stage" :data-photo-key="loaded?.key ?? ''">
    <svg ref="scene" id="scene" :class="{drawing: state.drawing, dragging: resizing}" xmlns="http://www.w3.org/2000/svg"
         aria-label="Frame with the boxes of the signs" :viewBox="'0 0 ' + row.width + ' ' + row.height"
         @pointerdown="onPointerDown($event, 'scene')" @pointermove="onPointerMove($event, 'scene')"
         @pointerup="finishDrag($event, 'scene')" @pointercancel="finishDrag($event, 'scene')"
         @touchstart="onTouchStart($event, 'scene')" @click="onSceneClick">
      <image id="photo" x="0" y="0" preserveAspectRatio="none" :href="row.preview_path" :width="row.width"
             :height="row.height" @load="photoDone($event, false)" @error="photoDone($event, true)"></image>
      <g id="hintLayer" :style="visibility">
        <template v-for="(hint, i) in drawing.hints" :key="i">
          <rect class="hint-under" :x="hint.rect.x" :y="hint.rect.y" :width="hint.rect.width" :height="hint.rect.height"/>
          <rect :class="'hint-box ' + hint.confidence" :x="hint.rect.x" :y="hint.rect.y" :width="hint.rect.width"
                :height="hint.rect.height"/>
          <text v-if="hint.label" :x="hint.label.x" :y="hint.label.y" class="hint-tag" :font-size="hint.label.fontSize"
                :stroke-width="hint.label.strokeWidth" paint-order="stroke">{{ hint.label.text }}</text>
        </template>
      </g>
      <g id="carLayer" :style="visibility">
        <template v-for="(box, i) in carBoxes" :key="i">
          <rect class="box-under" :x="box.rect.x" :y="box.rect.y" :width="box.rect.width" :height="box.rect.height"/>
          <rect :class="'car-box ' + box.use + (box.surest ? ' surest' : '')" :x="box.rect.x" :y="box.rect.y"
                :width="box.rect.width" :height="box.rect.height"/>
          <text :x="box.label.x" :y="box.label.y" :class="'car-tag ' + box.use" :font-size="box.label.fontSize"
                :stroke-width="box.label.strokeWidth" paint-order="stroke">{{ box.label.text }}</text>
        </template>
      </g>
      <g id="overlays" :style="visibility">
        <template v-for="box in drawing.boxes" :key="box.index">
          <rect :class="box.cls.replace('box', 'box-under')" :x="box.rect.x" :y="box.rect.y" :width="box.rect.width"
                :height="box.rect.height"/>
          <rect :class="box.cls" :data-index="box.index" :x="box.rect.x" :y="box.rect.y" :width="box.rect.width"
                :height="box.rect.height" :style="box.stroke === null ? undefined : {stroke: box.stroke}"/>
          <template v-if="box.road">
            <path :d="box.road" class="road-mark-under" :stroke-width="box.label.strokeWidth * 1.3"/>
            <path :d="box.road" class="road-mark" :stroke-width="box.label.strokeWidth * 0.55"/>
          </template>
          <text :x="box.label.x" :y="box.label.y" class="tag" :font-size="box.label.fontSize"
                :stroke-width="box.label.strokeWidth" paint-order="stroke">{{ box.label.text }}<tspan v-if="box.wet"
                class="tag-wet">{{ WET_TAG }}</tspan></text>
        </template>
        <rect v-for="handle in drawing.handles" :key="'mark-' + handle.corner" :x="handle.mark.x" :y="handle.mark.y"
              :width="handle.mark.width" :height="handle.mark.height" class="handle"/>
        <rect v-for="handle in drawing.handles" :key="'hit-' + handle.corner" :x="handle.hit.x" :y="handle.hit.y"
              :width="handle.hit.width" :height="handle.hit.height" :class="'handle-hit ' + handle.corner"
              :data-corner="handle.corner"/>
      </g>
      <g id="aiLayer" :style="visibility">
        <template v-for="(box, i) in aiBoxes" :key="i">
          <rect class="box-under" :x="box.rect.x" :y="box.rect.y" :width="box.rect.width" :height="box.rect.height"/>
          <rect :class="'ai-box ' + box.kind" :x="box.rect.x" :y="box.rect.y" :width="box.rect.width" :height="box.rect.height"/>
          <text :x="box.label.x" :y="box.label.y" class="ai-tag" :font-size="box.label.fontSize"
                :stroke-width="box.label.strokeWidth" paint-order="stroke">{{ box.label.text }}</text>
        </template>
      </g>
      <rect id="draft" :visibility="state.draft ? 'visible' : 'hidden'" :x="state.draft?.x" :y="state.draft?.y"
            :width="state.draft?.width" :height="state.draft?.height"></rect>
    </svg>
    <ZoomInset v-if="inset && selectedBox" :style="inset" :row="row" :detection="selectedBox"/>
    <div class="stage-hint" id="stageHint" :hidden="!state.drawing"><span>Drag on the image to draw the box{{ state.drawKind ? ': ' + (signKindOf(state.drawKind)?.chip ?? '') : '' }} · Shift: a square</span></div>
    <!-- After A: the boxes carried, not saved yet; A again saves and goes on (carry.ts). -->
    <div class="stage-hint carry-hint" id="carryHint" :hidden="state.drawing || carried !== row.frame_key"><span>Press A to confirm</span></div>
  </div>
  <PhotoGrip :row="row" :size="size"/>
</template>
