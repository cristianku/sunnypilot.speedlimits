<script setup lang="ts">
// The zoom of the selected box, inside the photo (Stage.vue places it in the bottom right corner):
// its corners and inside move the box too, with its size in pixels in the top right corner. A click elsewhere on the
// photo clears the selection and closes it (pointer.ts onSceneClick).
import {computed, ref, useTemplateRef, watchEffect} from 'vue';
import {coarsePointer} from './dom';
import {cropRectangle} from './geometry';
import {boxTitle, handlesOf, sceneScale, WET_TAG, zoomView, type Rect} from './overlay';
import {finishDrag, onPointerDown, onPointerMove, onTouchStart} from './pointer';
import {driveTime} from './frames';
import {report, state} from './store';
import type {Detection, FrameRow} from './types';
import {useScreenSize} from './use-layout';

const props = defineProps<{row: FrameRow; detection: Detection; index: number}>();

// The box's label as on the photo ("6 · light amber · OURS"), in the zoom's top left corner.
const title = computed(() => boxTitle(props.detection, props.index)
  + (props.detection.supplementary_panel === 'wet' ? WET_TAG : ''));

// Where the frame is, in the zoom's bottom right corner ("00000191 · 1/49 · 1:49"): a screenshot of the zoom alone
// still names the drive and the frame.
const where = computed(() => {
  const time = driveTime(props.row);
  return [report.routeId.split('--')[0], props.row.frame_key, time].filter(Boolean).join(' · ');
});

const zoom = useTemplateRef<SVGSVGElement>('zoom');
const zoomSize = useScreenSize(zoom);

// The box in pixels of the picture the classifier cuts it from, in the zoom's top right corner ("132×36 px"): rounded
// outwards as on the car (cropRectangle), in the photo's own pixels: twice the frame's on a drive at full definition.
const natural = ref<[number, number] | null>(null);
watchEffect(() => {
  const href = props.row.preview_path;
  natural.value = null;
  const image = new Image();
  image.onload = () => {
    if (href === props.row.preview_path) natural.value = [image.naturalWidth, image.naturalHeight];
  };
  image.src = href;
});
const pixels = computed(() => {
  const c = cropRectangle(props.detection.bbox_xyxy, props.row.width, props.row.height);
  const [width, height] = natural.value ?? [props.row.width, props.row.height];
  return Math.round((c[2] - c[0]) * width / props.row.width) + '×' + Math.round((c[3] - c[1]) * height / props.row.height)
    + ' px';
});

/** The zoom's view, as wide as the zoom on screen, the box's view centred in it. */
function fitted(v: Rect, size: {width: number; height: number}): Rect {
  if (!size.width || !size.height || !v.width || !v.height) return v;
  const scale = Math.min(size.width / v.width, size.height / v.height);
  const width = size.width / scale, height = size.height / scale;
  return {x: v.x - (width - v.width) / 2, y: v.y - (height - v.height) / 2, width, height};
}

// While a corner of the zoom is dragged its view stays still, so the corner stays under the finger.
const view = ref<Rect | null>(null);
watchEffect(() => {
  if (state.drag !== null && state.drag.svg === 'zoom') return;
  view.value = fitted(zoomView(props.detection.bbox_xyxy, props.row.width, props.row.height), zoomSize.value);
});
const viewBox = computed(() => {
  const v = view.value;
  return v === null ? undefined : v.x + ' ' + v.y + ' ' + v.width + ' ' + v.height;
});
const handles = computed(() => {
  const v = view.value;
  // The SVG keeps its view box in single precision.
  const kz = v && v.width && v.height ? sceneScale(zoomSize.value, Math.fround(v.width), Math.fround(v.height)) : 1;
  return handlesOf(props.detection.bbox_xyxy, 12 / kz, (coarsePointer() ? 36 : 22) / kz, false);
});
const box = computed(() => {
  const b = props.detection.bbox_xyxy;
  return {x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1]};
});
const resizing = computed(() => (state.drag?.kind === 'resize' || state.drag?.kind === 'move') && state.drag.svg === 'zoom');
</script>

<template>
  <div class="zoom-wrap zoom-inset">
    <svg ref="zoom" id="zoom" :class="{dragging: resizing}" xmlns="http://www.w3.org/2000/svg"
         preserveAspectRatio="xMidYMid meet" aria-label="Zoom of the selected box" :viewBox="viewBox"
         @pointerdown="onPointerDown($event, 'zoom')" @pointermove="onPointerMove($event, 'zoom')"
         @pointerup="finishDrag($event, 'zoom')" @pointercancel="finishDrag($event, 'zoom')"
         @touchstart="onTouchStart($event, 'zoom')">
      <image id="zoomPhoto" x="0" y="0" preserveAspectRatio="none" :href="row.preview_path" :width="row.width"
             :height="row.height"></image>
      <rect id="zoomBox" :x="box.x" :y="box.y" :width="box.width" :height="box.height"></rect>
      <g id="zoomHandles">
        <rect v-for="handle in handles" :key="'mark-' + handle.corner" :x="handle.mark.x" :y="handle.mark.y"
              :width="handle.mark.width" :height="handle.mark.height" class="handle"/>
        <rect v-for="handle in handles" :key="'hit-' + handle.corner" :x="handle.hit.x" :y="handle.hit.y"
              :width="handle.hit.width" :height="handle.hit.height" :class="'handle-hit ' + handle.corner"
              :data-corner="handle.corner"/>
      </g>
    </svg>
    <div class="zoom-title">{{ title }}</div>
    <div class="zoom-where">{{ where }}</div>
    <div class="zoom-pixels" id="zoomPixels" title="The box in pixels of the picture the classifier cuts it from">{{ pixels }}</div>
  </div>
</template>
