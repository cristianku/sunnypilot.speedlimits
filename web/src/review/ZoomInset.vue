<script setup lang="ts">
// The zoom of the selected box, inside the photo (Stage.vue places it in the top corner on the other side from the box):
// its corners and inside move the box too, with what the classifier gets in its corner. A click elsewhere on the photo
// clears the selection and closes it (pointer.ts onSceneClick).
import {computed, ref, useTemplateRef, watchEffect} from 'vue';
import ClassifierView from './ClassifierView.vue';
import {coarsePointer} from './dom';
import {handlesOf, sceneScale, zoomView, type Rect} from './overlay';
import {finishDrag, onPointerDown, onPointerMove, onTouchStart} from './pointer';
import {state} from './store';
import type {Detection, FrameRow} from './types';
import {useScreenSize} from './use-layout';

const props = defineProps<{row: FrameRow; detection: Detection}>();

const zoom = useTemplateRef<SVGSVGElement>('zoom');
const zoomSize = useScreenSize(zoom);

// The classifier's square, in the zoom's top right corner: the box is centred in the part of the zoom it leaves free.
const net = useTemplateRef<InstanceType<typeof ClassifierView>>('net');
const netSize = useScreenSize(computed(() => (net.value?.$el as Element | undefined) ?? null));
// Its margin from the zoom's edge and from the box (px).
const NET_GAP = 6;

/** The zoom's view, as wide as the zoom on screen: the box's view centred in the part left of the square. */
function besideSquare(v: Rect, size: {width: number; height: number}, square: number): Rect {
  if (!size.width || !size.height || !v.width || !v.height) return v;
  const free = Math.max(size.width - (square ? square + 2 * NET_GAP : 0), size.width / 2);
  const scale = Math.min(free / v.width, size.height / v.height);
  const height = size.height / scale;
  return {x: v.x - (free / scale - v.width) / 2, y: v.y - (height - v.height) / 2, width: size.width / scale, height};
}

// While a corner of the zoom is dragged its view stays still, so the corner stays under the finger.
const view = ref<Rect | null>(null);
watchEffect(() => {
  if (state.drag !== null && state.drag.svg === 'zoom') return;
  view.value = besideSquare(zoomView(props.detection.bbox_xyxy, props.row.width, props.row.height), zoomSize.value,
                            netSize.value.width);
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
    <ClassifierView ref="net" :row="row" :box="detection.bbox_xyxy"/>
  </div>
</template>
