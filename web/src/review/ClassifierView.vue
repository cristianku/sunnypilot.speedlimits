<script setup lang="ts">
// What the classifier gets for the selected box, beside the zoom: the box cut from the frame and stretched to a square,
// proportions lost, as the car and the training do. Made here from the photo on screen, so it follows a corner being
// dragged. Each of its pixels is one the network gets, so it stays blocky on a sharp screen.
import {computed, shallowRef, useTemplateRef, watch, watchEffect} from 'vue';
import {cropRectangle} from './geometry';
import type {Box, FrameRow} from './types';

// CLASSIFIER_SIZE of the car's vision.py and input_size of the training.
const SIZE = 160;

const props = defineProps<{row: FrameRow; box: Box | null}>();

const view = useTemplateRef<HTMLCanvasElement>('view');
const crop = computed(() => (props.box === null ? null : cropRectangle(props.box, props.row.width, props.row.height)));
const pixels = computed(() => {
  const c = crop.value;
  return c === null ? '' : (c[2] - c[0]) + ' × ' + (c[3] - c[1]) + ' px';
});

/** The photo of this frame once loaded: a crop is never cut from the previous one. */
const photo = shallowRef<HTMLImageElement | null>(null);
watch(() => props.row.preview_path, (href) => {
  photo.value = null;
  const image = new Image();
  image.onload = () => {
    if (href === props.row.preview_path) photo.value = image;
  };
  image.src = href;
}, {immediate: true});

watchEffect(() => {
  const context = view.value?.getContext('2d');
  if (!context) return;
  context.clearRect(0, 0, SIZE, SIZE);
  const image = photo.value;
  const c = crop.value;
  if (image === null || c === null) return;
  // Bilinear as the resize of the car and the training when enlarging, the usual case; "high" would be bicubic.
  context.imageSmoothingQuality = 'medium';
  const scaleX = image.naturalWidth / props.row.width;
  const scaleY = image.naturalHeight / props.row.height;
  context.drawImage(image, c[0] * scaleX, c[1] * scaleY, (c[2] - c[0]) * scaleX, (c[3] - c[1]) * scaleY, 0, 0, SIZE, SIZE);
});
</script>

<template>
  <figure class="net-view" id="netView">
    <canvas ref="view" id="netViewPicture" :width="SIZE" :height="SIZE" role="img"
            :aria-label="'What the classifier gets: ' + pixels + ' of the frame stretched to ' + SIZE + ' by ' + SIZE"></canvas>
    <figcaption title="The box as the classifier gets it: stretched to a square of 160 × 160 from this many pixels of the frame">Classifier · {{ pixels }}</figcaption>
  </figure>
</template>
