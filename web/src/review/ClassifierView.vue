<script setup lang="ts">
// What the classifier gets for the selected box, beside the zoom: the box cut from the frame and stretched to a square,
// proportions lost, as the car and the training do. Each of its pixels is one the network gets, so it stays blocky on
// a sharp screen.
//   - Once the box settles, the square is the training's own crop, made by the server (GET .../classifier-crop, the
//     training's crop_rectangle, frame at full definition and resize): its caption says which picture it comes from.
//   - While a corner is dragged, before that answer comes, and wherever there is none (the static report, a frame of
//     the video not saved, a failure), it is drawn here from the photo on screen, as close as a browser can.
//   - The grid button shows that crop small, top left, and beside it three of the box as the augmented training varies
//     it across epochs (moved and resized).
import {computed, onBeforeUnmount, ref, shallowRef, useTemplateRef, watch, watchEffect} from 'vue';
import {readLocal, releaseFocus, writeLocal} from './dom';
import {cropRectangle} from './geometry';
import {isVirtual, report, state} from './store';
import type {Box, FrameRow} from './types';

// CLASSIFIER_SIZE of the car's vision.py and input_size of the training.
const SIZE = 160;
// The jittered crops asked for, shown two by two in the square with the crop as drawn, top left.
const SAMPLES = 3;
// How long the box must stay still before the server is asked (ms): arrows held down, a drag's last moves.
const SETTLE_MS = 180;
// The server's answers kept in this page, for a box that goes back where it was (undo, another box and back).
const KEPT = 12;
const SAMPLES_KEY = 'nnslr-classifier-samples';

const props = defineProps<{row: FrameRow; box: Box | null}>();

const view = useTemplateRef<HTMLCanvasElement>('view');
const crop = computed(() => (props.box === null ? null : cropRectangle(props.box, props.row.width, props.row.height)));
/** The photo of this frame once loaded: a crop is never cut from the previous one. */
const photo = shallowRef<HTMLImageElement | null>(null);
const varied = ref(readLocal(SAMPLES_KEY, '0') === '1');

interface Served {
  url: string;
  bitmap: ImageBitmap;
  /** Width × height of the picture the server cut from, and of the box in it. */
  source: [number, number];
  pixels: [number, number];
  samples: number;
}

/** The address of the training's crop of this box, null where the server has none to give. */
const address = computed(() => {
  const box = props.box;
  if (box === null || report.apiBase === null || isVirtual(props.row.frame_key)) return null;
  const sorted = [Math.min(box[0], box[2]), Math.min(box[1], box[3]), Math.max(box[0], box[2]), Math.max(box[1], box[3])];
  return report.apiBase + '/frame/' + props.row.frame_key + '/classifier-crop?box=' + sorted.map(String).join(',') +
    (varied.value ? '&samples=' + SAMPLES : '') + '&run=' + encodeURIComponent(report.runId);
});
const kept = new Map<string, Served>();
const served = shallowRef<Served | null>(null);
/** The address whose crop the server could not make. */
const failed = ref<string | null>(null);
/** The server's square, only while it is the one of the box on screen. */
const shown = computed(() => (served.value !== null && served.value.url === address.value ? served.value : null));
const grid = computed(() => shown.value !== null && shown.value.samples > 0);

function sizeOf(header: string | null): [number, number] | null {
  const match = /^(\d+)x(\d+)$/.exec(header ?? '');
  return match ? [Number(match[1]), Number(match[2])] : null;
}

let timer = 0;
let pending: AbortController | null = null;

async function load(url: string): Promise<void> {
  pending?.abort();
  const mine = new AbortController();
  pending = mine;
  try {
    const response = await fetch(url, {credentials: 'same-origin', signal: mine.signal});
    const source = sizeOf(response.headers.get('X-Crop-Source'));
    const pixels = sizeOf(response.headers.get('X-Crop-Pixels'));
    if (!response.ok || source === null || pixels === null) throw new Error(String(response.status));
    // The CSP has no blob: images: decoded here, drawn on the canvas.
    const bitmap = await createImageBitmap(await response.blob());
    const answer: Served = {url, bitmap, source, pixels, samples: Number(response.headers.get('X-Crop-Samples') ?? 0)};
    kept.delete(url);
    kept.set(url, answer);
    while (kept.size > KEPT) kept.delete(kept.keys().next().value as string);
    if (address.value === url) served.value = answer;
  } catch {
    if (!mine.signal.aborted && address.value === url) failed.value = url;
  } finally {
    if (pending === mine) pending = null;
  }
}

// The box settled (no drag, no change for SETTLE_MS): its crop from the server, unless already here.
watch([address, () => state.drag !== null], ([url, dragging]) => {
  window.clearTimeout(timer);
  if (url === null || dragging) {
    pending?.abort();
    return;
  }
  const known = kept.get(url);
  if (known) {
    served.value = known;
    return;
  }
  if (failed.value === url) return;
  timer = window.setTimeout(() => void load(url), SETTLE_MS);
}, {immediate: true});

onBeforeUnmount(() => {
  window.clearTimeout(timer);
  pending?.abort();
});

function toggleSamples(event: Event): void {
  varied.value = !varied.value;
  writeLocal(SAMPLES_KEY, varied.value ? '1' : '0');
  releaseFocus(event);
}

watch(() => props.row.preview_path, (href) => {
  photo.value = null;
  const image = new Image();
  image.onload = () => {
    if (href === props.row.preview_path) photo.value = image;
  };
  image.src = href;
}, {immediate: true});

/** Width × height of the picture the square is cut from, and of the box in it (the photo's own pixels while drawn here:
 * twice the row's on a drive at full definition, as on the server). */
const measure = computed(() => {
  const answer = shown.value;
  if (answer !== null) return {source: answer.source, pixels: answer.pixels};
  const c = crop.value;
  if (c === null) return null;
  const width = photo.value?.naturalWidth ?? props.row.width;
  const height = photo.value?.naturalHeight ?? props.row.height;
  return {source: [width, height], pixels: [Math.round((c[2] - c[0]) * width / props.row.width),
    Math.round((c[3] - c[1]) * height / props.row.height)]};
});
// Two lines: what the square is and the box's pixels; then the picture the server cut it from, or that the browser
// drew it.
const caption = computed(() => {
  const size = measure.value;
  return size === null ? 'Classifier' : 'Classifier · ' + size.pixels[0] + '×' + size.pixels[1] + ' px';
});
const source = computed(() => {
  const answer = shown.value;
  return answer === null ? 'browser preview' : 'from ' + answer.source[0] + '×' + answer.source[1];
});
const describe = computed(() => {
  const size = measure.value;
  if (size === null) return 'The classifier';
  const what = size.pixels[0] + ' × ' + size.pixels[1] + ' pixels of the frame at ' + size.source[0] + ' × ' +
    size.source[1] + ', stretched to ' + SIZE + ' × ' + SIZE;
  if (shown.value !== null) {
    return grid.value ? 'The classifier, as its training cuts it (top left): ' + what + '; then ' + SAMPLES +
      ' crops of the box as the training varies it across epochs, moved and resized' :
      'The classifier, as its training cuts it: ' + what;
  }
  if (address.value === null) return 'The classifier, drawn from the photo on screen: ' + what;
  if (failed.value === address.value) {
    return 'The classifier, drawn from the photo on screen (the training\'s crop could not be made): ' + what;
  }
  return 'The classifier, drawn from the photo on screen until the box settles: ' + what;
});

watchEffect(() => {
  const canvas = view.value;
  const context = canvas?.getContext('2d');
  if (!canvas || !context) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
  const answer = shown.value;
  if (answer !== null) {
    context.imageSmoothingEnabled = false;
    if (answer.samples === 0) {
      context.drawImage(answer.bitmap, 0, 0, SIZE, SIZE, 0, 0, SIZE, SIZE);
      return;
    }
    // Two by two, each crop whole, the square twice as fine: on a sharp screen each pixel is still one of the crop.
    // Top left the crop as drawn (the strip's first, marked by the page's CSS), then the jittered ones.
    for (let place = 0; place <= Math.min(SAMPLES, answer.samples); place += 1) {
      context.drawImage(answer.bitmap, place * SIZE, 0, SIZE, SIZE, (place % 2) * SIZE,
                        Math.floor(place / 2) * SIZE, SIZE, SIZE);
    }
    context.fillStyle = 'rgba(128, 128, 128, 0.8)';
    context.fillRect(SIZE - 1, 0, 2, 2 * SIZE);
    context.fillRect(0, SIZE - 1, 2 * SIZE, 2);
    return;
  }
  const image = photo.value;
  const c = crop.value;
  if (image === null || c === null) return;
  // Bilinear as the resize of the car and the training when enlarging, the usual case; "high" would be bicubic.
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'medium';
  const scaleX = image.naturalWidth / props.row.width;
  const scaleY = image.naturalHeight / props.row.height;
  context.drawImage(image, c[0] * scaleX, c[1] * scaleY, (c[2] - c[0]) * scaleX, (c[3] - c[1]) * scaleY, 0, 0, SIZE, SIZE);
}, {flush: 'post'});
</script>

<template>
  <figure class="net-view" id="netView" :class="{varied: grid}">
    <canvas ref="view" id="netViewPicture" :width="grid ? 2 * SIZE : SIZE" :height="grid ? 2 * SIZE : SIZE" role="img"
            :aria-label="describe"></canvas>
    <figcaption :title="describe">
      <span class="net-caption" id="netCaption">{{ caption }}</span>
      <span class="net-source" id="netSource" :class="{served: shown !== null}">{{ source }}</span>
      <button v-if="address !== null" type="button" class="net-toggle" id="netSamples" :aria-pressed="varied"
              aria-label="Show the crop with three of the box as the training varies it across epochs"
              title="Top left the crop as drawn, then three as the training varies it across epochs: the box moved and resized (1 epoch in 5 keeps it as drawn); tilt and colour not shown"
              @click="toggleSamples">
        <svg viewBox="0 0 12 12" aria-hidden="true"><rect x="0.5" y="0.5" width="4.5" height="4.5" rx="1"/>
          <rect x="7" y="0.5" width="4.5" height="4.5" rx="1"/><rect x="0.5" y="7" width="4.5" height="4.5" rx="1"/>
          <rect x="7" y="7" width="4.5" height="4.5" rx="1"/></svg>
      </button>
    </figcaption>
  </figure>
</template>
