<script setup lang="ts">
// Models › Regression, a certified drive opened ("/regression?drive=<route>"): the drive player (DriveReviewDrive.vue) with one lane more,
// his speed and the curves of the models lit, and at its side each curve with its value at the time shown and how far it
// stays from his speed. It is open loop: the car drove as he drove, each curve is what a model said looking at that video.
// A driving model's plan is drawn on the moment it speaks of, a few seconds after the frame it saw (nnslr_tools.regression).
import {computed, onMounted, ref, shallowRef, watch} from 'vue';
import {api, messageOf, post} from '../common/api';
import {isoWhen} from '../common/format';
import {askConfirm} from '../composables/confirm';
import {clockOf} from './drive-review';
import DriveReviewDrive from './DriveReviewDrive.vue';
import {brainKey, type Curve, firstChoice, lit, type Mark, markOf, pathOf, type RegressionDrive, type RegressionPayload, scoreText, toggle,
  valueAt} from './regression';
import Segmented from './Segmented.vue';
import {goTo} from './tabs';

const props = defineProps<{route: string; readOnly: boolean}>();
const RATES = [1, 2, 4] as const;
const HEIGHT = 132;
const TOP = 6;
const BOTTOM = 6;
const HORIZONS = [{key: '2', label: '2 s'}, {key: '5', label: '5 s'}, {key: '10', label: '10 s'}] as const;

const seconds = ref(0);
const follow = (at: number): void => { seconds.value = at; };

const catalog = shallowRef<RegressionPayload | null>(null);
const drive = shallowRef<RegressionDrive | null>(null);
const error = ref('');
const horizon = ref<'2' | '5' | '10'>('5');
/** The brain that caps the "with Smartspeed" curves: the first one lit. */
const cap = computed(() => catalog.value?.brains.find((brain) => (lit.value ?? []).includes(brainKey(brain.key)))?.key ?? null);
async function read(): Promise<void> {
  try {
    const query = new URLSearchParams({horizon: horizon.value});
    if (cap.value) query.set('brain', cap.value);
    drive.value = await api<RegressionDrive>(`/api/regression/routes/${encodeURIComponent(props.route)}?${query.toString()}`);
    error.value = '';
  } catch (problem) {
    error.value = messageOf(problem);
  }
}
onMounted(async () => {
  try {
    catalog.value = await api<RegressionPayload>('/api/regression/routes');
    lit.value ??= firstChoice(catalog.value.brains);
    await read();
  } catch (problem) {
    error.value = messageOf(problem);
  }
});
watch([horizon, cap], () => { if (catalog.value) void read(); });

const curves = computed(() => drive.value?.curves ?? []);
const isLit = (key: string): boolean => (lit.value ?? []).includes(key);
const shown = computed(() => curves.value.filter((curve) => isLit(curve.key)));
/** The rows of the side: every model computed here, a model "with Smartspeed" only next to its model lit. */
const rows = computed(() => curves.value.filter((curve) => curve.kind !== 'car' || isLit(curve.key.slice(4))));
const hole = computed(() => drive.value?.hole_s ?? 3);
const mark = (key: string): Mark => markOf(key, catalog.value?.models ?? [], catalog.value?.brains ?? []);

// -- the lane ---------------------------------------------------------------------------------------------------------------
/** The scale of the lane: tens of km/h up to the fastest value lit, the same at every zoom. */
const top = computed(() => {
  const all = [drive.value?.own ?? [], ...shown.value.map((curve) => curve.points)];
  return Math.ceil(Math.max(50, ...all.flatMap((points) => points.map((point) => point[1]))) / 10) * 10;
});
const y = (kph: number): number => TOP + (1 - kph / top.value) * (HEIGHT - TOP - BOTTOM);
const ticks = computed(() => [0, 50, 100, 150].filter((kph) => kph <= top.value));
/** The paths of the time window: the lane is drawn again at every frame of the video, the curves only when the window moves. */
let drawn: {key: string; paths: Map<string, string>} | null = null;
function paths(from: number, to: number, width: number, x: (second: number) => number): Map<string, string> {
  const key = [from, to, width, top.value, horizon.value, shown.value.map((curve) => curve.key).join()].join('|');
  if (drawn?.key === key) return drawn.paths;
  const made = new Map<string, string>();
  made.set('own', pathOf(drive.value?.own ?? [], from, to, hole.value, x, y));
  for (const curve of shown.value) made.set(curve.key, pathOf(curve.points, from, to, hole.value, x, y));
  drawn = {key, paths: made};
  return made;
}
watch(drive, () => { drawn = null; });

// -- the values at a time ---------------------------------------------------------------------------------------------------
const round = (value: number | null): string => value === null ? '–' : String(Math.round(value));
const ownAt = (second: number): number | null => valueAt(drive.value?.own ?? [], second, hole.value);
const curveAt = (curve: Curve, second: number): number | null => valueAt(curve.points, second, hole.value);

/** The mouse over the lane: a line and the values there. On a phone the side follows the finger through the time shown. */
const hover = ref<{second: number; left: number; flip: boolean} | null>(null);
function point(event: PointerEvent, from: number, to: number, width: number): void {
  if (event.pointerType !== 'mouse') return;
  const box = (event.currentTarget as SVGElement).getBoundingClientRect();
  const left = event.clientX - box.left;
  hover.value = {second: from + left / width * (to - from), left, flip: left > width / 2};
}

// -- the drive --------------------------------------------------------------------------------------------------------------
async function remove(): Promise<void> {
  const yes = await askConfirm({title: 'Remove this drive from the certified ones?', action: 'Remove',
    text: 'Its results stay on the server: certifying it again brings them back.'});
  if (!yes) return;
  try {
    await post(`/api/regression/routes/${encodeURIComponent(props.route)}/certify`, {certified: false});
    goTo('regression');
  } catch (problem) {
    error.value = messageOf(problem);
  }
}
const capLabel = computed(() => catalog.value?.brains.find((brain) => brain.key === drive.value?.brain)?.label ?? '');
const certifiedText = computed(() => drive.value?.certified ? 'Certified ' + isoWhen(drive.value.certified.at) : 'Not certified.');
</script>

<template>
  <DriveReviewDrive class="rg-view" :route="route" :read-only="readOnly" tab="regression" back-label="Regression"
                    :rates="RATES" start-zoom="ten" @time="follow">
    <template #lane="{x, width, from, to}">
      <div class="dr-lane rg-lane">
        <span class="dr-lane-name">Models<small>km/h</small></span>
        <svg :width="width" :height="HEIGHT" :viewBox="`0 0 ${width} ${HEIGHT}`" role="img" aria-label="Your speed and the speed of each model lit"
             @pointermove="point($event, from, to, width)" @pointerleave="hover = null">
          <g v-for="kph in ticks" :key="kph">
            <line :x1="0" :x2="width" :y1="y(kph)" :y2="y(kph)" class="rg-grid"/>
            <text x="3" :y="y(kph) - 2" class="rg-tick">{{ kph }}</text>
          </g>
          <path v-for="curve in shown" :key="curve.key" :d="paths(from, to, width, x).get(curve.key)" class="rg-line"
                :stroke="mark(curve.key).color" :stroke-dasharray="mark(curve.key).dash"/>
          <path :d="paths(from, to, width, x).get('own')" class="rg-line rg-own"/>
          <line v-if="hover" :x1="hover.left" :x2="hover.left" y1="0" :y2="HEIGHT" class="rg-hover"/>
        </svg>
        <div v-if="hover" class="rg-tip" :style="hover.flip ? {right: `calc(${width - hover.left}px + 10px)`} : {left: `calc(var(--dr-name) + ${hover.left}px + 10px)`}">
          <b>{{ clockOf(hover.second) }}</b>
          <div><svg class="rg-key" viewBox="0 0 22 4" aria-hidden="true"><line x1="0" x2="22" y1="2" y2="2" class="rg-line rg-own"/></svg>
            <span>Your speed</span><em>{{ round(ownAt(hover.second)) }}</em></div>
          <div v-for="curve in shown" :key="curve.key">
            <svg class="rg-key" viewBox="0 0 22 4" aria-hidden="true"><line x1="0" x2="22" y1="2" y2="2" class="rg-line" :stroke="mark(curve.key).color"
                                                                            :stroke-dasharray="mark(curve.key).dash"/></svg>
            <span>{{ curve.label }}</span><em>{{ round(curveAt(curve, hover.second)) }}</em>
          </div>
        </div>
      </div>
    </template>
    <template #side>
      <section class="sr-card rg-card" aria-labelledby="rg-card-title">
        <h3 id="rg-card-title">Speeds at {{ clockOf(seconds) }}</h3>
        <p v-if="error" class="dr-hint">{{ error }}</p>
        <p v-else-if="!drive" class="dr-hint">Loading…</p>
        <template v-else>
          <ul class="rg-rows">
            <li>
              <div class="rg-row">
                <svg class="rg-key" viewBox="0 0 22 4" aria-hidden="true"><line x1="0" x2="22" y1="2" y2="2" class="rg-line rg-own"/></svg>
                <strong>Your speed</strong><em>{{ round(ownAt(seconds)) }}</em>
              </div>
            </li>
            <li v-for="curve in rows" :key="curve.key">
              <button type="button" class="rg-row" :aria-pressed="isLit(curve.key)" @click="toggle(curve.key)">
                <svg class="rg-key" viewBox="0 0 22 4" aria-hidden="true"><line x1="0" x2="22" y1="2" y2="2" class="rg-line" :stroke="mark(curve.key).color"
                                                                                :stroke-dasharray="mark(curve.key).dash"/></svg>
                <strong>{{ curve.label }}</strong><em>{{ round(curveAt(curve, seconds)) }}</em>
                <small>{{ scoreText(curve.score, drive.too_fast_kph) }}</small>
              </button>
            </li>
          </ul>
          <p v-if="!curves.length" class="dr-hint">Nothing computed on this drive yet: press Run in Regression.</p>
          <p v-if="capLabel && curves.some((curve) => curve.kind === 'car')" class="dr-hint">“With Smartspeed” is the lower of the model and Smartspeed
            {{ capLabel }}.</p>
          <div class="rg-horizon">
            <Segmented v-model="horizon" :options="HORIZONS" label="How far ahead the plan looks"/>
            <p class="dr-hint">The driving models say the speed {{ horizon }} s ahead, drawn on that moment.</p>
          </div>
          <p class="dr-hint">The car drove at your speed: each curve is what a model would have said from this video, not what the car would have
            done.</p>
          <div v-if="drive.failures.length" class="rg-out">
            <p v-for="failure in drive.failures" :key="failure.key" class="dr-hint">{{ failure.label }} failed: {{ failure.error }}</p>
          </div>
          <div class="rg-out rg-add">
            <p class="dr-hint">{{ certifiedText }}</p>
            <button v-if="drive.certified" type="button" class="btn small" :disabled="readOnly" @click="remove">Remove</button>
          </div>
        </template>
      </section>
    </template>
  </DriveReviewDrive>
</template>
