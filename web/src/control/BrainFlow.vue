<script setup lang="ts">
// Train › Brain, how the next training's brain is built, as one short flow that reads top to bottom on a phone too
// (docs/brain-training-diagram.md): the picture goes through MobileNetV3-small, the other inputs join its features, all of
// them go into the memory (GRU), and the memory gives the car's two outputs. Beside it (under it, narrow) what only the
// training has: the extra predictions that teach the shared network through their losses and the answers it is compared
// with. The inputs are the groups of brain-card.ts the training gives (the car's, the driver's only with Driver's inputs on):
// tests/review/test_brain_card.py keeps them those of the code. A tap on a block opens what it is under the flow.
import {computed, nextTick, onBeforeUnmount, onMounted, ref, watch} from 'vue';
import type {BrainChoice} from './brain-choices';
import {BRAIN_GROUPS, BRAIN_OUT, BRAIN_OUTPUTS, BRAIN_PICTURE, type BrainFact} from './brain-card';

const props = defineProps<{choices: BrainChoice}>();
const learns = computed(() => ({driver: props.choices.driver === 'on', works: props.choices.works === 'on',
  road: props.choices.road === 'on', types: props.choices.types === 'on'}));

type Kind = 'input' | 'net' | 'car';
interface Block {id: string; title: string; sub: string; kind: Kind; facts: BrainFact[]; note?: string}

const facts = (names: string[]): BrainFact[] => names.map((name) => {
  const group = BRAIN_GROUPS.find((item) => item.name === name)!;
  return {name: group.name + ' · ' + group.size, what: group.what};
});
const SIGN_GROUPS = ['signs', 'plate', 'plate_hours', 'plate_wet', 'sign_distance'];
const carGroups = BRAIN_GROUPS.filter((group) => group.use === 'car').map((group) => group.name);
/** The car's groups that are not the signs': hour, car ahead, country, and any group added after them. */
const placeGroups = carGroups.filter((name) => !SIGN_GROUPS.includes(name));
const driverGroups = BRAIN_GROUPS.filter((group) => group.use === 'driver').map((group) => group.name);

const camera: Block = {id: 'camera', title: 'Road image', sub: '256 × 144', kind: 'input', facts: [BRAIN_PICTURE[0]!]};
const features: Block = {id: 'features', title: 'MobileNetV3-small', sub: 'frozen · image → features', kind: 'net',
  facts: [{name: 'features', what: 'what the frozen MobileNetV3-small sees in the picture, the same on the car'}]};
const inputs = computed<Block[]>(() => [
  {id: 'signs', title: 'Signs read, plates, sign distance', sub: 'what our sign networks read, mistakes included', kind: 'input',
    facts: facts(SIGN_GROUPS.filter((name) => carGroups.includes(name))),
    note: 'As the car reads them, never the signs you confirmed in Sign review. A drive not read yet has them unknown.'},
  {id: 'place', title: 'Hour, country, car ahead', sub: 'from the car', kind: 'input', facts: facts(placeGroups)},
  ...(learns.value.driver ? [{id: 'driver', title: 'Your pedals and speed', sub: 'Driver’s inputs on', kind: 'input' as Kind,
    facts: facts(driverGroups), note: 'Off by default: the answer is your own speed, so with the car’s speed in, the network '
      + 'can learn to say it back.'}] : []),
]);
const memory: Block = {id: 'memory', title: 'GRU memory', sub: 'one step every ¼ s', kind: 'net', facts: [BRAIN_PICTURE[1]!],
  note: 'The image features and every other input go into it at each step; both outputs read its state.'};
const outputs: Block[] = [
  {id: 'speed', title: 'Speed to hold', sub: '1–200 km/h', kind: 'car', facts: [BRAIN_OUTPUTS[0]!]},
  {id: 'ahead', title: 'Next limit + metres', sub: 'or none', kind: 'car', facts: [BRAIN_OUTPUTS[1]!, BRAIN_OUTPUTS[2]!]},
];
const blocks = computed(() => [camera, features, ...inputs.value, memory, ...outputs]);

/** The predictions only the training has, by its switches: each only teaches the shared network through its loss. */
const extras = computed(() => [
  {on: learns.value.types, title: 'Road type', what: 'taught by the map’s road types, read offline and corrected in Speed review'},
  {on: learns.value.works, title: 'Roadworks', what: 'taught by the roadworks marked in Speed review'},
  {on: learns.value.road, title: 'Road characteristics', what: 'narrow, country, gravel, potholes: taught by the kinds of road '
    + 'marked in Speed review'},
].filter((extra) => extra.on));
const targets = [
  {title: 'Correct speed', what: 'your recorded speed on dashcam drives, corrected or left out in Speed review: what Speed to hold '
    + 'learns, never an input'},
  {title: 'Next limit / metres', what: 'from the sign events our networks read, your Sign review over them: what the car’s next '
    + 'limit and its metres learn'},
];
const out = computed(() => BRAIN_OUT.filter((reason) => learns.value.types || reason.name !== 'no_road_type'));

// -- the detail of a block: a tap (or click) keeps it open under the flow, a second tap or Esc closes it ------------------
const pinned = ref<string | null>(null);
const shown = computed(() => blocks.value.find((block) => block.id === pinned.value) ?? null);
const pin = (block: Block): void => { pinned.value = pinned.value === block.id ? null : block.id; };
const onKey = (event: KeyboardEvent): void => { if (event.key === 'Escape') pinned.value = null; };

// -- the lines, from where the blocks are: each input into a rail on the right, the rail into the memory, the memory to
// both outputs ---------------------------------------------------------------------------------------------------------
const main = ref<HTMLElement | null>(null);
const size = ref({w: 0, h: 0});
const lines = ref<{key: string; d: string; arrow: boolean}[]>([]);

function box(id: string, origin: DOMRect) {
  const rect = main.value?.querySelector<HTMLElement>(`[data-block="${id}"]`)?.getBoundingClientRect();
  if (!rect) return null;
  const l = rect.left - origin.left;
  const t = rect.top - origin.top;
  return {l, r: l + rect.width, t, b: t + rect.height, x: l + rect.width / 2, y: t + rect.height / 2};
}

function draw(): void {
  const element = main.value;
  if (!element) return;
  const origin = element.getBoundingClientRect();
  size.value = {w: element.clientWidth, h: element.clientHeight};
  const rows = ['features', ...inputs.value.map((block) => block.id)].map((id) => box(id, origin));
  const gru = box('memory', origin);
  if (rows.some((row) => !row) || !gru) return;
  const rail = size.value.w - 9;
  const found = rows.map((row, index) => ({key: 'in' + index, d: `M${row!.r},${row!.y} H${rail}`, arrow: false}));
  const join = (rows.at(-1)!.b + gru.t) / 2;
  found.push({key: 'rail', d: `M${rail},${rows[0]!.y} V${join} H${gru.x} V${gru.t - 2}`, arrow: true});
  const fork = gru.b + 14;
  for (const id of outputs.map((block) => block.id)) {
    const to = box(id, origin);
    if (to) found.push({key: id, d: `M${gru.x},${gru.b} V${fork} H${to.x} V${to.t - 2}`, arrow: true});
  }
  lines.value = found;
}

let observer: ResizeObserver | null = null;
watch(inputs, () => { void nextTick(draw); });
watch(shown, () => { void nextTick(draw); });
onMounted(() => {
  void nextTick(draw);
  observer = new ResizeObserver(() => draw());
  if (main.value) observer.observe(main.value);
  window.addEventListener('keydown', onKey);
  void document.fonts?.ready.then(draw);
});
onBeforeUnmount(() => {
  observer?.disconnect();
  window.removeEventListener('keydown', onKey);
});
</script>

<template>
  <div class="bd" id="train-brain-io">
    <p class="bf-legend">
      <span class="bf-key bf-key-input">Input</span>
      <span class="bf-key bf-key-net">The network</span>
      <span class="bf-key bf-key-car">Output on the car</span>
      <span class="bf-key bf-key-train">Only during training</span>
      <span class="bf-hint">Tap a block for what it is.</span>
    </p>
    <div class="bd-body">
      <div ref="main" class="bd-main">
        <svg class="bd-lines" :width="size.w" :height="size.h" :viewBox="`0 0 ${size.w} ${size.h}`" aria-hidden="true">
          <defs>
            <marker id="bd-arrow" class="bf-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7"
              orient="auto"><path d="M0,0 L8,4 L0,8 z"/></marker>
          </defs>
          <path v-for="line in lines" :key="line.key" :d="line.d" class="bf-line"
            :marker-end="line.arrow ? 'url(#bd-arrow)' : undefined"/>
        </svg>
        <h4 class="bd-head">Inputs · 4 times a second</h4>
        <div class="bd-ins">
          <div class="bd-row">
            <button v-for="(block, index) in [camera, features]" :key="block.id" type="button" class="bf-block"
              :class="['bf-' + block.kind, {on: pinned === block.id}, index ? 'bd-after' : '']" :data-block="block.id"
              :aria-expanded="pinned === block.id" @click="pin(block)">
              <span class="bf-title">{{ block.title }}</span>
              <span class="bf-sub">{{ block.sub }}</span>
            </button>
          </div>
          <button v-for="block in inputs" :key="block.id" type="button" class="bf-block"
            :class="['bf-' + block.kind, {on: pinned === block.id}]" :data-block="block.id" :aria-expanded="pinned === block.id"
            @click="pin(block)">
            <span class="bf-title">{{ block.title }}</span>
            <span class="bf-sub">{{ block.sub }}</span>
          </button>
        </div>
        <button type="button" class="bf-block bf-net bd-gru" :class="{on: pinned === memory.id}" :data-block="memory.id"
          :aria-expanded="pinned === memory.id" @click="pin(memory)">
          <span class="bf-title">{{ memory.title }}</span>
          <span class="bf-sub">{{ memory.sub }}</span>
        </button>
        <div class="bd-outs">
          <button v-for="block in outputs" :key="block.id" type="button" class="bf-block bf-car"
            :class="{on: pinned === block.id}" :data-block="block.id" :aria-expanded="pinned === block.id" @click="pin(block)">
            <span class="bf-title">{{ block.title }}</span>
            <span class="bf-sub">{{ block.sub }}</span>
          </button>
        </div>
        <div v-if="shown" class="bf-detail bf-inline bd-detail">
          <strong>{{ shown.title }}</strong>
          <ul><li v-for="fact in shown.facts" :key="fact.name"><code>{{ fact.name }}</code> {{ fact.what }}</li></ul>
          <p v-if="shown.note" class="bf-note">{{ shown.note }}</p>
        </div>
      </div>
      <aside class="bd-train">
        <h4>Only during training</h4>
        <p>Extra labels teach the shared network during training; they are not inputs and are not needed in the car.</p>
        <template v-if="extras.length">
          <h5>Extra predictions</h5>
          <ul><li v-for="extra in extras" :key="extra.title"><strong>{{ extra.title }}</strong><span>{{ extra.what }}</span></li></ul>
        </template>
        <h5>Training targets</h5>
        <ul><li v-for="target in targets" :key="target.title"><strong>{{ target.title }}</strong><span>{{ target.what }}</span></li></ul>
        <p class="bf-note">The map only makes labels offline: nothing of the map goes into the brain on the car.</p>
        <details class="bd-out">
          <summary>{{ out.length }} kinds of second answer nothing</summary>
          <ul><li v-for="reason in out" :key="reason.name"><span>{{ reason.what }}</span></li></ul>
          <p class="bf-note">They still go through the memory, as on the car.</p>
        </details>
      </aside>
    </div>
  </div>
</template>
