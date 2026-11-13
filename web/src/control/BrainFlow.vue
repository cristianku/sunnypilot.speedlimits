<script setup lang="ts">
// Train › Brain, how the next training's brain is built, as one short flow that reads top to bottom on a phone too: the
// picture goes through MobileNetV3-small, the other inputs join its features, all of them go into the memory (GRU), and the
// memory gives two groups of outputs: the car's one, the speed to hold, and those only the training has (by its switches), which teach the
// shared weights through their losses and are dropped from the car's file. Under each group, the correct labels it is
// compared with: the map only offline, for the road type. The inputs are the groups of brain-card.ts the training gives (the
// car's; never the driver's pedals nor the car's speed, nor the metres to a sign: the boxes of the signs instead):
// tests/review/test_brain_card.py keeps them those of the code. A tap
// on a block opens what it is under the flow.
import {computed, nextTick, onBeforeUnmount, onMounted, ref, watch} from 'vue';
import {plural} from '../common/format';
import type {BrainChoice} from './brain-choices';
import {BRAIN_GROUPS, BRAIN_OUT, BRAIN_OUTPUTS, BRAIN_PICTURE, type BrainFact} from './brain-card';

const props = defineProps<{choices: BrainChoice}>();
const learns = computed(() => ({works: props.choices.works === 'on', road: props.choices.road === 'on',
  types: props.choices.types === 'on'}));

type Kind = 'input' | 'net' | 'car' | 'train';
interface Block {id: string; title: string; sub: string; kind: Kind; facts: BrainFact[]; note?: string}

const facts = (names: string[]): BrainFact[] => names.map((name) => {
  const group = BRAIN_GROUPS.find((item) => item.name === name)!;
  return {name: group.name + ' · ' + group.size, what: group.what};
});
const SIGN_GROUPS = ['signs', 'plate', 'plate_hours', 'plate_wet', 'sign_boxes', 'road_signs', 'light_boxes'];
const carGroups = BRAIN_GROUPS.filter((group) => group.use === 'car').map((group) => group.name);
/** The car's groups that are not the signs': hour, car ahead, country, and any group added after them. */
const placeGroups = carGroups.filter((name) => !SIGN_GROUPS.includes(name));

const camera: Block = {id: 'camera', title: 'Road image', sub: '256 × 144', kind: 'input', facts: [BRAIN_PICTURE[0]!]};
const features: Block = {id: 'features', title: 'MobileNetV3-small', sub: 'frozen · image → features', kind: 'net',
  facts: [{name: 'features', what: 'what the frozen MobileNetV3-small sees in the picture, the same on the car'}]};
const inputs = computed<Block[]>(() => [
  {id: 'signs', title: 'Signs read, plates, sign boxes, traffic lights', sub: 'what our sign networks read, mistakes included', kind: 'input',
    facts: facts(SIGN_GROUPS.filter((name) => carGroups.includes(name))),
    note: 'As the car reads them, never the signs you confirmed in Sign review. A drive not read yet has them unknown. '
      + 'No metres to a sign: the brain gets the box of each sign, where it is in the picture and how big.'},
  {id: 'place', title: 'Hour, country, car ahead', sub: 'from the car', kind: 'input', facts: facts(placeGroups),
    note: 'Never your pedals nor the car’s speed: the answer is your own speed, and a brain that had them followed it.'},
]);
const memory: Block = {id: 'memory', title: 'GRU memory', sub: 'one step every ¼ s', kind: 'net', facts: [BRAIN_PICTURE[1]!],
  note: 'The image features and every other input go into it at each step; every output reads its state.'};
const outputs: Block[] = [
  {id: 'speed', title: 'Speed to hold', sub: '1–200 km/h', kind: 'car', facts: [BRAIN_OUTPUTS[0]!],
    note: 'All the car reads of the brain: it lowers this speed itself before a sign. No next limit and no metres to it.'},
];
/** The outputs only the training has, by its switches: each teaches the shared weights through its own loss and is dropped
 * from the file the car gets (brain_export). source: where its correct labels come from. */
const extras = computed<(Block & {source: string; map: boolean})[]>(() => [
  {on: learns.value.types, id: 'type', title: 'Road type', sub: 'motorway, rural, town…', map: true,
    source: 'OpenStreetMap, read offline, corrected in Speed review',
    what: 'the road type it sees, compared with the map’s, corrected in Speed review where it is wrong'},
  {on: learns.value.works, id: 'works', title: 'Roadworks', sub: 'inside roadworks or not',
    source: 'your marks in Speed review', what: 'whether the car is inside roadworks, compared with the ones marked in Speed review'},
  {on: learns.value.road, id: 'road', title: 'Road conditions', sub: 'narrow, country, gravel, potholes',
    source: 'your marks in Speed review', what: 'the kinds of road it sees, compared with the ones marked in Speed review'},
].filter((extra) => extra.on).map((extra) => ({id: extra.id, title: extra.title, sub: extra.sub, kind: 'train' as Kind,
  source: extra.source, map: extra.map === true, facts: [{name: extra.title, what: extra.what}],
  note: 'Only in training: its error updates the shared weights, then it is dropped from the car’s file.'})));
const blocks = computed(() => [camera, features, ...inputs.value, memory, ...outputs, ...extras.value]);
const out = computed(() => BRAIN_OUT.filter((reason) => learns.value.types || reason.name !== 'no_road_type'));

// -- the detail of a block: a tap (or click) keeps it open under the flow, a second tap or Esc closes it ------------------
const pinned = ref<string | null>(null);
const shown = computed(() => blocks.value.find((block) => block.id === pinned.value) ?? null);
const pin = (block: Block): void => { pinned.value = pinned.value === block.id ? null : block.id; };
const onKey = (event: KeyboardEvent): void => { if (event.key === 'Escape') pinned.value = null; };

// -- the lines, from where the blocks are: each input into a rail on the right, the rail into the memory, the memory to
// the outputs ----------------------------------------------------------------------------------------------------------
const main = ref<HTMLElement | null>(null);
const size = ref({w: 0, h: 0});
interface Line {key: string; d: string; arrow: boolean; compare?: boolean}
const lines = ref<Line[]>([]);

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
  const found: Line[] = rows.map((row, index) => ({key: 'in' + index, d: `M${row!.r},${row!.y} H${rail}`, arrow: false}));
  const join = (rows.at(-1)!.b + gru.t) / 2;
  found.push({key: 'rail', d: `M${rail},${rows[0]!.y} V${join} H${gru.x} V${gru.t - 2}`, arrow: true});
  const fork = gru.b + 14;
  const car = box('group-car', origin);
  for (const id of ['group-car', 'group-train']) {
    const to = box(id, origin);
    if (!to) continue;
    // Narrow, the training group sits under the car's: its line goes down the rail on the right, past the car's outputs.
    const under = id === 'group-train' && car && to.t > car.b;
    found.push({key: id, arrow: true, d: under ? `M${gru.r},${gru.y} H${rail} V${to.t - 12} H${to.x} V${to.t - 2}`
      : `M${gru.x},${gru.b} V${fork} H${to.x} V${to.t - 2}`});
  }
  // Each group of outputs and its correct labels: compared in training, nothing flows back into the network as an input.
  for (const id of ['car', 'train']) {
    const from = box('group-' + id, origin);
    const to = box('labels-' + id, origin);
    if (from && to) found.push({key: 'labels-' + id, d: `M${from.x},${from.b + 3} V${to.t - 3}`, arrow: false, compare: true});
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
      <span class="bf-key bf-key-train">Output only in training</span>
      <span class="bf-hint">Tap a block for what it is.</span>
    </p>
    <div ref="main" class="bd-main">
      <svg class="bd-lines" :width="size.w" :height="size.h" :viewBox="`0 0 ${size.w} ${size.h}`" aria-hidden="true">
        <defs>
          <marker id="bd-arrow" class="bf-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7"
            orient="auto"><path d="M0,0 L8,4 L0,8 z"/></marker>
          <marker id="bd-dot" class="bf-arrow" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="6" markerHeight="6">
            <circle cx="4" cy="4" r="3.2"/></marker>
        </defs>
        <path v-for="line in lines" :key="line.key" :d="line.d" class="bf-line" :class="{train: line.compare}"
          :marker-start="line.compare ? 'url(#bd-dot)' : undefined"
          :marker-end="line.compare ? 'url(#bd-dot)' : line.arrow ? 'url(#bd-arrow)' : undefined"/>
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
        <div class="bd-side">
          <section class="bd-group bd-group-car" data-block="group-car">
            <h5>On the car · {{ plural(outputs.length, 'output') }}</h5>
            <button v-for="block in outputs" :key="block.id" type="button" class="bf-block bf-car" :class="{on: pinned === block.id}"
              :data-block="block.id" :aria-expanded="pinned === block.id" @click="pin(block)">
              <span class="bf-title">{{ block.title }}</span>
              <span class="bf-sub">{{ block.sub }}</span>
            </button>
          </section>
          <div class="bd-labels" data-block="labels-car">
            <h5>Compared with</h5>
            <ul>
              <li><strong>Your speed</strong> recorded on dashcam drives, corrected in Speed review</li>
            </ul>
          </div>
        </div>
        <div v-if="extras.length" class="bd-side">
          <section class="bd-group bd-group-train" data-block="group-train">
            <h5>Only in training · {{ plural(extras.length, 'output') }}</h5>
            <button v-for="block in extras" :key="block.id" type="button" class="bf-block bf-train" :class="{on: pinned === block.id}"
              :data-block="block.id" :aria-expanded="pinned === block.id" @click="pin(block)">
              <span class="bf-title">{{ block.title }}</span>
              <span class="bf-sub">{{ block.sub }}</span>
            </button>
          </section>
          <div class="bd-labels" data-block="labels-train">
            <h5>Compared with the correct labels</h5>
            <ul>
              <li v-for="block in extras" :key="block.id"><strong>{{ block.title }}</strong> {{ block.source }}<span
                v-if="block.map" class="bd-map">map: offline only</span></li>
            </ul>
          </div>
        </div>
      </div>
      <div v-if="shown" class="bf-detail bf-inline bd-detail">
        <strong>{{ shown.title }}</strong>
        <ul><li v-for="fact in shown.facts" :key="fact.name"><code>{{ fact.name }}</code> {{ fact.what }}</li></ul>
        <p v-if="shown.note" class="bf-note">{{ shown.note }}</p>
      </div>
    </div>
    <p class="bd-sum">In training every output’s error updates the shared weights (the image projection and the GRU; MobileNetV3
      itself stays frozen). For the comma 4 the training-only outputs are dropped: the car runs the speed to hold alone, with no map.</p>
    <details class="bd-out">
      <summary>{{ out.length }} kinds of second answer nothing</summary>
      <ul><li v-for="reason in out" :key="reason.name">{{ reason.what }}</li></ul>
      <p class="bf-note">They still go through the memory, as on the car.</p>
    </details>
  </div>
</template>
