<script setup lang="ts">
// Train › Brain as a flow of blocks (brain-flow.ts): what the car gives the brain, the network, what it says, and apart what
// only the training has. Pointing at a block (or focusing it) shows its detail beside it; a click keeps it open, a second
// click or Esc closes it. Narrow, the columns stack, without arrows, and a tap opens the detail under the block.
import {computed, nextTick, onBeforeUnmount, onMounted, ref, watch} from 'vue';
import {FLOW_COLUMNS, flowBlocks, type FlowBlock} from './brain-flow';

const props = defineProps<{driver: boolean}>();
const blocks = computed(() => flowBlocks(props.driver));
const byId = computed(() => new Map(blocks.value.map((block) => [block.id, block])));
const columns = computed(() => FLOW_COLUMNS.map((title, column) => ({
  title, blocks: blocks.value.filter((block) => block.column === column)})));

/** Narrower than this the columns stack. */
const STACKED = 700;
const GAP = 12;
const POP_WIDTH = 330;
const grid = ref<HTMLElement | null>(null);
const width = ref(0);
const narrow = computed(() => width.value > 0 && width.value < STACKED);

const hover = ref<string | null>(null);
const pinned = ref<string | null>(null);
const shownId = computed(() => hover.value ?? pinned.value);
const shown = computed(() => shownId.value ? byId.value.get(shownId.value) ?? null : null);

function enter(block: FlowBlock, event: PointerEvent): void {
  if (event.pointerType !== 'touch') hover.value = block.id;
}
function leave(block: FlowBlock): void {
  if (hover.value === block.id) hover.value = null;
}
function pin(block: FlowBlock): void {
  pinned.value = pinned.value === block.id ? null : block.id;
  hover.value = null;
}
function close(): void {
  pinned.value = null;
  hover.value = null;
}

// -- the arrows and the place of the detail, from where the blocks are --------------------------------------------------
interface Line {key: string; d: string; kind: string; lit: boolean}
const lines = ref<Line[]>([]);
const size = ref({w: 0, h: 0});
const popAt = ref<{left: number; top: number} | null>(null);

function boxOf(id: string, origin: DOMRect): {l: number; r: number; t: number; b: number; x: number; y: number} | null {
  const element = grid.value?.querySelector<HTMLElement>(`[data-block="${id}"]`);
  if (!element) return null;
  const rect = element.getBoundingClientRect();
  const l = rect.left - origin.left;
  const t = rect.top - origin.top;
  return {l, r: l + rect.width, t, b: t + rect.height, x: l + rect.width / 2, y: t + rect.height / 2};
}

function lineKind(from: FlowBlock, to: FlowBlock): string {
  if (from.kind === 'train' || to.kind === 'train') return 'train';
  return from.kind === 'off' ? 'off' : 'car';
}

function draw(): void {
  const element = grid.value;
  if (!element) return;
  width.value = element.clientWidth;
  if (narrow.value) {
    lines.value = [];
    popAt.value = null;
    return;
  }
  const origin = element.getBoundingClientRect();
  size.value = {w: element.clientWidth, h: element.clientHeight};
  const found: Line[] = [];
  for (const block of blocks.value) {
    const a = boxOf(block.id, origin);
    for (const id of block.to ?? []) {
      const target = byId.value.get(id);
      const b = boxOf(id, origin);
      if (!a || !b || !target) continue;
      let d: string;
      if (b.l >= a.r - 1) {
        const c = Math.max(18, (b.l - a.r) / 2);
        d = `M${a.r},${a.y} C${a.r + c},${a.y} ${b.l - c},${b.y} ${b.l - 2},${b.y}`;
      } else if (b.r <= a.l + 1) {
        const c = Math.max(18, (a.l - b.r) / 2);
        d = `M${a.l},${a.y} C${a.l - c},${a.y} ${b.r + c},${b.y} ${b.r + 2},${b.y}`;
      } else {
        d = `M${a.x},${a.b} L${b.x},${b.t - 2}`;
      }
      found.push({key: block.id + '>' + id, d, kind: lineKind(block, target), lit: shownId.value === block.id || shownId.value === id});
    }
  }
  lines.value = found;
  place(origin);
}

/** The detail beside its block: on the side with more room, never out of the flow's width. */
function place(origin = grid.value?.getBoundingClientRect()): void {
  const id = shownId.value;
  if (!id || !origin || narrow.value) {
    popAt.value = null;
    return;
  }
  const box = boxOf(id, origin);
  if (!box) return;
  const w = size.value.w;
  const pop = Math.min(POP_WIDTH, w - 2 * GAP);
  const right = box.x < w / 2;
  const left = right ? Math.min(box.r + GAP, w - pop) : Math.max(0, box.l - GAP - pop);
  popAt.value = {left, top: Math.max(0, box.t - 4)};
}

watch(shownId, () => {
  for (const line of lines.value) {
    const [from, to] = line.key.split('>');
    line.lit = shownId.value === from || shownId.value === to;
  }
  place();
});
watch(blocks, () => { void nextTick(draw); });
watch(narrow, () => { void nextTick(draw); });

let observer: ResizeObserver | null = null;
const onKey = (event: KeyboardEvent): void => { if (event.key === 'Escape') close(); };
onMounted(() => {
  void nextTick(draw);
  observer = new ResizeObserver(() => draw());
  if (grid.value) observer.observe(grid.value);
  window.addEventListener('keydown', onKey);
  void document.fonts?.ready.then(draw);
});
onBeforeUnmount(() => {
  observer?.disconnect();
  window.removeEventListener('keydown', onKey);
});
</script>

<template>
  <div class="bf" :class="{stacked: narrow}" id="train-brain-io">
    <p class="bf-legend">
      <span class="bf-key bf-key-in">Gets on the car</span>
      <span class="bf-key bf-key-car">Says on the car</span>
      <span class="bf-key bf-key-train">Only in training</span>
      <span class="bf-key bf-key-off">Not given</span>
      <span class="bf-hint">{{ narrow ? 'Tap a block for what it is.' : 'Point at a block for what it is; a click keeps it open.' }}</span>
    </p>
    <div ref="grid" class="bf-grid" @click.self="close">
      <svg v-if="!narrow" class="bf-lines" :width="size.w" :height="size.h" :viewBox="`0 0 ${size.w} ${size.h}`" aria-hidden="true">
        <defs>
          <marker v-for="kind in ['car', 'train', 'off', 'lit']" :id="'bf-arrow-' + kind" :key="kind" class="bf-arrow" :class="kind"
            viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L8,4 L0,8 z"/>
          </marker>
        </defs>
        <path v-for="line in lines" :key="line.key" :d="line.d" class="bf-line" :class="[line.kind, {lit: line.lit}]"
          :marker-end="`url(#bf-arrow-${line.lit ? 'lit' : line.kind})`"/>
      </svg>
      <section v-for="(column, index) in columns" :key="index" class="bf-col" :class="'bf-col-' + index">
        <h4>{{ column.title }}</h4>
        <template v-for="block in column.blocks" :key="block.id">
          <p v-if="block.id === 'works'" class="bf-divider">only in training</p>
          <button type="button" class="bf-block" :class="['bf-' + block.kind, {on: shownId === block.id}]" :data-block="block.id"
            :aria-expanded="shownId === block.id" @pointerenter="enter(block, $event)" @pointerleave="leave(block)"
            @focus="hover = block.id" @blur="leave(block)" @click="pin(block)">
            <span class="bf-title">{{ block.title }}</span>
            <span class="bf-sub">{{ block.sub }}</span>
          </button>
          <div v-if="narrow && shownId === block.id" class="bf-detail bf-inline">
            <ul><li v-for="fact in block.facts" :key="fact.name"><code>{{ fact.name }}</code> {{ fact.what }}</li></ul>
            <p v-if="block.note" class="bf-note">{{ block.note }}</p>
          </div>
        </template>
        <p v-if="narrow && index < columns.length - 1" class="bf-down" aria-hidden="true">↓</p>
      </section>
      <div v-if="!narrow && shown && popAt" class="bf-detail bf-pop" role="tooltip"
        :style="{left: popAt.left + 'px', top: popAt.top + 'px', width: Math.min(POP_WIDTH, size.w - 2 * GAP) + 'px'}">
        <strong>{{ shown.title }}</strong>
        <ul><li v-for="fact in shown.facts" :key="fact.name"><code>{{ fact.name }}</code> {{ fact.what }}</li></ul>
        <p v-if="shown.note" class="bf-note">{{ shown.note }}</p>
      </div>
    </div>
  </div>
</template>
