<script setup lang="ts">
// A network as a flow of blocks in columns (Train › Brain, brain-flow.ts; the network in use in Models, model-flow.ts): the
// first columns stack their blocks, a column whose blocks have a row puts each on its row (an answer beside what it
// answers, an output beside where it goes), joined by lines: an arrow where something flows, a line with a dot at each end
// and no arrow from a block that is only compared with another (FlowBlock.compares). Pointing at a block (or focusing it) shows its detail beside it; a click keeps it open, a second
// click or Esc closes it. Narrow, the columns stack, without lines, and a tap opens the detail under the block.
import {computed, nextTick, onBeforeUnmount, onMounted, ref, watch} from 'vue';
import type {FlowBlock, FlowKind} from './brain-flow';

const props = defineProps<{
  blocks: FlowBlock[];
  /** The title of each column, in order. */
  columns: string[];
  legend: {kind: FlowKind; label: string}[];
  /** A line across a placed column on its row (from 1), before the block of the next row, and its text. */
  divider?: {column: number; row: number; text: string} | null;
  id: string;
}>();
const blocks = computed(() => props.blocks);
const byId = computed(() => new Map(blocks.value.map((block) => [block.id, block])));
const columns = computed(() => props.columns.map((title, column) => {
  const inColumn = blocks.value.filter((block) => block.column === column);
  return {title, blocks: inColumn, placed: inColumn.length > 0 && inColumn.every((block) => block.row !== undefined)};
}));
/** The rows of the placed columns, 1 the titles. */
const rowCount = computed(() => 1 + Math.max(1, ...blocks.value.map((block) => (block.row ?? 0) + (block.rows ?? 1) - 1)));

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

/** Wide, where a block of a placed column sits in the grid (row 1 is the titles), the divider and the titles. */
function at(block: FlowBlock): Record<string, string> | undefined {
  if (narrow.value || block.row === undefined) return undefined;
  return {gridColumn: String(block.column + 1), gridRow: `${block.row + 1} / span ${block.rows ?? 1}`};
}
const dividerAt = computed(() => narrow.value || !props.divider ? undefined
  : {gridColumn: String(props.divider.column + 1), gridRow: String(props.divider.row + 1)});
const titleAt = (column: number): Record<string, string> | undefined =>
  narrow.value || !columns.value[column]?.placed ? undefined : {gridColumn: String(column + 1), gridRow: '1'};
const gridStyle = computed(() => narrow.value ? undefined : {gridTemplateColumns: `repeat(${columns.value.length}, minmax(0, 1fr))`,
  gridTemplateRows: `repeat(${rowCount.value}, auto)`});

// -- the lines and the place of the detail, from where the blocks are ---------------------------------------------------
/** compare: from an answer to what it answers, a straight line with a dot at each end and no arrow. */
interface Line {key: string; d: string; kind: string; lit: boolean; compare: boolean}
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
  return from.kind === 'train' || to.kind === 'train' ? 'train' : 'car';
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
      const compare = block.compares === true;
      let d: string;
      if (compare) {
        const y = Math.min(Math.max(b.y, a.t + 10), a.b - 10);
        d = `M${a.l - 3},${y} L${b.r + 3},${y}`;
      } else if (b.l >= a.r - 1) {
        const c = Math.max(18, (b.l - a.r) / 2);
        d = `M${a.r},${a.y} C${a.r + c},${a.y} ${b.l - c},${b.y} ${b.l - 2},${b.y}`;
      } else if (b.r <= a.l + 1) {
        const c = Math.max(18, (a.l - b.r) / 2);
        d = `M${a.l},${a.y} C${a.l - c},${a.y} ${b.r + c},${b.y} ${b.r + 2},${b.y}`;
      } else {
        d = `M${a.x},${a.b} L${b.x},${b.t - 2}`;
      }
      found.push({key: block.id + '>' + id, d, kind: lineKind(block, target), compare,
        lit: shownId.value === block.id || shownId.value === id});
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
  <div class="bf" :class="{stacked: narrow}" :id="id">
    <p class="bf-legend">
      <span v-for="key in legend" :key="key.kind" class="bf-key" :class="'bf-key-' + key.kind">{{ key.label }}</span>
      <span class="bf-hint">{{ narrow ? 'Tap a block for what it is.' : 'Point at a block for what it is; a click keeps it open.' }}</span>
    </p>
    <div ref="grid" class="bf-grid" :style="gridStyle" @click.self="close">
      <svg v-if="!narrow" class="bf-lines" :width="size.w" :height="size.h" :viewBox="`0 0 ${size.w} ${size.h}`" aria-hidden="true">
        <defs>
          <marker v-for="kind in ['car', 'train', 'lit']" :id="'bf-arrow-' + kind" :key="kind" class="bf-arrow" :class="kind"
            viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L8,4 L0,8 z"/>
          </marker>
          <marker v-for="kind in ['train', 'lit']" :id="'bf-dot-' + kind" :key="'dot-' + kind" class="bf-arrow" :class="kind"
            viewBox="0 0 8 8" refX="4" refY="4" markerWidth="6" markerHeight="6">
            <circle cx="4" cy="4" r="3.2"/>
          </marker>
        </defs>
        <path v-for="line in lines" :key="line.key" :d="line.d" class="bf-line" :class="[line.kind, {lit: line.lit}]"
          :marker-start="line.compare ? `url(#bf-dot-${line.lit ? 'lit' : line.kind})` : undefined"
          :marker-end="`url(#bf-${line.compare ? 'dot' : 'arrow'}-${line.lit ? 'lit' : line.kind})`"/>
      </svg>
      <section v-for="(column, index) in columns" :key="index" class="bf-col"
        :class="['bf-col-' + index, column.placed ? 'bf-placed' : 'bf-stack', {'bf-answers': column.blocks.some((block) => block.compares)}]"
        :style="{'--bf-col': index + 1}">
        <h4 :style="titleAt(index)">{{ column.title }}</h4>
        <template v-for="block in column.blocks" :key="block.id">
          <p v-if="divider && block.column === divider.column && block.row === divider.row + 1" class="bf-divider" :style="dividerAt">{{ divider.text }}</p>
          <button type="button" class="bf-block" :class="['bf-' + block.kind, {on: shownId === block.id}]" :data-block="block.id"
            :style="at(block)"
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
