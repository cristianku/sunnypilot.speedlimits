<script setup lang="ts">
// Data › Speed review, a drive opened ("/speed-review?drive=<route>"): the drive player (DriveReviewDrive.vue)
// with one lane more, the answer the brain learns, and the commands at its side. The answer is his own dashboard speed
// at every moment (speed_review.py driver_speed): nothing to propose and nothing to confirm. A block is the only
// correction: "Block from here" keeps a start on the page, "Block to here" saves the block from that start to the time
// shown with its kph, which starts from his speed at the start, or left out of the training. A block is changed by saving
// it again with the same ends, and removed with its start. The lane draws his speed as a step line, the blocks over it with
// their kph, the roadworks and how far the drive was gone through for them. Roadworks are marked next to the speed, two
// taps on the frame shown; a third tap says that the drive was gone through up to the frame shown, so that no mark there
// means no works, and the drive opens there again. The picture shows the answer next to the car's own speed, so that the
// number read is the right one.
import {computed, onBeforeUnmount, onMounted, ref, shallowRef, useTemplateRef, watch} from 'vue';
import {api, messageOf, post} from '../common/api';
import {plural} from '../common/format';
import {isGuest} from '../common/guest';
import {toast} from '../composables/toast';
import {clockOf} from './drive-review';
import DriveReviewDrive from './DriveReviewDrive.vue';
import {type Block, blockAt, canEndWorks, clampKph, MIN_BLOCK_NS, type SpeedDrive, speedAt, speedIndex, worksAt,
  type WorksTap} from './speed-review';

const props = defineProps<{route: string; readOnly: boolean}>();
const RATES = [1, 2, 4] as const;
const HEIGHT = 84;
/** The speed is drawn from TOP to PLOT; under it the roadworks bar and the line of what was gone through. */
const TOP = 16;
const PLOT = 74;
/** A new block with no speed of his at its start begins at this kph. */
const FALLBACK_KPH = 50;

const player = useTemplateRef<InstanceType<typeof DriveReviewDrive>>('player');
const seconds = ref(0);
const t0 = ref(0);
function follow(at: number, start: number): void {
  seconds.value = at;
  t0.value = start;
}
const now = computed(() => Math.round(t0.value + seconds.value * 1e9));
const secondOf = (ns: number): number => (ns - t0.value) / 1e9;

// -- the drive -------------------------------------------------------------------------------------------------------------
const drive = shallowRef<SpeedDrive | null>(null);
const error = ref('');
const address = (): string => '/api/speed-review/drives/' + encodeURIComponent(props.route);
function show(answer: SpeedDrive): void {
  // A cut answer (the page left while it came) reads as {message} in api(): never a drive without its blocks.
  if (!answer || !Array.isArray(answer.blocks) || !Array.isArray(answer.speed)) throw new Error('The server sent a drive this page cannot read.');
  drive.value = answer;
}
async function read(): Promise<void> {
  try {
    show(await api<SpeedDrive>(address()));
    error.value = '';
  } catch (problem) {
    error.value = messageOf(problem);
  }
}
const blocks = computed(() => drive.value?.blocks ?? []);
const speed = computed(() => drive.value?.speed ?? []);
const maxKph = computed(() => drive.value?.max_kph ?? 200);
/** His dashboard speed at the time shown, and the block there: the answer is the block's when there is one. */
const driverKph = computed(() => speedAt(speed.value, now.value));
const block = computed(() => blockAt(blocks.value, now.value));
const label = computed(() => block.value ? block.value.kph : driverKph.value);

const go = (item: Block): void => {
  // A millisecond in: the time shown is on this block, not at the end of the one before.
  player.value?.seek(secondOf(item.start_ns) + 0.001);
};
// A drive opens where the roadworks were gone through to, the review going on where it stopped, else at its start.
// Opened from Sign review at a moment ("?segment=N&second=S"), it stays there: the player seeks to it.
const linked = new URLSearchParams(location.search).has('segment');
const toStart = watch([t0, drive], () => {
  if (!t0.value || !drive.value) return;
  toStart();
  const checked = drive.value.works_checked_ns;
  if (!linked && checked !== null) player.value?.seek(secondOf(checked));
});

// -- the commands ----------------------------------------------------------------------------------------------------------
const saving = ref(false);
const locked = computed(() => saving.value || props.readOnly);
async function send(path: string, body: object): Promise<boolean> {
  if (locked.value || !drive.value) return false;
  saving.value = true;
  try {
    show(await post<SpeedDrive>(address() + path, {...body, base_revision: drive.value.revision}));
    return true;
  } catch (problem) {
    toast(messageOf(problem), true);
    // A stale revision or anything else: what the server has now.
    void read();
    return false;
  } finally {
    saving.value = false;
  }
}

// A new block: its start stays on the page until its end is given; then it is saved at once. Its ends are in the
// drive, and an end before the start swaps them: the block goes from the earlier time to the later one.
const pending = ref<number | null>(null);
const pendingKph = ref(FALLBACK_KPH);
const pendingOut = ref(false);
const within = (ns: number): number => {
  const span = drive.value?.span_ns;
  return span ? Math.min(span[1], Math.max(span[0], ns)) : ns;
};
const canStart = computed(() => !props.readOnly && !!drive.value?.span_ns);
function startBlock(): void {
  if (!canStart.value) return;
  pending.value = within(now.value);
  pendingKph.value = speedAt(speed.value, pending.value) ?? FALLBACK_KPH;
  pendingOut.value = false;
}
const ends = computed((): [number, number] | null => pending.value === null ? null
  : [within(Math.min(pending.value, now.value)), within(Math.max(pending.value, now.value))]);
const canEnd = computed(() => !!ends.value && ends.value[1] - ends.value[0] >= MIN_BLOCK_NS);
async function endBlock(): Promise<void> {
  if (!ends.value || !canEnd.value) return;
  const [start, end] = ends.value;
  if (!await send('/blocks', {start_ns: start, end_ns: end, kph: pendingOut.value ? null : pendingKph.value, note: ''})) return;
  pending.value = null;
  // A block ends just before its end: a millisecond back, the card shows the block just made, ready to change.
  if (now.value >= end) player.value?.seek(secondOf(end) - 0.001);
}
const cancelBlock = (): void => { pending.value = null; };
function tapBlock(): void {
  if (pending.value === null) startBlock();
  else if (canEnd.value) void endBlock();
  else toast('A block lasts at least 1 s: go on to its end first.');
}

// A saved block is changed by saving it again with its ends and its note; Remove gives his speed back there.
const setBlock = (item: Block, kph: number | null): void => {
  void send('/blocks', {start_ns: item.start_ns, end_ns: item.end_ns, kph, note: item.note});
};
const removeBlock = (item: Block): void => { void send('/blocks', {remove: true, start_ns: item.start_ns}); };
/** A step of the kph of the new block, else of the block at the time shown. */
function step(delta: number): void {
  if (pending.value !== null) {
    if (!pendingOut.value) pendingKph.value = clampKph(pendingKph.value + delta, maxKph.value);
  } else if (block.value && block.value.kph !== null) {
    const kph = clampKph(block.value.kph + delta, maxKph.value);
    if (kph !== block.value.kph) setBlock(block.value, kph);
  }
}
const kphOf = (item: Block): number => speedAt(speed.value, item.start_ns) ?? FALLBACK_KPH;
const blockText = (item: Block): string => item.kph === null ? 'left out' : item.kph + ' km/h';

// -- the roadworks ---------------------------------------------------------------------------------------------------------
const works = computed(() => drive.value?.works ?? []);
const worksOpen = computed(() => works.value.find((item) => item.end_ns === null) ?? null);
const worksHere = computed(() => worksAt(works.value, now.value));
const worksShown = computed(() => worksHere.value ?? worksOpen.value);
const canMarkWorks = computed(() => worksOpen.value ? canEndWorks(works.value, worksOpen.value, now.value) : worksHere.value === null);
const worksChecked = computed(() => drive.value?.works_checked_ns ?? null);
const canCheckWorks = computed(() => !worksOpen.value || now.value < worksOpen.value.start_ns);
const tapWorks = (tap: WorksTap): void => { void send('/works', {tap, at_ns: now.value}); };
function markWorks(): void {
  if (canMarkWorks.value) tapWorks(worksOpen.value ? 'end' : 'start');
}

function onKey(event: KeyboardEvent): void {
  const target = event.target as HTMLElement | null;
  if (target?.closest('input, textarea, select, [contenteditable]') || event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.key === 'b') tapBlock();
  else if (event.key === 'Escape' && pending.value !== null) cancelBlock();
  else if (event.key === 'w') markWorks();
  else if (event.key === '-') step(-5);
  else if (event.key === '+' || event.key === '=') step(5);
  else if ((event.key === 'ArrowDown' || event.key === 'ArrowUp') && (pending.value !== null || typeof block.value?.kph === 'number')) {
    // The arrows step a kph only when there is one to step: else they scroll the page.
    event.preventDefault();
    step(event.key === 'ArrowDown' ? -1 : 1);
  }
}
onMounted(() => {
  window.addEventListener('keydown', onKey);
  void read();
});
onBeforeUnmount(() => window.removeEventListener('keydown', onKey));

// -- the lane --------------------------------------------------------------------------------------------------------------
/** The scale of the lane: tens of km/h up to the fastest speed or block of the drive, the same at every zoom. */
const top = computed(() => Math.ceil(Math.max(50, ...speed.value.map((point) => point[1] ?? 0),
  ...blocks.value.map((item) => item.kph ?? 0)) / 10) * 10);
const y = (kph: number): number => TOP + (1 - kph / top.value) * (PLOT - TOP);
const ticks = computed(() => [0, 50, 100, 150].filter((kph) => kph <= top.value));
const end = computed(() => drive.value?.span_ns ? secondOf(drive.value.span_ns[1]) : 0);
/** His speed in the time window as a step line, broken where the logs have none. The lane is drawn again at every
 * frame of the video, the line only when the window moves. */
let drawn: {key: string; path: string} | null = null;
function speedPath(from: number, to: number, width: number, x: (second: number) => number): string {
  const key = [from, to, width, top.value, t0.value].join('|');
  if (drawn?.key === key) return drawn.path;
  const points = speed.value;
  let out = '';
  let open = false;
  for (let index = Math.max(0, speedIndex(points, t0.value + from * 1e9)); index < points.length; index++) {
    const [ns, kph] = points[index]!;
    const at = secondOf(ns);
    if (at > to) break;
    const px = x(Math.max(at, from)).toFixed(1);
    if (kph === null) {
      if (open) out += 'H' + px;
      open = false;
    } else if (!open) {
      out += 'M' + px + ' ' + y(kph).toFixed(1);
      open = true;
    } else {
      out += 'H' + px + 'V' + y(kph).toFixed(1);
    }
  }
  if (open) out += 'H' + x(Math.min(to, end.value)).toFixed(1);
  drawn = {key, path: out};
  return out;
}
watch(drive, () => { drawn = null; });
/** The blocks inside the time window, with what the lane writes on them. */
function blocksIn(from: number, to: number) {
  return blocks.value.flatMap((item) => {
    const first = Math.max(secondOf(item.start_ns), from);
    const last = Math.min(secondOf(item.end_ns), to);
    return first < last ? [{key: item.start_ns, from: first, to: last, kph: item.kph, text: item.kph === null ? 'left out' : String(item.kph),
      current: item === block.value}] : [];
  });
}
/** The new block still on the page, from its start to the time shown, inside the time window: none or one. */
function pendingIn(from: number, to: number) {
  if (!ends.value || pending.value === null) return [];
  const start = secondOf(pending.value);
  return [{start: start >= from && start <= to ? start : null, from: Math.max(secondOf(ends.value[0]), from),
    to: Math.min(secondOf(ends.value[1]), to)}];
}
/** The part of the time window a person went through for roadworks. */
function checkedIn(from: number, to: number) {
  if (worksChecked.value === null) return [];
  const last = Math.min(secondOf(worksChecked.value), to);
  return from < last ? [{from, to: last}] : [];
}
/** The roadworks inside the time window. */
function worksIn(from: number, to: number) {
  return works.value.flatMap((item) => {
    const first = Math.max(secondOf(item.start_ns), from);
    const last = Math.min(item.end_ns === null ? end.value : secondOf(item.end_ns), to);
    return first < last ? [{key: item.start_ns, from: first, to: last, open: item.end_ns === null}] : [];
  });
}
</script>

<template>
  <DriveReviewDrive ref="player" class="sr-view" :route="route" :read-only="readOnly" tab="speedreview" :back-label="isGuest() ? 'My reviews' : 'Speed review'" simple-lanes
                    sharp :rates="RATES" start-zoom="ten" @time="follow">
    <template #hud>
      <span class="dr-box sr-box-label" :class="{empty: label === null}"><small>{{ block ? (block.kph === null ? 'OUT' : 'BLOCK') : 'LABEL' }}</small>
        <b>{{ label ?? '–' }}</b></span>
    </template>
    <template #lane="{x, w, width, from, to, fits}">
      <div class="dr-lane sr-lane" title="The answer the brain learns: the line is your dashboard speed; a box is a block of yours, its kph instead of your speed, grey left out; orange bar: roadworks; grey line: gone through for roadworks">
        <span class="dr-lane-name">Answer<small>km/h</small></span>
        <svg :width="width" :height="HEIGHT" :viewBox="`0 0 ${width} ${HEIGHT}`" aria-hidden="true">
          <g v-for="kph in ticks" :key="kph">
            <line :x1="0" :x2="width" :y1="y(kph)" :y2="y(kph)" class="rg-grid"/>
            <text x="3" :y="y(kph) - 2" class="rg-tick">{{ kph }}</text>
          </g>
          <g v-for="run in blocksIn(from, to)" :key="run.key">
            <rect :x="x(run.from)" y="1" :width="w(run.from, run.to)" :height="PLOT - 1" rx="3"
                  :class="['sr-block', {'sr-block-out': run.kph === null, 'sr-block-current': run.current}]"/>
            <line v-if="run.kph !== null" :x1="x(run.from)" :x2="x(run.to)" :y1="y(run.kph)" :y2="y(run.kph)" class="sr-block-kph"/>
            <text v-if="fits(run.from, run.to, run.text)" :x="x(run.from) + w(run.from, run.to) / 2" y="12" class="dr-run-text sr-block-text">{{ run.text }}</text>
          </g>
          <path :d="speedPath(from, to, width, x)" class="rg-line sr-speed"/>
          <g v-for="part in pendingIn(from, to)" :key="'new'">
            <rect v-if="part.from < part.to" :x="x(part.from)" y="1" :width="w(part.from, part.to)" :height="PLOT - 1" rx="3" class="sr-block-new"/>
            <line v-if="part.start !== null" :x1="x(part.start)" :x2="x(part.start)" y1="0" :y2="PLOT" class="sr-block-start"/>
          </g>
          <rect v-for="mark in worksIn(from, to)" :key="mark.key" :x="x(mark.from)" :y="PLOT + 2" :width="w(mark.from, mark.to)" height="4"
                :class="['sr-works', {'sr-works-open': mark.open}]"/>
          <rect v-for="part in checkedIn(from, to)" :key="part.from" :x="x(part.from)" :y="HEIGHT - 2" :width="w(part.from, part.to)" height="2"
                class="sr-works-checked"/>
        </svg>
      </div>
    </template>
    <template #side>
      <section class="sr-card" aria-labelledby="sr-card-title">
        <p v-if="error" class="dr-hint">{{ error }}</p>
        <p v-else-if="!drive" class="dr-hint">Loading…</p>
        <p v-else-if="!drive.span_ns" class="dr-hint">No speed to show: the logs of this drive are not read yet.</p>
        <template v-else>
          <h3 id="sr-card-title">The answer at {{ clockOf(seconds) }}</h3>
          <div class="sr-value">
            <span>{{ block ? 'Block' : 'Your speed here' }}</span><b>{{ label ?? '–' }}</b><span v-if="label !== null">km/h</span>
            <span v-if="block" class="sr-tag" :class="{'sr-tag-out': block.kph === null}">{{ block.kph === null ? 'Left out' : 'Your block' }}</span>
          </div>
          <p v-if="block" class="dr-hint">{{ clockOf(secondOf(block.start_ns)) }}–{{ clockOf(secondOf(block.end_ns)) }} ·
            your speed here {{ driverKph ?? '–' }} km/h<template v-if="block.reviewer"> · {{ block.reviewer }}</template></p>
          <p v-else class="dr-hint">Your dashboard speed is what the brain learns here. A block from a time to a time gives another
            speed, or leaves those moments out.</p>

          <div v-if="pending !== null" class="sr-new" role="group" aria-labelledby="sr-new-title">
            <p id="sr-new-title" class="sr-new-title">New block from {{ clockOf(secondOf(pending)) }}<template v-if="canEnd && ends">
              to {{ clockOf(secondOf(pending === ends[0] ? ends[1] : ends[0])) }}</template></p>
            <div class="sr-kph" role="group" aria-label="The speed of the new block">
              <button type="button" class="btn small" :disabled="pendingOut" title="Key −" @click="step(-5)">−5</button>
              <button type="button" class="btn small" :disabled="pendingOut" title="Key ↓" @click="step(-1)">−1</button>
              <output class="sr-kph-value">{{ pendingOut ? 'left out' : pendingKph }}</output>
              <button type="button" class="btn small" :disabled="pendingOut" title="Key ↑" @click="step(1)">+1</button>
              <button type="button" class="btn small" :disabled="pendingOut" title="Key +" @click="step(5)">+5</button>
            </div>
            <div class="sr-tools">
              <button type="button" class="btn small" id="sr-block-out" :aria-pressed="pendingOut"
                      title="These moments stay out of the training" @click="pendingOut = !pendingOut">Leave out</button>
              <button type="button" class="btn small" id="sr-block-cancel" title="Esc" @click="cancelBlock">Cancel</button>
            </div>
            <button type="button" class="btn primary sr-wide" id="sr-block-end" :disabled="locked || !canEnd" title="Key b" @click="endBlock">Block to here</button>
            <p v-if="!canEnd" class="dr-hint">Go to the end of the block: it lasts at least 1 s.</p>
          </div>
          <button v-else type="button" class="btn primary sr-wide" id="sr-block-start" :disabled="locked || !canStart" title="Key b"
                  @click="startBlock">Block from here</button>

          <template v-if="block && pending === null">
            <div class="sr-kph" role="group" aria-label="The speed of this block">
              <button type="button" class="btn small" :disabled="locked || block.kph === null" title="Key −" @click="step(-5)">−5</button>
              <button type="button" class="btn small" :disabled="locked || block.kph === null" title="Key ↓" @click="step(-1)">−1</button>
              <output class="sr-kph-value">{{ block.kph ?? 'left out' }}</output>
              <button type="button" class="btn small" :disabled="locked || block.kph === null" title="Key ↑" @click="step(1)">+1</button>
              <button type="button" class="btn small" :disabled="locked || block.kph === null" title="Key +" @click="step(5)">+5</button>
            </div>
            <div class="sr-tools">
              <button v-if="block.kph !== null" type="button" class="btn small" id="sr-block-leave" :disabled="locked"
                      @click="setBlock(block, null)">Leave out</button>
              <button v-else type="button" class="btn small" id="sr-block-give" :disabled="locked"
                      @click="setBlock(block, kphOf(block))">Give a speed ({{ kphOf(block) }})</button>
              <button type="button" class="btn small" id="sr-block-remove" :disabled="locked" @click="removeBlock(block)">Remove block</button>
            </div>
          </template>

          <div v-if="blocks.length" class="sr-blocks">
            <p class="sr-blocks-title">{{ plural(blocks.length, 'block') }}</p>
            <ul>
              <li v-for="item in blocks" :key="item.start_ns" :class="{current: item === block}">
                <button type="button" class="sr-block-go" @click="go(item)"><span>{{ clockOf(secondOf(item.start_ns)) }}–{{ clockOf(secondOf(item.end_ns)) }}</span>
                  <b>{{ blockText(item) }}</b></button>
                <button type="button" class="btn small sr-block-x" :disabled="locked" :aria-label="'Remove the block from ' + clockOf(secondOf(item.start_ns))"
                        @click="removeBlock(item)">✕</button>
              </li>
            </ul>
          </div>

          <div class="sr-tools">
            <button type="button" class="btn small" id="sr-works" :disabled="locked || !canMarkWorks" title="Key w" @click="markWorks">{{ worksOpen ? 'Roadworks end here' : 'Roadworks start here' }}</button>
            <button v-if="worksHere" type="button" class="btn small" id="sr-works-remove" :disabled="locked" @click="tapWorks('remove')">Remove roadworks</button>
            <button type="button" class="btn small" id="sr-works-checked" :disabled="locked || !canCheckWorks"
                    title="From the start of the drive to this frame every roadworks is marked: the rest is free of them" @click="tapWorks('checked')">Roadworks checked to here</button>
          </div>
          <p v-if="worksShown" class="dr-hint">Roadworks {{ clockOf(secondOf(worksShown.start_ns)) }}{{ worksShown.end_ns === null ? ', no end yet' : '–' + clockOf(secondOf(worksShown.end_ns)) }}</p>
          <p v-if="worksChecked !== null" class="dr-hint">Checked for roadworks up to {{ clockOf(secondOf(worksChecked)) }}</p>
        </template>
      </section>
    </template>
  </DriveReviewDrive>
</template>
