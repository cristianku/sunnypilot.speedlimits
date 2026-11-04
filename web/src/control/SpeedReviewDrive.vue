<script setup lang="ts">
// Data › Speed review, a drive opened ("/speed-review?drive=<route>"): the player of Drive review (DriveReviewDrive.vue)
// with one lane more, the speed to hold, and the commands at its side. The proposal is the limit the comma's vision held,
// and a person says what the speed to hold really was (speed_review.py). Every command saves at once and acts on the
// stretch under the time shown; Yes and Can't tell go on to the next stretch nobody looked at. In a stuck limit the right
// change is inside the stretch: Split here cuts it at the time shown, and saving one half makes the cut real. Roadworks
// are marked next to the speed, two taps on the frame shown, and cut no stretch; a third tap says that the drive was gone
// through up to the frame shown, so that no mark there means no works.
import {computed, onBeforeUnmount, onMounted, ref, shallowRef, useTemplateRef, watch} from 'vue';
import {api, messageOf, post} from '../common/api';
import {toast} from '../composables/toast';
import {clockOf} from './drive-review';
import DriveReviewDrive from './DriveReviewDrive.vue';
import {canCut, canEndWorks, cutOf, type Label, labelOf, nextOpen, openCuts, pieceAt, SPEEDS, type SpeedDrive, STATE_LABEL, STEP_KPH, stepped,
  type Stretch, withCuts, worksAt, type WorksTap} from './speed-review';

const props = defineProps<{route: string; readOnly: boolean}>();
const RATES = [1, 2, 4] as const;

const player = useTemplateRef<InstanceType<typeof DriveReviewDrive>>('player');
const seconds = ref(0);
const t0 = ref(0);
function follow(at: number, start: number): void {
  seconds.value = at;
  t0.value = start;
}
const now = computed(() => Math.round(t0.value + seconds.value * 1e9));
const secondOf = (ns: number): number => (ns - t0.value) / 1e9;

// -- the stretches ---------------------------------------------------------------------------------------------------------
const drive = shallowRef<SpeedDrive | null>(null);
const cuts = ref<number[]>([]);
const error = ref('');
const address = (): string => '/api/speed-review/drives/' + encodeURIComponent(props.route);
function show(answer: SpeedDrive): void {
  // A cut answer (the page left while it came) reads as {message} in api(): never a drive without its stretches.
  if (!answer || !Array.isArray(answer.stretches)) throw new Error('The server sent a drive this page cannot read.');
  drive.value = answer;
  cuts.value = openCuts(cuts.value, answer.stretches);
}
async function read(): Promise<void> {
  try {
    show(await api<SpeedDrive>(address()));
    error.value = '';
  } catch (problem) {
    error.value = messageOf(problem);
  }
}
const pieces = computed(() => withCuts(drive.value?.stretches ?? [], cuts.value));
const index = computed(() => pieceAt(pieces.value, now.value));
const piece = computed((): Stretch | null => pieces.value[index.value] ?? null);
const examined = computed(() => drive.value && drive.value.total_s ? drive.value.examined_s / drive.value.total_s : 0);

const go = (item: Stretch | undefined): void => {
  // A millisecond in: the time shown is on this stretch, not at the end of the one before.
  if (item) player.value?.seek(secondOf(item.start_ns) + 0.001);
};
function goOn(after: Stretch): void {
  const item = nextOpen(pieces.value, after.end_ns);
  if (item) go(item);
  else toast('Every stretch of this drive is examined.');
}
// A drive opens at the first stretch nobody looked at: the review goes on where it stopped.
const toFirstOpen = watch([t0, drive], () => {
  if (!t0.value || !drive.value) return;
  toFirstOpen();
  go(nextOpen(pieces.value, 0) ?? undefined);
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
const write = (item: Stretch, label: Label): Promise<boolean> => send('/stretches', {start_ns: item.start_ns, end_ns: item.end_ns, ...label});
const canYes = computed(() => piece.value?.state === 'not_examined' && piece.value.kph !== null);
async function yes(): Promise<void> {
  const item = piece.value;
  if (item && canYes.value && item.kph !== null && await write(item, labelOf(item, item.kph))) goOn(item);
}
function set(kph: number | null): void {
  if (piece.value && kph !== null) void write(piece.value, labelOf(piece.value, kph));
}
async function cannotTell(): Promise<void> {
  const item = piece.value;
  if (item && await write(item, {kph: null, state: 'undetermined', known_ns: null, note: item.note})) goOn(item);
}
const clear = (): void => { if (piece.value) void write(piece.value, {state: null}); };
const saved = computed(() => piece.value !== null && piece.value.state !== 'not_examined');
const hasValue = computed(() => saved.value && piece.value?.kph !== null);
function knownFrom(ns: number | null): void {
  const item = piece.value;
  if (item && item.kph !== null) void write(item, {...labelOf(item, item.kph), known_ns: ns});
}

const splittable = computed(() => canCut(piece.value, now.value));
function split(): void {
  if (splittable.value) cuts.value = [...cuts.value, now.value];
}
const removable = computed(() => cutOf(piece.value, cuts.value, now.value));
function join(): void {
  cuts.value = cuts.value.filter((cut) => cut !== removable.value);
}

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

// The note of a stretch already saved. The time can move while it is typed (a tap on the lanes ends the typing), so it
// goes to the stretch it was begun on.
const note = ref('');
const noteInput = useTemplateRef<HTMLInputElement>('noteInput');
let noteOf: Stretch | null = null;
function noteStart(): void {
  noteOf = piece.value;
  player.value?.pause();
}
function noteEnd(): void {
  const item = noteOf;
  const text = note.value.trim();
  noteOf = null;
  if (item && item.state !== 'not_examined' && text !== item.note) void write(item, {kph: item.kph, state: item.state, known_ns: item.known_ns, note: text});
}
watch(piece, (item) => {
  if (document.activeElement !== noteInput.value) note.value = item?.note ?? '';
});

function onKey(event: KeyboardEvent): void {
  const target = event.target as HTMLElement | null;
  if (target?.closest('input, textarea, select, [contenteditable]') || event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.key === 'y') void yes();
  else if (event.key === 's') split();
  else if (event.key === 'w') markWorks();
  else if (event.key === '-' && piece.value) set(stepped(piece.value, -STEP_KPH));
  else if ((event.key === '+' || event.key === '=') && piece.value) set(stepped(piece.value, STEP_KPH));
}
onMounted(() => {
  window.addEventListener('keydown', onKey);
  void read();
});
onBeforeUnmount(() => window.removeEventListener('keydown', onKey));

// -- the lane --------------------------------------------------------------------------------------------------------------
/** The stretches inside the window of the lanes, with what the lane writes on them. */
function runs(from: number, to: number) {
  return pieces.value.flatMap((item, at) => {
    const first = Math.max(secondOf(item.start_ns), from);
    const last = Math.min(secondOf(item.end_ns), to);
    return first < last ? [{key: item.start_ns, from: first, to: last, state: item.state, current: at === index.value,
      text: item.state === 'undetermined' ? '?' : item.kph === null ? '–' : String(item.kph),
      known: item.known_ns === null ? null : secondOf(item.known_ns)}] : [];
  });
}
/** The part of the window of the lanes a person went through for roadworks. */
function checkedIn(from: number, to: number) {
  if (worksChecked.value === null) return [];
  const last = Math.min(secondOf(worksChecked.value), to);
  return from < last ? [{from, to: last}] : [];
}
/** The roadworks inside the window of the lanes. */
function worksIn(from: number, to: number) {
  const end = pieces.value[pieces.value.length - 1]?.end_ns ?? 0;
  return works.value.flatMap((item) => {
    const first = Math.max(secondOf(item.start_ns), from);
    const last = Math.min(secondOf(item.end_ns ?? end), to);
    return first < last ? [{key: item.start_ns, from: first, to: last, open: item.end_ns === null}] : [];
  });
}
</script>

<template>
  <DriveReviewDrive ref="player" class="sr-view" :route="route" :read-only="readOnly" tab="speedreview" back-label="Speed review" no-episodes
                    :rates="RATES" start-zoom="ten" @time="follow">
    <template #lane="{x, w, width, from, to, fits}">
      <div class="dr-lane sr-lane" title="The speed to hold: dashed, what the comma's vision held, nobody looked at it yet; green confirmed; blue corrected; ? nobody can tell; orange bar: roadworks; grey line: gone through for roadworks">
        <span class="dr-lane-name">To hold<small>review</small></span>
        <svg :width="width" height="24" :viewBox="`0 0 ${width} 24`" aria-hidden="true">
          <g v-for="run in runs(from, to)" :key="run.key">
            <rect :x="x(run.from)" y="1" :width="w(run.from, run.to)" height="21" rx="3" :class="['sr-' + run.state, {'sr-current': run.current}]"/>
            <text v-if="fits(run.from, run.to, run.text)" :x="x(run.from) + w(run.from, run.to) / 2" y="15.5" class="dr-run-text" :class="'sr-text-' + run.state">{{ run.text }}</text>
            <line v-if="run.known !== null" :x1="x(run.known)" :x2="x(run.known)" y1="1" y2="22" class="sr-known"/>
          </g>
          <rect v-for="part in checkedIn(from, to)" :key="part.from" :x="x(part.from)" y="22" :width="w(part.from, part.to)" height="2" class="sr-works-checked"/>
          <rect v-for="mark in worksIn(from, to)" :key="mark.key" :x="x(mark.from)" y="18" :width="w(mark.from, mark.to)" height="4" :class="['sr-works', {'sr-works-open': mark.open}]"/>
        </svg>
      </div>
    </template>
    <template #side>
      <section class="sr-card" aria-labelledby="sr-card-title">
        <p v-if="error" class="dr-hint">{{ error }}</p>
        <p v-else-if="!drive" class="dr-hint">Loading…</p>
        <p v-else-if="!pieces.length" class="dr-hint">No limit to propose: the logs of this drive are not read yet.</p>
        <template v-else-if="piece">
          <div class="dr-ep-nav">
            <button type="button" class="btn small" :disabled="index <= 0" aria-label="Previous stretch" @click="go(pieces[index - 1])">‹</button>
            <h3 id="sr-card-title">Stretch {{ index + 1 }} of {{ pieces.length }}</h3>
            <button type="button" class="btn small" :disabled="index >= pieces.length - 1" aria-label="Next stretch" @click="go(pieces[index + 1])">›</button>
          </div>
          <div class="sr-bar" aria-hidden="true"><i :style="{width: examined * 100 + '%'}"></i></div>
          <p class="dr-hint">{{ clockOf(drive.examined_s) }} of {{ clockOf(drive.total_s) }} examined</p>

          <div class="sr-value">
            <b>{{ piece.state === 'undetermined' ? '?' : piece.kph ?? '–' }}</b><span v-if="piece.kph !== null">km/h</span>
            <span class="sr-tag" :class="'sr-tag-' + piece.state">{{ STATE_LABEL[piece.state] }}</span>
          </div>
          <p class="dr-hint">{{ clockOf(secondOf(piece.start_ns)) }}–{{ clockOf(secondOf(piece.end_ns)) }}
            <template v-if="piece.state === 'not_examined'"> · {{ piece.kph === null ? 'the comma\'s vision held none' : 'held by the comma\'s vision' }}</template>
            <template v-else-if="piece.reviewer"> · {{ piece.reviewer }}</template></p>
          <p class="dr-hint">The cap on a free road: queues, curves and stops stay with the planner.</p>

          <button type="button" class="btn primary sr-yes" id="sr-yes" :disabled="locked || !canYes" title="Key y" @click="yes">{{ piece.kph === null
            ? 'Pick a speed' : piece.state === 'not_examined' ? 'Yes, ' + piece.kph : 'Examined' }}</button>
          <div class="sr-steps">
            <button type="button" class="btn small" :disabled="locked || stepped(piece, -STEP_KPH) === null" title="Key −" @click="set(stepped(piece, -STEP_KPH))">−{{ STEP_KPH }}</button>
            <button type="button" class="btn small" :disabled="locked || stepped(piece, STEP_KPH) === null" title="Key +" @click="set(stepped(piece, STEP_KPH))">+{{ STEP_KPH }}</button>
          </div>
          <div class="sr-speeds" role="group" aria-label="The speed to hold">
            <button v-for="kph in SPEEDS" :key="kph" type="button" class="btn small" :aria-pressed="saved && piece.kph === kph" :disabled="locked" @click="set(kph)">{{ kph }}</button>
            <button type="button" class="btn small" :aria-pressed="piece.state === 'undetermined'" :disabled="locked" title="You looked and nobody can tell the speed here" id="sr-unknown" @click="cannotTell">?</button>
          </div>
          <div class="sr-tools">
            <button type="button" class="btn small" id="sr-split" :disabled="readOnly || !splittable" title="Key s" @click="split">Split here</button>
            <button v-if="removable !== null" type="button" class="btn small" id="sr-join" @click="join">Remove split</button>
            <button v-if="hasValue && piece.known_ns === null" type="button" class="btn small" id="sr-known" :disabled="locked"
                    title="From this moment the past is enough to work the speed out" @click="knownFrom(now)">Known from here</button>
            <button v-if="piece.known_ns !== null" type="button" class="btn small" :disabled="locked" @click="knownFrom(null)">Known from {{ clockOf(secondOf(piece.known_ns)) }} ✕</button>
            <button v-if="saved" type="button" class="btn small" id="sr-clear" :disabled="locked" @click="clear">Clear</button>
          </div>
          <div class="sr-tools">
            <button type="button" class="btn small" id="sr-works" :disabled="locked || !canMarkWorks" title="Key w" @click="markWorks">{{ worksOpen ? 'Roadworks end here' : 'Roadworks start here' }}</button>
            <button v-if="worksHere" type="button" class="btn small" id="sr-works-remove" :disabled="locked" @click="tapWorks('remove')">Remove roadworks</button>
            <button type="button" class="btn small" id="sr-works-checked" :disabled="locked || !canCheckWorks"
                    title="From the start of the drive to this frame every roadworks is marked: the rest is free of them" @click="tapWorks('checked')">Roadworks checked to here</button>
          </div>
          <p v-if="worksShown" class="dr-hint">Roadworks {{ clockOf(secondOf(worksShown.start_ns)) }}{{ worksShown.end_ns === null ? ', no end yet' : '–' + clockOf(secondOf(worksShown.end_ns)) }}</p>
          <p v-if="worksChecked !== null" class="dr-hint">Checked for roadworks up to {{ clockOf(secondOf(worksChecked)) }}</p>
          <label v-if="saved" class="sr-note"><span class="sr-only">A note on this stretch</span>
            <input ref="noteInput" v-model="note" type="text" maxlength="500" placeholder="A note on this stretch" :disabled="readOnly" @focus="noteStart" @change="noteEnd"/></label>
        </template>
      </section>
    </template>
  </DriveReviewDrive>
</template>
