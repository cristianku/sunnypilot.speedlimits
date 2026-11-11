<script setup lang="ts">
// Models › Drive, a drive opened ("/drive?drive=<route>"): the drive player (DriveReviewDrive.vue) with the comma's
// screen on its picture (CommaScreen.vue), the two squares as the car would draw them along the drive with a set of
// networks: the speed to hold, and under it the sign read with its confidence, the metres to it, a red border while it
// is in view and a pulse when it is read. Play shows what the car would have shown; a drive where all of it is right is
// certified, and a replay with another set is then compared with it frame by frame (the regression test).
// The source, at the top of the side, says whose squares they are: what the comma logged that day (the series of the
// player's own timeline, read here once more), a replay with the networks on the car or with those in use (GET
// /api/simulation/drives/<route>?set=, nnslr_tools.simulation.view: the same series by the same names), or the
// certified replay. Over the picture also SignNet's boxes of the frame read (the switch Boxes), in the frame's pixels
// as Speed review's, on the very frame of the video it was read on (its minute and frame k), and on the video's frames
// between two frames read on their way to the same sign's box of the next one (boxesAt). Three lanes over the player's:
// the speed to hold as runs, every sign read as a tick with what was
// read, and on a certified drive where the replay shown differs from it (red the speed to hold, orange the reading; a
// tap goes there). At the side: the values at the time shown; Compute, which replays the drive with the set shown on
// the GPU (the panel action "simulate", helpers/simulate_drive.sh) where it has no replay or one of other networks
// than today's; Certify this drive; and on a certified drive the answer of the regression test with the way to each
// difference. The squares follow the time shown alone, the pulse too: a paused video is a still screen. Nothing here
// changes the car, the labels or the networks in use.
import {computed, onBeforeUnmount, onMounted, ref, shallowRef, useTemplateRef, watch} from 'vue';
import {api, messageOf, post} from '../common/api';
import {isoWhen, sentence} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Job, PanelStatus} from '../common/panel-types';
import {askConfirm} from '../composables/confirm';
import {toast} from '../composables/toast';
import CommaScreen from './CommaScreen.vue';
import {clockOf, readText, runsOf, type ScanState, type Series, type Timeline, valueAt} from './drive-review';
import DriveReviewDrive from './DriveReviewDrive.vue';
import JobProgress from './JobProgress.vue';
import {jobOf, terminalActivities} from './jobs';
import Segmented from './Segmented.vue';
import {ROAD_TYPE_LABELS, roadTypeLabel} from './speed-review';
import {boxesAt, chosenSet, type Differences, FRAME, identical, inputTexts, isSet, PERIOD, pyRound, readingWords, readTimes,
  regressionText, screenAt, screenPlace, SET_LABEL, type SetKey, type SetStates, showBoxes, type SignBox, type SimulationDrive,
  type Source, type Summary, type VideoFrame} from './simulation';

const props = defineProps<{route: string; status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions]; stop: [job: Job];
  details: [job: Job]}>();
const RATES = [1, 2, 4] as const;

const player = useTemplateRef<InstanceType<typeof DriveReviewDrive>>('player');
const seconds = ref(0);
const follow = (at: number): void => { seconds.value = at; };

// -- what the drive has -----------------------------------------------------------------------------------------------------
const address = (): string => '/api/simulation/drives/' + encodeURIComponent(props.route);
/** The comma's own log of the drive, as the player reads it for itself: its series are the source "Comma that day". */
const timeline = shallowRef<Timeline | null>(null);
async function readTimeline(): Promise<void> {
  try {
    const answer = await api<Timeline>('/api/drive-review/drives/' + encodeURIComponent(props.route) + '/timeline');
    timeline.value = answer && typeof answer.series === 'object' && answer.series !== null ? answer : null;
  } catch {
    // The player says what is wrong with the drive's logs; here there is only no "Comma that day".
    timeline.value = null;
  }
}
/** A replay asked of the server: with a set, or the certified one. */
type Asked = Exclude<Source, 'comma'>;
/** The server's answers by what was asked, and from the last one what the drive has and the sets as they are now. */
const answers = shallowRef<Partial<Record<Asked, SimulationDrive>>>({});
const summary = shallowRef<Summary | null>(null);
const sets = shallowRef<SetStates>({});
const error = ref('');
/** Counted up at every reload: an answer asked before it is of the drive as it was. */
let generation = 0;
async function read(key: Asked): Promise<void> {
  const mine = generation;
  try {
    const answer = await api<SimulationDrive>(address() + '?set=' + key);
    if (mine !== generation) return;
    // A cut answer (the page left while it came) reads as {message} in api(): never a drive without its summary.
    if (!answer || typeof answer.summary !== 'object' || answer.summary === null) throw new Error('The server sent a drive this page cannot read.');
    answers.value = {...answers.value, [key]: answer};
    summary.value = answer.summary;
    sets.value = answer.sets ?? {};
    error.value = '';
  } catch (problem) {
    if (mine === generation) error.value = messageOf(problem);
  }
}

// -- whose squares ----------------------------------------------------------------------------------------------------------
const source = ref<Source>(chosenSet.value);
const SOURCE_LABEL: Record<Source, string> = {comma: 'Comma that day', car: SET_LABEL.car, 'in-use': SET_LABEL['in-use'],
  certified: 'Certified'};
/** Under the name of the lanes. */
const SOURCE_SHORT: Record<Source, string> = {comma: 'comma', car: 'on the car', 'in-use': 'in use', certified: 'certified'};
/** The comma logged its vision limit on that drive. */
const logged = computed(() => (timeline.value?.series['vis_limit'] ?? []).some((point) => typeof point[1] === 'number'));
const certified = computed(() => summary.value?.certified ?? null);
const sources = computed((): {key: Source; label: string}[] => (['comma', 'car', 'in-use', 'certified'] as const)
  .filter((key) => key === 'comma' ? logged.value : key === 'certified' ? certified.value !== null : true)
  .map((key) => ({key, label: SOURCE_LABEL[key]})));
/** The drive again after something changed it (a replay ended, it was certified or removed, its logs were read): the
 * answer shown stays until the new one comes, the others are asked again when they are shown. */
function reload(): void {
  generation += 1;
  const key: Asked = source.value === 'comma' ? chosenSet.value : source.value;
  const kept = answers.value[key];
  answers.value = kept ? {[key]: kept} : {};
  void read(key);
}
watch(source, (key) => {
  if (key === 'comma') return;
  if (answers.value[key]) error.value = '';
  else void read(key);
});
onMounted(() => {
  void readTimeline();
  reload();
});
/** The player's search of the logs: once it ends the drive has the clock of its logs, and everything is read again. */
let searching = false;
function onScan(state: ScanState): void {
  const busy = state.state === 'queued' || state.state === 'scanning';
  if (searching && !busy) {
    void readTimeline();
    reload();
  }
  searching = busy;
}

const answer = computed(() => source.value === 'comma' ? null : answers.value[source.value] ?? null);
const view = computed(() => answer.value?.view ?? null);
/** The set shown, null on the comma's own log and on the certified replay: what Compute and Certify act on. */
const set = computed((): SetKey | null => isSet(source.value) ? source.value : null);
const setState = computed(() => set.value ? sets.value[set.value] ?? null : null);
/** The drive's replay with the set shown, as the server has it. */
const result = computed(() => set.value ? summary.value?.sets[set.value] ?? null : null);
const shownSeries = computed((): Record<string, Series> | null => source.value !== 'comma' ? view.value?.series ?? null
  : logged.value ? timeline.value?.series ?? null : null);
const reads = computed(() => readTimes(source.value === 'comma' ? timeline.value?.events ?? [] : view.value?.events ?? []));
/** Whose replay it is and when it was made, in a line. */
const made = computed((): string => {
  if (source.value === 'comma') return logged.value ? 'What the comma logged on that drive.' : '';
  const shown = view.value;
  if (!shown) return '';
  const when = isoWhen(shown.made_at);
  return [shown.name || shown.label, when ? 'made ' + when : '',
    result.value && !result.value.current ? 'with other networks, car code or settings than today’s' : ''].filter(Boolean).join(' · ');
});

// -- on the picture ---------------------------------------------------------------------------------------------------------
const screen = computed(() => shownSeries.value ? screenAt(shownSeries.value, reads.value, seconds.value) : null);
const frame = computed(() => view.value?.frame ?? FRAME);
/** The stage's size, followed as it changes: the comma's screen goes on the picture inside it, not on its bars. */
const overlay = useTemplateRef<HTMLDivElement>('overlay');
const stage = ref({width: 0, height: 0});
let observer: ResizeObserver | null = null;
watch(overlay, (element) => {
  observer?.disconnect();
  observer = null;
  if (!element) return;
  const measure = (): void => { stage.value = {width: element.clientWidth, height: element.clientHeight}; };
  measure();
  observer = new ResizeObserver(measure);
  observer.observe(element);
}, {immediate: true});
onBeforeUnmount(() => observer?.disconnect());
const place = computed(() => screenPlace(stage.value, frame.value));
/** The frame of the video the player presents (its minute and frame k): the boxes go on it. */
const presented = shallowRef<VideoFrame | null>(null);
const boxes = computed(() => showBoxes.value && view.value ? boxesAt(view.value.signs, presented.value) : []);
const tagOf = (box: SignBox): string => (readText(box[4], undefined) || 'sign') + ' ' + Math.round(box[5] * 100) + '%';

// -- at the time shown ------------------------------------------------------------------------------------------------------
const now = computed(() => {
  const all = shownSeries.value;
  const shown = screen.value;
  if (!all || !shown) return null;
  const at = (name: string) => valueAt(all[name], seconds.value);
  const ahead = at('vis_ahead');
  const aheadMetres = at('vis_ahead_dist');
  const aheadKph = typeof ahead === 'number' ? Math.round(ahead) : 0;
  const read = readingWords(shown.rawClass, shown.rawValue);
  const taken = at('vis_raw_taken');
  const metres = pyRound(shown.distance);
  return {
    hold: shown.limitKph ? shown.limitKph + ' km/h' : 'none',
    // Only a replay says the limit read ahead of its sign; the comma's log of that day does not keep it.
    ahead: all['vis_ahead'] === undefined ? null : aheadKph > 0 ? aheadKph + ' km/h'
      + (typeof aheadMetres === 'number' && aheadMetres > 0 ? ' in ' + Math.round(aheadMetres) + ' m' : '') : 'none',
    // The confidence and the metres as the reading square rounds them (Python's round, a half to the even one).
    read: !read ? 'none yet' : read + ' · ' + pyRound(shown.confidence * 100) + '%'
      + (at('vis_raw_road') === 'other_road' ? ' · other road' : taken === true ? ' · taken' : taken === false ? ' · not taken' : ''),
    sign: metres > 0 ? 'in ' + metres + ' m' + (shown.seen ? ', in view' : '') : shown.seen ? 'in view' : 'none',
    road: shown.road,
    // A replay with a brain that learnt the road type: the type its type_head names (not on the car's screen).
    brainType: all['brain_type'] === undefined ? null : brainTypeText(at('brain_type'), at('brain_type_p')),
  };
});
function brainTypeText(type: unknown, chance: unknown): string {
  if (typeof type !== 'string' || !type) return 'none yet';
  return roadTypeLabel(type) + (typeof chance === 'number' ? ' · ' + Math.round(chance * 100) + '%' : '');
}

// -- compute ----------------------------------------------------------------------------------------------------------------
const readOnly = computed(() => props.status?.actions_enabled === false);
const state = computed((): ActionState | null => props.status?.actions?.simulate ?? null);
const job = computed(() => jobOf(props.status, ['simulate']));
const running = computed(() => job.value?.state === 'running' ? job.value : null);
/** A replay at work, the site's or one started from a terminal (helpers/simulate_drive.sh): when it ends, what it computed. */
const working = computed(() => running.value !== null
  || terminalActivities(props.status, ['train']).some((activity) => activity.label === 'drive replay'));
watch(working, (on, was) => { if (was && !on) reload(); });
/** Why Compute cannot start now, in the server's words; '' when it can. */
const blocked = computed((): string => {
  if (props.status && !state.value) return 'This server cannot replay a drive yet.';
  if (state.value && !state.value.allowed) return sentence(state.value.reason);
  // The release of the car's networks is the clone's: there is nothing Compute could make ready.
  return set.value === 'car' && setState.value && !setState.value.ready ? sentence(setState.value.reason) : '';
});
function compute(): void {
  if (state.value && set.value) emit('run', 'simulate', state.value, undefined, {route: props.route, set: set.value});
}

// -- certify, and the regression test ---------------------------------------------------------------------------------------
const saving = ref(false);
/** Certify pins the replay looked at (its made_at): the server refuses (409, its words in the toast) when the replay on
 * its disk is another one by then. */
async function mark(body: {certified: boolean; set?: SetKey; made_at?: string | null}): Promise<void> {
  saving.value = true;
  try {
    summary.value = (await post<{summary: Summary}>(address() + '/certify', body)).summary;
    if (source.value === 'certified' && !summary.value?.certified) source.value = chosenSet.value;
    // The certified replay, and each set against it, are others now.
    reload();
  } catch (problem) {
    toast(messageOf(problem), true);
  } finally {
    saving.value = false;
  }
}
async function certify(): Promise<void> {
  const shown = view.value;
  if (!set.value || !shown || working.value) return;
  const yes = await askConfirm({title: 'Certify this drive?', action: 'Certify',
    text: 'What the simulation shows along it becomes the reference of the regression test.'});
  if (yes) await mark({certified: true, set: set.value, made_at: shown.made_at});
}
async function remove(): Promise<void> {
  const yes = await askConfirm({title: 'Remove this drive from the certified ones?', action: 'Remove', danger: true,
    text: 'Its reference of the regression test is deleted. The replays of the drive stay: certifying it again makes a new one.'});
  if (yes) await mark({certified: false});
}
const certifiedText = computed((): string => {
  const entry = certified.value;
  if (!entry) return '';
  const label = entry.set === 'car' || entry.set === 'in-use' ? SET_LABEL[entry.set] : entry.set;
  return 'Certified ' + isoWhen(entry.at) + (entry.who ? ' by ' + entry.who : '') + ' with ' + label + (entry.name ? ', ' + entry.name : '') + '.';
});
/** The replay shown next to the certified one: only a set's replay of a certified drive has it. */
const found = computed((): Differences | null => set.value && certified.value ? answer.value?.differences ?? null : null);
/** Where a difference starts, the speed to hold's and the reading's together. */
const starts = computed(() => [...new Set([...(found.value?.limit ?? []), ...(found.value?.reading ?? [])].map((run) => run[0]))]
  .sort((a, b) => a - b));
const before = computed(() => starts.value.filter((at) => at < seconds.value - 0.05).pop() ?? null);
const after = computed(() => starts.value.find((at) => at > seconds.value + 0.05) ?? null);
const go = (at: number | null): void => { if (at !== null) player.value?.seek(at); };
/** A reading of simulation.reading_text in words: the sign in view by its kind, for which road, and the pulse. */
function readingName(text: string): string {
  const pulse = text === 'read' || text.endsWith(' read');
  let sign = (pulse ? text.slice(0, -4) : text).trim();
  const other = sign.endsWith(' other road');
  if (other) sign = sign.slice(0, -11);
  return (sign ? '“' + sign.replaceAll('_', ' ') + '” in view' + (other ? ' for another road' : '') : 'no sign in view') +
    (pulse ? ', pulsing' : '');
}
/** What differs at the time shown, said with both values. */
const here = computed((): string => {
  const at = seconds.value;
  const limit = found.value?.limit.find((run) => run[0] <= at && at < run[1] + PERIOD);
  const reading = found.value?.reading.find((run) => run[0] <= at && at < run[1] + PERIOD);
  return [limit ? 'Here the certified drive held ' + (limit[2] || 'no speed') + ', this one ' + (limit[3] || 'no speed') + '.' : '',
    reading ? 'Here the certified drive read ' + readingName(reading[2]) + ', this one ' + readingName(reading[3]) + '.' : '']
    .filter(Boolean).join(' ');
});

// -- the lanes --------------------------------------------------------------------------------------------------------------
interface Run {from: number; to: number; cls: string; text: string}
interface Tick {at: number; left: number; text: string; other: boolean; title: string}
interface Differ {from: number; to: number; start: number; title: string}
interface Lanes {
  from: number; to: number; width: number; series: Record<string, Series> | null; reads: number[]; found: Differences | null;
  hold: Run[]; types: Run[]; ticks: Tick[]; limit: Differ[]; reading: Differ[];
}
/** The lanes of a time window, kept while the window and what is shown stay the same: the player draws its lanes again
 * for other reasons, these only when they change. */
let drawn: Lanes | null = null;
function lanesIn(from: number, to: number, width: number, x: (seconds: number) => number): Lanes {
  const all = shownSeries.value;
  const times = reads.value;
  const differs = found.value;
  if (drawn && drawn.from === from && drawn.to === to && drawn.width === width && drawn.series === all && drawn.reads === times
    && drawn.found === differs) return drawn;
  // The speed to hold as the screen rounds it: a number, none held (0), or no frame read there (a hole of the replay).
  const hold = runsOf(all?.['vis_limit'], from, to, (value) => value === null ? -1 : typeof value === 'number' ? Math.max(0, Math.round(value)) : 0)
    .map((run): Run => ({from: run.from, to: run.to, cls: run.value < 0 ? 'dr-gap' : run.value ? 'dv-hold' : 'dr-none',
      text: run.value < 0 ? '' : run.value ? String(run.value) : '–'}));
  // The road type the brain names, in Speed review's colours; hatched where no frame was read.
  const types = runsOf(all?.['brain_type'], from, to, (value) => typeof value === 'string' ? value : '')
    .map((run): Run => ({from: run.from, to: run.to, cls: run.value ? 'sr-type sr-type-net rt-' + run.value : 'dr-gap',
      text: run.value ? ROAD_TYPE_LABELS[run.value]?.label.toLowerCase() ?? run.value : ''}));
  // What was read over its tick when there is room for it, kept whole at the ends of the lane.
  let lastX = -Infinity;
  const ticks = all === null ? [] : times.filter((at) => at >= from && at <= to).map((at): Tick => {
    const shown = screenAt(all, [], at);
    const text = readingWords(shown.rawClass, shown.rawValue) || '?';
    const other = valueAt(all['vis_raw_road'], at) === 'other_road';
    const room = x(at) - lastX >= text.length * 6 + 6;
    if (room) lastX = x(at);
    const half = text.length * 3 + 1;
    return {at, left: Math.min(Math.max(x(at), half), Math.max(half, width - half)), text: room ? text : '', other,
      title: 'Read ' + text + ' · ' + Math.round(shown.confidence * 100) + '%'
      + (other ? ' · other road' : '') + (shown.limitKph ? ' · holding ' + shown.limitKph : '')};
  });
  // A run ends on its last frame that differs, which lasts until the next one read.
  const inside = <T extends [number, number, unknown, unknown]>(runs: readonly T[], title: (run: T) => string): Differ[] => runs
    .filter((run) => run[1] + PERIOD >= from && run[0] <= to)
    .map((run) => ({from: Math.max(from, run[0]), to: Math.min(to, run[1] + PERIOD), start: run[0], title: title(run)}));
  drawn = {from, to, width, series: all, reads: times, found: differs, hold, types, ticks,
    limit: inside(differs?.limit ?? [], (run) => 'The speed to hold: certified ' + (run[2] || 'none') + ', now ' + (run[3] || 'none')),
    reading: inside(differs?.reading ?? [], (run) => 'The reading: certified ' + readingName(run[2]) + ', now ' + readingName(run[3]))};
  return drawn;
}
</script>

<template>
  <DriveReviewDrive ref="player" :route="route" :read-only="readOnly" tab="drive" back-label="Drive" :rates="RATES"
                    start-zoom="ten" :limits="false" full-screen @time="follow" @scan="onScan" @frame="presented = $event">
    <template #badge><b v-if="certified" class="dv-badge">Certified</b></template>
    <template #picture="{shown}">
      <div ref="overlay" class="dv-overlay" aria-hidden="true">
        <svg v-if="shown && boxes.length" class="dv-boxes" :viewBox="`0 0 ${frame[0]} ${frame[1]}`" preserveAspectRatio="xMidYMid meet">
          <g v-for="(box, index) in boxes" :key="index" class="dv-box" :class="{other: box[6] === 'other_road'}">
            <rect :x="box[0]" :y="box[1]" :width="Math.max(1, box[2] - box[0])" :height="Math.max(1, box[3] - box[1])"/>
            <text :x="box[0]" :y="Math.max(frame[1] / 30, box[1] - frame[1] / 120)" :font-size="frame[1] / 28">{{ tagOf(box) }}</text>
          </g>
        </svg>
        <CommaScreen v-if="screen && stage.width" v-bind="screen" :scale="place.scale"
                     :style="{right: place.right + 'px', top: place.top + 'px'}"/>
      </div>
    </template>
    <template #lane="{x, w, width, from, to, fits}">
      <div v-if="shownSeries" class="dr-lane" title="The speed to hold along the drive, as the vision square shows it; – none held; hatched: no frame read there">
        <span class="dr-lane-name">Hold<small>{{ SOURCE_SHORT[source] }}</small></span>
        <svg :width="width" height="24" :viewBox="`0 0 ${width} 24`" aria-hidden="true">
          <g v-for="(run, index) in lanesIn(from, to, width, x).hold" :key="index">
            <rect :x="x(run.from)" y="4" :width="w(run.from, run.to)" height="17" rx="2" :class="run.cls"/>
            <text v-if="run.text && fits(run.from, run.to, run.text)" :x="x(run.from) + w(run.from, run.to) / 2" y="16.5" class="dr-run-text">{{ run.text }}</text>
          </g>
        </svg>
      </div>
      <div v-if="shownSeries?.['brain_type']" class="dr-lane" id="dv-brain-type"
           title="The road type the brain names at each of its steps (its training head, applied to the car brain’s memory: the car does not show it); hatched: no frame read there">
        <span class="dr-lane-name">Road type<small>brain</small></span>
        <svg :width="width" height="18" :viewBox="`0 0 ${width} 18`" aria-hidden="true">
          <g v-for="(run, index) in lanesIn(from, to, width, x).types" :key="index">
            <rect :x="x(run.from)" y="2" :width="w(run.from, run.to)" height="14" rx="2" :class="run.cls"/>
            <text v-if="run.text && fits(run.from, run.to, run.text)" :x="x(run.from) + w(run.from, run.to) / 2" y="12.5"
                  class="dr-run-text sr-type-text">{{ run.text }}</text>
          </g>
        </svg>
      </div>
      <div v-if="shownSeries" class="dr-lane" title="Every sign read, where the reading square pulses, with what was read; dashed: a sign of another road">
        <span class="dr-lane-name">Signs<small>read</small></span>
        <svg :width="width" height="18" :viewBox="`0 0 ${width} 18`" aria-hidden="true">
          <g v-for="tick in lanesIn(from, to, width, x).ticks" :key="tick.at" :class="['dr-read', {dropped: tick.other}]">
            <line :x1="x(tick.at)" :x2="x(tick.at)" y1="11" y2="17"/>
            <text v-if="tick.text" :x="tick.left" y="9">{{ tick.text }}</text>
            <title>{{ tick.title }}</title>
          </g>
        </svg>
      </div>
      <div v-if="found" class="dr-lane" title="Where this replay differs from the certified drive: red the speed to hold, orange the reading; a tap goes to where it starts">
        <span class="dr-lane-name">Differs<small>certified</small></span>
        <svg :width="width" height="20" :viewBox="`0 0 ${width} 20`" aria-hidden="true">
          <g v-for="run in lanesIn(from, to, width, x).limit" :key="'limit' + run.start" class="dv-differ" @pointerdown.stop="go(run.start)">
            <rect class="dv-hit" :x="x(run.from) - 5" y="0" :width="Math.max(3, w(run.from, run.to)) + 10" height="10"/>
            <rect class="dv-differ-limit" :x="x(run.from)" y="1" :width="Math.max(3, w(run.from, run.to))" height="8" rx="2"/>
            <title>{{ run.title }}</title>
          </g>
          <g v-for="run in lanesIn(from, to, width, x).reading" :key="'reading' + run.start" class="dv-differ" @pointerdown.stop="go(run.start)">
            <rect class="dv-hit" :x="x(run.from) - 5" y="10" :width="Math.max(3, w(run.from, run.to)) + 10" height="10"/>
            <rect class="dv-differ-reading" :x="x(run.from)" y="11" :width="Math.max(3, w(run.from, run.to))" height="8" rx="2"/>
            <title>{{ run.title }}</title>
          </g>
        </svg>
      </div>
    </template>
    <template #side>
      <section class="sr-card dv-card" aria-labelledby="dv-card-title">
        <Segmented v-if="sources.length > 1" v-model="source" :options="sources" label="Whose squares are on the picture" class="dv-source"/>
        <h3 id="dv-card-title">{{ SOURCE_LABEL[source] }} at {{ clockOf(seconds) }}</h3>
        <p v-if="source !== 'comma' && error" class="dr-hint" id="dv-error">{{ error }}</p>
        <p v-else-if="source !== 'comma' && !answer" class="dr-hint">Loading…</p>
        <template v-else>
          <p v-if="made" class="dr-hint" id="dv-made">{{ made }}</p>
          <dl v-if="now" class="dv-now">
            <dt>Speed to hold</dt><dd>{{ now.hold }}</dd>
            <template v-if="now.ahead !== null"><dt>Limit ahead</dt><dd>{{ now.ahead }}</dd></template>
            <dt>Sign read</dt><dd>{{ now.read }}</dd>
            <dt>Sign ahead</dt><dd>{{ now.sign }}</dd>
            <template v-if="now.road"><dt>Road</dt><dd>{{ now.road }}</dd></template>
            <template v-if="now.brainType !== null"><dt>Road type (brain)</dt><dd id="dv-brain-type-now">{{ now.brainType }}</dd></template>
          </dl>
          <p v-else-if="set" class="dr-hint" id="dv-none">This drive is not replayed with these networks yet.</p>
          <div v-if="view" class="sr-tools">
            <button type="button" class="bc-switch" id="dv-boxes" :aria-pressed="showBoxes"
                    title="SignNet’s boxes of the frame read, on the picture: grey and dashed a sign of another road. Drawn on the frame the car read. The car reads one frame in 4 or 5: in between, a box follows its sign to the next frame read; one whose sign the next frame read lost shows on its own frame only"
                    @click="showBoxes = !showBoxes">Boxes</button>
          </div>
        </template>

        <div v-if="running" class="dv-job dv-job-side">
          <JobProgress :job="running" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
        </div>
        <template v-else-if="set && summary && !readOnly && (!result || !result.current)">
          <button type="button" class="btn sr-wide" :class="{primary: !result}" id="dv-compute" :disabled="!state?.allowed || blocked !== ''"
                  title="Replays this drive through the car’s own vision loop with these networks, on the GPU" @click="compute">Compute</button>
          <p v-if="blocked" class="dr-hint" id="dv-blocked">{{ blocked }}</p>
          <p v-else-if="setState && !setState.ready && setState.reason" class="dr-hint">{{ sentence(setState.reason) }}</p>
          <p v-if="job?.state === 'failed'" class="dr-hint">The last replay failed.
            <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>
        </template>

        <div v-if="certified && set && answer" class="dv-regression" id="dv-regression">
          <template v-if="found">
            <p class="dv-verdict" :class="{same: identical(found)}">{{ regressionText(found) }}</p>
            <div v-if="starts.length" class="sr-tools">
              <button type="button" class="btn small" id="dv-previous" :disabled="before === null" @click="go(before)">‹ Previous</button>
              <button type="button" class="btn small" id="dv-next" :disabled="after === null" @click="go(after)">Next difference ›</button>
            </div>
            <p v-if="here" class="dr-hint">{{ here }}</p>
            <p v-for="text in inputTexts(found)" :key="text" class="dr-hint">{{ text }}</p>
          </template>
          <p v-else-if="!view" class="dr-hint">Compute compares these networks with the certified drive.</p>
        </div>

        <template v-if="summary">
          <div v-if="certified" class="dv-certified">
            <p class="dr-hint" id="dv-certified">{{ certifiedText }}</p>
            <button v-if="!readOnly" type="button" class="btn small" id="dv-remove" :disabled="saving" @click="remove">Remove</button>
          </div>
          <!-- Not while a replay runs: the page cannot tell which drive it replays, and it may be this one. -->
          <button v-else-if="set && view && !readOnly" type="button" class="btn sr-wide" id="dv-certify" :disabled="saving || working"
                  :title="working ? 'A drive is being replayed: certify once it has ended' : 'This replay becomes the drive’s reference: the next sets are compared with it'"
                  @click="certify">Certify this drive</button>
        </template>
      </section>
    </template>
  </DriveReviewDrive>
</template>
