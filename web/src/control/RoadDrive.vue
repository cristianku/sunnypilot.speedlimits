<script setup lang="ts">
// Data › Roads, a drive opened ("/roads?drive=<route>"): its pictures one at a time with the type of road there, the
// map's or a person's (GET /api/panel/road-types/<route>), and the strip of the whole drive under it, coloured as the
// list, a dark line under the stretches a person set, an orange one under those to review, a blue one over the stretch a
// tap sets; a tap on it goes there, its knob is dragged from where it is. A long drive has a second strip under it, the
// minute or two around the picture, its pictures five pixels apart: a drag moves the drive under the line in its middle
// picture by picture, a tap goes to the picture tapped, a wheel or a trackpad scrolls it; its stretch is framed on the
// strip of the drive. Play, the arrows and space step as in Results › Drive (DriveView.vue). The map only proposes: a
// person confirms its type or taps the right one, for the stretch shown (or from a marked start to here), saved at once
// (POST, the drive comes back), after a question when it is longer than a minute, and only then the frames are usable in
// training; Next to review goes to the next stretch no person set, however sure the map is. On a phone the loop (Next to
// review, look, Confirm or tap a type) fits one screen; the rest is under it.
// The intersections of the drive (nnslr junctions: the map's, with a person's corrections over them) are dots under the
// strip; under the type buttons and Next to review (which also stops on the junctions to check) a person marks the
// picture shown "Junction here" or "Not a junction" (POST .../junction, kept apart from the map; Undo takes the last
// back).
import {computed, onBeforeUnmount, onMounted, ref, shallowRef, useTemplateRef, watch} from 'vue';
import {api, messageOf, post} from '../common/api';
import {plural, thousands} from '../common/format';
import type {RoadDriveView, RoadJunction, RoadType} from '../common/panel-types';
import {askConfirm} from '../composables/confirm';
import {toast} from '../composables/toast';
import {checkStretches, frameTime, nextToCheck, reasonOf, ROAD_LABEL, ROAD_TYPES, roadRuns, stretchAt} from './roads';
import Segmented from './Segmented.vue';
import {goTo} from './tabs';
import {SPEEDS, type Speed} from './watch';

const props = defineProps<{route: string; readOnly: boolean}>();

const MARKS = {
  play: 'M8 5.6v12.8a.8.8 0 0 0 1.2.7l10.3-6.4a.8.8 0 0 0 0-1.4L9.2 4.9A.8.8 0 0 0 8 5.6z',
  pause: 'M7 5h3.2v14H7zM13.8 5H17v14h-3.2z',
  back: 'M14.5 5.5 8 12l6.5 6.5',
  forward: 'M9.5 5.5 16 12l-6.5 6.5',
} as const;
/** The choices of a correction, null: the person cannot tell. */
const CHOICES: [key: RoadType | null, label: string][] = [['town', 'Town'], ['zone30', 'Zone 30'], ['rural', 'Rural'],
  ['expressway', 'Expressway'], ['motorway', 'Motorway'], ['ramp', 'Ramp'], ['merge', 'Merge lane'], ['parking', 'Parking'],
  ['garage', 'Garage'], [null, 'Can\'t tell']];
/** Longer than this (pictures, one a second), a correction asks first: one tap must not overwrite minutes of a drive. */
const ASK_ABOVE = 60;

// -- the drive -----------------------------------------------------------------------------------------------------------
const payload = shallowRef<RoadDriveView | null>(null);
const error = ref('');
let asked = 0;
async function read(route: string): Promise<void> {
  const ticket = ++asked;
  try {
    const answer = await api<RoadDriveView>('/api/panel/road-types/' + encodeURIComponent(route));
    if (ticket !== asked) return;
    payload.value = answer;
    error.value = '';
    document.title = 'Speed limits · ' + (answer.title ?? 'Roads');
  } catch (problem) {
    if (ticket === asked) error.value = messageOf(problem);
  }
}
const frames = computed(() => payload.value?.frames ?? []);
const runs = computed(() => roadRuns(frames.value));
/** The stretches to review: a line under each on the strip, Next to review goes from one to the next. */
const checks = computed(() => checkStretches(frames.value));
/** The narrowest mark of the strip, in frames: a stretch of one picture (the stretch a tap sets three times that, to show
 * beside the position). */
const thin = computed(() => Math.max(1, frames.value.length / 250));
const facts = computed(() => {
  const drive = payload.value;
  if (!drive) return '';
  return [drive.when, drive.minutes !== null ? drive.minutes + ' min' : null, plural(drive.frames.length, 'picture'),
    drive.usable ? thousands(drive.usable) + ' usable in training' : null,
    drive.cant_tell ? thousands(drive.cant_tell) + ' can\'t tell' : null,
    drive.to_review ? thousands(drive.to_review) + ' to review' : null].filter(Boolean).join(' · ');
});

/** Back to the list: back in the history when the list opened this drive, so it is as it was. */
function back(): void {
  if ((history.state as {fromList?: boolean} | null)?.fromList) history.back();
  else goTo('roads');
}

// -- the frame wanted, and the one shown (its picture decoded first) -----------------------------------------------------
const playing = ref(false);
const speed = ref<Speed>(2);
const cursor = ref(0);
const shown = ref(0);
const failed = ref(false);
const frame = computed(() => frames.value[shown.value] ?? null);
/** Mark start: the first frame of the stretch to correct, up to the one shown. */
const mark = ref<number | null>(null);

let cache = new Map<string, Promise<boolean>>();
function load(url: string): Promise<boolean> {
  let loaded = cache.get(url);
  if (loaded === undefined) {
    const image = new Image();
    image.decoding = 'async';
    image.src = url;
    loaded = image.decode().then(() => true, () => false);
    cache.set(url, loaded);
    if (cache.size > 48) cache.delete(cache.keys().next().value as string);
  }
  return loaded;
}
function showFrame(index: number): void {
  const url = frames.value[index]?.image ?? null;
  if (url === null) {
    shown.value = index;
    failed.value = false;
  } else {
    void load(url).then((ok) => {
      if (cursor.value !== index) return;
      shown.value = index;
      failed.value = !ok;
    });
  }
  // The next pictures (playing, the right arrow) and the two before (the left arrow).
  for (const step of [1, 2, 3, 4, 5, 6, -1, -2]) {
    const near = frames.value[index + step]?.image;
    if (near) void load(near);
  }
}
watch(cursor, showFrame);
watch(() => props.route, (route) => {
  playing.value = false;
  payload.value = null;
  mark.value = null;
  cache = new Map();
  cursor.value = 0;
  void read(route).then(() => showFrame(0));
}, {immediate: true});

function seek(index: number): void {
  cursor.value = Math.min(Math.max(0, frames.value.length - 1), Math.max(0, index));
}
let timer: number | undefined;
function schedule(): void {
  window.clearTimeout(timer);
  if (!playing.value || shown.value !== cursor.value) return;
  timer = window.setTimeout(() => {
    if (cursor.value >= frames.value.length - 1) playing.value = false;
    else seek(cursor.value + 1);
  }, 1000 / speed.value);
}
watch([playing, shown, speed], schedule);
function toggle(): void {
  if (!playing.value && cursor.value >= frames.value.length - 1) seek(0);
  playing.value = !playing.value;
}
function step(by: number): void {
  playing.value = false;
  seek(cursor.value + by);
}
const speedOptions = SPEEDS.map((value) => ({key: String(value), label: (value === 0.5 ? '½' : String(value)) + '×'}));
const speedKey = computed({
  get: (): string => String(speed.value),
  set: (value: string) => { speed.value = (SPEEDS.find((item) => String(item) === value) ?? 2) as Speed; },
});

const root = useTemplateRef<HTMLElement>('root');
function onKey(event: KeyboardEvent): void {
  if (event.altKey || event.ctrlKey || event.metaKey || !root.value || !(event.target instanceof Node)) return;
  const tag = event.target instanceof HTMLElement ? event.target.tagName : '';
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if (event.key === ' ' && tag !== 'BUTTON') {
    event.preventDefault();
    toggle();
  } else if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
    event.preventDefault();
    step(event.key === 'ArrowRight' ? 1 : -1);
  }
}
onMounted(() => document.addEventListener('keydown', onKey));
onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKey);
  window.clearTimeout(timer);
  sizes?.disconnect();
});

// -- the junctions: dots under the strip, the one at the picture shown, and a person's correction --------------------------
const junctions = computed(() => payload.value?.junctions?.marks ?? []);
/** The junctions to check (a GPS that stops in one, a track far from it, a long wait in it, the map's on another road):
 * Next to review stops on them too. */
const junctionChecks = computed(() => junctions.value.filter((mark) => mark.doubt !== null));
/** The dots of a strip, each at `place` of its middle (in %): those that are not junctions under, the junctions over
 * them, those to check on top. */
function dotsOf(marks: readonly RoadJunction[], place: (middle: number) => number) {
  const order = (mark: RoadJunction): number => mark.doubt !== null ? 2 : mark.label === 'junction' ? 1 : 0;
  return [...marks].sort((a, b) => order(a) - order(b))
    .map((mark) => ({key: mark.id + '/' + mark.first, left: place((mark.first + mark.last + 1) / 2),
      kind: mark.doubt !== null ? 'check' : mark.label === 'junction' ? 'on' : 'not'}));
}
const dots = computed(() => dotsOf(junctions.value, (middle) => 100 * middle / frames.value.length));
/** Where Next to review stops: the first frame of each stretch to review and of each junction to check, in drive order. */
const stops = computed(() => [...new Set([...checks.value.map((item) => item.first),
  ...junctionChecks.value.map((mark) => mark.first)])].sort((a, b) => a - b).map((first) => ({first})));
/** What is left to look at, in the title of its count: "3 stretches · 2 junctions". */
const left = computed(() => [checks.value.length ? plural(checks.value.length, 'stretch', 'stretches') : '',
  junctionChecks.value.length ? plural(junctionChecks.value.length, 'junction') : ''].filter(Boolean).join(' · '));
/** The junction at the picture shown: one it is inside, else the one it is just after (its "passed" frame), else a
 * junction of the map on another road right next to it (two pictures, as the server counts it). */
const junctionHere = computed((): RoadJunction | null => {
  const at = shown.value;
  return junctions.value.find((mark) => mark.label !== 'maybe' && mark.first <= at && at <= mark.last)
    ?? junctions.value.find((mark) => mark.label !== 'maybe' && mark.passed === at)
    ?? junctions.value.find((mark) => mark.label === 'maybe' && Math.abs(mark.first - at) <= 2) ?? null;
});
/** The junction at the picture shown in a few words: "Roundabout · from the map", "No junction here". */
const junctionText = computed((): string => {
  const mark = junctionHere.value;
  if (!payload.value?.junctions?.found) return 'Not found yet: Find the road types reads them';
  if (mark === null) return 'No junction here';
  if (mark.label === 'dropped') return 'Not a junction, set by you (the map has one)';
  if (mark.label === 'maybe') return mark.reviewed ? 'Not a junction, set by you' : 'Maybe a junction';
  const what = mark.kind === 'roundabout' ? 'Roundabout' : 'Junction';
  const by = mark.by === 'person' ? 'set by you' : mark.reviewed ? 'checked by you' : 'from the map';
  return (mark.passed === shown.value ? 'Just past a ' + what.toLowerCase() + ': a sign\'s limit ends here' : what)
    + ' · ' + by;
});
/** Why to check it, or the roads that meet there. */
const junctionDetail = computed((): string => {
  const mark = junctionHere.value;
  if (mark === null) return '';
  if (mark.doubt !== null) return mark.doubt;
  if (mark.label !== 'junction' || !mark.arms || !mark.roads) return '';
  return mark.arms + ' roads meet: ' + mark.roads.map((road) => road.replace('_', ' ')).join(', ');
});
/** A person's choice at the picture shown (a solid ring on its button), and the map's (a dashed one). */
const junctionSet = computed((): 'here' | 'none' | null => {
  const mark = junctionHere.value;
  if (mark === null || !mark.reviewed) return null;
  return mark.label === 'junction' ? 'here' : 'none';
});
const junctionMap = computed((): boolean => junctionHere.value?.label === 'junction' && !junctionHere.value.reviewed);
async function setJunction(choice: 'here' | 'none'): Promise<void> {
  const target = frame.value;
  if (!target || saving.value) return;
  saving.value = true;
  try {
    payload.value = await post<RoadDriveView>('/api/panel/road-types/' + encodeURIComponent(props.route) + '/junction',
      {frame_key: target.key, junction: choice});
    toast(choice === 'here' ? 'Junction saved' : 'Not a junction · saved');
  } catch (problem) {
    toast(messageOf(problem), true);
  } finally {
    saving.value = false;
  }
}

const stage = useTemplateRef<HTMLElement>('stage');
/** Next to review: the next stretch no person set, sure or not, or the next junction to check. */
function nextCheck(): void {
  const index = nextToCheck(stops.value, cursor.value);
  if (index === null) {
    toast('Nothing else to review on this drive.');
    return;
  }
  step(index - cursor.value);
  // The picture out of sight (a phone scrolled down to the buttons): back to its top, under the top bar.
  const box = stage.value?.getBoundingClientRect();
  const bar = document.querySelector('.topbar')?.getBoundingClientRect().bottom ?? 0;
  if (box && (box.top < bar || box.bottom > window.innerHeight)) window.scrollTo({top: window.scrollY + box.top - bar - 8});
}

// -- the strip of the drive: a tap goes to that point, the knob is dragged from where it is -------------------------------
const strip = useTemplateRef<HTMLElement>('strip');
/** The width of the strip in pixels: how many pictures the close strip shows. */
const width = ref(0);
let sizes: ResizeObserver | undefined;
watch(strip, (element) => {
  sizes?.disconnect();
  if (!element) return;
  sizes = new ResizeObserver(([entry]) => { width.value = entry?.contentRect.width ?? 0; });
  sizes.observe(element);
});
/** A finger or the mouse on a strip, for the time over the knob. */
const dragging = ref(false);
/** A drag on the strip: how far right of the knob it was taken (0: a tap away from it, the knob came to the finger). */
let grab: number | null = null;
/** Where the knob is, in % of the strip. */
const headAt = computed(() => 100 * (cursor.value + 0.5) / Math.max(1, frames.value.length) + '%');
/** The time of the picture wanted, over the knob while it is dragged. */
const headTime = computed(() => {
  const at = frames.value[cursor.value];
  return at && payload.value ? frameTime(at.key, payload.value.start) : '';
});
function stripAt(x: number): void {
  const box = strip.value?.getBoundingClientRect();
  if (!box || box.width <= 0 || frames.value.length === 0) return;
  seek(Math.floor(((x - box.left) / box.width) * frames.value.length));
}
function stripDown(event: PointerEvent): void {
  const box = strip.value?.getBoundingClientRect();
  if (!box || box.width <= 0 || frames.value.length === 0) return;
  playing.value = false;
  const head = box.left + box.width * (cursor.value + 0.5) / frames.value.length;
  // Near the knob (a finger is wide), it is taken where it is: no jump of a few pictures.
  grab = Math.abs(event.clientX - head) <= (event.pointerType === 'mouse' ? 8 : 22) ? event.clientX - head : 0;
  dragging.value = true;
  (event.currentTarget as Element).setPointerCapture(event.pointerId);
  if (grab === 0) stripAt(event.clientX);
}
function stripMove(event: PointerEvent): void {
  if (grab !== null) stripAt(event.clientX - grab);
}
function stripUp(): void {
  grab = null;
  dragging.value = false;
}

// -- the close strip: the minute or two around the picture, a drag moves the drive under its middle ----------------------
/** Pixels between two pictures on the close strip: a finger moves them one by one. */
const CLOSE_PX = 5;
/** The pictures it shows: as many as fit, an even count so the picture wanted is in the middle. */
const near = computed(() => Math.max(20, 2 * Math.round(width.value / CLOSE_PX / 2)));
/** Only on a drive the strip shows tighter than that. */
const zoomed = computed(() => width.value > 0 && frames.value.length > 2 * near.value);
/** The frame at its left edge (with a fraction: the middle of the picture wanted is the middle of the strip). */
const nearFrom = computed(() => cursor.value + 0.5 - near.value / 2);
function inView<T extends {first: number; count: number}>(items: readonly T[]): T[] {
  const from = nearFrom.value;
  return zoomed.value ? items.filter((item) => item.first < from + near.value && item.first + item.count > from) : [];
}
const nearRuns = computed(() => inView(runs.value));
const nearChecks = computed(() => inView(checks.value));
/** A notch every 10 pictures, a long one every 60, inside the drive. */
const ticks = computed((): number[] => {
  if (!zoomed.value) return [];
  const marks: number[] = [];
  const last = Math.min(frames.value.length, nearFrom.value + near.value);
  for (let tick = Math.max(10, Math.ceil(nearFrom.value / 10) * 10); tick < last; tick += 10) marks.push(tick);
  return marks;
});
const nearDots = computed(() => dotsOf(inView(junctions.value.map((mark) => ({...mark, count: mark.last - mark.first + 1}))),
  (middle) => 100 * (middle - nearFrom.value) / near.value));
/** Its stretch framed on the strip of the drive, in %. */
const lens = computed(() => {
  const count = frames.value.length;
  const from = Math.max(0, nearFrom.value);
  const to = Math.min(count, nearFrom.value + near.value);
  return {left: 100 * from / count + '%', width: 100 * (to - from) / count + '%'};
});
const close = useTemplateRef<HTMLElement>('close');
/** A drag on it: where it went down, the picture then, whether it moved (else a tap). */
let pull: {x: number; from: number; moved: boolean} | null = null;
function closeDown(event: PointerEvent): void {
  playing.value = false;
  pull = {x: event.clientX, from: cursor.value, moved: false};
  dragging.value = true;
  (event.currentTarget as Element).setPointerCapture(event.pointerId);
}
function closeMove(event: PointerEvent): void {
  if (pull === null) return;
  const moved = event.clientX - pull.x;
  if (Math.abs(moved) > 4) pull.moved = true;
  // The drive follows the finger: to the right, back in time.
  if (pull.moved) seek(pull.from - Math.round(moved / (width.value / near.value)));
}
function closeUp(event: PointerEvent): void {
  const box = close.value?.getBoundingClientRect();
  if (pull !== null && !pull.moved && event.type === 'pointerup' && box && box.width > 0) {
    seek(Math.floor(nearFrom.value + ((event.clientX - box.left) / box.width) * near.value));
  }
  pull = null;
  dragging.value = false;
}
/** A wheel or a trackpad: the pixels it scrolls, a picture every CLOSE_PX. */
let wheeled = 0;
function closeWheel(event: WheelEvent): void {
  const delta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
  if (!delta) return;
  event.preventDefault();
  playing.value = false;
  if (Math.sign(delta) !== Math.sign(wheeled)) wheeled = 0;
  wheeled += delta * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? width.value : 1);
  const by = Math.trunc(wheeled / (width.value / near.value));
  if (by === 0) return;
  wheeled -= by * (width.value / near.value);
  seek(cursor.value + by);
}

// -- the frame shown, and its correction -----------------------------------------------------------------------------------
const kind = computed(() => frame.value?.road_type ?? 'unknown');
/** Why the frame has its type, in a driver's words ('' when the map's reason has none). */
const reason = computed(() => frame.value ? reasonOf(frame.value) : '');
/** The stretch shown (stretchAt: of one kind, as sure, not set by a person when this frame is not). */
const stretch = computed((): [number, number] | null => frames.value.length ? stretchAt(frames.value, shown.value) : null);
/** Which part of the stretch a tap sets: all of it, its start up to the picture shown, or from it to its end, so a
 * long stretch (183 pictures without GPS: a garage, then the streets) is split where the picture shows the change. */
type Part = 'all' | 'before' | 'after';
const part = ref<Part>('all');
// Another stretch: the whole of it again.
watch(() => stretch.value?.[0], () => { part.value = 'all'; });
/** The part can be chosen when the picture shown is inside its stretch, not at one of its ends. */
const splittable = computed(() => mark.value === null && stretch.value !== null && shown.value > stretch.value[0]
  && shown.value < stretch.value[1]);
const partOptions = computed((): {key: Part; label: string}[] => [{key: 'all', label: 'Whole stretch'},
  {key: 'before', label: '✂ Before'}, {key: 'after', label: '✂ After'}]);
/** The frames a correction covers: from the mark to here, else the part chosen of the stretch shown. */
const span = computed((): [number, number] | null => {
  if (!frames.value.length) return null;
  if (mark.value !== null) return [Math.min(mark.value, shown.value), Math.max(mark.value, shown.value)];
  const whole = stretch.value;
  if (whole === null || !splittable.value || part.value === 'all') return whole;
  return part.value === 'before' ? [whole[0], shown.value] : [shown.value, whole[1]];
});
/** The times of the stretch a correction covers: "16:13:34 – 16:21:38". */
const spanTimes = computed((): string => {
  const [first, last] = span.value ?? [0, -1];
  const start = payload.value?.start ?? null;
  const from = frames.value[first];
  const to = frames.value[last];
  if (!from || !to) return '';
  return frameTime(from.key, start) + (last > first ? ' – ' + frameTime(to.key, start) : '');
});
/** "these 84 pictures", "this picture": what a tap sets, in the title of the buttons. */
const spanCount = computed((): string => {
  const [first, last] = span.value ?? [0, -1];
  const count = last - first + 1;
  return count === 1 ? 'this picture' : 'these ' + thousands(count) + ' pictures';
});
/** The map's type here, while a person has not said it: Confirm sets it, as a tap on its button. */
const proposal = computed((): RoadType | null => mark.value === null && frame.value && !frame.value.reviewed ? frame.value.road_type : null);
const saving = ref(false);
async function setType(type: RoadType | null): Promise<void> {
  const [first, last] = span.value ?? [0, -1];
  const from = frames.value[first];
  const to = frames.value[last];
  if (!from || !to || saving.value) return;
  const count = last - first + 1;
  const label = type === null ? 'Can\'t tell' : ROAD_LABEL[type];
  if (count > ASK_ABOVE && !await askConfirm({
    title: 'Set ' + plural(count, 'picture') + ' (' + Math.round(count / 60) + ' min) to ' + label + '?',
    text: 'The stretch marked in blue on the strip, ' + spanTimes.value + '. Undo last change takes it back.',
    action: 'Set to ' + label,
  })) return;
  if (saving.value) return;
  saving.value = true;
  try {
    payload.value = await post<RoadDriveView>('/api/panel/road-types/' + encodeURIComponent(props.route),
      {from: from.key, to: to.key, road_type: type});
    mark.value = null;
    part.value = 'all';
    toast(label + ' · ' + plural(count, 'picture') + ' saved');
  } catch (problem) {
    toast(messageOf(problem), true);
  } finally {
    saving.value = false;
  }
}
/** Undo the last road type (what '/undo'), or the last junction ('/junction/undo'). */
async function undo(what: '/undo' | '/junction/undo' = '/undo'): Promise<void> {
  if (saving.value) return;
  saving.value = true;
  try {
    payload.value = await post<RoadDriveView>('/api/panel/road-types/' + encodeURIComponent(props.route) + what);
    toast(what === '/undo' ? 'Last change undone' : 'Last junction change undone');
  } catch (problem) {
    toast(messageOf(problem), true);
  } finally {
    saving.value = false;
  }
}
</script>

<template>
  <section ref="root" class="card rt-view" aria-labelledby="rt-view-title">
    <div class="rt-view-head">
      <button type="button" class="link-button rt-back" @click="back"><span aria-hidden="true">‹</span> Roads</button>
      <div class="rt-view-name">
        <h2 id="rt-view-title">{{ payload ? payload.title ?? 'Places unknown' : 'Drive' }}</h2>
        <span>{{ facts }}</span>
      </div>
    </div>

    <p v-if="error" class="banner">The drive cannot be read: {{ error }}</p>
    <p v-else-if="!payload" class="empty">Loading the drive…</p>
    <p v-else-if="frames.length === 0" class="empty">No road types for this drive yet: Find the road types reads them.</p>
    <div v-else class="rt-view-main">
      <div ref="stage" class="rt-stage" tabindex="0" role="img"
           :aria-label="'Picture ' + (cursor + 1) + ' of ' + frames.length + ': ' + ROAD_LABEL[kind]" @click="toggle">
        <img v-if="frame?.image && !failed" :src="frame.image" alt="" draggable="false">
        <p v-else class="wp-noimage">{{ frame?.image ? 'The picture cannot be loaded.' : 'No picture of this frame.' }}</p>
        <span v-if="!playing" class="wp-paused" aria-hidden="true"><svg class="icon" viewBox="0 0 24 24"><path :d="MARKS.play" fill="currentColor"/></svg></span>
      </div>

      <div class="rt-time">
        <div ref="strip" class="rt-scrub" role="slider" :aria-valuemin="1" :aria-valuemax="frames.length"
             :aria-valuenow="cursor + 1" :aria-valuetext="headTime" aria-label="Where in the drive" @pointerdown="stripDown"
             @pointermove="stripMove" @pointerup="stripUp" @pointercancel="stripUp">
          <div class="rt-bar">
            <svg class="rt-track" :viewBox="'0 0 ' + frames.length + ' 20'" preserveAspectRatio="none" aria-hidden="true">
              <!-- The acceleration lane: the ramp's cyan over the motorway's blue, for both strips (control.html .rt-merge). -->
              <defs><linearGradient id="rt-merge-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0.4" class="rt-merge-top"/>
                <stop offset="0.4" class="rt-merge-bottom"/></linearGradient></defs>
              <rect v-for="run in runs" :key="run.first" :x="run.first" y="3" :width="run.count" height="11" :class="'rt-' + run.kind"/>
              <template v-for="run in runs" :key="'r' + run.first">
                <rect v-if="run.reviewed" :x="run.first" y="16" :width="run.count" height="3" class="rt-done"/>
              </template>
              <rect v-for="item in checks" :key="'c' + item.first" :x="item.first" y="16" :width="Math.max(item.count, thin)" height="3"
                    class="rt-tocheck"/>
              <rect v-if="span" :x="span[0]" y="0" :width="Math.max(span[1] - span[0] + 1, 3 * thin)" height="2.2" class="rt-marked"/>
              <rect v-if="splittable && part !== 'all'" :x="cursor" y="-2" :width="Math.max(thin, 1)" height="24" class="rt-cut"/>
            </svg>
            <!-- The knob and the frame of the close strip in HTML: round, as the SVG is stretched. -->
            <i v-if="zoomed" class="rt-lens" :style="lens" aria-hidden="true"></i>
            <i class="rt-knob" :style="{left: headAt}" aria-hidden="true"></i>
            <span v-if="dragging && headTime" class="rt-bubble" :style="{left: 'clamp(34px, ' + headAt + ', calc(100% - 34px))'}"
                  aria-hidden="true">{{ headTime }}</span>
          </div>
          <!-- The junctions: round dots in HTML, as the strip's SVG is stretched. -->
          <div v-if="dots.length" class="rt-jrow" aria-hidden="true">
            <i v-for="dot in dots" :key="dot.key" class="rt-jdot" :class="dot.kind" :style="{left: dot.left + '%'}"></i>
            <i class="rt-jhead" :style="{left: headAt}"></i>
          </div>
        </div>
        <!-- The minute or two around the picture: the drive moves under the line in the middle. -->
        <div v-if="zoomed" ref="close" class="rt-close" aria-hidden="true" @pointerdown="closeDown" @pointermove="closeMove"
             @pointerup="closeUp" @pointercancel="closeUp" @wheel="closeWheel">
          <svg class="rt-track" :viewBox="nearFrom + ' 0 ' + near + ' 20'" preserveAspectRatio="none">
            <rect v-for="run in nearRuns" :key="run.first" :x="run.first" y="3" :width="run.count" height="11" :class="'rt-' + run.kind"/>
            <template v-for="run in nearRuns" :key="'r' + run.first">
              <rect v-if="run.reviewed" :x="run.first" y="16" :width="run.count" height="3" class="rt-done"/>
            </template>
            <rect v-for="item in nearChecks" :key="'c' + item.first" :x="item.first" y="16" :width="item.count" height="3" class="rt-tocheck"/>
            <rect v-if="span" :x="span[0]" y="0" :width="span[1] - span[0] + 1" height="2.2" class="rt-marked"/>
            <rect v-for="tick in ticks" :key="'t' + tick" :x="tick - 0.15" y="3" width="0.3" :height="tick % 60 === 0 ? 11 : 3.5"
                  class="rt-tick"/>
          </svg>
          <div v-if="dots.length" class="rt-jrow">
            <i v-for="dot in nearDots" :key="dot.key" class="rt-jdot" :class="dot.kind" :style="{left: dot.left + '%'}"></i>
          </div>
          <i class="rt-close-head"></i>
        </div>
        <div class="rt-controls">
          <button type="button" class="btn small primary rt-play" :aria-pressed="playing" @click="toggle"><svg class="icon"
            viewBox="0 0 24 24" aria-hidden="true"><path :d="playing ? MARKS.pause : MARKS.play" fill="currentColor"/></svg>{{ playing ? 'Pause' : 'Play' }}</button>
          <button type="button" class="btn small rt-step" aria-label="Previous picture" :disabled="cursor === 0" @click="step(-1)"><svg
            class="icon" viewBox="0 0 24 24" aria-hidden="true"><path :d="MARKS.back" fill="none" stroke="currentColor" stroke-width="2.2"
            stroke-linecap="round" stroke-linejoin="round"/></svg></button>
          <button type="button" class="btn small rt-step" aria-label="Next picture" :disabled="cursor >= frames.length - 1" @click="step(1)"><svg
            class="icon" viewBox="0 0 24 24" aria-hidden="true"><path :d="MARKS.forward" fill="none" stroke="currentColor" stroke-width="2.2"
            stroke-linecap="round" stroke-linejoin="round"/></svg></button>
          <span class="rt-pos"><b>{{ frame ? frameTime(frame.key, payload.start) : '' }}</b><span class="rt-count"> ·
            {{ thousands(cursor + 1) }} / {{ thousands(frames.length) }}</span></span>
        </div>
      </div>

      <aside class="rt-side" aria-label="The road here">
        <div class="rt-now">
          <span class="rt-chip"><i class="rt-key" :class="'rt-' + kind" aria-hidden="true"></i>{{ ROAD_LABEL[kind] }}</span>
          <span v-if="frame?.reviewed" class="rt-flag" :class="frame.road_type === null ? 'none' : 'done'">{{
            frame.road_type === null ? 'Can\'t tell · not used' : 'Usable in training' }}</span>
          <span v-else-if="frame" class="rt-flag unsure">{{ frame.road_type === null ? 'To set' : frame.sure ? 'To confirm' : 'To confirm · not sure' }}</span>
        </div>
        <p v-if="reason" class="rt-basis">{{ reason }}</p>

        <div class="rt-fix">
          <!-- Inside a stretch: cut it at the picture shown (the white line on the strip) and set one side. -->
          <p v-if="splittable" class="rt-cut-head"><strong>✂ Cut here?</strong> <span>The stretch goes on before and after
            this picture: set the whole of it, or only the part before or after the cut.</span></p>
          <Segmented v-if="splittable" v-model="part" :options="partOptions" label="Cut the stretch at this picture" class="rt-part"/>
          <p class="rt-fix-head"><strong>{{ mark !== null ? 'Set from the mark to here to:' : 'Set ' + spanCount + ' to:' }}</strong>
            <span>{{ spanTimes }}</span></p>
          <button v-if="proposal" type="button" class="btn primary rt-confirm" :disabled="saving || readOnly"
                  @click="setType(proposal)">Confirm {{ ROAD_LABEL[proposal] }}</button>
          <!-- Real buttons, one a type: a person must see at once that a tap here changes the stretch. -->
          <div class="rt-pick" role="group" aria-label="Type of road">
            <button v-for="[key, label] in CHOICES" :key="key ?? 'none'" type="button" class="btn rt-type"
                    :aria-pressed="Boolean(frame?.reviewed) && mark === null && frame?.road_type === key"
                    :class="{'rt-map': key !== null && !frame?.reviewed && mark === null && frame?.road_type === key}"
                    :disabled="saving || readOnly" @click="setType(key)"><i class="rt-key" :class="'rt-' + (key ?? 'unknown')"
                    aria-hidden="true"></i>{{ label }}</button>
          </div>
        </div>

        <button type="button" class="btn rt-next" :disabled="stops.length === 0" @click="nextCheck">{{ stops.length ? 'Next to review'
          : 'Nothing left to review' }}<span v-if="stops.length" class="rt-left" :title="left">{{ thousands(stops.length) }} left</span></button>

        <!-- The junction at the picture shown: a person adds one the map misses, or drops one that is not (a driveway,
             a car park's exit, a bridge). -->
        <div class="rt-jx">
          <div class="rt-jx-head"><strong>Junction</strong> <span :class="{'rt-jx-doubt': junctionHere?.doubt}">{{ junctionText }}</span>
            <button v-if="(payload.junctions?.corrections ?? 0) > 0" type="button" class="link-button rt-jx-undo"
                    :disabled="saving || readOnly" aria-label="Undo the last junction change" @click="undo('/junction/undo')">Undo</button></div>
          <p v-if="junctionDetail" class="rt-jx-why">{{ junctionDetail }}</p>
          <div class="rt-jx-pick" role="group" aria-label="Junction at this picture">
            <button type="button" class="btn rt-type" :aria-pressed="junctionSet === 'here'" :class="{'rt-map': junctionMap}"
                    :disabled="saving || readOnly || !payload.junctions?.found" @click="setJunction('here')"><i class="rt-jdot on"
                    aria-hidden="true"></i>Junction here</button>
            <button type="button" class="btn rt-type" :aria-pressed="junctionSet === 'none'"
                    :disabled="saving || readOnly || !payload.junctions?.found" @click="setJunction('none')"><i class="rt-jdot not"
                    aria-hidden="true"></i>Not a junction</button>
          </div>
        </div>

        <div class="rt-more">
          <Segmented v-model="speedKey" :options="speedOptions" label="Speed" class="rt-speed"/>
          <div class="rt-fix-row">
            <button type="button" class="btn small" :disabled="readOnly" @click="mark = mark === null ? shown : null">{{ mark === null ? 'Mark start' : 'Clear the mark' }}</button>
            <button v-if="payload.corrections > 0" type="button" class="btn small" :disabled="saving || readOnly" @click="undo()">Undo last change</button>
          </div>
          <p class="rt-hint">{{ readOnly ? 'Read-only: corrections are turned off on this server.'
            : mark === null ? 'Confirm the proposal or tap the right type: the stretch marked in blue on the strip is saved at once, a long one after a question, and only then used in training. ✂ Before or After split it at the picture shown; Mark start chooses any start.'
              : 'Now go to the end of the stretch, then tap its type.' }}</p>
          <ul class="rt-legend rt-marks" aria-label="The strip">
            <li v-for="[key, label] in ROAD_TYPES" :key="key"><span class="rt-key" :class="'rt-' + key" aria-hidden="true"></span>{{ label }}</li>
            <li><span class="rt-line rt-tocheck" aria-hidden="true"></span>To review</li>
            <li><span class="rt-line rt-done" aria-hidden="true"></span>Reviewed by you</li>
            <li><span class="rt-line rt-marked" aria-hidden="true"></span>What a tap sets</li>
            <li><span class="rt-jdot on" aria-hidden="true"></span>Junction</li>
            <li><span class="rt-jdot check" aria-hidden="true"></span>Junction to check</li>
            <li><span class="rt-jdot not" aria-hidden="true"></span>Not a junction</li>
          </ul>
        </div>
      </aside>
    </div>
  </section>
</template>
