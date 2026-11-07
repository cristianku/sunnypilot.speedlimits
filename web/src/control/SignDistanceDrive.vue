<script setup lang="ts">
// Data › Sign distance, a drive opened ("/sign-distance?drive=<route>&sign=<key>"): one sign at a time. The road video
// frame by frame over the sign's window (only the frames the server lists: never a second kept only at a car park's
// speed), with what the car's screen showed on each frame (the "Vision 14 m" chip under the picture, in the car's
// unit). Under it vision's distance over the window, with the last sighting, vision's zero and the person's zero on it;
// a tap or a drag there or on the slider moves to a frame. The person marks the frame they call zero (the sign is about
// to leave the picture), or that they cannot tell: always the frame whose picture is on screen, and only once the one
// asked for is (never a frame still loading). Clear takes the mark away and Undo takes back the last line of the sign on
// screen (the marks are a file only ever added to, nnslr_tools.sign_distance). Then the figures of the sign: late_s
// first, late_m second. Only a person signed in marks: on a read-only server, or signed out, the buttons are not there,
// nor before the server has said it takes marks (readOnly null until then: no flash of them on a read-only server).
import {computed, onBeforeUnmount, onMounted, ref, shallowRef, watch} from 'vue';
import {api, ApiError, messageOf, post} from '../common/api';
import {plural} from '../common/format';
import {isGuest} from '../common/guest';
import {pictureCache} from '../composables/pictures';
import {toast} from '../composables/toast';
import {API, classText, type DriveView, type Drive, frameIndex, frameUrl, lateSentence, type MarkName,
  markLabel, nearestFrame, screenText, type Sign, signedMetres, type SignView, statusTone, timeText, toMark,
  type WindowFrame}
  from './sign-distance';
import {goTo, showSign, tabSign} from './tabs';

/** readOnly is null until the server's status is in (no buttons and no note yet). */
const props = defineProps<{route: string; readOnly: boolean | null}>();
const guest = isGuest();

// -- the drive and its signs ---------------------------------------------------------------------------------------------
const drive = shallowRef<DriveView | null>(null);
const error = ref('');
const view = shallowRef<SignView | null>(null);
const signError = ref('');
/** The sign being read (its view not here yet), so a second call for it is not sent. While it is another sign than the
 * one on screen nothing is marked: a mark goes to the sign the person sees, never to one leaving or one not shown yet. */
const loading = ref<string | null>(null);
const switching = computed(() => loading.value !== null && loading.value !== view.value?.sign.key);

const signs = computed(() => drive.value?.signs ?? []);
const sign = computed((): Sign | null => view.value?.sign ?? null);
const frames = computed(() => view.value?.window.frames ?? []);
const unit = computed(() => view.value?.unit ?? drive.value?.unit ?? 'm');
const factor = computed(() => unit.value === 'ft' ? 3.28084 : 1);

/** The sign to open first: the one in the address, else the first still to mark, else the first compared. */
function firstSign(list: Sign[]): string | null {
  const wanted = tabSign.value;
  if (wanted && list.some((item) => item.key === wanted)) return wanted;
  return (list.find(toMark) ?? list.find((item) => item.included) ?? list[0])?.key ?? null;
}

/** The drive again every 2 s while its logs are being read (Drive review reads them), until they are. */
let polling = 0;
function stopPolling(): void {
  window.clearTimeout(polling);
  polling = 0;
}
async function loadDrive(): Promise<void> {
  stopPolling();
  const route = props.route;
  try {
    const answer = await api<DriveView>(API + '/drives/' + encodeURIComponent(route));
    if (route !== props.route) return;
    if (!answer || typeof answer.status !== 'string' || !Array.isArray(answer.signs) || !Array.isArray(answer.orphans)) {
      throw new Error('The server sent a drive this page cannot read.');
    }
    drive.value = answer;
    error.value = '';
    const key = firstSign(answer.signs);
    if (key !== null) openSign(key);
    if (answer.status === 'reading') polling = window.setTimeout(() => void loadDrive(), 2000);
  } catch (problem) {
    if (route === props.route) error.value = messageOf(problem);
  }
}

async function loadSign(key: string): Promise<void> {
  loading.value = key;
  try {
    const answer = await api<SignView>(API + '/drives/' + encodeURIComponent(props.route) + '/signs/' + encodeURIComponent(key));
    if (loading.value !== key) return;
    if (!answer?.sign || !answer.window || !Array.isArray(answer.window.frames) || !Array.isArray(answer.warnings)
      || !answer.series) {
      throw new Error('The server sent a sign this page cannot read.');
    }
    view.value = answer;
    signError.value = '';
    index.value = startIndex(answer);
    shownIndex.value = -1;
    failed.value = false;
    void pump();
  } catch (problem) {
    if (loading.value === key) signError.value = messageOf(problem);
  } finally {
    if (loading.value === key) loading.value = null;
  }
}

/** Opens a sign of this drive: its key in the address (no new history entry), its frames read. */
function openSign(key: string): void {
  showSign(key);
  if (view.value?.sign.key !== key && loading.value !== key) void loadSign(key);
}
// Back and Forward between two addresses of this page.
watch(tabSign, (key) => {
  if (key && view.value?.sign.key !== key && loading.value !== key) void loadSign(key);
});
watch(() => props.route, () => {
  stopPolling();
  view.value = null;
  drive.value = null;
  void loadDrive();
});

/** A sign sent back by the server after a mark or an undo, in place of the one shown and the one in the list. */
function applySign(next: Sign): void {
  if (view.value && view.value.sign.key === next.key) view.value = {...view.value, sign: next};
  if (drive.value) drive.value = {...drive.value, signs: drive.value.signs.map((item) => item.key === next.key ? next : item)};
}

// -- the frame wanted, and the one shown (its picture decoded first) -----------------------------------------------------
const index = ref(0);
const shownIndex = ref(-1);
const failed = ref(false);
const wanted = computed(() => frames.value[index.value] ?? null);
/** The frame whose picture is on screen: the chip, the time and a mark are this one's. */
const onScreen = computed(() => frames.value[shownIndex.value] ?? null);
const pictureUrl = computed(() => onScreen.value ? frameUrl(props.route, onScreen.value) : null);
/** The picture of the frame asked for is on screen: only then is a frame marked (never the one still loading). */
const ready = computed(() => onScreen.value !== null && shownIndex.value === index.value && !failed.value);
/** The picture's own shape, width over height (the stage is a road camera's 1344 x 760 until it is in). */
const ratio = ref('');
const wideRatio = ref('');
function shapeOf(event: Event): string {
  const image = event.target as HTMLImageElement;
  return image.naturalWidth > 0 && image.naturalHeight > 0 ? (image.naturalWidth / image.naturalHeight).toFixed(4) : '';
}
function pictureLoaded(event: Event): void {
  ratio.value = shapeOf(event) || ratio.value;
}
function widePictureLoaded(event: Event): void {
  wideRatio.value = shapeOf(event) || wideRatio.value;
}

// -- the wide road camera: the same frame beside the road one, where its minute has the same frames -----------------
const wideSegments = computed(() => {
  const list = view.value?.window.wide_segments;
  return new Set(Array.isArray(list) ? list : []);
});
/** Some minute of the window has the wide camera: the page shows the two pictures. */
const twoCameras = computed(() => wideSegments.value.size > 0);
function hasWide(frame: WindowFrame): boolean {
  return wideSegments.value.has(frame.segment);
}
/** The wide picture of the frame on screen decoded (true), not there or not shown (false); it never holds up the road
 * picture's frame, and is never one of another frame. */
const wideShown = ref(false);
const widePictureUrl = computed(() => onScreen.value && wideShown.value ? frameUrl(props.route, onScreen.value, true) : null);

function startIndex(answer: SignView): number {
  const list = answer.window.frames;
  const mark = answer.sign.mark;
  const marked = mark ? frameIndex(list, mark.segment, mark.frame) : -1;
  if (marked >= 0) return marked;
  const seen = nearestFrame(list, answer.sign.last_seen_s);
  return seen >= 0 ? seen : 0;
}

const load = pictureCache();
/** One picture asked of the server at a time, the latest frame wanted winning: a drag never queues a decode a frame. */
let pumping = false;
async function pump(): Promise<void> {
  if (pumping) return;
  pumping = true;
  try {
    for (;;) {
      const target = index.value;
      const frame = frames.value[target];
      if (!frame) {
        shownIndex.value = -1;
        break;
      }
      if (target === shownIndex.value && !failed.value) break;
      const wide = hasWide(frame);
      const [ok, wideOk] = await Promise.all([load(frameUrl(props.route, frame)),
        wide ? load(frameUrl(props.route, frame, true)) : Promise.resolve(false)]);
      if (index.value !== target || frames.value[target] !== frame) continue;
      failed.value = !ok;
      if (ok) {
        shownIndex.value = target;
        wideShown.value = wideOk;
        for (let step = 1; step <= 5; step += 1) {
          const ahead = frames.value[target + step];
          if (ahead) void load(frameUrl(props.route, ahead));
          if (ahead && hasWide(ahead)) void load(frameUrl(props.route, ahead, true));
        }
      }
      break;
    }
  } finally {
    pumping = false;
  }
}
watch(index, () => void pump());

function seek(to: number): void {
  if (!frames.value.length) return;
  index.value = Math.max(0, Math.min(frames.value.length - 1, Math.round(to)));
}
function seekTime(seconds: number | null | undefined): void {
  const found = nearestFrame(frames.value, seconds);
  if (found >= 0) seek(found);
}

// -- what the screen showed, and the curve of the window -----------------------------------------------------------------
const chip = computed(() => {
  const frame = onScreen.value ?? wanted.value;
  if (!frame) return null;
  const text = screenText(frame.shown, unit.value);
  return text === null ? {text: 'Vision –', title: 'No distance in the logs at this frame', zero: false}
    : {text: 'Vision ' + text, title: frame.shown === 0 ? 'The screen shows no number at 0' : 'What the car\'s screen showed',
      zero: frame.shown === 0};
});
const markIndex = computed(() => sign.value?.mark ? frameIndex(frames.value, sign.value.mark.segment, sign.value.mark.frame) : -1);
const onMark = computed(() => markIndex.value >= 0 && markIndex.value === (shownIndex.value >= 0 ? shownIndex.value : index.value));

const W = 1000;
const H = 60;
const span = computed(() => {
  const window = view.value?.window;
  const from = window?.from_s ?? frames.value[0]?.t_s ?? 0;
  const to = window?.to_s ?? frames.value[frames.value.length - 1]?.t_s ?? from + 1;
  return {from, to: to > from ? to : from + 1};
});
const xOf = (seconds: number): number => (seconds - span.value.from) / (span.value.to - span.value.from) * W;
/** Vision's distance in the car's unit as steps (a gap where none was logged). */
const curve = computed(() => {
  const points = (view.value?.series.vision_m ?? []).filter(([t]) => t <= span.value.to);
  const top = Math.max(1, ...points.map(([, value]) => (value ?? 0) * factor.value));
  const yOf = (value: number): number => H - 4 - value * factor.value / top * (H - 10);
  let path = '';
  let open = false;
  for (let i = 0; i < points.length; i += 1) {
    const [t, value] = points[i]!;
    const start = Math.max(t, span.value.from);
    const end = Math.min(points[i + 1]?.[0] ?? span.value.to, span.value.to);
    if (end < span.value.from) continue;
    if (value === null) {
      open = false;
      continue;
    }
    const y = yOf(value);
    path += (open ? ' V' + y.toFixed(1) : ' M' + xOf(start).toFixed(1) + ',' + y.toFixed(1)) + ' H' + xOf(end).toFixed(1);
    open = true;
  }
  return {path: path.trim()};
});
/** Where the frames the page may show are: a second never shown leaves a gap. */
const covered = computed(() => {
  const out: {x: number; width: number}[] = [];
  let start: number | null = null;
  let previous = 0;
  for (const frame of frames.value) {
    if (start !== null && frame.t_s - previous > 0.2) {
      out.push({x: xOf(start), width: Math.max(1, xOf(previous) - xOf(start))});
      start = null;
    }
    if (start === null) start = frame.t_s;
    previous = frame.t_s;
  }
  if (start !== null) out.push({x: xOf(start), width: Math.max(1, xOf(previous) - xOf(start))});
  return out;
});
const markers = computed(() => {
  const out: {key: string; x: number; label: string}[] = [];
  const item = sign.value;
  if (!item) return out;
  if (item.last_seen_s !== null) out.push({key: 'seen', x: xOf(item.last_seen_s), label: 'Last seen'});
  if (item.zero_s !== null) out.push({key: 'vision', x: xOf(item.zero_s), label: 'Vision 0'});
  if (item.mark?.t_s !== null && item.mark?.t_s !== undefined) out.push({key: item.mark.mark === 'zero' ? 'mine' : 'cant', x: xOf(item.mark.t_s), label: 'Your mark'});
  return out;
});
const cursorX = computed(() => wanted.value ? xOf(wanted.value.t_s) : -10);

const strip = ref<SVGSVGElement | null>(null);
let dragging = false;
function stripSeek(event: PointerEvent): void {
  const box = strip.value?.getBoundingClientRect();
  if (!box || !box.width) return;
  const share = Math.max(0, Math.min(1, (event.clientX - box.left) / box.width));
  seekTime(span.value.from + share * (span.value.to - span.value.from));
}
function stripDown(event: PointerEvent): void {
  dragging = true;
  strip.value?.setPointerCapture(event.pointerId);
  stripSeek(event);
}
function stripMove(event: PointerEvent): void {
  if (dragging) stripSeek(event);
}
function stripUp(): void {
  dragging = false;
}

// -- marks ---------------------------------------------------------------------------------------------------------------
const saving = ref(false);
/** A 403 on a mark (a read-only server, signed out): the buttons go, as for a read-only server. */
const refused = ref('');
const writable = computed(() => props.readOnly === false && !refused.value);
const readOnlyNote = computed(() => refused.value || (props.readOnly === true ? 'This server is read-only: the marks are shown, not changed.' : ''));
const nameOfSign = (key: string): string => {
  const found = signs.value.find((item) => item.key === key);
  return found ? 'sign ' + found.number : 'sign ' + key;
};

function failedWith(problem: unknown): void {
  if (problem instanceof ApiError && problem.status === 403) refused.value = messageOf(problem);
  toast(messageOf(problem), true);
}

async function send(kind: MarkName): Promise<void> {
  const item = sign.value;
  // The frame on screen, never the one asked for while its picture is still loading.
  const frame = kind === 'clear' ? null : onScreen.value;
  if (!item || saving.value || switching.value || !writable.value
    || (kind !== 'clear' && (!frame || !ready.value || !item.included))) return;
  saving.value = true;
  try {
    const answer = await post<{sign: Sign; warnings: {message: string}[]}>(
      API + '/drives/' + encodeURIComponent(props.route) + '/marks',
      {sign: item.key, mark: kind, segment: frame?.segment ?? null, frame: frame?.frame ?? null});
    applySign(answer.sign);
    const said = kind === 'zero' ? 'Zero marked on sign ' + item.number + ' at ' + timeText(frame?.t_s)
      : kind === 'cant_tell' ? 'Sign ' + item.number + ': can\'t tell' : 'Mark cleared on sign ' + item.number;
    toast(said + (answer.warnings.length ? '. ' + answer.warnings[0]!.message : '.'));
  } catch (problem) {
    failedWith(problem);
  } finally {
    saving.value = false;
  }
}

const MARK_WORDS: Record<string, string> = {zero: 'the zero', cant_tell: 'the can\'t tell', clear: 'the clear'};
async function undo(): Promise<void> {
  const item = sign.value;
  if (!item || saving.value || switching.value || !writable.value) return;
  saving.value = true;
  try {
    const answer = await post<{sign: Sign | null; undone: {sign: string; mark: string}}>(
      API + '/drives/' + encodeURIComponent(props.route) + '/undo', {sign: item.key});
    if (answer.sign) applySign(answer.sign);
    toast('Undone on ' + nameOfSign(answer.undone.sign) + ': ' + (MARK_WORDS[answer.undone.mark] ?? answer.undone.mark) + '.');
  } catch (problem) {
    if (problem instanceof ApiError && problem.status === 409) toast('Nothing to undo on sign ' + item.number + '.');
    else failedWith(problem);
  } finally {
    saving.value = false;
  }
}

// -- the next sign, the next drive ---------------------------------------------------------------------------------------
const remaining = computed(() => signs.value.filter(toMark).length);
const compared = computed(() => signs.value.filter((item) => item.included).length);
/** The next sign still to mark after the one shown (from the start again), else the next one compared. */
const nextKey = computed((): string | null => {
  const list = signs.value;
  const here = list.findIndex((item) => item.key === sign.value?.key);
  const after = [...list.slice(here + 1), ...list.slice(0, Math.max(0, here))];
  return after.find(toMark)?.key ?? view.value?.next ?? null;
});
function nextSign(): void {
  if (nextKey.value) openSign(nextKey.value);
}
/** The signs before and after the one asked for, in the order of the numbers at the top (a sign left out too): from
 * the one in the address, so two quick steps go two signs on even while the first is still loading. */
const here = computed(() => signs.value.findIndex((item) => item.key === (tabSign.value ?? sign.value?.key)));
const signBefore = computed((): Sign | null => here.value > 0 ? signs.value[here.value - 1] ?? null : null);
const signAfter = computed((): Sign | null => here.value >= 0 ? signs.value[here.value + 1] ?? null : null);
function stepSign(step: -1 | 1): void {
  const to = step < 0 ? signBefore.value : signAfter.value;
  if (to) openSign(to.key);
}
const nextDriveBusy = ref(false);
async function nextDrive(): Promise<void> {
  nextDriveBusy.value = true;
  try {
    const answer = await api<{drives: Drive[]}>(API + '/drives');
    const list = answer.drives.filter((item) => item.status === 'signs' && item.route !== props.route
      && item.counts.included - item.counts.zero - item.counts.cant_tell > 0);
    const after = list.find((item) => item.route < props.route) ?? list[0];
    if (after) goTo('signdistance', '', after.route, null);
    else toast('Every sign of every drive is marked.');
  } catch (problem) {
    toast(messageOf(problem), true);
  } finally {
    nextDriveBusy.value = false;
  }
}

function back(): void {
  if ((history.state as {fromList?: boolean} | null)?.fromList) history.back();
  else goTo('signdistance');
}

// -- the keys ------------------------------------------------------------------------------------------------------------
function onKey(event: KeyboardEvent): void {
  if (event.altKey || event.ctrlKey || event.metaKey || !view.value) return;
  const target = event.target instanceof HTMLElement ? event.target : null;
  const tag = target?.tagName ?? '';
  const range = target instanceof HTMLInputElement && target.type === 'range';
  if ((tag === 'INPUT' && !range) || tag === 'SELECT' || tag === 'TEXTAREA') return;
  const key = event.key;
  if (key === 'ArrowLeft' || key === 'ArrowRight') {
    // Shift goes to the sign before or after; on the slider its own keys move one frame.
    if (event.shiftKey) {
      event.preventDefault();
      stepSign(key === 'ArrowLeft' ? -1 : 1);
      return;
    }
    if (range) return;
    event.preventDefault();
    seek(index.value + (key === 'ArrowLeft' ? -1 : 1));
    return;
  }
  if (event.shiftKey) return;
  const lower = key.toLowerCase();
  if (key === ',' || key === '.') seek(index.value + (key === ',' ? -5 : 5));
  else if (lower === 'z') void send('zero');
  else if (lower === 'c') void send('cant_tell');
  else if (lower === 'u') void undo();
  else if (lower === 'n') nextSign();
  else return;
  event.preventDefault();
}
onMounted(() => {
  document.addEventListener('keydown', onKey);
  void loadDrive();
});
onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKey);
  stopPolling();
});

// -- words ---------------------------------------------------------------------------------------------------------------
const heading = computed(() => {
  const item = sign.value;
  if (!item || !view.value) return '';
  return view.value.position !== null ? 'Sign ' + view.value.position + ' of ' + view.value.of : 'Sign ' + item.number + ' · left out';
});
const facts = computed(() => {
  const current = drive.value;
  if (!current) return '';
  return [current.when ? (current.when_certain ? '' : '≈ ') + current.when : null, current.route].filter(Boolean).join(' · ');
});
/** Where the picture on screen is ("minute 0, frame 130 · 110 of 220"), "loading…" while the one asked for is not. */
function placeOf(frame: {segment: number; frame: number} | null, at: number): string {
  if (!frame) return '';
  return 'minute ' + frame.segment + ', frame ' + frame.frame + ' · ' + (at + 1) + ' of ' + frames.value.length;
}
const position = computed(() => {
  const shownHere = onScreen.value !== null;
  const text = placeOf(shownHere ? onScreen.value : wanted.value, shownHere ? shownIndex.value : index.value);
  return text && !ready.value && !failed.value ? text + ' · loading…' : text;
});
const shownTime = computed(() => timeText((onScreen.value ?? wanted.value)?.t_s));
/** The slider's own place (the frame asked for), for a screen reader. */
const sliderText = computed(() => timeText(wanted.value?.t_s) + ', ' + placeOf(wanted.value, index.value));
const markWhen = computed(() => {
  const mark = sign.value?.mark;
  if (!mark) return '';
  return [mark.reviewer ? 'by ' + mark.reviewer : null, mark.at ? mark.at.replace('T', ' ').slice(0, 16) : null].filter(Boolean).join(', ');
});
const syncNote = computed(() => {
  const mark = sign.value?.mark;
  if (!mark || !mark.sync || mark.sync === 'frame') return '';
  return 'Kept out of the medians: the video and the logs of this minute are not in sync to the frame.';
});
const visionAtMark = computed(() => {
  const metrics = sign.value?.metrics;
  if (!metrics) return '';
  const shown = screenText(metrics.vision_shown, unit.value);
  const parts = [shown === null ? 'no distance logged at your zero' : shown === '0' ? 'the screen showed no number at your zero'
    : 'the screen showed ' + shown + ' at your zero'];
  if (metrics.v_kmh !== null) parts.push(Math.round(metrics.v_kmh) + ' km/h');
  return parts.join(' · ');
});
</script>

<template>
  <section class="card rt-view sd-view">
    <div class="rt-view-head">
      <button type="button" class="link-button rt-back" @click="back"><span aria-hidden="true">‹</span> {{ guest ? 'My reviews' : 'Sign distance' }}</button>
      <div class="rt-view-name">
        <h2>{{ drive?.title ?? (!drive ? 'Loading…' : guest ? 'Drive' : 'Places unknown') }}</h2>
        <span v-if="facts">{{ facts }}</span>
      </div>
    </div>
    <p v-if="error" class="empty">{{ error }}</p>
    <p v-else-if="!drive" class="empty">Loading…</p>
    <div v-else-if="drive.status !== 'signs'" class="dr-banner" :class="'dr-' + statusTone(drive.status)" id="sd-drive-state">
      <p>{{ drive.message }}</p>
      <ul v-if="drive.unreadable_files.length" class="sd-files">
        <li v-for="file in drive.unreadable_files" :key="file.segment + file.kind">minute {{ file.segment }} {{ file.kind }}: {{ file.error }}</li>
        <li class="sd-files-note">Their permissions are fixed on the server; this page never changes them.</li>
      </ul>
      <div v-if="!guest" class="dr-banner-actions">
        <a v-if="drive.status === 'unread' || drive.status === 'unreadable'" class="btn small" :href="'/drive-review?drive=' + encodeURIComponent(route)">Drive review</a>
        <a v-if="drive.status === 'no_video'" class="btn small" href="/prepare">Drives</a>
      </div>
    </div>
    <template v-else>
      <nav class="sd-signs" aria-label="Signs of this drive">
        <button v-for="item in signs" :key="item.key" type="button" class="sd-sign-chip"
                :class="[item.included ? (item.mark?.mark === 'zero' ? 'zero' : item.mark?.mark === 'cant_tell' ? 'cant' : 'todo') : 'out']"
                :aria-current="item.key === sign?.key ? 'true' : undefined" :title="'Sign ' + item.number + ': ' + markLabel(item)
                  + (item.excluded_message ? ' (' + item.excluded_message + ')' : '')" @click="openSign(item.key)">{{ item.number }}</button>
      </nav>
      <p v-if="drive.orphans.length" class="sd-warn sd-line">{{ plural(drive.orphans.length, 'mark') }} whose sign is no longer found (the logs were
        read again): kept, out of the figures.</p>

      <p v-if="signError" class="empty">{{ signError }}</p>
      <p v-else-if="!view || !sign" class="empty">Loading the sign…</p>
      <template v-else>
        <div class="sd-sign-head">
          <h3 id="sd-sign-title" :data-sign="sign.key">{{ heading }}</h3>
          <span>{{ classText(sign.raw_class) }}<template v-if="sign.tracks > 1"> · {{ sign.tracks }} tracks</template></span>
        </div>

        <div v-if="!sign.included" class="dr-banner dr-todo" id="sd-left-out">
          <p>Left out: {{ sign.excluded_message }}</p>
        </div>
        <template v-else>
          <div class="sd-stages" :class="{two: twoCameras}">
            <div class="sd-stage" :class="{empty: !pictureUrl}" :style="ratio ? {'--sd-aspect': ratio} : undefined">
              <img v-if="pictureUrl" :src="pictureUrl" alt="The road at the frame shown" draggable="false" @load="pictureLoaded">
              <span v-if="failed" class="dr-stage-note">This frame cannot be shown.</span>
              <span v-else-if="!pictureUrl" class="dr-stage-note">Loading the frame…</span>
              <span v-if="twoCameras" class="sd-cam">Road camera</span>
            </div>
            <div v-if="twoCameras" class="sd-stage sd-stage-wide" id="sd-wide" :class="{empty: !widePictureUrl}"
                 :style="wideRatio ? {'--sd-aspect': wideRatio} : ratio ? {'--sd-aspect': ratio} : undefined">
              <img v-if="widePictureUrl" :src="widePictureUrl" alt="The same frame from the wide road camera" draggable="false"
                   @load="widePictureLoaded">
              <span v-else-if="onScreen && !failed" class="dr-stage-note">{{ hasWide(onScreen) ? 'The wide picture of this frame cannot be shown.' : 'No wide picture in this minute.' }}</span>
              <span v-else class="dr-stage-note">Loading the frame…</span>
              <span class="sd-cam">Wide camera, same frame</span>
            </div>
          </div>
          <div class="sd-stage-bar">
            <span v-if="chip" class="sd-chip-vision" :class="{zero: chip.zero}" :title="chip.title" id="sd-vision-chip">{{ chip.text }}</span>
            <span v-if="onMark" class="sd-chip-mine" :class="{cant: sign.mark?.mark === 'cant_tell'}">{{ sign.mark?.mark === 'cant_tell' ? 'Can\'t tell' : 'Your zero' }}</span>
          </div>

          <div class="sd-time">
            <svg ref="strip" class="sd-curve" :viewBox="'0 0 ' + W + ' ' + H" preserveAspectRatio="none" role="presentation"
                 @pointerdown="stripDown" @pointermove="stripMove" @pointerup="stripUp" @pointercancel="stripUp">
              <rect class="sd-curve-base" x="0" y="0" :width="W" :height="H"/>
              <rect v-for="(part, n) in covered" :key="n" class="sd-curve-kept" :x="part.x" y="0" :width="part.width" :height="H"/>
              <path v-if="curve.path" class="sd-curve-line" :d="curve.path"/>
              <line v-for="mark in markers" :key="mark.key" :class="'sd-m-' + mark.key" :x1="mark.x" :x2="mark.x" y1="0" :y2="H"/>
              <line class="sd-m-now" :x1="cursorX" :x2="cursorX" y1="0" :y2="H"/>
            </svg>
            <input type="range" class="sd-range" min="0" :max="Math.max(0, frames.length - 1)" step="1" :value="index"
                   aria-label="Frame" :aria-valuetext="sliderText"
                   @input="seek(Number(($event.target as HTMLInputElement).value))">
            <div class="rt-controls sd-controls">
              <button type="button" class="btn small sd-step" aria-label="Back five frames" title="Back five frames (,)" @click="seek(index - 5)">«</button>
              <button type="button" class="btn small sd-step" aria-label="Back one frame" title="Back one frame (←)" @click="seek(index - 1)">‹</button>
              <button type="button" class="btn small sd-step" aria-label="Forward one frame" title="Forward one frame (→)" @click="seek(index + 1)">›</button>
              <button type="button" class="btn small sd-step" aria-label="Forward five frames" title="Forward five frames (.)" @click="seek(index + 5)">»</button>
              <span class="rt-pos sd-pos" id="sd-pos"><b>{{ shownTime }}</b> · {{ position }}</span>
            </div>
            <div class="sd-jumps">
              <button v-if="sign.last_seen_s !== null" type="button" class="dr-chip sd-j-seen" @click="seekTime(sign.last_seen_s)">Last seen
                {{ timeText(sign.last_seen_s) }}</button>
              <button v-if="sign.zero_s !== null" type="button" class="dr-chip sd-j-vision" @click="seekTime(sign.zero_s)">Vision 0
                {{ timeText(sign.zero_s) }}</button>
              <button v-if="sign.mark && sign.mark.t_s !== null" type="button" class="dr-chip sd-j-mine" @click="seekTime(sign.mark.t_s)">Your
                mark {{ timeText(sign.mark.t_s) }}</button>
            </div>
          </div>
        </template>

        <div v-if="writable && (sign.included || sign.mark)" class="sd-actions" role="group" aria-label="Mark this sign">
          <button v-if="sign.included" type="button" class="btn primary" id="sd-zero" :disabled="saving || switching || !ready" @click="send('zero')">Zero here
            <kbd class="sd-kbd">Z</kbd></button>
          <button v-if="sign.included" type="button" class="btn" id="sd-cant" :disabled="saving || switching || !ready" @click="send('cant_tell')">Can't tell
            <kbd class="sd-kbd">C</kbd></button>
          <button v-if="sign.mark" type="button" class="btn" id="sd-clear" :disabled="saving || switching" @click="send('clear')">Clear</button>
          <button type="button" class="btn" id="sd-undo" :disabled="saving || switching" @click="undo">Undo <kbd class="sd-kbd">U</kbd></button>
        </div>
        <p v-else-if="readOnlyNote" class="dr-hint sd-readonly" id="sd-readonly">{{ readOnlyNote }}</p>

        <div v-if="sign.included || sign.mark" class="sd-result" id="sd-result" aria-live="polite">
          <template v-if="sign.metrics">
            <strong class="sd-late">{{ lateSentence(sign.metrics.late_s) }}</strong>
            <span>{{ visionAtMark }}</span>
            <span class="sd-second">In metres: <template v-if="sign.metrics.late_m !== null">{{ signedMetres(sign.metrics.late_m) }}.</template><template
              v-else>the speed is not in the logs there.</template> {{ sign.metrics.late_m_note }}</span>
          </template>
          <template v-else-if="sign.mark?.mark === 'cant_tell'">
            <strong>You could not tell</strong>
            <span>Marked at {{ timeText(sign.mark.t_s) }}: this sign stays out of the figures.</span>
          </template>
          <template v-else-if="sign.mark?.mark === 'zero'">
            <strong>Zero marked at {{ timeText(sign.mark.t_s) }}</strong>
          </template>
          <template v-else-if="sign.included">
            <strong>Not marked yet</strong>
            <span>Find the frame where the sign is about to leave the picture, then Zero here.</span>
          </template>
          <span v-if="markWhen" class="sd-who">{{ markWhen }}</span>
          <span v-if="syncNote" class="sd-warn">{{ syncNote }}</span>
        </div>

        <ul v-if="view.warnings.length || !view.unit_known" class="sd-notes">
          <li v-for="(warning, n) in view.warnings" :key="n">{{ warning.message }}</li>
          <li v-if="!view.unit_known">The logs do not say the screen's unit: metres.</li>
        </ul>

        <div class="sd-next">
          <div class="sd-step-signs" role="group" aria-label="Sign before or after">
            <button type="button" class="btn" id="sd-sign-before" :disabled="!signBefore" @click="stepSign(-1)"><span
              aria-hidden="true">‹</span> Sign before <kbd class="sd-kbd">⇧←</kbd></button>
            <button type="button" class="btn" id="sd-sign-after" :disabled="!signAfter" @click="stepSign(1)">Sign after <span
              aria-hidden="true">›</span> <kbd class="sd-kbd">⇧→</kbd></button>
          </div>
          <button v-if="nextKey && nextKey !== sign.key" type="button" class="btn" id="sd-next" @click="nextSign">Next to mark
            <kbd class="sd-kbd">N</kbd></button>
          <template v-if="!remaining">
            <span v-if="compared" class="sd-done">All signs of this drive are marked.</span>
            <span v-else class="dr-hint" id="sd-none">No sign of this drive can be compared.</span>
            <button type="button" class="btn" :disabled="nextDriveBusy" @click="nextDrive">Next drive ›</button>
          </template>
          <span v-else class="dr-hint">{{ remaining }} to mark</span>
          <a v-if="!guest" class="link-button sd-car" :href="'/drive-review?drive=' + encodeURIComponent(route)">What the car did here ›</a>
        </div>
        <p class="dr-hint sd-keys">Keys: ← → one frame, , . five, Shift+← → the sign before or after;<template
          v-if="writable && sign.included"> Z zero here, C can't tell,</template><template v-if="writable && (sign.included
          || sign.mark)"> U undo,</template> N next to mark.</p>
      </template>
    </template>
  </section>
</template>
