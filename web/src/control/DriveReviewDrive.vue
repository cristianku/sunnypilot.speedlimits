<script setup lang="ts">
// The player of a drive, on Speed review (SpeedReviewDrive.vue) and Regression (RegressionDrive.vue): its video
// (nnslr_tools.drive_video, the same as Results › Drive) in step with the comma's logs (GET
// /api/drive-review/drives/<route>/timeline): a time of the logs is a picture of the video through the encode index of
// every frame (drive-review.ts videoTimeOf), never a file's date. On the picture the limit the comma's vision held; under
// it the lanes (DriveReviewLanes.vue) between two times, the whole drive as a strip, and at the side what the comma had
// at the time shown. Review the signs here opens Sign review on the frame of the time shown, and Sign review links back
// (?segment=N&second=S opens the drive at that time). Without logs the video plays all the same, with what to do to get
// them:
// the logs found on the server are imported by a search, others are sent from here (a folder of the comma's segments,
// or rlog/qlog files named with their drive and segment). Each page gives its own tab and back label (tab, backLabel)
// and a choice of playback speed (rates); Speed review also a lane of its own over the others (slot "lane"), only the
// lanes a speed is judged on (simpleLanes), the switch HD (sharp: standing still, the frame at the comma's full
// definition over the video) and its controls at the top of the side (slot "side"); each follows the time shown through
// the event time and moves it with seek.
import {computed, onBeforeUnmount, onMounted, ref, shallowRef, useTemplateRef, watch} from 'vue';
import {api, messageOf, post} from '../common/api';
import {plural} from '../common/format';
import {isGuest} from '../common/guest';
import {pictureCache} from '../composables/pictures';
import {toast} from '../composables/toast';
import {aheadAt, clockOf, FPS, kphText, limitOf, logTimeOf, NO_LOGS, previousChange, rawFrameOf, readText,
  REASON_LABEL, type ScanState, scanText, secondOf, setSpeedOf, signReviewHref, type Timeline, valueAt, videoTimeOf, type VideoState, visionAtSign} from './drive-review';
import DriveReviewLanes from './DriveReviewLanes.vue';
import {ROAD_LABEL} from './roads';
import Segmented from './Segmented.vue';
import {type ControlTab, goTo} from './tabs';

const props = withDefaults(defineProps<{route: string; readOnly: boolean; tab: ControlTab; backLabel: string;
  simpleLanes?: boolean; rates?: readonly number[]; startZoom?: Zoom; sharp?: boolean}>(), {simpleLanes: false, rates: () => [], sharp: false,
  startZoom: 'one'});
const guest = isGuest();
const emit = defineEmits<{time: [seconds: number, t0Ns: number]}>();

// The squares on the picture, as on the comma: an eye for the vision's limit, a car for the car's speed.
const HUD_ICONS = {
  eye: 'M12 1C6.6 1 2.4 4.4 1 8.5 2.4 12.6 6.6 16 12 16s9.6-3.4 11-7.5C21.6 4.4 17.4 1 12 1ZM12 4.3a4.2 4.2 0 1 0 0 8.4a4.2 4.2 0 1 0 0-8.4Z'
    + 'M12 6.5a2 2 0 1 0 0 4a2 2 0 1 0 0-4Z',
  car: 'M18.92 6.01C18.72 5.42 18.16 5 17.5 5h-11c-.66 0-1.21.42-1.42 1.01L3 12v8c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h12v1c0 .55.45 1 1 1h1'
    + 'c.55 0 1-.45 1-1v-8l-2.08-5.99zM6.5 16c-.83 0-1.5-.67-1.5-1.5S5.67 13 6.5 13s1.5.67 1.5 1.5S7.33 16 6.5 16zm11 0c-.83 0-1.5-.67-1.5-1.5'
    + 's.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zM5 11l1.5-4.5h11L19 11H5z',
};
const MARKS = {
  play: 'M8 5.6v12.8a.8.8 0 0 0 1.2.7l10.3-6.4a.8.8 0 0 0 0-1.4L9.2 4.9A.8.8 0 0 0 8 5.6z',
  pause: 'M7 5h3.2v14H7zM13.8 5H17v14h-3.2z',
} as const;

// -- the drive -----------------------------------------------------------------------------------------------------------
const view = shallowRef<Timeline | null>(null);
const error = ref('');
let asked = 0;
async function read(): Promise<void> {
  const ticket = ++asked;
  try {
    const answer = await api<Timeline>('/api/drive-review/drives/' + encodeURIComponent(props.route) + '/timeline');
    if (ticket !== asked) return;
    // A cut answer (the page left while it came) reads as {message} in api(): never a drive without its scan.
    if (!answer || typeof answer.scan !== 'object' || answer.scan === null || !Array.isArray(answer.scan.kinds)) {
      throw new Error('The server sent a drive this page cannot read.');
    }
    view.value = answer;
    error.value = '';
    document.title = 'Speed limits · ' + (answer.title ?? props.backLabel);
    followScan(answer.scan);
  } catch (problem) {
    if (ticket === asked) error.value = messageOf(problem);
  }
}
const scan = computed((): ScanState | null => view.value?.scan ?? null);
const hasLogs = computed(() => view.value !== null && view.value.t0_ns !== null);
const end = computed(() => view.value?.span_s?.[1] ?? video.value?.index?.duration ?? 0);
const t0 = computed(() => view.value?.t0_ns ?? 0);

/** While a search of the logs runs, its state every 2 s; the drive again when it ends. */
let scanTimer: number | undefined;
function followScan(state: ScanState): void {
  window.clearTimeout(scanTimer);
  if (state.state !== 'queued' && state.state !== 'scanning') return;
  scanTimer = window.setTimeout(async () => {
    try {
      const answer = await api<{scan: ScanState}>('/api/drive-review/drives/' + encodeURIComponent(props.route));
      if (answer.scan.state === 'queued' || answer.scan.state === 'scanning') {
        if (view.value) view.value = {...view.value, scan: answer.scan};
        followScan(answer.scan);
      } else {
        void read();
      }
    } catch {
      followScan(state);
    }
  }, 2000);
}
async function search(): Promise<void> {
  try {
    const answer = await post<{scan: ScanState}>('/api/drive-review/drives/' + encodeURIComponent(props.route) + '/scan');
    if (view.value) view.value = {...view.value, scan: answer.scan};
    followScan(answer.scan);
  } catch (problem) {
    toast(messageOf(problem), true);
  }
}

// -- logs sent from here ----------------------------------------------------------------------------------------------------
const LOG_NAME = /(^|[-/])(rlog|qlog)(\.zst|\.bz2)?$/;
const folderInput = useTemplateRef<HTMLInputElement>('folderInput');
const filesInput = useTemplateRef<HTMLInputElement>('filesInput');
const sending = ref<{done: number; total: number} | null>(null);
async function send(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const files = [...(input.files ?? [])].filter((file) => LOG_NAME.test(file.webkitRelativePath || file.name));
  input.value = '';
  if (!files.length) {
    toast('No rlog or qlog among these files.', true);
    return;
  }
  let added = 0;
  let known = 0;
  const failed: string[] = [];
  sending.value = {done: 0, total: files.length};
  for (const file of files) {
    const name = file.webkitRelativePath || file.name;
    try {
      const answer = await api<{added: boolean}>('/api/drive-review/drives/' + encodeURIComponent(props.route) + '/logs?name='
        + encodeURIComponent(name), {method: 'POST', headers: {'Content-Type': 'application/octet-stream'}, body: file});
      if (answer.added) added += 1;
      else known += 1;
    } catch (problem) {
      failed.push(messageOf(problem));
    }
    sending.value = {done: sending.value.done + 1, total: files.length};
  }
  sending.value = null;
  toast([added ? plural(added, 'log') + ' added' : null, known ? plural(known, 'log') + ' already here' : null,
    failed.length ? failed.length + ' refused: ' + failed[0] : null].filter(Boolean).join(' · ') + '.', failed.length > 0);
  void read();
}

// -- the video -------------------------------------------------------------------------------------------------------------
const video = ref<VideoState | null>(null);
const player = useTemplateRef<HTMLVideoElement>('player');
const ready = computed(() => video.value?.state === 'ready' && Boolean(video.value.url));
let videoTimer: number | undefined;
async function askVideo(start: boolean): Promise<void> {
  window.clearTimeout(videoTimer);
  try {
    const answer = await api<VideoState>('/api/panel/drive-video/' + encodeURIComponent(props.route) + (start ? '?start=1' : ''));
    const was = video.value?.state;
    video.value = answer;
    if (answer.state === 'waiting' || answer.state === 'making') videoTimer = window.setTimeout(() => void askVideo(false), 2000);
    // A video made now: the drive again, its sync with the video in it.
    if (answer.state === 'ready' && was && was !== 'ready') void read();
  } catch {
    video.value = null;
  }
}
const segments = computed(() => view.value?.segments ?? []);
/** The picture of the time shown: its time in the video, or null where the video has none. */
const videoTime = computed(() => ready.value && hasLogs.value ? videoTimeOf(segments.value, cursor.value) : null);
/** Without logs the video's own time is the drive's (no sync to make). */
const plainVideo = computed(() => ready.value && !hasLogs.value);

// -- the time shown, playing ---------------------------------------------------------------------------------------------
const cursor = ref(0);
watch([cursor, t0], () => emit('time', cursor.value, t0.value));
const playing = ref(false);
let frame = 0;
let last = 0;
function clamp(seconds: number): number {
  return Math.min(Math.max(0, seconds), end.value || seconds);
}
function seek(seconds: number): void {
  cursor.value = clamp(seconds);
  const element = player.value;
  if (!element) return;
  const at = plainVideo.value ? cursor.value : videoTime.value;
  if (at !== null) {
    element.currentTime = at;
  } else if (!element.paused) {
    element.pause();
  }
}
/** The video asked to play at a time, not playing yet: the time waits for it. */
let starting = false;
/** Why the browser does not play the video (its error, or play() refused), shown on the picture; '' while it plays. */
const videoProblem = ref('');
const MEDIA_ERRORS: Record<number, string> = {1: 'loading was stopped', 2: 'a network error', 3: 'it could not be decoded',
  4: 'this browser does not play this video'};
function onVideoError(): void {
  const code = player.value?.error?.code ?? 0;
  videoProblem.value = 'The video does not play here: ' + (MEDIA_ERRORS[code] ?? 'unknown error') + '.';
}
function start(element: HTMLVideoElement, at: number): void {
  element.currentTime = at;
  element.playbackRate = rate.value;
  starting = true;
  void element.play()
    .catch((problem: unknown) => {
      // A pause or a seek during the start is not a problem; anything else is said, and not tried again every frame.
      const name = problem instanceof DOMException ? problem.name : '';
      if (name !== 'AbortError') videoProblem.value = 'The browser did not start the video' + (name ? ' (' + name + ')' : '') + '.';
    })
    .finally(() => { starting = false; });
}
function tick(now: number): void {
  if (!playing.value) return;
  const element = player.value;
  if (element && starting) {
    // The picture is on its way.
  } else if (element && !element.paused) {
    const at = plainVideo.value ? element.currentTime : logTimeOf(segments.value, element.currentTime);
    if (at !== null) cursor.value = at;
  } else {
    // No picture here: the logs go on at their own pace, the video takes over where it has the time again.
    cursor.value = clamp(cursor.value + (last ? (now - last) / 1000 * rate.value : 0));
    const at = plainVideo.value ? cursor.value : videoTime.value;
    if (element && at !== null && !element.ended && !videoProblem.value) start(element, at);
  }
  last = now;
  if (end.value && cursor.value >= end.value - 0.05) {
    pause();
    return;
  }
  frame = requestAnimationFrame(tick);
}
function play(): void {
  if (playing.value) return;
  zoomAt.value = null;
  if (end.value && cursor.value >= end.value - 0.05) seek(0);
  playing.value = true;
  last = 0;
  videoProblem.value = '';
  const element = player.value;
  const at = plainVideo.value ? cursor.value : videoTime.value;
  if (element && at !== null) start(element, at);
  frame = requestAnimationFrame(tick);
}
function pause(): void {
  playing.value = false;
  cancelAnimationFrame(frame);
  player.value?.pause();
}
const toggle = (): void => (playing.value ? pause() : play());
/** One frame of the video on or back, paused (Shift+← →): to the middle of that frame, as a time on the edge between two
 * shows either. Where the video has no picture of the time, a frame's time of the logs. */
function stepFrame(direction: number): void {
  pause();
  const at = plainVideo.value ? cursor.value : videoTime.value;
  if (player.value && at !== null) {
    const target = (Math.floor(at * FPS + 1e-3) + direction + 0.5) / FPS;
    const logs = plainVideo.value ? target : logTimeOf(segments.value, target);
    if (logs !== null) {
      seek(logs);
      return;
    }
  }
  seek(cursor.value + direction / FPS);
}
// -- the full definition, standing still ------------------------------------------------------------------------------
// The switch HD, where the page offers it (sharp): while the video stands still (paused, a frame on or back), the frame on
// screen at the full definition of the comma's road video (1344×760, or 2688×1520), decoded on the server from its
// fcamera.hevc, over the smaller video that plays. Its frame comes from the video's own time, never from the time of the
// logs: the frame shown is the frame cut at. A tap on it zooms in where it was touched, another tap out.
const SHARP_KEY = 'nnslr-sharp';
function storedSharp(): boolean {
  try {
    return localStorage.getItem(SHARP_KEY) === '1';
  } catch {
    return false;
  }
}
const sharpOn = ref(storedSharp());
watch(sharpOn, (on) => {
  try {
    localStorage.setItem(SHARP_KEY, on ? '1' : '0');
  } catch {
    // Storage off: the choice lasts until the page is left.
  }
});
/** The video's own time while it stands still (after a seek, at a pause); null while it plays or seeks. */
const stillAt = ref<number | null>(null);
function onStill(): void {
  const element = player.value;
  stillAt.value = element && element.paused && !element.seeking ? element.currentTime : null;
}
const sharpFrame = computed(() => {
  if (!props.sharp || !sharpOn.value || playing.value || stillAt.value === null || (!plainVideo.value && videoTime.value === null)) return null;
  return rawFrameOf(video.value?.index, stillAt.value);
});
const sharpUrl = (segment: number, frame: number): string =>
  '/api/speed-review/drives/' + encodeURIComponent(props.route) + '/frames/' + segment + '/' + frame + '.jpg';
const wantedSharp = computed(() => sharpFrame.value ? sharpUrl(sharpFrame.value.segment, sharpFrame.value.frame) : null);
const shownSharp = ref<string | null>(null);
const sharpFailed = ref(false);
const loadPicture = pictureCache();
/** One frame asked of the server at a time, the latest wanted winning; then the five on either side, so a step into the
 * next group of pictures (a GOP, decoded whole) finds it ready. */
let pumping = false;
async function pumpSharp(): Promise<void> {
  if (pumping) return;
  pumping = true;
  try {
    for (;;) {
      const url = wantedSharp.value;
      const at = sharpFrame.value;
      if (url === null || at === null || url === shownSharp.value) break;
      const ok = await loadPicture(url);
      if (wantedSharp.value !== url) continue;
      sharpFailed.value = !ok;
      if (ok) {
        shownSharp.value = url;
        for (const step of [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]) {
          if (at.frame + step >= 0 && at.frame + step < at.frames) void loadPicture(sharpUrl(at.segment, at.frame + step));
        }
      }
      break;
    }
  } finally {
    pumping = false;
  }
}
watch(wantedSharp, () => {
  sharpFailed.value = false;
  void pumpSharp();
});
const sharpShown = computed(() => wantedSharp.value !== null && shownSharp.value === wantedSharp.value);
const sharpTag = computed(() => {
  if (!props.sharp || !sharpOn.value || playing.value || !ready.value) return '';
  if (sharpShown.value) return 'HD';
  if (sharpFailed.value) return 'No HD frame here';
  return wantedSharp.value ? 'HD…' : '';
});
/** Where the sharp frame is zoomed in, in % of the picture; null: the whole picture. */
const zoomAt = ref<{x: number; y: number} | null>(null);
function zoomSharp(event: MouseEvent): void {
  if (zoomAt.value) {
    zoomAt.value = null;
    return;
  }
  const box = (event.currentTarget as HTMLElement).getBoundingClientRect();
  zoomAt.value = {x: (event.clientX - box.left) / box.width * 100, y: (event.clientY - box.top) / box.height * 100};
}

/** The speed of the playback, where the page offers a choice. */
const rate = ref(1);
const RATES = computed(() => props.rates.map((key) => ({key: String(key), label: key + '×'})));
const rateKey = computed({get: () => String(rate.value), set: (key: string) => { rate.value = Number(key); }});
watch(rate, (value) => { if (player.value) player.value.playbackRate = value; });
function onKey(event: KeyboardEvent): void {
  const target = event.target as HTMLElement | null;
  if (target && (target.closest('input, textarea, select, [contenteditable]') || event.metaKey || event.ctrlKey || event.altKey)) return;
  if (event.key === ' ') {
    event.preventDefault();
    toggle();
  } else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.preventDefault();
    const direction = event.key === 'ArrowLeft' ? -1 : 1;
    if (event.shiftKey) stepFrame(direction);
    else seek(cursor.value + direction);
  }
}
onMounted(() => window.addEventListener('keydown', onKey));
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKey);
  cancelAnimationFrame(frame);
  window.clearTimeout(videoTimer);
  window.clearTimeout(scanTimer);
});

/** How a time of the logs lines up with the video here. */
const syncText = computed((): string => {
  if (!view.value || !hasLogs.value) return ready.value ? 'Video only: no logs to line up.' : '';
  const item = segments.value.find((row) => row.from_s <= cursor.value && cursor.value <= row.to_s);
  if (!item) return 'No logs at this time.';
  if (item.precision === 'frame') return 'Video and logs in step to the frame.';
  if (item.precision === 'degraded') return 'Video and logs in step to ±' + Math.round(item.max_error_ms ?? 0) + ' ms (frames missing).';
  if (item.precision === 'no_video') return ready.value ? 'No video of this minute.' : 'Logs only.';
  return 'No frame index in this minute\'s log: video not in step.';
});

/** Sign review of this drive on the frame nearest to the time shown (without logs: by the minute of the video). */
const signHref = computed((): string => {
  if (hasLogs.value) return signReviewHref(props.route, segments.value, cursor.value);
  const minute = [...(video.value?.index?.segments ?? [])].reverse().find((item) => item.start <= cursor.value + 1e-6);
  const base = '/route/' + encodeURIComponent(props.route) + '/';
  return minute ? base + '?segment=' + minute.segment + '&second=' + Math.max(0, Math.round((cursor.value - minute.start) * 10) / 10) : base;
});

// -- the window of the lanes ------------------------------------------------------------------------------------------------
type Zoom = 'one' | 'two' | 'ten' | 'all';
const zoom = ref<Zoom>(props.startZoom);
const ZOOMS: {key: Zoom; label: string}[] = [{key: 'one', label: '1 min'}, {key: 'two', label: '2 min'}, {key: 'ten', label: '10 min'},
  {key: 'all', label: 'Drive'}];
const windowSpan = computed(() => {
  const whole = Math.max(end.value, 1);
  if (zoom.value === 'all') return whole;
  if (zoom.value === 'ten') return Math.min(whole, 600);
  if (zoom.value === 'two') return Math.min(whole, 120);
  return Math.min(whole, 60);
});
const windowFrom = ref(0);
const windowTo = computed(() => windowFrom.value + windowSpan.value);
function place(): void {
  const span = windowSpan.value;
  if (cursor.value < windowFrom.value || cursor.value > windowFrom.value + span) {
    windowFrom.value = Math.max(0, Math.min(Math.max(0, end.value - span), cursor.value - span * 0.25));
  }
}
watch(cursor, place);
watch(zoom, () => {
  windowFrom.value = Math.max(0, Math.min(Math.max(0, end.value - windowSpan.value), cursor.value - windowSpan.value / 2));
});

// -- what the comma had at the time shown ----------------------------------------------------------------------------------
const series = (name: string) => view.value?.series[name];
const at = (name: string) => valueAt(series(name), cursor.value);
/** The vision limit from its sign on, and the one waiting for its sign (visionAtSign). */
const atSign = computed(() => view.value ? visionAtSign(view.value.series) : null);
const now = computed(() => {
  if (!hasLogs.value) return null;
  const vision = limitOf(valueAt(atSign.value?.limit, cursor.value));
  const visionAhead = aheadAt(atSign.value?.ahead ?? [], cursor.value);
  const mapValid = at('map_valid');
  const mapLimit = limitOf(at('map_limit'));
  const resolver = limitOf(at('res_limit'));
  const ahead = limitOf(at('map_ahead'));
  const aheadDistance = at('map_ahead_dist');
  const reason = at('vis_reason');
  const state = at('vis_state');
  const read = readText(at('vis_raw_class'), at('vis_raw_speed'));
  const confidence = at('vis_raw_conf');
  const signDistance = at('vis_sign_dist');
  return {
    why: reason === undefined || reason === null ? 'not in the log' : reason === '' ? 'no rule yet'
      : REASON_LABEL[String(reason)] ?? String(reason),
    state: state === 'none' ? 'no general limit' : state === 'unknown' ? 'unknown' : '',
    read: at('vis_raw_class') === undefined || at('vis_raw_class') === null ? 'not in the log' : !read ? 'none yet'
      : read + (typeof confidence === 'number' ? ' · ' + Math.round(confidence * 100) + '%' : '')
      + (at('vis_raw_taken') === true ? ' · taken' : ' · dropped'),
    signAhead: typeof signDistance === 'number' && signDistance > 0 ? Math.round(signDistance) + ' m' : '',
    vision: (vision === undefined ? 'not in the log' : vision === null ? 'none held' : vision + ' km/h')
      + (visionAhead ? ' · then ' + visionAhead.kph + ' km/h in ' + Math.round(visionAhead.metres) + ' m' : ''),
    visionKph: vision ?? null,
    // The limit the brain read, with the metres to its sign, under the one still in force.
    visionAhead: visionAhead ? visionAhead.kph + ' · ' + Math.round(visionAhead.metres) + ' m' : '',
    model: typeof at('vis_model') === 'string' ? String(at('vis_model')) : '',
    map: mapValid === undefined || mapValid === null ? 'not in the log' : mapValid === false ? 'no valid limit' : kphText(mapLimit ?? null, 'none'),
    mapKph: mapValid === true ? mapLimit ?? null : null,
    // The car's own speed, on the picture: what it really drove there, next to the limits.
    carKph: typeof at('v_ego') === 'number' ? Math.round(at('v_ego') as number) : null,
    ahead: typeof ahead === 'number' && typeof aheadDistance === 'number' ? ahead + ' km/h in ' + Math.round(aheadDistance) + ' m' : '',
    road: typeof at('road_name') === 'string' ? String(at('road_name')) : '',
    resolver: resolver === undefined ? 'not in the log' : (resolver === null ? 'none' : resolver + ' km/h') + (at('res_source') ? ' · ' + String(at('res_source')) : ''),
    sla: typeof at('sla_state') === 'string' ? String(at('sla_state')) : '',
    speed: kphText(typeof at('v_ego') === 'number' ? at('v_ego') as number : null),
    set: at('v_cruise') === undefined || at('v_cruise') === null ? 'not in the log' : kphText(setSpeedOf(at('v_cruise')), 'none'),
    target: typeof at('v_target') === 'number' ? kphText(at('v_target') as number) + (at('long_active') === true ? '' : ' (openpilot not in control)')
      : 'not in the log',
    openpilot: at('sd_state') === undefined || at('sd_state') === null ? 'not in the log'
      : String(at('sd_state')) + (at('long_active') === true ? ' · speed and steering' : at('lat_active') === true ? ' · steering' : ''),
    scene: typeof at('vis_road') === 'string' && at('vis_road') ? ROAD_LABEL[at('vis_road') as keyof typeof ROAD_LABEL] ?? String(at('vis_road')) : 'not in the log',
  };
});
const change = computed(() => previousChange(atSign.value?.limit, cursor.value));
defineExpose({seek, pause});
/** The server today at the time shown: our networks' sign within a second, the map's road and limit now. */
const today = computed(() => {
  const current = view.value;
  if (!current) return null;
  let sign: string | null = null;
  for (const item of current.inference?.frames ?? []) {
    const time = secondOf(current.segments, item.segment, item.second);
    if (time !== null && Math.abs(time - cursor.value) <= 1) {
      const best = [...item.signs].sort((a, b) => b.score - a.score)[0];
      if (best) sign = best.family === 'cancellation' ? 'end of limit' : (best.value_kph ?? '?') + ' (' + best.family.replace('_', ' ') + ')';
    }
  }
  let road: string | null = null;
  for (const [segment, first, lastSecond, type, maxspeed] of current.map_today?.runs ?? []) {
    const from = secondOf(current.segments, segment, first);
    const to = secondOf(current.segments, segment, lastSecond + 1);
    if (from !== null && to !== null && from <= cursor.value && cursor.value < to) {
      // A bare number is km/h; the map's own unit otherwise ("35 mph" in the US).
      const limit = maxspeed ? ' · ' + maxspeed + (/^\d+$/.test(String(maxspeed)) ? ' km/h' : '') : ' · no limit';
      road = (type ? ROAD_LABEL[type as keyof typeof ROAD_LABEL] ?? type : 'unknown road') + limit;
    }
  }
  return {sign, road, models: current.inference ? [current.inference.cutter, current.inference.classifier].filter(Boolean).join(' + ') : ''};
});

// -- the whole drive as a strip ------------------------------------------------------------------------------------------
const strip = useTemplateRef<SVGSVGElement>('strip');
const sx = (seconds: number): number => end.value ? seconds / end.value * 1000 : 0;
let dragging = false;
function stripSeek(event: PointerEvent): void {
  const box = strip.value?.getBoundingClientRect();
  if (!box || !box.width || !end.value) return;
  seek(Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)) * end.value);
}
function stripDown(event: PointerEvent): void {
  dragging = true;
  strip.value?.setPointerCapture(event.pointerId);
  stripSeek(event);
}
const stripMove = (event: PointerEvent): void => { if (dragging) stripSeek(event); };
const stripUp = (): void => { dragging = false; };

// -- the head ------------------------------------------------------------------------------------------------------------
const facts = computed(() => {
  const current = view.value;
  if (!current) return '';
  return [current.when ? (current.when_certain ? '' : '≈ ') + current.when : null, end.value ? clockOf(end.value) + ' long' : null,
    current.scan.kinds.length ? current.scan.kinds.join(' + ') : null, props.route].filter(Boolean).join(' · ');
});
const whenNote = computed(() => view.value && view.value.when && !view.value.when_certain
  ? 'The date comes from the files\' times (no GPS fix in the logs): it may be off.' : '');
function back(): void {
  pause();
  if ((history.state as {fromList?: boolean} | null)?.fromList) history.back();
  else goTo(props.tab);
}

onMounted(async () => {
  // A folder of the comma's segments (not in Vue's types of an input).
  folderInput.value?.setAttribute('webkitdirectory', '');
  await read();
  void askVideo(true);
  seekLinked();
});
/** A link from Sign review (?segment=N&second=S): the drive opens at that picture. */
function seekLinked(): void {
  const query = new URLSearchParams(window.location.search);
  const segment = Number(query.get('segment'));
  const second = Number(query.get('second') ?? 0);
  if (query.get('segment') === null || !Number.isInteger(segment) || !Number.isFinite(second) || !view.value) return;
  const at = secondOf(view.value.segments, segment, second);
  if (at !== null) seek(Math.max(0, at - 2));
}
watch(() => props.route, () => {
  pause();
  view.value = null;
  void read().then(() => askVideo(true));
});
</script>

<template>
  <section class="card rt-view dr-view" aria-labelledby="dr-view-title">
    <div class="rt-view-head">
      <button type="button" class="link-button rt-back" @click="back"><span aria-hidden="true">‹</span> {{ backLabel }}</button>
      <div class="rt-view-name">
        <h2 id="dr-view-title">{{ view?.title ?? (guest ? 'Drive' : 'Places unknown') }}</h2>
        <span>{{ facts }}</span>
        <span v-if="whenNote" class="dr-uncertain">{{ whenNote }}</span>
      </div>
    </div>
    <p v-if="error" class="empty">{{ error }}</p>
    <p v-else-if="!view" class="empty">Loading…</p>
    <template v-else>
      <div v-if="scan && scan.state !== 'done' && scan.state !== 'no_episodes'" class="dr-banner" :class="'dr-' + scanText(scan).tone" id="dr-scan-state">
        <p>{{ scanText(scan).text }}<template v-if="scan.state === 'no_logs' || scan.state === 'unreadable_logs'"> The video below plays all the same.</template></p>
        <div v-if="!guest" class="dr-banner-actions">
          <button v-if="scan.state !== 'queued' && scan.state !== 'scanning'" type="button" class="btn small" :class="{primary: scan.state !== 'no_logs'}"
                  :disabled="readOnly" @click="search">{{ scan.state === 'no_logs' || scan.state === 'unreadable_logs' ? 'Look on the server' : 'Search the logs' }}</button>
          <button type="button" class="btn small" :disabled="readOnly || sending !== null" @click="folderInput?.click()">Send a folder of logs</button>
          <button type="button" class="btn small" :disabled="readOnly || sending !== null" @click="filesInput?.click()">Send log files</button>
        </div>
      </div>
      <p v-if="sending" class="dr-hint">Sending the logs… {{ sending.done }} of {{ sending.total }}</p>

      <div class="dr-main">
        <div class="dr-stage" :class="{empty: !ready}" @click="toggle">
          <video v-if="ready" ref="player" :src="video!.url" muted playsinline preload="auto" @loadedmetadata="seek(cursor)"
                 @error="onVideoError" @playing="videoProblem = ''" @seeking="stillAt = null" @seeked="onStill" @pause="onStill"
                 @play="stillAt = null"></video>
          <img v-if="sharpShown" :src="shownSharp!" class="dr-sharp" :class="{zoomed: zoomAt}" alt="" draggable="false"
               :style="zoomAt ? {transform: 'scale(2.5)', transformOrigin: zoomAt.x + '% ' + zoomAt.y + '%'} : undefined"
               :title="zoomAt ? 'Tap to see the whole picture' : 'Tap to zoom in here'" @click.stop="zoomSharp">
          <span v-if="sharpTag" class="dr-sharp-tag" aria-live="polite">{{ sharpTag }}</span>
          <div v-if="ready && videoProblem" class="dr-stage-note">{{ videoProblem }}</div>
          <div v-else-if="ready && hasLogs && videoTime === null" class="dr-stage-note">No picture of this time: the logs go on.</div>
          <div v-else-if="!ready" class="dr-stage-note">{{ video === null ? 'Asking for the video…' : video.state === 'making' || video.state === 'waiting'
            ? 'Making the video of the drive… ' + Math.round((video.progress ?? 0) * 100) + '%'
            : video.state === 'failed' ? 'The video could not be made: ' + (video.reason ?? 'unknown error') + '.'
            : 'No video of this drive on the server.' }}</div>
          <div v-if="now" class="dr-hud" aria-hidden="true">
            <span class="dr-box dr-box-vision" :class="{empty: now.visionKph === null, ahead: !!now.visionAhead}"><svg viewBox="0 0 24 17">
              <path :d="HUD_ICONS.eye" fill="currentColor" fill-rule="evenodd"/></svg><b>{{ now.visionKph ?? '–' }}</b>
              <small v-if="now.visionAhead" class="dr-box-ahead">{{ now.visionAhead }}</small></span>
            <span class="dr-box dr-box-map"><small>MAP</small><b>{{ now.mapKph ?? '–' }}</b></span>
            <span class="dr-box dr-box-car"><svg viewBox="2 4 20 18"><path :d="HUD_ICONS.car" fill="currentColor"/></svg>
              <b>{{ now.carKph ?? '–' }}</b></span>
          </div>
        </div>

        <div class="dr-time">
          <div class="rt-controls">
            <button type="button" class="btn rt-play" @click="toggle"><svg class="icon" viewBox="0 0 24 24" aria-hidden="true">
              <path :d="playing ? MARKS.pause : MARKS.play" fill="currentColor"/></svg>{{ playing ? 'Pause' : 'Play' }}</button>
            <button type="button" class="btn small" aria-label="Back 5 seconds" @click="seek(cursor - 5)">−5 s</button>
            <button type="button" class="btn small" aria-label="On 5 seconds" @click="seek(cursor + 5)">+5 s</button>
            <button type="button" class="btn small" aria-label="Back one frame" title="Shift+←" @click="stepFrame(-1)">−1 frame</button>
            <button type="button" class="btn small" aria-label="On one frame" title="Shift+→" @click="stepFrame(1)">+1 frame</button>
            <button v-if="sharp" type="button" class="btn small" :aria-pressed="sharpOn" aria-label="Full definition when paused"
                    title="Paused, the frame at the comma's full definition; tap the picture to zoom" @click="sharpOn = !sharpOn">HD</button>
            <button type="button" class="btn small" :disabled="!change" id="dr-last-change"
                    :title="change ? 'The vision limit changed to ' + change.kph + ' at ' + clockOf(change.at) : 'No change before'"
                    @click="change && seek(change.at - 2)">Last limit change</button>
            <Segmented v-if="RATES.length" v-model="rateKey" :options="RATES" label="Playback speed" class="dr-rate"/>
            <span class="rt-pos"><b>{{ clockOf(cursor) }}</b> of {{ clockOf(end) }}</span>
          </div>
          <p class="dr-sync">{{ syncText }}</p>
          <svg ref="strip" class="dr-strip" viewBox="0 0 1000 20" preserveAspectRatio="none" role="slider" aria-label="The whole drive"
               :aria-valuemin="0" :aria-valuemax="Math.round(end)" :aria-valuenow="Math.round(cursor)" tabindex="0"
               @pointerdown="stripDown" @pointermove="stripMove" @pointerup="stripUp" @pointercancel="stripUp">
            <rect x="0" y="7" width="1000" height="6" rx="3" class="dr-strip-base"/>
            <rect :x="sx(windowFrom)" y="0.5" :width="Math.max(2, sx(windowTo) - sx(windowFrom))" height="19" rx="3" class="dr-strip-window"/>
            <rect :x="sx(cursor) - 1.5" y="0" width="3" height="20" class="dr-strip-head"/>
          </svg>
          <div class="dr-zoom-row">
            <Segmented v-model="zoom" :options="ZOOMS" label="Time shown in the lanes" class="dr-zoom"/>
            <span class="dr-window">{{ clockOf(windowFrom) }}–{{ clockOf(windowTo) }}</span>
          </div>
          <DriveReviewLanes v-if="hasLogs" :view="view" :from="windowFrom" :to="windowTo" :cursor="cursor"
                            :simple="simpleLanes" @seek="seek">
            <template #top="scale"><slot name="lane" v-bind="scale"></slot></template>
          </DriveReviewLanes>
          <p v-else class="dr-hint">{{ NO_LOGS }}</p>
        </div>

        <aside class="dr-side">
          <slot name="side"></slot>
          <section v-if="now" class="dr-now" aria-labelledby="dr-now-title">
            <h3 id="dr-now-title">At {{ clockOf(cursor) }}, the comma had</h3>
            <dl>
              <dt>Vision limit</dt><dd>{{ now.vision }}<small v-if="now.model"> · {{ now.model }}</small></dd>
              <dt>Why</dt><dd>{{ now.why }}<small v-if="now.state"> · {{ now.state }}</small></dd>
              <dt>Last sign read</dt><dd>{{ now.read }}<small v-if="now.signAhead"> · sign ahead in {{ now.signAhead }}</small></dd>
              <dt>Map limit</dt><dd>{{ now.map }}<small v-if="now.ahead"> · then {{ now.ahead }}</small><small v-if="now.road"> · {{ now.road }}</small></dd>
              <dt>Resolver</dt><dd>{{ now.resolver }}<small v-if="now.sla"> · assist {{ now.sla }}</small></dd>
              <dt>Speed</dt><dd>{{ now.speed }}</dd>
              <dt>Set speed</dt><dd>{{ now.set }}</dd>
              <dt>Target</dt><dd>{{ now.target }}</dd>
              <dt>openpilot</dt><dd>{{ now.openpilot }}</dd>
              <dt>Road (vision)</dt><dd>{{ now.scene }}</dd>
            </dl>
            <p v-if="change" class="dr-hint">Last limit change: {{ change.from ?? '–' }} → {{ change.kph }},
              {{ clockOf(cursor - change.at) }} ago.</p>
          </section>
          <section v-if="today && (view.inference || view.map_today)" class="dr-now dr-today-card" aria-labelledby="dr-today-title">
            <h3 id="dr-today-title">On the server today</h3>
            <p class="dr-hint">Found now, not what the comma knew while driving.</p>
            <dl>
              <dt>Our networks</dt><dd>{{ view.inference ? today.sign ?? 'no speed sign here' : 'not run on this drive' }}<small
                v-if="today.models"> · {{ today.models }}</small></dd>
              <dt>Map now</dt><dd>{{ view.map_today ? today.road ?? 'not read here' : 'not read for this drive' }}</dd>
            </dl>
          </section>

          <section v-if="!guest" class="dr-signs" aria-labelledby="dr-signs-title">
            <h3 id="dr-signs-title">The signs at {{ clockOf(cursor) }}</h3>
            <template v-if="view.sign_review">
              <p class="dr-hint">Sign review opens on the frame of this moment: confirm or correct the signs there.</p>
              <a class="btn primary" id="dr-sign-review" :href="signHref">Review the signs here</a>
            </template>
            <p v-else class="dr-hint">This drive has no frames in Sign review yet: prepare it in Drives first.</p>
          </section>
        </aside>
      </div>
    </template>
    <input ref="folderInput" type="file" class="sr-only" multiple @change="send" tabindex="-1" aria-hidden="true"/>
    <input ref="filesInput" type="file" class="sr-only" multiple accept=".zst,.bz2,application/octet-stream" @change="send" tabindex="-1" aria-hidden="true"/>
  </section>
</template>
