<script setup lang="ts">
// The lanes of a drive under its video (DriveReviewDrive.vue), between two times of the logs: what the comma recorded
// (its vision limit and the signs it read, every sign read with the reason the limit holds, the map's limit, the resolver's limit and source, the speeds, openpilot and
// the interventions), what the server finds today (our networks' signs on its frames, the map as it is now), and the
// episodes found in the logs. A stretch the log does not have is hatched: missing data is never a value. A tap or a drag
// goes to that time. The slot "top" is a lane of the page that shows them (Speed review's answer, his speed and his
// blocks), first of all, drawn with the scale of the others. Simple (Speed review) keeps only the vision's limit and the
// map's under it: what the answer is checked against; his dashboard speed is the top lane's, never the car's vEgo. With
// pickMap, a tap on a limit of the map's lane (no drag) gives that limit from its start to its end (pickMap).
import {computed, onBeforeUnmount, onMounted, ref, useTemplateRef} from 'vue';
import {combine, indexAt, limitOf, readText, REASON_LABEL, runsOf, secondOf, type Series, setSpeedOf, type Timeline,
  type Value, visionAtSign} from './drive-review';

const props = defineProps<{view: Timeline; from: number; to: number; cursor: number; simple?: boolean; pickMap?: boolean}>();
const emit = defineEmits<{seek: [seconds: number]; pickMap: [from: number, to: number, kph: number]}>();

const ruler = useTemplateRef<HTMLDivElement>('ruler');
const width = ref(600);
let observer: ResizeObserver | null = null;
onMounted(() => {
  if (!ruler.value) return;
  width.value = ruler.value.clientWidth || 600;
  observer = new ResizeObserver(() => { width.value = ruler.value?.clientWidth || width.value; });
  observer.observe(ruler.value);
});
onBeforeUnmount(() => observer?.disconnect());

const span = computed(() => Math.max(1e-3, props.to - props.from));
const x = (seconds: number): number => (seconds - props.from) / span.value * width.value;
const w = (from: number, to: number): number => Math.max(0, x(to) - x(from));
const cursorX = computed(() => x(props.cursor));
const series = (name: string): Series | undefined => props.view.series[name];

/** A lane of steps: its runs with a class and a text. */
interface Run {from: number; to: number; cls: string; text: string}
function steps(source: Series | undefined, key: (value: Value) => [cls: string, text: string]): Run[] {
  return runsOf(source, props.from, props.to, (value) => key(value).join('|')).map((run) => {
    const [cls = '', text = ''] = run.value.split('|');
    return {from: run.from, to: run.to, cls, text};
  });
}
const limitStep = (value: Value): [string, string] => {
  const limit = limitOf(value);
  return limit === undefined ? ['dr-gap', ''] : limit === null ? ['dr-none', '–'] : ['dr-known', String(limit)];
};

// A new limit of the brain from its sign on, as the map's (visionAtSign): the tick on top is where it was read.
const vision = computed(() => steps(visionAtSign(props.view.series).limit, limitStep));
const signs = computed(() => props.view.events.filter((event) => event[1] === 'sign' && event[0] >= props.from && event[0] <= props.to)
  .map((event) => ({at: event[0], text: typeof event[2] === 'number' && event[2] > 0 ? String(Math.round(event[2])) : '–'})));
// What the vision read and why it holds its limit: every sign read (solid: the limit was set from it, hollow: dropped,
// too unsure or overruled) over the rule that set the limit held.
const reads = computed(() => {
  const all = combine([series('vis_raw_class'), series('vis_raw_speed'), series('vis_raw_conf'), series('vis_raw_taken')],
    (values) => typeof values[0] === 'string' && values[0] ? JSON.stringify(values) : null);
  let lastX = -Infinity;
  return all.flatMap(([at, value]) => {
    if (typeof value !== 'string' || at < props.from || at > props.to) return [];
    const [signClass, speed, confidence, taken] = JSON.parse(value) as Value[];
    const text = readText(signClass, speed);
    const short = /^[0-9]+$/.test(text) || text === 'end' ? text : '◆';
    const shown = x(at) - lastX >= short.length * 6 + 6;
    if (shown) lastX = x(at);
    const sure = typeof confidence === 'number' ? ' · ' + Math.round(confidence * 100) + '%' : '';
    return [{at, text: shown ? short : '', taken: taken === true, title: text + sure + (taken === true ? ' · taken' : ' · dropped')}];
  });
});
const reasons = computed(() => steps(series('vis_reason'), (value) =>
  value === null || value === undefined ? ['dr-gap', ''] : value === '' ? ['dr-none', ''] : ['dr-reason', REASON_LABEL[String(value)] ?? String(value)]));
const mapSeries = computed(() => combine([series('map_valid'), series('map_limit')], ([valid, limit]) =>
  valid === undefined || valid === null ? null : valid === false ? 0 : limit ?? null));
const map = computed(() => steps(mapSeries.value, limitStep));
/** The whole run of the map's limit at a time, beyond the lanes shown: null where it has no number. */
function mapRunAt(seconds: number): {from: number; to: number; kph: number} | null {
  const run = runsOf(mapSeries.value, -Infinity, Infinity, (value) => limitOf(value) ?? 0)
    .find((item) => item.from <= seconds && seconds < item.to);
  return run && run.value > 0 ? {from: run.from, to: run.to, kph: run.value} : null;
}
defineExpose({mapRunAt});
const resolverSeries = computed(() => combine([series('res_limit'), series('res_source')], ([limit, source]) =>
  limit === undefined || limit === null ? null : (typeof limit === 'number' && limit > 0.5 ? Math.round(limit) : 0) + '|' + (source ?? '')));
const resolver = computed(() => steps(resolverSeries.value, (value) => {
  if (value === null || value === undefined) return ['dr-gap', ''];
  const [limit = '0', source = ''] = String(value).split('|');
  const tone = 'dr-src-' + (['vision', 'map', 'car', 'none'].includes(source) ? source : 'other');
  return limit === '0' ? [tone + ' dr-none', '– ' + source] : [tone, limit + ' ' + source];
}));
const openpilotSeries = computed(() => combine([series('sd_enabled'), series('long_active'), series('lat_active'), series('cruise_enabled')],
  ([enabled, long, lat, cruise]) => {
    if (long === true) return 'long';
    if (lat === true) return 'lat';
    if (enabled === true) return 'on';
    if (enabled === undefined && long === undefined && lat === undefined) return null;
    if (enabled === null && long === null && lat === null) return null;
    return cruise === true ? 'cruise' : 'off';
  }));
const OP_TEXT: Record<string, string> = {long: 'speed and steering', lat: 'steering', on: 'engaged', cruise: 'car\'s cruise', off: 'off'};
const openpilot = computed(() => steps(openpilotSeries.value, (value) =>
  value === null || value === undefined ? ['dr-gap', ''] : ['dr-op-' + String(value), OP_TEXT[String(value)] ?? String(value)]));

// The speeds: the car's (solid), the set speed (dashed), openpilot's target (dotted), on a scale from 0.
const SPEED_HEIGHT = 46;
const speedTop = computed(() => {
  let top = 60;
  for (const name of ['v_ego', 'v_target', 'v_cruise']) {
    const source = series(name) ?? [];
    for (let index = Math.max(0, indexAt(source, props.from)); index < source.length; index++) {
      const [time, value] = source[index]!;
      if (time > props.to) break;
      const kph = name === 'v_cruise' ? setSpeedOf(value) : typeof value === 'number' ? value : null;
      if (kph !== null) top = Math.max(top, kph);
    }
  }
  return Math.ceil(top / 20) * 20;
});
const y = (kph: number): number => SPEED_HEIGHT - 3 - kph / speedTop.value * (SPEED_HEIGHT - 8);
function path(source: Series | undefined, value: (raw: Value) => number | null, stepped: boolean): string {
  if (!source) return '';
  let out = '';
  let open = false;
  let last: number | null = null;
  for (let index = Math.max(0, indexAt(source, props.from)); index < source.length; index++) {
    const [time, raw] = source[index]!;
    if (time > props.to) break;
    const kph = value(raw);
    if (kph === null) {
      open = false;
      last = null;
      continue;
    }
    const px = x(Math.max(time, props.from)).toFixed(1);
    const py = y(kph).toFixed(1);
    if (!open) out += 'M' + px + ' ' + py;
    else if (stepped && last !== null) out += 'H' + px + 'V' + py;
    else out += 'L' + px + ' ' + py;
    open = true;
    last = kph;
  }
  if (open && stepped && last !== null) out += 'H' + x(props.to).toFixed(1);
  return out;
}
const number = (raw: Value): number | null => typeof raw === 'number' ? raw : null;
const egoPath = computed(() => path(series('v_ego'), number, false));
const setPath = computed(() => path(series('v_cruise'), (raw) => setSpeedOf(raw), true));
const targetPath = computed(() => path(series('v_target'), (raw) => typeof raw === 'number' && raw > 0 ? raw : null, false));

// The interventions and what openpilot said: a mark each, the clues of every episode of the drive.
const MARK: Record<string, string> = {gas: 'G', brake: 'B', steer: 'S', steer_contact: 's', set_speed_manual: '±', set_speed_auto: '±',
  set_speed_unknown: '±', disengage_user: 'D', disengage_auto: 'D', disengage_unknown: 'D'};
const marks = computed(() => {
  const seen = new Set<string>();
  const out: {at: number; text: string; cls: string}[] = [];
  const t0 = props.view.t0_ns ?? 0;
  for (const episode of props.view.episodes) {
    if (!episode.active) continue;
    for (const clue of episode.clues) {
      const at = (clue.t_ns - t0) / 1e9;
      const key = clue.kind + '@' + clue.t_ns;
      if (!(clue.kind in MARK) || at < props.from || at > props.to || seen.has(key)) continue;
      seen.add(key);
      out.push({at, text: MARK[clue.kind]!, cls: 'dr-m-' + clue.kind});
    }
  }
  return out;
});

// The server today: our networks' speed signs on its frames, and the map's road type and limit now.
const inferred = computed(() => {
  // A number over its tick when there is room for it; a run of ticks reading the same keeps the first.
  let lastX = -Infinity;
  let lastText = '';
  return (props.view.inference?.frames ?? []).flatMap((frame) => {
    const at = secondOf(props.view.segments, frame.segment, frame.second);
    if (at === null || at < props.from || at > props.to) return [];
    const best = [...frame.signs].sort((a, b) => b.score - a.score)[0];
    if (!best) return [];
    const text = best.family === 'cancellation' ? 'end' : String(best.value_kph ?? '?');
    const shown = x(at) - lastX >= text.length * 6 + 6 && (text !== lastText || x(at) - lastX >= 40);
    if (shown) {
      lastX = x(at);
      lastText = text;
    }
    return [{at, text: shown ? text : ''}];
  });
});
const mapToday = computed(() => (props.view.map_today?.runs ?? []).flatMap(([segment, first, last, type, maxspeed]) => {
  const from = secondOf(props.view.segments, segment, first);
  const to = secondOf(props.view.segments, segment, last + 1);
  if (from === null || to === null || to < props.from || from > props.to) return [];
  return [{from: Math.max(from, props.from), to: Math.min(to, props.to), cls: 'rt-' + (type ?? 'unknown'),
    text: maxspeed ? String(maxspeed) : ''}];
}));

// The episodes, by their kind: the vision disagreeing with the map, the driver intervening, or both.
const episodes = computed(() => props.view.episodes.filter((item) => item.active && item.end_s >= props.from && item.start_s <= props.to)
  .map((item) => ({id: item.id, from: Math.max(item.start_s, props.from), to: Math.min(Math.max(item.end_s, item.start_s + span.value / 200), props.to),
    channel: item.channel})));

// -- a tap or a drag on the lanes goes there -----------------------------------------------------------------------------
const track = useTemplateRef<HTMLDivElement>('track');
let dragging = false;
function secondAt(event: PointerEvent): number | null {
  const box = ruler.value?.getBoundingClientRect();
  if (!box || box.width <= 0) return null;
  return props.from + Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)) * span.value;
}
function seekAt(event: PointerEvent): void {
  const seconds = secondAt(event);
  if (seconds !== null) emit('seek', seconds);
}
/** Where a tap on a limit of the map's lane began: it gives the limit only if the pointer goes up where it went down. */
let picked: {x: number; seconds: number} | null = null;
function down(event: PointerEvent): void {
  dragging = true;
  const seconds = secondAt(event);
  picked = props.pickMap && seconds !== null && (event.target as Element | null)?.closest?.('.dr-pick')
    ? {x: event.clientX, seconds} : null;
  track.value?.setPointerCapture(event.pointerId);
  seekAt(event);
}
function move(event: PointerEvent): void {
  if (dragging) seekAt(event);
}
function up(event: PointerEvent): void {
  dragging = false;
  track.value?.releasePointerCapture?.(event.pointerId);
  const run = picked && event.type === 'pointerup' && Math.abs(event.clientX - picked.x) < 6 ? mapRunAt(picked.seconds) : null;
  picked = null;
  if (run) emit('pickMap', run.from, run.to, run.kph);
}
const textFits = (from: number, to: number, text: string): boolean => w(from, to) >= text.length * 6.5 + 6;
</script>

<template>
  <div class="dr-lanes" ref="track" @pointerdown="down" @pointermove="move" @pointerup="up" @pointercancel="up">
    <svg class="dr-defs" width="0" height="0" aria-hidden="true"><defs>
      <pattern id="dr-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="6" height="6" class="dr-hatch-back"/><line x1="1.5" y1="0" x2="1.5" y2="6" class="dr-hatch-line"/></pattern></defs></svg>
    <div class="dr-lane-ruler"><span class="dr-lane-name"></span><div ref="ruler"></div></div>
    <slot name="top" :x="x" :w="w" :width="width" :from="from" :to="to" :fits="textFits"></slot>
    <div class="dr-lane" title="The limit the comma's vision held (speedVisionSP), a new one of the brain from its sign on (the tick: where it was read); – none held; hatched: not in the log">
      <span class="dr-lane-name">Vision<small>comma</small></span>
      <svg :width="width" height="24" :viewBox="`0 0 ${width} 24`" aria-hidden="true">
        <g v-for="(run, index) in vision" :key="index">
          <rect :x="x(run.from)" y="5" :width="w(run.from, run.to)" height="17" rx="2" :class="run.cls"/>
          <text v-if="run.text && textFits(run.from, run.to, run.text)" :x="x(run.from) + w(run.from, run.to) / 2" y="17.5" class="dr-run-text">{{ run.text }}</text>
        </g>
        <path v-for="(sign, index) in signs" :key="'s' + index" :d="`M${x(sign.at) - 3.5} 0h7l-3.5 4.5z`" class="dr-sign-tick"><title>Sign read: {{ sign.text }}</title></path>
      </svg>
    </div>
    <div v-if="!simple" class="dr-lane" title="The signs the comma's vision read (solid: the limit was set from it; hollow: dropped, too unsure or overruled), over the reason its limit is what it is (speedVisionSP)">
      <span class="dr-lane-name">Reads<small>comma</small></span>
      <svg :width="width" height="30" :viewBox="`0 0 ${width} 30`" aria-hidden="true">
        <g v-for="(read, index) in reads" :key="'r' + index" :class="['dr-read', {dropped: !read.taken}]">
          <line :x1="x(read.at)" :x2="x(read.at)" y1="10" y2="15"/>
          <text v-if="read.text" :x="x(read.at)" y="8.5">{{ read.text }}</text>
          <title>{{ read.title }}</title>
        </g>
        <g v-for="(run, index) in reasons" :key="'w' + index">
          <rect :x="x(run.from)" y="17" :width="w(run.from, run.to)" height="12" rx="2" :class="run.cls"/>
          <text v-if="run.text && textFits(run.from, run.to, run.text)" :x="x(run.from) + w(run.from, run.to) / 2" y="26.5" class="dr-run-text dr-today-text">{{ run.text }}</text>
        </g>
      </svg>
    </div>
    <div class="dr-lane" :title="'The map\'s limit the comma had (liveMapDataSP); – no valid limit; hatched: not in the log'
      + (pickMap ? '. Tap a limit: a block with it, from its start to its end' : '')">
      <span class="dr-lane-name">Map<small>comma</small></span>
      <svg :width="width" height="20" :viewBox="`0 0 ${width} 20`" aria-hidden="true">
        <g v-for="(run, index) in map" :key="index">
          <rect :x="x(run.from)" y="1" :width="w(run.from, run.to)" height="17" rx="2"
                :class="[run.cls, {'dr-pick': pickMap && run.cls === 'dr-known'}]"/>
          <text v-if="run.text && textFits(run.from, run.to, run.text)" :x="x(run.from) + w(run.from, run.to) / 2" y="13.5" class="dr-run-text">{{ run.text }}</text>
        </g>
      </svg>
    </div>
    <div v-if="!simple" class="dr-lane" title="The limit the resolver chose and its source (longitudinalPlanSP)">
      <span class="dr-lane-name">Resolver<small>comma</small></span>
      <svg :width="width" height="20" :viewBox="`0 0 ${width} 20`" aria-hidden="true">
        <g v-for="(run, index) in resolver" :key="index">
          <rect :x="x(run.from)" y="1" :width="w(run.from, run.to)" height="17" rx="2" :class="run.cls"/>
          <text v-if="run.text && textFits(run.from, run.to, run.text)" :x="x(run.from) + w(run.from, run.to) / 2" y="13.5" class="dr-run-text">{{ run.text }}</text>
        </g>
      </svg>
    </div>
    <div v-if="!simple" class="dr-lane" title="Speed: the car's (solid), the set speed (dashed), openpilot's target (dotted)">
      <span class="dr-lane-name">Speed<small>0–{{ speedTop }}</small></span>
      <svg :width="width" :height="SPEED_HEIGHT" :viewBox="`0 0 ${width} ${SPEED_HEIGHT}`" aria-hidden="true">
        <line x1="0" :x2="width" :y1="y(0)" :y2="y(0)" class="dr-axis"/>
        <path :d="targetPath" class="dr-target"/>
        <path :d="setPath" class="dr-set-speed"/>
        <path :d="egoPath" class="dr-ego"/>
      </svg>
    </div>
    <div v-if="!simple" class="dr-lane" title="openpilot: green speed and steering, blue steering only; the marks are the interventions (G accelerator, B brake, S wheel held, D disengaged, ± set speed)">
      <span class="dr-lane-name">openpilot<small>comma</small></span>
      <svg :width="width" height="26" :viewBox="`0 0 ${width} 26`" aria-hidden="true">
        <rect v-for="(run, index) in openpilot" :key="index" :x="x(run.from)" y="15" :width="w(run.from, run.to)" height="9" rx="2" :class="run.cls">
          <title>{{ run.text }}</title></rect>
        <g v-for="(mark, index) in marks" :key="'m' + index" :class="['dr-mark', mark.cls]">
          <line :x1="x(mark.at)" :x2="x(mark.at)" y1="11" y2="26"/>
          <text :x="x(mark.at)" y="10">{{ mark.text }}</text>
        </g>
      </svg>
    </div>
    <div v-if="!simple" class="dr-lane dr-lane-server" title="Found on the server today, not what the comma knew: our networks' signs on its frames (ticks), the map as it is now (strip, with its limit)">
      <span class="dr-lane-name">Server<small>today</small></span>
      <svg :width="width" height="30" :viewBox="`0 0 ${width} 30`" aria-hidden="true">
        <g v-for="(sign, index) in inferred" :key="'i' + index" class="dr-inferred">
          <line :x1="x(sign.at)" :x2="x(sign.at)" y1="10" y2="15"/>
          <text v-if="sign.text" :x="x(sign.at)" y="8.5">{{ sign.text }}</text>
        </g>
        <g v-for="(run, index) in mapToday" :key="'t' + index">
          <rect :x="x(run.from)" y="17" :width="w(run.from, run.to)" height="12" rx="2" :class="run.cls" class="dr-today"/>
          <text v-if="run.text && textFits(run.from, run.to, run.text)" :x="x(run.from) + w(run.from, run.to) / 2" y="26.5" class="dr-run-text dr-today-text">{{ run.text }}</text>
        </g>
      </svg>
    </div>
    <div v-if="!simple" class="dr-lane" title="The episodes found in the logs (orange: the vision disagreed with the map, blue: the driver intervened, red: both at once)">
      <span class="dr-lane-name">Episodes</span>
      <svg :width="width" height="30" :viewBox="`0 0 ${width} 30`" aria-hidden="true">
        <rect v-for="item in episodes" :key="item.id" :x="x(item.from)" y="8" :width="Math.max(3, w(item.from, item.to))" height="14" rx="2"
              :class="['dr-episode', item.channel]"/>
      </svg>
    </div>
    <div class="dr-cursor" :style="{left: 'calc(var(--dr-name) + ' + cursorX + 'px)'}" aria-hidden="true"></div>
  </div>
</template>
