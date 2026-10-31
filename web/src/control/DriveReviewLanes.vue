<script setup lang="ts">
// The lanes of a drive under its video (DriveReviewDrive.vue), between two times of the logs: what the comma recorded
// (its vision limit and the signs it read, the map's limit, the resolver's limit and source, the speeds, openpilot and
// the interventions, the kind of road its vision saw), what the server finds today (our networks' signs on its frames,
// the map as it is now), and the review (the episodes, the limits a person set). A stretch the log does not have is
// hatched: missing data is never a value. A tap or a drag goes to that time.
import {computed, onBeforeUnmount, onMounted, ref, useTemplateRef} from 'vue';
import {type Annotation, combine, type Episode, indexAt, limitOf, runsOf, secondOf, type Series, setSpeedOf, statusOf, type Timeline,
  type Value} from './drive-review';

export interface DraftInterval {from: number; to: number | null; text: string}

const props = defineProps<{view: Timeline; from: number; to: number; cursor: number; episode: Episode | null;
  annotations: readonly Annotation[]; draft: readonly DraftInterval[]}>();
const emit = defineEmits<{seek: [seconds: number]}>();

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

const vision = computed(() => steps(series('vis_limit'), limitStep));
const signs = computed(() => props.view.events.filter((event) => event[1] === 'sign' && event[0] >= props.from && event[0] <= props.to)
  .map((event) => ({at: event[0], text: typeof event[2] === 'number' && event[2] > 0 ? String(Math.round(event[2])) : '–'})));
const mapSeries = computed(() => combine([series('map_valid'), series('map_limit')], ([valid, limit]) =>
  valid === undefined || valid === null ? null : valid === false ? 0 : limit ?? null));
const map = computed(() => steps(mapSeries.value, limitStep));
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
const scene = computed(() => steps(series('vis_road'), (value) =>
  value === null || value === undefined ? ['dr-gap', ''] : value === '' || value === 'none' ? ['rt-unknown', ''] : ['rt-' + String(value), String(value)]));

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

// The review: the episodes, and the limits a person set (saved, and the one being edited).
const episodes = computed(() => props.view.episodes.filter((item) => item.active && item.end_s >= props.from && item.start_s <= props.to)
  .map((item) => ({id: item.id, from: Math.max(item.start_s, props.from), to: Math.min(Math.max(item.end_s, item.start_s + span.value / 200), props.to),
    current: item.id === props.episode?.id, status: statusOf(props.annotations, item.id), spot: item.channel === 'spot_check'})));
const end = computed(() => props.view.span_s?.[1] ?? props.to);
const saved = computed(() => {
  const t0 = props.view.t0_ns ?? 0;
  const out: {from: number; to: number; text: string; cls: string}[] = [];
  for (const note of props.annotations) {
    if (note.author_kind !== 'human') continue;
    const list = note.payload.intervals;
    list.forEach((interval, index) => {
      const from = (interval.start_ns - t0) / 1e9;
      const next = list[index + 1];
      const to = interval.end_ns !== null ? (interval.end_ns - t0) / 1e9 : next ? (next.start_ns - t0) / 1e9 : end.value;
      if (to < props.from || from > props.to) return;
      out.push({from: Math.max(from, props.from), to: Math.min(to, props.to),
        text: interval.limit.state === 'unknown' ? '?' : String(interval.limit.value_kph),
        cls: interval.limit.state === 'unknown' ? 'dr-unknown' : interval.certain ? 'dr-set' : 'dr-set dr-unsure'});
    });
  }
  return out;
});
const drafted = computed(() => props.draft.filter((item) => (item.to ?? end.value) >= props.from && item.from <= props.to)
  .map((item) => ({from: Math.max(item.from, props.from), to: Math.min(item.to ?? end.value, props.to), text: item.text})));

// -- a tap or a drag on the lanes goes there -----------------------------------------------------------------------------
const track = useTemplateRef<HTMLDivElement>('track');
let dragging = false;
function seekAt(event: PointerEvent): void {
  const box = ruler.value?.getBoundingClientRect();
  if (!box || box.width <= 0) return;
  const share = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
  emit('seek', props.from + share * span.value);
}
function down(event: PointerEvent): void {
  dragging = true;
  track.value?.setPointerCapture(event.pointerId);
  seekAt(event);
}
function move(event: PointerEvent): void {
  if (dragging) seekAt(event);
}
function up(event: PointerEvent): void {
  dragging = false;
  track.value?.releasePointerCapture?.(event.pointerId);
}
const textFits = (from: number, to: number, text: string): boolean => w(from, to) >= text.length * 6.5 + 6;
</script>

<template>
  <div class="dr-lanes" ref="track" @pointerdown="down" @pointermove="move" @pointerup="up" @pointercancel="up">
    <svg class="dr-defs" width="0" height="0" aria-hidden="true"><defs>
      <pattern id="dr-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="6" height="6" class="dr-hatch-back"/><line x1="1.5" y1="0" x2="1.5" y2="6" class="dr-hatch-line"/></pattern></defs></svg>
    <div class="dr-lane-ruler"><span class="dr-lane-name"></span><div ref="ruler"></div></div>
    <div class="dr-lane" title="The limit the comma's vision held (speedVisionSP); – none held; hatched: not in the log">
      <span class="dr-lane-name">Vision<small>comma</small></span>
      <svg :width="width" height="24" :viewBox="`0 0 ${width} 24`" aria-hidden="true">
        <g v-for="(run, index) in vision" :key="index">
          <rect :x="x(run.from)" y="5" :width="w(run.from, run.to)" height="17" rx="2" :class="run.cls"/>
          <text v-if="run.text && textFits(run.from, run.to, run.text)" :x="x(run.from) + w(run.from, run.to) / 2" y="17.5" class="dr-run-text">{{ run.text }}</text>
        </g>
        <path v-for="(sign, index) in signs" :key="'s' + index" :d="`M${x(sign.at) - 3.5} 0h7l-3.5 4.5z`" class="dr-sign-tick"><title>Sign read: {{ sign.text }}</title></path>
      </svg>
    </div>
    <div class="dr-lane" title="The map's limit the comma had (liveMapDataSP); – no valid limit; hatched: not in the log">
      <span class="dr-lane-name">Map<small>comma</small></span>
      <svg :width="width" height="20" :viewBox="`0 0 ${width} 20`" aria-hidden="true">
        <g v-for="(run, index) in map" :key="index">
          <rect :x="x(run.from)" y="1" :width="w(run.from, run.to)" height="17" rx="2" :class="run.cls"/>
          <text v-if="run.text && textFits(run.from, run.to, run.text)" :x="x(run.from) + w(run.from, run.to) / 2" y="13.5" class="dr-run-text">{{ run.text }}</text>
        </g>
      </svg>
    </div>
    <div class="dr-lane" title="The limit the resolver chose and its source (longitudinalPlanSP)">
      <span class="dr-lane-name">Resolver<small>comma</small></span>
      <svg :width="width" height="20" :viewBox="`0 0 ${width} 20`" aria-hidden="true">
        <g v-for="(run, index) in resolver" :key="index">
          <rect :x="x(run.from)" y="1" :width="w(run.from, run.to)" height="17" rx="2" :class="run.cls"/>
          <text v-if="run.text && textFits(run.from, run.to, run.text)" :x="x(run.from) + w(run.from, run.to) / 2" y="13.5" class="dr-run-text">{{ run.text }}</text>
        </g>
      </svg>
    </div>
    <div class="dr-lane" title="Speed: the car's (solid), the set speed (dashed), openpilot's target (dotted)">
      <span class="dr-lane-name">Speed<small>0–{{ speedTop }}</small></span>
      <svg :width="width" :height="SPEED_HEIGHT" :viewBox="`0 0 ${width} ${SPEED_HEIGHT}`" aria-hidden="true">
        <line x1="0" :x2="width" :y1="y(0)" :y2="y(0)" class="dr-axis"/>
        <path :d="targetPath" class="dr-target"/>
        <path :d="setPath" class="dr-set-speed"/>
        <path :d="egoPath" class="dr-ego"/>
      </svg>
    </div>
    <div class="dr-lane" title="openpilot: green speed and steering, blue steering only; the marks are the interventions (G accelerator, B brake, S wheel held, D disengaged, ± set speed)">
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
    <div class="dr-lane" title="The kind of road the comma's vision saw (speedVisionSP.roadType)">
      <span class="dr-lane-name">Scene<small>comma</small></span>
      <svg :width="width" height="12" :viewBox="`0 0 ${width} 12`" aria-hidden="true">
        <rect v-for="(run, index) in scene" :key="index" :x="x(run.from)" y="2" :width="w(run.from, run.to)" height="9" rx="2" :class="run.cls">
          <title>{{ run.text }}</title></rect>
      </svg>
    </div>
    <div class="dr-lane dr-lane-server" title="Found on the server today, not what the comma knew: our networks' signs on its frames (ticks), the map as it is now (strip, with its limit)">
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
    <div class="dr-lane" title="The episodes found in the logs (orange: to review, red: wrong limit, green: right limit, grey: can't tell), and the limits a person set (a question mark: the limit is unknown)">
      <span class="dr-lane-name">Review</span>
      <svg :width="width" height="30" :viewBox="`0 0 ${width} 30`" aria-hidden="true">
        <rect v-for="item in episodes" :key="item.id" :x="x(item.from)" y="1" :width="Math.max(3, w(item.from, item.to))" height="8" rx="2"
              :class="['dr-episode', item.status, {current: item.current, spot: item.spot && item.status === 'open'}]"/>
        <g v-for="(item, index) in saved" :key="'a' + index">
          <rect :x="x(item.from)" y="12" :width="w(item.from, item.to)" height="17" rx="2" :class="item.cls"/>
          <text v-if="textFits(item.from, item.to, item.text)" :x="x(item.from) + w(item.from, item.to) / 2" y="24.5" class="dr-run-text">{{ item.text }}</text>
        </g>
        <g v-for="(item, index) in drafted" :key="'d' + index">
          <rect :x="x(item.from)" y="12" :width="Math.max(2, w(item.from, item.to))" height="17" rx="2" class="dr-draft"/>
          <text v-if="textFits(item.from, item.to, item.text)" :x="x(item.from) + w(item.from, item.to) / 2" y="24.5" class="dr-run-text">{{ item.text }}</text>
        </g>
      </svg>
    </div>
    <div class="dr-cursor" :style="{left: 'calc(var(--dr-name) + ' + cursorX + 'px)'}" aria-hidden="true"></div>
  </div>
</template>
