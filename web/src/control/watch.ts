// The Watch panel of the Evaluate tab (WatchSection.vue): the frames of one route of an evaluation played like a
// video, with the reviewed boxes and the cutter's boxes over each. Here: the frames asked of the server
// (GET /api/panel/runs/<run>/frames), what each frame shows at a score of the cutter, and the words for the labels.
import {ref, shallowRef, watch, type Ref} from 'vue';
import {api, messageOf} from '../common/api';
import type {PixelBox, WatchBox, WatchBoxRow, WatchFrame, WatchLevel, WatchPartition, WatchPayload,
             WatchPayloadRows} from '../common/panel-types';

export const NOT_A_SIGN = 'not_a_sign';
/** The scores of the slider (as nnslr_tools.evaluation.PER_FRAME_STEPS). */
export const SCORE_STEPS: readonly number[] = Array.from({length: 10}, (_, step) => (step + 1) * 5 / 100);
/** Frames per second at 1×: the frames were taken one a second, as the recording is sampled. */
export const SPEEDS = [0.5, 1, 2, 4] as const;
export type Speed = typeof SPEEDS[number];
export type FrameFilter = 'all' | 'signs' | 'misses';

/** A box row of the server as an object. */
export function boxOf(row: WatchBoxRow): WatchBox {
  const [x1, y1, x2, y2, score, match] = row;
  const item: WatchBox = {box: [x1, y1, x2, y2], score, match};
  if (row.length === 8) {
    item.label = row[6];
    item.confidence = row[7];
  }
  return item;
}

/** The frames of one route of a report, asked again when the run, report, partition, route or version change. */
export function useWatchFrames(run: Ref<string>, model: Ref<string | null>, partition: Ref<WatchPartition>,
                               route: Ref<string | null>, version: Ref<string | null>) {
  const payload = shallowRef<WatchPayload | null>(null);
  const loading = ref(false);
  const error = ref('');
  let asked = 0;
  watch([run, model, partition, route, version], async ([name, chosen, part, wanted]) => {
    if (chosen === null) {
      payload.value = null;
      return;
    }
    const ticket = ++asked;
    loading.value = true;
    const query = new URLSearchParams({model: chosen, partition: part});
    if (wanted) query.set('route', wanted);
    try {
      const answer = await api<WatchPayloadRows>('/api/panel/runs/' + encodeURIComponent(name) + '/frames?' + query.toString());
      if (ticket === asked) {
        payload.value = {...answer, frames: answer.frames.map((frame) => ({...frame, boxes: frame.boxes.map(boxOf)}))};
        error.value = '';
      }
    } catch (problem) {
      if (ticket === asked) {
        error.value = messageOf(problem);
        payload.value = null;
      }
    } finally {
      if (ticket === asked) loading.value = false;
    }
  }, {immediate: true});
  return {payload, loading, error};
}

/** The scores the slider offers for a report: the steps from the lowest box it kept (fixed for older chain reports). */
export function scoreSteps(payload: WatchPayload): number[] {
  if (payload.level === 'none') return [];
  if (payload.level === 'verdicts') return payload.main_score === null ? [] : [payload.main_score];
  const floor = payload.min_score ?? SCORE_STEPS[0] ?? 0.05;
  const steps = SCORE_STEPS.filter((step) => step >= floor - 1e-9);
  if (payload.level === 'above' && (steps.length === 0 || Math.abs((steps[0] ?? 0) - floor) > 1e-9)) steps.unshift(floor);
  return steps;
}

/** The step nearest a score. */
export function nearestStep(steps: readonly number[], score: number | null): number {
  if (steps.length === 0) return 0;
  const wanted = score ?? 0.1;
  let best = 0;
  steps.forEach((step, index) => {
    if (Math.abs(step - wanted) < Math.abs((steps[best] ?? 0) - wanted)) best = index;
  });
  return best;
}

export type BoxOutcome = 'found' | 'junk' | 'rejected' | 'survives';

export interface DrawnBox {
  box: PixelBox;
  score: number;
  outcome: BoxOutcome;
  /** The classifier's answer (a chain), and whether it reads the sign right. */
  label: string | null;
  right: boolean | null;
}

export interface DrawnTruth {
  box: PixelBox;
  family: string | null;
  /** Found by a box of the cutter at this score; null when the report does not say. */
  found: boolean | null;
  /** What the classifier called it once found (a chain). */
  verdict: string | null;
  right: boolean | null;
}

export interface FrameView {
  truth: DrawnTruth[];
  boxes: DrawnBox[];
  signs: number;
  found: number | null;
  missed: number | null;
  /** Boxes of the cutter on no reviewed box, drawn or not; null when the report does not say. */
  junk: number | null;
  /** Of those, the ones the classifier did not reject (a chain). */
  surviving: number | null;
  /** False boxes counted but not kept in the report (beyond its boxes per frame). */
  hidden: number;
}

/** What a frame shows at a score of the cutter. */
export function frameView(frame: WatchFrame, level: WatchLevel, score: number, steps: readonly number[] | null,
                          chain: boolean): FrameView {
  const kept = level === 'all' || level === 'above' ? frame.boxes.filter((item) => item.score >= score) : frame.boxes;
  const byTruth = new Map<number, WatchBox>();
  for (const item of kept) if (item.match >= 0 && !byTruth.has(item.match)) byTruth.set(item.match, item);
  const known = level === 'all' || level === 'above' || level === 'verdicts';
  const truth = frame.truth.map((item, index): DrawnTruth => {
    if (level === 'verdicts') {
      const verdict = item.found ? item.verdict ?? null : null;
      return {box: item.box, family: item.family, found: item.found ?? false, verdict,
              right: verdict === null ? null : verdict === item.label};
    }
    const match = byTruth.get(index);
    const verdict = chain && match ? match.label ?? null : null;
    return {box: item.box, family: item.family, found: known ? match !== undefined : null, verdict,
            right: verdict === null ? null : verdict === item.label};
  });
  const boxes = kept.map((item): DrawnBox => {
    const truthLabel = item.match >= 0 ? frame.truth[item.match]?.label ?? null : null;
    let outcome: BoxOutcome = item.match >= 0 ? 'found' : 'junk';
    if (chain && outcome === 'junk') outcome = item.label === NOT_A_SIGN ? 'rejected' : 'survives';
    return {box: item.box, score: item.score, outcome, label: chain ? item.label ?? null : null,
            right: chain && item.match >= 0 && item.label ? item.label === truthLabel : null};
  });
  const found = known ? truth.filter((item) => item.found).length : null;
  const drawnJunk = boxes.filter((item) => item.outcome !== 'found').length;
  let junk: number | null = level === 'verdicts' ? null : level === 'none' ? null : drawnJunk;
  let surviving: number | null = chain ? boxes.filter((item) => item.outcome === 'survives').length : null;
  const step = steps ? steps.findIndex((value) => Math.abs(value - score) < 1e-9) : -1;
  if (level === 'all' && step >= 0) {
    const counted = frame.counts?.[step];
    if (counted !== undefined && found !== null) junk = Math.max(drawnJunk, counted - boxes.filter((item) => item.outcome === 'found').length);
    const left = frame.surviving?.[step];
    if (chain && left !== undefined) surviving = left;
  }
  return {truth, boxes, signs: truth.length, found, missed: found === null ? null : truth.length - found, junk, surviving,
          hidden: junk === null ? 0 : Math.max(0, junk - drawnJunk)};
}

/** A label of the classifier, short enough to sit on a box: "80", "end", "zone 30", "give way", "not a sign". */
export function shortLabel(label: string | null | undefined): string {
  if (!label) return '';
  if (label === NOT_A_SIGN) return 'not a sign';
  const [family = '', value = '', hours = ''] = label.split(':');
  const shown = value === 'unknown' ? '?' : value.replace(/_/g, ' ');
  switch (family) {
    case 'maximum_speed': return value === 'unknown' ? 'speed ?' : shown || 'speed';
    case 'maximum_speed_hours': return shown + (hours ? ', hours ' + hours : ' with hours');
    case 'maximum_speed_wet': return shown + ' when wet';
    case 'cancellation': return 'end' + (shown ? ' ' + shown : '');
    case 'zone': return 'zone' + (shown ? ' ' + shown : '');
    case 'variable_display': return 'variable' + (shown ? ' ' + shown : '');
    case 'other_sign': return shown || 'other';
    case 'unreadable': return 'unreadable';
    default: return label.replace(/_/g, ' ');
  }
}

/** The colour of a reviewed box by family, as the review page draws it (shared.css --ov-*). */
export function familyColor(family: string | null): string {
  if (family === 'maximum_speed' || family === 'zone' || family === 'variable_display') return 'var(--ov-speed)';
  if (family === 'cancellation') return 'var(--ov-cancel)';
  if (family === 'other_sign') return 'var(--ov-other)';
  return 'var(--ov-none)';
}

/** The frames a filter keeps, by their index in the route, at a score. */
export function keptFrames(frames: readonly WatchFrame[], views: readonly FrameView[], filter: FrameFilter): number[] {
  const kept: number[] = [];
  frames.forEach((_, index) => {
    const view = views[index];
    if (!view) return;
    if (filter === 'signs' && view.signs === 0) return;
    if (filter === 'misses' && !view.missed) return;
    kept.push(index);
  });
  return kept;
}

/** "17/24" -> "segment 17 · image 24". */
export function frameName(frame: WatchFrame): string {
  if (frame.segment === null) return 'frame';
  return 'segment ' + frame.segment + (frame.index === null ? '' : ' · image ' + frame.index);
}
