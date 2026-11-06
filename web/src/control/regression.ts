// Models › Regression (nnslr_tools.regression): the certified drives, the models he lit and the curves of a drive. The
// switches are one choice for the whole tab, kept in this browser: what the next run computes is what a drive opened shows.
import {ref, watch} from 'vue';
import type {SpeedRow} from './speed-review';

export type ResultState = 'done' | 'missing' | 'failed' | 'no_wide' | 'more_wide' | 'no_frames';
export interface RegressionRow extends SpeedRow {
  certified: {at: string; who: string} | null;
  /** Segments with the wide camera, segments of the drive. */
  wide: [number, number];
  frames?: boolean;
  /** Seconds with his speed in the part of the drive the brains trained on; null before the split exists. */
  brain_learned_s?: number | null;
  /** Segments of the drive in openpilot.distill's data. */
  distill_segments?: number;
  results?: Record<string, ResultState>;
}
export interface DrivingModel {key: string; label: string; detail: string; needs_wide: boolean}
export interface BrainModel {key: string; label: string; detail: string; in_use: boolean}
export interface RegressionPayload {routes: RegressionRow[]; models: DrivingModel[]; brains: BrainModel[]; horizons: number[]; horizon: number}

export interface Score {seconds: number; mean_kph: number; abs_kph: number; above_s: number}
export interface Curve {key: string; label: string; kind: 'driving' | 'brain' | 'car'; points: [number, number][]; score: Score | null}
export interface RegressionDrive {
  route: string;
  horizon: number;
  brain: string | null;
  own: [number, number][];
  curves: Curve[];
  failures: {key: string; label: string; error: string}[];
  hole_s: number;
  too_fast_kph: number;
  certified: {at: string; who: string} | null;
}

export const brainKey = (run: string): string => 'brain:' + run;

/** The models lit, by their key in the results ("stock", "run:<folder>", "brain:<run>", "car:stock"); null: none chosen yet. */
export const lit = ref<string[] | null>(JSON.parse(localStorage.getItem('nnslr-regression-lit') ?? 'null') as string[] | null);
watch(lit, (keys) => localStorage.setItem('nnslr-regression-lit', JSON.stringify(keys)));
/** Small, Big and the brain in use, before he chose. */
export function firstChoice(brains: BrainModel[]): string[] {
  const brain = brains.find((item) => item.in_use) ?? brains[0];
  return ['stock', 'teacher', ...(brain ? [brainKey(brain.key)] : [])];
}
export function toggle(key: string): void {
  const keys = lit.value ?? [];
  lit.value = keys.includes(key) ? keys.filter((item) => item !== key) : [...keys, key];
}

/** A curve's mark: the colour follows the model, whatever else is lit. Three slots for the distilled runs and a dash after
 * them, never a colour used twice alone; "with Smartspeed" is its model's colour, dotted. */
export interface Mark {color: string; dash: string}
const FIXED: Record<string, number> = {stock: 1, teacher: 2, logged: 3};
const DASHES = ['', '8 4', '3 3'];
export function markOf(key: string, models: DrivingModel[], brains: BrainModel[]): Mark {
  if (key.startsWith('car:')) return {color: markOf(key.slice(4), models, brains).color, dash: '1 4'};
  if (key.startsWith('brain:')) {
    const index = Math.max(0, brains.findIndex((item) => brainKey(item.key) === key));
    return {color: 'var(--rg-4)', dash: DASHES[index % DASHES.length] ?? ''};
  }
  const slot = FIXED[key];
  if (slot !== undefined) return {color: `var(--rg-${slot})`, dash: ''};
  const index = Math.max(0, models.filter((item) => FIXED[item.key] === undefined).findIndex((item) => item.key === key));
  return {color: `var(--rg-${5 + index % 3})`, dash: DASHES[Math.floor(index / 3) % DASHES.length] ?? ''};
}

/** The value of a curve at a second, null in a hole. */
export function valueAt(points: [number, number][], second: number, hole: number): number | null {
  let low = 0;
  let high = points.length - 1;
  if (high < 0) return null;
  while (low < high) {
    const middle = (low + high) >> 1;
    if ((points[middle]?.[0] ?? 0) < second) low = middle + 1;
    else high = middle;
  }
  const after = points[low];
  const before = points[low - 1];
  const near = before && after && second - before[0] < after[0] - second ? before : after;
  return near && Math.abs(near[0] - second) <= hole ? near[1] : null;
}

/** The SVG path of a curve between two times; a hole in the data stays a hole. */
export function pathOf(points: [number, number][], from: number, to: number, hole: number, x: (second: number) => number,
                       y: (kph: number) => number): string {
  let path = '';
  let last: number | null = null;
  for (const [at, kph] of points) {
    if (at < from - hole || at > to + hole) continue;
    path += `${last === null || at - last > hole ? 'M' : 'L'}${x(at).toFixed(1)} ${y(kph).toFixed(1)}`;
    last = at;
  }
  return path;
}

const signed = (kph: number): string => (kph > 0 ? '+' : kph < 0 ? '−' : '') + Math.abs(kph).toFixed(1);
/** A curve against his speed, in words. */
export function scoreText(score: Score | null, tooFast: number): string {
  if (!score) return 'Never next to your speed';
  const fast = score.above_s ? ` · ${score.above_s} s over by ${tooFast}+` : '';
  return `${signed(score.mean_kph)} km/h on average · ${score.abs_kph.toFixed(1)} off${fast}`;
}
