// Models › Drive (nnslr_tools.simulation): a recorded drive replayed through the car's own vision loop with a set of
// networks, to see on its video what the comma's screen would show, and the drives certified as right, which a new set is
// compared with (the regression test). Here: the types of /api/simulation; the set chosen, one choice for the whole tab,
// and the switch of the boxes, both kept in this browser; what the comma's two squares show at a time, from the step
// series of a replay or of the comma's own log of that day (the same names, so one code draws both); SignNet's boxes of
// a time; where the comma's screen goes on the picture; and the regression test in words.
import {ref, watch} from 'vue';
import {plural, sentence} from '../common/format';
import {FPS, type Series, type Value, valueAt} from './drive-review';

/** A set of networks a drive is replayed with (simulation.SETS): the release the comma downloads, or the versions in
 * use on Models. */
export type SetKey = 'car' | 'in-use';
export const SET_KEYS: readonly SetKey[] = ['car', 'in-use'];
export const SET_LABEL: Record<SetKey, string> = {car: 'On the car', 'in-use': 'In use'};
/** What the squares of a drive opened show: what the comma logged that day, a replay with a set, or the certified one. */
export type Source = 'comma' | SetKey | 'certified';
export const isSet = (source: Source): source is SetKey => source === 'car' || source === 'in-use';

/** A network of a set: its role on the car (finder, classifier, decider, brain), its file and its version of Models. */
export interface SetNetwork {
  role: string;
  file: string;
  id: string;
  sha256: string;
}
/** A set as it is now (simulation.set_states). ready false: reason says why a replay cannot use it as it is. */
export interface SetState {
  label: string;
  name: string;
  networks: SetNetwork[];
  ready: boolean;
  reason: string;
  /** In use only: the versions in use on Models by role, which Compute exports first when they changed. */
  wanted?: Record<string, string>;
}
export type SetStates = Partial<Record<SetKey, SetState>>;

/** A drive's replay with a set. current false: made with other networks, car code or settings than today's. */
export interface SetResult {
  made_at: string | null;
  name: string;
  frames: number;
  current: boolean;
}
/** A certified drive: who certified which replay, and when. */
export interface Certified {
  at: string;
  who: string;
  set: string;
  name: string;
  networks: SetNetwork[];
  made_at: string | null;
  frames: number;
}
/** A replay next to the certified one, in frames read (simulation.compare): where the speed to hold differs, where the
 * sign read differs, the frames only one of the two has, and of the frames with another speed, how many gave the brain
 * the same inputs. inputs_frames: frames where the car's signals kept differ; sources: the minutes whose video or log is
 * not the one certified ("video of minute 3"), none for a replay made before they were kept. */
export interface Counts {
  frames: number;
  only_was: number;
  only_now: number;
  limit_frames: number;
  reading_frames: number;
  same_input: number;
  other_input: number;
  inputs_frames?: number;
  sources?: string[];
}
/** What a drive has (simulation.summary): its replays by set, its certification, and each replay against it. */
export interface Summary {
  route: string;
  sets: Partial<Record<SetKey, SetResult>>;
  certified: Certified | null;
  regression: Partial<Record<SetKey, Counts & {identical: boolean}>>;
}
/** A drive of the list: the row of Speed review's list with what it has here. */
export interface SimulationRow extends Omit<Summary, 'route'> {
  route: string;
  title: string | null;
  when: string | null;
  when_certain?: boolean;
  day: string | null;
  minutes: number | null;
  dashcam: boolean | null;
}
/** GET /api/simulation/drives. */
export interface SimulationList {
  sets: SetStates;
  drives: SimulationRow[];
}

/** A box of SignNet on a frame read, in pixels of the frame: its class as the classifier read it, the classifier's
 * confidence (0 to 1) and whose road the applicability network gave it to ('' | own_road | other_road | unresolved). */
export type SignBox = [x1: number, y1: number, x2: number, y2: number, signClass: string, confidence: number, road: string];
/** A replay as the page draws it (simulation.view): the series of speedVisionSP as the timeline names them, in seconds
 * from the start of the drive's logs; the readings that pulse; SignNet's boxes by frame read. */
export interface SimulationView {
  route: string;
  set: string;
  label: string;
  name: string;
  made_at: string | null;
  networks: SetNetwork[];
  frame: [number, number] | null;
  frames: number;
  series: Record<string, Series>;
  events: [number, 'sign', number][];
  signs: [number, SignBox[]][];
}
/** A replay against the certified one, with where (simulation.differences): runs of [from, to, certified, now] in
 * seconds, to being the last frame that differs. */
export interface Differences extends Counts {
  limit: [number, number, number, number][];
  limit_seconds: number;
  reading: [number, number, string, string][];
  reading_seconds: number;
}
/** GET /api/simulation/drives/<route>?set=<car|in-use|certified>. view null: never replayed with that set. differences:
 * only for a set's replay of a certified drive. */
export interface SimulationDrive {
  route: string;
  sets: SetStates;
  summary: Summary;
  view: SimulationView | null;
  differences: Differences | null;
}

// -- kept in this browser -------------------------------------------------------------------------------------------------
function stored(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function keep(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Storage off: the choice lasts until the page is left.
  }
}
/** The set chosen on the list: what the list shows of each drive, what the certified drives are replayed with, and what
 * a drive opened shows first. */
export const chosenSet = ref<SetKey>(stored('nnslr-drive-set') === 'car' ? 'car' : 'in-use');
watch(chosenSet, (key) => keep('nnslr-drive-set', key));
/** SignNet's boxes over the picture of a drive opened. */
export const showBoxes = ref(stored('nnslr-drive-boxes') !== '0');
watch(showBoxes, (on) => keep('nnslr-drive-boxes', on ? '1' : '0'));

// -- the networks of a set ------------------------------------------------------------------------------------------------
/** The roles as the page names them (simulation.ROLES; Models calls the finder the cutter), in the car's order. */
const ROLE_LABEL: Record<string, string> = {finder: 'SignNet', cutter: 'SignNet', classifier: 'Classifier', decider: 'Applicability',
  brain: 'Brain'};
const ROLE_ORDER = Object.keys(ROLE_LABEL);
/** The networks of a set, a line each: those of its release; for In use not exported yet, or exported from other
 * versions, the versions in use on Models, which Compute exports first. */
export function networksOf(state: SetState | undefined): {label: string; id: string}[] {
  if (!state) return [];
  const wanted = Object.entries(state.wanted ?? {});
  const list: [string, string][] = state.ready || !wanted.length ? state.networks.map((one) => [one.role, one.id]) : wanted;
  return list.sort((a, b) => ROLE_ORDER.indexOf(a[0]) - ROLE_ORDER.indexOf(b[0]))
    .map(([role, id]) => ({label: ROLE_LABEL[role] ?? role, id: id || 'version unknown'}));
}

// -- the comma's screen at a time -----------------------------------------------------------------------------------------
/** A number rounded as Python's round() does, as the car rounds what its squares show: to the nearest whole, a half to
 * the even one (round(12.5) is 12, round(13.5) 14), where Math.round takes it up. */
export function pyRound(value: number): number {
  const whole = Math.floor(value);
  const rest = value - whole;
  return rest < 0.5 ? whole : rest > 0.5 ? whole + 1 : whole % 2 === 0 ? whole : whole + 1;
}

/** The comma's screen in its own pixels (the rect of its camera view on the comma four): what CommaScreen.vue draws in. */
export const SCREEN = {width: 476, height: 240} as const;

/** What the comma's two squares show at a time (the props of CommaScreen.vue but its scale). */
export interface Screen {
  /** The speed to hold, km/h; 0: none. */
  limitKph: number;
  /** The last sign read: its class ('' before the first), its number as written, the classifier's confidence (0 to 1). */
  rawClass: string;
  rawValue: number;
  confidence: number;
  /** That sign is in view. */
  seen: boolean;
  /** Metres to the sign ahead; 0: none. */
  distance: number;
  /** The kind of road the network sees; '' without one. */
  road: string;
  /** The pulse of a sign read, 0 to 1. */
  pulse: number;
}
const numberOf = (value: Value | undefined): number => typeof value === 'number' ? value : 0;
const textOf = (value: Value | undefined): string => typeof value === 'string' ? value : '';

/** The reading square pulses this long after a sign read, up and back (sign_reading.PULSE_SECONDS). */
const PULSE_SECONDS = 0.6;
/** The index of the last of count sorted times at or before s, -1 before the first. */
function lastAt(count: number, time: (index: number) => number, s: number): number {
  let low = 0;
  let high = count - 1;
  let found = -1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (time(middle) <= s) {
      found = middle;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return found;
}
/** The pulse at a time, from the times of the signs read: it follows the time shown, so it moves with the video and
 * stands still with it. */
export function pulseAt(reads: readonly number[], s: number): number {
  const index = lastAt(reads.length, (at) => reads[at]!, s);
  const since = index < 0 ? Infinity : s - reads[index]!;
  return since < PULSE_SECONDS ? Math.sin(Math.PI * since / PULSE_SECONDS) : 0;
}
/** The times of the signs read (speedVisionSP.signRead), from the events of a replay or of the timeline. */
export function readTimes(events: readonly (readonly [number, string, unknown])[]): number[] {
  return events.filter((event) => event[1] === 'sign').map((event) => event[0]).sort((a, b) => a - b);
}

/** The screen at a time, from the series of a replay or of the comma's log. A log from before the car sent the number
 * as written and whether the sign is in view (rawSignValue, rawSignSeen) gives its speed in km/h and a sign with a size
 * on the camera instead. Where the frames stop (null) the squares are empty, as without speedvisiond. */
export function screenAt(series: Record<string, Series>, reads: readonly number[], s: number): Screen {
  const at = (name: string): Value | undefined => valueAt(series[name], s);
  const kph = pyRound(numberOf(at('vis_limit')));
  return {
    limitKph: kph > 0 ? kph : 0,
    rawClass: textOf(at('vis_raw_class')),
    rawValue: numberOf(series['vis_raw_value'] ? at('vis_raw_value') : at('vis_raw_speed')),
    confidence: numberOf(at('vis_raw_conf')),
    seen: series['vis_raw_seen'] ? at('vis_raw_seen') === true : numberOf(at('vis_sign_size')) > 0,
    distance: numberOf(at('vis_sign_dist')),
    road: textOf(at('vis_road')),
    pulse: pulseAt(reads, s),
  };
}

/** The road signs a reading can be, by their short name on the comma, crossed out at their end (sign_reading.ROAD_SIGNS). */
const ROAD_SIGNS: Record<string, [text: string, crossed: boolean]> = {
  'other_sign:motorway_start': ['MWY', false], 'other_sign:motorway_end': ['MWY', true],
  'other_sign:expressway_start': ['EXP', false], 'other_sign:expressway_end': ['EXP', true],
  'other_sign:town_entry': ['TOWN', false], 'other_sign:town_exit': ['TOWN', true],
};
/** What the reading square says of a sign read (sign_reading.reading_text): the number as written, END for the end of
 * a limit, a road sign's short name, crossed out at its end; '' before the first sign. */
export function readingText(signClass: string, value: number): {text: string; crossed: boolean} {
  if (!signClass) return {text: '', crossed: false};
  if (signClass === 'cancellation') return {text: 'END', crossed: false};
  const road = ROAD_SIGNS[signClass];
  if (road) return {text: road[0], crossed: road[1]};
  return {text: value > 0 ? String(pyRound(value)) : '?', crossed: false};
}
/** The same in words, for a line of text: "TOWN end" for a crossed one. */
export function readingWords(signClass: string, value: number): string {
  const {text, crossed} = readingText(signClass, value);
  return crossed ? text + ' end' : text;
}

// -- SignNet's boxes ------------------------------------------------------------------------------------------------------
/** A frame read has its time on the log's clock when speedvisiond got it, about 42 ms after its exposure started, which
 * is the time of its picture in the video: the boxes go on the picture they were found on, not on the next one. */
const FRAME_LEAD = 0.045;
/** The boxes hold until the next frame read, 0.2 s later (pipeline.PERIOD); older than this they are of a frame gone. */
const BOX_SECONDS = 0.35;
/** Two boxes of two frames read one after the other are the same sign when their centres are closer than this many times
 * the larger box's longer side: a sign close by moves about one and a half of its size in 0.25 s. */
const SAME_SIGN = 2;
/** A box on the picture of a time. The car reads 5 frames a second and the video has 20: on the frame read, SignNet's
 * box as it is; on the frames after it, on its way to the same sign's box of the next frame read, so that it stays on
 * the sign (the sign moves 25-30 px in 0.2 s close by); held: the next frame read has no box of that sign, so it stays
 * where it was read, behind the sign. */
export interface ShownBox {
  box: SignBox;
  held: boolean;
}
const centre = (box: SignBox): [number, number] => [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
const side = (box: SignBox): number => Math.max(box[2] - box[0], box[3] - box[1], 1);
/** For each box of a frame read, the same sign's box in the next frame read, or null: the nearest one, the same class
 * first, each box of the next frame taken once. */
export function sameSigns(now: readonly SignBox[], next: readonly SignBox[]): (SignBox | null)[] {
  const pairs: [number, number, number][] = [];
  now.forEach((box, i) => next.forEach((other, j) => {
    const [x, y] = centre(box);
    const [u, v] = centre(other);
    const apart = Math.hypot(u - x, v - y) / Math.max(side(box), side(other));
    if (apart <= SAME_SIGN) pairs.push([i, j, apart + (box[4] === other[4] ? 0 : SAME_SIGN)]);
  }));
  pairs.sort((a, b) => a[2] - b[2]);
  const found: (SignBox | null)[] = now.map(() => null);
  const taken = new Set<number>();
  for (const [i, j] of pairs) {
    if (found[i] || taken.has(j)) continue;
    found[i] = next[j]!;
    taken.add(j);
  }
  return found;
}
/** SignNet's boxes on the picture of a time: those of the last frame read up to it, each moved by the video's frames
 * since it towards the same sign's box of the next frame read (ShownBox); none once that frame is old. */
export function boxesAt(signs: readonly [number, SignBox[]][], s: number): ShownBox[] {
  const at = s + FRAME_LEAD;
  const index = lastAt(signs.length, (row) => signs[row]![0], at);
  const row = signs[index];
  if (!row || at - row[0] >= BOX_SECONDS) return [];
  // Whole frames of the video: 0 on the frame read, whatever the few ms between its exposure and FRAME_LEAD.
  const step = Math.round((at - row[0]) * FPS);
  const next = signs[index + 1];
  const steps = next ? Math.round((next[0] - row[0]) * FPS) : 0;
  const ahead = next && next[0] - row[0] < BOX_SECONDS && steps > step ? sameSigns(row[1], next[1]) : [];
  return row[1].map((box, i) => {
    const to = ahead[i];
    if (step <= 0) return {box, held: false};
    if (!to) return {box, held: true};
    const part = step / steps;
    const way = (from: number, end: number): number => from + (end - from) * part;
    return {box: [way(box[0], to[0]), way(box[1], to[1]), way(box[2], to[2]), way(box[3], to[3]), box[4], box[5], box[6]],
      held: false};
  });
}

// -- the picture ----------------------------------------------------------------------------------------------------------
/** The road camera's frame, when a replay does not say its own. */
export const FRAME: [number, number] = [1344, 760];
/** Where the comma's screen goes over a stage that shows a frame whole (object-fit: contain, bars above and below the
 * picture or at its sides): at the picture's top right corner, at the picture's width over the comma's, never smaller
 * than 0.62 (the three lines of the reading square stay readable on a phone) nor larger than 1.6. right, top: in the
 * stage's pixels. */
export function screenPlace(stage: {width: number; height: number}, frame: readonly [number, number]): {right: number; top: number; scale: number} {
  const shape = frame[0] / frame[1];
  const width = Math.min(stage.width, stage.height * shape);
  return {right: (stage.width - width) / 2, top: (stage.height - width / shape) / 2,
    scale: Math.min(1.6, Math.max(0.62, width / SCREEN.width))};
}

// -- the regression test in words -----------------------------------------------------------------------------------------
/** The time between two frames the car reads (pipeline.PERIOD): what a frame that differs lasts. */
export const PERIOD = 0.2;
const secondsText = (frames: number): string => Math.round(frames * PERIOD * 10) / 10 + ' s';
/** Nothing differs: the same video and logs, the same frames read, the same speed to hold and the same reading on each. */
export const identical = (found: Counts): boolean =>
  !(found.limit_frames || found.reading_frames || found.only_was || found.only_now || found.sources?.length);
/** A replay next to the certified one, in one sentence; first, when they did not read the same files, which ones. */
export function regressionText(found: Counts): string {
  if (identical(found)) return 'Identical to the certified drive.';
  const parts: string[] = [];
  const sources = found.sources ?? [];
  if (sources.length) {
    parts.push('not the drive certified: the ' + sources.join(', the ') + (sources.length === 1 ? ' is' : ' are') + ' another file');
  }
  if (found.limit_frames) parts.push('the speed to hold differs for ' + secondsText(found.limit_frames));
  if (found.reading_frames) parts.push('the reading ' + (found.limit_frames ? '' : 'differs ') + 'for ' + plural(found.reading_frames, 'frame'));
  const alone = found.only_was + found.only_now;
  if (alone) parts.push(plural(alone, 'frame') + (alone === 1 ? ' was' : ' were') + ' read in one of the two only');
  return sentence(parts.join(', '));
}
/** The same in two words, for a row of the list: "identical", "12.4 s differ". */
export function regressionShort(found: Counts): string {
  if (identical(found)) return 'identical';
  if (found.sources?.length) return 'other video or logs';
  const frames = found.limit_frames || found.reading_frames;
  return frames ? secondsText(frames) + ' differ' : 'other frames';
}
/** Of the frames with another speed to hold, what changed: the brain (it was given the same inputs) or what is read
 * before it (other inputs), and the frames where the car's own signals differ. A sentence each, none without such frames.
 * With another video or log the inputs are other by force: said as such, not put on what is read. */
export function inputTexts(found: Counts): string[] {
  const texts: string[] = [];
  if (found.same_input) {
    texts.push('In ' + found.same_input + ' of the frames with another speed the brain was given the same inputs: the brain changed.');
  }
  if (found.other_input) {
    texts.push('In ' + found.other_input + ' of the frames with another speed the brain was given other inputs: ' +
      (found.sources?.length ? 'the video or the logs read are not the same.' : 'what is read before it changed.'));
  }
  if (found.inputs_frames) {
    texts.push('In ' + plural(found.inputs_frames, 'frame') + ' the car\'s own signals differ (speed, steering, blinkers, yaw, ' +
      'pedals, openpilot on, the car ahead, GPS time or country)' +
      (found.sources?.length ? ', from the other logs.' : '.'));
  }
  return texts;
}
