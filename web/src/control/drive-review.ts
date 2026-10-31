// Data › Drive review (DriveReviewTab.vue, DriveReviewDrive.vue): the drives as the comma recorded them, against its
// own logs. The server reads the logs (nnslr_tools.drive_review: no networks, CPU only) and finds the episodes worth a
// look: the driver intervening (accelerator, brake, the wheel held, openpilot disengaged, the set speed changed by
// hand) and the vision limit disagreeing with the map's. Here: the types of /api/drive-review, a series' value at a
// time, and a time of the logs as a time of the drive's video (the encode index of every frame: never a file's date).

/** A point of a series of the comma's logs: seconds from the drive's first video frame, and the value (null: the
 * service was not in the log there). */
export type Value = number | string | boolean | null;
export type Series = [number, Value][];

export type ScanName = 'no_logs' | 'not_scanned' | 'queued' | 'scanning' | 'failed' | 'outdated' | 'no_episodes' | 'done';
export interface ScanState {
  route: string;
  state: ScanName;
  files: number;
  segments: number;
  kinds: string[];
  progress: number;
  error: string | null;
  finished_at: string | null;
  scanner_version: number | null;
  coverage: Record<string, number> | null;
  counts: Record<string, number> | null;
  dashcam: boolean | null;
  unreadable: {segment: number; error: string}[];
  episodes?: number;
}

export interface DriveRow {
  route: string;
  title: string | null;
  when: string | null;
  when_certain: boolean;
  minutes: number | null;
  scan: ScanState;
  /** Logs found on the server and not imported yet (only for a drive without logs). */
  logs_found: number | null;
  video_segments: number;
  episodes: {open: number; reviewed: number; undeterminable: number};
  spot_checks: number;
  annotations: number;
}

/** What the comma had at a time (drive_episodes.Drive.state). */
export interface Recorded {
  vision_kph: number | null;
  vision_logged: boolean;
  map_kph: number | null;
  map_logged: boolean;
  resolver_kph: number | null;
  resolver_source: string | null;
  sla_state: string | null;
  v_ego_kph: number | null;
  set_speed_kph: number | null;
  openpilot: string | null;
  engaged: boolean | null;
  long_active: boolean | null;
  lat_active: boolean | null;
  road_type: string | null;
  road_name: string | null;
}

export interface Availability {
  vision: boolean;
  map: boolean;
  resolver: boolean;
  state: boolean;
  road_type: boolean;
  log: 'rlog' | 'qlog' | 'mixed' | null;
}

export interface Hint {
  speed_kph: number;
  source: string;
  closer_to: 'vision' | 'map' | null;
}

export type Channel = 'intervention' | 'disagreement' | 'both' | 'spot_check';
export type EpisodeStatus = 'open' | 'reviewed' | 'undeterminable';

export interface QueueItem {
  id: string;
  route: string;
  start_ns: number;
  end_ns: number;
  channel: Channel;
  kinds: string[];
  priority: number;
  reasons: string[];
  hint: Hint | null;
  before: Recorded;
  data: Availability;
  video: boolean;
  status: EpisodeStatus;
  segments: [number, number] | null;
  duration_s: number;
}

export interface Clue {
  kind: string;
  t_ns: number;
  end_ns: number;
  detail: Record<string, unknown>;
}

export interface Episode {
  id: string;
  route: string;
  active: boolean;
  start_ns: number;
  end_ns: number;
  anchor_ns: number;
  start_s: number;
  end_s: number;
  anchor_s: number;
  channel: Channel;
  kinds: string[];
  clues: Clue[];
  priority: number;
  reasons: string[];
  hint: Hint | null;
  before: Recorded;
  data: Availability;
  segments: [number, number] | null;
  previous_change: {t_ns: number; vision_kph: number; from_kph: number | null} | null;
  scanner_version: number;
}

export type Verdict = 'wrong_limit' | 'correct_limit' | 'undeterminable';
export interface Interval {
  start_ns: number;
  end_ns: number | null;
  limit: {state: 'known' | 'unknown'; value_kph: number | null};
  change: string | null;
  sign_read_kph: number | null;
  scene: string | null;
  certain: boolean;
  note: string | null;
}
export interface AnnotationPayload {
  verdict: Verdict;
  reason: string | null;
  undeterminable_reason: string | null;
  note: string | null;
  intervals: Interval[];
  context?: {intervals: {recorded: Recorded | null}[]};
}
export interface Annotation {
  id: string;
  route: string;
  episode_id: string | null;
  revision: number;
  payload: AnnotationPayload;
  author: string;
  author_kind: 'human' | 'rule' | 'model' | 'auto';
  origin: string;
  created_at: string;
  updated_at: string;
}

/** How a segment of the logs lines up with the video (drive_logs.sync). */
export interface SyncSegment {
  segment: number;
  frames_log: number;
  frames_video: number | null;
  video_start: number | null;
  start: number | null;
  period: number | null;
  max_error_ms: number | null;
  missing_ids: number;
  precision: 'frame' | 'degraded' | 'no_video' | 'no_index';
  exact: [number, number][] | null;
  log: {kind: string; sha256: string} | null;
  from_s: number;
  to_s: number;
}

export interface VideoState {
  state: 'ready' | 'waiting' | 'making' | 'failed' | 'missing';
  progress?: number;
  reason?: string;
  url?: string;
  index?: {fps: number; duration: number; segments: {segment: number; start: number; frames: number}[]};
}

export interface Timeline {
  route: string;
  scan: ScanState;
  t0_ns: number | null;
  span_s?: [number, number];
  civil_ms?: number | null;
  meta?: {build?: Record<string, unknown>; car?: Record<string, unknown>};
  /** Missing segments of the logs: [the segment before, the segment after]. */
  holes?: [number, number][];
  left_out?: {segment: number; reason: string}[];
  segments: SyncSegment[];
  series: Record<string, Series>;
  events: [number, 'sign' | 'button' | 'onroad', unknown][];
  episodes: Episode[];
  annotations: Annotation[];
  gps?: [number, number, number][];
  video: VideoState;
  title: string | null;
  when: string | null;
  when_certain: boolean;
  inference: {run_id: string; cutter: string | null; classifier: string | null;
    frames: {segment: number; second: number; signs: {family: string; value_kph: number | null; score: number}[]}[]} | null;
  map_today: {source: string; runs: [number, number, number, string | null, number | null][]} | null;
}

// -- words ---------------------------------------------------------------------------------------------------------------

/** The no-logs state, as the spec of the review says it. */
export const NO_LOGS = 'Logs not available: import the logs to look for the episodes of the recorded drive.';

/** A kind of clue in a few words (drive_episodes.KIND_TEXT is the sentence). */
export const KIND_LABEL: Record<string, string> = {
  gas: 'Accelerator', brake: 'Brake', steer: 'Wheel held', set_speed_manual: 'Set speed by hand',
  disengage_user: 'Disengaged by the driver', disengage_auto: 'Disengaged by itself', disengage_unknown: 'Disengaged',
  disagreement: 'Vision ≠ map', steer_contact: 'Wheel touched', set_speed_auto: 'Set speed by the car',
  set_speed_unknown: 'Set speed changed', spot_check: 'Spot check',
};
export const CHANNEL_LABEL: Record<Channel, string> = {
  intervention: 'Intervention', disagreement: 'Disagreement', both: 'Intervention and disagreement', spot_check: 'Spot check',
};
export const STATUS_LABEL: Record<EpisodeStatus, string> = {open: 'To review', reviewed: 'Reviewed', undeterminable: 'Can\'t tell'};

export const VERDICTS: [Verdict, string][] = [['wrong_limit', 'Wrong limit'], ['correct_limit', 'Right limit, another reason'],
  ['undeterminable', 'Can\'t tell']];
export const OTHER_REASONS: [string, string][] = [['traffic', 'Traffic'], ['road_or_curve', 'Road or curve'],
  ['weather_or_visibility', 'Weather or visibility'], ['driver_choice', 'Driver\'s choice'], ['test', 'A test'], ['other', 'Other']];
export const UNDETERMINABLE_REASONS: [string, string][] = [['video_missing', 'No video'], ['data_missing', 'Data missing'],
  ['sign_not_visible', 'Sign not visible'], ['sign_ambiguous', 'Sign ambiguous'],
  ['conflicting_information', 'Clues disagree'], ['other', 'Other']];
export const CHANGES: [string, string][] = [['sign', 'A sign'], ['end_of_limit', 'End of limit'], ['scene', 'The kind of road'],
  ['carried', 'Carried on from before'], ['other', 'Other']];
/** The usual limits, one tap each. */
export const QUICK_LIMITS = [30, 40, 50, 60, 70, 80, 100, 120];

/** The state of a drive's logs and scan in words, and its tone. */
export function scanText(scan: ScanState): {text: string; tone: 'muted' | 'todo' | 'ok' | 'bad' | 'busy'} {
  switch (scan.state) {
    case 'no_logs': return {text: NO_LOGS, tone: 'muted'};
    case 'not_scanned': return {text: 'Logs imported, not searched yet.', tone: 'todo'};
    case 'queued': return {text: 'Waiting to search the logs…', tone: 'busy'};
    case 'scanning': return {text: 'Searching the logs… ' + Math.round((scan.progress || 0) * 100) + '%', tone: 'busy'};
    case 'failed': return {text: 'The search failed: ' + (scan.error ?? 'unknown error') + '.', tone: 'bad'};
    case 'outdated': return {text: 'New logs or a newer search: search again.', tone: 'todo'};
    case 'no_episodes': return {text: 'Searched: no episodes.', tone: 'ok'};
    case 'done': return {text: 'Searched.', tone: 'ok'};
  }
}

/** "1:05" or "1:02:05" from seconds of the drive. */
export function clockOf(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '–';
  const sign = seconds < 0 ? '−' : '';
  const whole = Math.floor(Math.abs(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor(whole / 60) % 60;
  const rest = String(whole % 60).padStart(2, '0');
  return sign + (hours ? hours + ':' + String(minutes).padStart(2, '0') : String(minutes)) + ':' + rest;
}

/** How long an episode lasts: "0:12 long", or "a moment" for a single event. */
export function lengthOf(seconds: number): string {
  return seconds < 1 ? 'a moment' : clockOf(seconds) + ' long';
}

/** A km/h value, or the reason there is none. */
export function kphText(value: number | null | undefined, missing = 'not in the log'): string {
  return value === null || value === undefined ? missing : Math.round(value) + ' km/h';
}

// -- the series ----------------------------------------------------------------------------------------------------------

/** The index of the last point at or before s, -1 before the first. */
export function indexAt(series: Series | undefined, s: number): number {
  if (!series || !series.length) return -1;
  let low = 0;
  let high = series.length - 1;
  if (series[0]![0] > s) return -1;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (series[middle]![0] <= s) low = middle;
    else high = middle - 1;
  }
  return low;
}

/** The value of a series at s; undefined when the drive has no such series or s is before its first point. */
export function valueAt(series: Series | undefined, s: number): Value | undefined {
  const index = indexAt(series, s);
  return index < 0 ? undefined : series![index]![1];
}

/** A limit of the logs (km/h): 0 is "none held, or not sure" (null); not in the log (null, or no series) is undefined. */
export function limitOf(value: Value | undefined): number | null | undefined {
  if (value === undefined || value === null) return undefined;
  return typeof value === 'number' && value > 0.5 ? Math.round(value) : null;
}

/** The set speed of carState.vCruise: 255 (and 0) mean none. */
export function setSpeedOf(value: Value | undefined): number | null {
  return typeof value === 'number' && value > 0 && value < 250 ? Math.round(value) : null;
}

/** Several series as one, on all their times: f of the value of each there (undefined before its first point); a
 * value equal to the one before is not repeated. */
export function combine(list: readonly (Series | undefined)[], f: (values: (Value | undefined)[]) => Value): Series {
  const times = [...new Set(list.flatMap((series) => (series ?? []).map((point) => point[0])))].sort((a, b) => a - b);
  const at = list.map(() => -1);
  const out: Series = [];
  for (const time of times) {
    const values = list.map((series, index) => {
      if (!series) return undefined;
      let next = at[index]!;
      while (next + 1 < series.length && series[next + 1]![0] <= time) next += 1;
      at[index] = next;
      return next < 0 ? undefined : series[next]![1];
    });
    const value = f(values);
    if (!out.length || out[out.length - 1]![1] !== value) out.push([time, value]);
  }
  return out;
}

/** The parts of a series between two times, each with its value, consecutive equal values joined (by key). */
export function runsOf<T>(series: Series | undefined, from: number, to: number, key: (value: Value) => T):
  {from: number; to: number; value: T}[] {
  const out: {from: number; to: number; value: T}[] = [];
  if (!series || !series.length || to <= from) return out;
  let index = Math.max(0, indexAt(series, from));
  for (; index < series.length; index++) {
    const [time, value] = series[index]!;
    if (time >= to) break;
    const end = Math.min(to, index + 1 < series.length ? series[index + 1]![0] : to);
    const start = Math.max(from, time);
    if (end <= start) continue;
    const mapped = key(value);
    const last = out[out.length - 1];
    if (last && last.value === mapped && Math.abs(last.to - start) < 1e-6) last.to = end;
    else out.push({from: start, to: end, value: mapped});
  }
  return out;
}

/** The last change of the vision limit before s to another known value (drive_episodes.previous_change). */
export function previousChange(series: Series | undefined, s: number): {at: number; kph: number; from: number | null} | null {
  if (!series) return null;
  const known: [number, number][] = [];
  for (const [time, value] of series) {
    if (time >= s) break;
    const limit = limitOf(value);
    if (typeof limit === 'number') known.push([time, limit]);
  }
  for (let index = known.length - 1; index >= 0; index--) {
    if (index === 0 || known[index]![1] !== known[index - 1]![1]) {
      return {at: known[index]![0], kph: known[index]![1], from: index ? known[index - 1]![1] : null};
    }
  }
  return null;
}

// -- the video -------------------------------------------------------------------------------------------------------------

const FPS = 20;

/** The segment of the logs at s: the one whose span holds it, else the last one starting before it. */
function segmentAt(segments: readonly SyncSegment[], s: number): SyncSegment | null {
  let found: SyncSegment | null = null;
  for (const item of segments) {
    if (item.from_s <= s + 1e-6) found = item;
    if (item.from_s <= s && s <= item.to_s) return item;
  }
  return found;
}

/** The frame id of a time in a segment: from the drifting frames' own times when the index has them, else the line. */
function frameOf(item: SyncSegment, s: number): number {
  if (item.exact && item.exact.length) {
    let best = item.exact[0]!;
    for (const pair of item.exact) if (pair[1] <= s) best = pair;
    return best[0] + (s - best[1]) / (item.period ?? 1 / FPS);
  }
  return (s - item.start!) / item.period!;
}

/** The time in the drive's video of a time of the logs, or null where the video has no picture of it. */
export function videoTimeOf(segments: readonly SyncSegment[], s: number): number | null {
  const item = segmentAt(segments, s);
  if (!item || item.video_start === null || item.start === null || !item.period || !item.frames_video) return null;
  const frame = frameOf(item, s);
  if (frame < -1 || frame > item.frames_video) return null;
  return item.video_start + Math.min(Math.max(frame, 0), item.frames_video - 1) / FPS;
}

/** The time of the logs of a time of the video, or null. */
export function logTimeOf(segments: readonly SyncSegment[], videoTime: number): number | null {
  for (const item of segments) {
    if (item.video_start === null || item.start === null || !item.period || !item.frames_video) continue;
    const end = item.video_start + item.frames_video / FPS;
    if (videoTime >= item.video_start - 1e-3 && videoTime < end) {
      const frame = (videoTime - item.video_start) * FPS;
      if (item.exact && item.exact.length) {
        let best = item.exact[0]!;
        for (const pair of item.exact) if (pair[0] <= frame) best = pair;
        return best[1] + (frame - best[0]) * item.period;
      }
      return item.start + frame * item.period;
    }
  }
  return null;
}

/** The time of the logs of a picture of the video: second N of a segment (the server's frames, one a second). */
export function secondOf(segments: readonly SyncSegment[], segment: number, second: number): number | null {
  const item = segments.find((row) => row.segment === segment);
  if (!item) return null;
  if (item.start !== null && item.period) return item.start + second * FPS * item.period;
  return item.from_s + second;
}
