// Data › Sign distance (SignDistanceTab.vue, SignDistanceDrive.vue): how late or early the comma's countdown to the sign
// ahead reaches zero. The car shows the distance to it (speedVisionSP.signDistance, vision.SignDistance in sunnypilot:
// a round sign by its size, a motorway or expressway sign by its height); a person marks, sign by sign, the frame of the road video they call "zero" (the sign is about to leave
// the picture), and the page shows how far from it vision's own zero comes. Only the comma's camera-side logs, the
// car's speed and the marks: never the map. The server (nnslr_tools.sign_distance) finds the signs, picks the frames
// that may be shown (never a second kept only at a car park's speed) and works out every figure: the page only shows
// them. Here: the types of /api/sign-distance and the words and numbers of the page.

/** What Sign distance can do with a drive (sign_distance.STATUS_MESSAGES). */
export type DriveStatus = 'reading' | 'unread' | 'unreadable' | 'no_logs' | 'build_too_old' | 'no_sign_seen' | 'no_video'
  | 'signs';
/** Why a sign is left out of the figures (sign_distance.EXCLUDED_MESSAGES). */
export type Excluded = 'cut' | 'dropped' | 'coarse' | 'no_index' | 'no_video' | 'slow_only';
export type MarkName = 'zero' | 'cant_tell' | 'clear';
/** How a minute's video and logs agree (drive_logs.precision): to the frame, or a mark is kept out of the medians. */
export type SyncPrecision = 'frame' | 'degraded' | 'unknown' | 'no_video' | 'no_index';

/** A person's mark in effect on a sign (seconds from the drive's first video frame). */
export interface Mark {
  mark: MarkName;
  segment: number | null;
  frame: number | null;
  t_s: number | null;
  sync: SyncPrecision | null;
  reviewer: string | null;
  at: string | null;
}

/** The figures of a 'zero' mark against vision's zero. */
export interface Metrics {
  /** Vision's screen zero minus the person's zero, in seconds: + vision's zero comes after it (late). */
  late_s: number;
  /** What vision said at the person's zero, in metres, and what the screen showed then (in the car's unit). */
  vision_m: number | null;
  vision_shown: number | null;
  v_kmh: number | null;
  /** Secondary: in metres, with the processing delay times the speed in it (late_m_note says so). */
  late_m: number | null;
  late_m_note: string;
  speed_missing: boolean;
}

export interface Sign {
  key: string;
  number: number;
  start_s: number | null;
  /** Vision's screen zero: the first sample that rounds to 0 in the car's unit. */
  zero_s: number | null;
  d0_s: number | null;
  end_s: number | null;
  end: 'zero' | 'cut';
  last_seen_s: number | null;
  last_seen_m: number | null;
  last_positive_m: number | null;
  raw_class: string | null;
  tracks: number;
  jumps: number;
  segments: number[];
  included: boolean;
  excluded: Excluded | null;
  excluded_message: string | null;
  frames: number;
  mark: Mark | null;
  metrics: Metrics | null;
}

export interface Build {
  version: string | null;
  branch: string | null;
  commit: string | null;
  commit_date: string | null;
  verdict: 'new' | 'old' | 'unknown';
}

/** A log found that this server cannot open (no path): its permissions are Cristian's to fix, never the page's. */
export interface UnreadableFile {
  segment: number;
  kind: string;
  error: string;
  seen_at: string | null;
}

export interface Counts {
  signs: number;
  included: number;
  excluded: Partial<Record<Excluded, number>>;
  zero: number;
  cant_tell: number;
  orphans: number;
}

export interface Drive {
  route: string;
  title: string | null;
  when: string | null;
  when_certain: boolean;
  day: string | null;
  status: DriveStatus;
  message: string | null;
  scan: string;
  unreadable_files: UnreadableFile[];
  unit: 'm' | 'ft';
  unit_known: boolean;
  build: Build | null;
  counts: Counts;
}

/** A mark whose sign is no longer found (the logs read again): kept, never dropped, out of the figures. */
export interface Orphan extends Mark {
  sign: string;
}

export interface DriveView extends Drive {
  ok: true;
  signs: Sign[];
  orphans: Orphan[];
}

/** A frame of a sign's window that may be shown, with what vision said there. */
export interface WindowFrame {
  segment: number;
  frame: number;
  t_s: number;
  vision_m: number | null;
  /** The number on the car's screen (m or ft): 0 shows no number, null no value logged. */
  shown: number | null;
}

export interface SignView {
  ok: true;
  route: string;
  unit: 'm' | 'ft';
  unit_known: boolean;
  sign: Sign;
  /** wide_segments: the minutes whose wide road camera has the same frames, shown beside the road camera. */
  window: {from_s: number | null; to_s: number | null; frames: WindowFrame[]; wide_segments?: number[]};
  series: {vision_m: [number, number | null][]; size_px: [number, number | null][]; speed_kmh: [number, number | null][]};
  sync: {segment: number; precision: SyncPrecision; max_error_ms: number | null; frames_log: number | null;
    frames_video: number | null}[];
  warnings: {code: string; segment?: number; message: string}[];
  /** The included signs before and after this one, its place among them (null for a sign left out). */
  prev: string | null;
  next: string | null;
  position: number | null;
  of: number;
}

export interface Stats {
  median: number | null;
  count: number;
  min: number | null;
  max: number | null;
}

export interface Summary {
  drives: number;
  statuses: Partial<Record<DriveStatus, number>>;
  signs: number;
  included: number;
  excluded: Partial<Record<Excluded, number>>;
  zero: number;
  cant_tell: number;
  late_s: Stats;
  late_m: Stats & {note: string};
  degraded_marks: number;
  orphans: number;
  by_drive: {route: string; title: string | null; included: number; zero: number; late_s: Stats}[];
}

export const API = '/api/sign-distance';

export function driveHref(route: string, sign: string | null = null): string {
  return '/sign-distance?drive=' + encodeURIComponent(route) + (sign ? '&sign=' + encodeURIComponent(sign) : '');
}

/** The picture of a frame of the road camera, or (wide) the same frame of the wide road camera. */
export function frameUrl(route: string, frame: Pick<WindowFrame, 'segment' | 'frame'>, wide = false): string {
  return API + '/drives/' + encodeURIComponent(route) + '/frames/' + frame.segment + '/' + frame.frame
    + (wide ? '-wide' : '') + '.jpg';
}

/** The tone of a drive's state, with the classes of Drive review (dr-muted, dr-todo, dr-ok, dr-bad, dr-busy). */
export function statusTone(status: DriveStatus): 'muted' | 'todo' | 'ok' | 'bad' | 'busy' {
  switch (status) {
    case 'reading': return 'busy';
    case 'unread': return 'todo';
    case 'unreadable': return 'bad';
    case 'no_video': return 'todo';
    case 'signs': return 'ok';
    default: return 'muted';
  }
}

/** After a number of signs: "3 cut short by the logs, 1 without video yet". */
export const EXCLUDED_LABEL: Record<Excluded, string> = {
  cut: 'cut short by the logs', dropped: 'lost by vision', coarse: 'in qlogs only', no_index: 'without an encode index',
  no_video: 'without video yet', slow_only: 'with slow-only video',
};

const ROUND = (value: number, digits: number): string => (Math.round(Math.abs(value) * 10 ** digits) / 10 ** digits).toFixed(digits);

/** late_s in words: "0.51 s late", "0.20 s early", "on time". */
export function lateText(late: number | null | undefined): string {
  if (late === null || late === undefined || !Number.isFinite(late)) return '–';
  if (Math.abs(late) < 0.005) return 'on time';
  return ROUND(late, 2) + ' s ' + (late > 0 ? 'late' : 'early');
}

/** late_s signed, for a list: "+0.51 s", "−0.20 s". */
export function signedSeconds(late: number | null | undefined): string {
  if (late === null || late === undefined || !Number.isFinite(late)) return '–';
  return (late > 0 ? '+' : late < 0 ? '−' : '±') + ROUND(late, 2) + ' s';
}

/** late_m signed: "+2.6 m". */
export function signedMetres(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '–';
  return (value > 0 ? '+' : value < 0 ? '−' : '±') + ROUND(value, 1) + ' m';
}

/** The sentence of a sign's main figure: "Vision's zero comes 0.51 s after yours". */
export function lateSentence(late: number): string {
  if (Math.abs(late) < 0.005) return 'Vision\'s zero comes with yours';
  return 'Vision\'s zero comes ' + ROUND(late, 2) + ' s ' + (late > 0 ? 'after' : 'before') + ' yours';
}

/** What the car's screen showed: "14 m", "0 (no number)", or null for no value logged. */
export function screenText(shown: number | null | undefined, unit: 'm' | 'ft'): string | null {
  if (shown === null || shown === undefined) return null;
  return shown > 0 ? shown + ' ' + unit : '0';
}

/** "1:07.25": a time of the drive to the hundredth. */
export function timeText(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '–';
  const negative = seconds < 0;
  const hundredths = Math.round(Math.abs(seconds) * 100);
  const minutes = Math.floor(hundredths / 6000);
  const rest = (hundredths % 6000) / 100;
  return (negative ? '−' : '') + minutes + ':' + rest.toFixed(2).padStart(5, '0');
}

/** The index of the frame closest to a time (-1 for no frames). */
export function nearestFrame(frames: WindowFrame[], seconds: number | null | undefined): number {
  if (!frames.length || seconds === null || seconds === undefined) return -1;
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (frames[middle]!.t_s < seconds) low = middle + 1;
    else high = middle;
  }
  if (low > 0 && Math.abs(frames[low - 1]!.t_s - seconds) <= Math.abs(frames[low]!.t_s - seconds)) return low - 1;
  return low;
}

/** The index of a frame (minute and frame number), -1 when it is not one of the window's. */
export function frameIndex(frames: WindowFrame[], segment: number | null, frame: number | null): number {
  if (segment === null || frame === null) return -1;
  return frames.findIndex((item) => item.segment === segment && item.frame === frame);
}

/** A sign's class as the car logged it ("maximum_speed:30") in words: "Maximum speed 30"; a number with hours
 * ("maximum_speed_hours:30"): "Speed limit 30, with hours". */
export function classText(raw: string | null): string {
  if (!raw) return 'Sign';
  const [family, value] = raw.split(':');
  if (family === 'maximum_speed_hours') return 'Speed limit ' + value + ', with hours';
  const name = (family ?? raw).replace(/_/g, ' ');
  return value ? name.charAt(0).toUpperCase() + name.slice(1) + ' ' + value : name.charAt(0).toUpperCase() + name.slice(1);
}

/** The words of a sign's mark: "Zero marked", "Can't tell", "To mark". */
export function markLabel(sign: Sign): string {
  if (!sign.included) return 'Left out';
  if (!sign.mark) return 'To mark';
  return sign.mark.mark === 'zero' ? 'Zero marked' : sign.mark.mark === 'cant_tell' ? 'Can\'t tell' : 'To mark';
}

/** A sign still to mark: included and with no mark in effect. */
export function toMark(sign: Sign): boolean {
  return sign.included && (!sign.mark || sign.mark.mark === 'clear');
}
