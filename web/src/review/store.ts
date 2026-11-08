// The page's data and its one reactive state: the payload embedded by the server (or by write_review for the
// offline report), the working copies of the frames, what is on screen. The modules change this state; the
// components of ReviewApp.vue follow it.
import {reactive, shallowReactive} from 'vue';
import {readLocal, writeLocal} from './dom';
import {isFilter} from './frames';
import {CAMERA_FPS, READ_INDEX_BASE, compareFrames, frameSecond} from './order';
import type {
  Busy, Drag, Filter, FrameEdit, FrameRow, Hint, HintStats, LastSave, Notice, Payload, RouteLink, ServerState,
  Suspicious,
} from './types';

// The working copies of a drive in this browser, by run: STORAGE_PREFIX + run id.
const STORAGE_PREFIX = 'nnslr-review-v1-';
// Whether the list of keys under Confirm is open, in this browser.
const KEYS_SHOWN = 'nnslr-review-keys-shown';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** What the payload says, fixed for the life of the page (initStore fills it before the app mounts). */
export interface Report {
  runId: string;
  routeId: string;
  rows: readonly FrameRow[];
  routes: readonly RouteLink[];
  /** Country code: English name, for the country of a frame. */
  countries: Readonly<Record<string, string>>;
  hintStats: HintStats | null;
  routeHasHints: boolean;
  /** With the review server every confirmation is saved in its database; opened as a plain file the page
   * works offline (null). */
  apiBase: string | null;
  storageKey: string;
  lastKeyStorage: string;
  /** Who proposed the signs (vietnam, ours, qwen), null when not known. */
  proposer: string | null;
  /** Propose again worked on this drive when the page was made: read only until reloaded (review/busy.ts). */
  readOnly: boolean;
  /** The page shows the new proposals made so far, not the report (the drive at work). */
  atWork: boolean;
  /** Segment: the frames of its road video (null: not counted yet), for the segments that have one on the server;
   * null offline. A cell of the timeline is the 20 frames of its second (seconds.ts). Replaced, never changed in
   * place, when video.ts learns a count. */
  videos: Readonly<Record<number, number | null>> | null;
}

// Shallow: only its fields are followed (the rows of the drive at work grow while the page is open, appendRows).
export const report: Report = shallowReactive({
  runId: '', routeId: '', rows: [], routes: [], countries: {}, hintStats: null, routeHasHints: false, apiBase: null,
  storageKey: '', lastKeyStorage: '', proposer: null, readOnly: false, atWork: false, videos: null,
});

let hintsByFrame: Record<string, Hint[]> = {};
let rowIndex = new Map<string, number>();
let segmentRows = new Map<number, FrameRow[]>();
// The frames of the video between the seconds the reviewer went through (virtualRowOf), by key.
const virtualRows = new Map<string, FrameRow>();

/** The export of "Download reviews (JSONL)": its message under the button. */
export interface ExportMessage {
  text: string;
  ok: boolean;
}

export interface ReviewState {
  /** The working copy of every frame touched, by frame key (also what localStorage keeps). */
  edits: Record<string, FrameEdit>;
  suspicious: Suspicious;
  currentKey: string | null;
  /** Index of the selected box in the working copy, -1 for none. */
  selected: number;
  /** The reviewer chose that box (click, arrows, a new box), rather than the page opening on it: only then does
   * Del delete it. */
  selectedByUser: boolean;
  /** The digits of a speed being typed on the keyboard ("8" before "80"), empty when none. */
  typing: string;
  drawing: boolean;
  drag: Drag | null;
  /** The rectangle being drawn on the photo (image pixels), while drawing. */
  draft: {x: number; y: number; width: number; height: number} | null;
  /** The click that follows a drag is ignored until then (ms). */
  ignoreClickUntil: number;
  filter: Filter;
  filterNote: string;
  showHints: boolean;
  /** The reviewer field of the top bar, as typed. */
  reviewer: string;
  /** A save needed the reviewer's name: the field is marked until something is typed. */
  reviewerNeeded: boolean;
  /** Counts the requests to put the focus on the reviewer field. */
  reviewerFocus: number;
  /** The "Other value" field holds a number out of range. */
  valueError: boolean;
  exportMessage: ExportMessage | null;
  /** Frames being sent to the server now. */
  inFlight: Set<string>;
  notice: Notice | null;
  storageWarning: string;
  serverWarning: string;
  retrying: boolean;
  /** The frame whose "Withdraw confirmation" was pressed once: the second press withdraws. */
  withdrawArmed: string | null;
  /** "<frame key>#<version>": Confirm already asked about the same sign drawn twice; pressed again it saves. */
  duplicateAsked: string | null;
  saveCount: number;
  lastSave: LastSave | null;
  server: ServerState;
  /** How the drive is now for Propose again (asked again while the page is read only); null: not busy (any more). */
  busy: Busy | null;
  /** The list of keys under Confirm is open (Keys, or ?): kept per browser, closed at first. */
  showKeys: boolean;
  /** The newest updated_at of the saves the page has read (review/ai-live.ts asks for those after it). */
  newestSave: string | null;
}

export const state: ReviewState = reactive({
  edits: {},
  suspicious: {},
  currentKey: null,
  selected: -1,
  selectedByUser: false,
  typing: '',
  drawing: false,
  drag: null,
  draft: null,
  ignoreClickUntil: 0,
  filter: 'todo',
  filterNote: '',
  showHints: true,
  reviewer: '',
  reviewerNeeded: false,
  reviewerFocus: 0,
  valueError: false,
  exportMessage: null,
  inFlight: new Set<string>(),
  notice: null,
  storageWarning: '',
  serverWarning: '',
  retrying: false,
  withdrawArmed: null,
  duplicateAsked: null,
  saveCount: 0,
  lastSave: null,
  server: 'offline',
  busy: null,
  showKeys: readLocal(KEYS_SHOWN, '0') === '1',
  newestSave: null,
});

/** Opens or closes the list of keys under Confirm, for this browser. */
export function toggleKeys(): void {
  state.showKeys = !state.showKeys;
  writeLocal(KEYS_SHOWN, state.showKeys ? '1' : '0');
}

/** The payload of the page (<script id="payload">), read once before the app mounts. */
export function initStore(raw: Partial<Payload>): void {
  const routeId = raw.route_id ?? '';
  hintsByFrame = isRecord(raw.hints) ? raw.hints as Record<string, Hint[]> : {};
  Object.assign(report, {
    runId: String(raw.run_id),
    routeId,
    rows: Array.isArray(raw.frames) ? raw.frames : [],
    routes: Array.isArray(raw.routes) ? raw.routes : [],
    countries: isRecord(raw.countries) ? raw.countries as Record<string, string> : {},
    hintStats: isRecord(raw.hint_stats) ? raw.hint_stats as unknown as HintStats : null,
    routeHasHints: Object.keys(hintsByFrame).length > 0,
    apiBase: raw.server ? '/api/route/' + encodeURIComponent(routeId) : null,
    storageKey: STORAGE_PREFIX + String(raw.run_id),
    lastKeyStorage: 'nnslr-review-last-' + String(raw.run_id),
    proposer: typeof raw.proposer === 'string' ? raw.proposer : null,
    readOnly: isRecord(raw.busy),
    atWork: isRecord(raw.busy) && raw.busy['phase'] === 'proposing again',
    videos: raw.server && isRecord(raw.videos) ? videosOf(raw.videos) : null,
  } satisfies Report);
  state.busy = isRecord(raw.busy) ? raw.busy as unknown as Busy : null;
  reindex();
  state.suspicious = isRecord(raw.suspicious) ? raw.suspicious as Suspicious : {};
}

function videosOf(raw: Record<string, unknown>): Record<number, number | null> {
  const videos: Record<number, number | null> = {};
  for (const [segment, count] of Object.entries(raw)) {
    if (/^[0-9]+$/.test(segment)) videos[Number(segment)] = typeof count === 'number' && count > 0 ? count : null;
  }
  return videos;
}

/** The frames of a segment's video, learnt from the server (video.ts). */
export function videoCounted(segment: number, count: number): void {
  const videos = report.videos;
  if (videos === null || videos[segment] === count) return;
  report.videos = {...videos, [segment]: count};
}

function reindex(): void {
  rowIndex = new Map(report.rows.map((row, index) => [row.frame_key, index]));
  segmentRows = new Map();
  for (const row of report.rows) {
    const list = segmentRows.get(row.segment_index);
    if (list === undefined) segmentRows.set(row.segment_index, [row]);
    else list.push(row);
  }
}

/** The new proposals of the drive at work (review/busy.ts), after the rows the page has. */
export function appendRows(rows: readonly FrameRow[]): void {
  const fresh = rows.filter((row) => !rowIndex.has(row.frame_key));
  if (fresh.length === 0) return;
  report.rows = [...report.rows, ...fresh];
  reindex();
}

/** A frame between the seconds just saved (server.ts send): a picture of the report from now on, in time order. */
export function insertRow(row: FrameRow): void {
  if (rowIndex.has(row.frame_key)) return;
  report.rows = [...report.rows, row].sort(compareFrames);
  virtualRows.delete(row.frame_key);
  reindex();
}

export function indexOf(key: string): number {
  return rowIndex.get(key) ?? -1;
}

/** The place of a frame in the route: its index, or for a frame of the video not saved (virtualRowOf) the index of
 * the frame just before it (-1 when there is none). */
export function anchorIndex(key: string): number {
  const index = rowIndex.get(key);
  if (index !== undefined) return index;
  const row = virtualRowOf(key);
  if (row === null) return -1;
  let anchor = -1;
  for (let i = 0; i < report.rows.length && compareFrames(report.rows[i] as FrameRow, row) < 0; i += 1) anchor = i;
  return anchor;
}

export function rowOf(key: string): FrameRow | null {
  const index = rowIndex.get(key);
  return index === undefined ? virtualRowOf(key) : report.rows[index] ?? null;
}

/** A frame of the video between the seconds that is not a picture of the report (video.ts): its row, shown from
 * the server's video (preview video/<segment>/<frame>.jpg), with the size, position and country of the nearest
 * picture of its segment. Only with the review server, which takes it into the report when it is confirmed
 * (review_service.video_frame_row). Null for any other key. */
export function virtualRowOf(key: string): FrameRow | null {
  const known = virtualRows.get(key);
  if (known !== undefined) return known;
  const match = /^([0-9]+)\/([0-9]+)$/.exec(key);
  if (report.apiBase === null || match === null || rowIndex.has(key)) return null;
  const segment = Number(match[1]);
  const index = Number(match[2]);
  const rows = segmentRows.get(segment);
  if (index <= READ_INDEX_BASE || rows === undefined) return null;
  const decoded = index - READ_INDEX_BASE;
  const second = decoded / CAMERA_FPS;
  let near = rows[0] as FrameRow;
  for (const row of rows) {
    if (Math.abs(frameSecond(row) - second) < Math.abs(frameSecond(near) - second)) near = row;
  }
  const row: FrameRow = {
    frame_key: key, segment_index: segment, output_index: index, decoded_frame_index: decoded, media_time_s: second,
    width: near.width, height: near.height, preview_path: 'video/' + segment + '/' + decoded + '.jpg', detections: [],
    gps: near.gps ?? null, country_proposed: near.country_proposed ?? null,
    country_from: near.country_proposed ? 'nearest_fix' : null,
  };
  virtualRows.set(key, row);
  return row;
}

/** Not a picture of the report: a frame of the video between the seconds, never saved. */
export function isVirtual(key: string): boolean {
  return !rowIndex.has(key) && virtualRowOf(key) !== null;
}

export function hintsOf(key: string): Hint[] {
  const list = hintsByFrame[key];
  return Array.isArray(list) ? list : [];
}

/** What this browser keeps of a drive: with the review server only the frames it does not have yet (changed and not
 * saved, or confirmed and not sent), since it gives the others back at every load; offline (the static report)
 * every frame. Kept whole, the copies of a few drives the AI reviewed filled the browser's room. */
function kept(edits: Record<string, FrameEdit>): Record<string, FrameEdit> {
  if (!report.apiBase) return edits;
  const result: Record<string, FrameEdit> = {};
  for (const [key, edit] of Object.entries(edits)) {
    if (edit.saved === false) result[key] = edit;
  }
  return result;
}

/** Room in this browser: the copies of the other drives keep only their frames the server does not have. */
function compactOthers(): void {
  for (let index = localStorage.length - 1; index >= 0; index -= 1) {
    const key = localStorage.key(index);
    if (key === null || !key.startsWith(STORAGE_PREFIX) || key === report.storageKey) continue;
    try {
      const value: unknown = JSON.parse(localStorage.getItem(key) || '{}');
      if (!isRecord(value)) continue;
      const pending = Object.entries(value).filter(([, edit]) => isRecord(edit) && edit['saved'] === false);
      if (pending.length === 0) localStorage.removeItem(key);
      else localStorage.setItem(key, JSON.stringify(Object.fromEntries(pending)));
    } catch {
      // Left as it is.
    }
  }
}

/** Keeps the working copies in this browser; full, it makes room first; a failure (storage full or off) becomes a
 * warning. */
export function store(): void {
  const text = JSON.stringify(kept(state.edits));
  try {
    localStorage.setItem(report.storageKey, text);
    state.storageWarning = '';
    return;
  } catch {
    // Full: room is made below.
  }
  try {
    if (report.apiBase) compactOthers();
    localStorage.setItem(report.storageKey, text);
    state.storageWarning = '';
  } catch {
    state.storageWarning = report.apiBase
      ? 'This browser cannot keep changes not confirmed yet: confirm a frame before leaving it. Confirmed frames are '
        + 'saved on the server.'
      : 'Saving in this browser is not available: download the JSONL before closing.';
  }
}

/** What this browser kept: the working copies of the frames, reviewer, filter, timeline fold, hints shown. */
export function restore(): void {
  try {
    const saved: unknown = JSON.parse(localStorage.getItem(report.storageKey) || '{}');
    if (!isRecord(saved)) throw new Error('Invalid review storage');
    state.edits = saved as Record<string, FrameEdit>;
  } catch {
    state.storageWarning = 'Earlier reviews cannot be read back from this browser. Export the JSONL before closing.';
  }
  state.reviewer = readLocal('nnslr-reviewer', '') || '';
  const savedFilter = readLocal('nnslr-review-filter', 'todo');
  if (isFilter(savedFilter)) state.filter = savedFilter;
  state.showHints = readLocal('nnslr-show-hints', '1') !== '0';
}
