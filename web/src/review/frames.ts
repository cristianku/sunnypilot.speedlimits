// Frames and their review state: the working copy, the status of a frame, its doubts, the filters.
import {hasValue, isSpeedFamily} from './catalog';
import {clone} from './dom';
import {READ_INDEX_BASE} from './order';
import {indexOf, isVirtual, report, rowOf, state} from './store';
import {trialKeys} from './trial';
import type {
  ConfirmedCopy, Detection, Filter, FrameEdit, FrameRow, FrameStatus, QualityFilter, SuspiciousReason,
} from './types';

export function currentRow(): FrameRow | null {
  return state.currentKey === null ? null : rowOf(state.currentKey);
}

/** The second of its segment's video a frame was taken at, else null. */
export function segmentSecond(row: FrameRow): number | null {
  const fps = row.requested_sample_fps;
  const second = typeof row.media_time_s === 'number' ? row.media_time_s
    : typeof fps === 'number' && fps > 0 ? row.output_index / fps : null;
  return second === null || !Number.isFinite(second) ? null : second;
}

/** A frame between the pictures at 1 a second: a sign the comma read there, or a frame of the video (video.ts). */
export function isBetweenFrame(row: FrameRow): boolean {
  return row.output_index >= READ_INDEX_BASE;
}

/** The frame of a sign the comma's vision read (nnslr_tools.sign_reads): only those carry the value it read. */
export function isReadFrame(row: FrameRow): boolean {
  return 'comma_read_kph' in row;
}

/** "the comma read 20 here" (its value unknown: "the comma read a sign here"). */
export function commaReadText(row: FrameRow): string {
  const kph = row.comma_read_kph;
  return 'the comma read ' + (typeof kph === 'number' ? kph : 'a sign') + ' here';
}

/** The first segment of the drive's frames: the drive's 0:00, as Speed review's player counts (a drive whose first
 * minutes were not uploaded starts at its first segment there, not at segment 0). */
let firstOf: {rows: readonly FrameRow[]; first: number} = {rows: [], first: 0};
function firstSegment(): number {
  if (firstOf.rows !== report.rows) {
    let first = Infinity;
    for (const row of report.rows) if (row.segment_index < first) first = row.segment_index;
    firstOf = {rows: report.rows, first: Number.isFinite(first) ? first : 0};
  }
  return firstOf.first;
}

/** How far into the drive the frame is, "12:46" (1:02:05 past an hour), on the clock of Speed review's player: the
 * minutes since the drive's first segment and the second in it; null when the extraction did not record the second. */
export function driveTime(row: FrameRow): string | null {
  const second = segmentSecond(row);
  if (second === null) return null;
  const total = Math.max(0, Math.floor((row.segment_index - firstSegment()) * 60 + second + 1e-6));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const pad = (value: number): string => String(value).padStart(2, '0');
  // A frame between two seconds (20 a second): "19:32.85".
  const hundredths = isBetweenFrame(row) ? '.' + pad(Math.floor(second * 100 + 1e-6) % 100) : '';
  return (hours > 0 ? hours + ':' + pad(minutes) : String(minutes)) + ':' + pad(total % 60) + hundredths;
}

/** "12:46 into the drive", else the segment and the image ("Segment 12 · image 46"). */
export function framePlace(row: FrameRow): string {
  const time = driveTime(row);
  return time === null ? 'Segment ' + row.segment_index + ' · image ' + row.output_index : time + ' into the drive';
}

export function proposalsOf(row: FrameRow): Detection[] {
  return Array.isArray(row.detections) ? row.detections : [];
}

/** The boxes a frame starts with: its proposals, numbered from left to right as they appear on the photo. */
export function startingDetections(row: FrameRow): Detection[] {
  const detections = clone(proposalsOf(row));
  detections.sort((a, b) => a.bbox_xyxy[0] - b.bbox_xyxy[0]);
  return detections;
}

// The page's working copy of a frame: the proposals until the reviewer changes or confirms it.
export function workOf(row: FrameRow): FrameEdit {
  const key = row.frame_key;
  let work = state.edits[key];
  if (!work) {
    const detections = startingDetections(row);
    state.edits[key] = {review_state: 'pending', detections, saved: true, revision: 0};
    // The reactive copy: changes made through it reach the page.
    work = state.edits[key];
  }
  return work as FrameEdit;
}

/** The country of a frame: the one picked, else the one its GPS position gives. */
export function countryOf(row: FrameRow, s: FrameEdit | undefined): string | null {
  if (s && s.country !== undefined) return s.country;
  return row.country_proposed ?? null;
}

/** The box selected in the frame on screen. */
export function selectedDetection(): Detection | null {
  const row = currentRow();
  if (!row) return null;
  return workOf(row).detections[state.selected] ?? null;
}

/** The working copy of a review read from the server (or of the last confirmed version, to go back to it). */
export function fromServer(item: ConfirmedCopy): FrameEdit {
  return {
    review_state: item.confirmed ? 'reviewed' : 'pending',
    detections: clone(item.detections ?? []),
    reviewer: item.reviewer,
    reviewer_kind: item.reviewer_kind || 'human',
    needs_person: item.needs_person === true,
    channel: item.channel,
    review_timestamp: item.review_timestamp,
    site_group: item.site_group,
    encounter_id: item.encounter_id,
    // A review saved before the country existed has none: the GPS proposal stands.
    country: item.country,
    users_complete: item.users_complete === true,
    revision: item.revision,
    saved: true,
    error: null,
    server_confirmed: item.confirmed,
    base: clone(item),
  };
}

// True when the working copy is exactly what the server has confirmed (an agent may edit it without changed()).
export function sameAsServer(s: FrameEdit | undefined): boolean {
  if (!s || !s.base) return false;
  if (JSON.stringify(s.detections) !== JSON.stringify(s.base.detections || [])) return false;
  if ((s.site_group ?? null) !== (s.base.site_group ?? null)) return false;
  if (s.country !== undefined && (s.country ?? null) !== (s.base.country ?? null)) return false;
  if ((s.users_complete === true) !== (s.base.users_complete === true)) return false;
  return (s.encounter_id ?? null) === (s.base.encounter_id ?? null);
}

export function isConfirmed(key: string): boolean {
  const s = state.edits[key];
  if (!s) return false;
  if (report.apiBase) return s.server_confirmed === true;
  return s.review_state === 'reviewed';
}

export function reasonsOf(key: string): SuspiciousReason[] {
  if (isConfirmed(key)) return [];
  const list = state.suspicious[key];
  return Array.isArray(list) ? list : [];
}

export function frameStatus(key: string): FrameStatus {
  if (report.apiBase && state.inFlight.has(key)) return 'saving';
  const s = state.edits[key];
  if (!s) return 'todo';
  if (s.review_state === 'reviewed') {
    if (!report.apiBase) return 'done';
    // Saved by the site's AI reviewer and not changed here: still to confirm, by a person.
    if (s.needs_person && s.saved === true && s.server_confirmed === true && sameAsServer(s)) return 'todo';
    if (s.saved === true && s.server_confirmed === true) return 'done';
    return 'error';
  }
  if (s.saved === false) return 'dirty';
  return 'todo';
}

// Confirmed on the server by an AI reviewer (REST API or MCP) and not changed here. The confirmation is final,
// like a person's: this only shows where it comes from (filter, striped tick, save line).
export function aiConfirmed(key: string): boolean {
  const s = state.edits[key];
  return Boolean(report.apiBase && s && s.reviewer_kind === 'ai' && frameStatus(key) === 'done' && sameAsServer(s));
}

export function statusLabel(key: string): string {
  const status = frameStatus(key);
  const s = state.edits[key];
  if (status === 'done' && aiConfirmed(key)) return 'Confirmed · AI';
  if (status === 'todo' && s?.needs_person) return 'AI · to confirm';
  if (status === 'done') return 'Confirmed';
  if (status === 'saving') return 'Saving…';
  if (status === 'error') return s && s.error ? 'Not saved' : 'Waiting';
  // Not "To confirm": too close to the "To review" filter beside it.
  if (status === 'dirty') return 'Unsaved changes';
  if (reasonsOf(key).length > 0) return 'Suspicious';
  return 'To review';
}

export interface Doubts {
  unknown: number;
  unreadable: number;
  road: number;
}

// Doubts left in a confirmed frame: a speed sign without its number, an unreadable sign, a road not chosen.
export function doubtsOf(key: string): Doubts {
  const out = {unknown: 0, unreadable: 0, road: 0};
  const s = state.edits[key];
  if (!s || !isConfirmed(key)) return out;
  for (const d of s.detections) {
    const speed = isSpeedFamily(d.family);
    if (d.family === 'unreadable') {
      out.unreadable += 1;
    } else if ((speed || d.family === 'road_marking_candidate') && !hasValue(d)) {
      out.unknown += 1;
    }
    if (speed && (!d.applicability || d.applicability === 'unresolved')) out.road += 1;
  }
  return out;
}

export function isQualityFilter(filter: string | undefined): filter is QualityFilter {
  return filter === 'unknown' || filter === 'unreadable' || filter === 'road';
}

// Filters over confirmed frames: a confirmation does not take the frame out of them by itself.
export function confirmedFilter(filter: string): boolean {
  return isQualityFilter(filter) || filter === 'ai';
}

// "AI trial" is not kept for the next visit: its frames are known only once the page has read the trial.
export function isFilter(value: string | null | undefined): value is Filter {
  return value === 'todo' || value === 'suspicious' || value === 'proposals' || value === 'all'
    || (value !== null && value !== undefined && confirmedFilter(value));
}

export function matchesFilter(row: FrameRow, filter: Filter): boolean {
  const key = row.frame_key;
  switch (filter) {
    case 'todo': return frameStatus(key) !== 'done';
    case 'suspicious': return reasonsOf(key).length > 0;
    case 'proposals': return proposalsOf(row).length > 0;
    case 'ai': return aiConfirmed(key);
    case 'trial': return trialKeys.has(key) && frameStatus(key) !== 'done';
    case 'unknown': return doubtsOf(key).unknown > 0;
    case 'unreadable': return doubtsOf(key).unreadable > 0;
    case 'road': return doubtsOf(key).road > 0;
    default: return true;
  }
}

// Frames of the current filter, plus the frame on screen so that it never disappears under the reviewer.
export function visibleRows(): FrameRow[] {
  return report.rows.filter((row) => row.frame_key === state.currentKey || matchesFilter(row, state.filter));
}

/** Frames confirmed here and not stored on the server yet (a frame of the video between the seconds too: the
 * server takes it into the report when it is saved). */
export function unsavedKeys(): string[] {
  const keys = report.rows.map((row) => row.frame_key)
    .concat(Object.keys(state.edits).filter((key) => indexOf(key) < 0 && isVirtual(key)));
  return keys.filter((key) => {
    const s = state.edits[key];
    return s && s.saved === false && s.review_state === 'reviewed';
  });
}
