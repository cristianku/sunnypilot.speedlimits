// The data of the review page: the payload pages.render_review embeds in the page, the answers of the review
// API (nnslr_tools.review: GET /api/route/<id>/reviews, POST /api/route/<id>/frame/<segment>/<image>) and
// the page's own working copy of each frame, kept in localStorage.
import type {AiAction} from '../common/ai-types';

/** x1, y1, x2, y2 in pixels of the full image. */
export type Box = [number, number, number, number];

/** One box: a proposal of the model (report row) or the reviewer's version of it. Proposals carry more
 * fields (evidence_source...): they are kept as they are and sent back unchanged. */
export interface Detection {
  bbox_xyxy: Box;
  family: string;
  value_kph?: number | null | undefined;
  sign_value?: number | null | undefined;
  sign_unit?: 'km/h' | 'mph' | 'unknown' | null | undefined;
  value_state?: 'unknown' | 'reviewed';
  score?: number | null;
  model_label?: string | null;
  class_id?: number | null;
  review_state?: string;
  applicability?: string;
  supplementary_panel?: string;
  /** The hours read on a plate of hours, "22-6" (nnslr_tools.sign_types.PLATE_HOURS); absent or null: not said. */
  plate_hours?: string | null;
  visibility?: string;
  occlusion?: string;
  temporary?: boolean | null;
  variable_display?: boolean | null;
  /** What an "Other sign" is: town_entry, direction_right... (catalog.SIGN_KIND_GROUPS); null: not said. */
  sign_type?: string | null;
  /** An "Other sign" painted on the road (the children's triangle with "SCHULE"): true, or absent. */
  painted?: boolean | null;
  notes?: string | null;
  detection_id?: string;
  /** "Apply previous": the frame its labels come from. The server drops it when saving. */
  carried_from?: string;
  /** "Apply previous": the box is where the sign should be, a guess. No save accepts it until the reviewer has
   * looked (moved the box onto the sign, or pressed "Here") and the flag is gone. */
  check_position?: boolean;
  /** An AI suggestion applied (review/ai.ts): what the AI did with the box. The page shows it; the server keeps
   * only the labels when saving, so the frame is saved as the reviewer's own review. */
  ai_action?: AiAction;
}

/** One frame of the pre-annotation report. The rows carry more fields (hashes, models...): the JSONL export
 * copies them as they are. */
export interface FrameRow {
  frame_key: string;
  segment_index: number;
  output_index: number;
  width: number;
  height: number;
  /** Relative to the page: images/<n>.jpg */
  preview_path: string;
  detections?: Detection[];
  site_group?: string | null;
  encounter_id?: string | null;
  /** Where the car was (nnslr gps), null without a fix; missing when the GPS was not read for the route. */
  gps?: {latitude: number; longitude: number; speed_mps: number | null; bearing_deg: number | null} | null;
  /** The country of that position (ISO code), the one saved unless the reviewer picks another. */
  country_proposed?: string | null;
  /** gps: from this frame's position; nearest_fix: from the nearest frame with one. */
  country_from?: string | null;
  /** Seconds into its segment (one minute each), when the extraction knows it. */
  media_time_s?: number | null;
  /** The frame of its segment's video this picture is (20 a second), when the extraction knows it. */
  decoded_frame_index?: number | null;
  requested_sample_fps?: number | null;
  /** The value of the sign the comma read at this frame (km/h; null: a sign without a number), only on the frames
   * added at its reads (output index 1000 + the frame in the video). */
  comma_read_kph?: number | null;
}

/** One sign seen by the second reading (nnslr_tools.review_hints). */
export interface Hint {
  type: string;
  value_kph?: number | null;
  box: Box;
  confidence: string;
  /** "A" or "B": the two passes of the second reading. */
  pass?: string;
  side?: string;
  note?: string;
  source?: string;
}

export interface HintStats {
  files: number;
  lines: number;
  frames: number;
  signs: number;
  skipped_files: number;
  skipped_lines: number;
  skipped_signs: number;
  unknown_frames: number;
}

/** Why a frame is suspicious: rule a..d of review_hints.suspicious_frames, and the Italian text. */
export interface SuspiciousReason {
  rule: string;
  text: string;
}

export type Suspicious = Record<string, SuspiciousReason[]>;

export interface RouteLink {
  route_id: string;
  url: string;
  /** The drive: "Herznach → Zürich · Fri 23 Oct 2026 · 08:01"; null when nothing is known of it. */
  name?: string | null;
  /** The same apart, for the two lines of the menu: "Herznach → Zürich", "Fri 23 Oct 2026 · 08:01", 49. */
  title?: string | null;
  when?: string | null;
  minutes?: number | null;
  /** Recorded at 2688 x 1520 (drives.full_definition): its pictures here are sharper than the frame's pixels. */
  full_definition?: boolean;
  /** A dashcam drive, sunnypilot off in all of it (review.dashcam_of); null before its logs are read. */
  dashcam?: boolean | null;
}

/** The JSON in <script id="payload">. server is false in the static index.html of write_review (offline). */
export interface Payload {
  run_id: string;
  route_id: string;
  server: boolean;
  frames: FrameRow[];
  hints: Record<string, Hint[]>;
  hint_stats: HintStats;
  suspicious: Suspicious;
  routes: RouteLink[];
  /** The countries the review knows, code: English name (nnslr_tools.gps.country_names). */
  countries?: Record<string, string>;
  /** Who proposed the signs: vietnam, ours or qwen; null offline (not known). */
  proposer?: string | null;
  /** Propose again works on this drive: read only (nnslr_tools.reproposing.public); null otherwise. */
  busy?: Busy | null;
  /** The segments with their road video on the server, with its frames when counted (review_service.videos); null
   * offline. */
  videos?: Record<string, number | null> | null;
  /** The seconds kept of the segments kept only at a car park's speed (review_service.kept_seconds): a frame of their
   * video in another second cannot be saved. Absent before the server sent it. */
  kept_seconds?: Record<string, number[]>;
}

/** A drive Propose again works on, read only until its new proposals are in. */
export interface Busy {
  phase: 'waiting' | 'proposing again';
  /** Who proposes ("Qwen3-VL"), null when not known. */
  proposer: string | null;
  message: string;
  /** At work: the frames proposed so far, of the frames of its current report. */
  done?: number;
  total?: number | null;
  /** At work: the seconds left for the drive (an estimate), null when not known yet. */
  left_seconds?: number | null;
  /** At work: the frames being proposed now, how long such a step takes and the seconds since the last one was
   * written; null when the steps are too quick to show or not known yet. */
  step?: {frames: number; seconds: number; since: number} | null;
  /** Waiting: the drives before it in the batch. */
  ahead?: number;
}

/** One saved frame as the server sends it (review_db.public_review). */
export interface ServerReview {
  route_id: string;
  run_id: string;
  frame_key: string;
  confirmed: boolean;
  revision: number;
  reviewer: string | null;
  reviewer_kind: 'human' | 'ai';
  /** web, rest or mcp. */
  channel: string;
  updated_at: string;
  /** Saved by the site's AI reviewer (Review with AI): shown to confirm, left out of training until a person does. */
  needs_person?: boolean;
  review_timestamp: string | null;
  detections: Detection[];
  site_group: string | null;
  encounter_id: string | null;
  country?: string | null;
}

/** GET /api/route/<id>/reviews */
export interface ReviewsAnswer {
  route_id: string;
  run_id: string;
  frame_count: number;
  confirmed_count: number;
  reviews: ServerReview[];
  suspicious: Suspicious;
}

/** POST /api/route/<id>/frame/<key>: a confirmation or a withdrawal stored. */
export interface SaveAnswer {
  ok: true;
  review: ServerReview;
  confirmed_count: number;
  suspicious: Suspicious;
}

/** POST /api/route/<id>/frame/<key>/apply-previous: the boxes of the frame with the previous one carried over
 * (review_service.apply_previous). Nothing is saved. */
export interface CarryAnswer {
  frame_key: string;
  source_frame_key: string;
  detections: Detection[];
  matched: number;
  estimated: number;
  dropped: number;
  kept: number;
  /** Model boxes removed: the same sign as a carried box. */
  duplicates: number;
  /** 1-based numbers of the boxes to check. */
  to_check: number[];
  message: string;
}

/** An error answer of the API (400, 403, 409 revision_conflict or stale_report...). */
export interface Problem {
  ok?: false;
  error?: string;
  message?: string;
}

/** The last confirmed version of a frame: on the server (the whole review), or offline the confirmation made
 * in this browser (detections, groups, reviewer and time only). */
export type ConfirmedCopy = Partial<ServerReview>;

/** The page's working copy of a frame, kept in localStorage: the proposals until the reviewer changes or
 * confirms it. Old copies in a browser may miss any of the optional fields. */
export interface FrameEdit {
  review_state: 'pending' | 'reviewed';
  detections: Detection[];
  /** false: changed here, or confirmed here and not yet on the server. */
  /** The server's copy was saved by the site's AI reviewer: a person still has to confirm it. */
  needs_person?: boolean;
  saved?: boolean | undefined;
  /** The server revision these edits are based on (0: not on the server). */
  revision?: number | undefined;
  /** Counts the edits, to tell whether the frame changed while it was being saved. */
  version?: number | undefined;
  error?: string | null | undefined;
  /** false after an error the automatic retry should not repeat (4xx). */
  retry?: boolean | undefined;
  reviewer?: string | null | undefined;
  reviewer_kind?: 'human' | 'ai' | undefined;
  channel?: string | undefined;
  review_timestamp?: string | null | undefined;
  site_group?: string | null | undefined;
  encounter_id?: string | null | undefined;
  /** The country picked for the frame; undefined: the one its GPS gives (FrameRow.country_proposed). */
  country?: string | null | undefined;
  server_confirmed?: boolean | undefined;
  base?: ConfirmedCopy | null | undefined;
}

/** Which frames the arrows walk through: the top filters, the AI ones and the doubt counters. trial: the frames of
 * the route's last "Try without saving" still to review. */
export type Filter = 'todo' | 'suspicious' | 'proposals' | 'all' | 'ai' | 'trial' | 'unknown' | 'unreadable' | 'road';
export type QualityFilter = 'unknown' | 'unreadable' | 'road';

/** todo, dirty (edited, not confirmed), saving, error (confirmed, not on the server yet) or done. */
export type FrameStatus = 'todo' | 'dirty' | 'saving' | 'error' | 'done';

/** body[data-server] for whoever drives the page. */
export type ServerState = 'offline' | 'loading' | 'ready' | 'stale' | 'error';

export type SaveResult = 'saving' | 'ok' | 'error' | 'unchanged' | 'need-reviewer' | 'need-check' | 'need-duplicate-check'
  | 'withdrawn';

/** The outcome of the last save, as body data attributes. */
export interface LastSave {
  key: string;
  result: SaveResult;
  revision: number;
}

/** A message for one frame on the status line (and the phone bar when it is an error). */
export interface Notice {
  key: string;
  text: string;
  error: boolean;
  warn?: boolean;
}

export type Point = [number, number];
export type Corner = 'nw' | 'ne' | 'sw' | 'se';

/** The two pictures a pointer can work on: the photo and the zoom of the selected box. */
export type Surface = 'scene' | 'zoom';

/** A pointer drag on the photo or on the zoom: drawing a new box, moving a corner of the selected one, or
 * moving the whole selected box (box: where it was when the drag started). */
export type Drag =
  | {kind: 'draw'; svg: Surface; start: Point}
  | {kind: 'resize'; svg: Surface; corner: Corner; moved: boolean}
  | {kind: 'move'; svg: Surface; start: Point; box: Box; moved: boolean};
