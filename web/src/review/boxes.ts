// The list of the boxes of the frame in the panel ("Signs"), as data for BoxList.vue: what each one is,
// where it comes from, what the reviewer changed, the boxes that are the same sign twice.
import type {AiAction} from '../common/ai-types';
import {CAR_BOX, describe, shortName} from './catalog';
import {proposalsOf} from './frames';
import {boxArea, overlap} from './geometry';
import type {Detection, FrameEdit, FrameRow} from './types';

/** "added by hand", "added by the AI (claude)", "model · 87%", or what the reviewer changed: "model 40 " + "→ 60". */
export interface Source {
  text: string;
  changed: string | null;
}

export interface BoxItem {
  index: number;
  detectionId: string;
  what: string;
  source: Source;
  /** Placed by "Apply previous" where the sign should be: the reviewer must look at it. */
  check: boolean;
  /** The earlier box this one repeats (same sign drawn twice), or null. */
  duplicateOf: number | null;
  /** Which road it is for, as the photo says it: ours, another, or not said yet. */
  road: 'ours' | 'another' | 'unclear';
  /** Made by the AI: its card and its box on the photo get the AI's violet-pink border. */
  ai: boolean;
}

// The proposal a reviewed box comes from, by detection_id; null for a box added by hand.
export function originalOf(row: FrameRow, d: Detection): Detection | null {
  if (!d.detection_id) return null;
  return proposalsOf(row).find((p) => p.detection_id === d.detection_id) ?? null;
}

/** A box of an AI suggestion applied: what the AI did with it. */
const AI_SOURCE: Record<AiAction, string> = {
  keep: 'AI suggestion · kept', correct: 'AI suggestion · corrected', add: 'AI suggestion · added',
  unjudged: 'AI suggestion · not judged',
};

/** Who added a box no proposal of the model had: an AI suggestion applied here, the AI reviewer that saved the frame
 * (REST API, MCP, Review with AI) when the box is in its saved review, else the reviewer, by hand. */
/** A box the AI made: an AI suggestion taken, or a box of the frame as the AI saved it (Review with AI). */
export function madeByAi(work: FrameEdit, d: Detection): boolean {
  if (d.ai_action) return true;
  const base = work.base;
  return base?.reviewer_kind === 'ai' && Boolean(d.detection_id)
    && (base.detections ?? []).some((saved) => saved.detection_id === d.detection_id);
}

export function addedBy(work: FrameEdit, d: Detection): string {
  if (d.ai_action) return 'added by the AI suggestion';
  const base = work.base;
  if (!madeByAi(work, d)) return d.detection_id?.startsWith(CAR_BOX) ? 'car networks' : 'added by hand';
  const name = String(base?.reviewer ?? '').replace(/^ai\//, '');
  return name ? 'added by the AI (' + name + ')' : 'added by the AI';
}

function sourceOf(row: FrameRow, work: FrameEdit, d: Detection): Source {
  if (d.check_position) return {text: 'from ' + d.carried_from + ' · check it', changed: null};
  if (d.carried_from) return {text: 'as in ' + d.carried_from, changed: null};
  if (d.ai_action) return {text: AI_SOURCE[d.ai_action] ?? 'AI suggestion', changed: null};
  const original = originalOf(row, d);
  if (original === null) return {text: addedBy(work, d), changed: null};
  if (original.family !== d.family || original.value_kph !== d.value_kph) {
    return {text: 'model ' + shortName(original) + ' ', changed: '→ ' + shortName(d)};
  }
  if (original.score === null || original.score === undefined) return {text: 'model', changed: null};
  return {text: 'model · ' + Math.round(Number(original.score) * 100) + '%', changed: null};
}

/** A box this much smaller than the one around it is a sign on it (a limit on a panel of lanes), not the same sign
 * drawn twice. */
const NESTED_AREA = 0.4;

/** Two boxes of about one size where one covers most of the other: the same sign drawn twice. */
function sameSign(a: Detection, b: Detection): boolean {
  const small = Math.min(boxArea(a.bbox_xyxy), boxArea(b.bbox_xyxy));
  const large = Math.max(boxArea(a.bbox_xyxy), boxArea(b.bbox_xyxy));
  return overlap(a.bbox_xyxy, b.bbox_xyxy) > 0.6 && large > 0 && small / large >= NESTED_AREA;
}

/** For each box, the earlier box it repeats (the same sign drawn twice: about its size, it covers most of it), or null.
 * A small box inside a large one is a sign on a panel, never a repeat. */
export function duplicatesOf(work: FrameEdit): (number | null)[] {
  return work.detections.map((d, i) => {
    const earlier = work.detections.slice(0, i).findIndex((other) => sameSign(other, d));
    return earlier >= 0 ? earlier : null;
  });
}

export function boxItems(row: FrameRow, work: FrameEdit): BoxItem[] {
  const duplicates = duplicatesOf(work);
  return work.detections.map((d, i) => ({
    index: i,
    detectionId: String(d.detection_id || ''),
    what: describe(d),
    source: sourceOf(row, work, d),
    check: d.check_position === true,
    duplicateOf: duplicates[i] ?? null,
    road: d.applicability === 'own_road' ? 'ours' : d.applicability === 'other_road' ? 'another' : 'unclear',
    ai: madeByAi(work, d),
  }));
}

export function boxCountText(count: number): string {
  if (count === 0) return '';
  return count === 1 ? '1 box' : count + ' boxes';
}
