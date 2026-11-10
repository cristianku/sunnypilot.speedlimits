// Recent: the labels already given to boxes of this drive, offered again in one tap for the selected box (the row
// under the zoom). A label is what the sign is: type, number, kind, painted, road, panel and its hours, temporary;
// never where the box is nor how well the sign is seen.
//   - From what the page holds: the frames confirmed on the server (all of them, read when the page opens) and this
//     page's edits. In a frame not confirmed only the boxes the reviewer changed count, never a proposal left as the
//     model made it.
//   - Nearest first: the other boxes of the frame on screen, the frames before it, then the ones after.
import {originalOf} from './boxes';
import {ROADS, PANELS, describe, familyHasValue, shortName, signKindOf} from './catalog';
import {isConfirmed} from './frames';
import {anchorIndex, indexOf, isVirtual, report, state} from './store';
import type {Detection, FrameRow} from './types';

/** How many at most: one line under the zoom. */
const SHOWN = 6;

export const LABEL_FIELDS = ['family', 'value_kph', 'sign_unit', 'sign_type', 'painted', 'applicability',
  'supplementary_panel', 'plate_hours', 'temporary', 'variable_display'] as const;
export type Label = Pick<Detection, (typeof LABEL_FIELDS)[number]>;

export interface Recent {
  /** The same for the same label. */
  key: string;
  /** The number of a speed sign, drawn as a small disc before the text; null for the other signs. */
  disc: number | null;
  text: string;
  title: string;
  label: Label;
}

/** The label of a box, its empty fields left out: a number on a type that has none is no part of it. */
export function labelOf(d: Detection): Label {
  const label: Label = {family: d.family};
  for (const field of LABEL_FIELDS) {
    const value = d[field];
    if (value !== undefined && value !== null) Object.assign(label, {[field]: value});
  }
  if (!familyHasValue(d.family)) delete label.value_kph;
  return label;
}

export function labelKey(label: Label): string {
  return JSON.stringify(LABEL_FIELDS.map((field) => label[field] ?? null));
}

// Our road, the usual one, goes without saying on a chip.
const ROAD_SHORT: Record<string, string> = {other_road: 'another', unresolved: 'unclear'};

function asDetection(label: Label): Detection {
  return {...label, bbox_xyxy: [0, 0, 0, 0]};
}

function recentOf(label: Label): Recent {
  const d = asDetection(label);
  const disc = label.family === 'maximum_speed' && label.value_kph != null && label.sign_unit !== 'mph'
    ? label.value_kph : null;
  const parts: string[] = [];
  if (disc === null) {
    const kind = label.family === 'other_sign' ? signKindOf(label.sign_type) : null;
    const name = kind !== null ? kind.chip : label.family === 'other_sign' ? 'Other sign' : shortName(d);
    parts.push((label.painted === true ? 'Painted ' : '') + name.charAt(0).toUpperCase() + name.slice(1));
  }
  if (label.supplementary_panel === 'time') parts.push(label.plate_hours ?? 'hours');
  else if (label.supplementary_panel === 'wet') parts.push('wet');
  else if (label.supplementary_panel === 'present') parts.push('panel');
  if (label.temporary === true) parts.push('temporary');
  const road = ROAD_SHORT[label.applicability ?? 'unresolved'];
  if (road) parts.push(road);
  const roadName = ROADS.find((item) => item.id === label.applicability)?.label ?? 'Unclear';
  const panel = PANELS.find((item) => item.id === label.supplementary_panel)?.label;
  const title = describe(d) + ' · road: ' + roadName.toLowerCase() + (panel ? ' · panel below: ' + panel.toLowerCase() : '')
    + (label.temporary === true ? ' · temporary' : '');
  return {key: labelKey(label), disc, text: parts.join(' · '), title, label};
}

/** The boxes of a frame labelled by a reviewer, but the one left out (the selected box of the frame on screen). */
function labelled(row: FrameRow, skip: number): Detection[] {
  const work = state.edits[row.frame_key];
  if (!work) return [];
  const confirmed = isConfirmed(row.frame_key);
  return work.detections.filter((d, i) => {
    if (i === skip || !d.family) return false;
    if (confirmed) return true;
    const original = originalOf(row, d);
    return original === null || labelKey(labelOf(original)) !== labelKey(labelOf(d));
  });
}

/** Up to SHOWN labels, nearest first, each once (two that would read the same: the nearest). */
export function recentLabels(row: FrameRow, selected: number): Recent[] {
  const found: Recent[] = [];
  const seen = new Set<string>();
  const take = (rows: Iterable<Detection>): boolean => {
    for (const d of rows) {
      const recent = recentOf(labelOf(d));
      const shown = recent.disc + '|' + recent.text;
      if (seen.has(shown)) continue;
      seen.add(shown);
      found.push(recent);
      if (found.length >= SHOWN) return true;
    }
    return false;
  };
  if (take(labelled(row, selected))) return found;
  // A frame of the video not saved sits just after the frame of the report before it.
  const key = row.frame_key;
  const here = isVirtual(key) ? anchorIndex(key) + 0.5 : indexOf(key);
  for (let i = Math.ceil(here) - 1; i >= 0; i -= 1) {
    if (take(labelled(report.rows[i] as FrameRow, -1))) return found;
  }
  for (let i = Math.floor(here) + 1; i < report.rows.length; i += 1) {
    if (take(labelled(report.rows[i] as FrameRow, -1))) return found;
  }
  return found;
}
