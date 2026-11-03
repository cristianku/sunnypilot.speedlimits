// Changes to the frame on screen: type, value, road and details of the selected box, boxes added (drawn or
// taken from the second reading) and removed, drawing mode and second reading on or off.
import {nextTick, toRaw} from 'vue';
import {ON_VEHICLE, dropStaleKind, familyForHint, familyHasValue} from './catalog';
import {writeLocal} from './dom';
import {currentRow, selectedDetection, workOf} from './frames';
import {isBox} from './geometry';
import {refuseChange} from './busy';
import {revealEditor} from './scroll';
import {hintsOf, state, store} from './store';
import type {Box, Detection, FrameEdit, FrameRow} from './types';

/** The frame was changed here: it waits for a new confirmation. */
export function changed(row: FrameRow): void {
  const s = workOf(row);
  s.review_state = 'pending';
  s.saved = false;
  s.error = null;
  s.version = (s.version || 0) + 1;
  state.withdrawArmed = null;
  if (state.notice && state.notice.key === row.frame_key) state.notice = null;
  store();
}

// The boxes the reviewer moved or resized by hand since a fit last moved them, in this page only: C (every box of
// the frame) leaves them where they are; the Fit box of their editor still fits them.
const placedByHand = new WeakSet<Detection>();

export function markPlacedByHand(detection: Detection, placed: boolean): void {
  if (placed) placedByHand.add(toRaw(detection));
  else placedByHand.delete(toRaw(detection));
}

export function isPlacedByHand(detection: Detection): boolean {
  return placedByHand.has(toRaw(detection));
}

/** Applies a change to the selected box, if there is one, and marks the frame changed. */
function editSelected(change: (d: Detection) => void): void {
  const d = selectedDetection();
  const row = currentRow();
  if (!d || !row || refuseChange()) return;
  change(d);
  // value_kph is the compatible written-number alias. Keep original digits coherent when edited.
  if (d.sign_value !== undefined) d.sign_value = d.value_kph ?? null;
  dropStaleKind(d);
  changed(row);
}

function applyFamily(d: Detection, id: string): void {
  d.family = id;
  if (!familyHasValue(id)) d.value_kph = null;
  // The kind and "painted" belong to "Other sign" only.
  if (id !== 'other_sign') {
    delete d.sign_type;
    delete d.painted;
  }
  d.value_state = d.value_kph === null || d.value_kph === undefined ? 'unknown' : 'reviewed';
  state.valueError = false;
}

export function setFamily(id: string): void {
  editSelected((d) => applyFamily(d, id));
}

/** L: the lit display over the road, a variable speed limit, and back to a speed limit; its number stays. */
export function toggleVariable(): void {
  editSelected((d) => applyFamily(d, d.family === 'variable_display' ? 'maximum_speed' : 'variable_display'));
}

/** The most common box: an "Other sign" (its kind kept when it has one) for the road the car drives on (S). */
export function setOtherOurs(): void {
  editSelected((d) => {
    if (d.family !== 'other_sign') {
      delete d.sign_type;
      delete d.painted;
    }
    d.family = 'other_sign';
    d.value_kph = null;
    d.value_state = 'unknown';
    d.applicability = 'own_road';
    state.valueError = false;
  });
}

/** Not a sign but a sign-like disc or panel on a vehicle, such as a truck's top-speed sticker (V): a negative example
 * for the classifier, and for the decider not ours (another road), counted apart. */
export function setOnVehicle(): void {
  editSelected((d) => {
    d.family = 'not_a_sign';
    d.sign_type = ON_VEHICLE;
    d.applicability = 'other_road';
    d.value_kph = null;
    d.value_state = 'unknown';
    state.valueError = false;
  });
}

/** A kind of "Other sign" (town entry, direction →...); pressed again, the kind goes and the sign stays. */
/** The kind of the selected box, an other sign. Pressed again it stays: a press to make sure never takes it off. */
export function setKind(id: string): void {
  editSelected((d) => {
    d.family = 'other_sign';
    d.value_kph = null;
    d.value_state = 'unknown';
    d.sign_type = id;
    state.valueError = false;
  });
}

/** Painted on the road, on or off: the box becomes an "Other sign" and keeps its kind (the children's triangle with
 * "SCHULE" is "Children", painted). */
export function togglePainted(): void {
  editSelected((d) => {
    const painted = !(d.family === 'other_sign' && d.painted === true);
    if (d.family !== 'other_sign') delete d.sign_type;
    d.family = 'other_sign';
    d.value_kph = null;
    d.value_state = 'unknown';
    if (painted) d.painted = true;
    else delete d.painted;
    state.valueError = false;
  });
}

export function setValue(value: number | null): void {
  editSelected((d) => {
    if (!familyHasValue(d.family)) d.family = 'maximum_speed';
    d.value_kph = value;
    d.sign_value = value;
    d.value_state = value === null ? 'unknown' : 'reviewed';
    state.valueError = false;
  });
}

export function setField<K extends keyof Detection>(key: K, value: Detection[K]): void {
  editSelected((d) => {
    d[key] = value;
  });
}

export function removeBox(index: number): void {
  const row = currentRow();
  if (!row || refuseChange()) return;
  const work = workOf(row);
  if (index < 0 || index >= work.detections.length) return;
  work.detections.splice(index, 1);
  state.selected = -1;
  changed(row);
}

export function removeSelected(): void {
  removeBox(state.selected);
}

/** D: every box of the frame on screen goes, as an unsaved change (Discard changes brings them back; Confirm
 * then saves the frame without signs). */
export function removeAllBoxes(): void {
  const row = currentRow();
  if (!row || refuseChange()) return;
  const work = workOf(row);
  const count = work.detections.length;
  if (count === 0) return;
  work.detections.splice(0);
  state.selected = -1;
  state.drawing = false;
  changed(row);
  state.notice = {key: row.frame_key, error: false, warn: true,
    text: (count === 1 ? 'The box is' : 'All ' + count + ' boxes are') + ' removed: Confirm saves the frame without signs, Discard changes brings them back.'};
}

/** A detection id no other box of the frame has: manual-<time>[-n]. */
function manualId(work: FrameEdit): string {
  const stamp = Date.now();
  const taken = new Set(work.detections.map((d) => d.detection_id));
  let id = 'manual-' + stamp;
  for (let n = 1; taken.has(id); n++) id = 'manual-' + stamp + '-' + n;
  return id;
}

// A box added by the reviewer: drawn by hand or taken from the second reading or the car's networks. Same fields
// every way; signType: the kind of an other sign.
export function addManualBox(row: FrameRow, box: Box, family: string, value: number | null,
  signType: string | null = null): void {
  if (refuseChange()) return;
  const work = workOf(row);
  work.detections.push({
    bbox_xyxy: box,
    family,
    value_kph: value,
    value_state: value === null ? 'unknown' : 'reviewed',
    score: null,
    model_label: null,
    class_id: null,
    review_state: 'pending',
    applicability: 'unresolved',
    supplementary_panel: 'unknown',
    detection_id: manualId(work),
    ...(signType && family === 'other_sign' ? {sign_type: signType} : {}),
  });
  state.selected = work.detections.length - 1;
  state.selectedByUser = true;
  state.drawing = false;
  changed(row);
}

/** "Use": the sign of the second reading becomes a box of the frame. */
export function useHint(index: number): void {
  const row = currentRow();
  if (!row) return;
  const hint = hintsOf(row.frame_key)[index];
  if (!hint || !isBox(hint.box)) return;
  const family = familyForHint(hint.type);
  const value = familyHasValue(family) && Number.isInteger(hint.value_kph) ? hint.value_kph ?? null : null;
  addManualBox(row, [...hint.box], family, value);
}

/** Physical site or encounter of the frame (under "More"); empty: none. */
export function setFrameField(row: FrameRow, key: 'site_group' | 'encounter_id', value: string): void {
  if (refuseChange()) return;
  const trimmed = value.trim();
  workOf(row)[key] = trimmed === '' ? null : trimmed;
  changed(row);
}

/** A box chosen in the list (or on the photo, drawing: false there already): its editor opens. */
export function selectBox(index: number): void {
  state.selected = index;
  state.selectedByUser = true;
  state.drawing = false;
  void nextTick(revealEditor);
}

export function toggleDrawing(): void {
  if (!state.drawing && refuseChange()) return;
  state.drawing = !state.drawing;
  state.selected = -1;
}

export function toggleHints(): void {
  state.showHints = !state.showHints;
  writeLocal('nnslr-show-hints', state.showHints ? '1' : '0');
}

/** Esc: no drawing, no box selected. */
export function cancelAll(): void {
  state.drawing = false;
  state.selected = -1;
}
