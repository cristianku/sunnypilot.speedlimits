// "Car networks" (key T, its button in the inspector): what the car's cutter (SignNet) and classifier read on the
// frame on screen, run by the car's own code on the server (GET <api>/car/<segment>/<index>,
// nnslr_tools.car_reading), with whom each box is for from the applicability network in use. On, it reads every frame
// the page shows, a picture at 1 a second or any frame between them, and on a frame not confirmed each box goes
// straight into the frame's boxes, the list on the right, in fuchsia (its detection_id starts with "car-"). On a
// confirmed frame they are only drawn, and listed in CarSection.vue. Nothing is saved: Confirm saves the frame as
// always, and a frame only passed over stays to review. Hidden offline.
import {reactive, watch} from 'vue';
import {isGuest} from '../common/guest';
import {familyHasValue, signKindOf} from './catalog';
import {readLocal, writeLocal} from './dom';
import {pushBox} from './edits';
import {currentRow, frameStatus, workOf} from './frames';
import {overlap} from './geometry';
import {call} from './server';
import {report, state} from './store';
import type {Box, Detection, FrameEdit, FrameRow} from './types';

/** What the car does with a box: a number for the limit, a sign of the kind of road, the green board (only an input
 * of the road-type network), nothing (read sure enough, no limit in it), or drops it (below min_confidence). */
export type CarUse = 'speed' | 'road' | 'board' | 'nothing' | 'unsure';

export interface CarSign {
  box: Box;
  /** SignNet's score of the box. */
  score: number;
  /** The classifier's class ("maximum_speed:80", "other_sign:give_way", "not_a_sign"...) and how sure it is. */
  class: string;
  confidence: number;
  speed: number | null;
  use: CarUse;
  /** Whom it applies to, from the applicability network in use and how sure: not on a box read as not a sign. */
  applicability?: string;
  applicability_score?: number;
  /** The class as a box of the review. */
  review: {family: string; value_kph: number | null; sign_type?: string; supplementary_panel?: string;
    plate_hours?: string};
}

export interface CarRelease {
  name: string;
  cutter: string | null;
  classifier: string | null;
  models_commit: string | null;
  sunnypilot_commit: string | null;
  min_confidence: number;
  min_box_score: number;
  max_boxes: number;
}

export interface CarReading {
  frame_key: string;
  signs: CarSign[];
  /** The sign the car's logs give as the frame's reading (vision.raw_reading), by its index; null: none. */
  surest: number | null;
  width: number;
  height: number;
  release: CarRelease;
  /** Why the applicability network gave no answer on this frame; null when it did. */
  applicability_note?: string | null;
  ms: number;
}

interface CarAsk {
  phase: 'reading' | 'ready' | 'failed';
  reading: CarReading | null;
  error: string;
}

// The readings kept in the page, the oldest dropped first.
const KEPT = 64;
// A frame is read once it stays on screen this long: an arrow held down goes through 20 frames a second.
const SETTLE_MS = 150;

export const car = reactive({shown: false, asks: {} as Record<string, CarAsk>});

// The frames whose boxes went to the list on the right, in this page: a box taken away there does not come back.
const placed = new Set<string>();

export function carAvailable(): boolean {
  return Boolean(report.apiBase) && !isGuest();
}

async function readCar(key: string): Promise<void> {
  const known = car.asks[key];
  if (known && known.phase !== 'failed') return;
  const apiBase = report.apiBase;
  car.asks[key] = {phase: 'reading', reading: null, error: ''};
  try {
    const reading = await call<CarReading>('GET', apiBase + '/car/' + key);
    // Another drive opened meanwhile (the editor in Speed review): its frames have the same keys.
    if (report.apiBase !== apiBase) return;
    car.asks[key] = {phase: 'ready', reading, error: ''};
  } catch (error) {
    if (report.apiBase !== apiBase) return;
    car.asks[key] = {phase: 'failed', reading: null, error: error instanceof Error ? error.message : String(error)};
  }
  const keys = Object.keys(car.asks);
  for (const old of keys.slice(0, Math.max(0, keys.length - KEPT))) {
    if (old !== state.currentKey) delete car.asks[old];
  }
}

// One reading at a time, of the last frame shown.
let wanted: string | null = null;
let reading = false;
let settle: number | undefined;

async function readWanted(): Promise<void> {
  if (!car.shown) wanted = null;
  if (reading || wanted === null) return;
  const key = wanted;
  wanted = null;
  reading = true;
  try {
    await readCar(key);
  } finally {
    reading = false;
  }
  void readWanted();
}

/** The page's choice from the last visit, and a reading of each frame shown while it is on. */
export function useCarReading(): void {
  car.shown = readLocal('nnslr-car-networks', '0') === '1';
  watch(() => [car.shown, state.currentKey] as const, ([shown, key]) => {
    window.clearTimeout(settle);
    if (!shown || key === null || !carAvailable()) {
      // Hidden: the frame waiting for its reading is not read either.
      wanted = null;
      return;
    }
    wanted = key;
    settle = window.setTimeout(() => void readWanted(), SETTLE_MS);
  }, {immediate: true});
  watch(() => (state.currentKey === null ? null : carAskOf(state.currentKey)?.reading ?? null), (reading) => {
    const row = currentRow();
    if (reading && row) placeCarSigns(row, reading);
  }, {immediate: true});
}

/** Forgets the drive's readings (the editor on the Signs side of Speed review opens another one, review/embed.ts);
 * whether the networks are shown stays. */
export function resetCar(): void {
  for (const key of Object.keys(car.asks)) delete car.asks[key];
  placed.clear();
  wanted = null;
  window.clearTimeout(settle);
}

/** T: on (the frame on screen is read now, then every frame shown) or off. */
export function toggleCar(): void {
  if (!carAvailable()) return;
  car.shown = !car.shown;
  writeLocal('nnslr-car-networks', car.shown ? '1' : '0');
}

/** The reading of a frame while the car's networks are on, else null. */
export function carAskOf(key: string): CarAsk | null {
  return car.shown ? car.asks[key] ?? null : null;
}

/** A box of the car as a box of the review (family, value, kind, panel, whom it is for). */
export function carDetection(sign: CarSign): Detection {
  const {family, value_kph: value, sign_type: kind, supplementary_panel: panel, plate_hours: hours} = sign.review;
  const d: Detection = {bbox_xyxy: sign.box, family, value_kph: familyHasValue(family) ? value : null};
  if (kind && signKindOf(kind)) d.sign_type = kind;
  if (panel) d.supplementary_panel = panel;
  if (hours) d.plate_hours = hours;
  if (sign.applicability) d.applicability = sign.applicability;
  return d;
}

/** True while a sign of the car is not among the frame's boxes: no box of the frame is on it. */
export function carSignLeft(work: FrameEdit, sign: CarSign): boolean {
  return !work.detections.some((box) => overlap(box.bbox_xyxy, sign.box) > 0.3);
}

/** What the car read becomes boxes of a frame not confirmed, once, inside the picture. */
function placeCarSigns(row: FrameRow, reading: CarReading): void {
  const status = frameStatus(row.frame_key);
  if (report.readOnly || placed.has(row.frame_key) || (status !== 'todo' && status !== 'dirty')) return;
  placed.add(row.frame_key);
  const work = workOf(row);
  for (const sign of reading.signs) {
    const [x1, y1, x2, y2] = sign.box;
    const box: Box = [Math.max(0, x1), Math.max(0, y1), Math.min(row.width, x2), Math.min(row.height, y2)];
    if (box[2] - box[0] < 2 || box[3] - box[1] < 2 || !carSignLeft(work, sign)) continue;
    const d = carDetection(sign);
    pushBox(work, box, d.family, d.value_kph ?? null, d.sign_type ?? null, d.supplementary_panel, d.plate_hours,
      d.applicability, 'car', 'car');
  }
}
