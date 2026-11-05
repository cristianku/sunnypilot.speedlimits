// "Car networks" (key T, its button in the inspector): what the car's cutter (SignNet) and classifier read on the
// frame on screen, run by the car's own code on the server (GET <api>/car/<segment>/<index>,
// nnslr_tools.car_reading). On, it reads every frame the page shows, a picture at 1 a second or any frame between
// them; its boxes are drawn on the photo in their own colour and listed in CarSection.vue, each with what the car
// does with it and "Use", which makes it a box of the frame. Nothing is saved: Confirm saves the frame as always.
// Hidden offline.
import {reactive, watch} from 'vue';
import {familyHasValue, holdsPlate, signKindOf} from './catalog';
import {readLocal, writeLocal} from './dom';
import {addManualBox} from './edits';
import {overlap, sameBox} from './geometry';
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
  /** The class as a box of the review. */
  review: {family: string; value_kph: number | null; sign_type?: string; supplementary_panel?: string};
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

export function carAvailable(): boolean {
  return Boolean(report.apiBase);
}

async function readCar(key: string): Promise<void> {
  const known = car.asks[key];
  if (known && known.phase !== 'failed') return;
  car.asks[key] = {phase: 'reading', reading: null, error: ''};
  try {
    const reading = await call<CarReading>('GET', report.apiBase + '/car/' + key);
    car.asks[key] = {phase: 'ready', reading, error: ''};
  } catch (error) {
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

/** A box of the car as a box of the review (family, value, kind, panel), for its name and its "Use". */
export function carDetection(sign: CarSign): Detection {
  const {family, value_kph: value, sign_type: kind, supplementary_panel: panel} = sign.review;
  const d: Detection = {bbox_xyxy: sign.box, family, value_kph: familyHasValue(family) ? value : null};
  if (kind && signKindOf(kind)) d.sign_type = kind;
  if (panel) d.supplementary_panel = panel;
  return d;
}

/** True when a box of the frame already is this one: same box, or same family, value and plate of hours over it. */
export function carSignUsed(work: FrameEdit, sign: CarSign): boolean {
  const d = carDetection(sign);
  return work.detections.some((box) => sameBox(box.bbox_xyxy, sign.box)
    || (overlap(box.bbox_xyxy, sign.box) > 0.3 && box.family === d.family && (box.value_kph ?? null) === d.value_kph
      && holdsPlate(box) === holdsPlate(d)));
}

/** "Use": the box of the car becomes a box of the frame, inside the picture, with its family, value, kind and panel. */
export function useCarSign(row: FrameRow, sign: CarSign): void {
  const d = carDetection(sign);
  const [x1, y1, x2, y2] = sign.box;
  const box: Box = [Math.max(0, x1), Math.max(0, y1), Math.min(row.width, x2), Math.min(row.height, y2)];
  if (box[2] - box[0] < 2 || box[3] - box[1] < 2) return;
  addManualBox(row, box, d.family, d.value_kph ?? null, d.sign_type ?? null, d.supplementary_panel);
}
