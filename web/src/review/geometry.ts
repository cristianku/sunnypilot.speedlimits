// Boxes in image pixels: comparison, overlap, and where the labels on the photo go.
import type {Box} from './types';

export function isBox(value: unknown): value is Box {
  return Array.isArray(value) && value.length === 4 && value.every((item) => typeof item === 'number');
}

export function sameBox(a: unknown, b: unknown): boolean {
  if (!isBox(a) || !isBox(b)) return false;
  return Math.abs(a[0] - b[0]) <= 0.01 && Math.abs(a[1] - b[1]) <= 0.01
    && Math.abs(a[2] - b[2]) <= 0.01 && Math.abs(a[3] - b[3]) <= 0.01;
}

/** The pixels of a box the classifier gets, as crop_rectangle in training.py and the car's vision.py: rounded outwards,
 * inside the frame, at least 2 pixels on each side. */
export function cropRectangle(box: Box, width: number, height: number): Box {
  const widen = (from: number, to: number, limit: number): [number, number] => {
    const side = Math.min(2, limit);
    let low = Math.max(0, Math.floor(from));
    let high = Math.min(limit, Math.ceil(to));
    if (high - low < side) {
      low = Math.floor((low + high) / 2 - side / 2);
      high = low + side;
    }
    low = Math.max(0, Math.min(low, limit - side));
    return [low, Math.min(limit, Math.max(high, low + side))];
  };
  const [left, right] = widen(Math.min(box[0], box[2]), Math.max(box[0], box[2]), width);
  const [top, bottom] = widen(Math.min(box[1], box[3]), Math.max(box[1], box[3]), height);
  return [left, top, right, bottom];
}

export function boxesOverlap(a: Box, b: Box): boolean {
  if (a[2] < b[0] || b[2] < a[0]) return false;
  return !(a[3] < b[1] || b[3] < a[1]);
}

// Shared area over the area of the smaller box: 1 when one box contains the other.
export function overlap(a: Box, b: Box): number {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  if (x2 <= x1 || y2 <= y1) return 0;
  const shared = (x2 - x1) * (y2 - y1);
  const smaller = Math.min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]));
  if (smaller <= 0) return 0;
  return shared / smaller;
}

export function boxArea(b: Box): number {
  return Math.abs((b[2] - b[0]) * (b[3] - b[1]));
}

/** The x, y, width, height attributes of an SVG rect. */
export function rectOf(b: Box): {x: number; y: number; width: number; height: number} {
  return {x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1]};
}

/** A label already on the photo: its baseline at y, h high. */
export interface PlacedLabel {
  x: number;
  y: number;
  w: number;
  h: number;
  text: string;
  /** The box of a second-reading label, to drop a second label on the same sign. */
  box?: Box;
}

// Labels must not cover each other: move a label by whole lines until it is free (rough text width).
export function labelWidth(text: string, fontSize: number): number {
  return text.length * fontSize * 0.6;
}

function labelClash(placed: readonly PlacedLabel[], x: number, y: number, width: number, height: number): boolean {
  return placed.some((p) => x < p.x + p.w && p.x < x + width && y - height < p.y && p.y - p.h < y);
}

export function freeLabelY(placed: readonly PlacedLabel[], x: number, y: number, width: number, height: number,
  step: number): number {
  let candidate = y;
  for (let attempt = 0; attempt < 4; attempt++) {
    if (!labelClash(placed, x, candidate, width, height)) return candidate;
    candidate += step;
  }
  return candidate;
}
