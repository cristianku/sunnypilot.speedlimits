// What is drawn on the photo, as data for Stage.vue and BoxEditor.vue: the boxes with their dark under-stroke
// and labels, the corner handles of the selected box, the boxes of the second reading, the AI suggestion not
// applied yet, and the view of the zoom of the selected box. Sizes are in image pixels: k is the on-screen size
// of one image pixel.
import type {AiAction, AiSuggestion} from '../common/ai-types';
import type {CarSign, CarUse} from './car';
import {boxColor, hintLabel, shortName} from './catalog';
import {boxArea, boxesOverlap, freeLabelY, labelWidth, rectOf, type PlacedLabel} from './geometry';
import type {Box, Corner, Detection, Hint} from './types';

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** A label on the photo: SVG text with its halo. */
export interface Label {
  x: number;
  y: number;
  text: string;
  fontSize: number;
  strokeWidth: number;
}

export interface DrawnBox {
  index: number;
  /** box, proposal (dashed), selected. */
  cls: string;
  rect: Rect;
  /** The family's colour; none on the selected box (the CSS colours it). */
  stroke: string | null;
  label: Label;
  /** For our road: the path of the road drawn before the label (two lanes narrowing to the horizon); else null. */
  road: string | null;
}

export interface DrawnHint {
  confidence: string;
  rect: Rect;
  /** null: the other pass of the second reading already labelled the same sign. */
  label: Label | null;
}

export interface Handle {
  corner: Corner;
  /** The visible square and the larger area that takes the pointer. */
  mark: Rect;
  hit: Rect;
}

/** A box of the AI suggestion: one it keeps, corrects or adds, or one of the frame it would delete. */
export interface DrawnAi {
  kind: AiAction | 'delete';
  rect: Rect;
  label: Label;
}

/** A box of the car's networks (car.ts): what the car does with it, and whether it is the frame's reading. */
export interface DrawnCar {
  use: CarUse;
  surest: boolean;
  rect: Rect;
  label: Label;
}

export interface StageDrawing {
  boxes: DrawnBox[];
  hints: DrawnHint[];
  handles: Handle[];
}

/** On-screen pixels per image pixel of an SVG of this on-screen size showing width x height. */
export function sceneScale(size: {width: number; height: number}, width: number, height: number): number {
  if (!size.width || !size.height || !width || !height) return 1;
  return Math.min(size.width / width, size.height / height);
}

function label(text: string, x: number, y: number, fontSize: number, k: number): Label {
  return {x, y, text, fontSize, strokeWidth: 3.5 / k};
}

/** The road icon of a box for our road, from x on the baseline y of a label of ``size``: two lane lines
 * narrowing to the horizon and a dash of the centre line between them, ROAD_ICON times the size wide. */
const ROAD_ICON = 1.25;
function roadIcon(x: number, y: number, size: number): string {
  const bottom = y + size * 0.12;
  const top = y - size * 0.95;
  const middle = x + size * ROAD_ICON / 2;
  const at = (value: number): string => value.toFixed(2);
  return 'M' + at(x) + ' ' + at(bottom) + 'L' + at(middle - size * 0.16) + ' ' + at(top)
    + 'M' + at(x + size * ROAD_ICON) + ' ' + at(bottom) + 'L' + at(middle + size * 0.16) + ' ' + at(top)
    + 'M' + at(middle) + ' ' + at(bottom - size * 0.05) + 'L' + at(middle) + ' ' + at(bottom - size * 0.42);
}

const CORNERS: readonly Corner[] = ['nw', 'ne', 'sw', 'se'];

function cornerPoint(b: Box, corner: Corner): [number, number] {
  const x = corner === 'nw' || corner === 'sw' ? b[0] : b[2];
  const y = corner === 'nw' || corner === 'ne' ? b[1] : b[3];
  return [x, y];
}

// On the photo the corners sit outside the box, so they never cover a small sign.
export function handlesOf(b: Box, size: number, hit: number, outside: boolean): Handle[] {
  return CORNERS.map((corner) => {
    const [x, y] = cornerPoint(b, corner);
    let hx = x - size / 2;
    let hy = y - size / 2;
    if (outside) {
      hx = corner === 'nw' || corner === 'sw' ? x - size : x;
      hy = corner === 'nw' || corner === 'ne' ? y - size : y;
    }
    return {
      corner,
      mark: {x: hx, y: hy, width: size, height: size},
      hit: {x: x - hit / 2, y: y - hit / 2, width: hit, height: hit},
    };
  });
}

/** Where a label starts: at the left of its box, moved left when it would run off the right edge of the photo. */
function labelX(left: number, width: number, imageWidth: number, k: number): number {
  return Math.max(0, Math.min(left, imageWidth - width - 4 / k));
}

/** Boxes, labels, hints and handles of a frame shown at scale k. */
export function drawStage(detections: readonly Detection[], selected: number, hints: readonly Hint[],
  imageHeight: number, k: number, coarse: boolean, imageWidth = Number.POSITIVE_INFINITY): StageDrawing {
  const fontSize = 14 / k;
  const placed: PlacedLabel[] = [];
  const handleSize = 8 / k;
  const hitSize = (coarse ? 36 : 20) / k;
  // Larger boxes first, so that a small box inside a large one stays clickable.
  const order = detections.map((_, index) => index);
  const areaAt = (index: number): number => {
    const d = detections[index];
    return d ? boxArea(d.bbox_xyxy) : 0;
  };
  order.sort((a, b) => areaAt(b) - areaAt(a));
  const boxes: DrawnBox[] = [];
  for (const i of order) {
    const d = detections[i];
    if (d === undefined) continue;
    const b = d.bbox_xyxy;
    const isSelected = i === selected;
    let cls = 'box';
    if (d.model_label) cls += ' proposal';
    if (d.check_position) cls += ' check';
    if (isSelected) cls += ' selected';
    // The road too, when said: our road in capitals after a road icon ("⟋⟍ 2 · 50 · OURS"), another in words.
    const ours = d.applicability === 'own_road';
    const road = ours ? ' · OURS' : d.applicability === 'other_road' ? ' · another' : '';
    const text = (i + 1) + ' · ' + shortName(d) + road + (d.check_position ? ' · check' : '');
    const icon = ours ? fontSize * (ROAD_ICON + 0.4) : 0;
    const width = labelWidth(text, fontSize) + icon;
    const x = labelX(b[0], width, imageWidth, k);
    // The selected box has its corners outside: its label goes above them.
    const top = isSelected ? b[1] - handleSize - fontSize * 0.2 : b[1] - fontSize * 0.35;
    const y = Math.max(fontSize, freeLabelY(placed, x, Math.max(fontSize, top), width, fontSize, -fontSize));
    placed.push({x, y, w: width, h: fontSize, text});
    boxes.push({
      index: i, cls, rect: rectOf(b), stroke: isSelected ? null : d.check_position ? 'var(--warn)' : boxColor(d),
      label: label(text, x + icon, y, fontSize, k), road: ours ? roadIcon(x, y, fontSize) : null,
    });
  }
  const drawnHints: DrawnHint[] = [];
  const hintFont = 12 / k;
  for (const h of hints) {
    const b = h.box;
    const text = hintLabel(h);
    const width = labelWidth(text, hintFont);
    let hintText: Label | null = null;
    // The two passes often see the same sign: one label is enough.
    if (!placed.some((p) => p.text === text && p.box && boxesOverlap(p.box, b))) {
      // Below the box, where the label of a reviewed box never is.
      const x = labelX(b[0], width, imageWidth, k);
      const y = Math.min(imageHeight, freeLabelY(placed, x, b[3] + hintFont * 1.05, width, hintFont, hintFont));
      placed.push({x, y, w: width, h: hintFont, text, box: b});
      hintText = label(text, x, y, hintFont, k);
    }
    drawnHints.push({confidence: h.confidence, rect: rectOf(b), label: hintText});
  }
  const chosen = detections[selected];
  const handles = chosen === undefined ? [] : handlesOf(chosen.bbox_xyxy, handleSize, hitSize, true);
  return {boxes, hints: drawnHints, handles};
}

/**
 * The AI suggestion on the photo, in its own colour: "AI n · 80" under each box it suggests (n: the box's number once
 * applied), "✕ n" under each box of the frame it would delete. A box it did not judge is the frame's own, already drawn.
 */
export function drawAi(suggestion: Pick<AiSuggestion, 'detections' | 'deleted'>, imageHeight: number, k: number,
  imageWidth = Number.POSITIVE_INFINITY): DrawnAi[] {
  const fontSize = 12 / k;
  const placed: PlacedLabel[] = [];
  const drawn: DrawnAi[] = [];
  const add = (kind: DrawnAi['kind'], b: Box, text: string): void => {
    const width = labelWidth(text, fontSize);
    const x = labelX(b[0], width, imageWidth, k);
    // Below the box, where the label of a box of the frame never is.
    const y = Math.min(imageHeight, freeLabelY(placed, x, b[3] + fontSize * 1.05, width, fontSize, fontSize));
    placed.push({x, y, w: width, h: fontSize, text});
    drawn.push({kind, rect: rectOf(b), label: label(text, x, y, fontSize, k)});
  };
  suggestion.detections.forEach((d, i) => {
    if (d.ai_action !== 'unjudged') add(d.ai_action, d.bbox_xyxy, 'AI ' + (i + 1) + ' · ' + shortName(d));
  });
  for (const d of suggestion.deleted) add('delete', d.bbox_xyxy, '✕ ' + d.number);
  return drawn;
}

/**
 * The boxes of the car's networks not among the frame's, on the photo in fuchsia: "car n · 80 · 97%" under each (n: its row in
 * "Car networks"; names: each as a box of the review, car.ts carDetection); dashed when the car takes nothing for
 * the limit from it, dotted when it drops it.
 */
export function drawCar(signs: readonly CarSign[], names: readonly Detection[], surest: number | null,
  imageHeight: number, k: number, imageWidth = Number.POSITIVE_INFINITY): DrawnCar[] {
  const fontSize = 12 / k;
  const placed: PlacedLabel[] = [];
  return signs.map((sign, i) => {
    const b = sign.box;
    const name = names[i];
    const road = sign.applicability === 'own_road' ? ' · OURS' : sign.applicability === 'other_road' ? ' · another' : '';
    const text = 'car ' + (i + 1) + ' · ' + (name ? shortName(name) : sign.class) + road + ' · '
      + Math.round(sign.confidence * 100) + '%';
    const width = labelWidth(text, fontSize);
    const x = labelX(b[0], width, imageWidth, k);
    const y = Math.min(imageHeight, freeLabelY(placed, x, b[3] + fontSize * 1.05, width, fontSize, fontSize));
    placed.push({x, y, w: width, h: fontSize, text});
    return {use: sign.use, surest: i === surest, rect: rectOf(b), label: label(text, x, y, fontSize, k)};
  });
}

/** The part of the photo the zoom shows: the box with a margin, inside the image. */
export function zoomView(b: Box, imageWidth: number, imageHeight: number): Rect {
  const w = b[2] - b[0];
  const h = b[3] - b[1];
  const pad = Math.max(w, h) * 0.35 + 4;
  const x = Math.max(0, b[0] - pad);
  const y = Math.max(0, b[1] - pad);
  let vw = w + pad * 2;
  let vh = h + pad * 2;
  if (x + vw > imageWidth) vw = imageWidth - x;
  if (y + vh > imageHeight) vh = imageHeight - y;
  return {x, y, width: vw, height: vh};
}
