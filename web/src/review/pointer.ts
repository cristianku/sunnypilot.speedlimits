// Mouse and touch on the photo and on the zoom: draw a new box (drawing mode, N), select a box, move the
// corners of the selected one or drag the whole box; Shift held makes the box drawn or resized a square. The handlers of both pictures (Stage.vue, BoxEditor.vue); the point under the
// pointer is the SVG's own math.
import {nextTick} from 'vue';
import {holdsPlate} from './catalog';
import {datasetOf} from './dom';
import {addManualBox, changed, markPlacedByHand} from './edits';
import {currentRow, selectedDetection} from './frames';
import {revealEditor} from './scroll';
import {report, state} from './store';
import type {Box, Corner, FrameRow, Point, Surface} from './types';

function isCorner(value: string | undefined): value is Corner {
  return value === 'nw' || value === 'ne' || value === 'sw' || value === 'se';
}

/** The corner handle under the pointer, if any. */
function cornerOf(target: EventTarget | null): Corner | null {
  const corner = datasetOf(target)?.['corner'];
  return isCorner(corner) ? corner : null;
}

/** The picture the handler is on (the listeners are on the <svg> elements). */
function svgOf(event: Event): SVGSVGElement | null {
  return event.currentTarget instanceof SVGSVGElement ? event.currentTarget : null;
}

/** The point of the image under the pointer, inside the image. An SVGPoint (single precision), as the boxes
 * drawn so far: a DOMPoint would give the same box with more decimals. */
function toImagePoint(svg: SVGSVGElement, event: PointerEvent, row: FrameRow): Point {
  const matrix = svg.getScreenCTM();
  let point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  if (matrix !== null) point = point.matrixTransform(matrix.inverse());
  return [Math.min(Math.max(point.x, 0), row.width), Math.min(Math.max(point.y, 0), row.height)];
}

/** Shift held: the point that makes a square with the corner ``a``, its side the longer one dragged, inside the image. */
function squared(a: Point, b: Point, row: FrameRow, shift: boolean): Point {
  if (!shift) return b;
  const sx = b[0] < a[0] ? -1 : 1;
  const sy = b[1] < a[1] ? -1 : 1;
  const room = Math.min(sx > 0 ? row.width - a[0] : a[0], sy > 0 ? row.height - a[1] : a[1]);
  const side = Math.min(Math.max(Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])), room);
  return [a[0] + sx * side, a[1] + sy * side];
}

/** The corner opposite to the one dragged. */
const OPPOSITE: Record<Corner, (b: Box) => Point> = {
  nw: (b) => [b[2], b[3]], ne: (b) => [b[0], b[3]], sw: (b) => [b[2], b[1]], se: (b) => [b[0], b[1]],
};

function drawDraft(a: Point, b: Point): void {
  state.draft = {
    x: Math.min(a[0], b[0]), y: Math.min(a[1], b[1]), width: Math.abs(a[0] - b[0]), height: Math.abs(a[1] - b[1]),
  };
}

export function onPointerDown(event: PointerEvent, surface: Surface): void {
  const row = currentRow();
  const svg = svgOf(event);
  // Read only (review/busy.ts): a box is chosen by a click, never drawn, moved or resized.
  if (!row || svg === null || report.readOnly) return;
  if (surface === 'scene' && state.drawing) {
    event.preventDefault();
    const start = toImagePoint(svg, event, row);
    state.drag = {kind: 'draw', svg: surface, start};
    svg.setPointerCapture(event.pointerId);
    drawDraft(start, start);
    return;
  }
  const corner = cornerOf(event.target);
  if (corner !== null) {
    event.preventDefault();
    state.drag = {kind: 'resize', svg: surface, corner, moved: false};
    svg.setPointerCapture(event.pointerId);
    return;
  }
  // Inside the selected box: the whole box follows the pointer.
  const d = selectedDetection();
  if (d && onSelectedBox(event, surface, toImagePoint(svg, event, row), d.bbox_xyxy)) {
    event.preventDefault();
    state.drag = {kind: 'move', svg: surface, start: toImagePoint(svg, event, row), box: [...d.bbox_xyxy], moved: false};
    svg.setPointerCapture(event.pointerId);
  }
}

/** On the photo the selected box itself, on the zoom anywhere inside it (its outline lets the pointer through). */
function onSelectedBox(event: Event, surface: Surface, p: Point, b: Box): boolean {
  if (surface === 'scene') return Number(datasetOf(event.target)?.['index']) === state.selected;
  return p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3];
}

export function onPointerMove(event: PointerEvent, surface: Surface): void {
  const drag = state.drag;
  const svg = svgOf(event);
  if (!drag || drag.svg !== surface || svg === null) return;
  const row = currentRow();
  if (!row) return;
  const p = toImagePoint(svg, event, row);
  if (drag.kind === 'draw') {
    drawDraft(drag.start, squared(drag.start, p, row, event.shiftKey));
    return;
  }
  const d = selectedDetection();
  if (!d) return;
  if (drag.kind === 'move') {
    // Inside the image: the box stops at its edges.
    const dx = Math.min(Math.max(p[0] - drag.start[0], -drag.box[0]), row.width - drag.box[2]);
    const dy = Math.min(Math.max(p[1] - drag.start[1], -drag.box[1]), row.height - drag.box[3]);
    d.bbox_xyxy = [drag.box[0] + dx, drag.box[1] + dy, drag.box[2] + dx, drag.box[3] + dy];
    if (Math.abs(dx) + Math.abs(dy) > 0) drag.moved = true;
    return;
  }
  // Shift: a square on the opposite corner (never a box of a sign and its plate, taller on purpose).
  const anchor = OPPOSITE[drag.corner](d.bbox_xyxy);
  const q = squared(anchor, p, row, event.shiftKey && !holdsPlate(d));
  const b: Box = [anchor[0], anchor[1], q[0], q[1]];
  const x1 = Math.min(b[0], b[2]);
  const x2 = Math.max(b[0], b[2]);
  const y1 = Math.min(b[1], b[3]);
  const y2 = Math.max(b[1], b[3]);
  if (x2 - x1 >= 2 && y2 - y1 >= 2) {
    d.bbox_xyxy = [x1, y1, x2, y2];
    drag.moved = true;
  }
}

export function finishDrag(event: PointerEvent, surface: Surface): void {
  const row = currentRow();
  const drag = state.drag;
  const svg = svgOf(event);
  if (!row || !drag || drag.svg !== surface || svg === null) return;
  state.drag = null;
  // The click that follows a drag must not select or deselect anything.
  state.ignoreClickUntil = Date.now() + 400;
  if (drag.kind === 'draw') {
    state.draft = null;
    if (event.type === 'pointercancel') return;
    const p = squared(drag.start, toImagePoint(svg, event, row), row, event.shiftKey);
    const box: Box = [
      Math.min(drag.start[0], p[0]), Math.min(drag.start[1], p[1]),
      Math.max(drag.start[0], p[0]), Math.max(drag.start[1], p[1]),
    ];
    if (box[2] - box[0] < 2 || box[3] - box[1] < 2) return;
    // W Y K pressed while drawing: the box is that road user at once; else "Unreadable" until a type is picked.
    if (state.drawKind) addManualBox(row, box, 'other_sign', null, state.drawKind);
    else addManualBox(row, box, 'unreadable', null);
    return;
  }
  if (drag.moved) {
    // A carried box moved (or resized) onto its sign: its position is checked.
    const d = selectedDetection();
    if (d) {
      delete d.check_position;
      markPlacedByHand(d, true);
    }
    changed(row);
  }
}

// A touch on a corner or on the selected box, or anywhere while drawing, must not scroll the page.
export function onTouchStart(event: TouchEvent, surface: Surface): void {
  if (surface === 'scene' && state.drawing) {
    event.preventDefault();
    return;
  }
  if (cornerOf(event.target) !== null) {
    event.preventDefault();
    return;
  }
  const onBox = surface === 'scene' ? Number(datasetOf(event.target)?.['index']) === state.selected
    : state.selected >= 0;
  if (onBox) event.preventDefault();
}

/** A click on a box selects it; on the photo elsewhere it clears the selection. */
export function onSceneClick(event: MouseEvent): void {
  if (Date.now() < state.ignoreClickUntil || state.drawing) return;
  const data = datasetOf(event.target);
  const index = data?.['index'];
  if (index !== undefined) {
    state.selected = Number(index);
    state.selectedByUser = true;
    void nextTick(revealEditor);
    return;
  }
  if (data?.['corner']) return;
  if (state.selected !== -1) state.selected = -1;
}
