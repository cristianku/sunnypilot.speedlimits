// The height the reviewer gave the drive's picture on a computer, with the grip under it (StageGrip.vue): the page's own
// height is the lowest, and dragged down the picture grows past its column. The drive then leaves the page's 1180 px,
// centred, and once the picture and the side no longer fit side by side in the window the side goes under the lanes.
// Kept in this browser; null is the page's own height.
import {ref} from 'vue';

const KEY = 'nnslr-drive-stage-height';
/** A height kept below this is no height: the picture is never smaller than the page's own. */
export const STAGE_MIN = 180;
/** The picture's width over its height: the comma's road camera. */
export const ASPECT = 1928 / 1208;
/** Wider than this the drive has the grip and its two columns (control.html's min-width:861px). */
const DESKTOP = 861;
/** Around the picture, in control.html and shared.css: the page's 1180 px and its padding (16 a side), the card's padding
 * (.rt-view, 18 a side), the side column with its gap (.dr-main, 330 + 20), what the page keeps under the picture at its
 * own height (100dvh - 420px) and the narrowest it makes it (.dr-stage's min-width). */
const PAGE = 1180;
const PAGE_PAD = 32;
const CARD_PAD = 36;
const SIDE = 350;
const BELOW = 420;
const NARROWEST = 420;
/** Kept under the picture at its highest: the window shows it whole under the top bar. */
const ROOM = 24;

function stored(): number | null {
  try {
    const value = Number(localStorage.getItem(KEY) ?? '');
    return Number.isFinite(value) && value >= STAGE_MIN ? value : null;
  } catch {
    return null;
  }
}

/** The picture's height in CSS pixels while the grip is dragged and after, or null for the page's default. */
export const stageHeight = ref<number | null>(stored());

/** Sets the height (null: back to the default) and keeps it for the next visit. */
export function setStageHeight(height: number | null): void {
  stageHeight.value = height === null ? null : Math.round(height);
  try {
    if (stageHeight.value === null) localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, String(stageHeight.value));
  } catch {
    // Kept for this page only.
  }
}

/** The window's room for the page (no scrollbar), followed as it changes. */
const measure = () => ({width: document.documentElement.clientWidth, height: window.innerHeight});
export const viewport = ref(measure());
window.addEventListener('resize', () => { viewport.value = measure(); });

/** The picture's height at the page's own size: the lowest. */
export function lowestHeight(): number {
  const {width, height} = viewport.value;
  const column = Math.min(PAGE, width) - PAGE_PAD - CARD_PAD - SIDE;
  return Math.min(column, Math.max(NARROWEST, (height - BELOW) * ASPECT)) / ASPECT;
}

/** The highest the picture goes: as wide as the window, and whole in it under the top bar. */
export function highestHeight(): number {
  const {width, height} = viewport.value;
  const top = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--topbar-h')) || 57;
  return Math.max(lowestHeight(), Math.min((width - PAGE_PAD - CARD_PAD) / ASPECT, height - top - ROOM));
}

/** Whether the picture is bigger than the page's own on a computer, and whether the side then goes under the lanes. */
export function grownLayout(height: number | null): {grown: boolean; under: boolean} {
  const {width} = viewport.value;
  const grown = height !== null && width >= DESKTOP && height > lowestHeight() + 0.5;
  return {grown, under: grown && height * ASPECT > width - PAGE_PAD - CARD_PAD - SIDE};
}
