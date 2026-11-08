// The height the reviewer gave the drive's picture on a computer, with the grip under it (StageGrip.vue), as in Sign
// review: the lanes under it get the room left. Kept in this browser; null is the page's own height.
import {ref} from 'vue';

const KEY = 'nnslr-drive-stage-height';
/** The lowest the picture goes: its signs still readable. */
export const STAGE_MIN = 180;

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
