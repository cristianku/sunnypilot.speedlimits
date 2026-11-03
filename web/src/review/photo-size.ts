// The height the reviewer gave the photo on a computer, with the grip under it (PhotoGrip.vue): the timeline under
// the photo gets the room left. Kept in this browser; null is the page's own height.
import {ref} from 'vue';

const KEY = 'nnslr-review-photo-height';
/** The lowest the photo goes: its signs still readable. */
export const PHOTO_MIN = 140;

function stored(): number | null {
  try {
    const value = Number(localStorage.getItem(KEY) ?? '');
    return Number.isFinite(value) && value >= PHOTO_MIN ? value : null;
  } catch {
    return null;
  }
}

/** The photo's height in CSS pixels while the grip is dragged and after, or null for the page's default. */
export const photoHeight = ref<number | null>(stored());

/** Sets the height (null: back to the default) and keeps it for the next visit. */
export function setPhotoHeight(height: number | null): void {
  photoHeight.value = height === null ? null : Math.round(height);
  try {
    if (photoHeight.value === null) localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, String(photoHeight.value));
  } catch {
    // Kept for this page only.
  }
}
