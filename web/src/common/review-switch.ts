// The switch between Speed review and Sign review of the same drive at the same moment, always at the same place of
// the top bar (components/ReviewSwitch.vue, mounted by mount.ts before the comma): the page shown sets which one it is
// and the address of the other (DriveReviewDrive.vue's signHref, Inspector.vue's driveHref), null when there is none.
import {reactive} from 'vue';

export type ReviewPage = 'speed' | 'signs';

export const reviewSwitch = reactive<{current: ReviewPage | null; other: string | null}>({current: null, other: null});

export function setReviewSwitch(current: ReviewPage | null, other: string | null): void {
  reviewSwitch.current = current;
  reviewSwitch.other = other;
}
