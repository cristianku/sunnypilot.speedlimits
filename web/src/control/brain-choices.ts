// What the next training of the brain learns, kept once for Train › Brain and for its step of Train › Pipeline (the same
// choice in both): its roadworks head, the picture's features as a mean or as a grid, and the driver's inputs (the gas, the
// brake and the car's speed), only to measure what they are worth: the car does not give them, such a brain cannot go there.
import {computed, reactive, ref} from 'vue';
import type {ActionOptions} from '../common/panel-types';

export type OnOff = 'on' | 'off';
export type BrainPool = 'mean' | 'grid';
export const ON_OFF: {key: OnOff; label: string}[] = [{key: 'on', label: 'On'}, {key: 'off', label: 'Off'}];
export const POOL_OPTIONS: {key: BrainPool; label: string}[] = [{key: 'mean', label: 'Mean'}, {key: 'grid', label: 'Grid'}];

export function useBrainChoices() {
  const works = ref<OnOff>('on');
  const driver = ref<OnOff>('off');
  const pool = ref<BrainPool>('mean');
  /** The options of its training (ActionOptions.works, driver, pool). */
  const option = computed((): ActionOptions => ({works: works.value, driver: driver.value, pool: pool.value}));
  /** What it learns in words, after the training's own: "", or ", without the roadworks". */
  const words = computed(() => (works.value === 'off' ? ', without the roadworks' : '')
    + (pool.value === 'grid' ? ', with the picture as a grid' : '')
    + (driver.value === 'on' ? ', with the driver’s inputs (not for the car)' : ''));
  /** A line under the choices. */
  const note = computed(() => (driver.value === 'on' ? 'With the driver’s inputs it is only a measure: the car does not give '
    + 'them, so this brain cannot be chosen in Models.' : 'Grid keeps where things are in the picture, Mean only what is in it.'));
  return reactive({works, driver, pool, option, words, note});
}
export type BrainChoice = ReturnType<typeof useBrainChoices>;
