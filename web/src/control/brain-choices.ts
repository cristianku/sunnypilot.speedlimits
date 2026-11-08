// What the next training of the brain learns, kept once for Train › Brain and for its step of Train › Pipeline (the same
// choice in both): its roadworks head, its head of the kinds of road of Speed review, its head of the road type (the map's,
// changed in Speed review: then only the seconds with one), the picture's features as a grid or as a mean, and the driver's inputs (the gas, the
// brake and the car's speed), off by default since 8 Nov 2026: his own speed is the answer, so they are only a comparison.
import {computed, reactive, ref} from 'vue';
import type {ActionOptions} from '../common/panel-types';

export type OnOff = 'on' | 'off';
export type BrainPool = 'mean' | 'grid';
export const POOL_OPTIONS: {key: BrainPool; label: string}[] = [{key: 'mean', label: 'Mean'}, {key: 'grid', label: 'Grid'}];

export function useBrainChoices() {
  const works = ref<OnOff>('on');
  const road = ref<OnOff>('on');
  const types = ref<OnOff>('on');
  const driver = ref<OnOff>('off');
  const pool = ref<BrainPool>('grid');
  /** The options of its training (ActionOptions.works, road, types, driver, pool). */
  const option = computed((): ActionOptions => ({works: works.value, road: road.value, types: types.value,
    driver: driver.value, pool: pool.value}));
  /** What it learns in words, after the training's own: "", or ", without the roadworks". */
  const words = computed(() => (works.value === 'off' ? ', without the roadworks' : '')
    + (road.value === 'off' ? ', without the kinds of road' : '')
    + (types.value === 'off' ? ', without the road types' : '')
    + (pool.value === 'mean' ? ', with the picture as a mean' : '')
    + (driver.value === 'on' ? ', with the driver’s inputs' : ''));
  /** What Mean and Grid are, on the picture's choice. */
  const note = 'Grid keeps where things are in the picture, Mean only what is in it.';
  return reactive({works, road, types, driver, pool, option, words, note});
}
export type BrainChoice = ReturnType<typeof useBrainChoices>;
