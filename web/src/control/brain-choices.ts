// What the next training of the brain learns, kept once for Train › Brain and for its step of Train › Pipeline (the same
// choice in both): its roadworks head, its head of the kinds of road of Speed review, its head of the road type (the map's,
// changed in Speed review: then only the seconds with one), the picture's features as a grid or as a mean, seed 0 alone
// (the default since 11 Nov 2026) or its three seeds (to compare in Models), three times the time. The driver's
// inputs (the gas, the brake and the car's speed) are gone since 10 Nov 2026: the brain trained with them followed his speed.
import {computed, reactive, ref} from 'vue';
import type {ActionOptions} from '../common/panel-types';

export type OnOff = 'on' | 'off';
export type BrainPool = 'mean' | 'grid';
export const POOL_OPTIONS: {key: BrainPool; label: string}[] = [{key: 'mean', label: 'Mean'}, {key: 'grid', label: 'Grid'}];
export type BrainSeeds = 'all' | 'one';
export const SEEDS_OPTIONS: {key: BrainSeeds; label: string}[] = [{key: 'one', label: 'One'}, {key: 'all', label: 'Three'}];

export function useBrainChoices() {
  const works = ref<OnOff>('on');
  const road = ref<OnOff>('on');
  const types = ref<OnOff>('on');
  const pool = ref<BrainPool>('grid');
  const seeds = ref<BrainSeeds>('one');
  /** The options of its training (ActionOptions.works, road, types, pool; seeds only for seed 0 alone, the three when
   * not given). */
  const option = computed((): ActionOptions => ({works: works.value, road: road.value, types: types.value,
    pool: pool.value, ...(seeds.value === 'one' ? {seeds: 'one' as const} : {})}));
  /** What it learns in words, after the training's own: "", or ", without the roadworks". */
  const words = computed(() => (works.value === 'off' ? ', without the roadworks' : '')
    + (road.value === 'off' ? ', without the kinds of road' : '')
    + (types.value === 'off' ? ', without the road types' : '')
    + (pool.value === 'mean' ? ', with the picture as a mean' : ''));
  /** What Mean and Grid are, on the picture's choice. */
  const note = 'Grid keeps where things are in the picture, Mean only what is in it.';
  /** What the seeds are, on their choice. */
  const seedsNote = 'Seed 0 alone, or three networks that differ only by chance, to compare in Models, in three times the time.';
  return reactive({works, road, types, pool, seeds, option, words, note, seedsNote});
}
export type BrainChoice = ReturnType<typeof useBrainChoices>;
