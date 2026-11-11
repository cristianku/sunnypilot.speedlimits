<script setup lang="ts">
// The choices of the brain's training (brain-choices.ts) in one short row, in Train › Brain above its flow (which follows
// them) and in its step of Train › Pipeline: the three heads only the training has as switches that light up when on, the
// picture's features as Mean or Grid, its seeds as Three or One.
import {POOL_OPTIONS, SEEDS_OPTIONS, type BrainChoice} from './brain-choices';
import Segmented from './Segmented.vue';

const props = defineProps<{choices: BrainChoice; id: string}>();
type Switch = 'works' | 'road' | 'types';
const SWITCHES: {key: Switch; label: string; title: string}[] = [
  {key: 'works', label: 'Roadworks', title: 'The brain also learns the roadworks on the road, a head only the training has'},
  {key: 'road', label: 'Road conditions', title: 'The brain also learns the road conditions you mark in Speed review: narrow, country, gravel, potholes…'},
  {key: 'types', label: 'Road type (map)', title: 'The brain also learns the road type of the map (motorway, rural, town…), corrected in Speed review; only the seconds that have one count'},
];
function flip(key: Switch): void {
  props.choices[key] = props.choices[key] === 'on' ? 'off' : 'on';
}
</script>

<template>
  <div class="bc" :id="id + '-choices'" role="group" aria-label="What the next training of the brain learns">
    <button v-for="item in SWITCHES" :key="item.key" type="button" class="bc-switch" :aria-pressed="choices[item.key] === 'on'"
            :title="item.title" @click="flip(item.key)">{{ item.label }}</button>
    <span class="bc-pick" :title="choices.note"><span>Picture</span>
      <Segmented v-model="choices.pool" :options="POOL_OPTIONS" label="The picture’s features: Grid keeps where things are, Mean only what is in it"/></span>
    <span class="bc-pick" :title="choices.seedsNote"><span>Seeds</span>
      <Segmented v-model="choices.seeds" :options="SEEDS_OPTIONS" label="The seeds: seed 0 alone, or three to compare in Models"/></span>
  </div>
</template>
