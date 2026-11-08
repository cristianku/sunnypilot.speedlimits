<script setup lang="ts">
// The choices of the brain's training (brain-choices.ts) in one short row, in Train › Brain above its flow (which follows
// them) and in its step of Train › Pipeline: the three heads only the training has and the driver's inputs as switches that
// light up when on, the picture's features as Mean or Grid.
import {POOL_OPTIONS, type BrainChoice} from './brain-choices';
import Segmented from './Segmented.vue';

const props = defineProps<{choices: BrainChoice; id: string}>();
type Switch = 'works' | 'road' | 'types' | 'driver';
const SWITCHES: {key: Switch; label: string; title: string}[] = [
  {key: 'works', label: 'Roadworks', title: 'The brain also learns the roadworks on the road, a head only the training has'},
  {key: 'road', label: 'Kinds of road', title: 'The brain also learns the kinds of road marked in Speed review'},
  {key: 'types', label: 'Road types', title: 'The brain also learns the road type, only from the seconds that have one'},
  {key: 'driver', label: 'Driver’s inputs', title: 'The gas, the brake and the car’s speed as inputs: off by default'},
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
  </div>
</template>
