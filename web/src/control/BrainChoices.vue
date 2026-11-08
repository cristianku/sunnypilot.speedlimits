<script setup lang="ts">
// The choices of the brain's training (brain-choices.ts), in Train › Brain and in its step of Train › Pipeline.
import {ON_OFF, POOL_OPTIONS, type BrainChoice} from './brain-choices';
import Segmented from './Segmented.vue';

defineProps<{choices: BrainChoice; id: string}>();
</script>

<template>
  <div class="seg-row"><span class="seg-label">Roadworks</span>
    <Segmented v-model="choices.works" :options="ON_OFF" label="The brain also learns the roadworks"/></div>
  <div class="seg-row"><span class="seg-label">Kinds of road</span>
    <Segmented v-model="choices.road" :options="ON_OFF" label="The brain also learns the kinds of road of Speed review"/></div>
  <div class="seg-row"><span class="seg-label">Road types</span>
    <Segmented v-model="choices.types" :options="ON_OFF" label="The brain also learns the road type, only from the seconds that have one"/></div>
  <div class="seg-row"><span class="seg-label">Picture</span>
    <Segmented v-model="choices.pool" :options="POOL_OPTIONS" label="The picture’s features"/></div>
  <div class="seg-row"><span class="seg-label">Driver’s inputs</span>
    <Segmented v-model="choices.driver" :options="ON_OFF" label="The gas, the brake and the car’s speed as inputs"/></div>
  <p class="latest-none" :id="id + '-note'">{{ choices.note }}</p>
</template>
