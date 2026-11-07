<script setup lang="ts">
// The choices of the brain's training (brain-choices.ts), in Train › Brain and in its step of Train › Pipeline.
import {ON_OFF, POOL_OPTIONS, type BrainChoice} from './brain-choices';
import Segmented from './Segmented.vue';

defineProps<{choices: BrainChoice; id: string}>();
</script>

<template>
  <div class="seg-row"><span class="seg-label">Roadworks</span>
    <Segmented v-model="choices.works" :options="ON_OFF" label="The brain also learns the roadworks"/></div>
  <div class="seg-row"><span class="seg-label">Picture</span>
    <Segmented v-model="choices.pool" :options="POOL_OPTIONS" label="The picture’s features"/></div>
  <div class="seg-row"><span class="seg-label">Driver’s inputs</span>
    <Segmented v-model="choices.driver" :options="ON_OFF" label="The gas, the brake, the car’s speed and openpilot on or off as inputs"/></div>
  <p class="latest-none" :id="id + '-note'">{{ choices.note }}</p>
</template>
