<script setup lang="ts">
// How the next training of the scene network reads the signs (road-signs.ts), in its section and in its step of
// Train › Pipeline: the classifier, and the board input.
import PickerSheet from '../components/PickerSheet.vue';
import {BOARDS_OPTIONS, type RoadSignsChoice} from './road-signs';
import Segmented from './Segmented.vue';

defineProps<{signs: RoadSignsChoice; id: string}>();
</script>

<template>
  <div v-if="signs.classifierOptions.length > 1" class="cls-start">
    <PickerSheet v-model="signs.classifier" :options="signs.classifierOptions" label="Signs read by" :id="id + '-classifier'"/>
  </div>
  <div class="seg-row">
    <Segmented v-model="signs.boards" :options="BOARDS_OPTIONS" label="Green boards"/>
  </div>
  <p class="latest-none" :id="id + '-note'">{{ signs.note }}</p>
</template>
