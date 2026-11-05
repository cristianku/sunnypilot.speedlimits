<script setup lang="ts">
// How the next training of the scene network reads the signs (road-signs.ts), in its section and in its step of
// Train › Pipeline: the classifier.
import PickerSheet from '../components/PickerSheet.vue';
import type {RoadSignsChoice} from './road-signs';

defineProps<{signs: RoadSignsChoice; id: string}>();
</script>

<template>
  <div v-if="signs.classifierOptions.length > 1" class="cls-start">
    <PickerSheet v-model="signs.classifier" :options="signs.classifierOptions" label="Signs read by" :id="id + '-classifier'"/>
  </div>
</template>
