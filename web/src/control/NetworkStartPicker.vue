<script setup lang="ts">
// Where the next training of the cutter or of applicability starts (network-start.ts), in its
// section and in its step of Train › Pipeline; nothing while it has no finished version to go on from.
import PickerSheet from '../components/PickerSheet.vue';
import type {NetworkStartChoice} from './network-start';

defineProps<{start: NetworkStartChoice; id: string}>();
</script>

<template>
  <template v-if="start.options.length > 1">
    <div class="cls-start">
      <PickerSheet v-model="start.key" :options="start.options" label="Start from" :id="id"/>
    </div>
    <p v-if="start.note" class="latest-none" :class="{'warn-text': start.note.warn}" :id="id + '-note'">{{ start.note.text }}</p>
  </template>
</template>
