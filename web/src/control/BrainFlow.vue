<script setup lang="ts">
// Train › Brain as a flow of blocks (brain-flow.ts, drawn by FlowChart): only what the next training uses, by the switches
// above it: what it gives the brain, the network, what it says, and the answers only the training has, each on the row of
// what it answers, joined by a line without arrows: it is compared with it, nothing flows from it.
import {computed} from 'vue';
import type {BrainChoice} from './brain-choices';
import {DIVIDER_ROW, FLOW_COLUMNS, flowBlocks} from './brain-flow';
import FlowChart from './FlowChart.vue';

const props = defineProps<{choices: BrainChoice}>();
const blocks = computed(() => flowBlocks({driver: props.choices.driver === 'on', works: props.choices.works === 'on',
  road: props.choices.road === 'on', types: props.choices.types === 'on'}));
const divider = computed(() => blocks.value.some((block) => block.column === 2 && block.row === DIVIDER_ROW + 1)
  ? {column: 2, row: DIVIDER_ROW, text: 'only in training'} : null);
const LEGEND = [{kind: 'input', label: 'Input'}, {kind: 'car', label: 'Output, also on the car'},
  {kind: 'train', label: 'Only in training'}] as const;
</script>

<template>
  <FlowChart :blocks="blocks" :columns="FLOW_COLUMNS" :legend="[...LEGEND]" :divider="divider" id="train-brain-io"/>
</template>
