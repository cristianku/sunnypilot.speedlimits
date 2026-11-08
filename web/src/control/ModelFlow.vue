<script setup lang="ts">
// Models: the version in use of the network picked, as the car runs it (model-flow.ts), drawn by FlowChart.
import {computed} from 'vue';
import type {ModelVersion} from '../common/panel-types';
import FlowChart from './FlowChart.vue';
import {MODEL_COLUMNS, MODEL_LEGEND, modelBlocks} from './model-flow';

const props = defineProps<{version: ModelVersion}>();
const blocks = computed(() => modelBlocks(props.version));
</script>

<template>
  <FlowChart v-if="blocks" :blocks="blocks" :columns="MODEL_COLUMNS" :legend="MODEL_LEGEND" :id="'model-flow-' + version.role"/>
</template>
