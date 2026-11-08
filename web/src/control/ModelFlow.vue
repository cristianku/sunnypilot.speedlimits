<script setup lang="ts">
// Models: the version in use of the network picked, as the car runs it (model-flow.ts), drawn by FlowChart; a brain with the
// outputs only its training had draws those too, under a line.
import {computed} from 'vue';
import type {ModelVersion} from '../common/panel-types';
import FlowChart from './FlowChart.vue';
import {modelLayout} from './model-flow';

const props = defineProps<{version: ModelVersion}>();
const layout = computed(() => modelLayout(props.version));
</script>

<template>
  <FlowChart v-if="layout" :blocks="layout.blocks" :columns="layout.columns" :legend="layout.legend" :divider="layout.divider"
             :id="'model-flow-' + version.role"/>
</template>
