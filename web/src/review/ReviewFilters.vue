<script setup lang="ts">
// Which frames to show (To review, Suspicious, With proposals, All, AI when an AI confirmed some, and AI trial when
// the route's last "Try without saving" left frames to review), next to the frame navigation under the photo; a
// doubt filter in use is shown here too.
import {computed} from 'vue';
import {DOUBT_LABELS, doubtCounts, routeCounts} from './current';
import {isQualityFilter, matchesFilter} from './frames';
import {pickFilter} from './navigation';
import {report, state} from './store';
import type {Filter} from './types';

interface FilterButton {
  filter: Filter;
  label: string;
  title?: string;
}
const FILTERS: readonly FilterButton[] = [
  {filter: 'todo', label: 'To review', title: 'Not confirmed yet'},
  {filter: 'suspicious', label: 'Suspicious', title: 'The model and the second reading disagree'},
  {filter: 'proposals', label: 'With proposals', title: 'With proposals of the model'},
  {filter: 'all', label: 'All'},
];
const counts = routeCounts;
const trialCount = computed(() => report.rows.filter((row) => matchesFilter(row, 'trial')).length);
// The doubt filter in use, if any ("26 without a number").
const doubt = computed(() => (isQualityFilter(state.filter) ? state.filter : null));
const doubtText = computed(() => (doubt.value === null ? ''
  : doubtCounts.value[doubt.value] + ' ' + DOUBT_LABELS[doubt.value]));
</script>

<template>
  <div class="filters" role="group" aria-label="Which frames to show">
    <button v-for="item in FILTERS" :key="item.filter" type="button" :data-filter="item.filter"
            :aria-pressed="state.filter === item.filter" :title="item.title"
            @click="pickFilter(item.filter)">{{ item.label }}</button>
    <!-- Only when an AI confirmed some frames (or while it is the filter in use). -->
    <button type="button" data-filter="ai" :aria-pressed="state.filter === 'ai'"
            title="Confirmed by an AI reviewer (REST API or MCP)"
            :hidden="counts.byAi === 0 && state.filter !== 'ai'" @click="pickFilter('ai')">AI · {{ counts.byAi }}</button>
    <!-- Only when the last "Try without saving" of the route left frames to review. -->
    <button type="button" data-filter="trial" :aria-pressed="state.filter === 'trial'"
            title="Frames the AI tried without saving, still to review"
            :hidden="trialCount === 0 && state.filter !== 'trial'" @click="pickFilter('trial')">AI trial · {{ trialCount }}</button>
    <!-- A doubt filter in use (its counter is under the drive's bar): shown pressed here too, × removes it. -->
    <button v-if="doubt !== null" type="button" class="doubt-on" :data-filter="doubt" aria-pressed="true"
            :aria-label="doubtText + ': remove this filter'" title="Remove this filter"
            @click="pickFilter(doubt)">{{ doubtText }}<span class="x" aria-hidden="true">×</span></button>
  </div>
</template>
