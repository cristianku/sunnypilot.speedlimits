<script setup lang="ts">
// One route of the table: id (link to its review), segments, frames, confirmed frames, objects in the
// dataset and state.
import {computed} from 'vue';
import {thousands} from '../common/format';
import type {RouteRow, RouteState} from '../common/panel-types';
import Bar from '../components/Bar.vue';
import {signsHref} from '../common/sign-link';

const props = defineProps<{row: RouteRow}>();

// State of a route: label, pill classes, and a prefix dropped on phones ("ready to import" -> "to import").
const STATE: Record<RouteState, [label: string, kind: string, prefix?: string]> = {
  collegare: ['to link', 'todo'], estrarre: ['to extract', 'todo'], preannotare: ['to pre-annotate', 'todo'],
  revisionare: ['to review', 'muted hollow'], importare: ['to import', 'ok hollow', 'ready '],
  pronta: ['in the dataset', 'ok'],
};

const state = computed(() => STATE[props.row.state]);
// The leading zeros of "0000011e--…" go in their own span, hidden on phones so the part that tells
// routes apart stays visible.
const pad = computed(() => /^0+(?=[0-9a-f]+--)/.exec(props.row.route)?.[0] ?? '');
const late = computed(() => props.row.run_id ? props.row.segments - props.row.extracted : 0);
const reviewed = computed(() => props.row.report_frames ? Math.min(100, props.row.confirmed * 100 / props.row.report_frames) : 0);
</script>

<template>
  <tr>
    <td class="route"><a :href="signsHref(row.route)" :title="row.route"><span v-if="pad"
      class="rid-pad">{{ pad }}</span>{{ row.route.slice(pad.length) }}</a></td>
    <td class="num">{{ row.extracted }} / {{ row.segments }}</td>
    <td class="num">{{ thousands(row.frames) }}</td>
    <td v-if="row.run_id"><div class="reviewed"><Bar :percent="reviewed" kind="ok"/><span
      class="num">{{ thousands(row.confirmed) }} / {{ thousands(row.report_frames) }}</span></div></td>
    <td v-else class="none">–</td>
    <td class="num">{{ thousands(row.objects) }}</td>
    <td><span class="pill" :class="state[1]"><span v-if="state[2]" class="pill-long">{{ state[2] }}</span>{{ state[0] }}</span><span
      v-if="row.state === 'importare'" class="note">{{ thousands(row.new_reviews) }} new frames</span><span
      v-if="late > 0" class="note">{{ late }} {{ late === 1 ? 'segment' : 'segments' }} arrived later, not used</span></td>
  </tr>
</template>
