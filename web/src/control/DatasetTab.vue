<script setup lang="ts">
// Data › Training dataset: what the networks learn from. First the Sign review frames not in the dataset yet, with Import
// sign reviews (Speed review is not imported: the brain reads it as it is)
// (helpers/3_train.sh --dataset-only, the panel action "dataset"), or its import running; then what the dataset holds
// (DatasetSection.vue). Importing is always a click: opening this page or another starts nothing.
import {computed} from 'vue';
import {plural, thousands} from '../common/format';
import type {ActionName, ActionState, Job, PanelStatus} from '../common/panel-types';
import DatasetSection from './DatasetSection.vue';
import JobProgress from './JobProgress.vue';
import {JOB_STATE, jobOf, startedWhen} from './jobs';
import SectionHead from './SectionHead.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState]; stop: [job: Job]; details: [job: Job]}>();

const job = computed(() => jobOf(props.status, ['dataset']));
/** Running or failed: in place of the button; one done is a line. */
const bigJob = computed(() => job.value && (job.value.state === 'running' || job.value.state === 'failed') ? job.value : null);
const state = computed(() => props.status?.actions.dataset ?? null);
const fresh = computed(() => {
  const routes = props.status?.routes ?? [];
  return {reviews: routes.reduce((total, row) => total + row.new_reviews, 0), drives: routes.filter((row) => row.new_reviews > 0).length};
});
const short = computed(() => props.status !== null && props.status.reviewed_routes < props.status.min_routes);
</script>

<template>
  <SectionHead tab="dataset"/>

  <section class="card import-card" id="import" aria-labelledby="import-title">
    <h2 id="import-title" class="sr-only">Import sign reviews</h2>
    <JobProgress v-if="bigJob" :job="bigJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
    <div v-else class="import-row">
      <span class="import-icon" :class="{fresh: fresh.reviews > 0}" aria-hidden="true">{{ fresh.reviews > 0 ? '↓' : '✓' }}</span>
      <div class="import-what">
        <strong>{{ status === null ? 'Loading…' : fresh.reviews ? plural(fresh.reviews, 'sign review') + ' to import'
          : 'Dataset is up to date' }}</strong>
        <span v-if="status && fresh.reviews">Sign review frames new or changed on {{ plural(fresh.drives, 'drive') }} · a few
          minutes</span>
        <span v-else-if="status">Every confirmed Sign review frame is in.</span>
        <span v-if="status">Speed review needs no import: the brain reads it as it is.</span>
      </div>
      <button v-if="fresh.reviews" type="button" class="btn primary" id="import-reviews" :disabled="!state?.allowed"
              @click="state && emit('run', 'dataset', state)">Import sign reviews</button>
    </div>
    <p v-if="!bigJob && fresh.reviews && state && !state.allowed" class="import-why">{{ state.reason }}</p>
    <p v-if="job && !bigJob" class="import-last">Last import {{ startedWhen(job) }} ·
      {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
      <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>
    <p v-if="short" class="import-why warn">Training needs {{ status?.min_routes }} reviewed drives (now
      {{ thousands(status?.reviewed_routes ?? 0) }}): some are kept apart to test on roads the network never saw.</p>
  </section>

  <DatasetSection :dataset="status?.dataset ?? null" :loading="status === null"/>
</template>
