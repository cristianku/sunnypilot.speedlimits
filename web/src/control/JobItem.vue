<script setup lang="ts">
// One job of the list of recent jobs, a row as in an iOS list: a round mark of its state, name, when it started and
// how it went; the row opens the details, Stop when it runs. A Review with AI opens its drive on the Signs side instead
// (the frames the AI reviewed in violet there, #79): its log says nothing worth reading.
import {computed} from 'vue';
import type {Job} from '../common/panel-types';
import {signsHref} from '../common/sign-link';
import Icon from '../components/Icon.vue';
import {failedStep, JOB_STATE, startedWhen} from './jobs';

const props = defineProps<{job: Job}>();
defineEmits<{details: [job: Job]; stop: [job: Job]}>();
/** The mark of each state: a check, a cross, a square (stopped), a ring (running). */
const MARK: Record<Job['state'], string> = {ok: '✓', failed: '✕', interrupted: '■', running: ''};
/** The drive of a Review with AI, which its row opens. */
const drive = computed(() => props.job.action === 'ai_review' && props.job.route ? props.job.route : null);
const sub = computed(() => {
  const job = props.job;
  const parts = [startedWhen(job)];
  if (drive.value) parts.push(drive.value + (job.saving ? ' · saved' : job.saving === false ? ' · trial' : ''));
  if (job.state === 'running') {
    if (job.progress.percent !== null) parts.push(Math.floor(job.progress.percent) + '%');
    if (job.progress.detail) parts.push(job.progress.detail);
  } else if (job.state === 'failed') {
    const step = failedStep(job);
    parts.push(step ? 'failed at ' + step : 'failed');
  } else if (job.state === 'interrupted') {
    parts.push('stopped');
  } else if (job.progress.detail) {
    parts.push(job.progress.detail);
  }
  return parts.join(' · ');
});
</script>

<template>
  <li class="job-item" :data-job="job.id">
    <component :is="drive ? 'a' : 'button'" class="job-open" v-bind="drive ? {href: signsHref(drive)} : {type: 'button'}"
               :title="drive ? 'Open the drive the AI reviewed, on its signs' : 'Details of ' + job.label"
               @click="drive || $emit('details', job)">
      <span class="job-mark" :class="JOB_STATE[job.state][1]" :aria-label="JOB_STATE[job.state][0]">{{ MARK[job.state] }}</span>
      <span class="grow"><span class="what">{{ job.label }}</span><span class="sub">{{ sub }}</span></span>
      <span class="job-go" aria-hidden="true">›</span>
    </component>
    <button v-if="job.stoppable" class="btn small danger" type="button" @click="$emit('stop', job)"><Icon name="stop"/>Stop</button>
  </li>
</template>
