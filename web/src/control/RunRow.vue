<script setup lang="ts">
// One running job in one row (the Overview): its name, how far it is, what it does now and about how long is left,
// with a thin bar, and Stop beside it; a tap opens its sheet (LogSheet.vue: the steps, the log). A job that failed is
// the same row in red with what went wrong. Work started from a terminal has no sheet: the row only says what it is.
// With two GPUs in the machine, the card each holds opens its line ("GPU 1").
import {computed} from 'vue';
import Icon from '../components/Icon.vue';
import type {Activity, Job} from '../common/panel-types';
import {useNow} from '../composables/now';
import {etaText, failedStep, GPU_CHIP_STYLE, gpuChipText, stepText} from './jobs';

const props = defineProps<{job?: Job | null; activity?: Activity | null}>();
defineEmits<{open: [job: Job]; stop: [job: Job]}>();
const now = useNow();

const failed = computed(() => props.job?.state === 'failed');
const percent = computed(() => {
  const value = props.job ? props.job.progress.percent : props.activity?.progress?.percent ?? null;
  return value === null ? null : Math.max(0, Math.min(100, value));
});
const label = computed(() => props.job?.label ?? (props.activity ? props.activity.label.charAt(0).toUpperCase() + props.activity.label.slice(1) : ''));
/** "26% · epoch 5 of 20 · about 19 min left"; a failed job: where and why it stopped. */
const line = computed(() => {
  const job = props.job;
  if (job && failed.value) return ['Failed', failedStep(job), job.progress.failure ?? ''].filter(Boolean).join(' · ');
  const parts: string[] = [];
  if (percent.value !== null) parts.push(Math.floor(percent.value) + '%');
  if (job) {
    const step = stepText(job);
    if (step) parts.push(step);
    const eta = etaText(job, now.value);
    if (eta) parts.push(eta);
  } else {
    if (props.activity?.progress?.detail) parts.push(props.activity.progress.detail);
    parts.push('started from a terminal');
  }
  return parts.join(' · ') || 'running';
});
/** The card it holds, as a chip at the start of its line (none once it ended). */
const gpu = computed(() => gpuChipText(props.job ?? props.activity ?? {}));
const CHIP_STYLE = {...GPU_CHIP_STYLE, marginLeft: '0', marginRight: '6px'};
</script>

<template>
  <div class="run-item">
  <component :is="job ? 'button' : 'div'" :type="job ? 'button' : undefined" class="run-row" :class="{failed}"
             :data-job="job?.id" @click="job && $emit('open', job)">
    <span class="run-icon" aria-hidden="true"><span v-if="!failed" class="spinner"></span><template v-else>!</template></span>
    <span class="run-main">
      <strong class="run-name">{{ label }}</strong>
      <span class="run-line"><span v-if="gpu" :style="CHIP_STYLE">{{ gpu }}</span>{{ line }}</span>
      <span v-if="!failed" class="run-bar" :class="{indeterminate: percent === null}" aria-hidden="true"><span
        :style="percent === null ? undefined : {width: percent + '%'}"></span></span>
    </span>
    <span v-if="job" class="run-chev" aria-hidden="true">›</span>
  </component>
  <button v-if="job && job.state === 'running' && job.stoppable" class="btn small danger run-stop" type="button"
          :aria-label="'Stop ' + label" @click="$emit('stop', job)"><Icon name="stop"/>Stop</button>
  </div>
</template>
