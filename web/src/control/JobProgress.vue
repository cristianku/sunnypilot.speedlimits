<script setup lang="ts">
// A job, live: name and state, the big percent (or "running" with a moving bar when the server does not
// know it), the steps, what it is doing, the time it has been running (ticking) and, when the percent is
// known, about how long is left; the GPU it holds, when the machine has two; Stop and Details (the log in the
// sheet); while it runs, the machine in small under the buttons. A finished job shows how it ended; a failed one
// what went wrong in a sentence when the server knows (the log, with its paths and exit code, is in Details) and
// Try again when it simply runs again.
import {computed, inject} from 'vue';
import type {Job} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import {useNow} from '../composables/now';
import JobSteps from './JobSteps.vue';
import {elapsedText, etaText, failedStep, GPU_CHIP_STYLE, gpuChipText, JOB_STATE, PIPELINE_ACTIONS, startedAt,
  startedWhen} from './jobs';
import {RESUME_JOB, RETRY_JOB} from './use-actions';

const props = defineProps<{job: Job; heading?: string | undefined}>();
defineEmits<{details: [job: Job]; stop: [job: Job]}>();
const now = useNow();

const running = computed(() => props.job.state === 'running');
const resumeJob = inject(RESUME_JOB, null);
// A training of the pipeline that stopped after writing its run: the missing steps can run again on that run
// (helpers/3_train.sh --resume; not the cutter's or the classifier's training, which are other scripts).
const resumable = computed(() => resumeJob !== null && Boolean(props.job.run)
  && (props.job.state === 'failed' || props.job.state === 'interrupted') && PIPELINE_ACTIONS.includes(props.job.action));
const retryJob = inject(RETRY_JOB, null);
// A failed download, preparation, install, import or measure: the same action again, when it can start now.
const retryable = computed(() => retryJob !== null && props.job.state === 'failed' && retryJob.can(props.job));
const state = computed(() => JOB_STATE[props.job.state]);
const percent = computed(() => {
  const value = props.job.state === 'ok' ? 100 : props.job.progress.percent;
  return value === null ? null : Math.max(0, Math.min(100, value));
});
/** Never rounded up: 99.6 is still 99 until the job says it is done. */
const percentText = computed(() => percent.value === null ? '' : percent.value > 0 && percent.value < 1 ? '<1' : String(Math.floor(percent.value)));
const elapsed = computed(() => {
  const start = startedAt(props.job);
  return start === null ? null : now.value - start.getTime();
});
/** About how long is left (jobs.ts etaText). */
const eta = computed(() => etaText(props.job, now.value));
/** When it started and, while it runs, for how long; the exit code of a failed job is in Details. */
const when = computed(() => {
  const at = 'Started ' + startedWhen(props.job);
  if (running.value) return elapsed.value === null ? at : at + ' · for ' + elapsedText(elapsed.value);
  return at;
});
/** What went wrong, in one sentence, when the server knows (an out-of-memory GPU, a full disk, ...). */
const failure = computed(() => props.job.state === 'failed' ? props.job.progress.failure ?? '' : '');
const ended = computed((): string => {
  if (props.job.state === 'ok') return 'Done';
  if (props.job.state === 'failed') return 'Failed';
  if (props.job.state === 'interrupted') return 'Interrupted';
  return '';
});
const stoppedAt = computed(() => props.job.state === 'failed' ? failedStep(props.job) : '');
const steps = computed(() => props.job.progress.steps.length > 1 ? props.job.progress.steps : []);
/** The card it runs on, as a chip after its times ("GPU 1"), with two cards in the machine. */
const gpu = computed(() => gpuChipText(props.job));
const barKind = computed(() => ({ok: 'ok', failed: 'bad', interrupted: 'warn', running: ''})[props.job.state]);
</script>

<template>
  <article class="jp" :class="['jp-' + job.state, {'has-steps': steps.length > 0}]" :data-job="job.id" :aria-label="job.label">
    <header class="jp-head">
      <span class="jp-icon" :class="state[1]" aria-hidden="true"><span v-if="running" class="spinner"></span><svg
        v-else-if="job.state === 'ok'" viewBox="0 0 16 16"><path d="M3.5 8.4l3 3 6-6.4" fill="none" stroke="currentColor" stroke-width="2.2"
        stroke-linecap="round" stroke-linejoin="round"/></svg><svg v-else viewBox="0 0 16 16"><path d="M5 5l6 6M11 5l-6 6"
        fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg></span>
      <div class="jp-title">
        <span v-if="heading" class="jp-kicker">{{ heading }}</span>
        <h3>{{ job.label }}</h3>
        <span class="jp-when">{{ when }}<span v-if="gpu" :style="GPU_CHIP_STYLE">{{ gpu }}</span></span>
      </div>
      <span class="pill" :class="state[1]">{{ state[0] }}</span>
    </header>
    <ul v-if="job.settings?.length" class="jp-settings" aria-label="Launched with">
      <li v-for="setting in job.settings" :key="setting">{{ setting }}</li>
    </ul>

    <div class="jp-body">
      <div class="jp-main">
        <div class="jp-figure">
          <span v-if="running && percent === null" class="jp-big jp-big-word">running</span>
          <span v-else-if="running" class="jp-big">{{ percentText }}<small>%</small></span>
          <span v-else class="jp-big jp-big-word" :class="state[1]">{{ ended }}<template v-if="job.state === 'ok'"> ✓</template></span>
          <span v-if="eta" class="jp-eta">{{ eta }}</span>
        </div>
        <div class="jp-bar" :class="[barKind, {indeterminate: running && percent === null}]" role="progressbar"
             :aria-label="job.label" aria-valuemin="0" aria-valuemax="100"
             :aria-valuenow="percent === null ? undefined : Math.floor(percent)"
             :aria-valuetext="percent === null ? 'running, progress unknown' : undefined"><span
          :style="percent === null ? undefined : {width: percent + '%'}"></span></div>
        <p v-if="job.progress.detail" class="jp-detail">{{ job.progress.detail }}</p>
        <p v-if="stoppedAt" class="jp-stopped">It stopped at the step <strong>{{ stoppedAt }}</strong>.</p>
        <p v-if="failure" class="jp-failure">{{ failure }}</p>
        <p v-else-if="job.progress.last_line" class="jp-line" :class="{bad: job.state === 'failed'}"
           :title="job.progress.last_line">{{ job.progress.last_line }}</p>
        <slot></slot>
        <div class="jp-actions">
          <button class="btn small" type="button" @click="$emit('details', job)"><Icon name="details"/>Details</button>
          <button v-if="running && job.stoppable" class="btn small danger" type="button" @click="$emit('stop', job)"><Icon name="stop"/>Stop</button>
          <button v-if="resumable" class="btn small primary" type="button" @click="resumeJob?.(job)"><Icon name="resume"/>Resume</button>
          <button v-if="retryable" class="btn small primary" type="button" @click="retryJob?.run(job)"><Icon name="resume"/>Try again</button>
        </div>
      </div>
      <JobSteps v-if="steps.length" :steps="steps" :current="job.state === 'running' ? job.progress.step_progress : null"/>
    </div>
  </article>
</template>
