<script setup lang="ts">
// Work started from a terminal (or by another program): not a job of the site, so no log and no Stop.
// A preparation shows how far it is, from what it already wrote; anything else a moving bar.
import {computed} from 'vue';
import type {Activity, Preparation} from '../common/panel-types';
import {GPU_CHIP_STYLE, gpuChipText} from './jobs';
import JobSteps from './JobSteps.vue';
import PrepRoutes from './PrepRoutes.vue';

const props = defineProps<{activity: Activity; preparation: Preparation | null}>();
const prep = computed(() => props.activity.group === 'prepare' ? props.preparation : null);
// A training started from a terminal has the progress of its log (data/logs/train-<run>.log).
const progress = computed(() => props.activity.progress ?? null);
const percent = computed(() => prep.value?.percent ?? progress.value?.percent ?? null);
const where = computed(() => props.activity.group === 'other' ? 'another program on the server' : 'started from a terminal');
/** The card it holds, with two in the machine ("GPU 1"), as a chip after where it comes from. */
const gpu = computed(() => gpuChipText(props.activity));
</script>

<template>
  <article class="jp jp-running jp-terminal" :aria-label="activity.label">
    <header class="jp-head">
      <span class="jp-icon run" aria-hidden="true"><span class="spinner"></span></span>
      <div class="jp-title">
        <h3>{{ activity.label.charAt(0).toUpperCase() + activity.label.slice(1) }}</h3>
        <span class="jp-when">{{ where }} · pid {{ activity.pid }}<span v-if="gpu" :style="GPU_CHIP_STYLE">{{ gpu }}</span></span>
      </div>
      <span class="pill run">running</span>
    </header>
    <div class="jp-body">
      <div class="jp-main">
        <div class="jp-figure">
          <span v-if="percent === null" class="jp-big jp-big-word">running</span>
          <span v-else class="jp-big">{{ Math.floor(percent) }}<small>%</small></span>
        </div>
        <div class="jp-bar" :class="{indeterminate: percent === null}" role="progressbar" :aria-label="activity.label"
             aria-valuemin="0" aria-valuemax="100" :aria-valuenow="percent ?? undefined"
             :aria-valuetext="percent === null ? 'running, progress unknown' : undefined"><span
          :style="percent === null ? undefined : {width: percent + '%'}"></span></div>
        <p v-if="prep && prep.routes_total" class="jp-detail">{{ prep.routes_done }} of {{ prep.routes_total }} drives ready</p>
        <p v-if="progress?.detail" class="jp-detail">{{ progress.detail }}</p>
        <p v-if="progress?.last_line" class="jp-line">{{ progress.last_line }}</p>
        <PrepRoutes v-if="prep" :routes="prep.working"/>
        <p class="jp-line">It stops from the terminal it started in: here it is only shown.</p>
      </div>
      <JobSteps v-if="progress?.steps.length" :steps="progress.steps"/>
    </div>
  </article>
</template>
