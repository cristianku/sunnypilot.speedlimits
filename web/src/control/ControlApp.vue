<script setup lang="ts">
// Overview, Drives, Roads, Training dataset, Training, Results, Versions, Publish: one app, the tab shown follows the address (tabs.ts). One status for all
// tabs (/api/panel/status every 4 s, which also feeds the machine's history), the actions, and the sheet with the
// details of a job. Evaluate reads the training runs itself (/api/panel/runs), Models the networks by role
// (/api/panel/models).
import {computed, provide, ref, watch} from 'vue';
import type {Job} from '../common/panel-types';
import {markBusy} from '../composables/busy';
import {usePanelStatus} from '../composables/panel-status';
import {busyTabs} from './jobs';
import LogSheet from './LogSheet.vue';
import {recordMachine} from './machine-history';
import DatasetTab from './DatasetTab.vue';
import EvaluateTab from './EvaluateTab.vue';
import ModelsTab from './ModelsTab.vue';
import OverviewTab from './OverviewTab.vue';
import PrepareTab from './PrepareTab.vue';
import PublishTab from './PublishTab.vue';
import RoadsTab from './RoadsTab.vue';
import {currentTab, useTabRouting} from './tabs';
import TrainTab from './TrainTab.vue';
import {RESUME_JOB, RETRY_ACTIONS, RETRY_JOB, usePanelActions} from './use-actions';

useTabRouting();
const {status, refresh} = usePanelStatus((value) => recordMachine(value.system));
const {run, stop, resume, retry, startPipeline, stopPipeline} = usePanelActions(refresh);
// "Resume" on the card of a stopped training, wherever the card is shown.
provide(RESUME_JOB, resume);
// "Try again" on the card of a failed job that needs no choice but its proposer, when its action can start now.
provide(RETRY_JOB, {
  can: (job) => RETRY_ACTIONS.includes(job.action) && (job.action !== 'measure_proposer' || Boolean(job.proposer))
    && status.value?.actions[job.action]?.allowed === true,
  run: (job) => {
    const state = status.value?.actions[job.action];
    if (state) void retry(job, state);
  },
});

/** The job in the sheet, by id: the sheet shows its latest copy from the status. */
const shownId = ref<string | null>(null);
const shownJob = computed(() => status.value?.jobs.find((job) => job.id === shownId.value) ?? null);
function details(job: Job): void {
  shownId.value = job.id;
}

/** The tabs with work running, by address: a job lights the tab that shows it (jobs.ts ACTION_TAB: a measure is
 * Evaluate's even if it uses the GPU as a training does); work started from a terminal is known by its group. */
const busyPaths = computed((): ReadonlySet<string> => status.value ? busyTabs(status.value.jobs, status.value.activities) : new Set());
/** A dot on their link in the top bar and the phone menu (markBusy). */
watch(busyPaths, markBusy);
</script>

<template>
  <main class="page" :data-tab="currentTab">
    <div v-if="status && !status.actions_enabled" class="banner" id="readonly">Read-only: the buttons are turned off on this server.</div>
    <OverviewTab v-if="currentTab === 'overview'" :status="status" @run="run" @stop="stop" @details="details"/>
    <PrepareTab v-else-if="currentTab === 'prepare'" :status="status" @run="run" @stop="stop" @details="details"/>
    <TrainTab v-else-if="currentTab === 'train'" :status="status" @run="run" @stop="stop" @details="details"
              @pipeline="startPipeline" @stop-pipeline="stopPipeline"/>
    <EvaluateTab v-else-if="currentTab === 'evaluate'" :status="status" @run="run" @stop="stop" @details="details"/>
    <DatasetTab v-else-if="currentTab === 'dataset'" :status="status" @run="run" @stop="stop" @details="details"/>
    <RoadsTab v-else-if="currentTab === 'roads'" :status="status" @run="run" @stop="stop" @details="details"/>
    <PublishTab v-else-if="currentTab === 'publish'" :status="status" @run="run" @stop="stop" @details="details"/>
    <ModelsTab v-else :status="status" @run="run" @stop="stop" @details="details"/>
  </main>
  <LogSheet :job="shownJob" @close="shownId = null" @stop="stop"/>
</template>
