<script setup lang="ts">
// Overview, Drives, Speed review, Training dataset, Training, Versions, Drive, Results, Publish: one app, the tab shown follows the address (tabs.ts). One status for all
// tabs (/api/panel/status every 4 s, which also feeds the machine's history), the actions, and the sheet with the
// details of a job. Evaluate reads the training runs itself (/api/panel/runs), Models the networks by role
// (/api/panel/models). Speed review and Sign distance read only whether the buttons are on from the status, which costs
// the server seconds of CPU at each reading: there it is read once a minute (and in a hidden browser tab everywhere), the
// server pill and the dots of the busy tabs from the light /api/panel/system every 4 s, as on the other pages.
import {computed, onBeforeUnmount, onMounted, provide, ref, watch} from 'vue';
import {poll} from '../common/api';
import type {Activity, AiLive, Job, SystemInfo} from '../common/panel-types';
import {aiLive} from '../composables/ai-live';
import {markBusy} from '../composables/busy';
import {showConnection} from '../composables/connection';
import {usePanelStatus} from '../composables/panel-status';
import {usePoll} from '../composables/poll';
import {busyTabs} from './jobs';
import LogSheet from './LogSheet.vue';
import {recordMachine} from './machine-history';
import DatasetTab from './DatasetTab.vue';
import DriveTab from './DriveTab.vue';
import EvaluateTab from './EvaluateTab.vue';
import ModelsTab from './ModelsTab.vue';
import OverviewTab from './OverviewTab.vue';
import PrepareTab from './PrepareTab.vue';
import PublishTab from './PublishTab.vue';
import SignDistanceTab from './SignDistanceTab.vue';
import CrossingsTab from './CrossingsTab.vue';
import ReviewersTab from './ReviewersTab.vue';
import SpeedReviewTab from './SpeedReviewTab.vue';
import {currentTab, useTabRouting} from './tabs';
import TrainTab from './TrainTab.vue';
import {RESUME_JOB, RETRY_ACTIONS, RETRY_JOB, usePanelActions} from './use-actions';

useTabRouting();
/** The tabs that need of the status only actions_enabled: their own data comes from their own addresses. */
const LIGHT_TABS: ReadonlySet<string> = new Set(['speedreview', 'signdistance', 'crossings']);
const STATUS_MS = 4000;
const LIGHT_STATUS_MS = 60000;
const hidden = ref(document.hidden);
const light = computed(() => LIGHT_TABS.has(currentTab.value));
const {status, refresh} = usePanelStatus((value) => { recordMachine(value.system); aiLive.value = value.ai_live ?? null; },
  () => light.value || hidden.value ? LIGHT_STATUS_MS : STATUS_MS);
// Back on a tab that shows the status, or on the browser tab: a fresh one now.
watch([light, hidden], ([isLight, isHidden], [wasLight, wasHidden]) => {
  if (!isLight && !isHidden && (wasLight || wasHidden)) void refresh();
});
function onVisibility(): void {
  hidden.value = document.hidden;
}
onMounted(() => document.addEventListener('visibilitychange', onVisibility));
onBeforeUnmount(() => document.removeEventListener('visibilitychange', onVisibility));
/** On the light tabs, and until the first status comes, the server pill and the busy dots every 4 s from
 * /api/panel/system (0.15 s, 1 kB): the pill shows at once after a reload, not after the whole status. */
usePoll(async () => {
  if (hidden.value || (!light.value && status.value !== null)) return;
  try {
    const answer = await poll<{system: SystemInfo; running?: {jobs: Pick<Job, 'action' | 'state'>[]; activities: Activity[]; ai_live?: AiLive | null}}>(
      '/api/panel/system');
    recordMachine(answer.system);
    if (answer.running) {
      markBusy(busyTabs(answer.running.jobs, answer.running.activities));
      aiLive.value = answer.running.ai_live ?? null;
    }
    showConnection(null);
  } catch (error) {
    showConnection(error);
  }
}, STATUS_MS);
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
    <SpeedReviewTab v-else-if="currentTab === 'speedreview'" :read-only="status?.actions_enabled === false"/>
    <DriveTab v-else-if="currentTab === 'drive'" :status="status" @run="run" @stop="stop" @details="details"/>
    <SignDistanceTab v-else-if="currentTab === 'signdistance'" :read-only="status ? !status.actions_enabled : null"/>
    <CrossingsTab v-else-if="currentTab === 'crossings'" :read-only="status ? !status.actions_enabled : null"/>
    <PublishTab v-else-if="currentTab === 'publish'" :status="status" @run="run" @stop="stop" @details="details"/>
    <ReviewersTab v-else-if="currentTab === 'reviewers'" :status="status" @run="run" @stop="stop" @details="details"/>
    <ModelsTab v-else :status="status" @run="run" @stop="stop" @details="details"/>
  </main>
  <LogSheet :job="shownJob" @close="shownId = null" @stop="stop"/>
</template>
