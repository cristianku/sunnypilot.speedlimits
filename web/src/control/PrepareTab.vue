<script setup lang="ts">
// Prepare: from the drives copied from the comma (the comma pill of the top bar downloads them) to frames with the
// proposed signs: who proposes (with its measure), the actions, the download or preparation running (or the last
// one), the map's road and limit of the prepared drives (Read the map, RoadTypesCard.vue), the routes.
import {computed, onMounted, ref, shallowRef} from 'vue';
import {api, messageOf, post} from '../common/api';
import {plural, sentence, thousands} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Job, PanelStatus, Proposer, ProposersPayload,
  RouteState} from '../common/panel-types';
import {toast} from '../composables/toast';
import Segmented from './Segmented.vue';
import ActionList from './ActionList.vue';
import JobProgress from './JobProgress.vue';
import {activityTab, JOB_STATE, jobOf, PREPARE_ACTIONS, startedWhen, terminalActivities} from './jobs';
import {cardLabel, cardMemoryText, freestCard, gpuCards} from './machine-history';
import PrepRoutes from './PrepRoutes.vue';
import {networksText, PROPOSER_LABELS, proposerText} from './proposer-words';
import {reviewSummary} from './review-routes';
import RoadTypesCard from './RoadTypesCard.vue';
import RouteTable from './RouteTable.vue';
import TerminalActivity from './TerminalActivity.vue';
import SectionHead from './SectionHead.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions]; stop: [job: Job];
  details: [job: Job]}>();

/** Who proposes the signs: the choice is kept on the server for every browser (POST /api/panel/proposer); before one
 * is made, the server offers the one measured best (Evaluate › Proposers). */
const PROPOSERS: {key: Proposer; label: string}[] = (['vietnam', 'ours', 'qwen'] as const).map((key) => ({key, label: PROPOSER_LABELS[key]}));
/** What each one is; its measure follows (measureText). */
const PROPOSER_TEXT: Record<Proposer, string> = {
  vietnam: 'A ready-made model trained on Vietnamese signs: it boxes and reads them by itself.',
  ours: 'Our cutter draws the boxes, our classifier reads them and our applicability network says which road each is for, all trained on your reviews.',
  qwen: 'A model that looks at the whole picture, boxes every sign and reads it in one go.',
};
/** The proposers as measured and the choice kept on the server (GET /api/panel/proposers, once per visit). */
const measures = shallowRef<ProposersPayload | null>(null);
const chosen = ref<Proposer | null>(null);
async function loadProposers(): Promise<void> {
  try {
    measures.value = await api<ProposersPayload>('/api/panel/proposers');
    chosen.value = measures.value.chosen ?? 'vietnam';
  } catch {
    // An older server, or the network: the default of the helpers until it answers.
    chosen.value ??= 'vietnam';
  }
}
onMounted(() => void loadProposers());
async function choose(value: Proposer): Promise<void> {
  const before = chosen.value;
  chosen.value = value;
  try {
    measures.value = await post<ProposersPayload>('/api/panel/proposer', {proposer: value});
  } catch (error) {
    chosen.value = before;
    toast(messageOf(error), true);
  }
}
/** A preparation of the site running with a known proposer: its choice is shown, and cannot change until it ends. */
const runningProposer = computed((): Proposer | null => job.value?.state === 'running'
  && (job.value.action === 'prepare' || job.value.action === 'prepare_again') ? job.value.proposer ?? null : null);
const proposer = computed<Proposer>({
  get: () => runningProposer.value ?? chosen.value ?? 'vietnam',
  set: (value) => void choose(value),
});
/** "On 604 test frames it found 60 of 80 speed signs and read 51 of them right, at 0.8 s a picture.", or how to
 * measure it. */
function measureText(key: Proposer): string {
  if (measures.value === null) return '';
  const report = measures.value.reports.find((item) => item.proposer === key && item.partition === 'test');
  if (!report) return 'Not measured yet: Evaluate › Proposers measures it.';
  const speed = report.speed_signs;
  const share = speed.read_right_share ?? 0;
  return 'On ' + thousands(report.frames) + ' test frames' + (report.limited ? ' (a few only)' : '') + ' it found '
    + speed.found + ' of ' + speed.truth + ' speed signs and read ' + (share < 0.1 ? 'only ' : '') + speed.read_right
    + ' of them right, at ' + report.seconds_per_frame.toFixed(1) + ' s a picture.'
    + (measures.value.best === key ? ' The best so far.' : '');
}
/** When more than half of the memory is taken on the GPU with the most room (Qwen3-VL needs about 17 GB to load), how much
 * ("31.5 of 32 GB"), in a sentence; '' when it fits. */
const gpuTaken = computed(() => {
  const system = props.status?.system;
  const gpu = freestCard(system);
  if (!gpu || gpu.memory_used_mb === null || !gpu.memory_total_mb || gpu.memory_used_mb < gpu.memory_total_mb / 2) return '';
  const cards = gpuCards(system);
  if (cards.length < 2) {
    return 'The GPU is taken now: ' + cardMemoryText(gpu) + ' of its memory are in use by another program. Install tries'
      + ' Qwen3-VL on the GPU: free it first.';
  }
  return (cards.length === 2 ? 'Both GPUs are' : 'Every GPU is') + ' taken now: the one with the most room, '
    + cardLabel(gpu, cards.indexOf(gpu), cards.length) + ', has ' + cardMemoryText(gpu) + ' in use by other programs.'
    + ' Install tries Qwen3-VL on a GPU: free one first.';
});
/** Why the chosen proposer cannot run here, if it cannot. */
const unavailable = computed(() => proposer.value === 'qwen' ? props.status?.proposers?.qwen ?? null : null);
/** Why Our networks cannot propose yet: they propose only whole, cutter, classifier and decider in use. */
const oursMissing = computed(() => proposer.value === 'ours' ? props.status?.proposers?.ours ?? null : null);
/** Why Prepare and Propose again cannot start now with the proposer shown: Qwen3-VL missing, or the GPU memory it
 * needs taken (the server refuses it then: said on the cards before the click, not after the confirmation). */
const proposerBlock = computed((): string => {
  if (oursMissing.value) return sentence(oursMissing.value);
  if (proposer.value !== 'qwen') return '';
  if (unavailable.value) return 'Qwen3-VL is not ready on this server yet: install it above, or choose another';
  return props.status?.proposers?.qwen_gpu ?? '';
});
const prepareActions = computed((): PanelStatus['actions'] | null => {
  const actions = props.status?.actions ?? null;
  const why = proposerBlock.value;
  if (actions === null || !why) return actions;
  const blocked = (state: ActionState): ActionState => state.allowed ? {...state, allowed: false, reason: why} : state;
  return {...actions, prepare: blocked(actions.prepare), prepare_again: blocked(actions.prepare_again)};
});
/** Routes with proposals and frames still to confirm: "Propose again" redoes those frames, the confirmed ones stay. */
const unstarted = computed(() => (props.status?.routes ?? []).filter((row) => row.state === 'revisionare').length);
const PROPOSE_ACTIONS = computed((): [ActionName, string][] => unstarted.value ? [['prepare_again',
  'New proposals for the frames not confirmed yet of ' + plural(unstarted.value, 'drive')
  + '. The confirmed frames stay as they are.']] : []);

/** The frames and the proposals, with the proposer chosen above them (the drives come from the comma pill of the
 * top bar). */
/** Under the Qwen3-VL choice while it is missing (scripts/setup_vlm.sh). */
const INSTALL_ACTIONS: [ActionName, string][] = [
  ['setup_vlm', 'Checks the files you copied (or downloads the missing ones, 17.5 GB), installs what it needs and tries it on the GPU.'],
];
const PREPARE_ITEMS: [ActionName, string][] = [
  ['prepare', 'Takes the pictures out of the new drives and proposes the signs, ready to review.'],
];
/** How many drives wait for each step, in plain words: their videos found, their pictures taken out, their signs found. */
const COUNTS: [RouteState, string][] = [['collegare', 'waiting for their videos'], ['estrarre', 'waiting for their pictures'],
  ['preannotare', 'waiting for their signs']];

const job = computed(() => jobOf(props.status, PREPARE_ACTIONS));
// The map reading shares the group of the preparation: its card shows it (RoadTypesCard.vue), with its job.
const activities = computed(() => terminalActivities(props.status, ['fetch', 'prepare'])
  .filter((activity) => activityTab(activity) === '/prepare' && activity.label !== 'map reading'));
const counts = computed(() => COUNTS.map(([state, label]) => ({state, label,
  count: (props.status?.routes ?? []).filter((row) => row.state === state).length})));
const total = computed(() => counts.value.reduce((sum, item) => sum + item.count, 0));
const running = computed(() => job.value?.state === 'running' || activities.value.length > 0);
/** The job shown as a card: running, or failed (its error must be read); one done or stopped by hand is a line. */
const bigJob = computed(() => job.value && (job.value.state === 'running' || job.value.state === 'failed') ? job.value : null);
const WAITING: RouteState[] = ['collegare', 'estrarre', 'preannotare'];
const notReady = computed(() => (props.status?.routes ?? []).filter((row) => WAITING.includes(row.state)));
/** The drives with frames left to review (in progress or not started), and those reviewed to the end. */
const review = computed(() => reviewSummary(props.status));
const toReview = computed(() => review.value.inProgress + review.value.toStart);
/** Who proposes in a preparation, with Our networks' parts by their run: "Our networks: cutter SignNet, 30 Oct 10:46 · …". */
function proposerLine(job: Job): string {
  return job.proposer && job.proposer in PROPOSER_LABELS ? proposerText(job.proposer as Proposer, job.networks)
    : String(job.proposer);
}
/** The networks Our networks run now (the ones on the car, models/on-car.json; a part it has none of, the one in use in
 * Models), in words; '' while unknown. */
const oursInUse = computed(() => networksText(props.status?.proposers?.networks ?? measures.value?.networks));

/** The comma pill of the top bar, opened as a click on it would: Download new drives is there. */
function openComma(): void {
  const pill = document.querySelector<HTMLButtonElement>('#comma-pill .tm-pill');
  if (pill === null) return;
  if (pill.getAttribute('aria-expanded') !== 'true') pill.click();
  pill.focus();
}
</script>

<template>
  <SectionHead tab="prepare"/>

  <section class="card tab-card" aria-labelledby="prepare-title">
    <h2 class="sr-only" id="prepare-title">New drives</h2>
    <p v-if="total" class="figure-note prep-state" id="prepare-big">{{ plural(total, 'drive') }} copied from the comma, waiting
      to be prepared.</p>
    <p v-else-if="status" class="figure-note prep-state" id="prepare-big">No new drive to prepare. New drives come from the comma:
      <button type="button" class="link-button" @click="openComma">Download new drives</button></p>
    <ul v-if="total" class="chips">
      <li v-for="item in counts.filter((entry) => entry.count)" :key="item.state"><strong>{{ item.count }}</strong> {{ item.label }}</li>
    </ul>
    <h3 class="prep-step">Find the signs</h3>
    <p class="figure-note prep-state">Draws a box around every sign in every picture and reads it, for you to confirm in Review.</p>
    <div class="proposer">
      <span class="proposer-label">Found by</span>
      <Segmented v-model="proposer" :options="PROPOSERS" label="Who proposes the signs" :disabled="runningProposer !== null"/>
      <p class="figure-note">{{ PROPOSER_TEXT[proposer] }} {{ measureText(proposer) }}<template v-if="runningProposer"> It is
        proposing now: you can choose again when it ends.</template></p>
      <p v-if="proposer === 'ours' && oursInUse" class="figure-note" id="proposer-networks">The car's networks: {{ oursInUse }}.
        <a href="/publish">Publish puts new ones on the car</a></p>
      <p v-if="oursMissing" class="figure-note proposer-missing">{{ sentence(oursMissing) }}</p>
      <template v-if="unavailable">
        <p class="figure-note proposer-missing" :title="unavailable">{{ unavailable.includes('not checked yet')
          ? 'Qwen3-VL is copied here but not checked yet.' : 'Qwen3-VL is not installed on this server yet.' }}</p>
        <p v-if="gpuTaken" class="figure-note proposer-missing">{{ gpuTaken }}</p>
        <ActionList :items="INSTALL_ACTIONS" :actions="status?.actions ?? null" @run="(name, state) => emit('run', name, state)"/>
      </template>
    </div>
    <ActionList :items="[...PREPARE_ITEMS, ...PROPOSE_ACTIONS]" :actions="prepareActions"
                @run="(name, state) => emit('run', name, state, undefined, {proposer})"/>
  </section>

  <!-- Only what is happening now, or a job that failed: a finished job is one line. -->
  <section v-if="bigJob || activities.length" class="live-section" :class="{now: running}" aria-labelledby="prepare-job-title">
    <h2 class="section-title" id="prepare-job-title">{{ running ? 'Running' : 'Last job failed' }}</h2>
    <div class="jp-list" aria-live="polite">
      <JobProgress v-if="bigJob" :job="bigJob" @details="$emit('details', $event)" @stop="$emit('stop', $event)">
        <p v-if="bigJob.proposer" class="figure-note prep-by">Found by {{ proposerLine(bigJob) }}</p>
        <PrepRoutes v-if="(bigJob.action === 'prepare' || bigJob.action === 'prepare_again') && bigJob.state === 'running' && status?.preparation" :routes="status.preparation.working"/>
      </JobProgress>
      <TerminalActivity v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity" :preparation="status?.preparation ?? null"/>
    </div>
  </section>
  <p v-else-if="job" class="last-line">Last: {{ job.label }}, {{ startedWhen(job) }} · {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
    <button type="button" class="link-button" @click="$emit('details', job)">Details</button></p>

  <!-- The map's road and limit of the prepared drives: Map now and the Server today lane of the drive player. -->
  <RoadTypesCard :status="status" @run="(name, state) => emit('run', name, state)" @stop="emit('stop', $event)"
                 @details="emit('details', $event)"/>

  <!-- The routes still on their way; the ready ones are on the Review tab, as drives. -->
  <RouteTable v-if="notReady.length" :routes="notReady"/>
  <p v-if="toReview" class="last-line"><a href="/speed-review">{{ plural(toReview, 'drive') }} to review ›</a></p>
  <p v-else-if="review.reviewed" class="last-line"><a href="/speed-review">Every drive is reviewed ›</a></p>

  <!-- What the networks learn from: the reviews imported, by sign family, who confirmed them, the split. -->
</template>
