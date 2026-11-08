<script setup lang="ts">
// The road labeler on Models (docs/ROAD_LABELER.md): an offline network that prepares Speed review (the kinds of road and
// the roadworks, from the camera at high definition, the map, the GPS and the speed), never on the car. Train it
// (helpers/road_labeler.sh train, the panel action "road_labeler_train"), a dry run on the CPU first if wanted; its job
// while it runs; then its runs from GET /api/panel/road-labeler (nnslr_tools.labeler_runs): the road type on validation
// beside the map's, each head's precision and recall, and whether it writes into Speed review or why not. A run with its
// checkpoint prepares Speed review ("road_labeler_predict"): its proposals, and the marks of the heads that write. A real
// run is refused while another process has the GPU (the server and the helper check it); a dry run is not. Hidden on a
// server without the actions, and the runs left out when the endpoint is not there yet.
import {computed, onMounted, ref, shallowRef, watch} from 'vue';
import {api} from '../common/api';
import {isoWhen, plural, sentence} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Job, LabelerHead, LabelerPayload, LabelerRun, PanelStatus} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import {pct} from './evaluate';
import JobProgress from './JobProgress.vue';
import {JOB_STATE, jobOf, startedWhen, terminalActivities} from './jobs';
import TerminalActivity from './TerminalActivity.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions];
  stop: [job: Job]; details: [job: Job]}>();

const ACTIONS: ActionName[] = ['road_labeler_train', 'road_labeler_predict'];
const trainState = computed((): ActionState | null => props.status?.actions?.road_labeler_train ?? null);
const predictState = computed((): ActionState | null => props.status?.actions?.road_labeler_predict ?? null);
/** A server from before the road labeler's actions: no card. */
const shown = computed(() => trainState.value !== null);
const job = computed(() => jobOf(props.status, ACTIONS));
const bigJob = computed(() => job.value && (job.value.state === 'running' || job.value.state === 'failed') ? job.value : null);
const activities = computed(() => terminalActivities(props.status, ['train']).filter((activity) => activity.label === 'road labeler'));
const idle = computed(() => !bigJob.value && !activities.value.length);
/** Another process on the GPU: only a dry run can start (start() refuses the others). */
const gpuBusy = computed(() => (props.status?.system?.gpu?.processes ?? 0) > 0);

const payload = shallowRef<LabelerPayload | null>(null);
async function load(): Promise<void> {
  try {
    payload.value = await api<LabelerPayload>('/api/panel/road-labeler');
  } catch {
    // Not on this server yet (it comes with a restart), or not readable now: the card shows no runs.
    payload.value = null;
  }
}
onMounted(() => void load());
const working = computed(() => job.value?.state === 'running' || activities.value.length > 0);
watch(working, (now, before) => {
  if (before && !now) void load();
});

const runs = computed((): LabelerRun[] => payload.value?.runs ?? []);
const drivesText = computed(() => typeof payload.value?.drives === 'number'
  ? ', ' + plural(payload.value.drives, 'drive') + ' with their pictures' : '');
const trained = computed(() => runs.value.filter((run) => !run.dry_run).length);
/** Prepare Speed review with the proposals only: no mark written. */
const proposalsOnly = ref(false);

function train(dry: boolean): void {
  if (trainState.value) emit('run', 'road_labeler_train', trainState.value, undefined, dry ? {dry: 'on'} : undefined);
}
function prepare(run: LabelerRun, dry: boolean): void {
  if (!predictState.value) return;
  const options: ActionOptions = {labeler: run.run, ...(dry ? {dry: 'on'} : {}), ...(proposalsOnly.value ? {marks: 'off'} : {})};
  emit('run', 'road_labeler_predict', predictState.value, undefined, options);
}

const headName = (head: LabelerHead): string => head.head === 'works' ? 'roadworks' : head.head.replace(/_/g, ' ');
const headNumbers = (head: LabelerHead): string => head.precision === null && head.recall === null
  ? (head.inside || head.outside ? 'one side alone on validation' : 'nothing to measure on validation')
  : 'precision ' + pct(head.precision) + ' · recall ' + pct(head.recall);
const headWrites = (head: LabelerHead): string => !head.writes ? 'held back: ' + (head.why ?? '')
  : head.with_write_only ? 'writes with --write only' : 'writes';
function typesText(run: LabelerRun): string {
  const types = run.types;
  if (types.right === null) return 'No road type to measure on validation.';
  const person = types.right_on_person === null ? ''
    : '; on the seconds a person set ' + pct(types.right_on_person) + ', the map ' + pct(types.map_on_person);
  return 'Road type right on ' + pct(types.right) + ' of validation' + person + '.';
}
const runFacts = (run: LabelerRun): string => [isoWhen(run.finished), plural(run.drives, 'drive'), plural(run.epochs, 'epoch'),
  run.dry_run ? 'dry run' : ''].filter(Boolean).join(' · ');
</script>

<template>
  <section v-if="shown" class="card import-card labeler-card" id="road-labeler" aria-labelledby="rl-title">
    <h2 id="rl-title" class="rl-title">Road labeler</h2>
    <p class="rl-what">An offline network that prepares Speed review: the kinds of road and the roadworks, from the camera at high
      definition, the map, the GPS and the speed. It never runs on the car; what it writes is a proposal you check.</p>
    <JobProgress v-if="bigJob" :job="bigJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
    <TerminalActivity v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity" :preparation="null"/>
    <div v-if="idle" class="import-row">
      <span class="import-icon" :class="{fresh: !trained}" aria-hidden="true"><Icon name="train"/></span>
      <div class="import-what">
        <strong>{{ trained ? plural(trained, 'run trained', 'runs trained') : 'Not trained yet' }}</strong>
        <span>Learns from the corrections of Speed review on the GPU{{ drivesText }}. The first run decodes the videos of every
          drive.</span>
      </div>
      <div class="rl-buttons">
        <button type="button" class="btn" id="labeler-dry" :disabled="!trainState?.allowed" @click="train(true)">Dry run</button>
        <button type="button" class="btn primary" id="labeler-train" :disabled="!trainState?.allowed || gpuBusy"
                @click="train(false)">Train</button>
      </div>
    </div>
    <p v-if="idle && trainState && !trainState.allowed" class="import-why">{{ sentence(trainState.reason) }}</p>
    <p v-else-if="idle && gpuBusy" class="import-why warn" id="labeler-gpu">The GPU is busy with another process: only a dry run
      on the CPU can start now.</p>
    <p v-if="job && !bigJob" class="import-last">Last {{ job.action === 'road_labeler_train' ? 'training' : 'preparation' }}
      {{ job.run ? job.run + ', ' : '' }}{{ startedWhen(job) }} ·
      {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
      <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>

    <ul v-if="runs.length" class="rl-runs" id="labeler-runs">
      <li v-for="run in runs" :key="run.run" class="rl-run" :data-run="run.run">
        <div class="rl-run-head"><b>{{ run.run }}</b><small>{{ runFacts(run) }}</small></div>
        <p class="rl-types">{{ typesText(run) }}</p>
        <ul class="rl-heads">
          <li v-for="head in run.heads" :key="head.head" class="rl-head" :class="{writes: head.writes}">
            <span class="rl-head-name">{{ headName(head) }}</span>
            <span class="rl-head-num">{{ headNumbers(head) }}</span>
            <span class="rl-head-writes">{{ headWrites(head) }}</span>
          </li>
        </ul>
        <div v-if="run.checkpoint && idle" class="rl-run-actions">
          <button type="button" class="btn" :disabled="!predictState?.allowed" @click="prepare(run, true)">Dry run</button>
          <button v-if="!run.dry_run" type="button" class="btn primary" :disabled="!predictState?.allowed || gpuBusy"
                  @click="prepare(run, false)">Prepare Speed review</button>
        </div>
      </li>
    </ul>
    <label v-if="runs.some((run) => run.checkpoint && !run.dry_run) && idle" class="rl-marks">
      <input type="checkbox" v-model="proposalsOnly"> Proposals only: Prepare Speed review writes no mark</label>
    <p v-if="idle && runs.length && predictState && !predictState.allowed" class="import-why">{{ sentence(predictState.reason) }}</p>
    <p v-if="payload && payload.total > runs.length" class="import-why">The {{ runs.length }} newest of {{ payload.total }} runs; the others
      are in data/runs.</p>
  </section>
</template>
