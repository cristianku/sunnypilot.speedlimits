<script setup lang="ts">
// Train › Pipeline: our networks trained one after the other, each with the choices of its own section (its signs, where
// it starts...), each a job of
// its own (POST /api/panel/pipeline; the server starts the next one when one ends well). A classifier after the
// cutter learns the mistakes of that new cutter. While it runs: the queue, the training at work, "Stop after this
// one". Below, folded: the first pipeline of all, the detector and the reader (helpers/3_train.sh).
import {computed, ref} from 'vue';
import {isoWhen, sentence} from '../common/format';
import type {ActionName, ActionOptions, ActionState, CutterFamilies, Job, PanelStatus, PipelineAction,
  PipelineStep, TrainingStart} from '../common/panel-types';
import type {ClassifierChoices as Choices} from './classifier-choices';
import ClassifierChoices from './ClassifierChoices.vue';
import type {CutterChoices} from './cutter-choices';
import JobProgress from './JobProgress.vue';
import MapillaryChoices from './MapillaryChoices.vue';
import type {NetworkStartChoice, StartRole} from './network-start';
import NetworkStartPicker from './NetworkStartPicker.vue';
import type {BrainChoice} from './brain-choices';
import BrainChoices from './BrainChoices.vue';
import Segmented from './Segmented.vue';
import TrainStart from './TrainStart.vue';

const props = defineProps<{status: PanelStatus | null; choices: Choices; cutterChoices: CutterChoices;
  starts: Record<StartRole, NetworkStartChoice>;
  brainChoices: BrainChoice; fresh: number; familyOptions: {key: CutterFamilies; label: string}[]}>();
const families = defineModel<CutterFamilies>('families', {required: true});
const emit = defineEmits<{pipeline: [steps: {action: PipelineAction; options: ActionOptions}[], text: string];
  stopPipeline: []; stop: [job: Job]; details: [job: Job]; run: [name: ActionName, state: ActionState, start?: TrainingStart]}>();

const STEPS: {action: PipelineAction; label: string; text: string}[] = [
  {action: 'dataset', label: 'Import the sign reviews', text: 'The Sign review frames not imported yet join the dataset, and its split is made again.'},
  {action: 'train_cutter', label: 'Cutter', text: 'A new SignNet. About 80 minutes.'},
  {action: 'train_classifier', label: 'Classifier', text: 'Learns the signs and the cutter’s mistakes. A few hours.'},
  {action: 'train_decider', label: 'Applicability', text: 'Three versions, compared. About 15 minutes.'},
  {action: 'train_brain', label: 'Brain', text: 'Three seeds, from your speed on every drive with sunnypilot off and its logs read, the next limit from the signs our networks in use read. About 15 minutes, and about 40 s for each drive they have not read yet.'},
];
const LABEL = Object.fromEntries(STEPS.map((step) => [step.action, step.label])) as Record<PipelineAction, string>;

/** The trainings chosen: every network; the import only with reviews still to import. */
const picked = ref<Partial<Record<PipelineAction, boolean>>>({});
const on = (action: PipelineAction): boolean => picked.value[action] ?? (action !== 'dataset' || props.fresh > 0);
function toggle(action: PipelineAction, value: boolean): void {
  picked.value = {...picked.value, [action]: value};
}
const chosen = computed(() => STEPS.filter((step) => on(step.action)));
const withCutter = computed(() => on('train_cutter'));

function optionsOf(action: PipelineAction): ActionOptions {
  if (action === 'train_cutter') return {families: families.value, ...props.starts.cutter.option, ...props.cutterChoices.options};
  if (action === 'train_classifier') return props.choices.options;
  if (action === 'train_decider') return props.starts.decider.option;
  if (action === 'train_brain') return {...props.brainChoices.option, ...props.starts.brain.option};
  return {};
}
/** Why it cannot start now ('' when it can): the reason of a training chosen (the classifier after a new cutter needs
 * no cutter in use), or the share of Mapillary of the cutter or of the classifier. */
const reason = computed(() => {
  const actions = props.status?.actions;
  if (!actions) return '';
  if (!chosen.value.length) return 'Choose at least one training.';
  for (const step of chosen.value) {
    const state = actions[step.action];
    if (step.action === 'train_classifier' && withCutter.value) continue;
    if (state && !state.allowed) return sentence(step.label + ': ' + state.reason);
  }
  if (on('train_cutter') && props.cutterChoices.invalid) return sentence('Cutter: ' + props.cutterChoices.invalid);
  if (on('train_classifier') && props.choices.invalid) return sentence('Classifier: ' + props.choices.invalid);
  return '';
});
const startLabel = computed(() => chosen.value.length === 1 ? 'Train ' + chosen.value[0]!.label.toLowerCase()
  : 'Train these ' + chosen.value.length + ' in a row');

function start(): void {
  const steps = chosen.value.map((step) => ({action: step.action, options: optionsOf(step.action)}));
  const words = chosen.value.map((step) => step.action === 'train_cutter'
    ? 'the cutter (' + (families.value === 'all' ? 'every sign' : 'speed signs only') + props.starts.cutter.words
      + props.cutterChoices.words + ')'
    : step.action === 'train_classifier' ? 'the classifier' + props.choices.words + (withCutter.value ? ', on that new cutter' : '')
      : step.action === 'dataset' ? 'the import of the reviews'
        : step.action === 'train_brain' ? 'the brain' + props.brainChoices.words + props.starts.brain.words
          : LABEL[step.action].toLowerCase() + props.starts.decider.words);
  emit('pipeline', steps, 'One after the other on the GPU, each a training of its own: ' + words.join(', then ')
    + '. The next one starts when one ends well; a failed one stops the pipeline. Each run is a new one, named by the time it starts.');
}

/** The last pipeline (status.pipeline): the queue while it runs, a line once it ended. */
const pipeline = computed(() => props.status?.pipeline ?? null);
const running = computed(() => pipeline.value?.state === 'running');
const jobOfStep = (step: PipelineStep): Job | null => props.status?.jobs.find((job) => job.id === step.job) ?? null;
const current = computed(() => {
  const step = pipeline.value?.steps.find((item) => item.state === 'running');
  return step ? jobOfStep(step) : null;
});
const MARK: Record<PipelineStep['state'], [kind: string, text: string, word: string]> = {
  ok: ['ok', '✓', 'done'], failed: ['bad', '✕', 'failed'], interrupted: ['bad', '✕', 'stopped'],
  running: ['run', '', 'training'], todo: ['warn', '', 'next'], skipped: ['warn', '–', 'not started'],
};
const ENDED: Record<string, string> = {done: 'done', failed: 'failed', stopped: 'stopped'};
</script>

<template>
  <!-- The queue at work: each training with its state, the one at work with its progress. -->
  <template v-if="pipeline && running">
    <ol class="pl-queue" id="pipeline-queue">
      <li v-for="(step, index) in pipeline.steps" :key="step.action" class="pl-row" :data-state="step.state">
        <span class="job-mark" :class="MARK[step.state][0]" aria-hidden="true">{{ step.state === 'todo' ? index + 1 : MARK[step.state][1] }}</span>
        <span class="pl-what"><span class="pl-label">{{ LABEL[step.action] ?? step.label }}</span>
          <span class="pl-sub">{{ MARK[step.state][2] }}<template v-if="step.run"> · {{ step.run }}</template></span></span>
        <button v-if="jobOfStep(step)" type="button" class="link-button" @click="emit('details', jobOfStep(step)!)">Details</button>
      </li>
    </ol>
    <p v-if="pipeline.reason" class="latest-none" id="pipeline-reason">{{ pipeline.reason }}</p>
    <JobProgress v-if="current" :job="current" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
    <div class="pl-go">
      <button class="btn" type="button" id="pipeline-stop" @click="emit('stopPipeline')">Stop after this one</button>
    </div>
  </template>

  <!-- The trainings to chain, each with the choices of its own section. -->
  <template v-else>
    <ol class="pl-steps" id="pipeline-steps">
      <li v-for="step in STEPS" :key="step.action" class="pl-step" :class="{off: !on(step.action)}" :data-action="step.action">
        <label class="pl-head">
          <input type="checkbox" :checked="on(step.action)" :id="'pipeline-' + step.action"
                 @change="toggle(step.action, ($event.target as HTMLInputElement).checked)">
          <span class="pl-what"><span class="pl-label">{{ step.label }}</span>
            <span class="pl-sub">{{ step.action === 'dataset' && fresh === 0 ? 'Nothing new to import: the split is made again.'
              : step.action === 'train_cutter' ? 'A new SignNet. ' + cutterChoices.minutes + '.' : step.text }}</span></span>
        </label>
        <div v-if="on(step.action) && step.action === 'train_cutter'" class="pl-choices">
          <NetworkStartPicker :start="starts.cutter" id="pipeline-cutter-start"/>
          <div class="seg-row"><Segmented v-model="families" :options="familyOptions" label="Signs the cutter finds"/></div>
          <MapillaryChoices :choices="cutterChoices" :status="status" label="Frames the cutter learns from" id-prefix="cutter-mapillary"
                            note-id="pipeline-cutter-mapillary-note"
                            @locate="status && emit('run', 'mapillary_locations', status.actions.mapillary_locations)"/>
          <p v-if="cutterChoices.withMapillary" class="latest-none" id="pipeline-cutter-mapillary-note">The Mapillary windows join its
            training only: every measure stays on your drives.</p>
        </div>
        <div v-else-if="on(step.action) && step.action === 'train_classifier'" class="pl-choices">
          <ClassifierChoices :choices="choices" :status="status"
                             @locate="status && emit('run', 'mapillary_locations', status.actions.mapillary_locations)"/>
          <p class="latest-none" id="pipeline-cutter-note">{{ withCutter ? 'It learns the mistakes of the cutter trained just before.'
            : 'It learns the mistakes of the cutter in use (Models).' }}</p>
        </div>
        <div v-else-if="on(step.action) && step.action === 'train_decider' && starts.decider.options.length > 1" class="pl-choices">
          <NetworkStartPicker :start="starts.decider" id="pipeline-decider-start"/>
        </div>
        <div v-else-if="on(step.action) && step.action === 'train_brain'" class="pl-choices">
          <NetworkStartPicker :start="starts.brain" id="pipeline-brain-start"/>
          <BrainChoices :choices="brainChoices" id="pipeline-brain"/>
        </div>
      </li>
    </ol>
    <div class="pl-go">
      <button class="btn primary" type="button" id="pipeline-start" :disabled="!status?.actions_enabled || Boolean(reason)"
              :aria-describedby="reason ? 'pipeline-reason' : undefined" @click="start">{{ startLabel }}</button>
      <p v-if="reason" class="act-reason" id="pipeline-reason">{{ reason }}</p>
    </div>
    <p v-if="pipeline" class="last-line" id="pipeline-last">Last pipeline: {{ pipeline.steps.map((step) => LABEL[step.action] ?? step.label).join(', ') }},
      {{ pipeline.started_at ? isoWhen(pipeline.started_at) : '' }} · {{ ENDED[pipeline.state] }}<template
      v-if="pipeline.reason"> · {{ pipeline.reason }}</template></p>
  </template>

  <!-- The first pipeline of all: the detector and the reader, the way every model was trained before the networks. -->
  <details class="latest-row pl-old" id="pipeline-old">
    <summary><span class="latest-what"><span class="latest-name">Detector and reader</span>
      <span class="latest-text">The first pipeline: import, split, detector and reader, to a package</span></span><span
      class="latest-chev" aria-hidden="true">›</span></summary>
    <div class="pl-old-body">
      <TrainStart :starts="status ? status.training_starts ?? [] : null" :state="status?.actions.train ?? null"
                  @start="(start) => status && emit('run', 'train', status.actions.train, start)"/>
    </div>
  </details>
</template>
