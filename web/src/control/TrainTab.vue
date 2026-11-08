<script setup lang="ts">
// Training: from the reviews to a model, as an app on the phone. One line says what the networks learn from (objects,
// frames, drives; the reviews not imported yet), a link to Data › Training dataset, where they are imported. Then one network at a time, chosen
// in a switcher with a dot on the one training: the cutter (scripts/run_signnet.sh: SignNet, the cutter the comma runs,
// at the car's band; the signs it finds, its Mapillary switch and the cut of its windows), the classifier with "not a sign" (scripts/run_classifier.sh), the decider (in preparation:
// experiment B), the brain (scripts/run_brain.sh: its three seeds, from every drive with its logs read: his own speed with
// the blocks of Speed review, the next limit of Sign review, which it says before the click), the pipeline (PipelineSection: those trainings one
// after the other; the first one, helpers/3_train.sh, folded under it). Each has the same place for its work:
// the Train button, or its training running (or failed) with its steps, its Settings for the next training
// (TrainSettingsSection), then its curves and its latest version
// (GET /api/panel/models, a link to its versions in Models). The page does not move while work runs.
import {computed, ref, shallowRef, toRef, watch} from 'vue';
import {api, messageOf} from '../common/api';
import {isoWhen, plural, thousands} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Activity, BrainLabels, CutterFamilies, Job, JobProgress as Progress,
  ModelRole, ModelVersion, PanelStatus, PipelineAction, TrainingStart, TrainSettingsAction} from '../common/panel-types';
import ActionList from './ActionList.vue';
import {useBrainChoices} from './brain-choices';
import BrainChoices from './BrainChoices.vue';
import {useClassifierChoices} from './classifier-choices';
import ClassifierChoices from './ClassifierChoices.vue';
import {useCutterChoices} from './cutter-choices';
import JobProgress from './JobProgress.vue';
import MapillaryChoices from './MapillaryChoices.vue';
import {CHARTED_ACTIONS, JOB_STATE, jobOf, PIPELINE_ACTIONS, startedWhen, terminalActivities} from './jobs';
import {pct} from './evaluate';
import {sizeText, useModels} from './models';
import {useNetworkStart} from './network-start';
import NetworkStartPicker from './NetworkStartPicker.vue';
import PipelineSection from './PipelineSection.vue';
import Segmented from './Segmented.vue';
import TerminalActivity from './TerminalActivity.vue';
import TrainingCharts from './TrainingCharts.vue';
import TrainSettingsSection from './TrainSettingsSection.vue';
import SectionHead from './SectionHead.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: TrainingStart, options?: ActionOptions]; stop: [job: Job];
  details: [job: Job]; pipeline: [steps: {action: PipelineAction; options: ActionOptions}[], text: string]; stopPipeline: []}>();

/** The page as an app: one line of data, then one network at a time (Cutter, Classifier, Applicability, Brain, the whole
 * Pipeline), whose Train button becomes its progress, in the same place, while it trains. */
type Section = 'cutter' | 'classifier' | 'decider' | 'brain' | 'pipeline';
const SECTIONS: {key: Section; label: string}[] = [{key: 'cutter', label: 'Cutter'}, {key: 'classifier', label: 'Classifier'},
  {key: 'decider', label: 'Applicability'}, {key: 'brain', label: 'Brain'}, {key: 'pipeline', label: 'Pipeline'}];
function sectionOf(action: ActionName): Section | null {
  if (action === 'train_cutter') return 'cutter';
  if (action === 'train_classifier') return 'classifier';
  if (action === 'train_decider') return 'decider';
  if (action === 'train_brain') return 'brain';
  return PIPELINE_ACTIONS.includes(action) ? 'pipeline' : null;
}
/** The newest job of a section (running or the last one). */
function jobFor(section: Section): Job | null {
  return props.status?.jobs.find((item) => sectionOf(item.action) === section) ?? null;
}
/** The work started from a terminal, by section (its label names the network; else the pipeline). */
function activitiesFor(section: Section): Activity[] {
  return activities.value.filter((activity) => {
    const named: Section = activity.label.includes('cutter') ? 'cutter' : activity.label.includes('classifier') ? 'classifier'
      : activity.label.includes('brain') ? 'brain' : 'pipeline';
    return named === section;
  });
}
/** The pipeline of our networks at work (status.pipeline): its section is marked and shown first. */
const pipelineRunning = computed(() => props.status?.pipeline?.state === 'running');
const runningSections = computed(() => SECTIONS.map((item) => item.key)
  .filter((key) => jobFor(key)?.state === 'running' || activitiesFor(key).length > 0 || (key === 'pipeline' && pipelineRunning.value))
  .sort((a, b) => Number(b === 'pipeline' && pipelineRunning.value) - Number(a === 'pipeline' && pipelineRunning.value)));
/** #pipeline in the address (Train all of the Overview) opens the pipeline. */
const picked = ref<Section | null>(location.hash === '#pipeline' ? 'pipeline' : null);
/** The section shown: the one picked, else the one training (the pipeline first), else the cutter. */
const section = computed({
  get: (): Section => picked.value ?? runningSections.value[0] ?? 'cutter',
  set: (value: Section) => { picked.value = value; },
});
const job = computed(() => jobFor(section.value));
/** The training whose settings the section shows (Settings, under its action): none for the whole pipeline. */
const SETTINGS_OF: Partial<Record<Section, TrainSettingsAction>> = {cutter: 'train_cutter', classifier: 'train_classifier',
  decider: 'train_decider', brain: 'train_brain'};
const settingsAction = computed(() => SETTINGS_OF[section.value] ?? null);
/** The job of the section in its slot: running or failed; one done or stopped by hand is a line. In the pipeline: only a
 * first pipeline (detector and reader) at work; the last line is PipelineSection's. */
const bigJob = computed(() => job.value && (job.value.state === 'running' || (job.value.state === 'failed' && section.value !== 'pipeline'))
  ? job.value : null);
/** Importing the reviews (Data › Training dataset): said in the line of data. */
const importing = computed(() => jobOf(props.status, ['dataset'])?.state === 'running');

/** The networks by role: every version trained here, newest first (the Models tab's data). */
const {payload: models} = useModels();
const roleOf = (role: ModelRole) => models.value?.roles.find((item) => item.role === role) ?? null;
/** The latest version of a role (newest weights), or null. */
const latest = (role: ModelRole): ModelVersion | null => roleOf(role)?.versions[0] ?? null;
const latestCutter = computed(() => latest('cutter'));
/** The cutter in use (Models): the classifier's training mines its false boxes and measures the chain with it. */
const cutterInUse = computed(() => roleOf('cutter')?.versions.find((version) => version.in_use) ?? null);
/** Its numbers as the car reads it (best 5 boxes of 0.3 or more), on the test, else validation: the lenient count (a box
 * on the sign counts, car_centre) when measured, else the strict one (IoU 0.5); strict: the strict one beside the
 * lenient, shown in small (null without the lenient). */
const cutterInUseNumbers = computed(() => {
  const version = cutterInUse.value;
  if (version?.role !== 'cutter') return null;
  const numbers = version.test?.car ? version.test : version.validation?.car ? version.validation : null;
  return numbers?.car ? {main: numbers.car_centre ?? numbers.car, strict: numbers.car_centre ? numbers.car : null} : null;
});
const cutterInUseCar = computed(() => cutterInUseNumbers.value?.main ?? null);
const cutterInUseStrict = computed(() => cutterInUseNumbers.value?.strict ?? null);
const cutterInUsePartition = computed(() => cutterInUse.value?.role === 'cutter' && cutterInUse.value.test?.car ? 'test' : 'validation');
const latestClassifier = computed(() => latest('classifier'));
const latestDecider = computed(() => latest('decider'));
const roleLabel = (role: ModelRole): string => roleOf(role)?.label ?? role;
const sectionLatest = computed(() => section.value === 'cutter' ? latestCutter.value : section.value === 'classifier'
  ? latestClassifier.value : section.value === 'decider' ? latestDecider.value : section.value === 'brain' ? latest('brain') : null);
/** The headline of a version in one line: its main number and what it counts (the numbers of ModelItem). */
function headlineOf(version: ModelVersion): {value: string; text: string} | null {
  if (version.role === 'cutter') {
    const cut = version.test ?? version.validation ?? null;
    if (cut === null) return null;
    const speed = version.families === 'all' ? cut.by_family?.['speed'] ?? null : null;
    if (speed) return {value: pct(speed.recall), text: 'speed signs found, ' + thousands(speed.found) + ' of ' + thousands(speed.truth ?? 0)
      + ' · every sign ' + pct(cut.recall)};
    return {value: pct(cut.recall), text: 'signs found'};
  }
  if (version.role === 'classifier') {
    if (version.chain) return {value: pct(version.chain.recall), text: 'speed signs found and read right'};
    if (version.test) return {value: pct(version.test.accuracy), text: 'test signs read right'};
    return null;
  }
  if (version.role === 'brain') return version.test ? {value: pct(version.test.right), text: 'speed right on the test'} : null;
  return version.validation ? {value: pct(version.validation.other_road_recall), text: '“Another” found on validation'} : null;
}
const latestHeadline = computed(() => sectionLatest.value ? headlineOf(sectionLatest.value) : null);
/** How many versions the section's network has: Models lists them, to compare and to choose. */
const sectionVersions = computed(() => section.value === 'cutter' || section.value === 'classifier' || section.value === 'decider'
  || section.value === 'brain' ? roleOf(section.value)?.versions.length ?? 0 : 0);

/** The cutter: a SignNet at the car's band (status.cutter_options.band), and the signs it learns to find. */
const FAMILY_OPTIONS: {key: CutterFamilies; label: string}[] = [{key: 'all', label: 'Every sign'}, {key: 'speed', label: 'Speed signs only'}];
const pickedFamilies = ref<CutterFamilies | null>(null);
const cutterBand = computed(() => sizeText(props.status?.cutter_options?.band ?? '896x288'));
const cutterFamilies = computed({
  get: (): CutterFamilies => pickedFamilies.value ?? props.status?.cutter_options?.family ?? 'all',
  set: (value: CutterFamilies) => { pickedFamilies.value = value; },
});
/** Where the cutter's and applicability's next trainings start (network-start.ts), the same in their sections and in the
 * pipeline. */
const versionsOf = (role: ModelRole) => roleOf(role)?.versions;
const starts = {cutter: useNetworkStart('cutter', toRef(props, 'status'), versionsOf),
  decider: useNetworkStart('decider', toRef(props, 'status'), versionsOf)};
/** The cutter's Mapillary switch (cutter-choices.ts), the same in its section and in the pipeline. */
const cutterChoices = useCutterChoices(toRef(props, 'status'));
const CUTTER_ACTIONS = computed((): [ActionName, string][] => [
  ['train_cutter', 'A new SignNet, the cutter the comma runs, for ' + (cutterFamilies.value === 'all' ? 'every sign'
    : 'speed signs only') + starts.cutter.words + cutterChoices.words + ': it looks at the ' + cutterBand.value
    + ' band at the top of the frame. ' + cutterChoices.minutes + ' on the GPU.'],
]);
/** The cut of the Mapillary windows, in the cutter's card until they are cut. */
const WINDOWS_ACTIONS: [ActionName, string][] = [
  ['mapillary_windows', 'Cuts the Mapillary photos into windows the size of the car’s band, around their speed signs, for the '
    + 'switch above. The CPU for a while; the photos are read from the TrueNAS share.'],
];
/** The cutter's button: also not with a share that is not a percentage. */
const cutterActions = computed(() => {
  const actions = props.status?.actions ?? null;
  if (actions === null || !cutterChoices.invalid) return actions;
  return {...actions, train_cutter: {...actions.train_cutter, allowed: false, reason: cutterChoices.invalid}};
});
/** The speed signs of the windows (every family but the other signs). */
const windowSigns = computed(() => Object.values(cutterChoices.mapillaryState?.boxes_by_family ?? {})
  .reduce((total, count) => total + count, 0));
/** The classifier's training, in its card. */
/** The decider's training, in its card: experiment B, its three variants. */
const DECIDER_ACTIONS = computed((): [ActionName, string][] => [
  ['train_decider', 'Three versions of applicability (is this sign for our road?): the crop alone, with position and speed, with '
    + 'the scene' + starts.decider.words + '. About 15 minutes on the GPU.'],
]);
/** What the brain learns (brain-choices.ts), the same in its section and in the pipeline. */
const brainChoices = useBrainChoices();
/** The brain's training, in its card: three seeds, to choose one in Models. */
const BRAIN_ACTIONS = computed((): [ActionName, string][] => [
  ['train_brain', 'Three seeds of the brain' + brainChoices.words + ', one after the other, about 3 minutes each on the GPU, after '
    + 'reading the signs of the drives as the car does. Then you choose one in Models and Publish it.'],
]);
/** The classifier's choices (classifier-choices.ts), the same in its section and in the pipeline. */
const choices = useClassifierChoices(toRef(props, 'status'), versionsOf);
const CLASSIFIER_ACTIONS = computed((): [ActionName, string][] => [
  ['train_classifier', 'A new classifier, taught on the signs and on the cutter’s mistakes' + choices.words
    + '. Takes a few hours on the GPU.'],
]);
/** The classifier's button: also not with a share that is not a percentage. */
const classifierActions = computed(() => {
  const actions = props.status?.actions ?? null;
  if (actions === null || !choices.invalid) return actions;
  return {...actions, train_classifier: {...actions.train_classifier, allowed: false, reason: choices.invalid}};
});
const activities = computed(() => terminalActivities(props.status, ['train']));
const sectionActivities = computed(() => activitiesFor(section.value));
/** The curves of a training: while it runs (a placeholder before the first epoch), or of the last one if it has any. */
const charted = computed((): {progress: Progress; running: boolean} | null => {
  // A training started from a terminal: the server reads its progress from data/logs/train-<run>.log.
  const terminal = sectionActivities.value.find((activity) => activity.progress);
  if (terminal?.progress && !(job.value?.state === 'running')) return {progress: terminal.progress, running: true};
  const value = job.value;
  if (value === null || !CHARTED_ACTIONS.includes(value.action)) return null;
  const curves = value.progress.curves ?? {};
  const shown = value.state === 'running' || Object.values(curves).some((curve) => Boolean(curve?.points.length));
  return shown ? {progress: value.progress, running: value.state === 'running'} : null;
});
/** What every drive with its logs read gives the brain today (his speed, the blocks of Speed review as its corrections, the
 * next limit of Sign review), asked when its section opens and when a training of it ends (about 10 s on the server:
 * never with the status). */
const brainLabels = shallowRef<BrainLabels | null>(null);
watch([section, () => jobFor('brain')?.state], async ([value]) => {
  if (value !== 'brain') return;
  try {
    brainLabels.value = await api<BrainLabels>('/api/panel/brain-labels');
  } catch (problem) {
    brainLabels.value = {problem: messageOf(problem)};
  }
}, {immediate: true});
const numbers = computed(() => {
  const status = props.status;
  if (status === null) return null;
  const routes = status.routes;
  const dataset = status.dataset;
  return {
    reviewed: status.reviewed_routes,
    needed: status.min_routes,
    fresh: routes.reduce((total, row) => total + row.new_reviews, 0),
    // What the dataset file holds (status.dataset), else what the routes say they put there.
    objects: dataset ? dataset.objects : routes.reduce((total, row) => total + row.objects, 0),
    frames: dataset ? dataset.frames : null,
    inDataset: dataset ? dataset.routes : routes.filter((row) => row.objects > 0).length,
  };
});
</script>

<template>
  <SectionHead tab="train"/>
  <!-- Which network: first, under the title, as the segments of an iOS page. -->
  <Segmented v-model="section" class="train-sections" :options="SECTIONS" :marked="runningSections" label="Network"/>


  <!-- What the networks learn from, in one line; the reviews not imported yet lead to Data › Training dataset, where they are
       imported. -->
  <a class="card data-strip" href="/dataset" aria-label="What the networks learn from: open the dataset">
    <span class="data-texts"><span class="data-line"><strong>{{ numbers ? thousands(numbers.objects) : '–' }}</strong> objects on
      <strong>{{ numbers?.frames === null || !numbers ? '–' : thousands(numbers.frames) }}</strong> frames of
      {{ numbers ? plural(numbers.inDataset, 'drive') : '–' }}</span>
    <span v-if="importing" class="data-new">Importing your reviews…</span>
    <span v-else-if="numbers?.fresh" class="data-new"><strong>{{ plural(numbers.fresh, 'review') }}</strong> not imported</span>
    <span v-else-if="numbers" class="data-new muted">Dataset is up to date</span>
    <span v-if="numbers && numbers.reviewed < numbers.needed" class="data-new warn">Training needs {{ numbers.needed }} reviewed
      drives (now {{ numbers.reviewed }})</span></span>
    <span class="data-chev" aria-hidden="true">›</span>
  </a>

  <section class="card role-card train-section" :id="'train-' + section" :data-role="section" aria-live="polite">
    <p v-if="section === 'cutter'" class="role-what">SignNet, the same network the comma runs: it finds every box that could be a sign at the top of the frame. It must miss nothing: the classifier drops its false boxes.</p>
    <p v-else-if="section === 'classifier'" class="role-what">Says what each box of the cutter is: a speed value, end of limit, another sign and its kind, or not a sign.</p>
    <p v-else-if="section === 'decider'" class="role-what">Says whom each sign applies to: our road, another road, or abstain.</p>
    <p v-else-if="section === 'brain'" class="role-what">Says the speed to hold, four times a second, from the camera, the signs it reads and
      the car’s own motion, and whether it is in roadworks. It learns your own speed on every drive whose logs are read, with the
      blocks you set in Speed review as the corrections, the roadworks you mark there, and the next limit from the signs of Sign review.</p>
    <p v-else class="role-what">Our networks one after the other, each with the choices of its own section: the next one starts
      when one ends well. The classifier after a new cutter learns that cutter’s mistakes.</p>

    <!-- The slot of the work: the training running in place of its button, a failed one above it (it can start again): what runs and what to start come before the versions. -->
    <div class="train-slot">
      <JobProgress v-if="bigJob" :job="bigJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
      <template v-if="bigJob?.state === 'running' || sectionActivities.length">
        <TerminalActivity v-for="activity in sectionActivities" :key="activity.pid + activity.label" :activity="activity" :preparation="null"/>
      </template>
      <template v-else-if="section === 'cutter'">
        <div class="seg-row">
          <Segmented v-model="cutterFamilies" :options="FAMILY_OPTIONS" label="Signs the cutter finds"/>
        </div>
        <MapillaryChoices :choices="cutterChoices" :status="status" label="Frames the cutter learns from" id-prefix="cutter-mapillary"
                          note-id="train-cutter-mapillary-note"
                          @locate="status && emit('run', 'mapillary_locations', status.actions.mapillary_locations)"/>
        <NetworkStartPicker :start="starts.cutter" id="train-cutter-start"/>
        <ActionList :items="CUTTER_ACTIONS" :actions="cutterActions"
                    @run="(name, state) => emit('run', name, state, undefined, {families: cutterFamilies, ...starts.cutter.option,
                      ...cutterChoices.options})"/>
        <JobProgress v-if="cutterChoices.cutJob?.state === 'running'" :job="cutterChoices.cutJob" @details="emit('details', $event)"
                     @stop="emit('stop', $event)"/>
        <p v-else-if="cutterChoices.cutting" class="latest-none" id="train-cutter-cutting">Cutting the Mapillary windows, started from a
          terminal: the switch is ready when it ends.</p>
        <template v-else-if="cutterChoices.mapillaryState?.available">
          <p class="latest-none" id="train-cutter-mapillary-note">Mapillary: {{ thousands(cutterChoices.mapillaryState.windows) }} windows
            of {{ thousands(cutterChoices.mapillaryState.photos) }} street photos, with {{ thousands(windowSigns) }} speed signs, added to
            train only: every measure stays on your drives.<template v-if="cutterChoices.mapillary === 'on'"> The same share always
            takes the same photos; then at most as many speed signs as ours (Settings › Mapillary signs per one of ours).</template><template
            v-if="cutterChoices.mapillary === 'pretrain'"> The cutter first learns on them alone, from the start chosen above, then on
            your drives, starting from there.</template> Photos {{ cutterChoices.mapillaryState.licence }}, Mapillary Traffic Sign
            Dataset · <button type="button" class="link-button" id="cutter-windows-again"
            :disabled="!status?.actions.mapillary_windows?.allowed"
            @click="status && emit('run', 'mapillary_windows', status.actions.mapillary_windows)">Cut again</button></p>
        </template>
        <template v-else-if="cutterChoices.mapillaryState">
          <p class="latest-none" id="train-cutter-mapillary-note">Mapillary windows: not cut yet. With them the cutter can also
            learn from the speed signs of street photos taken around the world.</p>
          <ActionList :items="WINDOWS_ACTIONS" :actions="status?.actions ?? null" @run="(name, state) => emit('run', name, state)"/>
        </template>
        <p v-if="cutterChoices.cutJob?.state === 'failed'" class="latest-none warn-text" id="train-cutter-cut-failed">Cutting the
          windows failed · <button type="button" class="link-button" @click="cutterChoices.cutJob && emit('details', cutterChoices.cutJob)">Details</button></p>
        <p v-if="cutterChoices.mapillaryState?.available && cutterChoices.locateJob?.state === 'failed'" class="latest-none warn-text"
           id="train-cutter-locate-failed">Locating the photos failed · <button type="button" class="link-button"
           @click="cutterChoices.locateJob && emit('details', cutterChoices.locateJob)">Details</button></p>
      </template>
      <template v-else-if="section === 'classifier'">
        <ClassifierChoices :choices="choices" :status="status"
                           @locate="status && emit('run', 'mapillary_locations', status.actions.mapillary_locations)"/>
        <ActionList :items="CLASSIFIER_ACTIONS" :actions="classifierActions"
                    @run="(name, state) => emit('run', name, state, undefined, choices.options)"/>
        <p class="latest-none" id="train-classifier-cutter"><template v-if="cutterInUse">Cutter: the one in use,
          <strong>{{ cutterInUse.label }}</strong> · {{ cutterInUse.trained_at ? isoWhen(cutterInUse.trained_at) : cutterInUse.run }}.<template
          v-if="cutterInUseCar"> On the {{ cutterInUsePartition }}, as the car reads it: it finds <strong>{{ pct(cutterInUseCar.recall) }}</strong>
          of the signs, and <strong>{{ pct(cutterInUseCar.precision) }}</strong> of its boxes are signs<template
          v-if="cutterInUseCar.false_per_frame !== null"> ({{ cutterInUseCar.false_per_frame.toFixed(2) }} false a frame)</template>.<span
          v-if="cutterInUseStrict" class="strict-note" title="Counted when the box sits on the sign: its centre on it, half to twice its size. The strict count needs the boxes to overlap by IoU 0.5, unfair to far signs boxed a pixel or two tight.">
          Strict count (IoU 0.5): {{ pct(cutterInUseStrict.recall) }} / {{ pct(cutterInUseStrict.precision) }}.</span></template>
          Its mistakes are the “not a sign” the classifier learns, and the chain is measured with it. Another one: Models › Use this one.</template><template
          v-else-if="models">No cutter in use: choose one in Models first.</template></p>
        <p v-if="choices.mapillaryState?.available" class="latest-none" id="train-mapillary-note">Mapillary: {{ thousands(choices.mapillaryState.crops) }}
          street-photo signs of {{ plural(choices.mapillaryState.labels, 'kind') }}, added to train only: every measure stays on your drives.<template
          v-if="choices.mapillary === 'on'"> The same share always takes the same signs; then at most 300 of each kind.</template><template
          v-if="choices.mapillary === 'pretrain'"> The classifier first learns on them alone (the same share always takes the same
          signs), then on your signs and the cutter’s mistakes, starting from there.</template></p>
        <p v-else-if="choices.mapillaryState" class="latest-none" id="train-mapillary-note">Mapillary signs: not prepared yet
          (helpers/mapillary.sh).</p>
        <p v-if="choices.mapillaryState?.available && choices.locateJob?.state === 'failed'" class="latest-none warn-text" id="train-locate-failed">Locating the
          photos failed · <button type="button" class="link-button" @click="choices.locateJob && emit('details', choices.locateJob)">Details</button></p>
      </template>
      <template v-else-if="section === 'decider'">
        <NetworkStartPicker :start="starts.decider" id="train-decider-start"/>
        <ActionList :items="DECIDER_ACTIONS" :actions="status?.actions ?? null"
                    @run="(name, state) => emit('run', name, state, undefined, starts.decider.option)"/>
        <p class="latest-none" id="train-decider-note">The one in use (Models) sets OURS or another on the speed signs Our networks propose.</p>
      </template>
      <template v-else-if="section === 'brain'">
        <BrainChoices :choices="brainChoices" id="train-brain"/>
        <ActionList :items="BRAIN_ACTIONS" :actions="status?.actions ?? null"
                    @run="(name, state) => emit('run', name, state, undefined, brainChoices.option)"/>
        <p v-if="brainLabels?.problem" class="latest-none warn-text" id="train-brain-labels">{{ brainLabels.problem }}</p>
        <p v-else-if="brainLabels?.partitions" class="latest-none" id="train-brain-labels">Your speed on
          {{ plural(brainLabels.drives?.length ?? 0, 'drive') }} with their logs read, the blocks of Speed review as its corrections:
          <strong>{{ thousands(brainLabels.partitions.train.speed) }}</strong> seconds to learn from,
          <strong>{{ thousands(brainLabels.partitions.validation.speed) }}</strong> to watch it on,
          <strong>{{ thousands(brainLabels.partitions.test.speed) }}</strong> for the test. Sign review answers the next limit on
          <strong>{{ thousands(brainLabels.partitions.train.next) }}</strong> of the seconds to learn from.<template
          v-if="brainLabels.to_read"> First it reads the signs of {{ plural(brainLabels.to_read, 'drive') }} as the car does today.</template></p>
        <p v-else class="latest-none" id="train-brain-labels">Loading…</p>
      </template>
      <PipelineSection v-else v-model:families="cutterFamilies" :status="status" :choices="choices" :cutter-choices="cutterChoices" :starts="starts" :brain-choices="brainChoices" :fresh="numbers?.fresh ?? 0"
                       :family-options="FAMILY_OPTIONS" @pipeline="(steps, text) => emit('pipeline', steps, text)"
                       @stop-pipeline="emit('stopPipeline')" @stop="emit('stop', $event)" @details="emit('details', $event)"
                       @run="(name, state, start) => emit('run', name, state, start)"/>
      <p v-if="job && !bigJob && !sectionActivities.length && section !== 'pipeline'" class="last-line">Last: {{ job.label }}, {{ startedWhen(job) }} ·
        {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
        <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>
    </div>

    <!-- The settings of the next training of this network, also while one runs (it keeps its own). -->
    <TrainSettingsSection v-if="settingsAction" :training="settingsAction"
                          :running="job?.state === 'running' || sectionActivities.length > 0"/>

    <!-- The latest version in one row, a link to Models: its versions there, opened on this one, to compare and to
         choose (the choice is made only there). -->
    <a v-if="sectionLatest" class="latest-row latest-link" :href="'/models#role-' + sectionLatest.role + ':' + encodeURIComponent(sectionLatest.id)" id="latest-version">
      <span class="latest-what"><span class="latest-kicker">{{ section === 'decider' ? 'Latest prototype' : 'Latest' }} ·
        {{ sectionLatest.trained_at ? isoWhen(sectionLatest.trained_at) : sectionLatest.run }}<span
        v-if="sectionLatest.in_use" class="latest-use">in use</span><span v-else-if="!sectionLatest.finished" class="latest-wait">not finished</span></span>
        <span class="latest-name">{{ sectionLatest.label }}</span>
        <span class="latest-text">{{ latestHeadline ? latestHeadline.text + ' · ' : '' }}{{ plural(sectionVersions, 'version', 'versions') }} in Models</span></span>
      <strong v-if="latestHeadline" class="latest-value">{{ latestHeadline.value }}</strong>
      <span class="latest-chev" aria-hidden="true">›</span>
    </a>
    <p v-else-if="section === 'cutter'" class="latest-none">{{ models === null ? 'Loading…'
      : 'None trained yet: train a SignNet, then choose it in Models.' }}</p>
    <p v-else-if="section === 'classifier'" class="latest-none">{{ models === null ? 'Loading…'
      : 'None trained yet: the reader of a whole pipeline plays this role.' }}</p>
    <p v-else-if="section === 'brain'" class="latest-none">{{ models === null ? 'Loading…' : 'None trained yet.' }}</p>

    <!-- The curves stay open once the training ends: closing them is a tap. -->
    <TrainingCharts v-if="charted?.running" :progress="charted.progress" :running="true"/>
    <details v-else-if="charted" class="latest-row curves-row" open>
      <summary><span class="latest-what"><span class="latest-name">Curves of the last training</span>
        <span class="latest-text">{{ section === 'decider' ? 'the chosen measure' : section === 'brain' ? 'km/h off' : 'recall' }} and loss, epoch by epoch</span></span><span class="latest-chev" aria-hidden="true">›</span></summary>
      <TrainingCharts :progress="charted.progress" :running="false"/>
    </details>

  </section>
</template>
