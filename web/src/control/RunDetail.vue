<script setup lang="ts">
// One run in detail: what it is (when, where its weights started from, how far it got, the error that stopped it),
// its headline numbers on the test, then the reader label by label, the detector by sign size and route, the dataset
// and split it used, its training curves (TrainingCharts, as on the Train tab) and experiment B when it has one. A run
// of the classifier with “not a sign” (scripts/run_classifier.sh): the classifier alone, the chain end to end against
// the old reader, the false boxes it learnt from, the comparison in plain text. A run of the cutter: SignNet
// (scripts/run_signnet.sh) in its 5 best boxes of each frame, as the car reads it (SignNetSection); an old Detector run
// (scripts/run_detector_sizes.sh), each size as the Models tab judges a cutter. Watch plays the frames of a route with
// the cutter's boxes over them, for every report of the run that has frames (WatchSection). A UserNet run
// (scripts/run_usernet.sh): the road users found at its score, its false boxes a frame, what it learnt from.
import {computed} from 'vue';
import {thousands} from '../common/format';
import type {CurveNet, JobProgress, RunDetail, RunSummary} from '../common/panel-types';
import ChainSection from './ChainSection.vue';
import ClassifierSection from './ClassifierSection.vue';
import CutterSection from './CutterSection.vue';
import DetectorSection from './DetectorSection.vue';
import {RUN_STATE, modeText, notEvaluatedText, pct, plain, runWhen, shortSha, usernetStageText} from './evaluate';
import ExperimentSection from './ExperimentSection.vue';
import {perFrame} from './models';
import NegativesSection from './NegativesSection.vue';
import SignNetSection from './SignNetSection.vue';
import ReaderSection from './ReaderSection.vue';
import SplitSection from './SplitSection.vue';
import TrainingCharts from './TrainingCharts.vue';
import WatchSection from './WatchSection.vue';

const props = defineProps<{summary: RunSummary; detail: RunDetail | null; loading: boolean; error: string; noWatch?: boolean}>();
/** The summary is refreshed with the list every 10 s; the detail is asked again only when the run changed. */
const run = computed((): RunSummary => props.summary);
const detail = computed(() => props.detail?.name === props.summary.name ? props.detail : null);

/** The networks whose curves the run has: the pipeline's two (and the classifier when it was trained into the same
 * run), or the classifier alone (after its pretraining on Mapillary, in two stages). */
const nets = computed((): CurveNet[] => {
  const value = detail.value;
  if (value === null || value.kind === 'experiment') return [];
  if (value.kind === 'classifier') return value.curves.pretrain?.points.length ? ['pretrain', 'classifier'] : ['classifier'];
  if (value.kind === 'cutter') return value.signnet_run || value.curves.signnet?.points.length ? ['signnet'] : ['detector'];
  if (value.kind === 'usernet') return ['usernet'];
  return value.curves.classifier?.points.length ? ['detector', 'reader', 'classifier'] : ['detector', 'reader'];
});
/** The curves as the Train tab's charts take them (a job's progress without steps). */
const curves = computed((): JobProgress | null => {
  const value = detail.value;
  if (value === null || nets.value.length === 0) return null;
  const points = nets.value.reduce((sum, net) => sum + (value.curves[net]?.points.length ?? 0), 0);
  if (!points && run.value.state !== 'running') return null;
  return {steps: [], percent: null, detail: '', last_line: '', curves: value.curves};
});
const training = computed(() => detail.value?.training ?? null);
/** What scripts/run_classifier.sh wrote here, and its training. */
const classifierRun = computed(() => detail.value?.classifier_run ?? null);
const classifierTraining = computed(() => classifierRun.value?.training ?? null);
const hasClassifierEval = computed(() => Boolean(classifierRun.value?.evaluations.test || classifierRun.value?.evaluations.validation));
const hasChain = computed(() => Boolean(classifierRun.value?.chains.test || classifierRun.value?.chains.validation));
/** An old Detector's cutter run: its (last) size in the tiles (a SignNet run's numbers are on its SignNet card). */
const cutterLine = computed(() => {
  const line = run.value.cutters?.[run.value.cutters.length - 1] ?? null;
  return line?.kind === 'signnet' ? null : line;
});
const where = computed(() => {
  const value = run.value;
  if (value.state === 'running') return 'Running: ' + (value.progress || value.stage_label) + '.';
  if (value.state === 'finished') {
    return value.kind === 'experiment' ? 'Finished: the three variants of experiment B are measured.'
      : value.kind === 'classifier' ? 'Finished: the classifier is trained, measured alone and in the chain with the cutter.'
        : value.kind === 'cutter' ? 'Finished: the cutter is trained and measured on test and validation.'
        : value.kind === 'usernet' ? 'Finished: UserNet is trained and measured on ' + (detail.value?.usernet_run?.test
          ? 'validation and test.' : 'validation (no test frames yet).')
        : 'Finished: trained, evaluated, exported and packaged.';
  }
  return 'Stopped while ' + value.stage_label + '.';
});
</script>

<template>
  <section class="card run-detail" id="run-detail" :data-run="run.name" aria-labelledby="run-title">
    <div class="rd-head">
      <div class="rd-title">
        <span class="jp-kicker">Run</span>
        <h2 id="run-title">{{ run.name }}</h2>
      </div>
      <span class="pill" :class="RUN_STATE[run.state][1]">{{ RUN_STATE[run.state][0] }}</span>
    </div>
    <dl class="facts">
      <div><dt>Started</dt><dd>{{ runWhen(run) || '–' }}</dd></div>
      <div><dt>Started from</dt><dd>{{ run.start.label }}<span v-if="modeText(run)" class="muted"> · {{ modeText(run) }}</span></dd></div>
      <div v-if="training?.detector || training?.reader"><dt>Best validation</dt><dd>
        <template v-if="training?.detector">detector loss {{ plain(training.detector.best_validation_loss) }}{{
          training.detector.best_epoch ? ' (epoch ' + training.detector.best_epoch + ')' : '' }}</template><template
          v-if="training?.detector && training?.reader"> · </template><template v-if="training?.reader">reader {{
          pct(training.reader.best_validation_accuracy) }}{{ training.reader.best_epoch ? ' (epoch ' + training.reader.best_epoch + ')' : '' }}</template></dd></div>
      <div v-if="classifierTraining"><dt>Classifier, best validation</dt><dd>{{
        pct(classifierTraining.best_validation_balanced_accuracy ?? classifierTraining.best_validation_accuracy) }}{{
          classifierTraining.best_epoch ? ' (epoch ' + classifierTraining.best_epoch + (classifierTraining.epochs_run ? ' of ' + classifierTraining.epochs_run : '') + ')' : ''
        }}<span v-if="classifierTraining.best_validation_sign_accuracy !== null || classifierTraining.best_validation_rejection !== null" class="muted">
          · signs {{ pct(classifierTraining.best_validation_sign_accuracy) }}, “not a sign” {{ pct(classifierTraining.best_validation_rejection) }}</span></dd></div>
      <div v-if="classifierTraining && (classifierTraining.classes !== null || classifierTraining.not_a_sign_weight !== null)"><dt>Classifier</dt><dd>{{
        classifierTraining.classes ?? '–' }} classes<template v-if="classifierTraining.not_a_sign_weight !== null"> · “not a sign” weighs {{
          plain(classifierTraining.not_a_sign_weight) }} in the loss</template><template v-if="classifierTraining.augmentation !== null"> · {{
          classifierTraining.augmentation ? 'augmented' : 'no augmentation' }}</template><template v-if="classifierTraining.stopped_early"> · stopped early</template></dd></div>
      <div v-if="training?.detector?.gpu || training?.reader?.gpu || classifierTraining?.gpu"><dt>GPU</dt><dd>{{
        training?.detector?.gpu ?? training?.reader?.gpu ?? classifierTraining?.gpu }}</dd></div>
      <div v-if="run.split_sha256"><dt>Split</dt><dd><code :title="run.split_sha256">{{ shortSha(run.split_sha256) }}</code></dd></div>
    </dl>
    <p class="rd-where" :class="run.state">{{ where }}</p>
    <p v-if="run.state === 'incomplete' && run.error" class="rd-error"><code>{{ run.error }}</code></p>
    <p v-if="run.kind === 'training' && run.experiment_running" class="rd-where running">Also running here: {{ run.experiment_progress }}.</p>
    <p v-if="run.start.source" class="note">Weights: <code>{{ run.start.source }}</code></p>
    <p v-for="problem in run.problems" :key="problem" class="note ds-note">{{ problem }}.</p>

    <div v-if="run.evaluated && run.kind === 'classifier'" class="tiles" id="run-tiles">
      <div class="tile">
        <span class="tile-label">Classifier, signs read right</span>
        <span class="tile-value">{{ pct(run.classifier?.accuracy) }}</span>
        <span class="tile-note">{{ run.classifier?.samples ? thousands(run.classifier.samples) + ' test sign crops' : 'no evaluation of the classifier' }}<template
          v-if="run.classifier?.human_accuracy != null"> · human-reviewed {{ pct(run.classifier.human_accuracy) }}</template></span>
      </div>
      <div class="tile">
        <span class="tile-label">“Not a sign” rejected</span>
        <span class="tile-value">{{ pct(run.classifier?.rejection_rate) }}</span>
        <span class="tile-note">{{ run.classifier?.not_a_sign_crops ? 'of ' + thousands(run.classifier.not_a_sign_crops) + ' test crops that are no sign' : '' }}</span>
      </div>
      <div class="tile">
        <span class="tile-label">Chain, end to end</span>
        <span class="tile-value">{{ pct(run.chain?.recall) }}</span>
        <span class="tile-note">{{ run.chain ? thousands(run.chain.correct ?? 0) + ' of ' + thousands(run.chain.truth ?? 0) + ' speed signs found and read right, score '
          + run.chain.score_threshold : 'no evaluation of the chain' }}</span>
      </div>
      <div class="tile">
        <span class="tile-label">False boxes left per frame</span>
        <span class="tile-value">{{ perFrame(run.chain?.surviving_false_per_frame) }}</span>
        <span class="tile-note">{{ run.chain ? 'of ' + perFrame(run.chain.false_per_frame) + ' from the cutter' : '' }}<template
          v-if="run.chain_old_reader"> · old reader {{ pct(run.chain_old_reader.recall) }} end to end</template></span>
      </div>
    </div>
    <div v-else-if="run.evaluated && run.kind === 'cutter' && cutterLine" class="tiles" id="run-tiles">
      <div class="tile">
        <span class="tile-label">Cutter, signs found</span>
        <span class="tile-value">{{ pct(cutterLine.recall) }}</span>
        <span class="tile-note">{{ thousands(cutterLine.found ?? 0) }} of {{ thousands(cutterLine.truth_boxes ?? 0) }} at score {{ cutterLine.score_threshold ?? '–' }},
          {{ (cutterLine.size ?? '–').replace('x', ' × ') }}</span>
      </div>
      <div class="tile">
        <span class="tile-label">Speed signs alone</span>
        <span class="tile-value">{{ cutterLine.families === 'all' ? pct(cutterLine.speed_recall) : pct(cutterLine.recall) }}</span>
        <span class="tile-note">{{ cutterLine.families === 'all' ? 'it also finds the other signs' : 'it finds the speed signs only' }}</span>
      </div>
      <div class="tile">
        <span class="tile-label">Boxes per frame</span>
        <span class="tile-value">{{ perFrame(cutterLine.boxes_per_frame) }}</span>
        <span class="tile-note">{{ perFrame(cutterLine.false_per_frame) }} false per frame</span>
      </div>
      <div class="tile">
        <span class="tile-label">Time per frame</span>
        <span class="tile-value">{{ cutterLine.ms_per_frame === null ? '–' : perFrame(cutterLine.ms_per_frame) + ' ms' }}</span>
        <span class="tile-note">the model alone</span>
      </div>
    </div>
    <div v-else-if="run.evaluated && run.kind === 'usernet' && run.usernet" class="tiles" id="run-tiles">
      <div class="tile">
        <span class="tile-label">Road users found</span>
        <span class="tile-value">{{ pct(run.usernet.recall) }}</span>
        <span class="tile-note">at score {{ run.usernet.score_threshold }} on {{ run.usernet.partition }}{{
          run.usernet.frames ? ', ' + thousands(run.usernet.frames) + ' frames' : '' }}</span>
      </div>
      <div class="tile">
        <span class="tile-label">Its boxes that are road users</span>
        <span class="tile-value">{{ pct(run.usernet.precision) }}</span>
        <span class="tile-note">best epoch {{ run.usernet.best_epoch ?? '–' }}</span>
      </div>
      <div class="tile">
        <span class="tile-label">False boxes per frame</span>
        <span class="tile-value">{{ run.usernet.false_per_frame === null ? '–' : run.usernet.false_per_frame.toFixed(2) }}</span>
        <span class="tile-note">validated on {{ run.usernet.validation_source === 'bdd' ? 'BDD100K pictures' : 'your ticked frames' }}</span>
      </div>
      <div class="tile">
        <span class="tile-label">Learnt from</span>
        <span class="tile-value">{{ run.usernet.stage === 'bdd' ? 'BDD100K' : run.usernet.stage === 'drives' ? 'Drives' : '–' }}</span>
        <span class="tile-note">{{ usernetStageText(run.usernet.stage, run.usernet.uses_bdd) }}{{
          run.usernet.uses_bdd ? ': BDD100K’s licence goes with the weights' : '' }}</span>
      </div>
    </div>
    <div v-else-if="run.evaluated && run.kind !== 'cutter' && run.kind !== 'usernet'" class="tiles" id="run-tiles">
      <div class="tile">
        <span class="tile-label">Reader accuracy</span>
        <span class="tile-value">{{ pct(run.reader?.accuracy) }}</span>
        <span class="tile-note">{{ run.reader ? thousands(run.reader.samples) + ' test crops' : 'no reader evaluation' }}</span>
      </div>
      <div class="tile">
        <span class="tile-label">Reader, human-reviewed</span>
        <span class="tile-value">{{ pct(run.reader?.human_accuracy) }}</span>
        <span class="tile-note">{{ run.reader?.human_samples ? thousands(run.reader.human_samples) + ' crops a person confirmed' : 'none confirmed by a person' }}</span>
      </div>
      <div class="tile">
        <span class="tile-label">Detector recall</span>
        <span class="tile-value">{{ pct(run.detector?.recall) }}</span>
        <span class="tile-note">{{ run.detector ? thousands(run.detector.found) + ' of ' + thousands(run.detector.truth_boxes) + ' signs found' : 'no detector evaluation' }}</span>
      </div>
      <div class="tile">
        <span class="tile-label">Detector precision</span>
        <span class="tile-value">{{ pct(run.detector?.precision) }}</span>
        <span class="tile-note">{{ run.detector ? thousands(run.detector.false_boxes) + ' false boxes' : '' }}</span>
      </div>
    </div>
    <p v-else-if="run.kind !== 'experiment' && !(run.evaluated && run.kind === 'cutter')" class="net-wait" id="run-not-evaluated">{{ notEvaluatedText(run) }}</p>
  </section>

  <p v-if="error" class="banner">The details of {{ run.name }} cannot be read: {{ error }}</p>
  <p v-else-if="loading && detail === null" class="card empty">Loading {{ run.name }}…</p>
  <template v-if="detail">
    <ReaderSection v-if="detail.reader_eval" :reader="detail.reader_eval"/>
    <DetectorSection v-if="detail.detector_eval" :detector="detail.detector_eval"/>
    <SignNetSection v-if="detail.signnet_run" :signnet="detail.signnet_run"/>
    <CutterSection v-if="detail.cutter_run" :sizes="detail.cutter_run.sizes" :comparison="detail.cutter_run.comparison"/>
    <ClassifierSection v-if="classifierRun && hasClassifierEval" :evaluations="classifierRun.evaluations"/>
    <ChainSection v-if="classifierRun && hasChain" :chains="classifierRun.chains"/>
    <WatchSection v-if="!noWatch && detail.watch?.length" :run="run.name" :models="detail.watch" :version="run.version"/>
    <TrainingCharts v-if="curves" :progress="curves" :running="run.state === 'running'" :nets="nets" title="Training curves"/>
    <NegativesSection v-if="classifierRun" :negatives="classifierRun.negatives"
                      :by-source="detail.dataset?.classifier?.negatives ?? classifierTraining?.negatives?.by_split ?? null"/>
    <section v-if="classifierRun?.comparison" class="card eval-card" id="classifier-comparison" aria-labelledby="comparison-title">
      <div class="ds-head"><h2 id="comparison-title">Against the old reader<span class="eval-what">comparison.txt, as the script wrote it</span></h2></div>
      <pre class="term">{{ classifierRun.comparison }}</pre>
    </section>
    <SplitSection v-if="detail.dataset" :dataset="detail.dataset"/>
    <ExperimentSection v-if="detail.experiment" :experiment="detail.experiment" :split="detail.split_sha256"/>
  </template>
</template>
