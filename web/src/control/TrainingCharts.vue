<script setup lang="ts">
// How a training is going, drawn: a few figures (the epoch now, the best validation result and when, time
// per epoch and about how long the network still needs), the detector's loss, the reader's accuracy and
// loss, train against validation; for the classifier with “not a sign”, the sign crops read right and the “not a
// sign” crops rejected on validation, their mean that chooses its best epoch, and its loss; before it, in two stages,
// its pretraining on the Mapillary signs alone (accuracy on our validation signs, and loss); for SignNet (the cutter the
// comma runs), the validation signs found in its 5 best boxes of each frame at any score (the car also drops the boxes
// below 0.3: those numbers come once it is measured, on the run's SignNet card), its false boxes per frame among those
// 5, and its train loss (before it, in two stages, its stage on the Mapillary windows alone: the speed signs of our
// validation it finds, which choose that stage's best epoch); for applicability (experiment B), its variants in the same two charts, one
// line each: the mean of its reading accuracy and the F1 of the signs of another road (which chooses its best epoch; before
// #72 the ownership macro F1 in its place), and its validation loss; for the brain, its three seeds in the same way: how far its speed is from the labels on validation, the frames it says
// right, and its train loss. Data: progress.curves of the job, one point per finished epoch.
import {computed} from 'vue';
import type {BrainNet, CurveNet, DeciderNet, EpochPoint, JobProgress, TrainingCurve} from '../common/panel-types';
import LineChart, {type ChartSeries} from './charts/LineChart.vue';
import {elapsedText, roughText} from './jobs';
import {ROAD_TYPE_LABELS} from './speed-review';

// title: the heading of the card (the Evaluate tab shows the curves of a past run); nets: the networks to show, else
// those of the job's steps, else those with curves, else the detector and the reader.
const props = defineProps<{progress: JobProgress; running: boolean; title?: string; nets?: CurveNet[]}>();
type Net = CurveNet;
/** Applicability's variants (experiment B), in the order they train. */
const DECIDER_NETS: DeciderNet[] = ['experiment_b_crop', 'experiment_b_crop_geom', 'experiment_b_full'];
/** The brain's seeds (scripts/run_brain.sh). */
const BRAIN_NETS: BrainNet[] = ['brain_s0', 'brain_s1', 'brain_s2'];
const ALL_NETS: Net[] = ['signnet_pretrain', 'signnet', 'detector', 'reader', 'pretrain', 'classifier', ...DECIDER_NETS, ...BRAIN_NETS];
const NET_NAME: Record<Net, string> = {signnet_pretrain: 'SignNet on Mapillary', signnet: 'SignNet', detector: 'Detector', reader: 'Reader', pretrain: 'Pretraining on Mapillary',
  classifier: 'Classifier', experiment_b_crop: 'The crop alone',
  experiment_b_crop_geom: 'Crop, position and speed', experiment_b_full: 'Crop, position and the scene',
  brain_s0: 'Seed 0', brain_s1: 'Seed 1', brain_s2: 'Seed 2'};
/** A network in a sentence ("the classifier ends in ..."). */
const NET_WORD: Record<Net, string> = {signnet_pretrain: 'Mapillary stage', signnet: 'SignNet', detector: 'detector', reader: 'reader', pretrain: 'pretraining',
  classifier: 'classifier', experiment_b_crop: 'variant with the crop alone', experiment_b_crop_geom: 'variant with crop, position and speed',
  experiment_b_full: 'variant with crop, position and the scene', brain_s0: 'seed 0', brain_s1: 'seed 1', brain_s2: 'seed 2'};
const isSignnet = (net: Net): boolean => net === 'signnet' || net === 'signnet_pretrain';
const isDecider = (net: Net): net is DeciderNet => (DECIDER_NETS as Net[]).includes(net);
const isBrain = (net: Net): net is BrainNet => (BRAIN_NETS as Net[]).includes(net);
const TRAIN = 'var(--c-train)';
const VAL = 'var(--c-val)';
const REJECT = 'var(--split-3)';
const BALANCED = 'var(--split-4)';
/** Applicability's variants and the brain's seeds share their charts: slots 1 to 3 of the categorical palette, in their order. */
const VARIANT_COLOR: Record<DeciderNet | BrainNet, string> = {experiment_b_crop: 'var(--c-train)', experiment_b_crop_geom: 'var(--c-val)',
  experiment_b_full: 'var(--split-3)', brain_s0: 'var(--c-train)', brain_s1: 'var(--c-val)', brain_s2: 'var(--split-3)'};

const running = computed(() => props.running);
const curves = computed(() => props.progress.curves ?? {});
const detector = computed(() => curves.value.detector ?? null);
const reader = computed(() => curves.value.reader ?? null);
const classifier = computed(() => curves.value.classifier ?? null);
const pretrain = computed(() => curves.value.pretrain ?? null);
const signnet = computed(() => curves.value.signnet ?? null);
const signnetPretrain = computed(() => curves.value.signnet_pretrain ?? null);
const hasPoints = (curve: TrainingCurve | null | undefined): curve is TrainingCurve => curve !== null && curve !== undefined
  && curve.points.length > 0;
/** The networks shown, in the order they train. */
const nets = computed((): Net[] => {
  if (props.nets?.length) return props.nets;
  const stepKeys = props.progress.steps.map((step) => step.key);
  // The pretraining is a step of its own; in the logs before 30 Oct 2026 it was inside the classifier's: shown once it has a curve.
  const fromSteps = ALL_NETS.filter((net) => stepKeys.includes(net)
    || (net === 'pretrain' && stepKeys.includes('classifier') && hasPoints(curves.value.pretrain)));
  if (fromSteps.length) return fromSteps;
  const withCurves = ALL_NETS.filter((net) => hasPoints(curves.value[net]));
  return withCurves.length ? withCurves : ['detector', 'reader'];
});
const shows = (net: Net): boolean => nets.value.includes(net);
const hasAny = computed(() => nets.value.some((net) => hasPoints(curves.value[net])));
/** The x axis of a curve: its planned epochs, or the epoch it stopped at when it stopped early. */
const span = (curve: TrainingCurve): number => curve.stopped_early ?? curve.epochs;

/** The values of one metric by epoch (index = epoch - 1), null where missing. */
function column(curve: TrainingCurve, pick: (point: EpochPoint) => number | undefined, scale = 1): (number | null)[] {
  const last = Math.max(0, ...curve.points.map((point) => point.epoch));
  const values: (number | null)[] = Array.from({length: last}, () => null);
  for (const point of curve.points) {
    const value = pick(point);
    if (value !== undefined && Number.isFinite(value) && point.epoch >= 1) values[point.epoch - 1] = value * scale;
  }
  return values;
}
function best(curve: TrainingCurve, pick: (point: EpochPoint) => number | undefined, highest: boolean): EpochPoint & {value: number} | null {
  let found: (EpochPoint & {value: number}) | null = null;
  for (const point of curve.points) {
    const value = pick(point);
    if (value === undefined || !Number.isFinite(value)) continue;
    if (found === null || (highest ? value > found.value : value < found.value)) found = {...point, value};
  }
  return found;
}
const loss = (value: number): string => value >= 10 ? value.toFixed(1) : value >= 1 ? value.toFixed(2) : value.toFixed(3);
const lossAxis = (value: number): string => value === 0 ? '0' : Number(value.toPrecision(3)).toString();
const percent = (value: number): string => (Math.round(value * 10) / 10).toString() + '%';
const percentAxis = (value: number): string => Math.round(value) + '%';
const kph = (value: number): string => value.toFixed(1) + ' km/h';
const kphAxis = (value: number): string => Number(value.toFixed(1)).toString();
const cell = (value: number | undefined, text: (value: number) => string): string => value === undefined ? '–' : text(value);

function lossSeries(curve: TrainingCurve): ChartSeries[] {
  return [
    {key: 'train', label: 'train', color: TRAIN, values: column(curve, (point) => point.train_loss)},
    {key: 'val', label: 'validation', color: VAL, values: column(curve, (point) => point.val_loss)},
  ];
}
const detectorBest = computed(() => shows('detector') && hasPoints(detector.value) ? best(detector.value, (point) => point.val_loss, false) : null);
/** The cutter's best epoch: its validation recall at score 0.1, when the log has it. */
const detectorRecall = computed(() => hasPoints(detector.value) && detector.value.points.some((point) => point.val_recall !== undefined));
const detectorBestRecall = computed(() => shows('detector') && detectorRecall.value && hasPoints(detector.value)
  ? best(detector.value, (point) => point.val_recall, true) : null);
/** SignNet's best epoch: the most validation signs found in its 5 best boxes of each frame, at any score (the weights
 * it keeps). */
const signnetBest = computed(() => shows('signnet') && hasPoints(signnet.value) ? best(signnet.value, (point) => point.val_recall, true) : null);
const signnetLowest = computed(() => hasPoints(signnet.value) ? best(signnet.value, (point) => point.train_loss, false) : null);
/** Its first stage on the Mapillary windows alone (Mapillary, then ours): the speed signs of our validation it finds in
 * its 5 best boxes choose that stage's best epoch, the weights the second one starts from. */
const pretrainSpeed = computed(() => hasPoints(signnetPretrain.value)
  && signnetPretrain.value.points.some((point) => point.val_speed_recall !== undefined));
const signnetPretrainBest = computed(() => shows('signnet_pretrain') && hasPoints(signnetPretrain.value)
  ? best(signnetPretrain.value, (point) => pretrainSpeed.value ? point.val_speed_recall : point.val_recall, true) : null);
const signnetPretrainLowest = computed(() => hasPoints(signnetPretrain.value)
  ? best(signnetPretrain.value, (point) => point.train_loss, false) : null);
const signnetFalse = computed(() => hasPoints(signnet.value) && signnet.value.points.some((point) => point.val_false_per_frame !== undefined));
const readerBestLoss = computed(() => hasPoints(reader.value) ? best(reader.value, (point) => point.val_loss, false) : null);
const readerBestAccuracy = computed(() => shows('reader') && hasPoints(reader.value)
  ? best(reader.value, (point) => point.val_accuracy, true) : null);

/** The classifier with “not a sign” writes the sign crops and the rejection apart; an older log only the accuracy. */
const withRejection = computed(() => hasPoints(classifier.value)
  && classifier.value.points.some((point) => point.val_sign_accuracy !== undefined || point.val_rejection !== undefined));
const classifierBestLoss = computed(() => hasPoints(classifier.value) ? best(classifier.value, (point) => point.val_loss, false) : null);
/** Its best epoch: by the balanced accuracy (the mean of signs and rejection), else by the accuracy. */
const classifierBest = computed(() => {
  if (!shows('classifier') || !hasPoints(classifier.value)) return null;
  return withRejection.value ? best(classifier.value, (point) => point.val_balanced_accuracy, true)
    : best(classifier.value, (point) => point.val_accuracy, true);
});
const classifierSeries = computed((): ChartSeries[] => {
  const curve = classifier.value;
  if (!hasPoints(curve)) return [];
  if (!withRejection.value) {
    return [{key: 'train', label: 'train', color: TRAIN, values: column(curve, (point) => point.train_accuracy, 100)},
      {key: 'val', label: 'validation', color: VAL, values: column(curve, (point) => point.val_accuracy, 100)}];
  }
  return [
    {key: 'signs', label: 'signs read right', color: VAL, values: column(curve, (point) => point.val_sign_accuracy, 100)},
    {key: 'reject', label: '“not a sign” rejected', color: REJECT, values: column(curve, (point) => point.val_rejection, 100)},
    {key: 'val', label: 'balanced', color: BALANCED, values: column(curve, (point) => point.val_balanced_accuracy, 100)},
  ];
});

/** The pretraining on Mapillary alone: its accuracy on our validation signs chooses its best epoch. */
const pretrainBest = computed(() => shows('pretrain') && hasPoints(pretrain.value)
  ? best(pretrain.value, (point) => point.val_accuracy, true) : null);
const pretrainBestLoss = computed(() => hasPoints(pretrain.value) ? best(pretrain.value, (point) => point.val_loss, false) : null);

type Best = (EpochPoint & {value: number}) | null;
interface NetCard {net: Net; curve: TrainingCurve | null; best: Best; lowest: Best; validation: ChartSeries[]; scored: boolean}
/** Applicability's variants: the mean of the signs read right and the other-road F1 chooses the best epoch (#72). */
const deciderCards = computed((): NetCard[] => nets.value.filter(isDecider).map((net) => {
  const curve = curves.value[net];
  if (!hasPoints(curve)) return {net, curve: null, best: null, lowest: null, validation: [], scored: false};
  return {net, curve, scored: true, best: best(curve, (point) => point.val_selection_score, true),
    lowest: best(curve, (point) => point.val_loss, false), validation: []};
}));
/** The brain's seeds: each keeps its last epoch (brain_train), so that epoch is what its tile says. */
const brainCards = computed((): NetCard[] => nets.value.filter(isBrain).map((net) => {
  const curve = curves.value[net];
  if (!hasPoints(curve)) return {net, curve: null, best: null, lowest: null, validation: [], scored: false};
  const measured = curve.points.filter((point) => point.val_error_kph !== undefined);
  const last = measured.reduce<EpochPoint | null>((found, point) => found === null || point.epoch > found.epoch ? point : found, null);
  return {net, curve, scored: true, best: last === null ? null : {...last, value: last.val_error_kph as number},
    lowest: best(curve, (point) => point.train_loss, false), validation: []};
}));
/** A tile per network with a best epoch. */
const bestCards = computed(() => deciderCards.value.flatMap((card) => card.best ? [{net: card.net, best: card.best}] : []));

/** The network being trained now (from the steps; the pretraining until the classifier's first epoch), or the last
 * one with points. */
const current = computed((): Net | null => {
  const step = props.progress.steps.find((item) => item.state === 'current')?.key;
  if (running.value && step === 'classifier' && shows('pretrain') && !hasPoints(classifier.value)) return 'pretrain';
  if (running.value && nets.value.some((net) => net === step)) return step as Net;
  return [...nets.value].reverse().find((net) => hasPoints(curves.value[net])) ?? null;
});
const currentCurve = computed(() => current.value === null ? null : curves.value[current.value] ?? null);
const lastEpoch = computed(() => Math.max(0, ...(currentCurve.value?.points ?? []).map((point) => point.epoch)));
/** Mean of the last three epochs (the first is often slower). */
const perEpoch = computed(() => {
  const seconds = (currentCurve.value?.points ?? []).map((point) => point.seconds).filter((value): value is number => value !== undefined && value > 0);
  const recent = seconds.slice(-3);
  return recent.length ? recent.reduce((sum, value) => sum + value, 0) / recent.length : null;
});
const left = computed(() => {
  const curve = currentCurve.value;
  if (!running.value || curve === null || perEpoch.value === null || curve.stopped_early !== undefined) return '';
  const epochsLeft = curve.epochs - lastEpoch.value;
  return epochsLeft > 0 ? 'the ' + NET_WORD[current.value as Net] + ' ends in about ' + roughText(epochsLeft * perEpoch.value * 1000) : '';
});
/** Still in the steps before the first network. */
const beforeEpochs = computed(() => running.value && ['import', 'split', 'gpu_env', 'checks', 'mine', 'frames', 'signs']
  .includes(props.progress.steps.find((item) => item.state === 'current')?.key ?? ''));
const beforeText = computed(() => shows('reader')
  ? 'First the training imports the reviews, checks the dataset and prepares the GPU: a few minutes.'
  : deciderCards.value.length ? 'First it checks the dataset and the GPU: a minute or two.'
  : brainCards.value.length ? 'First it checks the labels and the GPU, then reads the signs of the drives it learns from as the car does: a few minutes.'
  : shows('detector') || shows('signnet') ? 'First it checks the frames of the training and of the evaluation, and the GPU: a few minutes.'
    : 'First it checks the dataset, the split and the cutter in use, then runs the cutter on every frame to mine its false boxes.');
const isRunning = (net: Net): boolean => running.value && current.value === net;

/** The charts of applicability and of the brain: a line per variant or seed with points, and the one training now before
 * its first epoch. */
const linesOf = (cards: NetCard[]): NetCard[] => cards.filter((card) => card.curve !== null || isRunning(card.net));
const spanOf = (cards: NetCard[]): number => Math.max(1, ...cards.map((card) => card.curve === null ? 1 : span(card.curve)));
const seriesOf = (cards: NetCard[], pick: (point: EpochPoint) => number | undefined, scale = 1): ChartSeries[] => cards
  .map((card) => ({key: card.net, label: NET_NAME[card.net], color: VARIANT_COLOR[card.net as DeciderNet | BrainNet],
    values: card.curve === null ? [] : column(card.curve, pick, scale)}));
const deciderLines = computed(() => linesOf(deciderCards.value));
const deciderSpan = computed(() => spanOf(deciderLines.value));
const deciderSeries = (pick: (point: EpochPoint) => number | undefined, scale = 1): ChartSeries[] => seriesOf(deciderLines.value, pick, scale);
const brainLines = computed(() => linesOf(brainCards.value));
const brainSpan = computed(() => spanOf(brainLines.value));
const brainSeries = (pick: (point: EpochPoint) => number | undefined, scale = 1): ChartSeries[] => seriesOf(brainLines.value, pick, scale);
/** The road types the brain's epochs measure on validation (brain trainings from 10 Nov 2026 on), in ROAD_TYPES' order. */
const brainTypes = computed(() => brainCards.value.some((card) => card.curve?.points.some((point) => point.val_types !== undefined)));
const brainTypeKinds = computed((): string[] => {
  const kinds = new Set<string>();
  for (const card of brainCards.value) for (const point of card.curve?.points ?? []) Object.keys(point.val_type ?? {}).forEach((kind) => kinds.add(kind));
  return [...kinds];
});
/** The best variant's best epoch, marked on its line. */
function deciderMark(pick: (card: NetCard) => Best, highest: boolean, label: string): {series: string; epoch: number; label: string} | null {
  let found: {net: Net; point: EpochPoint & {value: number}} | null = null;
  for (const card of deciderCards.value) {
    const point = pick(card);
    if (point !== null && (found === null || (highest ? point.value > found.point.value : point.value < found.point.value))) found = {net: card.net, point};
  }
  return found === null ? null : {series: found.net, epoch: found.point.epoch, label};
}
/** The variant training now: its line alone pulses, the band follows its last epoch. */
const deciderRunning = computed(() => deciderCards.value.find((card) => isRunning(card.net))?.net ?? false);
const brainRunning = computed(() => brainCards.value.find((card) => isRunning(card.net))?.net ?? false);
/** Every value, for the table under the charts. */
const tables = computed(() => nets.value.flatMap((net) => {
  const curve = curves.value[net];
  return curve && curve.points.length ? [{net, rows: [...curve.points].sort((a, b) => a.epoch - b.epoch)}] : [];
}));
</script>

<template>
  <section class="card training" aria-labelledby="training-title">
    <h2 id="training-title">{{ title ?? 'How the training is going' }}</h2>
    <div v-if="!hasAny" class="training-wait">
      <svg viewBox="0 0 48 32" aria-hidden="true"><path d="M3 27h42M3 27V4" fill="none" stroke="currentColor" stroke-width="1.5"
        stroke-linecap="round"/><path d="M7 9c6 2 9 9 15 12s12 2 21 3" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-dasharray="2 4"/></svg>
      <p><strong>The curves appear at the end of the first epoch.</strong><span v-if="beforeEpochs">{{ beforeText }}</span></p>
    </div>
    <template v-else>
      <div class="tiles">
        <div class="tile">
          <span class="tile-label">{{ running ? 'Epoch' : 'Epochs done' }}</span>
          <span class="tile-value">{{ lastEpoch }}<small> of {{ currentCurve?.epochs ?? '–' }}</small></span>
          <span class="tile-note">{{ current === null ? '' : NET_WORD[current] + (currentCurve?.stopped_early !== undefined
            ? ', stopped early after epoch ' + currentCurve.stopped_early : running ? ', running' : '') }}</span>
        </div>
        <div v-if="signnetBest" class="tile" id="signnet-best">
          <span class="tile-label">Best 5 boxes, any score</span>
          <span class="tile-value">{{ percent(signnetBest.value * 100) }}</span>
          <span class="tile-note">SignNet on validation, epoch {{ signnetBest.epoch }}<template v-if="signnetBest.val_false_per_frame !== undefined">,
            {{ signnetBest.val_false_per_frame.toFixed(2) }} false boxes per frame</template></span>
        </div>
        <div v-if="signnetPretrainBest && !signnetBest" class="tile" id="signnet-pretrain-best">
          <span class="tile-label">{{ pretrainSpeed ? 'Best on our speed signs' : 'Best 5 boxes, any score' }}</span>
          <span class="tile-value">{{ percent(signnetPretrainBest.value * 100) }}</span>
          <span class="tile-note">Mapillary stage on validation, epoch {{ signnetPretrainBest.epoch }}: the start of the second stage</span>
        </div>
        <div v-if="detectorBestRecall" class="tile" id="detector-best-recall">
          <span class="tile-label">Best validation recall</span>
          <span class="tile-value">{{ percent(detectorBestRecall.value * 100) }}</span>
          <span class="tile-note">detector at score 0.1, epoch {{ detectorBestRecall.epoch }}</span>
        </div>
        <div v-else-if="detectorBest" class="tile">
          <span class="tile-label">Best validation loss</span>
          <span class="tile-value">{{ loss(detectorBest.value) }}</span>
          <span class="tile-note">detector, epoch {{ detectorBest.epoch }}</span>
        </div>
        <div v-if="readerBestAccuracy" class="tile">
          <span class="tile-label">Best validation accuracy</span>
          <span class="tile-value">{{ percent(readerBestAccuracy.value * 100) }}</span>
          <span class="tile-note">reader, epoch {{ readerBestAccuracy.epoch }}</span>
        </div>
        <div v-if="classifierBest" class="tile" id="classifier-best">
          <span class="tile-label">{{ withRejection ? 'Best validation, balanced' : 'Best validation accuracy' }}</span>
          <span class="tile-value">{{ percent(classifierBest.value * 100) }}</span>
          <span class="tile-note">classifier, epoch {{ classifierBest.epoch }}<template v-if="withRejection && classifierBest.val_sign_accuracy !== undefined
            && classifierBest.val_rejection !== undefined">: signs {{ percent(classifierBest.val_sign_accuracy * 100) }}, “not a sign” {{
            percent(classifierBest.val_rejection * 100) }}</template></span>
        </div>
        <div v-if="pretrainBest && !classifierBest" class="tile" id="pretrain-best">
          <span class="tile-label">Best on our signs</span>
          <span class="tile-value">{{ percent(pretrainBest.value * 100) }}</span>
          <span class="tile-note">pretraining on Mapillary, epoch {{ pretrainBest.epoch }}</span>
        </div>
        <div v-for="card in bestCards" :key="'best-' + card.net" class="tile" :id="card.net + '-best'">
          <span class="tile-label">Best on validation</span>
          <span class="tile-value">{{ percent(card.best.value * 100) }}</span>
          <span class="tile-note">{{ NET_NAME[card.net] }}, epoch {{ card.best.epoch }}<template v-if="card.best.val_accuracy !== undefined">:
            read right {{ percent(card.best.val_accuracy * 100) }}, <template v-if="card.best.val_other_road_f1 !== undefined">another
            road {{ percent(card.best.val_other_road_f1 * 100) }}</template><template v-else>whose sign {{ card.best.val_ownership_macro_f1 === undefined ? '–'
            : percent(card.best.val_ownership_macro_f1 * 100) }}</template></template></span>
        </div>
        <template v-for="card in brainCards" :key="'kept-' + card.net">
          <div v-if="card.best" class="tile" :id="card.net + '-kept'">
            <span class="tile-label">{{ NET_NAME[card.net] }} on validation</span>
            <span class="tile-value">{{ kph(card.best.value) }}</span>
            <span class="tile-note">off the labels at epoch {{ card.best.epoch }}<template v-if="card.best.val_accuracy !== undefined">:
              right {{ percent(card.best.val_accuracy * 100) }}</template><template v-if="card.best.val_types !== undefined">,
              road type right {{ percent(card.best.val_types * 100) }}</template></span>
          </div>
        </template>
        <div v-if="perEpoch !== null" class="tile">
          <span class="tile-label">Time per epoch</span>
          <span class="tile-value">{{ elapsedText(perEpoch * 1000) }}</span>
          <span class="tile-note">{{ left || ((current === null ? '' : NET_WORD[current]) + ', mean of the last epochs') }}</span>
        </div>
      </div>

      <div v-if="shows('signnet_pretrain')" class="net" id="signnet-pretrain-curves">
        <h3>{{ NET_NAME.signnet_pretrain }}<span>the first stage, on the Mapillary windows alone · on our validation, found in its 5
          best boxes of each frame, higher is better, and loss</span></h3>
        <div v-if="hasPoints(signnetPretrain)" class="net-charts">
          <LineChart title="Mapillary stage on validation" :top="100" :format="percentAxis" :epochs="span(signnetPretrain)"
                     :series="[...(pretrainSpeed ? [{key: 'speed', label: 'speed signs', color: VAL,
                       values: column(signnetPretrain, (point) => point.val_speed_recall, 100)}] : []),
                       {key: 'val', label: 'every sign', color: pretrainSpeed ? REJECT : VAL, values: column(signnetPretrain, (point) => point.val_recall, 100)}]"
                     :best="signnetPretrainBest ? {series: pretrainSpeed ? 'speed' : 'val', epoch: signnetPretrainBest.epoch, label: 'best'} : null"
                     :running="isRunning('signnet_pretrain')"/>
          <LineChart title="Mapillary stage loss" :epochs="span(signnetPretrain)" :format="lossAxis"
                     :series="[{key: 'train', label: 'train', color: TRAIN, values: column(signnetPretrain, (point) => point.train_loss)}]"
                     :best="signnetPretrainLowest ? {series: 'train', epoch: signnetPretrainLowest.epoch, label: 'lowest'} : null"
                     :running="isRunning('signnet_pretrain')"/>
        </div>
        <p v-else class="net-wait">The Mapillary stage’s curves appear at the end of its first epoch.</p>
        <p v-if="hasPoints(signnetPretrain)" class="note">The windows hold speed signs only, so it keeps the epoch that finds the
          most speed signs of our validation; the second stage, on our drives, starts from those weights.</p>
      </div>

      <div v-if="shows('signnet')" class="net" id="signnet-curves">
        <h3>{{ NET_NAME.signnet }}<span>finds the signs at the top of the frame · on validation, found in its 5 best boxes of each
          frame at any score, higher is better, and loss</span></h3>
        <div v-if="hasPoints(signnet)" class="net-charts">
          <LineChart title="SignNet on validation" :top="100" :format="percentAxis" :epochs="span(signnet)"
                     :series="[{key: 'val', label: 'best 5 boxes, any score', color: VAL, values: column(signnet, (point) => point.val_recall, 100)}]"
                     :best="signnetBest ? {series: 'val', epoch: signnetBest.epoch, label: 'best'} : null"
                     :running="isRunning('signnet')"/>
          <LineChart title="SignNet loss" :epochs="span(signnet)" :format="lossAxis"
                     :series="[{key: 'train', label: 'train', color: TRAIN, values: column(signnet, (point) => point.train_loss)}]"
                     :best="signnetLowest ? {series: 'train', epoch: signnetLowest.epoch, label: 'lowest'} : null"
                     :running="isRunning('signnet')"/>
        </div>
        <p v-else class="net-wait">SignNet’s curves appear at the end of its first epoch.</p>
        <p v-if="hasPoints(signnet)" class="note">It keeps the weights of the epoch that finds the most validation signs in its 5
          best boxes of each frame, at any score; then it is measured on validation and test, also as the car reads it (its best
          5 boxes of score 0.3 or more): those numbers are on the run's SignNet card.</p>
      </div>

      <div v-if="shows('detector')" class="net">
        <h3>{{ NET_NAME.detector }}<span>finds the signs in the frame · {{ detectorRecall ? 'recall on validation, higher is better, and loss'
          : 'loss, lower is better' }}</span></h3>
        <div v-if="hasPoints(detector) && detectorRecall" class="net-charts">
          <LineChart title="Detector recall on validation" :top="100" :format="percentAxis" :epochs="span(detector)"
                     :series="[{key: 'val', label: 'validation, score 0.1', color: VAL, values: column(detector, (point) => point.val_recall, 100)}]"
                     :best="detectorBestRecall ? {series: 'val', epoch: detectorBestRecall.epoch, label: 'best'} : null"
                     :running="isRunning('detector')"/>
          <LineChart title="Detector loss" :series="lossSeries(detector)" :epochs="span(detector)" :format="lossAxis"
                     :best="detectorBest ? {series: 'val', epoch: detectorBest.epoch, label: 'lowest'} : null"
                     :running="isRunning('detector')"/>
        </div>
        <LineChart v-else-if="hasPoints(detector)" title="Detector loss" :series="lossSeries(detector)" :epochs="span(detector)"
                   :format="lossAxis" :best="detectorBest ? {series: 'val', epoch: detectorBest.epoch, label: 'best'} : null"
                   :running="isRunning('detector')"/>
        <p v-else class="net-wait">The detector’s curves appear at the end of its first epoch.</p>
      </div>

      <div v-if="shows('reader')" class="net">
        <h3>{{ NET_NAME.reader }}<span>reads the limit on the sign · accuracy, higher is better, and loss</span></h3>
        <div v-if="hasPoints(reader)" class="net-charts">
          <LineChart title="Reader accuracy" :top="100" :format="percentAxis" :epochs="span(reader)"
                     :series="[{key: 'train', label: 'train', color: TRAIN, values: column(reader, (point) => point.train_accuracy, 100)},
                               {key: 'val', label: 'validation', color: VAL, values: column(reader, (point) => point.val_accuracy, 100)}]"
                     :best="readerBestAccuracy ? {series: 'val', epoch: readerBestAccuracy.epoch, label: 'best'} : null"
                     :running="isRunning('reader')"/>
          <LineChart title="Reader loss" :series="lossSeries(reader)" :epochs="span(reader)" :format="lossAxis"
                     :best="readerBestLoss ? {series: 'val', epoch: readerBestLoss.epoch, label: 'best'} : null"
                     :running="isRunning('reader')"/>
        </div>
        <p v-else class="net-wait">{{ running ? 'The reader starts after the detector: its curves appear at the end of its first epoch.'
          : 'The reader did not reach its first epoch.' }}</p>
      </div>

      <div v-if="shows('pretrain')" class="net" id="pretrain-curves">
        <h3>{{ NET_NAME.pretrain }}<span>the classifier learns on the Mapillary signs alone, measured on our validation signs ·
          accuracy, higher is better, and loss</span></h3>
        <div v-if="hasPoints(pretrain)" class="net-charts">
          <LineChart title="Pretraining accuracy" :top="100" :format="percentAxis" :epochs="span(pretrain)"
                     :series="[{key: 'train', label: 'train, Mapillary', color: TRAIN, values: column(pretrain, (point) => point.train_accuracy, 100)},
                               {key: 'val', label: 'validation, our signs', color: VAL, values: column(pretrain, (point) => point.val_accuracy, 100)}]"
                     :best="pretrainBest ? {series: 'val', epoch: pretrainBest.epoch, label: 'best'} : null"
                     :running="isRunning('pretrain')"/>
          <LineChart title="Pretraining loss" :series="lossSeries(pretrain)" :epochs="span(pretrain)" :format="lossAxis"
                     :best="pretrainBestLoss ? {series: 'val', epoch: pretrainBestLoss.epoch, label: 'lowest'} : null"
                     :running="isRunning('pretrain')"/>
        </div>
        <p class="note">The classifier below starts from its best epoch, then learns on your signs and “not a sign”.</p>
      </div>

      <div v-if="shows('classifier')" class="net" id="classifier-curves">
        <h3>{{ NET_NAME.classifier }}<span>says what each box is, “not a sign” included · on validation, higher is better, and loss</span></h3>
        <div v-if="hasPoints(classifier)" class="net-charts">
          <LineChart :title="withRejection ? 'Classifier on validation' : 'Classifier accuracy'" :top="100" :format="percentAxis"
                     :epochs="span(classifier)" :series="classifierSeries"
                     :best="classifierBest ? {series: 'val', epoch: classifierBest.epoch, label: 'best'} : null"
                     :running="isRunning('classifier')"/>
          <LineChart title="Classifier loss" :series="lossSeries(classifier)" :epochs="span(classifier)" :format="lossAxis"
                     :best="classifierBestLoss ? {series: 'val', epoch: classifierBestLoss.epoch, label: 'lowest'} : null"
                     :running="isRunning('classifier')"/>
        </div>
        <p v-else class="net-wait">{{ running ? (shows('pretrain') ? 'The classifier starts from the pretraining’s best epoch: its curves appear at the end of its first epoch.'
          : 'The classifier starts after the mining of the cutter’s false boxes: its curves appear at the end of its first epoch.')
          : 'The classifier did not reach its first epoch.' }}</p>
        <p v-if="withRejection" class="note">Balanced: the mean of the sign crops read right and the “not a sign” crops rejected; it
          chooses the epoch whose weights are kept.</p>
      </div>


      <div v-if="deciderCards.length" class="net" id="decider-curves">
        <h3>Applicability<span>is this sign for our road? · a line per variant, on validation: the chosen measure, higher is
          better, and the loss</span></h3>
        <div v-if="deciderCards.some((card) => card.curve)" class="net-charts">
          <LineChart title="Chosen measure on validation" :top="100" :format="percentAxis" :epochs="deciderSpan"
                     :series="deciderSeries((point) => point.val_selection_score, 100)"
                     :best="deciderMark((card) => card.best, true, 'best')" :running="deciderRunning"/>
          <LineChart title="Validation loss" :format="lossAxis" :epochs="deciderSpan" :series="deciderSeries((point) => point.val_loss)"
                     :best="deciderMark((card) => card.lowest, false, 'lowest')" :running="deciderRunning"/>
        </div>
        <p v-else class="net-wait">{{ running ? 'The curves appear at the end of the first variant’s first epoch.'
          : 'No variant reached its first epoch.' }}</p>
        <p class="note" id="decider-score-note">Chosen measure: the mean of the signs read right and the F1 of the signs of
          another road; it chooses the epoch whose weights each variant keeps. Abstain (unclear) is measured, out of the choice.
          The two apart are in the tiles above and in the values per epoch. A training of before 11 Nov 2026 chose on the macro
          F1 of whose sign it is (our road, another road, abstain) in place of the other-road F1.</p>
      </div>

      <div v-if="brainCards.length" class="net" id="brain-curves">
        <h3>The brain<span>the speed to hold, once a second · a line per seed, on validation: km/h off the labels, lower is better,
          the frames it says right and the road type it names right, higher is better, and train loss</span></h3>
        <div v-if="brainCards.some((card) => card.curve)" class="net-charts">
          <LineChart title="Km/h off on validation" :format="kphAxis" :epochs="brainSpan"
                     :series="brainSeries((point) => point.val_error_kph)" :running="brainRunning"/>
          <LineChart title="Right on validation" :top="100" :format="percentAxis" :epochs="brainSpan"
                     :series="brainSeries((point) => point.val_accuracy, 100)" :running="brainRunning"/>
          <LineChart v-if="brainTypes" title="Road type right on validation" :top="100" :format="percentAxis" :epochs="brainSpan"
                     :series="brainSeries((point) => point.val_types, 100)" :running="brainRunning"/>
          <LineChart title="Train loss" :format="lossAxis" :epochs="brainSpan" :series="brainSeries((point) => point.train_loss)"
                     :running="brainRunning"/>
        </div>
        <p v-else class="net-wait">{{ running ? 'The curves appear at the end of the first seed’s first epoch.'
          : 'No seed reached its first epoch.' }}</p>
        <p class="note" id="brain-note">Each seed keeps the weights of its last epoch, then it is measured on the test: its numbers
          are in Models › Brain, where you choose the one for the car.</p>
      </div>

      <details class="chart-table">
        <summary>Values per epoch</summary>
        <div v-for="table in tables" :key="table.net" class="table-wrap">
          <table>
            <caption>{{ NET_NAME[table.net] }}</caption>
            <thead><tr><th class="num">Epoch</th><th class="num">Train loss</th><th v-if="!isSignnet(table.net) && !isBrain(table.net)" class="num">Val loss</th>
              <template v-if="table.net !== 'detector' && !isSignnet(table.net) && !isBrain(table.net)"><th class="num">Train acc.</th><th class="num">Val acc.</th></template>
              <template v-if="isBrain(table.net)"><th class="num">Km/h off</th><th class="num">Right</th>
                <template v-if="brainTypes"><th class="num" title="The road type named right, on all types">Road type</th>
                  <th v-for="kind in brainTypeKinds" :key="kind" class="num" :title="'Road type right on ' + (ROAD_TYPE_LABELS[kind]?.label ?? kind)">{{
                    ROAD_TYPE_LABELS[kind]?.short ?? kind }}</th></template></template>
              <template v-if="isDecider(table.net)"><th class="num" title="Macro F1 of our road, another road and abstain">Whose sign</th><th
                class="num" title="The signs of another road found">Other road</th><th class="num" title="F1 of the signs of another road">Other
                F1</th><th class="num">Chosen</th></template>
              <th v-if="table.net === 'detector' && detectorRecall" class="num">Val recall</th>
              <template v-if="isSignnet(table.net)"><th class="num" title="Its 5 best boxes of each frame, at any score">Best 5, any score</th><th
                v-if="table.net === 'signnet_pretrain' && pretrainSpeed" class="num" title="The speed signs among them">Speed signs</th><th
                v-if="signnetFalse" class="num">False per frame</th></template>
              <template v-if="table.net === 'classifier' && withRejection"><th class="num">Signs</th><th class="num">Not a sign</th>
                <th class="num">Balanced</th></template><th class="num">Time</th></tr></thead>
            <tbody>
              <tr v-for="point in table.rows" :key="point.epoch">
                <td class="num">{{ point.epoch }}</td>
                <td class="num">{{ cell(point.train_loss, loss) }}</td>
                <td v-if="!isSignnet(table.net) && !isBrain(table.net)" class="num">{{ cell(point.val_loss, loss) }}</td>
                <template v-if="isBrain(table.net)">
                  <td class="num">{{ cell(point.val_error_kph, (value) => value.toFixed(1)) }}</td>
                  <td class="num">{{ cell(point.val_accuracy, (value) => percent(value * 100)) }}</td>
                  <template v-if="brainTypes"><td class="num">{{ cell(point.val_types, (value) => percent(value * 100)) }}</td>
                    <td v-for="kind in brainTypeKinds" :key="kind" class="num">{{ cell(point.val_type?.[kind], (value) => percent(value * 100)) }}</td></template>
                </template>
                <template v-if="table.net !== 'detector' && !isSignnet(table.net) && !isBrain(table.net)">
                  <td class="num">{{ cell(point.train_accuracy, (value) => percent(value * 100)) }}</td>
                  <td class="num">{{ cell(point.val_accuracy, (value) => percent(value * 100)) }}</td>
                </template>
                <td v-if="table.net === 'detector' && detectorRecall" class="num">{{ cell(point.val_recall, (value) => percent(value * 100)) }}</td>
                <template v-if="isSignnet(table.net)">
                  <td class="num">{{ cell(point.val_recall, (value) => percent(value * 100)) }}</td>
                  <td v-if="table.net === 'signnet_pretrain' && pretrainSpeed" class="num">{{ cell(point.val_speed_recall, (value) => percent(value * 100)) }}</td>
                  <td v-if="signnetFalse" class="num">{{ cell(point.val_false_per_frame, (value) => value.toFixed(2)) }}</td>
                </template>
                <template v-if="isDecider(table.net)">
                  <td class="num">{{ cell(point.val_ownership_macro_f1, (value) => percent(value * 100)) }}</td>
                  <td class="num">{{ cell(point.val_other_road_recall, (value) => percent(value * 100)) }}</td>
                  <td class="num">{{ cell(point.val_other_road_f1, (value) => percent(value * 100)) }}</td>
                  <td class="num">{{ cell(point.val_selection_score, (value) => percent(value * 100)) }}</td>
                </template>
                <template v-if="table.net === 'classifier' && withRejection">
                  <td class="num">{{ cell(point.val_sign_accuracy, (value) => percent(value * 100)) }}</td>
                  <td class="num">{{ cell(point.val_rejection, (value) => percent(value * 100)) }}</td>
                  <td class="num">{{ cell(point.val_balanced_accuracy, (value) => percent(value * 100)) }}</td>
                </template>
                <td class="num">{{ point.seconds === undefined ? '–' : elapsedText(point.seconds * 1000) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </details>
    </template>
  </section>
</template>
