<script setup lang="ts">
// One trained version of a role: its run, what it is and when its weights were saved, the headline numbers that judge
// that role side by side (MetricTiles: the cutter: the speed signs and every sign found at a low score, as on Evaluate,
// then by width, boxes per frame, time per frame; the
// classifier: the speed signs and every sign read right on the test, and for the classifier with “not a sign” the “not a sign” crops it rejects and
// the chain end to end at the cutter's score, the speed signs found and read right and the false boxes per frame that
// survive it; the decider: the "Another" road found, on validation and test; the brain: the test frames it says right),
// the button that puts it in use and the link to its run on Evaluate. “Use this one” only on a version the server lets be chosen (version.selectable: finished,
// and for the cutter a SignNet at the car's band; an old Detector is kept to compare), else why not in a small note.
// A SignNet (the cutter the comma runs) has its own details and, next to “Use this one”, “Try on the comma now”
// (scripts/try_on_comma.sh: copied straight onto the comma until its next software update). The car gets the networks
// in use with Publish, on the Publish tab (CarPublishSection.vue).
// “On the car”: one of the networks the comma downloads; “Trying on the comma”: the one copied there now.
import {computed} from 'vue';
import {isoWhen, thousands} from '../common/format';
import type {DeciderNumbers, ModelVersion} from '../common/panel-types';
import {PARTITION_LABEL, pct} from './evaluate';
import {ROLE_TEXT, epochText, evaluateHref, metricsOf, openRun, perFrame, roleWords, sizeText, type CarState, type Metric} from './models';
import MetricTiles from './MetricTiles.vue';

const props = defineProps<{version: ModelVersion; roleLabel: string; score: number; readOnly: boolean; busy: string | null;
  car?: CarState | null}>();
const emit = defineEmits<{use: [version: ModelVersion]; try: [version: ModelVersion]}>();

const cutter = computed(() => props.version.role === 'cutter' ? props.version : null);
const signnet = computed(() => cutter.value?.kind === 'signnet');
/** A SignNet the comma can run (its band and grid): only it can be tried on or published to the car. */
const forCar = computed(() => signnet.value && cutter.value?.fits_the_car !== false);
const classifier = computed(() => props.version.role === 'classifier' ? props.version : null);
const decider = computed(() => props.version.role === 'decider' ? props.version : null);
const brain = computed(() => props.version.role === 'brain' ? props.version : null);
/** The earlier run a SignNet or an applicability variant went on from (Train › Start from). */
const networkStart = computed(() => props.version.role === 'cutter' || props.version.role === 'decider'
  ? props.version.start ?? null : null);
/** The cutter's numbers: on the test, else on validation (a size compared before its test was measured). */
const cut = computed(() => cutter.value?.test ?? cutter.value?.validation ?? null);
const cutterMetrics = computed((): Metric[] => cutter.value ? metricsOf(cutter.value) : []);
const cutterCaption = computed(() => cut.value === null ? ''
  : 'Found on the ' + (PARTITION_LABEL[cut.value.partition] ?? cut.value.partition).toLowerCase() + ' at score ' + (cut.value.score_threshold ?? '–')
    + (signnet.value ? ', as the car reads them (its best ' + (cut.value.car?.max_boxes ?? 5) + ' boxes of each frame of score '
      + (cut.value.car?.score_threshold ?? 0.3) + ' or more)' : '')
    + ', and in the ' + (cut.value.budget?.boxes ?? 5) + ' best boxes of each frame at any score'
    + (signnet.value && cut.value.car_centre ? '. As the car reads them, a box counts when it sits on the sign (its centre on '
      + 'it, half to twice its size); in small, the strict count at IoU 0.5' : ''));
const evaluated = computed(() => cut.value !== null || (classifier.value?.test ?? null) !== null || (classifier.value?.chain ?? null) !== null
  || decider.value?.validation != null || decider.value?.test != null || brain.value?.test != null);
const epoch = computed(() => epochText(props.version.best_epoch, props.version.epochs_run ?? props.version.epochs));
/** The classifier with “not a sign” (classifier:<run>/classifier); an old reader has none of its numbers. */
const rejects = computed(() => classifier.value?.class_scheme !== undefined);
const chain = computed(() => classifier.value?.chain ?? null);
const oldChain = computed(() => classifier.value?.chain_old_reader ?? null);
const classifierMetrics = computed((): Metric[] => classifier.value ? metricsOf(classifier.value) : []);
/** It can be put in use: the server says so (an older server: finished, and not a SignNet). */
const selectable = computed(() => props.version.selectable ?? (props.version.finished && !signnet.value));
/** Why not, in a few words, in place of the button (nothing on the one in use, nor on one still training: its pill
 * and its line say so). */
const notSelectable = computed((): string => props.version.in_use || selectable.value || !props.version.finished ? ''
  : props.version.not_selectable_reason ?? '');
/** Why the button is off, as its title. */
const blocked = computed((): string => {
  if (props.readOnly) return 'Read-only: the choice is turned off on this server.';
  return '';
});

/** Try on the comma now: on a SignNet neither on the car nor on trial there already. */
const onTrial = computed(() => signnet.value && props.car?.onTrial === props.version.run);
const triable = computed(() => forCar.value && !props.version.on_car && !onTrial.value);
const trying = computed(() => props.car?.trying === props.version.run);
/** Why “Try on the comma now” is off, as its title. */
const tryBlocked = computed((): string => {
  if (props.readOnly) return 'Read-only: the buttons are turned off on this server.';
  if (!props.version.finished) return 'Its training has not finished: its weights may still change.';
  if (props.car?.trying != null) return 'Networks are being copied onto the comma now.';
  if (props.car?.sending != null) return 'The networks are being published to the car now.';
  const state = props.car?.tryState ?? null;
  if (state === null || state === undefined) return 'Trying on the comma is not available on this server.';
  return state.allowed ? '' : state.reason.charAt(0).toUpperCase() + state.reason.slice(1) + '.';
});
const DECIDER_ROWS: {key: keyof DeciderNumbers; label: string; title: string}[] = [
  {key: 'other_road_recall', label: '“Another” found', title: 'Share of the signs for another road it calls “Another” (recall)'},
  {key: 'other_road_precision', label: '“Another” right', title: 'Share of its “Another” answers that are right (precision)'},
  {key: 'macro_f1', label: 'Macro-F1', title: 'Ours, Another and Unclear weighted alike'},
  {key: 'abstain_rate', label: 'Abstains', title: 'Share of the signs it leaves undecided'},
  {key: 'reading_accuracy', label: 'Reading', title: 'Share of the boxes it reads right'},
];
/** A cell of the decider's table; "Another" found on a partition without such a sign is "no case" (its precision
 * there still counts the wrong "Another" answers). */
function deciderCell(numbers: DeciderNumbers | null, key: keyof DeciderNumbers): string {
  if (numbers === null) return '–';
  if (key === 'other_road_recall' && numbers.other_road_support === 0) return 'no case';
  return pct(numbers[key]);
}
const noOtherOnTest = computed(() => decider.value?.test?.other_road_support === 0);

</script>

<template>
  <li class="model" :class="{'in-use': version.in_use}" :data-model="version.id">
    <div class="model-head">
      <div class="model-title">
        <span class="model-name" :title="version.id">{{ version.label }}</span>
        <span class="model-sub">{{ version.trained_at ? isoWhen(version.trained_at) : version.run }} · {{ version.detail }}</span>
      </div>
      <span v-if="version.in_use || version.on_car || onTrial || !version.finished" class="model-pills">
        <span v-if="version.in_use" class="pill ok">In use</span>
        <span v-else-if="!version.finished" class="pill warn">Not finished</span>
        <span v-if="version.on_car" class="pill car">In the branch</span>
        <span v-if="onTrial" class="pill trial">Trying on the comma</span>
      </span>
    </div>

    <template v-if="cutter">
      <template v-if="cut">
        <MetricTiles :metrics="cutterMetrics" :caption="cutterCaption"/>
        <details class="model-more"><summary>Details</summary>
        <ul v-if="cut.widths" class="model-widths" aria-label="Signs found by width in the frame">
          <li v-for="row in cut.widths" :key="row.key" :title="row.found + ' of ' + row.truth + ' signs ' + row.label + ' wide'">
            <span>{{ row.label }}</span><b>{{ row.truth ? pct(row.recall) : '–' }}</b><span>{{ row.found }} of {{ row.truth }}</span>
          </li>
        </ul>
        <dl class="model-facts">
          <div><dt>Boxes per frame</dt><dd>{{ perFrame(cut.boxes_per_frame) }}</dd></div>
          <div><dt>False per frame</dt><dd>{{ perFrame(cut.false_per_frame) }}</dd></div>
          <template v-if="signnet">
          <div><dt title="The top rows of the frame, scaled to this width">Input band</dt><dd>{{ cutter.input_size ? sizeText(cutter.input_size) : '–' }}</dd></div>
          <div><dt title="The step of its output grid, in pixels of the band">Grid step</dt><dd>{{ cutter.stride ? cutter.stride + ' px' : '–' }}</dd></div>
          <div><dt>Weights</dt><dd>{{ cutter.parameters ? thousands(cutter.parameters) : '–' }}</dd></div>
          <div v-if="cutter.train_share"><dt>Training drives</dt><dd>{{ pct(cutter.train_share) }}</dd></div>
          </template>
          <template v-else>
          <div><dt>Time per frame</dt><dd :title="cut.device ?? ''">{{ cut.ms_per_frame === null ? '–' : perFrame(cut.ms_per_frame) + ' ms' }}</dd></div>
          <div><dt>Input</dt><dd>{{ sizeText(cutter.input_size) }}<span v-if="!cutter.input_size_recorded" class="muted"> (default)</span></dd></div>
          <div><dt>Default box</dt><dd>{{ cutter.anchor_min_ratio ?? '–' }}<span v-if="!cutter.anchor_recorded" class="muted"> (default)</span></dd></div>
          </template>
          <div><dt>Best epoch</dt><dd>{{ epoch }}</dd></div>
          <div><dt>Finds</dt><dd>{{ cutter.families === 'all' ? 'speed and other signs' : 'speed signs' }}<span v-if="!cutter.families_recorded"
            class="muted"> (not recorded)</span></dd></div>
        </dl>
        <p v-if="!cut.at_cutter_score" class="note">At the report’s score {{ cut.score_threshold ?? '–' }}: it was evaluated before the cutter’s {{ score }} was measured.</p>
        <p v-if="cut.human" class="note">Frames a person confirmed: {{ pct(cut.human.recall) }} found ({{ cut.human.found ?? 0 }} of {{ cut.human.truth_boxes ?? 0 }}).</p>
        </details>
      </template>
    </template>

    <template v-else-if="classifier && rejects">
      <template v-if="classifier.test || chain">
        <MetricTiles :metrics="classifierMetrics" :caption="classifier.test ? 'Crops of the test signs; every sign without the kind of an other sign' : ''"/>
        <p v-if="classifier.start?.another_split" class="note warn-text">Started from a classifier trained on another split: it may
          have seen some of these test signs, so its test is not fair.</p>
        <details class="model-more"><summary>Details</summary>
        <dl class="model-facts">
          <div v-if="classifier.start"><dt>Started from</dt><dd>{{
            classifier.start.mapillary_only && classifier.start.run === classifier.run ? 'Its Mapillary stage'
            : classifier.start.run ? classifier.start.run + (classifier.start.mapillary_only ? ' (its Mapillary stage)' : '')
            : 'Another classifier' }}</dd></div>
          <div v-if="classifier.test"><dt>Human-reviewed</dt><dd>{{ pct(classifier.test.human_accuracy) }}<span v-if="classifier.test.human_samples"
            class="muted"> of {{ thousands(classifier.test.human_samples) }}</span></dd></div>
          <div v-if="classifier.test"><dt title="Share of the “not a sign” test crops it answers “not a sign”">“Not a sign” rejected</dt><dd>{{
            pct(classifier.test.rejection_rate) }}<span v-if="classifier.test.not_a_sign_crops" class="muted"> of {{ thousands(classifier.test.not_a_sign_crops) }}</span></dd></div>
          <div v-if="classifier.test"><dt title="Sign crops it throws away as “not a sign”">Signs thrown away</dt><dd>{{
            pct(classifier.test.signs_rejected_rate) }}<span v-if="classifier.test.signs_rejected !== null && classifier.test.signs_rejected !== undefined"
            class="muted"> ({{ thousands(classifier.test.signs_rejected) }})</span></dd></div>
          <div v-if="chain"><dt :title="'Speed signs of the test found by the cutter at score ' + chain.score_threshold + ' and read right by this classifier'">Chain, end to end</dt><dd>{{
            pct(chain.recall) }}<span v-if="chain.truth" class="muted"> ({{ thousands(chain.correct ?? 0) }} of {{ thousands(chain.truth) }})</span></dd></div>
          <div v-if="chain"><dt :title="'False boxes of the cutter at score ' + chain.score_threshold + ' that this classifier lets through, per frame'">Surviving false per frame</dt><dd>{{
            perFrame(chain.surviving_false_per_frame) }}<span class="muted"> of {{ perFrame(chain.false_per_frame) }}</span></dd></div>
          <div><dt title="Mean of the sign crops read right and the “not a sign” crops rejected, on validation: it chose the best epoch">Best validation</dt><dd>{{
            pct(classifier.best_validation_balanced_accuracy ?? classifier.best_validation_accuracy) }}</dd></div>
          <div><dt>Best epoch</dt><dd>{{ epoch }}</dd></div>
          <div><dt>Classes</dt><dd>{{ classifier.classes ?? '–' }}</dd></div>
        </dl>
        <p v-if="chain" class="note">Chain at the cutter’s score {{ chain.score_threshold }}: the cutter alone finds {{ pct(chain.recall_found) }}
          of the speed signs<template v-if="chain.human_recall !== null">; on the frames a person confirmed, {{ pct(chain.human_recall) }} end to end</template>.</p>
        <p v-if="chain && oldChain" class="note">The old reader in the same chain: {{ pct(oldChain.recall) }} end to end, every false box
          let through ({{ perFrame(oldChain.surviving_false_per_frame) }} per frame).</p>
        <p v-if="classifier.test?.family_accuracy != null" class="note">Every sign with the kind of an other sign counted too:
          {{ pct(classifier.test.accuracy) }} right.</p>
        </details>
      </template>
    </template>

    <template v-else-if="classifier">
      <template v-if="classifier.test">
        <div class="model-big">
          <strong>{{ pct(classifier.test.accuracy) }}</strong>
          <span>right answers on {{ thousands(classifier.test.samples ?? 0) }} test crops</span>
        </div>
        <details class="model-more"><summary>Details</summary>
        <dl class="model-facts">
          <div><dt>Human-reviewed</dt><dd>{{ pct(classifier.test.human_accuracy) }}<span v-if="classifier.test.human_samples" class="muted"> of {{ thousands(classifier.test.human_samples) }}</span></dd></div>
          <div><dt>Classes</dt><dd>{{ classifier.classes ?? '–' }}</dd></div>
          <div><dt>Best validation</dt><dd>{{ pct(classifier.best_validation_accuracy) }}</dd></div>
          <div><dt>Best epoch</dt><dd>{{ epoch }}</dd></div>
        </dl>
        </details>
      </template>
    </template>

    <template v-else-if="decider">
      <template v-if="decider.validation || decider.test">
        <div class="model-big">
          <strong>{{ pct(decider.validation?.other_road_recall) }}</strong>
          <span>of the “Another” signs found on validation<template v-if="decider.validation?.other_road_support">:
            {{ decider.validation.other_road_found ?? 0 }} of {{ decider.validation.other_road_support }}</template></span>
        </div>
        <details class="model-more"><summary>Details</summary>
        <div class="table-wrap"><table class="model-table">
          <thead><tr><th></th><th class="num">Validation</th><th class="num">Test</th></tr></thead>
          <tbody>
            <tr v-for="row in DECIDER_ROWS" :key="row.key" :data-metric="row.key">
              <th scope="row" :title="row.title">{{ row.label }}</th>
              <td class="num">{{ deciderCell(decider.validation, row.key) }}</td>
              <td class="num">{{ deciderCell(decider.test, row.key) }}</td>
            </tr>
          </tbody>
        </table></div>
        <p v-if="decider.inputs" class="note">Reads: {{ decider.inputs.join(' + ') }} · best epoch {{ epoch }}.</p>
        <p v-if="noOtherOnTest" class="note">The test has no “Another” sign yet: whom a sign applies to is measured on validation only.</p>
        <p v-if="decider.smoke_limit !== null" class="note ds-note">A smoke run on {{ decider.smoke_limit }} frames per partition, not a measure.</p>
        </details>
      </template>
    </template>

    <template v-else-if="brain">
      <template v-if="brain.test">
        <div class="model-big">
          <strong>{{ pct(brain.test.right) }}</strong>
          <span>right on {{ thousands(brain.test.frames ?? 0) }} test frames</span>
        </div>
        <details class="model-more"><summary>Details</summary>
        <dl class="model-facts">
          <div><dt>Off on average</dt><dd>{{ brain.test.mean_error_kph?.toFixed(1) ?? '–' }} km/h</dd></div>
          <div><dt>On validation</dt><dd>{{ pct(brain.validation?.right) }}</dd></div>
          <div><dt title="What it reads each second beside the picture; never the map">Inputs</dt><dd>{{ brain.groups?.join(', ') ?? '–' }}</dd></div>
          <div><dt title="Whether the frames it trained on had a limit with a plate of hours">Hours plates</dt><dd>{{ brain.hours ? 'Reads them and their hours' : brain.plates ? 'Reads them' : 'Does not read them' }}</dd></div>
          <div><dt title="Whether the frames it trained on had a limit valid only on a wet road">Wet plates</dt><dd>{{ brain.wet ? 'Reads them' : 'Does not read them' }}</dd></div>
          <div><dt>Seed</dt><dd>{{ brain.seed ?? '–' }}</dd></div>
        </dl>
        </details>
      </template>
    </template>

    <p v-if="networkStart" class="note" :class="{'warn-text': networkStart.another_split}" data-start>Went on from
      {{ networkStart.run ?? 'an earlier run' }}<template v-if="networkStart.epoch !== null">, its epoch {{ networkStart.epoch }}</template><template
      v-if="networkStart.another_split">: trained on another split, so this test is not fair</template>.</p>
    <p v-if="!evaluated" class="model-wait">{{ version.finished ? 'Not evaluated: no numbers on the test yet.' : 'Still training, or stopped before its end.' }}</p>
    <p v-for="problem in version.problems" :key="problem" class="note ds-note">{{ problem }}.</p>
    <p v-if="notSelectable" class="note model-why-not">{{ notSelectable }}</p>

    <div class="model-actions">
      <a class="flow-link" :href="evaluateHref(version)" @click="openRun($event, version)">Open the run ›</a>
      <span v-if="forCar || (selectable && !version.in_use)" class="model-buttons">
        <button v-if="selectable && !version.in_use" class="btn small" type="button" :disabled="blocked !== '' || busy !== null"
                :title="blocked || ROLE_TEXT[version.role].uses"
                :aria-label="'Use ' + version.run + ' · ' + version.label + ' as the ' + roleWords(roleLabel)"
                @click="emit('use', version)">{{ busy === version.id ? 'Saving…' : 'Use this one' }}</button>
        <template v-if="forCar">
          <button v-if="triable" class="btn small" type="button" :disabled="tryBlocked !== ''"
                  :title="tryBlocked || 'Copies it with the classifier in use straight onto the comma, until its next software update.'"
                  :aria-label="'Try ' + version.run + ' · ' + version.label + ' on the comma now'" :data-try="version.run"
                  @click="emit('try', version)">{{ trying ? 'Copying…' : 'Try on the comma now' }}</button>
          <span v-if="version.on_car" class="model-car-note">The comma runs this one.</span>
        </template>
      </span>
    </div>
  </li>
</template>
