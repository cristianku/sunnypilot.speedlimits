<script setup lang="ts">
// Results, the scores: by network, choose what to judge (the whole chain, the cutter, the classifier, the decider); every
// version of it in one table, judged by the one number that matters for that network, next to the version in use;
// the version chosen (or the one “Open the run ›” of Models and Train asks for, ?model=) played frame by frame on the
// test (WatchSection), and all the numbers of its run folded below. Chains measured with another cutter are marked
// not comparable. A SignNet cutter is judged as the car reads it (its best 5 boxes a frame of score 0.3 or more), an old
// Detector at score 0.1: the two are marked not comparable.
// The proposers of Prepare (the Vietnamese YOLO, our networks, Qwen3-VL) are judged the same way, each measured on the
// reviewed test frames with its Measure button (helpers/measure_proposer.sh); so are the AI reviewers of Settings, each
// asked about the same frames a person reviewed (helpers/measure_ai_reviewer.sh).
import {computed, ref, shallowRef, watch} from 'vue';
import type {AiProvider, AiReviewerReport, AiReviewersPayload} from '../common/ai-types';
import {api, messageOf} from '../common/api';
import {isoWhen, thousands} from '../common/format';
import type {ActionName, ActionOptions, ActionState, ClassifierModel, CutterModel, DeciderModel, ModelRole, ModelVersion,
  PanelStatus, Proposer, ProposersPayload, WatchModel} from '../common/panel-types';
import type {Job} from '../common/panel-types';
import ActionList from './ActionList.vue';
import {pct, points, selectedModel, selectedRun, useRunDetail, useRuns} from './evaluate';
import JobProgress from './JobProgress.vue';
import {EVALUATE_ACTIONS, jobOf} from './jobs';
import {perFrame, sizeText, useModels} from './models';
import RunDetailView from './RunDetail.vue';
import Segmented from './Segmented.vue';
import WatchSection from './WatchSection.vue';
import SectionHead from './SectionHead.vue';
import {usePoll} from '../composables/poll';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions]; stop: [job: Job];
  details: [job: Job]}>();
/** A measure (of a proposer, of an AI reviewer) running, or the last one when it failed: its card at the top (its dot
 * in the top bar leads here, jobs.ts ACTION_TAB). */
const measureJob = computed(() => {
  const job = jobOf(props.status, EVALUATE_ACTIONS);
  return job && (job.state === 'running' || job.state === 'failed') ? job : null;
});
/** The words of a proposer or AI reviewer not measured yet: Measure is pressed only when no measure runs. */
const notMeasured = computed(() => measureJob.value?.state === 'running' ? 'not measured yet' : 'not measured yet: press Measure');

type View = 'proposers' | 'chain' | 'cutter' | 'classifier' | 'decider' | 'ai';
const VIEWS: {key: View; label: string}[] = [
  {key: 'proposers', label: 'Proposers'}, {key: 'chain', label: 'Chain'}, {key: 'cutter', label: 'Cutter'},
  {key: 'classifier', label: 'Classifier'}, {key: 'decider', label: 'Applicability'}, {key: 'ai', label: 'AI reviewers'},
];
/** What each network must do, and the number that judges it. */
const GOAL: Record<View, string> = {
  proposers: 'Who proposes the signs in Prepare, each run on the same reviewed test frames. What counts: of the speed '
    + 'signs, how many it proposes with the right number, and how many false proposals per frame you would have to delete.',
  chain: 'The cutter and the classifier together, on the test roads no training saw. What counts: of the test speed '
    + 'signs, how many are found and read right, and how many false boxes are left per frame.',
  cutter: 'Finds every box that could be a sign. What counts: how many of the test speed signs it finds; a SignNet as '
    + 'the car reads it, among its best 5 boxes of each frame of score 0.3 or more. False boxes are expected: the '
    + 'classifier drops them.',
  classifier: 'Says what each box is, or “not a sign”. What counts: the test signs read right, and the false boxes '
    + 'it throws away.',
  decider: 'Says whom a sign applies to: our road or another. What counts: the “Another road” signs it finds. '
    + 'Only on validation: the test has no such sign yet.',
  ai: 'The models of Settings that review the signs, each asked about the same frames a person reviewed, as it would '
    + 'review them. What counts: of the speed signs, how many it keeps or draws with the right number; then the frames '
    + 'without signs it leaves empty, and the false boxes per frame you would have to delete.',
};

interface Row {
  id: string;
  name: string;
  when: string;
  /** The judging share (0-1) and its words. */
  main: number | null;
  mainText: string;
  second: string;
  inUse: boolean;
  run: string;
  /** The report of the run the player shows for this row. */
  watch: string | null;
  /** The run of that report when it is not the row's own (the old reader is played in the chain of a classifier run). */
  watchRun?: string | undefined;
  /** Its own comparison (the chain: with the old reader on the same frames). */
  vs?: {text: string; kind: string} | undefined;
  /** Why its number cannot be compared with the others (a chain measured with another cutter). */
  apart?: string | undefined;
  /** What the mark of the one in use says ("in use"; an AI reviewer: "reviews"). */
  badge?: string | undefined;
  /** What its number is measured on ("classifier of 381"): only rows with the same basis are compared. */
  basis?: string | undefined;
}

const storedView = localStorage.getItem('nnslr-evaluate-view');
const view = ref<View>(VIEWS.find((item) => item.key === storedView)?.key ?? 'chain');
watch(view, (value) => localStorage.setItem('nnslr-evaluate-view', value));
/** The proposers as measured (helpers/measure_proposer.sh), asked again every 15 s: a measure takes minutes. */
const proposers = shallowRef<ProposersPayload | null>(null);
const proposersError = ref('');
/** The AI reviewers and their measures (helpers/measure_ai_reviewer.sh), every 15 s while their view is shown. */
const aiReviewers = shallowRef<AiReviewersPayload | null>(null);
const aiError = ref('');
async function askProposers(): Promise<void> {
  try {
    proposers.value = await api<ProposersPayload>('/api/panel/proposers');
    proposersError.value = '';
  } catch (problem) {
    proposersError.value = messageOf(problem);
  }
}
async function askAiReviewers(): Promise<void> {
  try {
    aiReviewers.value = await api<AiReviewersPayload>('/api/panel/ai-reviewers');
    aiError.value = '';
  } catch (problem) {
    aiError.value = messageOf(problem);
  }
}
usePoll(() => Promise.all([askProposers(), view.value === 'ai' ? askAiReviewers() : null]).then(() => undefined), 15000);
watch(view, (value) => { if (value === 'ai') void askAiReviewers(); });
const picked = ref<string | null>(null);
const {payload: models, error: modelsError} = useModels();
const {payload: runsPayload} = useRuns();

function versionsOf(role: string): ModelVersion[] {
  return models.value?.roles.find((item) => item.role === role)?.versions ?? [];
}
function share(found: number | null | undefined, truth: number | null | undefined): number | null {
  return found === null || found === undefined || !truth ? null : found / truth;
}
function ofText(found: number | null | undefined, truth: number | null | undefined, what: string): string {
  return found === null || found === undefined || !truth ? 'not measured' : pct(found / truth) + ' · ' + found + ' of ' + truth + ' ' + what;
}

/** A row of an AI reviewer: a provider of Settings with its latest measure for its model, or an older measure. */
function aiRow(name: string, model: string, providerId: string, report: AiReviewerReport | null, reviews: boolean): Row {
  const speed = report?.speed_signs;
  return {id: 'ai:' + providerId + '/' + model, name, when: model + (report ? ' · ' + isoWhen(report.measured_at) : ''),
    main: speed ? speed.read_right_share ?? share(speed.read_right, speed.truth) : null,
    mainText: speed ? ofText(speed.read_right, speed.truth, 'speed signs with the right number') : notMeasured.value,
    second: report ? aiSecond(report) : '', inUse: reviews, badge: 'reviews', run: '', watch: null};
}
function aiSecond(report: AiReviewerReport): string {
  const empty = report.no_sign;
  const parts = [empty.truth ? 'left ' + empty.agreed + ' of ' + empty.truth + ' frames without signs empty' : 'no frame without signs',
    perFrame(report.false_per_frame) + ' false boxes per frame'];
  if (report.seconds_per_frame !== null) parts.push(report.seconds_per_frame.toFixed(1) + ' s per frame');
  if (report.frames_failed) parts.push(report.frames_failed + ' failed');
  return parts.join(' · ');
}

const rows = computed((): Row[] => {
  const list: Row[] = [];
  if (view.value === 'proposers') {
    for (const item of proposers.value?.proposers ?? []) {
      const report = proposers.value?.reports.find((entry) => entry.proposer === item.key && entry.partition === 'test');
      const speed = report?.speed_signs;
      list.push({id: 'proposer:' + item.key, name: item.label, when: report ? isoWhen(report.measured_at) + (report.limited ? ' · a few frames only' : '') : '',
        main: speed ? share(speed.read_right, speed.truth) : null,
        mainText: speed ? ofText(speed.read_right, speed.truth, 'speed signs proposed with the right number')
          : item.unavailable ? (item.key === 'ours' ? 'not whole yet: a part is not in use' : 'not installed') : notMeasured.value,
        second: report ? 'found ' + speed!.found + ' of ' + speed!.truth + ' · ' + perFrame(report.false_per_frame)
          + ' false proposals per frame · ' + report.seconds_per_frame.toFixed(1) + ' s per frame' : '',
        inUse: false, run: '', watch: null});
    }
  } else if (view.value === 'ai') {
    const payload = aiReviewers.value;
    const shownReports = new Set<AiReviewerReport>();
    for (const provider of payload?.providers ?? []) {
      const report = payload?.reports.find((entry) => entry.provider_id === provider.id && entry.model === provider.model) ?? null;
      if (report) shownReports.add(report);
      list.push(aiRow(provider.name, provider.model, provider.id, report, provider.is_reviewer));
    }
    // Measures of a model a provider no longer uses, or of a provider removed: still there to compare with.
    for (const report of payload?.reports ?? []) {
      if (!shownReports.has(report)) list.push(aiRow(report.provider_name, report.model, report.provider_id, report, false));
    }
  } else if (view.value === 'chain') {
    const versions = (versionsOf('classifier') as ClassifierModel[]).filter((version) => version.chain);
    // The cutter the chains are compared with: the one in use when a chain was measured with it, else the newest
    // chain's. A chain measured with another cutter found other boxes: its number is not comparable.
    const cutters = versions.map((version) => version.chain?.cutter_id ?? null);
    const inUseCutter = versionsOf('cutter').find((version) => version.in_use)?.id ?? null;
    const common = inUseCutter !== null && cutters.includes(inUseCutter) ? inUseCutter : cutters[0] ?? null;
    const commonSize = versions.find((version) => version.chain?.cutter_id === common)?.chain?.cutter_size ?? null;
    const apart = 'Measured with another cutter: not comparable with the rows of '
      + (commonSize ? 'the cutter ' + sizeText(commonSize) : common === inUseCutter ? 'the cutter in use' : 'the newest') + '.';
    for (const version of versions) {
      const chain = version.chain!;
      const main = share(chain.correct, chain.truth);
      const old = version.chain_old_reader;
      const oldMain = old ? share(old.correct, old.truth) : null;
      const other = common !== null && chain.cutter_id != null && chain.cutter_id !== common;
      list.push({id: version.id, name: 'Classifier',
        when: isoWhen(version.trained_at) + (chain.cutter_size ? ' · cutter ' + sizeText(chain.cutter_size) : '')
          + ' · at score ' + chain.score_threshold, main,
        mainText: ofText(chain.correct, chain.truth, 'found and read right'),
        second: perFrame(chain.surviving_false_per_frame) + ' false boxes per frame left, of '
          + perFrame(chain.false_per_frame) + ' from the cutter',
        inUse: version.in_use, run: version.run, watch: 'chain',
        vs: main === null || oldMain === null ? undefined : {text: points(main - oldMain) + ' vs the old reader (it keeps every false box)',
          kind: Math.abs(main - oldMain) < 0.005 ? 'same' : main > oldMain ? 'better' : 'worse'},
        apart: other ? apart : undefined});
    }
  } else if (view.value === 'cutter') {
    for (const version of versionsOf('cutter') as CutterModel[]) {
      const test = version.test;
      const car = version.kind === 'signnet' ? test?.car ?? null : null;
      if (car) {
        // A SignNet as the car reads it: its best 5 boxes a frame of score 0.3 or more; compared with SignNets only.
        const read = version.families === 'all' ? car.by_family?.speed ?? null : null;
        const truth = read ? read.truth : car.truth_boxes;
        list.push({id: version.id, name: version.label + (version.families === 'all' ? ' · all signs' : ' · speed signs'),
          when: isoWhen(version.trained_at), main: read ? share(read.found, read.truth) : share(car.found, car.truth_boxes),
          mainText: ofText(read ? read.found : car.found, truth, (read || version.families !== 'all' ? 'speed signs' : 'signs')
            + ' found as the car reads them (best ' + car.max_boxes + ' at ' + car.score_threshold + ')'),
          second: perFrame(car.false_per_frame) + ' false boxes per frame', inUse: version.in_use, run: version.run,
          watch: version.part, basis: 'car ' + (read ? 'speed' : 'all') + ' of ' + (truth ?? '?')});
        continue;
      }
      const speed = test?.by_family?.speed;
      list.push({id: version.id, name: (version.kind === 'signnet' ? version.label : 'Cutter ' + sizeText(version.input_size))
        + (version.families === 'all' ? ' · all signs' : ' · speed signs'),
        when: isoWhen(version.trained_at), main: speed ? share(speed.found, speed.truth) : share(test?.found, test?.truth_boxes),
        mainText: speed ? ofText(speed.found, speed.truth, 'speed signs found')
          : ofText(test?.found, test?.truth_boxes, version.families === 'all' ? 'signs found' : 'speed signs found'),
        second: test ? perFrame(test.boxes_per_frame) + ' boxes per frame' : '', inUse: version.in_use, run: version.run,
        watch: version.part, basis: speed ? 'speed of ' + speed.truth : 'all of ' + (test?.truth_boxes ?? '?')});
    }
  } else if (view.value === 'classifier') {
    const versions = versionsOf('classifier') as ClassifierModel[];
    for (const version of versions) {
      const test = version.test;
      const old = !version.class_scheme;
      // An old reader is played in the chain of the newest classifier run that measured it (its own run keeps no chain).
      const chained = old ? versions.find((other) => other.chain_old_reader
        && (other.chain_old_reader.classifier_id ?? version.id) === version.id) ?? null : null;
      list.push({id: version.id, name: old ? 'Old reader (no “not a sign”)' : 'Classifier', when: isoWhen(version.trained_at),
        main: test?.accuracy ?? null,
        mainText: test?.accuracy == null ? 'not measured' : pct(test.accuracy) + ' of ' + thousands(test.samples ?? 0) + ' signs read right',
        second: test?.rejection_rate == null ? (old ? 'cannot say “not a sign”' : '') : pct(test.rejection_rate) + ' of the false boxes thrown away',
        inUse: version.in_use, run: version.run, watch: old ? (chained ? 'chain-old-reader' : null) : 'chain', watchRun: chained?.run,
        // The old reader reads only signs, never "not a sign": its share is not the classifier's.
        basis: (old ? 'reader of ' : 'classifier of ') + (test?.samples ?? '?')});
    }
  } else {
    for (const version of versionsOf('decider') as DeciderModel[]) {
      const numbers = version.validation;
      list.push({id: version.id, name: version.label, when: isoWhen(version.trained_at),
        main: share(numbers?.other_road_found, numbers?.other_road_support),
        mainText: ofText(numbers?.other_road_found, numbers?.other_road_support, '“Another road” signs found'),
        second: numbers?.reading_accuracy == null ? '' : pct(numbers.reading_accuracy) + ' read right',
        inUse: version.in_use, run: version.run, watch: null, basis: 'decider of ' + (numbers?.other_road_support ?? '?')});
    }
  }
  return list;
});
/** The version the others are compared with: the one in use, else the best. */
const reference = computed(() => rows.value.find((row) => row.inUse)
  ?? [...rows.value].filter((row) => row.main !== null).sort((a, b) => (b.main ?? 0) - (a.main ?? 0))[0] ?? null);
/** Two rows are compared only when measured the same way on the same test (a chain: with the same cutter). */
const comparable = (row: Row, other: Row | null): boolean => other !== null && !row.apart && row.basis === other.basis;
const best = computed(() => Math.max(...rows.value.filter((row) => comparable(row, reference.value)).map((row) => row.main ?? -1)));
function versus(row: Row): {text: string; kind: string} {
  if (view.value === 'chain') return row.vs ?? {text: '', kind: ''};
  const ref0 = reference.value;
  if (row.main === null || !ref0 || ref0.main === null || ref0.id === row.id) return {text: '', kind: ''};
  if (!comparable(row, ref0)) {
    const carRead = (one: Row) => one.basis?.startsWith('car ') ?? false;
    return {text: row.basis?.startsWith('reader') ? 'another kind of network: not comparable'
      : carRead(row) !== carRead(ref0) ? 'judged another way (a SignNet as the car reads, a Detector at 0.1): not comparable'
        : 'measured on another test: not comparable', kind: 'same'};
  }
  const difference = row.main - ref0.main;
  return {text: points(difference) + ' vs ' + (ref0.inUse ? 'the one in use' : 'the best'),
    kind: Math.abs(difference) < 0.005 ? 'same' : difference > 0 ? 'better' : 'worse'};
}

const shown = computed(() => rows.value.find((row) => row.id === picked.value) ?? reference.value ?? rows.value[0] ?? null);
/** The proposer of the row chosen, its latest report and its Measure button. */
const shownProposer = computed((): Proposer | null => shown.value?.id.startsWith('proposer:') ? shown.value.id.slice(9) as Proposer : null);
const proposerReport = computed(() => proposers.value?.reports.find((entry) => entry.proposer === shownProposer.value && entry.partition === 'test') ?? null);
const proposerMissing = computed(() => proposers.value?.proposers.find((item) => item.key === shownProposer.value)?.unavailable ?? null);
const MEASURE: [ActionName, string][] = [['measure_proposer', 'Proposes the signs of the reviewed test frames as Prepare would, '
  + 'and compares them with the boxes you kept. Nothing changes in the reviews; minutes to an hour.']];
/** The row to choose once another view is shown (a view shown starts from its reference row). */
let pendingPick: string | null = null;
watch(view, () => { picked.value = pendingPick; pendingPick = null; });

/** The view of each role, for a version asked for by the address (the road type has none here). */
const ROLE_VIEW: Partial<Record<ModelRole, View>> = {cutter: 'cutter', classifier: 'classifier', decider: 'decider'};
/** The version “Open the run ›” of Models and Train asks for (?model=, else the newest version of ?run=): chosen in the
 * view of its role once the versions have come, then forgotten. */
function openWanted(): void {
  if (selectedModel.value === null && selectedRun.value === null) return;
  const versions = models.value?.roles.flatMap((role) => role.versions);
  if (versions === undefined) return;
  const version = versions.find((item) => item.id === selectedModel.value) ?? versions.find((item) => item.run === selectedRun.value) ?? null;
  selectedModel.value = null;
  selectedRun.value = null;
  const target = version === null ? undefined : ROLE_VIEW[version.role];
  if (version === null || target === undefined) return;
  if (view.value === target) {
    picked.value = version.id;
  } else {
    pendingPick = version.id;
    view.value = target;
  }
}
watch(models, openWanted, {immediate: true});

/** The AI reviewer of the row chosen: its measure, and its provider when it still uses that model (for Measure). */
const shownAi = computed((): {report: AiReviewerReport | null; provider: AiProvider | null} | null => {
  const id = shown.value?.id;
  if (!id?.startsWith('ai:')) return null;
  const payload = aiReviewers.value;
  const same = (providerId: string, model: string): boolean => 'ai:' + providerId + '/' + model === id;
  return {report: payload?.reports.find((entry) => same(entry.provider_id, entry.model)) ?? null,
    provider: payload?.providers.find((entry) => same(entry.id, entry.model)) ?? null};
});
const AI_MEASURE: [ActionName, string][] = [['measure_ai_reviewer', 'Asks the AI about up to 60 frames a person reviewed, '
  + 'the same ones for every model, and compares its answers with theirs. Nothing is saved; minutes to an hour.']];
/** "25 of 31", or "–" without any. */
function countText(found: number, truth: number): string {
  return truth ? found + ' of ' + truth : '–';
}

/** Each view's data has come (the versions of the networks, the proposers, the AI reviewers), or why not. */
const loaded = computed(() => (view.value === 'proposers' ? proposers.value !== null
  : view.value === 'ai' ? aiReviewers.value !== null : models.value !== null));
const loadError = computed(() => (view.value === 'proposers' ? proposersError.value
  : view.value === 'ai' ? aiError.value : modelsError.value));
const emptyText = computed(() => (view.value === 'proposers' ? 'No proposer on this server.'
  : view.value === 'ai' ? 'No AI provider yet: add one in Settings.' : 'Nothing measured yet: train one on the Train tab.'));

const runSummary = computed(() => runsPayload.value?.runs.find((run) => run.name === shown.value?.run) ?? null);
const {detail, loading, error} = useRunDetail(computed(() => runSummary.value?.name ?? null),
  computed(() => runSummary.value?.version ?? null));
const watchModels = computed((): WatchModel[] => {
  const key = shown.value?.watch;
  if (!key) return [];
  // The old reader in the chain of a classifier run: measured on the test only (runs.watch_file).
  if (shown.value?.watchRun) return [{model: key, label: 'Chain, old reader', kind: 'chain', partitions: ['test']}];
  const list = detail.value?.name === shown.value?.run ? detail.value?.watch ?? [] : [];
  return list.filter((item) => item.model === key);
});
/** The run of the report played, and its version (so the player asks again when it changes). */
const watchRun = computed(() => shown.value?.watchRun ?? shown.value?.run ?? '');
const watchVersion = computed(() => runsPayload.value?.runs.find((run) => run.name === watchRun.value)?.version ?? null);
/** Why there is nothing to play. */
const nothingToPlay = computed((): string => {
  if (view.value === 'decider') return 'Nothing to play: applicability is judged on its table.';
  if (shown.value?.watch === null) return 'Nothing to play yet: this reader is played in the chain of a classifier run, and none has measured it.';
  return 'No frames to play for this version: its run kept none.';
});
</script>

<template>
  <SectionHead tab="evaluate"/>

  <section v-if="measureJob" class="live-section" :class="{now: measureJob.state === 'running'}" aria-labelledby="measure-job-title">
    <h2 class="section-title" id="measure-job-title">{{ measureJob.state === 'running' ? 'Running' : 'Last measure failed' }}</h2>
    <div class="jp-list" aria-live="polite">
      <JobProgress :job="measureJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
    </div>
  </section>

  <section class="card ev-card" aria-labelledby="eval-title">
    <div class="ev-head">
      <h2 id="eval-title">What to judge</h2>
      <Segmented v-model="view" :options="VIEWS" label="Network to judge"/>
    </div>
    <p class="ev-goal">{{ GOAL[view] }}</p>
    <p v-if="!loaded" class="empty">{{ loadError || 'Loading…' }}</p>
    <p v-else-if="rows.length === 0" class="empty">{{ emptyText }} <a v-if="view === 'ai'" href="/account#ai-reviewer">Open Settings</a></p>
    <ul v-else class="ev-list" role="listbox" aria-label="Versions">
      <li v-for="row in rows" :key="row.id" role="option" :aria-selected="shown?.id === row.id"
          :class="['ev-row', {picked: shown?.id === row.id}]" tabindex="0"
          @click="picked = row.id" @keydown.enter="picked = row.id">
        <div class="ev-name"><strong>{{ row.name }}</strong><span v-if="row.inUse" class="ev-in-use">{{ row.badge ?? 'in use' }}</span>
          <span class="ev-when">{{ row.when }}</span></div>
        <div class="ev-main" :class="{top: row.main !== null && row.main === best && rows.length > 1}">{{ row.mainText }}</div>
        <div class="ev-second">{{ row.second }}<span v-if="versus(row).text" :class="['ev-vs', versus(row).kind]">{{ versus(row).text }}</span><span
          v-if="row.apart" class="ev-vs apart">{{ row.apart }}</span></div>
      </li>
    </ul>
  </section>

  <section v-if="shown && shownProposer" class="card ev-card" aria-labelledby="proposer-title">
    <h2 id="proposer-title">{{ shown.name }} <span class="ev-when">{{ shown.when }}</span></h2>
    <p class="ev-goal">{{ shown.mainText }}<template v-if="shown.second"> · {{ shown.second }}</template></p>
    <template v-if="proposerReport">
      <p class="ev-goal">On {{ thousands(proposerReport.frames) }} test frames. Every sign, speed signs and the others: found
        {{ proposerReport.every_sign.found }} of {{ proposerReport.every_sign.truth }}. Speed signs on the frames a person
        reviewed: {{ ofText(proposerReport.speed_signs_person.read_right, proposerReport.speed_signs_person.truth, 'with the right number') }}.</p>
      <ul class="ev-widths">
        <li v-for="item in proposerReport.by_width" :key="item.key"><span class="ev-when">{{ item.label }} wide</span>
          <strong>{{ item.truth ? item.found + ' of ' + item.truth : '–' }}</strong><span class="ev-when">found</span></li>
      </ul>
    </template>
    <p v-if="proposerMissing" class="ev-goal ev-missing">{{ proposerMissing }}</p>
    <ActionList v-else :items="MEASURE" :actions="props.status?.actions ?? null"
                @run="(name, state) => shownProposer && emit('run', name, state, undefined, {proposer: shownProposer})"/>
  </section>

  <section v-if="shown && shownAi" class="card ev-card" aria-labelledby="ai-title">
    <h2 id="ai-title">{{ shown.name }} <span class="ev-when">{{ shown.when }}</span></h2>
    <p class="ev-goal">{{ shown.mainText }}<template v-if="shown.second"> · {{ shown.second }}</template></p>
    <template v-if="shownAi.report">
      <ul class="ev-widths">
        <li><span class="ev-when">Speed signs found</span><strong>{{ countText(shownAi.report.speed_signs.found, shownAi.report.speed_signs.truth) }}</strong>
          <span class="ev-when">{{ pct(shownAi.report.speed_signs.found_share) }}</span></li>
        <li><span class="ev-when">Right kind</span><strong>{{ countText(shownAi.report.speed_signs.kind_right, shownAi.report.speed_signs.truth) }}</strong>
          <span class="ev-when">{{ pct(shownAi.report.speed_signs.kind_right_share) }}</span></li>
        <li><span class="ev-when">Right road</span><strong>{{ countText(shownAi.report.speed_signs.road_right, shownAi.report.speed_signs.truth) }}</strong>
          <span class="ev-when">{{ pct(shownAi.report.speed_signs.road_right_share) }}</span></li>
        <li><span class="ev-when">Other signs found</span><strong>{{ countText(shownAi.report.other_signs.found, shownAi.report.other_signs.truth) }}</strong>
          <span class="ev-when">{{ countText(shownAi.report.other_signs.kind_right, shownAi.report.other_signs.truth) }} right kind</span></li>
        <li><span class="ev-when">Every sign found</span><strong>{{ countText(shownAi.report.every_sign.found, shownAi.report.every_sign.truth) }}</strong>
          <span class="ev-when">{{ pct(shownAi.report.every_sign.found_share) }}</span></li>
        <li><span class="ev-when">No sign, left empty</span><strong>{{ countText(shownAi.report.no_sign.agreed, shownAi.report.no_sign.truth) }}</strong>
          <span class="ev-when">{{ shownAi.report.no_sign.said_no_sign_wrongly }} said “no sign” wrongly</span></li>
        <li><span class="ev-when">AI boxes per frame</span><strong>{{ perFrame(shownAi.report.ai_boxes_per_frame) }}</strong>
          <span class="ev-when">{{ perFrame(shownAi.report.false_per_frame) }} false</span></li>
        <li><span class="ev-when">Seconds per frame</span><strong>{{ shownAi.report.seconds_per_frame === null ? '–' : shownAi.report.seconds_per_frame.toFixed(1) }}</strong>
          <span class="ev-when">{{ shownAi.report.frames_failed }} failed</span></li>
      </ul>
      <p class="ev-goal">On {{ thousands(shownAi.report.frames_measured) }} of {{ thousands(shownAi.report.frames) }} frames a person reviewed
        (the same {{ shownAi.report.frames }} for every model: sample <code>{{ shownAi.report.sample_sha256.slice(0, 10) }}</code>)<template
          v-if="shownAi.report.frames_incomplete">; on {{ shownAi.report.frames_incomplete }} it did not judge every box</template>.</p>
      <details v-if="shownAi.report.failures.length" class="ev-failures">
        <summary>{{ shownAi.report.frames_failed }} frames without an answer</summary>
        <ul><li v-for="(failure, i) in shownAi.report.failures" :key="i"><code>{{ failure.frame_key }}</code> {{ failure.reason }}: {{ failure.message }}</li></ul>
      </details>
    </template>
    <p v-if="!shownAi.provider" class="ev-goal ev-missing">This model is no longer in Settings: only its last measure is left.</p>
    <ActionList v-else :items="AI_MEASURE" :actions="props.status?.actions ?? null"
                @run="(name, state) => shownAi?.provider && emit('run', name, state, undefined, {provider: shownAi.provider.id})"/>
  </section>

  <template v-if="shown && !shownProposer && !shownAi">
    <section class="card ev-card" aria-labelledby="watch-title">
      <h2 id="watch-title">{{ shown.name }} <span class="ev-when">{{ shown.when }}</span></h2>
      <p class="ev-goal">{{ shown.mainText }}<template v-if="shown.second"> · {{ shown.second }}</template></p>
      <WatchSection v-if="watchModels.length" :key="shown.id" :run="watchRun" :models="watchModels" :version="watchVersion"/>
      <p v-else-if="loading" class="empty">Loading the frames…</p>
      <p v-else class="empty">{{ nothingToPlay }}</p>
    </section>
    <details v-if="runSummary" class="card ev-more">
      <summary>All the numbers of run {{ shown.run }}</summary>
      <RunDetailView :key="runSummary.name" :summary="runSummary" :detail="detail" :loading="loading" :error="error" no-watch/>
    </details>
  </template>
</template>
