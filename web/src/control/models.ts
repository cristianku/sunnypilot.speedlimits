// The Models tab: the three networks of the pipeline (cutter, classifier, decider) and the road type (docs/NEURAL_NETWORK.md), every
// version trained on this server (GET /api/panel/models 10 s after each answer while the tab is open), the one in use for each
// role (POST /api/panel/models/active, asked first in the confirm dialog), and the words the tab shows.
import {ref, shallowRef} from 'vue';
import {api, messageOf, post} from '../common/api';
import {isoWhen, thousands} from '../common/format';
import type {ActionState, ActiveChosen, ActiveModel, BrainNumbers, ModelRole, ModelVersion, ModelsPayload} from '../common/panel-types';
import {askConfirm, type ConfirmOptions} from '../composables/confirm';
import {toast} from '../composables/toast';
import {chooseRun, pct} from './evaluate';
import {goTo} from './tabs';
import {usePoll} from '../composables/poll';

const INTERVAL_MS = 10000;

/** What each role does, what plays it today, the words when none is trained yet, and what uses the one in use
 * (Prepare's “Our networks” proposer, nnslr_tools.preannotate.our_networks; the classifier's training, which learns
 * from the false boxes of the cutter in use and is measured with it, scripts/run_classifier.sh). */
export const ROLE_TEXT: Record<ModelRole, {what: string; today: string; empty: string; emptyMore: string; uses: string}> = {
  cutter: {
    what: 'Finds every box that could be a sign. It must miss nothing: the classifier drops its false boxes. '
      + 'The cutter is SignNet, the same network the comma runs; the old Detectors are kept to compare.',
    today: 'Today: SignNet, judged on the speed signs as the car reads them: its 5 best boxes of each frame of score 0.3 or more.',
    empty: 'No cutter trained yet: train a SignNet, then choose it here.',
    emptyMore: '“Train the cutter” on the Train tab; its SignNet appears here.',
    uses: 'Prepare’s “Our networks” finds the signs with it. The next “Train the classifier” learns from its '
      + 'false boxes and is measured with it.',
  },
  classifier: {
    what: 'Says what each box is: a speed value, end of limit, another sign, or not a sign.',
    today: 'Today: the reader, judged on right answers per sign; the classifier with “not a sign” also on the false boxes '
      + 'it rejects and on the chain end to end.',
    empty: 'No classifier trained yet: the reader of a training run plays this role.',
    emptyMore: 'Start a training, or “Train the classifier”, on the Train tab; it appears here.',
    uses: 'Prepare’s “Our networks” reads the signs with it.',
  },
  decider: {
    what: 'Applicability: is this sign for our road? Our road, another road, or it abstains.',
    today: 'Today: only the prototypes of experiment B.',
    empty: 'No applicability network trained yet: experiment B is the first prototype.',
    emptyMore: 'Its variants appear here once scripts/run_experiment_b.sh has trained them.',
    uses: 'Nothing uses the applicability network yet: the choice is only noted for later.',
  },
  road_type: {
    what: 'Says from the camera what road the car is on, once a second: town, zone 30, rural, expressway, motorway, ramp, '
      + 'the merge lane after it, a car park or a garage. '
      + 'With memory it remembers the seconds before, and how long ago a motorway sign was read, so the 60 of a ramp ends '
      + 'where the merge lane starts, without a sign.',
    today: 'Today: prototypes, with memory and the sign, without the sign and without memory, judged on the test drives.',
    empty: 'No scene network trained yet.',
    emptyMore: '“Train the scene network” on the Train tab; its two networks appear here.',
    uses: 'The comma runs the scene network in use once Publish puts it in its software.',
  },
  brain: {
    what: 'Says the speed to hold, once a second, from the camera, the signs read, the car ahead, the hour and the country. '
      + 'On the comma it is the vision limit: the rules of the signs and the scene network no longer run.',
    today: 'Today: trained on the speeds of Speed review, judged on the frames of drives it never learned.',
    empty: 'No brain trained yet.',
    emptyMore: 'nnslr train-brain on the server; it appears here.',
    uses: 'The comma runs the brain in use once Publish puts it in its software, in place of the scene network.',
  },
};

/** The models, asked 10 s after each answer while the tab is shown. */
export function useModels() {
  const payload = shallowRef<ModelsPayload | null>(null);
  const error = ref('');
  async function ask(): Promise<void> {
    try {
      payload.value = await api<ModelsPayload>('/api/panel/models');
      error.value = '';
    } catch (problem) {
      error.value = messageOf(problem);
    }
  }
  return {payload, error, refresh: usePoll(ask, INTERVAL_MS)};
}

/** "2026-10-24_1452 · Detector 320 × 320": a version in one line. */
export function versionName(version: ModelVersion): string {
  return version.run + ' · ' + version.label;
}

/** A role's name inside a sentence: “scene network” for the role road_type (its label, “Scene”, is a name), else
 * its label in small letters. */
export function roleWords(version: ModelVersion, roleLabel: string): string {
  return version.role === 'road_type' ? 'scene network' : roleLabel.toLowerCase();
}

/** Asks, then makes a version the one in use for its role; true when it is. */
export async function putInUse(version: ModelVersion, roleLabel: string): Promise<boolean> {
  const role = roleWords(version, roleLabel);
  const ok = await askConfirm({
    title: 'Use this ' + role + '?', action: 'Use this one',
    text: versionName(version) + ' becomes the ' + role + ' in use on this server. ' + ROLE_TEXT[version.role].uses
      + ' Nothing is copied or sent to the comma.',
  });
  if (!ok) return false;
  try {
    await post<ActiveChosen>('/api/panel/models/active', {role: version.role, id: version.id});
    toast(roleLabel + ' in use: ' + versionName(version));
    return true;
  } catch (problem) {
    toast(messageOf(problem), true);
    return false;
  }
}

/** The SignNet versions and the comma: the run of the SignNet being published to the car ('' when the run is not known
 * yet; null when none is), and “Try on the comma now”. */
export interface CarState {
  sending: string | null;
  /** “Try on the comma now”: its state, the run being copied ('' while not known; null: none), the run on trial. */
  tryState?: ActionState | null;
  trying?: string | null;
  onTrial?: string | null;
}

/** The question before Publish (Publish tab, scripts/send_to_car.sh): which SignNet, which classifier and which road
 * type (the ones in use) the comma gets, what the job prepares, publishes and builds, and what is left to do on the comma. */
export function publishQuestion(signnet: ModelVersion, classifier: ModelVersion | null,
                                roadType: ModelVersion | null = null, brain: ModelVersion | null = null): ConfirmOptions {
  const reader = classifier ? classifier.label + ' ' + (classifier.trained_at ? isoWhen(classifier.trained_at) : classifier.run)
    : 'the classifier in use';
  const scene = brain
    ? ' and the brain of ' + brain.run + ' to say the speed to hold, in place of the rules and the scene network'
    : roadType
    ? ' and the scene network ' + roadType.label + ' of ' + roadType.run + ' to read the scene where no sign gives the limit '
      + '(a garage 10, a car park 20, a town 50)'
    : '; no scene network is in use, so it reads the signs alone';
  return {
    title: 'Publish the networks in use to the car?', action: 'Publish',
    text: 'The comma gets ' + signnet.label + ' of ' + signnet.run + ' to find the signs, ' + reader
      + ' to read them' + scene + '. The package is prepared and checked, published in speed-limits-models, the '
      + 'sunnypilot speed-limits branch moves to it and psa-torque-sunny-testing is built again: a few minutes. Then, on '
      + 'the comma: Settings, Software, Check for updates, and install.',
  };
}

/** The question before "Try on the comma now" (scripts/try_on_comma.sh): the same two networks copied straight onto the
 * comma, checked first with the comma's own code; no publication, no build. */
export function tryQuestion(signnet: ModelVersion, classifier: ModelVersion | null): ConfirmOptions {
  const reader = classifier ? classifier.label + ' ' + (classifier.trained_at ? isoWhen(classifier.trained_at) : classifier.run)
    : 'the classifier in use';
  return {
    title: 'Try this SignNet on the comma now?', action: 'Try on the comma',
    text: 'The networks go onto the comma now: ' + signnet.label + ' of ' + signnet.run + ' and ' + reader
      + ' (the classifier in use), checked first with the comma\'s own code. Parked, the comma prepares them within a '
      + 'few minutes; they are used from the next drive. Nothing is published or built: the comma\'s next software '
      + 'update puts back the build\'s networks.',
  };
}

/** The question before "Back to the build's networks" (scripts/try_on_comma.sh --restore). */
export function restoreQuestion(): ConfirmOptions {
  return {
    title: 'Back to the build\'s networks?', action: 'Put them back',
    text: 'The comma gets back the networks of the build it runs, from the next drive. Nothing else on the comma is touched.',
  };
}

/** Who chose the model in use, and when: "chosen 24 Oct 16:40 by admin". */
export function chosenText(active: ActiveModel): string {
  const when = isoWhen(active.at);
  const who = active.via === 'sign-in' && active.who ? 'by ' + active.who
    : active.via === 'token' ? 'by an agent with the access token'
      : active.via === 'local' ? 'on this computer' : '';
  return ['chosen', when, who].filter(Boolean).join(' ');
}

/** The address of a version on the Evaluate tab: its run and its id. */
export function evaluateHref(version: ModelVersion): string {
  return '/evaluate?run=' + encodeURIComponent(version.run) + '&model=' + encodeURIComponent(version.id);
}

/** Opens a version on the Evaluate tab, chosen in the view of its role (a click with a modifier key opens the link as
 * usual). */
export function openRun(event: MouseEvent, version: ModelVersion): void {
  if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  goTo('evaluate');
  chooseRun(version.run, version.id);
}

/** "320x320" -> "320 × 320". */
export function sizeText(size: string): string {
  return size.replace('x', ' × ');
}

/** 1.234 -> "1.23"; null -> "–". */
export function perFrame(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '–';
  return value >= 100 ? value.toFixed(0) : value >= 10 ? value.toFixed(1) : value.toFixed(2);
}

/** "8 of 20 epochs", "8", "–". */
export function epochText(best: number | null, epochs: number | null): string {
  if (best === null) return '–';
  return epochs ? best + ' of ' + epochs : String(best);
}

/** Where the weights started. */
export function initText(initialization: string | null): string {
  if (initialization === 'imagenet1k_v1') return 'ImageNet';
  if (initialization === 'random') return 'random weights';
  if (initialization === 'mapillary_pretraining') return 'Mapillary pretraining';
  if (initialization === 'checkpoint') return 'another classifier';
  if (initialization === 'start') return 'an earlier run';
  return initialization ?? '–';
}

/** A headline number of a version: what it counts (and a short name for a table header), the share (0-1, null: not
 * measured or not trained for it), the count it is of ("145 of 208") and a line under it. */
export interface Metric {
  key: string;
  label: string;
  short?: string;
  value: number | null;
  count: string;
  detail: string;
  /** A second, smaller line under detail (the cutter as the car reads it: the strict count at IoU 0.5). */
  note?: string;
}

/** The numbers that judge a version, side by side (MetricTiles, the Models table): the cutter: a SignNet's speed signs
 * as the car reads them (its best 5 boxes a frame of score 0.3 or more), the speed signs and every sign found on the
 * test at its score (a detector passes on a couple of hundred boxes a frame at that score, SignNet a few), and the
 * speed signs among its 5 best boxes a frame at any score; the classifier with “not a sign”: the speed-sign crops and every sign
 * crop read right (the kind of an other sign aside), and the chain with the cutter; an old reader: its crops read
 * right; the decider: the “Another” signs found on validation and on the test; the road type: the merges where the
 * ramp's limit ends in time by the comma's rule (since the second version), the motorway said right in the first minute
 * after each ramp, the intersections passed (the junction head, since the third version: F1 of "passed" within 2 s, and
 * the false ones an hour, each a limit lifted too early), the mean F1 of the road types and the town's F1, on the test. */
export function metricsOf(version: ModelVersion): Metric[] {
  const of = (part: number | null | undefined, whole: number | null | undefined): string =>
    whole ? thousands(part ?? 0) + ' of ' + thousands(whole) : '';
  if (version.role === 'cutter') {
    const cut = version.test ?? version.validation ?? null;
    if (cut === null) return [];
    const every: Metric = {key: 'every', label: 'Every sign', value: cut.recall, count: of(cut.found, cut.truth_boxes),
      detail: cut.truth_boxes === null ? '' : of(cut.found, cut.truth_boxes) + ' found'};
    const boxes = cut.budget?.boxes ?? 5;
    const kept = cut.budget ? (version.families === 'all' ? cut.budget.by_family?.['speed'] ?? null
      : {truth: cut.budget.truth_boxes, found: cut.budget.found, recall: cut.budget.recall}) : null;
    const budgetLabel = 'Speed signs, best ' + boxes + ' boxes, any score';
    const budget: Metric = kept
      ? {key: 'budget', label: budgetLabel, short: 'Best ' + boxes + ', any score', value: kept.recall,
        count: of(kept.found, kept.truth), detail: of(kept.found, kept.truth) + ' in the ' + boxes + ' best boxes of each frame, whatever their score'}
      : {key: 'budget', label: budgetLabel, short: 'Best ' + boxes + ', any score', value: null,
        count: 'not measured', detail: 'Its report did not keep the boxes of each frame'};
    // A SignNet as the car reads it: its best 5 boxes a frame of score 0.3 or more (missing from an older server). With
    // the lenient count (car_centre: a box on the sign counts, even a pixel or two tight) as the main number when it was
    // measured, the strict one (IoU 0.5) in small.
    const metrics: Metric[] = [];
    if (version.kind === 'signnet') {
      const strict = cut.car ?? null;
      const car = cut.car_centre ?? strict;
      const rule = 'best ' + (car?.max_boxes ?? 5) + ' at ' + (car?.score_threshold ?? 0.3);
      const speedOf = (numbers: typeof strict) => numbers ? (version.families === 'all' ? numbers.by_family?.['speed'] ?? null
        : {truth: numbers.truth_boxes ?? 0, found: numbers.found ?? 0, recall: numbers.recall}) : null;
      const read = speedOf(car);
      const strictRead = cut.car_centre ? speedOf(strict) : null;
      metrics.push(read
        ? {key: 'car', label: 'Speed signs as the car reads them (' + rule + ')', short: 'As the car reads', value: read.recall,
          count: of(read.found, read.truth), detail: of(read.found, read.truth) + ' found, '
            + perFrame(car?.false_per_frame) + ' false boxes per frame',
          ...(strictRead ? {note: 'IoU 0.5: ' + pct(strictRead.recall) + ' found, ' + perFrame(strict?.false_per_frame) + ' false a frame'} : {})}
        : {key: 'car', label: 'Speed signs as the car reads them (' + rule + ')', short: 'As the car reads', value: null,
          count: 'not measured', detail: car === null && cut.car === undefined ? 'Not sent by this server' : 'Its report did not measure it'});
    }
    const speed = version.families === 'all' ? cut.by_family?.['speed'] ?? null : null;
    if (speed) {
      return [...metrics, {key: 'speed', label: 'Speed signs', value: speed.recall, count: of(speed.found, speed.truth),
        detail: of(speed.found, speed.truth) + ' found'}, every, budget];
    }
    if (version.families === 'all') return [...metrics, every, budget];
    return [...metrics, {...every, key: 'speed', label: 'Speed signs'},
      {key: 'every', label: 'Every sign', value: null, count: 'speed only', detail: 'Trained on speed signs only'}, budget];
  }
  if (version.role === 'classifier') {
    const test = version.test;
    const metrics: Metric[] = [];
    if (version.class_scheme === undefined) {
      if (test) metrics.push({key: 'every', label: 'Read right', value: test.accuracy, count: of(test.accuracy === null || test.samples === null
        ? null : Math.round(test.accuracy * test.samples), test.samples), detail: 'right answers on ' + thousands(test.samples ?? 0) + ' test crops'});
      return metrics;
    }
    if (test) {
      metrics.push({key: 'speed', label: 'Speed signs', value: test.speed_accuracy ?? null, count: of(test.speed_correct, test.speed_samples),
        detail: test.speed_samples ? of(test.speed_correct, test.speed_samples) + ' read right' : 'Not counted apart in this report'});
      const every = test.family_accuracy ?? test.accuracy;
      const right = every === null || test.samples === null ? null : Math.round(every * test.samples);
      metrics.push({key: 'every', label: 'Every sign', value: every, count: of(right, test.samples), detail: of(right, test.samples) + ' read right'});
    }
    const chain = version.chain;
    if (chain) {
      metrics.push({key: 'chain', label: 'With the cutter' + (chain.cutter_size ? ' ' + sizeText(chain.cutter_size) : ''), value: chain.recall,
        count: of(chain.correct, chain.truth), detail: chain.truth ? of(chain.correct, chain.truth) + ' speed signs, end to end' : 'speed signs, end to end'});
    }
    return metrics;
  }
  if (version.role === 'brain') {
    const speed = (label: string, key: string, numbers: BrainNumbers | null): Metric => ({key, label, value: numbers?.right ?? null,
      count: numbers?.frames ? thousands(numbers.frames) + ' frames' : '',
      detail: numbers?.mean_error_kph == null ? '' : numbers.mean_error_kph.toFixed(1) + ' km/h off on average'});
    return version.test === null ? [] : [speed('Right on the test', 'test', version.test),
      speed('On validation', 'validation', version.validation)];
  }
  if (version.role === 'road_type') {
    const test = version.test;
    if (test === null) return [];
    const merge = test.after_merge;
    const town = test.classes['town'] ?? null;
    const release = version.release?.test ?? null;
    const late = release ? release.late + release.not_held : 0;
    const junction = version.junctions?.test ?? null;
    const needed = version.junctions?.needed ?? null;
    const hourly = (value: number | null): string => value === null ? '–' : value < 10 ? value.toFixed(1) : String(Math.round(value));
    return [
      ...(release ? [{key: 'release', label: 'Ramp limit ends in time', short: 'Ends in time',
        value: release.merges ? release.in_time / release.merges : null,
        count: release.merges ? of(release.in_time, release.merges) + (release.merges === 1 ? ' merge' : ' merges') : 'no merge',
        detail: release.merges ? 'The comma’s rule on the ' + release.merges + ' merges of the test: '
          + (release.on_ramp ? release.on_ramp + ' ended on the ramp, ' : 'none on the ramp, ')
          + (late ? late + ' late or never' : 'none late')
          + (release.median_release === null ? '' : ', median ' + release.median_release + ' s after the merge')
          : 'The test has no merge'}] : []),
      {key: 'merge', label: 'Motorway after a ramp', short: 'After a ramp', value: merge.merges ? merge.accuracy : null,
        count: merge.merges ? merge.seconds + ' s, ' + merge.merges + (merge.merges === 1 ? ' ramp' : ' ramps') : 'no ramp',
        detail: merge.merges ? 'The first minute on the motorway after each of the ' + merge.merges + ' ramps of the test'
          : 'The test has no ramp onto the motorway'},
      ...(junction ? [{key: 'junction', label: 'Junctions passed', short: 'Junctions', value: junction.f1_2s,
        count: hourly(junction.false_per_hour) + ' false/h',
        detail: 'F1 of “passed” within 2 s on the ' + (junction.passed ?? 0) + ' intersections of the test; '
          + hourly(junction.false_per_hour) + ' false an hour (town ' + hourly(junction.town_per_hour) + ', outside '
          + hourly(junction.outside_per_hour) + '): each lifts a limit too early'
          + (needed?.needed ? '; ' + needed.caught_2s + ' of the ' + needed.needed + ' after a number sign caught' : '')}] : []),
      {key: 'macro', label: 'Road types', value: test.macro_f1, count: test.frames === null ? '' : thousands(test.frames) + ' s',
        detail: 'Mean F1 of the road types the test has'},
      {key: 'town', label: 'Town', value: town?.f1 ?? null, count: town ? thousands(town.support) + ' s' : '',
        detail: 'F1 of the town on the test'},
    ];
  }
  const another = (numbers: typeof version.validation): Metric['count'] =>
    numbers?.other_road_support ? of(numbers.other_road_found, numbers.other_road_support) : '';
  return [
    {key: 'validation', label: '“Another” found', value: version.validation?.other_road_recall ?? null, count: another(version.validation),
      detail: another(version.validation) + ' on validation'},
    {key: 'test', label: 'On the test', value: version.test?.other_road_support ? version.test.other_road_recall : null,
      count: version.test?.other_road_support ? another(version.test) : 'no case', detail: version.test?.other_road_support ? another(version.test) + ' on the test' : 'The test has no “Another” sign yet'},
  ];
}
