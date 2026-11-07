// The Overview as the loop of the work: new drives are prepared, their signs reviewed, the networks trained on the
// reviews, and the ones in use published to the car. One row each, from the data of the status (never the sentence of
// route_status): how much waits there, in one line, and the one thing to do about it; its work running in place of
// its button. The rows with work waiting are marked; the first of them has the main button.
import {plural, reviewPercent, sentence, thousands} from '../common/format';
import type {ActionName, ActionState, Job, PanelStatus, RouteState} from '../common/panel-types';
import type {IconName} from '../components/icons/shapes';
import {networkText, proposerText} from './proposer-words';
import {reviewSummary} from './review-routes';

export type LoopKey = 'drives' | 'review' | 'train' | 'car';

export interface LoopRow {
  key: LoopKey;
  /** The stop of the loop, and its page. */
  label: string;
  href: string;
  /** The colour (k-<kind> in the page CSS). */
  kind: 'prepare' | 'review' | 'train' | 'done';
  /** How much waits, in a few words ("4 drives to prepare"), and the line under it. */
  big: string;
  note: string;
  /** A second line in orange: why its work cannot run as chosen. */
  warn: string;
  /** A link after the note ("Change" who proposes), or null. */
  more: {href: string; label: string} | null;
  /** Work waits for you here. */
  waiting: boolean;
  /** Its work is done: a tick when it has no button. */
  ok: boolean;
  /** Its work running now ("Prepare new drives · 42%"), in place of the button; '' when none runs. */
  running: string;
  /** The button: an action of the panel, or a link. */
  action: {name: ActionName; state: ActionState; label: string} | {href: string; label: string; icon: IconName} | null;
  /** The main one: the first row with work waiting, nothing of it running. */
  primary: boolean;
}

const TO_PREPARE: RouteState[] = ['collegare', 'estrarre', 'preannotare'];
/** The jobs of each stop: one of them running takes the place of the button. */
const ROW_ACTIONS: Record<LoopKey, ActionName[]> = {
  drives: ['fetch', 'prepare', 'prepare_again', 'road_types'],
  review: ['ai_review'],
  train: ['dataset', 'train_cutter', 'train_classifier', 'train_decider', 'train_road_type', 'train_brain', 'train', 'train_continue', 'train_resume'],
  car: ['send_to_car', 'try_on_comma', 'comma_restore'],
};
/** The terminal work of each stop, by its group (nnslr_tools.panel.ACTIVITIES). */
const ROW_GROUPS: Partial<Record<LoopKey, string[]>> = {drives: ['fetch', 'prepare'], train: ['train'], car: ['publish']};
const CAR_ROLES = ['cutter', 'classifier', 'road_type', 'brain'] as const;

const wholePercent = (percent: number | null): string =>
  percent === null ? '' : Math.floor(Math.max(0, Math.min(100, percent))) + '%';

/** "Prepare new drives · 42% · 2 of 5 drives ready" for a job, from its progress. */
function runningText(job: Job): string {
  return [job.label, wholePercent(job.progress.percent), job.progress.detail].filter(Boolean).join(' · ');
}

function running(status: PanelStatus, key: LoopKey): string {
  const job = status.jobs.find((item) => item.state === 'running' && ROW_ACTIONS[key].includes(item.action));
  if (key === 'train' && status.pipeline?.state === 'running') {
    const steps = status.pipeline.steps;
    const index = steps.findIndex((step) => step.state === 'running');
    const at = index < 0 ? '' : ': ' + steps[index]!.label + ', ' + (index + 1) + ' of ' + steps.length;
    return ['Train all' + at, job ? wholePercent(job.progress.percent) : ''].filter(Boolean).join(' · ');
  }
  if (job) return runningText(job);
  const activity = status.activities.find((item) => ROW_GROUPS[key]?.includes(item.group));
  return activity ? activity.label.charAt(0).toUpperCase() + activity.label.slice(1) + ', started from a terminal' : '';
}

function drives(status: PanelStatus): Omit<LoopRow, 'primary'> {
  const count = status.routes.filter((row) => TO_PREPARE.includes(row.state)).length;
  const proposers = status.proposers;
  const chosen = proposers?.chosen ?? null;
  const missing = chosen === 'ours' ? proposers?.ours ?? '' : chosen === 'qwen' ? proposers?.qwen ?? proposers?.qwen_gpu ?? '' : '';
  // Prepared, but their road types not found yet (Data › Speed review, Find the road types): that is the work of the row until it is done.
  const roads = !count ? status.road_types_missing ?? 0 : 0;
  if (roads && status.actions.road_types) {
    return {
      key: 'drives', label: 'Drives', href: '/speed-review', kind: 'prepare',
      big: plural(roads, 'drive') + ' without road types',
      note: 'Prepared: the map still has to say the type of road of each picture.', warn: '',
      more: {href: '/speed-review', label: 'Speed review'}, waiting: true, ok: false, running: running(status, 'drives'),
      action: {name: 'road_types', state: status.actions.road_types, label: 'Find the road types'},
    };
  }
  // Nothing to prepare: the button downloads the new drives from the comma (also in the comma's panel, at the top).
  const name: ActionName = count ? 'prepare' : 'fetch';
  return {
    key: 'drives', label: 'Drives', href: '/prepare', kind: 'prepare',
    big: count ? plural(count, 'drive') + ' to prepare' : status.routes.length ? 'All prepared' : 'No drives yet',
    note: count ? (chosen ? 'Found by ' + proposerText(chosen, proposers?.networks) + '.' : 'Found by the best measured proposer.')
      : 'New drives come from the comma.',
    warn: count && missing ? sentence(missing) : '', more: count ? {href: '/prepare', label: 'Change'} : null,
    waiting: count > 0 || status.next_action === 'fetch', ok: !count && status.routes.length > 0, running: running(status, 'drives'),
    // "Download from the comma" is too long for a column of four: the row says where they come from.
    action: status.actions[name] ? {name, state: status.actions[name], label: count ? status.actions[name].label : 'Download'} : null,
  };
}

function review(status: PanelStatus): Omit<LoopRow, 'primary'> {
  const summary = reviewSummary(status);
  const left = status.routes.filter((row) => row.state === 'revisionare')
    .reduce((total, row) => total + Math.max(0, row.report_frames - row.confirmed), 0);
  const drivesLeft = summary.inProgress + summary.toStart;
  return {
    key: 'review', label: 'Sign review', href: '/routes', kind: 'review',
    big: !summary.frames ? 'Nothing to review yet' : left ? thousands(left) + ' frames to review' : 'All confirmed',
    note: summary.frames ? (drivesLeft ? 'In ' + plural(drivesLeft, 'drive') + ' · ' : '')
      + reviewPercent(summary.confirmed, summary.frames) + '% of ' + thousands(summary.frames) + ' frames confirmed' : '',
    warn: '', more: null, waiting: left > 0, ok: summary.frames > 0 && !left, running: running(status, 'review'),
    action: summary.resume ? {href: summary.resume.href, label: summary.resume.started ? 'Resume review' : 'Start review',
      icon: 'resume'} : null,
  };
}

function train(status: PanelStatus): Omit<LoopRow, 'primary'> {
  const fresh = status.routes.reduce((total, row) => total + row.new_reviews, 0);
  const objects = status.dataset?.objects ?? status.routes.reduce((total, row) => total + row.objects, 0);
  const enough = status.reviewed_routes >= status.min_routes;
  return {
    key: 'train', label: 'Train', href: '/train', kind: 'train',
    big: !enough ? status.reviewed_routes + ' of ' + status.min_routes + ' drives reviewed'
      : fresh ? plural(fresh, 'new review') : 'Dataset up to date',
    note: !enough ? 'Training needs ' + status.min_routes + ' reviewed drives.'
      : (fresh ? 'Not in the dataset yet: Train all imports them first. ' : '') + thousands(objects) + ' objects in the dataset.',
    warn: '', more: null, waiting: enough && fresh > 0, ok: enough && !fresh, running: running(status, 'train'),
    action: enough ? {href: '/train#pipeline', label: 'Train all', icon: 'train'} : null,
  };
}

function car(status: PanelStatus): Omit<LoopRow, 'primary'> {
  const summary = status.car;
  const base = {key: 'car' as const, label: 'Car', href: '/publish', kind: 'done' as const, more: null, running: running(status, 'car')};
  if (!summary) return {...base, big: '–', note: '', warn: '', waiting: false, ok: false, action: null};
  const differ = CAR_ROLES.filter((role) => (summary.in_use[role] ?? null) !== (summary.on_car?.[role] ?? null));
  const publish = {href: '/publish', label: summary.build_pending ? 'Finish publishing' : 'Publish', icon: 'upload' as const};
  const inUse = CAR_ROLES.filter((role) => summary.in_use[role]).map((role) => networkText(role, summary.in_use[role]!));
  if (summary.problem) {
    return {...base, big: 'Cannot read', note: '', warn: sentence(summary.problem), waiting: false, ok: false, action: null};
  }
  if (summary.build_pending) {
    return {...base, big: 'Publishing not finished', note: 'psa-torque-sunny-testing is still to build: the comma finds nothing new yet.',
      warn: '', waiting: true, ok: false, action: publish};
  }
  if (!differ.length) {
    return {...base, big: summary.on_car ? 'Branch ready' : 'Nothing in use',
      note: inUse.length ? 'The branch has the networks in use: ' + inUse.join(' · ') + '.' : '', warn: '',
      waiting: false, ok: summary.on_car !== null, action: null};
  }
  // "classifier, 30 Oct 11:22 (the car has 29 Oct 09:00)"; a network the car has and none in use: "scene: none in use".
  const words = differ.map((role) => {
    const used = summary.in_use[role];
    const there = summary.on_car?.[role];
    if (!used) return networkText(role, there!).split(',')[0] + ': none in use';
    return networkText(role, used) + (there ? ' (the car has ' + networkText(role, there).split(', ').pop() + ')' : '');
  });
  return {...base, big: summary.on_car ? plural(differ.length, 'network') + ' to publish' : 'Not published yet',
    note: 'Not in the branch yet: ' + words.join(' · ') + '.', warn: '', waiting: true, ok: false, action: publish};
}

/** The four rows; null before the first status. */
export function loopRows(status: PanelStatus | null): LoopRow[] | null {
  if (!status) return null;
  const rows = [drives(status), review(status), train(status), car(status)];
  const first = rows.find((row) => row.waiting && !row.running && row.action);
  return rows.map((row) => ({...row, primary: row === first}));
}
