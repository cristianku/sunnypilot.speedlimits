// The jobs of the control pages: their state in words, their times, which job a tab shows, the work started
// from a terminal, the GPUs they hold.
import {shortWhen} from '../common/format';
import type {ActionName, Activity, Gpu, Job, JobState, PanelStatus, SystemInfo} from '../common/panel-types';
import type {IconName} from '../components/icons/shapes';
import {cardLabel, gpuCards, machineNow} from './machine-history';

/** State of a job: label and pill class. */
export const JOB_STATE: Record<JobState, [label: string, kind: string]> = {
  running: ['running', 'run'], ok: ['done', 'ok'], failed: ['failed', 'bad'], interrupted: ['interrupted', 'warn'],
};

/** The group of each action (nnslr_tools.panel.ACTIONS): a terminal activity of the same group is that work. */
export const ACTION_GROUP: Record<ActionName, string> = {
  fetch: 'fetch', prepare: 'prepare', prepare_again: 'prepare', measure_proposer: 'train', setup_vlm: 'train', dataset: 'train', train: 'train', train_continue: 'train', train_resume: 'train',
  train_classifier: 'train', train_cutter: 'train', train_decider: 'train', train_brain: 'train', regression: 'train', update: 'update', ai_review: 'ai',
  measure_ai_reviewer: 'ai',
  networks_prepare: 'publish', networks_publish: 'publish', road_types: 'prepare', send_to_car: 'publish',
  try_on_comma: 'publish', comma_restore: 'publish', mapillary_locations: 'mapillary', mapillary_windows: 'mapillary',
  review_box: 'review_box', road_labeler_train: 'train', road_labeler_predict: 'train', sign_tracks: 'train',
  box_fit_train: 'box_fit',
};

/** The icon of each action, on its button. */
export const ACTION_ICON: Record<ActionName, IconName> = {
  fetch: 'download', prepare: 'prepare', prepare_again: 'resume', measure_proposer: 'train', setup_vlm: 'download', dataset: 'import', train: 'train', train_continue: 'train',
  train_resume: 'resume', train_classifier: 'train', train_cutter: 'train', train_decider: 'train', train_brain: 'train', regression: 'train',
  update: 'update', ai_review: 'ai',
  measure_ai_reviewer: 'ai', networks_prepare: 'package', networks_publish: 'upload', road_types: 'routes', send_to_car: 'upload',
  try_on_comma: 'upload', comma_restore: 'resume', mapillary_locations: 'routes', mapillary_windows: 'prepare',
  review_box: 'upload', road_labeler_train: 'train', road_labeler_predict: 'routes', sign_tracks: 'routes',
  box_fit_train: 'train',
};

/**
 * The tab that shows the job of each action, by its address: its card, the dot on the tab's link while it runs (also
 * in the phone menu), the link of "Running now" in the server's panel. The update has no tab: the page reloads.
 */
export const ACTION_TAB: Record<ActionName, string | null> = {
  fetch: '/prepare', prepare: '/prepare', prepare_again: '/prepare', setup_vlm: '/prepare',
  dataset: '/dataset', road_types: '/prepare', review_box: '/reviewers',
  train: '/train', train_continue: '/train', train_resume: '/train', train_classifier: '/train', train_brain: '/train',
  train_cutter: '/train', train_decider: '/train', ai_review: '/train', mapillary_locations: '/train', mapillary_windows: '/train',
  measure_proposer: '/evaluate', measure_ai_reviewer: '/evaluate', regression: '/regression',
  networks_prepare: '/publish', networks_publish: '/publish',
  send_to_car: '/publish', try_on_comma: '/models', comma_restore: '/models',
  road_labeler_train: '/models', road_labeler_predict: '/models', sign_tracks: '/models', box_fit_train: '/models',
  update: null,
};
const actionsOf = (tab: string): ActionName[] => (Object.keys(ACTION_TAB) as ActionName[]).filter((name) => ACTION_TAB[name] === tab);
/** The actions whose jobs the Prepare, Train and Evaluate tabs show; the map reading is on Prepare too, in a card of its
 * own (RoadTypesCard.vue) that shows its job. */
export const PREPARE_ACTIONS: ActionName[] = actionsOf('/prepare').filter((name) => name !== 'road_types');
export const TRAIN_ACTIONS: ActionName[] = actionsOf('/train');
export const EVALUATE_ACTIONS: ActionName[] = actionsOf('/evaluate');

/** The tab of work started from a terminal, by its group (nnslr_tools.panel.ACTIVITIES); a measure is Evaluate's, the
 * map reading is Prepare's like the rest of its group (in its own card there, RoadTypesCard.vue), the Mapillary positions
 * and windows Train's (its classifier and its cutter). */
export function activityTab(activity: Pick<Activity, 'group' | 'label'>): string | null {
  if (activity.group === 'mapillary') return '/train';
  if (activity.group === 'box_fit') return '/models';
  if (activity.label === 'networks on the comma' || activity.label === 'sign tracks') return '/models';
  if (activity.group === 'ai') return activity.label.includes('measure') ? '/evaluate' : '/train';
  if (activity.group === 'publish') return '/publish';
  return activity.group === 'fetch' || activity.group === 'prepare' ? '/prepare' : activity.group === 'train' ? '/train' : null;
}

/** The addresses of the tabs with work running: the running jobs of the site, and the work started from a terminal. */
export function busyTabs(jobs: readonly Pick<Job, 'action' | 'state'>[], activities: readonly Pick<Activity, 'group' | 'label'>[]): Set<string> {
  const tabs = new Set<string>();
  for (const job of jobs) {
    const tab = job.state === 'running' ? ACTION_TAB[job.action] : null;
    if (tab) tabs.add(tab);
  }
  for (const activity of activities) {
    const tab = activityTab(activity);
    if (tab) tabs.add(tab);
  }
  return tabs;
}
/** The trainings whose job shows its curves on the Train tab. */
export const CHARTED_ACTIONS: ActionName[] = ['train', 'train_continue', 'train_classifier', 'train_cutter', 'train_decider',
  'train_brain'];
/** The trainings of the whole pipeline (helpers/3_train.sh): a stopped one can be resumed on its run. */
export const PIPELINE_ACTIONS: ActionName[] = ['train', 'train_continue', 'train_resume'];

/**
 * When the job started. The server writes started_at with its offset when it has it; otherwise "started",
 * the server's clock without a zone, which on this server (the LXC) is UTC.
 */
export function startedAt(job: Job): Date | null {
  if (job.started_at) {
    const parsed = Date.parse(job.started_at);
    if (!Number.isNaN(parsed)) return new Date(parsed);
  }
  const match = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})/.exec(job.started);
  if (!match) return null;
  const [year = 0, month = 1, day = 1, hour = 0, minute = 0, second = 0] = match.slice(1).map(Number);
  return new Date(Date.UTC(year, month - 1, day, hour, minute, second));
}

const pad = (value: number): string => String(value).padStart(2, '0');

/** "18:55" today, otherwise "23 Oct 18:55", in the browser's time. */
export function startedWhen(job: Job): string {
  const start = startedAt(job);
  return start === null ? job.started : shortWhen(start);
}

/** "42 s", "12 min 03 s", "2 h 05 min". */
export function elapsedText(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  if (seconds < 60) return seconds + ' s';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return minutes + ' min ' + pad(seconds % 60) + ' s';
  return Math.floor(minutes / 60) + ' h ' + pad(minutes % 60) + ' min';
}

/** A rough remaining time, in whole minutes: "less than a minute", "40 min", "1 h 10 min". */
export function roughText(ms: number): string {
  const minutes = Math.round(ms / 60000);
  if (minutes < 1) return 'less than a minute';
  if (minutes < 60) return minutes + ' min';
  return Math.floor(minutes / 60) + ' h' + (minutes % 60 ? ' ' + (minutes % 60) + ' min' : '');
}

/** The job a tab shows for these actions: the running one, else the latest. */
/** "about 2 h 50 min", "about 12 min", "under a minute" (job_progress.duration_text). */
export function durationText(seconds: number): string {
  const minutes = Math.round(seconds / 60);
  if (minutes < 1) return 'under a minute';
  return 'about ' + (minutes >= 60 ? Math.floor(minutes / 60) + ' h ' + (minutes % 60) + ' min' : minutes + ' min');
}

export function jobOf(status: PanelStatus | null, actions: ActionName[]): Job | null {
  const jobs = (status?.jobs ?? []).filter((job) => actions.includes(job.action));
  return jobs.find((job) => job.state === 'running') ?? jobs[0] ?? null;
}

/** The work started from a terminal (or by another program) that is not a job of the site. A server that ties each
 * work to the panel job whose process session it is in says so (job: its id, null from a terminal), so a training from
 * a terminal shows beside a training of the site; an older one is read by the group of the site's running jobs. */
export function terminalActivities(status: PanelStatus | null, groups?: string[]): Activity[] {
  if (status === null) return [];
  const fromSite = new Set(status.jobs.filter((job) => job.state === 'running').map((job) => ACTION_GROUP[job.action]));
  // A process whose command line matches two labels (same pid) is shown once.
  const pids = new Set<number>();
  return status.activities.filter((activity) => {
    if (pids.has(activity.pid)) return false;
    pids.add(activity.pid);
    const ours = activity.job === undefined ? fromSite.has(activity.group) : activity.job !== null;
    return !ours && (groups === undefined || groups.includes(activity.group));
  });
}

/** The actions whose job uses the GPU (the proposers of Prepare and their measure, Qwen3-VL's test, the trainings, the
 * road labeler and the signs on every frame but for their dry runs). */
const GPU_ACTIONS: ActionName[] = ['prepare', 'prepare_again', 'measure_proposer', 'setup_vlm', 'train', 'train_continue',
  'train_resume', 'train_classifier', 'train_cutter', 'train_decider', 'train_brain', 'regression', 'road_labeler_train',
  'road_labeler_predict', 'sign_tracks'];
/** The same for work started from a terminal, by group ("other": the driving model of openpilot.distill). */
const GPU_GROUPS = ['prepare', 'train', 'other'];

/** Whether a running job or a work from a terminal may be on a card (its nvidia-smi index; undefined: the one card of
 * an older server): by the cards the server says it holds ([] none, null every card), else by its kind (usual). */
function mayBeOn(item: Pick<Job, 'gpus'>, index: number | undefined, usual: boolean): boolean {
  if (item.gpus === undefined) return usual;
  if (item.gpus === null) return true;
  return index === undefined ? item.gpus.length > 0 : item.gpus.includes(index);
}

/**
 * The name of each process on a GPU (card.process_list; no card: system.gpu, the processes of every card). In this
 * container nvidia-smi cannot name the host's processes ("[Not Found]"), the site's own jobs included: when one process
 * of the card has no name and one job of the site, or one work from a terminal, may be on it, the row is that one;
 * otherwise "A program the server cannot name". A job whose card the server does not say may be on any card: it names
 * a process only when no other card has one without a name.
 */
export function gpuProcessNames(status: PanelStatus | null, card?: Gpu | null): string[] {
  const list = (card ?? status?.system.gpu)?.process_list ?? [];
  const unnamed = (name: string): boolean => !name || name.startsWith('[');
  const index = card?.index;
  const label = (activity: Activity): string => (activity.label.startsWith('openpilot') ? activity.label
    : activity.label.charAt(0).toUpperCase() + activity.label.slice(1)) + ', from a terminal';
  const known = [
    ...(status?.jobs ?? []).filter((job) => job.state === 'running' && mayBeOn(job, index, GPU_ACTIONS.includes(job.action)))
      .map((job) => ({name: job.label, placed: Array.isArray(job.gpus)})),
    ...terminalActivities(status, GPU_GROUPS).filter((activity) => mayBeOn(activity, index, true))
      .map((activity) => ({name: label(activity), placed: Array.isArray(activity.gpus)})),
  ];
  const here = list.filter((process) => unnamed(process.name)).length;
  const everywhere = card ? gpuCards(status?.system).reduce((total, item) => total
    + item.process_list.filter((process) => unnamed(process.name)).length, 0) : here;
  const only = known.length === 1 ? known[0] : undefined;
  const guess = only && here === 1 && (only.placed || everywhere === 1) ? only.name : undefined;
  return list.map((process) => (unnamed(process.name) ? guess ?? 'A program the server cannot name' : process.name));
}

/** The cards a running job or a work from a terminal holds, for the chip beside it: "GPU 0", "GPU 0 + 1"; '' for none,
 * for a GPU job whose card the server cannot tell (it holds every card), for an ended job, and on a machine with one
 * card, where it says nothing. */
export function gpuChipText(item: Pick<Job, 'gpus'> & {state?: JobState}): string {
  if (item.state !== undefined && item.state !== 'running') return '';
  if (!item.gpus?.length || gpuCards(machineNow.value).length < 2) return '';
  return 'GPU ' + [...item.gpus].sort((a, b) => a - b).join(' + ');
}
/** The chip of gpuChipText: a small grey capsule after the line it sits in (the page's stylesheets are not the
 * components': the style comes with it). */
export const GPU_CHIP_STYLE = {display: 'inline-block', marginLeft: '6px', padding: '0 7px', borderRadius: '999px',
  background: 'var(--surface-2)', color: 'var(--muted)', fontSize: '11.5px', fontWeight: '600', lineHeight: '18px',
  letterSpacing: '0', whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums', verticalAlign: '1px'} as const;

/** A holder's label inside a sentence: "a cutter training" for "A cutter training"; a name stays as it is. */
const inSentence = (label: string): string => /^(A|An|The) /.test(label) ? label.charAt(0).toLowerCase() + label.slice(1) : label;

/** Whether no GPU can take a real run of the road labeler or of the signs on every frame now: every card held, by the
 * server's word (each card's free); an older server: any process on the GPU, as its start() refuses then. A machine
 * without nvidia-smi: never (the helpers check for themselves). */
export function noFreeCard(system: SystemInfo | null | undefined): boolean {
  const cards = gpuCards(system);
  if (cards.some((card) => card.free !== undefined)) return !cards.some((card) => card.free === true);
  return (system?.gpu?.processes ?? 0) > 0;
}

/** What the server calls a card's holder it cannot tell (panel.py UNNAMED). */
const UNNAMED = 'a program the server cannot name';

/** What holds the GPUs, in the words of the server's reasons (panel.py _gpu_busy): "Both GPUs are busy: a cutter
 * training on GPU 0, a program the server cannot name on GPU 1", "Both GPUs are busy with a cutter training" (one
 * holder on both), "The GPU is busy with a cutter training"; one card that only an unnamed program holds, and an older
 * server, which does not say who: "The GPU is busy with another process". */
export function gpusBusyText(system: SystemInfo | null | undefined): string {
  const cards = gpuCards(system);
  const holders = (card: Gpu): string => card.holders?.length
    ? [...new Set(card.holders.map((holder) => inSentence(holder.label)))].join(' and ') : UNNAMED;
  if (!cards.some((card) => card.holders !== undefined)) return 'The GPU is busy with another process';
  const [first, second] = cards;
  if (cards.length === 1 && first) {
    return holders(first) === UNNAMED ? 'The GPU is busy with another process' : 'The GPU is busy with ' + holders(first);
  }
  if (cards.length === 2 && first && second && holders(first) === holders(second)) {
    return 'Both GPUs are busy with ' + holders(first);
  }
  return (cards.length === 2 ? 'Both GPUs are busy: ' : `All ${cards.length} GPUs are busy: `) + cards
    .filter((card) => card.free !== true)
    .map((card) => holders(card) + ' on ' + cardLabel(card, cards.indexOf(card), cards.length)).join(', ');
}

/** The tooltip where trainings start, on a machine with two GPUs or more from a server that gives each its own job (it
 * says which cards are free); undefined otherwise (no tooltip): a server from before only reads both cards, one training
 * at a time. */
export function twoGpusTip(system: SystemInfo | null | undefined): string | undefined {
  const cards = gpuCards(system);
  return cards.length > 1 && cards.some((card) => card.free !== undefined)
    ? 'Two GPUs: two trainings can run at once, one per card. The brain uses one card for now.' : undefined;
}

/** About how long a running job has left: the server's time when it sends one (Propose again), else from the time so
 * far and the percent, only past 3% and only "about"; '' when unknown. now: the clock, in ms. */
export function etaText(job: Job, now: number): string {
  if (job.state !== 'running') return '';
  const server = job.progress.seconds_left;
  if (server !== undefined) return server === null ? '' : durationText(server) + ' left';
  const percent = job.progress.percent;
  const start = startedAt(job);
  const elapsed = start === null ? null : now - start.getTime();
  if (percent === null || percent <= 3 || percent >= 100 || elapsed === null || elapsed <= 0) return '';
  return 'about ' + roughText(elapsed * (100 - percent) / percent) + ' left';
}

/** What a job does now, in words: its detail ("detector · epoch 5 of 20"), else its current step. */
export function stepText(job: Job): string {
  return job.progress.detail || job.progress.steps.find((step) => step.state === 'current')?.label || '';
}

/** The step a failed job stopped at, if the log said. */
export function failedStep(job: Job): string {
  return job.progress.steps.find((step) => step.state === 'failed')?.label ?? '';
}
