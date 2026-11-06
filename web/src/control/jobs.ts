// The jobs of the control pages: their state in words, their times, which job a tab shows, the work started
// from a terminal.
import {shortWhen} from '../common/format';
import type {ActionName, Activity, Job, JobState, PanelStatus} from '../common/panel-types';
import type {IconName} from '../components/icons/shapes';

/** State of a job: label and pill class. */
export const JOB_STATE: Record<JobState, [label: string, kind: string]> = {
  running: ['running', 'run'], ok: ['done', 'ok'], failed: ['failed', 'bad'], interrupted: ['interrupted', 'warn'],
};

/** The group of each action (nnslr_tools.panel.ACTIONS): a terminal activity of the same group is that work. */
export const ACTION_GROUP: Record<ActionName, string> = {
  fetch: 'fetch', prepare: 'prepare', prepare_again: 'prepare', measure_proposer: 'train', setup_vlm: 'train', dataset: 'train', train: 'train', train_continue: 'train', train_resume: 'train',
  train_classifier: 'train', train_cutter: 'train', train_decider: 'train', train_road_type: 'train', train_brain: 'train', regression: 'train', update: 'update', ai_review: 'ai',
  measure_ai_reviewer: 'ai',
  networks_prepare: 'publish', networks_publish: 'publish', road_types: 'prepare', send_to_car: 'publish',
  try_on_comma: 'publish', comma_restore: 'publish', mapillary_locations: 'mapillary', mapillary_windows: 'mapillary',
};

/** The icon of each action, on its button. */
export const ACTION_ICON: Record<ActionName, IconName> = {
  fetch: 'download', prepare: 'prepare', prepare_again: 'resume', measure_proposer: 'train', setup_vlm: 'download', dataset: 'import', train: 'train', train_continue: 'train',
  train_resume: 'resume', train_classifier: 'train', train_cutter: 'train', train_decider: 'train', train_road_type: 'train', train_brain: 'train', regression: 'train',
  update: 'update', ai_review: 'ai',
  measure_ai_reviewer: 'ai', networks_prepare: 'package', networks_publish: 'upload', road_types: 'routes', send_to_car: 'upload',
  try_on_comma: 'upload', comma_restore: 'resume', mapillary_locations: 'routes', mapillary_windows: 'prepare',
};

/**
 * The tab that shows the job of each action, by its address: its card, the dot on the tab's link while it runs (also
 * in the phone menu), the link of "Running now" in the server's panel. The update has no tab: the page reloads.
 */
export const ACTION_TAB: Record<ActionName, string | null> = {
  fetch: '/prepare', prepare: '/prepare', prepare_again: '/prepare', setup_vlm: '/prepare',
  dataset: '/dataset', road_types: '/roads',
  train: '/train', train_continue: '/train', train_resume: '/train', train_classifier: '/train', train_road_type: '/train', train_brain: '/train',
  train_cutter: '/train', train_decider: '/train', ai_review: '/train', mapillary_locations: '/train', mapillary_windows: '/train',
  measure_proposer: '/evaluate', measure_ai_reviewer: '/evaluate', regression: '/regression',
  networks_prepare: '/publish', networks_publish: '/publish',
  send_to_car: '/publish', try_on_comma: '/models', comma_restore: '/models',
  update: null,
};
const actionsOf = (tab: string): ActionName[] => (Object.keys(ACTION_TAB) as ActionName[]).filter((name) => ACTION_TAB[name] === tab);
/** The actions whose jobs the Prepare, Train and Evaluate tabs show. */
export const PREPARE_ACTIONS: ActionName[] = actionsOf('/prepare');
export const TRAIN_ACTIONS: ActionName[] = actionsOf('/train');
export const EVALUATE_ACTIONS: ActionName[] = actionsOf('/evaluate');

/** The tab of work started from a terminal, by its group (nnslr_tools.panel.ACTIVITIES); a measure is Evaluate's, the
 * road types (in the group of the preparation) are Roads', the Mapillary positions and windows Train's (its classifier
 * and its cutter). */
export function activityTab(activity: Pick<Activity, 'group' | 'label'>): string | null {
  if (activity.label === 'road types') return '/roads';
  if (activity.group === 'mapillary') return '/train';
  if (activity.label === 'networks on the comma') return '/models';
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
  'train_road_type', 'train_brain'];
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

/** The work started from a terminal (or by another program) that is not a job of the site. */
export function terminalActivities(status: PanelStatus | null, groups?: string[]): Activity[] {
  if (status === null) return [];
  const fromSite = new Set(status.jobs.filter((job) => job.state === 'running').map((job) => ACTION_GROUP[job.action]));
  // A process whose command line matches two labels (same pid) is shown once.
  const pids = new Set<number>();
  return status.activities.filter((activity) => {
    if (pids.has(activity.pid)) return false;
    pids.add(activity.pid);
    return !fromSite.has(activity.group) && (groups === undefined || groups.includes(activity.group));
  });
}

/** The actions whose job uses the GPU (the proposers of Prepare and their measure, Qwen3-VL's test, the trainings). */
const GPU_ACTIONS: ActionName[] = ['prepare', 'prepare_again', 'measure_proposer', 'setup_vlm', 'train', 'train_continue',
  'train_resume', 'train_classifier', 'train_cutter', 'train_decider', 'train_road_type', 'train_brain', 'regression'];
/** The same for work started from a terminal, by group ("other": the driving model of openpilot.distill). */
const GPU_GROUPS = ['prepare', 'train', 'other'];

/**
 * The name of each process on the GPU (system.gpu.process_list). In this container nvidia-smi cannot name the host's
 * processes ("[Not Found]"), the site's own jobs included: when one process has no name and one job of the site, or
 * one work from a terminal, uses the GPU, the row is that one; otherwise "A program the server cannot name".
 */
export function gpuProcessNames(status: PanelStatus | null): string[] {
  const list = status?.system.gpu?.process_list ?? [];
  const unnamed = (name: string): boolean => !name || name.startsWith('[');
  const known = [
    ...(status?.jobs ?? []).filter((job) => job.state === 'running' && GPU_ACTIONS.includes(job.action)).map((job) => job.label),
    ...terminalActivities(status, GPU_GROUPS).map((activity) => (activity.label.startsWith('openpilot') ? activity.label
      : activity.label.charAt(0).toUpperCase() + activity.label.slice(1)) + ', from a terminal'),
  ];
  const guess = known.length === 1 && list.filter((process) => unnamed(process.name)).length === 1 ? known[0] : undefined;
  return list.map((process) => (unnamed(process.name) ? guess ?? 'A program the server cannot name' : process.name));
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
