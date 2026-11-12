// The Evaluate tab: the runs (GET /api/panel/runs 10 s after each answer while the tab is open), the run chosen (kept in
// ?run= of the address, so a link opens it), its detail (GET /api/panel/runs/<run>, asked again only when the
// run changes), and the words and numbers the tab shows.
import {ref, shallowRef, watch, type Ref} from 'vue';
import {api, messageOf} from '../common/api';
import {isoWhen, thousands} from '../common/format';
import type {RunDetail, RunState, RunSummary, RunsPayload} from '../common/panel-types';
import {elapsedText} from './jobs';
import {usePoll} from '../composables/poll';

const INTERVAL_MS = 10000;

/** State of a run: label and pill class. */
export const RUN_STATE: Record<RunState, [label: string, kind: string]> = {
  running: ['running', 'run'], finished: ['finished', 'ok'], incomplete: ['stopped', 'warn'],
};

/** The run chosen on the page, from ?run= at the first load. */
export const selectedRun = ref<string | null>(new URLSearchParams(location.search).get('run'));
/** The version chosen in that run (an id of the Models tab, "cutter:<run>/detector-sizes/960x544"), from ?model=. */
export const selectedModel = ref<string | null>(new URLSearchParams(location.search).get('model'));

/** Chooses a run, and a version of it, and puts them in the address (no new history entry). The Evaluate tab shows
 * that version in the view of its role, then forgets them. */
export function chooseRun(name: string, model: string | null = null): void {
  selectedRun.value = name;
  selectedModel.value = model;
  history.replaceState(history.state, '', '/evaluate?run=' + encodeURIComponent(name)
    + (model === null ? '' : '&model=' + encodeURIComponent(model)));
}

/** The runs, asked 10 s after each answer while the tab is shown. */
export function useRuns() {
  const payload = shallowRef<RunsPayload | null>(null);
  const error = ref('');
  async function ask(): Promise<void> {
    try {
      payload.value = await api<RunsPayload>('/api/panel/runs');
      error.value = '';
    } catch (problem) {
      error.value = messageOf(problem);
    }
  }
  return {payload, error, refresh: usePoll(ask, INTERVAL_MS)};
}

/** The detail of a run, asked when the run shown or its version changes. */
export function useRunDetail(name: Ref<string | null>, version: Ref<string | null>) {
  const detail = shallowRef<RunDetail | null>(null);
  const loading = ref(false);
  const error = ref('');
  let asked = 0;
  watch([name, version], async ([current]) => {
    if (current === null) {
      detail.value = null;
      return;
    }
    const ticket = ++asked;
    // A new run: no old numbers while the new ones load; the same run: keep showing the old ones.
    if (detail.value?.name !== current) {
      detail.value = null;
      loading.value = true;
    }
    try {
      const answer = await api<RunDetail>('/api/panel/runs/' + encodeURIComponent(current));
      if (ticket === asked) {
        detail.value = answer;
        error.value = '';
      }
    } catch (problem) {
      if (ticket === asked) error.value = messageOf(problem);
    } finally {
      if (ticket === asked) loading.value = false;
    }
  }, {immediate: true});
  return {detail, loading, error};
}

/** 0.873 -> "87.3%", 0.29 -> "29.0%" (one decimal, so a column lines up), 1 -> "100%", 0 -> "0%"; null -> "–". */
export function pct(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '–';
  const tenths = Math.round(value * 1000);
  return (tenths === 0 || tenths === 1000 ? String(tenths / 10) : (tenths / 10).toFixed(1)) + '%';
}

/** A difference of two shares in percentage points: "+2.1 pp", "−0.4 pp", "±0 pp". */
export function points(difference: number): string {
  const value = Math.round(difference * 1000) / 10;
  if (value === 0) return '±0 pp';
  return (value > 0 ? '+' : '−') + Math.abs(value).toString() + ' pp';
}

/** A difference of two counts: "+12", "−3", "±0". */
export function signed(difference: number): string {
  if (difference === 0) return '±0';
  return (difference > 0 ? '+' : '−') + thousands(Math.abs(difference));
}

/** A loss or another plain number, as the training charts show it. */
export function plain(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '–';
  return value >= 10 ? value.toFixed(1) : value >= 1 ? value.toFixed(2) : value.toFixed(3);
}

/** "24 Oct 14:52 · 3 h 05 min" (how long it took, or has been running). */
export function runWhen(run: RunSummary): string {
  const started = isoWhen(run.started_at);
  const end = run.finished_at ?? (run.state === 'running' ? new Date().toISOString() : run.updated_at);
  if (!run.started_at || !end) return started;
  const ms = Date.parse(end) - Date.parse(run.started_at);
  return Number.isFinite(ms) && ms > 0 ? started + ' · ' + elapsedText(ms) : started;
}

/** The mode of the pipeline in words, with "resumed" when it was. */
export function modeText(run: RunSummary): string {
  const mode = run.mode === 'fresh' ? 'new run' : run.mode === 'continue' ? 'continued' : run.mode === 'resume' ? 'resumed' : '';
  return [mode, run.resumed && run.mode !== 'resume' ? 'resumed' : ''].filter(Boolean).join(', ');
}

/** Where a run is: "detector · epoch 5 of 20", "stopped while training the reader", "bundle packaged". */
export function stageText(run: RunSummary): string {
  if (run.state === 'running') return run.progress || run.stage_label;
  if (run.state === 'finished') {
    return run.kind === 'experiment' ? 'every variant measured' : run.kind === 'classifier' ? 'classifier and chain measured'
      : run.kind === 'cutter' ? 'cutter measured' : run.kind === 'usernet' ? 'UserNet measured' : 'bundle packaged';
  }
  return 'while ' + run.stage_label;
}

/** Why a run has no headline numbers. */
export function notEvaluatedText(run: RunSummary): string {
  if (run.kind === 'experiment') {
    return 'Experiment B only: ' + run.experiment_b.length + ' of 3 variants measured' + (run.experiment_b.length ? ', open it for the table.' : '.');
  }
  if (run.state === 'running') {
    return run.kind === 'classifier' ? 'Not evaluated yet: the classifier and the chain are measured after the training.'
      : run.kind === 'cutter' || run.kind === 'usernet' ? 'Not evaluated yet: test and validation come after the training.'
        : 'Not evaluated yet: the test comes after the training.';
  }
  return 'No evaluation: the run stopped before it.';
}

/** What a UserNet learnt from, in words: "BDD100K" or "your drives" (with "after BDD100K" when it went on from it). */
export function usernetStageText(stage: string | null | undefined, usesBdd: boolean): string {
  if (stage === 'bdd') return 'BDD100K';
  if (stage === 'drives') return usesBdd ? 'your drives, after BDD100K' : 'your drives';
  return '–';
}

/** A UserNet run in one line: "UserNet (BDD100K), at score 0.5 on test: 62.0% of the road users found · 1.80 false boxes
 * per frame". */
export function usernetLine(run: RunSummary): string {
  const line = run.usernet;
  if (!line || !line.partition) return 'UserNet: open the run for its numbers.';
  return 'UserNet (' + usernetStageText(line.stage, line.uses_bdd) + '), at score ' + line.score_threshold + ' on '
    + line.partition + ': ' + pct(line.recall) + ' of the road users found · '
    + (line.false_per_frame === null ? '–' : line.false_per_frame.toFixed(2)) + ' false boxes per frame';
}

/** The first characters of a hash, for the page. */
export function shortSha(sha: string | null | undefined): string {
  return sha ? sha.slice(0, 10) : '–';
}

/** The partitions of a split in the order the training uses them, then any other. */
export function partitionOrder(keys: string[]): string[] {
  const order = ['train', 'validation', 'test', 'buffer'];
  return [...order.filter((key) => keys.includes(key)), ...keys.filter((key) => !order.includes(key)).sort()];
}

export const PARTITION_LABEL: Record<string, string> = {
  train: 'Train', validation: 'Validation', test: 'Test', buffer: 'Buffer', result: 'Result',
};
