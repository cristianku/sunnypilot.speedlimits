<script setup lang="ts">
// Models › Regression: the drives he drove well and certified once, and on them every model he lights. A run computes only
// what is missing (scripts/run_regression.sh, the panel action "regression"), so after a training only the new model runs;
// a certified drive opened ("/regression?drive=<route>", RegressionDrive.vue) shows his speed under the video with what each
// model would have said. Nothing here changes the car, the labels or the model in use.
import {computed, onMounted, ref, shallowRef, watch} from 'vue';
import {api, messageOf, post} from '../common/api';
import {plural} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Job, PanelStatus} from '../common/panel-types';
import {clockOf} from './drive-review';
import JobProgress from './JobProgress.vue';
import {jobOf} from './jobs';
import {brainKey, firstChoice, lit, type RegressionPayload, type RegressionRow, toggle} from './regression';
import RegressionDrive from './RegressionDrive.vue';
import SectionHead from './SectionHead.vue';
import {tabDrive} from './tabs';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions]; stop: [job: Job];
  details: [job: Job]}>();
const SHORT_LIST = 12;

const payload = shallowRef<RegressionPayload | null>(null);
const error = ref('');
async function load(): Promise<void> {
  try {
    show(await api<RegressionPayload>('/api/regression/routes'));
  } catch (problem) {
    error.value = messageOf(problem);
  }
}
function show(answer: RegressionPayload): void {
  payload.value = answer;
  lit.value ??= firstChoice(answer.brains);
  error.value = '';
}

const job = computed(() => jobOf(props.status, ['regression']));
const bigJob = computed(() => job.value && (job.value.state === 'running' || job.value.state === 'failed') ? job.value : null);
const state = computed(() => props.status?.actions.regression ?? null);
const readOnly = computed(() => props.status?.actions_enabled === false);
// A run that ends: what it computed.
watch(() => job.value?.state, (now, before) => { if (before === 'running' && now !== 'running') void load(); });
// Back on the list: a drive may have left the certified ones.
watch(tabDrive, (route) => { if (route === null) void load(); });
onMounted(() => { if (tabDrive.value === null) void load(); });

const startOf = (drive: RegressionRow): string => (drive.day ?? '') + (drive.when ?? '').slice(-5);
const newest = computed(() => [...(payload.value?.routes ?? [])].sort((a, b) => startOf(b).localeCompare(startOf(a))));
const certified = computed(() => newest.value.filter((drive) => drive.certified));
const others = computed(() => newest.value.filter((drive) => !drive.certified));
const showAll = ref(false);
const offered = computed(() => showAll.value ? others.value : others.value.slice(0, SHORT_LIST));

const isLit = (key: string): boolean => (lit.value ?? []).includes(key);
const chosenDriving = computed(() => (payload.value?.models ?? []).filter((model) => isLit(model.key)));
const chosenBrains = computed(() => (payload.value?.brains ?? []).filter((brain) => isLit(brainKey(brain.key))));
const labels = computed(() => [...chosenDriving.value.map((model) => ({key: model.key, label: model.label})),
  ...chosenBrains.value.map((brain) => ({key: brainKey(brain.key), label: 'Smartspeed ' + brain.label}))]);

const stateOf = (drive: RegressionRow, key: string): string | undefined => drive.results?.[key];
/** What a run would compute: the lit models missing on the certified drives, in runs of one model on one drive. */
const missing = computed(() => certified.value.reduce((sum, drive) => sum + labels.value.filter(({key}) =>
  ['missing', 'more_wide', 'failed'].includes(stateOf(drive, key) ?? '')).length, 0));
const words = (drive: RegressionRow, wanted: string[]): string[] => labels.value.filter(({key}) => wanted.includes(stateOf(drive, key) ?? ''))
  .map(({label}) => label);
function resultsText(drive: RegressionRow): string {
  const parts: string[] = [];
  const done = words(drive, ['done']);
  const open = words(drive, ['missing', 'more_wide']);
  const failed = words(drive, ['failed']);
  if (done.length) parts.push('Ready: ' + done.join(', '));
  if (open.length) parts.push('To run: ' + open.join(', '));
  if (failed.length) parts.push('Failed: ' + failed.join(', '));
  return parts.join(' · ') || 'Light a model above.';
}
function limitsText(drive: RegressionRow): string {
  const parts: string[] = [];
  if (drive.wide[0] === 0) parts.push('No wide camera: only In the car can run here.');
  else if (drive.wide[0] < drive.wide[1]) parts.push(`Wide camera on ${drive.wide[0]} of ${drive.wide[1]} segments.`);
  if (drive.frames === false) parts.push('Not prepared: Smartspeed cannot run here.');
  return parts.join(' ');
}
function learnedText(drive: RegressionRow): string {
  const parts: string[] = [];
  if (drive.brain_learned_s) parts.push(`The brain trained on ${clockOf(drive.brain_learned_s)} of it`);
  if (drive.distill_segments) parts.push(`${parts.length ? 'the' : 'The'} distilled models on ${plural(drive.distill_segments, 'segment')}`);
  return parts.length ? parts.join(', ') + ': not a fair test there.' : 'No model trained on it.';
}
const wideText = (drive: RegressionRow): string => drive.wide[0] ? '' : ' · no wide camera';
const hrefOf = (drive: RegressionRow): string => '/regression?drive=' + encodeURIComponent(drive.route);

const certifying = ref('');
async function certify(drive: RegressionRow): Promise<void> {
  certifying.value = drive.route;
  try {
    show(await post<RegressionPayload>(`/api/regression/routes/${encodeURIComponent(drive.route)}/certify`, {certified: true}));
  } catch (problem) {
    error.value = messageOf(problem);
  } finally {
    certifying.value = '';
  }
}
function run(): void {
  if (!state.value) return;
  emit('run', 'regression', state.value, undefined, {openpilot: chosenDriving.value.map((model) => model.key).join(','),
    brains: chosenBrains.value.map((brain) => brain.key).join(',')});
}
const canRun = computed(() => Boolean(state.value?.allowed) && missing.value > 0);
const runText = computed(() => {
  if (state.value && !state.value.allowed) return state.value.reason;
  if (!certified.value.length) return 'Certify a drive first.';
  if (!labels.value.length) return 'Light a model first.';
  return missing.value ? `${plural(missing.value, 'run')} to compute: only what is missing, on the GPU.`
    : 'Everything lit is computed on the certified drives.';
});
</script>

<template>
  <RegressionDrive v-if="tabDrive" :key="tabDrive" :route="tabDrive" :read-only="readOnly"/>
  <template v-else>
    <SectionHead tab="regression"/>
    <section class="card" aria-labelledby="rg-models-title">
      <h2 id="rg-models-title">Models</h2>
      <p class="import-why">Light what you want to compare on the certified drives: each one is a curve next to your speed.</p>
      <p v-if="error" class="empty">{{ error }}</p>
      <p v-else-if="payload === null" class="empty">Loading…</p>
      <template v-else>
        <p class="rg-group">openpilot</p>
        <div class="rg-switches">
          <button v-for="model in payload.models" :key="model.key" type="button" class="btn small" :aria-pressed="isLit(model.key)"
                  :title="model.detail" @click="toggle(model.key)">{{ model.label }}</button>
        </div>
        <p class="rg-group">Smartspeed</p>
        <div v-if="payload.brains.length" class="rg-switches">
          <button v-for="brain in payload.brains" :key="brain.key" type="button" class="btn small" :aria-pressed="isLit(brainKey(brain.key))"
                  :title="brain.detail" @click="toggle(brainKey(brain.key))">{{ brain.label }}{{ brain.in_use ? ' · in use' : '' }}</button>
        </div>
        <p v-else class="import-why">No brain trained yet.</p>
        <JobProgress v-if="bigJob" :job="bigJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
        <div v-if="job?.state !== 'running'" class="rg-run">
          <p class="import-why">{{ runText }}</p>
          <button type="button" class="btn primary" :disabled="!canRun" @click="run">Run</button>
        </div>
      </template>
    </section>

    <section v-if="payload" class="card sr-drives" aria-labelledby="rg-drives-title">
      <div class="ds-head">
        <h2 id="rg-drives-title">Certified drives</h2>
        <span class="ds-when">{{ plural(certified.length, 'drive') }}</span>
      </div>
      <p v-if="!certified.length" class="import-why">None yet: certify below a drive where the car went well.</p>
      <ul v-else class="rt-list">
        <li v-for="drive in certified" :key="drive.route" class="rt-drive dr-drive">
          <a class="rt-open" :href="hrefOf(drive)">
            <div class="rt-name">
              <strong>{{ drive.title ?? 'Places unknown' }}</strong>
              <span><template v-if="drive.when">{{ drive.when_certain ? '' : '≈ ' }}{{ drive.when }}</template>{{
                drive.minutes !== null ? ' · ' + drive.minutes + ' min' : '' }}</span>
            </div>
            <p class="dr-drive-state">{{ resultsText(drive) }}</p>
            <p v-if="limitsText(drive)" class="dr-drive-state dr-muted">{{ limitsText(drive) }}</p>
            <p class="dr-drive-state dr-muted">{{ learnedText(drive) }}</p>
            <span class="data-chev rt-go" aria-hidden="true">›</span>
          </a>
        </li>
      </ul>
    </section>

    <section v-if="payload && others.length" class="card sr-drives" aria-labelledby="rg-add-title">
      <div class="ds-head">
        <h2 id="rg-add-title">Certify a drive</h2>
        <span class="ds-when">{{ plural(others.length, 'drive') }}</span>
      </div>
      <p class="import-why">Pick the drives where the car went well: they stay here for every model to come. The driving models need the wide
        camera.</p>
      <ul class="rt-list">
        <li v-for="drive in offered" :key="drive.route" class="rt-drive rg-add">
          <div class="rt-name">
            <strong>{{ drive.title ?? 'Places unknown' }}</strong>
            <span><template v-if="drive.when">{{ drive.when_certain ? '' : '≈ ' }}{{ drive.when }}</template>{{
              drive.minutes !== null ? ' · ' + drive.minutes + ' min' : '' }}{{ wideText(drive) }}</span>
          </div>
          <button type="button" class="btn small" :disabled="readOnly || certifying === drive.route" @click="certify(drive)">Certify</button>
        </li>
      </ul>
      <button v-if="others.length > SHORT_LIST" type="button" class="btn small rg-more" @click="showAll = !showAll">{{
        showAll ? 'Show fewer' : 'Show all ' + others.length }}</button>
    </section>
  </template>
</template>
