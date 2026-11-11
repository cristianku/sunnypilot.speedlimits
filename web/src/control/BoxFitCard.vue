<script setup lang="ts">
// Fit box on Models: the review page's key C, which fits a rough box to its sign on this server's CPU
// (nnslr_tools.box_fit): two rankers (colour trees and edge networks), then the box refiner, a small network, when it
// beat them. Train them again on the boxes people drew (scripts/train_box_fit.sh, the panel action "box_fit_train"), the
// refiner too or not, a dry run first if wanted (it only counts the boxes); its job while it runs; then the last summary
// of each part from GET /api/panel/box-fit: the mean IoU with the person's box on drives left out, of the rough box, the
// rankers and the refiner, and whether refiner.onnx was written. Trained on a V100 (#65), the candidates cut on the CPU
// first; its dry run takes no card.
// The site loads the new models at its next restart. Hidden on a server without the action, and the summaries left out
// when the endpoint is not there yet.
import {computed, onMounted, ref, shallowRef, watch} from 'vue';
import {api} from '../common/api';
import {isoWhen, plural, sentence} from '../common/format';
import type {ActionName, ActionOptions, ActionState, BoxFitMeasure, BoxFitPayload, Job, PanelStatus} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import JobProgress from './JobProgress.vue';
import {JOB_STATE, jobOf, startedWhen, terminalActivities} from './jobs';
import TerminalActivity from './TerminalActivity.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions];
  stop: [job: Job]; details: [job: Job]}>();

const state = computed((): ActionState | null => props.status?.actions?.box_fit_train ?? null);
/** A server from before the action: no card. */
const shown = computed(() => state.value !== null);
const job = computed(() => jobOf(props.status, ['box_fit_train']));
const bigJob = computed(() => job.value && (job.value.state === 'running' || job.value.state === 'failed') ? job.value : null);
const activities = computed(() => terminalActivities(props.status, ['box_fit']));
const idle = computed(() => !bigJob.value && !activities.value.length);

const payload = shallowRef<BoxFitPayload | null>(null);
async function load(): Promise<void> {
  try {
    payload.value = await api<BoxFitPayload>('/api/panel/box-fit');
  } catch {
    // Not on this server yet (it comes with a restart), or not readable now: the card shows no summary.
    payload.value = null;
  }
}
onMounted(() => void load());
const working = computed(() => job.value?.state === 'running' || activities.value.length > 0);
watch(working, (now, before) => {
  if (before && !now) void load();
});

/** The rankers only, without the hour of the refiner. */
const withoutRefiner = ref(false);
function start(dry: boolean): void {
  if (!state.value) return;
  const options: ActionOptions = {...(dry ? {dry: 'on'} : {}), ...(withoutRefiner.value ? {refiner: 'off'} : {})};
  emit('run', 'box_fit_train', state.value, undefined, Object.keys(options).length ? options : undefined);
}

const rankers = computed(() => payload.value?.rankers ?? null);
const refiner = computed(() => payload.value?.refiner ?? null);
/** A mean IoU as written in the summaries, 0 to 1. */
const iou = (value: number | null | undefined): string => typeof value === 'number' ? value.toFixed(3) : '–';
const rankersFacts = computed(() => {
  const item = rankers.value;
  if (!item) return '';
  return [isoWhen(item.at), item.boxes !== null ? plural(item.boxes, 'box', 'boxes') : '',
    item.routes !== null ? plural(item.routes, 'drive') : '', item.folds ? item.folds + ' groups of drives' : '']
    .filter(Boolean).join(' · ');
});
const refinerFacts = computed(() => {
  const item = refiner.value;
  if (!item) return '';
  return [isoWhen(item.at), item.boxes !== null ? plural(item.boxes, 'box', 'boxes') + ' to learn' : '',
    item.held_out_boxes !== null ? item.held_out_boxes + ' left out' : '',
    item.best_epoch !== null && item.epochs !== null ? 'best epoch ' + item.best_epoch + ' of ' + item.epochs : '',
    item.minutes !== null ? item.minutes + ' min' : ''].filter(Boolean).join(' · ');
});
const measureText = (measure: BoxFitMeasure): string => 'rough ' + iou(measure.rough) + ' · rankers ' + iou(measure.rankers)
  + ' · refiner ' + iou(measure.refiner);
const beats = (measure: BoxFitMeasure): boolean => typeof measure.refiner === 'number' && typeof measure.rankers === 'number'
  && measure.refiner > measure.rankers;
const rows = computed(() => {
  const item = refiner.value;
  if (!item) return [];
  const out: {key: string; name: string; measure: BoxFitMeasure}[] = [];
  if (item.all) out.push({key: 'all', name: 'All boxes', measure: item.all});
  if (item.small) out.push({key: 'small', name: 'Small signs', measure: item.small});
  return out;
});
const writtenText = computed(() => {
  const item = refiner.value;
  if (!item) return '';
  return item.written ? 'refiner.onnx written: Fit box uses it after the site restarts.'
    : 'Not better than the rankers in all and on the small signs: refiner.onnx was left as it was.';
});
</script>

<template>
  <section v-if="shown" class="card import-card labeler-card" id="box-fit" aria-labelledby="bf-title">
    <h2 id="bf-title" class="rl-title">Fit box</h2>
    <p class="rl-what">Key C of the review page fits a rough box to its sign on this server: two rankers (colour trees and
      edge networks), then the box refiner, a small network, where it beat them. They learn from the boxes people drew. The
      review page uses new models after the site restarts.</p>
    <JobProgress v-if="bigJob" :job="bigJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
    <TerminalActivity v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity" :preparation="null"/>
    <div v-if="idle" class="import-row">
      <span class="import-icon" :class="{fresh: !rankers}" aria-hidden="true"><Icon name="train"/></span>
      <div class="import-what">
        <strong>{{ rankers ? 'Last trained on ' + (rankers.boxes !== null ? plural(rankers.boxes, 'box', 'boxes') : 'the boxes')
          : 'Not trained here yet' }}</strong>
        <span>On a V100: the candidates cut on the CPU (about 2 minutes), then the rankers{{ withoutRefiner ? '' : ' and the refiner' }} trained on the GPU.</span>
      </div>
      <div class="rl-buttons">
        <button type="button" class="btn" id="box-fit-dry" :disabled="!state?.allowed" @click="start(true)">Dry run</button>
        <button type="button" class="btn primary" id="box-fit-train" :disabled="!state?.allowed"
                @click="start(false)">Train</button>
      </div>
    </div>
    <p v-if="idle && state && !state.allowed" class="import-why">{{ sentence(state.reason) }}</p>
    <label v-if="idle" class="rl-marks"><input type="checkbox" id="box-fit-no-refiner" v-model="withoutRefiner"> Without the
      refiner (the rankers only)</label>
    <p v-if="job && !bigJob" class="import-last">Last run {{ startedWhen(job) }} · {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
      <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>

    <ul v-if="rankers || refiner" class="rl-runs" id="box-fit-summaries">
      <li v-if="rankers" class="rl-run" data-part="rankers">
        <div class="rl-run-head"><b>Rankers</b><small>{{ rankersFacts }}</small></div>
        <p class="rl-types">Mean IoU with the person's box on drives left out: rough box {{ iou(rankers.rough) }}, after the
          rankers {{ iou(rankers.fit) }}.</p>
      </li>
      <li v-if="refiner" class="rl-run" data-part="refiner">
        <div class="rl-run-head"><b>Refiner</b><small>{{ refinerFacts }}</small><small v-if="refiner.written"
          class="rl-current">written</small></div>
        <ul v-if="rows.length" class="rl-heads">
          <li v-for="row in rows" :key="row.key" class="rl-head" :class="{writes: beats(row.measure)}">
            <span class="rl-head-name" style="text-transform: none">{{ row.name }}</span>
            <span class="rl-head-num">{{ measureText(row.measure) }}</span>
            <span class="rl-head-writes">{{ beats(row.measure) ? 'refiner better' : 'rankers better' }}</span>
          </li>
        </ul>
        <p class="rl-types">{{ writtenText }}</p>
      </li>
    </ul>
  </section>
</template>
