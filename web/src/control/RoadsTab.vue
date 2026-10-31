<script setup lang="ts">
// Data › Roads: the type of road of every picture of the prepared drives (town, zone 30, rural, expressway, motorway, ramp,
// merge lane, parking, garage), from where the car was and the map (helpers/road_types.sh, the panel action "road_types"). First the
// action, with Find the road types or its job running; then each drive as a thin strip of its whole length coloured by
// road type, from GET /api/panel/road-types, read when the page opens and again when a search for the road types ends.
// Nothing here needs a review: the map's types are a proposal; a tap on a drive opens it ("/roads?drive=<route>",
// RoadDrive.vue), to look at its pictures and correct the map, and the list is read again on the way back.
import {computed, onMounted, ref, shallowRef, watch} from 'vue';
import {api, messageOf} from '../common/api';
import {isoWhen, plural, sentence, thousands} from '../common/format';
import type {ActionName, ActionState, Job, PanelStatus, RoadTypesPayload} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import JobProgress from './JobProgress.vue';
import {activityTab, JOB_STATE, jobOf, startedWhen, terminalActivities} from './jobs';
import RoadDrive from './RoadDrive.vue';
import {ROAD_LABEL as LABEL, ROAD_TYPES as TYPES} from './roads';
import SectionHead from './SectionHead.vue';
import {goTo, tabDrive as roadsDrive} from './tabs';
import TerminalActivity from './TerminalActivity.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState]; stop: [job: Job]; details: [job: Job]}>();

const job = computed(() => jobOf(props.status, ['road_types']));
/** Running or failed: in place of the button; one done is a line. */
const bigJob = computed(() => job.value && (job.value.state === 'running' || job.value.state === 'failed') ? job.value : null);
const state = computed(() => props.status?.actions.road_types ?? null);
/** The road types started from a terminal: shown here, stopped there. */
const activities = computed(() => terminalActivities(props.status, ['prepare']).filter((activity) => activityTab(activity) === '/roads'));
/** Nothing searching now: the button, and why it cannot start. */
const idle = computed(() => !bigJob.value && !activities.value.length);

const payload = shallowRef<RoadTypesPayload | null>(null);
const error = ref('');
async function load(): Promise<void> {
  try {
    payload.value = await api<RoadTypesPayload>('/api/panel/road-types');
    error.value = '';
  } catch (problem) {
    error.value = messageOf(problem);
  }
}
onMounted(() => void load());
// A search for the road types that ends (done, failed or stopped, from here or a terminal): its drives again.
const searching = computed(() => job.value?.state === 'running' || activities.value.length > 0);
watch(searching, (now, before) => {
  if (before && !now) void load();
});
// Back from a drive: its corrections in the list.
watch(roadsDrive, (now, before) => {
  if (before && !now) void load();
});

const percentText = (part: number, whole: number): string => {
  const value = whole ? part * 100 / whole : 0;
  return value > 0 && value < 1 ? '<1%' : Math.round(value) + '%';
};
/** Each drive: its stretches as the parts of its strip, and the share of each kind of road under it. */
const drives = computed(() => (payload.value?.routes ?? []).map((drive) => ({
  ...drive,
  parts: drive.stretches.map((stretch) => {
    const kind = stretch.type ?? 'unknown';
    return {kind, frames: stretch.frames, tip: LABEL[kind] + ' · ' + plural(stretch.frames, 'picture')};
  }),
  shares: TYPES.filter(([key]) => (drive.types[key] ?? 0) > 0)
    .map(([key, label]) => ({key, label, share: percentText(drive.types[key] ?? 0, drive.frames)})),
})));
const without = computed(() => payload.value?.drives_without ?? 0);
/** Prepared drives still without road types, or none read yet. */
const fresh = computed(() => payload.value !== null && (without.value > 0 || !payload.value.routes.length));
/** Drives copied from the comma but not prepared yet: no pictures, so no road types until Prepare new drives. */
const waiting = computed(() => (props.status?.routes ?? []).filter((row) => ['collegare', 'estrarre', 'preannotare'].includes(row.state)).length);
const madeAt = computed(() => isoWhen(payload.value?.made_at));
</script>

<template>
  <RoadDrive v-if="roadsDrive" :route="roadsDrive" :read-only="status?.actions_enabled === false"/>
  <template v-else>
    <SectionHead tab="roads"/>

    <section class="card import-card" id="road-types" aria-labelledby="rt-title">
      <h2 id="rt-title" class="sr-only">Road types from the map</h2>
      <JobProgress v-if="bigJob" :job="bigJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
      <TerminalActivity v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity" :preparation="null"/>
      <div v-if="idle" class="import-row">
        <span class="import-icon" :class="{fresh}" aria-hidden="true"><Icon v-if="fresh" name="routes"/><template v-else>✓</template></span>
        <div class="import-what">
          <strong>{{ payload === null ? 'Road types from the map' : fresh ? 'Road types to find' : 'Road types are up to date' }}</strong>
          <span v-if="payload !== null && !fresh">Every prepared drive has them, from the map{{ madeAt ? ' · ' + madeAt : '' }}.</span>
          <span v-else>Reads where the car was and the type of road there on the map, for every picture of the prepared drives. No need to
            review anything first.</span>
        </div>
        <button type="button" class="btn" :class="{primary: fresh}" id="find-road-types" :disabled="!state?.allowed"
                @click="state && emit('run', 'road_types', state)">{{ fresh ? 'Find the road types' : 'Find again' }}</button>
      </div>
      <p v-if="idle && without > 0" class="import-why" id="rt-without">{{ plural(without, 'prepared drive has', 'prepared drives have') }}
        no road types yet.</p>
      <p v-if="idle && waiting > 0" class="import-why" id="rt-waiting">{{ plural(waiting, 'new drive needs', 'new drives need') }} their pictures
        first: <button type="button" class="link-button" @click="goTo('prepare')">Prepare new drives</button> in Drives.</p>
      <p v-if="idle && state && !state.allowed" class="import-why">{{ sentence(state.reason) }}</p>
      <p v-if="job && !bigJob" class="import-last">Last search {{ startedWhen(job) }} ·
        {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
        <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>
    </section>

    <section class="card rt-drives" aria-labelledby="rt-drives-title">
      <div class="ds-head">
        <h2 id="rt-drives-title">By drive</h2>
        <span v-if="drives.length" class="ds-when">{{ plural(drives.length, 'drive') }}</span>
      </div>
      <p v-if="error" class="empty">{{ error }}</p>
      <p v-else-if="payload === null" class="empty">Loading…</p>
      <p v-else-if="!drives.length" class="empty">No road types yet: Find the road types reads them for the prepared drives.</p>
      <template v-else>
        <ul class="rt-legend" aria-label="Kinds of road">
          <li v-for="[key, label] in TYPES" :key="key"><span class="rt-key" :class="'rt-' + key" aria-hidden="true"></span>{{ label }}</li>
        </ul>
        <ul class="rt-list">
          <li v-for="drive in drives" :key="drive.route" class="rt-drive"><a class="rt-open" :href="'/roads?drive=' + encodeURIComponent(drive.route)">
            <div class="rt-name">
              <strong>{{ drive.title ?? 'Places unknown' }}</strong>
              <span>{{ [drive.when, drive.minutes !== null ? drive.minutes + ' min' : null].filter(Boolean).join(' · ') }}</span>
            </div>
            <div class="rt-strip" role="img"
                 :aria-label="(drive.title ?? 'Drive') + ': ' + drive.shares.map((item) => item.label + ' ' + item.share).join(', ')">
              <span v-for="(part, index) in drive.parts" :key="index" :class="'rt-' + part.kind" :style="{flexGrow: part.frames}"
                    :title="part.tip"></span>
            </div>
            <p class="rt-shares">
              <span v-for="item in drive.shares" :key="item.key">{{ item.label }} {{ item.share }}</span>
              <span v-if="drive.reviewed" class="rt-reviewed">{{ thousands(drive.reviewed) }} reviewed</span>
              <span class="rt-frames">{{ thousands(drive.frames) }} pictures</span>
            </p>
            <span class="data-chev rt-go" aria-hidden="true">›</span>
          </a></li>
        </ul>
      </template>
    </section>
  </template>
</template>
