<script setup lang="ts">
// Read the map: the map's limits of the prepared drives (helpers/road_types.sh, the panel action "road_types"), at the
// top of Data › Speed review: the OpenStreetMap limit it finds is the proposal of each stretch there.
// The button, or its job running; then the drives still without, the maps of the server and those the drives still need
// (the reading downloads them first, scripts/osm_maps.py), from GET /api/panel/road-types, read when the page opens and
// again when a reading ends.
import {computed, onMounted, ref, shallowRef, watch} from 'vue';
import {api, messageOf} from '../common/api';
import {isoWhen, plural, sentence} from '../common/format';
import type {ActionName, ActionState, Job, PanelStatus, RoadMapNeed, RoadTypesPayload} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import JobProgress from './JobProgress.vue';
import {activityTab, JOB_STATE, jobOf, startedWhen, terminalActivities} from './jobs';
import {goTo} from './tabs';
import TerminalActivity from './TerminalActivity.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState]; stop: [job: Job]; details: [job: Job]}>();

const job = computed(() => jobOf(props.status, ['road_types']));
/** Running or failed: in place of the button; one done is a line. */
const bigJob = computed(() => job.value && (job.value.state === 'running' || job.value.state === 'failed') ? job.value : null);
const state = computed(() => props.status?.actions.road_types ?? null);
/** The search started from a terminal: shown here, stopped there. */
const activities = computed(() => terminalActivities(props.status, ['prepare']).filter((activity) => activityTab(activity) === '/speed-review'));
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
// A search that ends (done, failed or stopped, from here or a terminal): its drives again.
const searching = computed(() => job.value?.state === 'running' || activities.value.length > 0);
watch(searching, (now, before) => {
  if (before && !now) void load();
});

const without = computed(() => payload.value?.drives_without ?? 0);
/** The maps of the server (absent from a server before them). */
const maps = computed(() => payload.value?.maps ?? null);
/** Maps the next reading downloads for the drives outside the maps (one that failed is tried again; one bigger than a
 * reading downloads is not): their limits are still to find there. */
const mapsToGet = computed(() => (maps.value?.needed ?? []).some((item) => item.state !== 'too big'));
/** Prepared drives still without, none read yet, or maps to download first. */
const fresh = computed(() => payload.value !== null && (without.value > 0 || !payload.value.routes.length || mapsToGet.value));
/** Only maps to download: every prepared drive has been read, but not where it went outside the maps. */
const onlyMaps = computed(() => fresh.value && without.value === 0 && !!payload.value?.routes.length);
/** Drives copied from the comma but not prepared yet: no pictures, so no map until Prepare new drives. */
const waiting = computed(() => (props.status?.routes ?? []).filter((row) => ['collegare', 'estrarre', 'preannotare'].includes(row.state)).length);
const madeAt = computed(() => isoWhen(payload.value?.made_at));

const sizeText = (bytes: number): string => bytes >= 1e9 ? (bytes / 1e9).toFixed(1) + ' GB' : Math.max(1, Math.round(bytes / 1e6)) + ' MB';
const andList = (names: string[]): string => names.length < 2 ? names.join('') : names.slice(0, -1).join(', ') + ' and ' + names.at(-1);
const nameOf = (item: RoadMapNeed): string => item.name ?? 'a map';
/** The drives outside the maps, and what the next reading downloads: "2 drives outside the maps: Read again downloads
 * Alsace and Michigan (≈ 448 MB)."; then why the last download failed, and the maps more than a search downloads. */
const outsideText = computed(() => {
  const needed = maps.value?.needed ?? [];
  const items = needed.filter((item) => item.state !== 'too big');
  const total = items.reduce((sum, item) => sum + (item.bytes ?? 0), 0);
  const big = needed.filter((item) => item.state === 'too big');
  const failed = needed.find((item) => item.state === 'failed');
  return [
    plural(maps.value?.outside ?? 0, 'drive') + ' outside the maps' + (items.length
      ? ': ' + (fresh.value ? 'Read the map' : 'Read again') + ' downloads ' + andList(items.map(nameOf))
        + (total ? ' (≈ ' + sizeText(total) + ')' : '') : '') + '.',
    failed ? 'The last download failed' + (failed.reason ? ': ' + failed.reason : '') + '.' : '',
    big.length ? andList(big.map(nameOf)) + (big.length === 1 ? ' is' : ' are') + ' more than the '
      + (maps.value?.max_gb ?? 5) + ' GB a reading downloads.' : '',
  ].filter(Boolean).join(' ');
});
</script>

<template>
  <section class="card import-card" id="road-types" aria-labelledby="rt-title">
    <h2 id="rt-title" class="sr-only">The map's limits</h2>
    <JobProgress v-if="bigJob" :job="bigJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
    <TerminalActivity v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity" :preparation="null"/>
    <div v-if="idle" class="import-row">
      <span class="import-icon" :class="{fresh}" aria-hidden="true"><Icon v-if="fresh" name="routes"/><template v-else>✓</template></span>
      <div class="import-what">
        <strong>{{ payload === null ? 'The map\'s limits' : !fresh ? 'The map is read for every drive' : onlyMaps ? 'Maps to download'
          : 'Drives to read on the map' }}</strong>
        <span v-if="payload !== null && !fresh">Every prepared drive has the map's limits{{ madeAt ? ' · ' + madeAt : '' }}: the
          proposals below.</span>
        <span v-else-if="onlyMaps">Some drives went where the server has no map: the reading downloads it first, then reads them
          there.</span>
        <span v-else-if="error">{{ error }}</span>
        <span v-else>Reads where the car was and the limit there on the map, for every picture of the prepared drives: the proposal of each
          stretch below.</span>
      </div>
      <button type="button" class="btn" :class="{primary: fresh}" id="find-road-types" :disabled="!state?.allowed"
              @click="state && emit('run', 'road_types', state)">{{ fresh ? 'Read the map' : 'Read again' }}</button>
    </div>
    <p v-if="idle && without > 0" class="import-why" id="rt-without">{{ plural(without, 'prepared drive has', 'prepared drives have') }}
      not been read on the map yet.</p>
    <p v-if="idle && waiting > 0" class="import-why" id="rt-waiting">{{ plural(waiting, 'new drive needs', 'new drives need') }} their pictures
      first: <button type="button" class="link-button" @click="goTo('prepare')">Prepare new drives</button> in Drives.</p>
    <p v-if="idle && maps && maps.checked_at && maps.names.length" class="import-why" id="rt-maps">Maps: {{ maps.names.join(', ') }}.</p>
    <p v-if="idle && maps && maps.outside > 0" class="import-why" id="rt-outside">{{ outsideText }}</p>
    <p v-if="idle && maps && maps.nowhere > 0" class="import-why" id="rt-nowhere">{{ plural(maps.nowhere, 'drive has', 'drives have') }}
      positions on no map to download (the sea, a lake, a GPS fix that jumped): no map limit there.</p>
    <p v-if="idle && state && !state.allowed" class="import-why">{{ sentence(state.reason) }}</p>
    <p v-if="job && !bigJob" class="import-last">Last reading {{ startedWhen(job) }} ·
      {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
      <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>
  </section>
</template>
