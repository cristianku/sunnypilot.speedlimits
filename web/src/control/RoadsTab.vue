<script setup lang="ts">
// Data › Roads: the type of road of every picture of the prepared drives (town, zone 30, rural, expressway, motorway, ramp,
// merge lane, parking, garage), from where the car was and the map (helpers/road_types.sh, the panel action "road_types"). First the
// action, with Find the road types or its job running; then each drive as a thin strip of its whole length coloured by
// road type, from GET /api/panel/road-types, read when the page opens and again when a search for the road types ends.
// The map's types are proposals, and only what a person confirms or corrects is used in training: each drive says how
// many frames are usable and how many to review, and the title of the list the sum. A tap on a drive opens it
// ("/roads?drive=<route>", RoadDrive.vue), to look at its pictures, confirm or correct them, and the list is read again on
// the way back. Under the action, the maps of the server and those the drives still need: the search downloads them
// first (scripts/osm_maps.py).
import {computed, onMounted, ref, shallowRef, watch} from 'vue';
import {api, messageOf} from '../common/api';
import {isoWhen, plural, sentence, thousands} from '../common/format';
import type {ActionName, ActionState, Job, PanelStatus, RoadMapNeed, RoadTypesPayload} from '../common/panel-types';
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
/** The maps of the server (absent from a server before them). */
const maps = computed(() => payload.value?.maps ?? null);
/** Maps the next search downloads for the drives outside the maps (one that failed is tried again; one bigger than a
 * search downloads is not): their road types are still to find there. */
const mapsToGet = computed(() => (maps.value?.needed ?? []).some((item) => item.state !== 'too big'));
/** Prepared drives still without road types, none read yet, or maps to download first. */
const fresh = computed(() => payload.value !== null && (without.value > 0 || !payload.value.routes.length || mapsToGet.value));
/** Only maps to download: every prepared drive has its road types, but not where it went outside the maps. */
const onlyMaps = computed(() => fresh.value && without.value === 0 && !!payload.value?.routes.length);
/** Drives copied from the comma but not prepared yet: no pictures, so no road types until Prepare new drives. */
const waiting = computed(() => (props.status?.routes ?? []).filter((row) => ['collegare', 'estrarre', 'preannotare'].includes(row.state)).length);
const madeAt = computed(() => isoWhen(payload.value?.made_at));
/** What all the frames are worth to training: "7,759 usable in training · 66,268 to review". */
const reviewText = (counts: Partial<Pick<RoadTypesPayload['review'], 'usable' | 'cant_tell' | 'to_review'>> | undefined): string[] => [
  counts?.usable ? thousands(counts.usable) + ' usable in training' : '',
  counts?.cant_tell ? thousands(counts.cant_tell) + ' can\'t tell' : '',
  counts?.to_review ? thousands(counts.to_review) + ' to review' : '',
].filter(Boolean);

const sizeText = (bytes: number): string => bytes >= 1e9 ? (bytes / 1e9).toFixed(1) + ' GB' : Math.max(1, Math.round(bytes / 1e6)) + ' MB';
const andList = (names: string[]): string => names.length < 2 ? names.join('') : names.slice(0, -1).join(', ') + ' and ' + names.at(-1);
const nameOf = (item: RoadMapNeed): string => item.name ?? 'a map';
/** The drives outside the maps, and what the next search downloads: "2 drives outside the maps: Find again downloads
 * Alsace and Michigan (≈ 448 MB)."; then why the last download failed, and the maps more than a search downloads. */
const outsideText = computed(() => {
  const needed = maps.value?.needed ?? [];
  const items = needed.filter((item) => item.state !== 'too big');
  const total = items.reduce((sum, item) => sum + (item.bytes ?? 0), 0);
  const big = needed.filter((item) => item.state === 'too big');
  const failed = needed.find((item) => item.state === 'failed');
  return [
    plural(maps.value?.outside ?? 0, 'drive') + ' outside the maps' + (items.length
      ? ': ' + (fresh.value ? 'Find the road types' : 'Find again') + ' downloads ' + andList(items.map(nameOf))
        + (total ? ' (≈ ' + sizeText(total) + ')' : '') : '') + '.',
    failed ? 'The last download failed' + (failed.reason ? ': ' + failed.reason : '') + '.' : '',
    big.length ? andList(big.map(nameOf)) + (big.length === 1 ? ' is' : ' are') + ' more than the '
      + (maps.value?.max_gb ?? 5) + ' GB a search downloads.' : '',
  ].filter(Boolean).join(' ');
});
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
          <strong>{{ payload === null ? 'Road types from the map' : !fresh ? 'Road types are up to date' : onlyMaps ? 'Maps to download'
            : 'Road types to find' }}</strong>
          <span v-if="payload !== null && !fresh">Every prepared drive has its proposals, from the map{{ madeAt ? ' · ' + madeAt : '' }}. Only
            what you confirm or correct is used in training.</span>
          <span v-else-if="onlyMaps">Some drives went where the server has no map: the search downloads it first, then reads their road types
            there.</span>
          <span v-else>Reads where the car was and the type of road there on the map, for every picture of the prepared drives: proposals,
            used in training only once you confirm or correct them.</span>
        </div>
        <button type="button" class="btn" :class="{primary: fresh}" id="find-road-types" :disabled="!state?.allowed"
                @click="state && emit('run', 'road_types', state)">{{ fresh ? 'Find the road types' : 'Find again' }}</button>
      </div>
      <p v-if="idle && without > 0" class="import-why" id="rt-without">{{ plural(without, 'prepared drive has', 'prepared drives have') }}
        no road types yet.</p>
      <p v-if="idle && waiting > 0" class="import-why" id="rt-waiting">{{ plural(waiting, 'new drive needs', 'new drives need') }} their pictures
        first: <button type="button" class="link-button" @click="goTo('prepare')">Prepare new drives</button> in Drives.</p>
      <p v-if="idle && maps && maps.checked_at && maps.names.length" class="import-why" id="rt-maps">Maps: {{ maps.names.join(', ') }}.</p>
      <p v-if="idle && maps && maps.outside > 0" class="import-why" id="rt-outside">{{ outsideText }}</p>
      <p v-if="idle && maps && maps.nowhere > 0" class="import-why" id="rt-nowhere">{{ plural(maps.nowhere, 'drive has', 'drives have') }}
        positions on no map to download (the sea, a lake, a GPS fix that jumped): they stay without a road type there.</p>
      <p v-if="idle && state && !state.allowed" class="import-why">{{ sentence(state.reason) }}</p>
      <p v-if="job && !bigJob" class="import-last">Last search {{ startedWhen(job) }} ·
        {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
        <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>
    </section>

    <section class="card rt-drives" aria-labelledby="rt-drives-title">
      <div class="ds-head">
        <h2 id="rt-drives-title">By drive</h2>
        <span v-if="drives.length" class="ds-when" id="rt-review">{{ [plural(drives.length, 'drive'), ...reviewText(payload?.review)].join(' · ') }}</span>
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
              <span v-for="(item, index) in reviewText(drive)" :key="item" :class="{'rt-reviewed': index === 0 && drive.usable}">{{ item }}</span>
              <span class="rt-frames">{{ thousands(drive.frames) }} pictures</span>
            </p>
            <span class="data-chev rt-go" aria-hidden="true">›</span>
          </a></li>
        </ul>
      </template>
    </section>
  </template>
</template>
