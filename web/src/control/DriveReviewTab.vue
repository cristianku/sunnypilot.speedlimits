<script setup lang="ts">
// Data › Drive review: the recorded drives against the comma's own logs (drive-review.ts). First the search: Find
// episodes reads the logs of every drive not searched yet (POST /api/drive-review/scan-all: no networks, CPU only);
// then the episodes to review, the ones with more agreeing clues first, filtered by drive, kind, data and status; then
// every drive with the state of its logs. A tap on an episode or a drive opens the drive
// ("/drive-review?drive=<route>&episode=<id>", DriveReviewDrive.vue); its video plays even without logs.
import {computed, onBeforeUnmount, onMounted, ref, shallowRef, watch} from 'vue';
import {api, messageOf, post} from '../common/api';
import {plural} from '../common/format';
import {toast} from '../composables/toast';
import DriveReviewDrive from './DriveReviewDrive.vue';
import {CHANNEL_LABEL, type DriveRow, KIND_LABEL, kphText, lengthOf, type QueueItem, scanText, STATUS_LABEL} from './drive-review';
import SectionHead from './SectionHead.vue';
import Segmented from './Segmented.vue';
import {tabDrive} from './tabs';

defineProps<{readOnly: boolean}>();

// -- the drives ----------------------------------------------------------------------------------------------------------
const drives = shallowRef<DriveRow[] | null>(null);
const error = ref('');
/** The first day whose drives are reviewed ('' every drive): the vision network before it is not worth judging. */
const reviewFrom = ref('');
async function loadDrives(): Promise<void> {
  try {
    const answer = await api<{drives: DriveRow[]; review_from: string}>('/api/drive-review/drives');
    drives.value = answer.drives;
    // Never over a day being picked (the list reloads every 2 s while it searches).
    if (document.activeElement?.id !== 'dr-review-from' && typeof answer.review_from === 'string') {
      reviewFrom.value = answer.review_from;
    }
    error.value = '';
  } catch (problem) {
    error.value = messageOf(problem);
  }
}
const busy = computed(() => (drives.value ?? []).some((drive) => drive.scan.state === 'queued' || drive.scan.state === 'scanning'));
/** The drives reviewed: from the first day on (and those of a day unknown). */
const reviewed = computed(() => (drives.value ?? []).filter((drive) => !drive.before_review));
const early = computed(() => (drives.value ?? []).length - reviewed.value.length);
/** Drives whose logs wait for a search: not searched, out of date, or found on the server and not imported. */
const toSearch = computed(() => reviewed.value.filter((drive) => drive.scan.state === 'not_scanned' || drive.scan.state === 'outdated'
  || (drive.scan.state === 'no_logs' && (drive.logs_found ?? 0) > 0)).length);
const withoutLogs = computed(() => (drives.value ?? []).filter((drive) => drive.scan.state === 'no_logs' && !drive.logs_found).length);
const searched = computed(() => (drives.value ?? []).filter((drive) => drive.scan.state === 'done' || drive.scan.state === 'no_episodes').length);
const titleOf = (route: string): string => {
  const drive = drives.value?.find((item) => item.route === route);
  return drive?.title ?? drive?.when ?? route;
};

let fromTimer = 0;
/** A day picked: saved once it is a whole date. While a year is typed digit by digit the input passes through
 * 0002, 0020, 0202: those are never saved. */
function fromChanged(): void {
  window.clearTimeout(fromTimer);
  if (reviewFrom.value && Number(reviewFrom.value.slice(0, 4)) < 2020) return;
  fromTimer = window.setTimeout(() => void saveFrom(), 700);
}
async function saveFrom(): Promise<void> {
  try {
    const answer = await post<{review_from: string}>('/api/drive-review/settings', {review_from: reviewFrom.value});
    reviewFrom.value = answer.review_from;
    toast(answer.review_from ? 'Drives reviewed from ' + dayText(answer.review_from) + '.' : 'Every drive is reviewed.');
    await Promise.all([loadDrives(), loadQueue()]);
  } catch (problem) {
    toast(messageOf(problem), true);
    await loadDrives();
  }
}
const dayText = (day: string): string => new Date(day + 'T12:00:00').toLocaleDateString('en-GB', {day: 'numeric', month: 'short', year: 'numeric'});

const starting = ref(false);
async function searchAll(): Promise<void> {
  starting.value = true;
  try {
    const answer = await post<{queued: string[]}>('/api/drive-review/scan-all');
    toast(answer.queued.length ? 'Searching ' + plural(answer.queued.length, 'drive') + '.' : 'Nothing new to search.');
    await loadDrives();
  } catch (problem) {
    toast(messageOf(problem), true);
  } finally {
    starting.value = false;
  }
}
async function searchOne(route: string): Promise<void> {
  try {
    await post('/api/drive-review/drives/' + encodeURIComponent(route) + '/scan');
    await loadDrives();
  } catch (problem) {
    toast(messageOf(problem), true);
  }
}

// -- the queue -----------------------------------------------------------------------------------------------------------
type StatusFilter = 'open' | 'reviewed' | 'undeterminable' | 'all';
const STATUS_OPTIONS: {key: StatusFilter; label: string}[] = [{key: 'open', label: 'To review'}, {key: 'reviewed', label: 'Reviewed'},
  {key: 'undeterminable', label: 'Can\'t tell'}, {key: 'all', label: 'All'}];
const KIND_OPTIONS: [string, string][] = [['', 'Every kind'], ['intervention', 'Interventions'], ['disagreement', 'Vision ≠ map'],
  ['both', 'Both at once'], ['gas', 'Accelerator'], ['brake', 'Brake'], ['steer', 'Wheel held'], ['set_speed_manual', 'Set speed by hand'],
  ['disengage_user', 'Disengaged by the driver'], ['disengage_auto', 'Disengaged by itself'], ['disengage_unknown', 'Disengaged, cause unknown'],
  ['spot_check', 'Spot checks']];
const DATA_OPTIONS: [string, string][] = [['', 'Any data'], ['vision_and_map', 'Vision and map logged'], ['no_vision', 'No vision in the log'],
  ['no_map', 'No map in the log'], ['no_video', 'No video'], ['qlog', 'From qlogs only']];
const status = ref<StatusFilter>('open');
const kind = ref('');
const data = ref('');
const route = ref('');
const queue = shallowRef<QueueItem[] | null>(null);
const queueError = ref('');
let asked = 0;
async function loadQueue(): Promise<void> {
  const ticket = ++asked;
  const query = new URLSearchParams();
  if (status.value !== 'all') query.set('status', status.value);
  if (kind.value) query.set('kind', kind.value);
  if (data.value) query.set('data', data.value);
  if (route.value) query.set('route', route.value);
  try {
    const answer = await api<{episodes: QueueItem[]}>('/api/drive-review/episodes?' + query.toString());
    if (ticket !== asked) return;
    queue.value = answer.episodes;
    queueError.value = '';
  } catch (problem) {
    if (ticket === asked) queueError.value = messageOf(problem);
  }
}
watch([status, kind, data, route], () => void loadQueue());
const SHOWN = 40;
const showAll = ref(false);
const shown = computed(() => showAll.value ? queue.value ?? [] : (queue.value ?? []).slice(0, SHOWN));

/** What the comma had just before the episode, in a line: "vision 60 · map 120 · 98 km/h". */
function beforeText(item: QueueItem): string {
  const before = item.before;
  const parts = [
    'vision ' + (before.vision_logged ? before.vision_kph ?? 'none' : 'not logged'),
    'map ' + (before.map_logged ? before.map_kph ?? 'none' : 'not logged'),
  ];
  if (before.v_ego_kph !== null) parts.push(kphText(before.v_ego_kph));
  if (before.vision_model) parts.push('network ' + before.vision_model);
  return parts.join(' · ');
}
const episodeHref = (item: QueueItem): string => '/drive-review?drive=' + encodeURIComponent(item.route) + '&episode=' + item.id;
const driveHref = (drive: DriveRow): string => '/drive-review?drive=' + encodeURIComponent(drive.route);

// -- polling while a search runs, and back from a drive ----------------------------------------------------------------
let timer: number | undefined;
function schedule(): void {
  window.clearTimeout(timer);
  if (busy.value && !tabDrive.value) timer = window.setTimeout(() => void loadDrives().then(schedule), 2000);
}
watch(busy, (now, before) => {
  schedule();
  if (before && !now) void loadQueue();
});
watch(tabDrive, (now, before) => {
  if (before && !now) void Promise.all([loadDrives(), loadQueue()]).then(schedule);
});
onMounted(() => {
  if (!tabDrive.value) void Promise.all([loadDrives(), loadQueue()]).then(schedule);
});
onBeforeUnmount(() => window.clearTimeout(timer));
</script>

<template>
  <DriveReviewDrive v-if="tabDrive" :route="tabDrive" :read-only="readOnly"/>
  <template v-else>
    <SectionHead tab="drivereview"/>

    <section class="card import-card" id="dr-search" aria-labelledby="dr-search-title">
      <h2 id="dr-search-title" class="sr-only">Search the logs</h2>
      <div class="import-row">
        <span class="import-icon" :class="{fresh: toSearch > 0 || busy}" aria-hidden="true">
          <span v-if="busy || (!drives && !error)" class="dr-spin"></span><template v-else-if="toSearch || !drives">!</template><template v-else>✓</template></span>
        <div class="import-what">
          <strong>{{ !drives ? (error ? 'The drives could not be read' : 'Reading the drives…') : busy ? 'Searching the logs…'
            : toSearch ? plural(toSearch, 'drive', 'drives') + ' to search' : 'Every drive with logs is searched' }}</strong>
          <span>Reads the comma's logs (rlog, qlog) for the moments worth a look: the driver taking over, or the vision limit disagreeing
            with the map's. No networks: only the logs, on the CPU.</span>
        </div>
        <button type="button" class="btn" :class="{primary: toSearch > 0}" id="dr-search-all" :disabled="readOnly || starting || busy || !toSearch"
                @click="searchAll">Find episodes</button>
      </div>
      <div class="dr-search-facts">
        <p v-if="drives" class="import-why">{{ plural(searched, 'drive') }} searched<template v-if="withoutLogs"> ·
          {{ plural(withoutLogs, 'drive') }} without logs on the server</template><template v-if="early"> ·
          {{ plural(early, 'drive') }} before {{ dayText(reviewFrom) }} left out</template>.</p>
        <label class="dr-from" title="Drives before this day are not searched and stay out of the episodes: the vision network on the car was not yet the one to judge. Empty: every drive.">
          <span>Drives from</span>
          <input type="date" id="dr-review-from" v-model="reviewFrom" :disabled="readOnly" @change="fromChanged"></label>
      </div>
    </section>

    <section class="card dr-queue" aria-labelledby="dr-queue-title">
      <div class="ds-head">
        <h2 id="dr-queue-title">Episodes</h2>
        <span v-if="queue" class="ds-when">{{ plural(queue.length, 'episode') }}</span>
      </div>
      <Segmented v-model="status" :options="STATUS_OPTIONS" label="Status" class="dr-status"/>
      <div class="dr-filters">
        <label class="wp-field"><span>Drive</span>
          <select v-model="route" id="dr-filter-drive">
            <option value="">Every drive</option>
            <option v-for="drive in reviewed" :key="drive.route" :value="drive.route"
                    :disabled="drive.scan.state !== 'done' && drive.scan.state !== 'no_episodes'">{{ drive.title ?? drive.when ?? drive.route }}</option>
          </select></label>
        <label class="wp-field"><span>Kind</span>
          <select v-model="kind" id="dr-filter-kind"><option v-for="[key, label] in KIND_OPTIONS" :key="key" :value="key">{{ label }}</option></select></label>
        <label class="wp-field"><span>Data</span>
          <select v-model="data" id="dr-filter-data"><option v-for="[key, label] in DATA_OPTIONS" :key="key" :value="key">{{ label }}</option></select></label>
      </div>
      <p v-if="queueError" class="empty">{{ queueError }}</p>
      <p v-else-if="queue === null" class="empty">Loading…</p>
      <p v-else-if="!queue.length" class="empty">{{ status === 'open' && !kind && !data && !route
        ? (searched ? 'Nothing left to review.' : 'No episodes yet: Find episodes searches the logs of the drives.')
        : 'No episode matches these filters.' }}</p>
      <ul v-else class="dr-list" id="dr-episodes">
        <li v-for="item in shown" :key="item.id">
          <a class="dr-item" :href="episodeHref(item)">
            <span class="dr-item-head">
              <strong>{{ CHANNEL_LABEL[item.channel] }}</strong>
              <span class="pill" :class="item.status === 'open' ? 'todo' : item.status === 'reviewed' ? 'ok' : 'muted'">{{ STATUS_LABEL[item.status] }}</span>
            </span>
            <span class="dr-item-where"><b v-if="item.when" class="dr-item-when">{{ item.when_certain ? '' : '≈ ' }}{{ item.when }}</b><template
              v-if="item.when"> · </template>{{ titleOf(item.route) }} · {{ lengthOf(item.duration_s) }}</span>
            <span class="dr-chips">
              <span v-for="name in item.kinds" :key="name" class="dr-chip" :class="'dr-k-' + name">{{ KIND_LABEL[name] ?? name }}</span>
            </span>
            <span class="dr-item-facts">Before: {{ beforeText(item) }}<template v-if="item.hint"> · driver {{ item.hint.speed_kph }} km/h,
              closer to the {{ item.hint.closer_to ?? 'neither' }} (a hint)</template></span>
            <span v-if="!item.video || item.data.log === 'qlog' || item.data.log === 'mixed'" class="dr-item-warn">{{
              [!item.video ? 'no video' : null, item.data.log === 'qlog' ? 'qlog only: fewer messages a second' : item.data.log === 'mixed' ? 'part qlog' : null]
                .filter(Boolean).join(' · ') }}</span>
            <span class="data-chev dr-go" aria-hidden="true">›</span>
          </a>
        </li>
      </ul>
      <button v-if="queue && queue.length > SHOWN && !showAll" type="button" class="btn small" @click="showAll = true">
        Show all {{ queue.length }}</button>
    </section>

    <section class="card dr-drives" aria-labelledby="dr-drives-title">
      <div class="ds-head">
        <h2 id="dr-drives-title">Drives</h2>
        <span v-if="drives" class="ds-when">{{ plural(drives.length, 'drive') }}</span>
      </div>
      <p v-if="error" class="empty">{{ error }}</p>
      <p v-else-if="drives === null" class="empty">Loading…</p>
      <p v-else-if="!drives.length" class="empty">No drives on the server yet.</p>
      <ul v-else class="rt-list">
        <li v-for="drive in drives" :key="drive.route" class="rt-drive dr-drive">
          <a class="rt-open" :href="driveHref(drive)">
            <div class="rt-name">
              <strong>{{ drive.title ?? 'Places unknown' }}</strong>
              <span><template v-if="drive.when">{{ drive.when_certain ? '' : '≈ ' }}{{ drive.when }}</template>{{
                drive.minutes !== null ? ' · ' + drive.minutes + ' min' : '' }}</span>
            </div>
            <p class="dr-drive-state" :class="drive.before_review ? 'dr-muted' : 'dr-' + scanText(drive.scan).tone">
              <template v-if="drive.before_review">Before {{ dayText(reviewFrom) }}: not reviewed.</template>
              <template v-else-if="drive.scan.state === 'no_logs' && drive.logs_found">Logs found on the server, not imported yet: Import and search reads them.</template>
              <template v-else>{{ scanText(drive.scan).text }}</template>
              <template v-if="drive.scan.state === 'done' && !drive.before_review"> {{ drive.episodes.open ? plural(drive.episodes.open, 'episode') + ' to review' : 'All reviewed' }}<template
                v-if="drive.episodes.reviewed + drive.episodes.undeterminable"> · {{ drive.episodes.reviewed + drive.episodes.undeterminable }} done</template>.</template>
            </p>
            <p class="rt-shares">
              <span v-if="drive.scan.kinds.length">{{ drive.scan.kinds.join(' + ') }} · {{ plural(drive.scan.segments, 'minute') }} of logs</span>
              <span v-if="drive.scan.dashcam === true">openpilot never engaged</span>
              <span>{{ drive.video_segments ? plural(drive.video_segments, 'minute') + ' of video' : 'no video' }}</span>
              <span v-if="drive.annotations" class="rt-reviewed">{{ plural(drive.annotations, 'note') }}</span>
              <span class="rt-frames dr-route">{{ drive.route }}</span>
            </p>
            <span class="data-chev rt-go" aria-hidden="true">›</span>
          </a>
          <div v-if="!readOnly && !drive.before_review && (drive.scan.state === 'not_scanned' || drive.scan.state === 'outdated' || drive.scan.state === 'failed'
                 || (drive.scan.state === 'no_logs' && drive.logs_found))" class="dr-drive-actions">
            <button type="button" class="btn small" @click="searchOne(drive.route)">{{ drive.scan.state === 'no_logs' ? 'Import and search' : 'Search again' }}</button>
          </div>
        </li>
      </ul>
    </section>
  </template>
</template>
