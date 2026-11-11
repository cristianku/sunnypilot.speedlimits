<script lang="ts">
import type {SpeedRow} from './speed-review';

/** A drive's frames of its signs, as GET /api/routes sends them (review.route_summaries). */
interface SignDrive {
  route_id: string;
  frame_count: number;
  confirmed_count: number;
  waiting_for_person?: number;
  /** Where its signs are reviewed (review.signs_url): the Signs side, or the page of Sign review of a drive without video. */
  url?: string;
  drive?: {title: string | null; when: string | null; minutes: number | null; start?: string | null} | null;
  dashcam?: boolean | null;
}
/** The lists as they were last read, kept while the page is open: back on the tab (from another tab, or from a drive
 * opened by its address) they stand at once and are read again behind, as on the way back from a drive. */
const shown: {drives: SpeedRow[] | null; signs: SignDrive[] | null} = {drives: null, signs: null};
</script>

<script setup lang="ts">
// Data › Speed review, the answers the brain learns from (speed_review.py): his own dashboard speed at every moment of a
// drive is the answer, with nothing to confirm; a block corrects it from a time to a time, with another speed or left out
// of the training, and roadworks are marked on the frames. Nothing here changes the car. Every drive the brain learns
// from (sunnypilot off in all of it, speed_review.dashcam_drive, or not read yet: DASHCAM MODE on the first), the newest
// first, with its blocks, roadworks and kinds of road marked; a tap opens it ("/speed-review?drive=<route>", SpeedReviewDrive.vue). Above the
// list, Read the logs reads the comma's logs of every drive (ReadLogsCard.vue), the list read again every 2 s while it
// does: a drive has a speed only once its logs are read. One list since 11 Nov 2026: every drive is one row with its
// speed and its signs (GET /api/routes, sunnypilot on or off; a frame the AI saved waits for a person too), Signs beside
// it opening the drive on its Signs side at the first frame still to confirm; a drive with sunnypilot on is there only
// for its signs, and its row opens them. Every dashcam drive enters the brain training, each whole
// (brain_targets.training_routes): there is nothing to mark for it.
import {computed, onBeforeUnmount, onMounted, ref, shallowRef, watch} from 'vue';
import {api, messageOf} from '../common/api';
import {isGuest} from '../common/guest';
import {plural, thousands} from '../common/format';
import {signsHref} from '../common/sign-link';
import {clockOf, scanText} from './drive-review';
import ReadLogsCard from './ReadLogsCard.vue';
import SectionHead from './SectionHead.vue';
import SpeedReviewDrive from './SpeedReviewDrive.vue';
import {tabDrive} from './tabs';

defineProps<{readOnly: boolean}>();

const drives = shallowRef<SpeedRow[] | null>(shown.drives);
const error = ref('');
async function load(): Promise<void> {
  try {
    drives.value = shown.drives = (await api<{drives: SpeedRow[]}>('/api/speed-review/drives')).drives;
    error.value = '';
  } catch (problem) {
    error.value = messageOf(problem);
  }
}
// By date and time ("when" ends in the time): the number of a drive is not its date, the comma's count started again.
const startOf = (drive: SpeedRow): string => (drive.day ?? '') + (drive.when ?? '').slice(-5);
/** The drives the list can show: not those with sunnypilot on, the newest first. */
const listed = computed(() => (drives.value ?? []).filter((drive) => drive.dashcam !== false)
  .sort((a, b) => startOf(b).localeCompare(startOf(a))));

/** Drives where sunnypilot was on: the brain does not learn their speed, the list has them only for their signs. */
const engaged = computed(() => (drives.value ?? []).filter((drive) => drive.dashcam === false).length);
/** What a drive has: its length, its blocks, its roadworks marks and its marks of a kind of road; before its logs are read,
 * what reading them does. */
const stateOf = (drive: SpeedRow): string => !drive.total_s ? scanText(drive.scan).text
  : clockOf(drive.total_s) + ' long · ' + (drive.blocks ? plural(drive.blocks, 'block') : 'no block')
    + (drive.works ? ' · ' + plural(drive.works, 'roadworks', 'roadworks') : '')
    + (drive.road ? ' · ' + plural(drive.road, 'road mark') : '');
const marked = (drive: SpeedRow): boolean => drive.blocks > 0 || drive.works > 0 || drive.road > 0;
const hrefOf = (drive: SpeedRow): string => '/speed-review?drive=' + encodeURIComponent(drive.route);

const signDrives = shallowRef<SignDrive[] | null>(shown.signs);
/** One row a drive: a dashcam drive with its speed, and every drive with frames of its signs, the newest first. */
interface Row {
  route: string;
  speed: SpeedRow | null;
  signs: SignDrive | null;
}
/** When a drive started, in ms: its signs' start (UTC), else its speed's local day and time. */
const timeOf = (row: Row): number => {
  const start = row.signs?.drive?.start;
  if (start) return Date.parse(start);
  const speed = row.speed;
  return speed?.day && speed.when ? Date.parse(speed.day + 'T' + speed.when.slice(-5)) || 0 : 0;
};
const rows = computed((): Row[] => {
  const byRoute = new Map<string, Row>();
  for (const drive of listed.value) byRoute.set(drive.route, {route: drive.route, speed: drive, signs: null});
  for (const item of signDrives.value ?? []) {
    if (item.frame_count <= 0) continue;
    const row = byRoute.get(item.route_id);
    if (row) row.signs = item;
    else byRoute.set(item.route_id, {route: item.route_id, speed: null, signs: item});
  }
  return [...byRoute.values()].sort((a, b) => timeOf(b) - timeOf(a));
});
const titleOf = (row: Row): string => row.speed?.title ?? row.signs?.drive?.title ?? 'Places unknown';
const whenOf = (row: Row): string => {
  const when = row.speed?.when ? (row.speed.when_certain ? '' : '≈ ') + row.speed.when : row.signs?.drive?.when ?? '';
  const minutes = row.speed?.minutes ?? row.signs?.drive?.minutes ?? null;
  return when + (minutes !== null ? ' · ' + minutes + ' min' : '');
};
const signsUrl = (item: SignDrive): string => item.url ?? signsHref(item.route_id);
/** Its signs in words: "160 of 1,062 frames confirmed", "1,062 frames confirmed" when all are. */
const signsText = (item: SignDrive): string => byPerson(item) >= item.frame_count
  ? 'Signs: all ' + thousands(item.frame_count) + ' frames confirmed'
  : 'Signs: ' + thousands(byPerson(item)) + ' of ' + thousands(item.frame_count) + ' frames confirmed'
    + (item.waiting_for_person ? ' · ' + thousands(item.waiting_for_person) + ' saved by the AI, to confirm' : '');
async function loadSigns(): Promise<void> {
  // The administrator's; a server without the list, or a failure, leaves the card out quietly.
  if (isGuest()) return;
  try {
    const answer = await api<{routes: SignDrive[]}>('/api/routes');
    signDrives.value = shown.signs = Array.isArray(answer?.routes) ? answer.routes : null;
  } catch {
    signDrives.value = shown.signs = null;
  }
}
/** The frames a person confirmed: confirmed_count counts the AI's saves too, which wait for a person. */
const byPerson = (drive: SignDrive): number => Math.max(0, drive.confirmed_count - (drive.waiting_for_person ?? 0));
const signsLeft = (item: SignDrive): boolean => byPerson(item) < item.frame_count;

// While logs are read, the list again every 2 s (never under an opened drive).
const reading = computed(() => (drives.value ?? []).some((drive) => drive.scan.state === 'queued' || drive.scan.state === 'scanning'));
let timer: number | undefined;
function schedule(): void {
  window.clearTimeout(timer);
  if (reading.value && tabDrive.value === null) timer = window.setTimeout(() => void load().then(schedule), 2000);
}
const reload = (): Promise<void> => load().then(schedule);
// Back on the list: the blocks and the roadworks may have changed.
watch(tabDrive, (route) => {
  if (route !== null) window.clearTimeout(timer);
  else {
    void reload();
    void loadSigns();
  }
});
onMounted(() => {
  if (tabDrive.value !== null) return;
  void reload();
  void loadSigns();
});
onBeforeUnmount(() => window.clearTimeout(timer));
</script>

<template>
  <SpeedReviewDrive v-if="tabDrive" :key="tabDrive" :route="tabDrive" :read-only="readOnly"/>
  <template v-else>
    <SectionHead tab="speedreview"/>
    <ReadLogsCard v-if="!isGuest()" :drives="drives" :error="error" :read-only="readOnly" @changed="reload"/>
    <section class="card sr-drives" aria-labelledby="sr-drives-title">
      <div class="ds-head">
        <h2 id="sr-drives-title">Drives</h2>
        <span v-if="drives" class="ds-when">{{ plural(rows.length, 'drive') }}</span>
      </div>
      <p class="import-why">The brain network learns the speed to hold from your own driving: your dashboard speed at every moment
        is its answer, with nothing to confirm. Where it should have been another speed, open the drive and put a block from a
        time to a time, with that speed or left out of the training; mark the roadworks and the kinds of road there too.
        <b>Signs</b> opens the drive on its signs, at the first frame still to confirm. Nothing here changes the car.</p>
      <p v-if="engaged" class="import-why" id="sr-engaged">The speed is yours only with sunnypilot off (sunnylink › Toggles ›
        Enable sunnypilot): {{ plural(engaged, 'drive') }} with sunnypilot on {{ engaged === 1 ? 'is' : 'are' }} here only for
        {{ engaged === 1 ? 'its' : 'their' }} signs.</p>
      <p v-if="error" class="empty">{{ error }}</p>
      <p v-else-if="drives === null" class="empty">Loading…</p>
      <ul v-else class="rt-list">
        <li v-for="row in rows" :key="row.route" class="rt-drive dr-drive sr-row">
          <a class="rt-open" :href="row.speed ? hrefOf(row.speed) : signsUrl(row.signs!)">
            <div class="rt-name">
              <strong>{{ titleOf(row) }} <b v-if="row.speed?.dashcam" class="dashcam-badge">DASHCAM MODE</b></strong>
              <span>{{ whenOf(row) }}</span>
            </div>
            <p v-if="row.speed" class="dr-drive-state" :class="{'dr-muted': !marked(row.speed)}">{{ stateOf(row.speed) }}</p>
            <p v-if="row.signs" class="dr-drive-state" :class="{'dr-muted': !signsLeft(row.signs)}">{{ signsText(row.signs) }}</p>
            <p class="rt-shares"><span class="rt-frames dr-route">{{ row.route }}</span></p>
            <span v-if="!row.speed || !row.signs" class="data-chev rt-go" aria-hidden="true">›</span>
          </a>
          <a v-if="row.speed && row.signs" class="sr-signs" :class="{'sr-signs-done': !signsLeft(row.signs)}"
             :href="signsUrl(row.signs)" :title="'Open the signs of this drive at the first frame still to confirm'">Signs ›</a>
        </li>
      </ul>
    </section>
  </template>
</template>
