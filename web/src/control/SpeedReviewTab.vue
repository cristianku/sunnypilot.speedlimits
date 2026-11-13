<script lang="ts">
import type {SpeedRow} from './speed-review';

/** A drive's frames of its signs, as GET /api/routes sends them (review.route_summaries). */
interface SignDrive {
  route_id: string;
  frame_count: number;
  confirmed_count: number;
  waiting_for_person?: number;
  /** The frames of it the AI reviewer saved (reviewer_kind ai), in confirmed_count too. */
  confirmed_by_ai?: number;
  /** Where its signs are reviewed (review.signs_url): the Signs side, or the page of Sign review of a drive without video. */
  url?: string;
  drive?: {title: string | null; when: string | null; minutes: number | null; start?: string | null} | null;
  dashcam?: boolean | null;
}
/** The lists as they were last read, kept while the page is open: back on the tab (from another tab, or from a drive
 * opened by its address) they stand at once and are read again behind, as on the way back from a drive. */
const shown: {drives: SpeedRow[] | null; signs: SignDrive[] | null} = {drives: null, signs: null};

/** The groups of the banner over the list, by the signs of a drive (as Sign review's list grouped them): some frames
 * confirmed, none yet, all; and the drives the AI reviewed part of, whatever their group. */
type Group = 'doing' | 'todo' | 'ai' | 'done';
const GROUPS: readonly {key: Group; label: string}[] = [{key: 'todo', label: 'To review'},
  {key: 'doing', label: 'Started, not finished'}, {key: 'ai', label: 'Reviewed by AI'}, {key: 'done', label: 'Done'}];
/** The group the list shows (?show=<group> in the address, kept while the page is open); null: every drive. */
const asked = new URLSearchParams(location.search).get('show');
let showing: Group | null = GROUPS.some((group) => group.key === asked) ? asked as Group : null;
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
import AiRouteSheet from '../components/AiRouteSheet.vue';
import Icon from '../components/Icon.vue';
import ReadLogsCard from './ReadLogsCard.vue';
import SectionHead from './SectionHead.vue';
import SpeedReviewDrive from './SpeedReviewDrive.vue';
import {tabDrive} from './tabs';

const props = defineProps<{readOnly: boolean}>();
/** "Review with AI" of a drive's signs, beside Signs (components/AiRouteSheet.vue): the administrator's, where it can save. */
const canAi = !props.readOnly && !isGuest();
const aiLeft = (signs: SignDrive): number => Math.max(0, signs.frame_count - signs.confirmed_count);

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
const byAi = (item: SignDrive): number => Math.min(item.confirmed_by_ai ?? 0, item.confirmed_count);
function inGroup(item: SignDrive, group: Group): boolean {
  if (group === 'ai') return byAi(item) > 0;
  const confirmed = Math.min(item.confirmed_count, item.frame_count);
  return group === 'done' ? confirmed >= item.frame_count : group === 'todo' ? confirmed === 0
    : confirmed > 0 && confirmed < item.frame_count;
}
const show = ref<Group | null>(showing);
/** A tile of the banner: the list of its group, again: every drive. The address keeps it (back from a drive, a reload). */
function pick(group: Group): void {
  show.value = showing = show.value === group ? null : group;
  const url = new URL(location.href);
  if (showing) url.searchParams.set('show', showing);
  else url.searchParams.delete('show');
  history.replaceState(history.state, '', url.pathname + url.search + url.hash);
}
const withSigns = computed(() => rows.value.flatMap((row) => row.signs ? [row.signs] : []));
/** The banner: the frames confirmed of every drive, by you and by the AI, and how many drives each group has. */
const glance = computed(() => {
  const items = withSigns.value;
  const frames = items.reduce((total, item) => total + item.frame_count, 0);
  const confirmed = items.reduce((total, item) => total + Math.min(item.confirmed_count, item.frame_count), 0);
  const ai = items.reduce((total, item) => total + byAi(item), 0);
  const tiles = GROUPS.map((group) => {
    const members = items.filter((item) => inGroup(item, group.key));
    const left = members.reduce((total, item) => total + Math.max(0, item.frame_count - item.confirmed_count), 0);
    const sub = !members.length ? 'none' : group.key === 'ai' ? thousands(members.reduce((total, item) => total + byAi(item), 0)) + ' frames'
      : group.key === 'done' ? thousands(members.reduce((total, item) => total + item.frame_count, 0)) + ' frames' : thousands(left) + ' left';
    return {...group, count: members.length, sub};
  });
  return {frames, confirmed, ai, percent: frames ? Math.floor(confirmed * 1000 / frames) / 10 : 0,
    people: frames ? (confirmed - ai) * 100 / frames : 0, aiShare: frames ? ai * 100 / frames : 0, drives: items.length, tiles};
});
/** The rows of the group shown (a drive without signs only in every drive). */
const shownRows = computed(() => show.value === null ? rows.value
  : rows.value.filter((row) => row.signs !== null && inGroup(row.signs, show.value!)));
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
        <span v-if="drives" class="ds-when">{{ show ? shownRows.length + ' of ' : '' }}{{ plural(rows.length, 'drive') }}</span>
      </div>
      <div v-if="glance.drives" class="rv-overview sr-glance" id="sr-glance" aria-label="The signs of the drives at a glance">
        <p class="rv-big"><strong>{{ glance.percent }}%</strong><span>of the frames confirmed · {{ thousands(glance.confirmed) }} of
          {{ thousands(glance.frames) }} in {{ plural(glance.drives, 'drive') }}</span></p>
        <span class="rv-bar" aria-hidden="true"><span :style="{width: glance.people + '%'}"></span><span v-if="glance.ai" class="ai"
          :style="{width: glance.aiShare + '%'}"></span></span>
        <p v-if="glance.ai" class="rv-notes">{{ thousands(glance.confirmed - glance.ai) }} by you · <span class="sr-ai-word">{{ thousands(glance.ai) }} by the AI</span></p>
        <div class="rv-tiles" role="group" aria-label="Show only">
          <button v-for="tile in glance.tiles" :key="tile.key" type="button" class="rv-tile" :class="tile.key" :id="'sr-show-' + tile.key"
                  :aria-pressed="show === tile.key" :disabled="!tile.count && show !== tile.key"
                  :title="show === tile.key ? 'Show every drive again' : 'Show only these drives'" @click="pick(tile.key)">
            <span class="rv-tile-label"><i v-if="tile.key === 'ai'" class="sr-ai-square" aria-hidden="true"></i>{{ tile.label }}</span>
            <span class="rv-tile-n">{{ tile.count }}</span><span class="rv-tile-sub">{{ tile.sub }}</span>
          </button>
        </div>
      </div>
      <p v-if="error" class="empty">{{ error }}</p>
      <p v-else-if="drives === null" class="empty">Loading…</p>
      <p v-else-if="show && !shownRows.length" class="empty">No drive here. <button type="button" class="link-button" @click="pick(show)">Show every drive</button></p>
      <ul v-else class="rt-list">
        <li v-for="row in shownRows" :key="row.route" class="rt-drive dr-drive sr-row">
          <a class="rt-open" :href="row.speed ? hrefOf(row.speed) : signsUrl(row.signs!)">
            <div class="rt-name">
              <strong>{{ titleOf(row) }} <b v-if="row.speed?.dashcam" class="dashcam-badge">DASHCAM MODE</b>
                <b v-if="row.signs && byAi(row.signs)" class="ai-badge" :title="thousands(byAi(row.signs)) + ' frames of its signs reviewed by the AI'"><i class="sr-ai-square" aria-hidden="true"></i>AI REVIEW</b></strong>
              <span>{{ whenOf(row) }}</span>
            </div>
            <p v-if="row.speed" class="dr-drive-state" :class="{'dr-muted': !marked(row.speed)}">{{ stateOf(row.speed) }}</p>
            <p v-if="row.signs" class="dr-drive-state" :class="{'dr-muted': !signsLeft(row.signs)}">{{ signsText(row.signs) }}</p>
            <p class="rt-shares"><span class="rt-frames dr-route">{{ row.route }}</span></p>
            <span v-if="!row.speed || !row.signs" class="data-chev rt-go" aria-hidden="true">›</span>
          </a>
          <a v-if="row.speed && row.signs" class="sr-signs" :class="{'sr-signs-done': !signsLeft(row.signs)}"
             :href="signsUrl(row.signs)" :title="'Open the signs of this drive at the first frame still to confirm'">Signs ›</a>
          <button v-if="canAi && row.signs && aiLeft(row.signs) > 0" type="button" class="ai-route sr-ai-row"
                  :data-ai-route="row.route" :data-ai-name="titleOf(row)" :data-ai-left="aiLeft(row.signs)"
                  title="The AI reviewer of Settings reviews the frames of the signs still to confirm: all signs, or only those at a traffic light"><Icon name="ai"/>Review with AI</button>
        </li>
      </ul>
    </section>
    <Teleport v-if="canAi" to="body"><AiRouteSheet/></Teleport>
  </template>
</template>
