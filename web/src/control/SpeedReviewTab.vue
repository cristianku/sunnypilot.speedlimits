<script setup lang="ts">
// Data › Speed review, the answers the brain learns from (speed_review.py): his own dashboard speed at every moment of a
// drive is the answer, with nothing to confirm; a block corrects it from a time to a time, with another speed or left out
// of the training, and roadworks are marked on the frames. Nothing here changes the car. Every drive the brain learns
// from (sunnypilot off in all of it, speed_review.dashcam_drive, or not read yet: DASHCAM MODE on the first), the newest
// first, with its blocks and roadworks marks; a tap opens it ("/speed-review?drive=<route>", SpeedReviewDrive.vue). Above the
// list, Read the logs reads the comma's logs of every drive (ReadLogsCard.vue), the list read again every 2 s while it
// does: a drive has a speed only once its logs are read.
import {computed, onBeforeUnmount, onMounted, ref, shallowRef, watch} from 'vue';
import {api, messageOf} from '../common/api';
import {isGuest} from '../common/guest';
import {plural} from '../common/format';
import {clockOf, scanText} from './drive-review';
import ReadLogsCard from './ReadLogsCard.vue';
import SectionHead from './SectionHead.vue';
import type {SpeedRow} from './speed-review';
import SpeedReviewDrive from './SpeedReviewDrive.vue';
import {tabDrive} from './tabs';

defineProps<{readOnly: boolean}>();

const drives = shallowRef<SpeedRow[] | null>(null);
const error = ref('');
async function load(): Promise<void> {
  try {
    drives.value = (await api<{drives: SpeedRow[]}>('/api/speed-review/drives')).drives;
    error.value = '';
  } catch (problem) {
    error.value = messageOf(problem);
  }
}
// By date and time ("when" ends in the time): the number of a drive is not its date, the comma's count started again.
const startOf = (drive: SpeedRow): string => (drive.day ?? '') + (drive.when ?? '').slice(-5);
const rows = computed(() => (drives.value ?? []).filter((drive) => drive.dashcam !== false)
  .sort((a, b) => startOf(b).localeCompare(startOf(a))));
/** Drives where sunnypilot was on: the brain does not learn from them, the list leaves them out. */
const engaged = computed(() => (drives.value ?? []).filter((drive) => drive.dashcam === false).length);
/** What a drive has: its length, its blocks and its roadworks marks; before its logs are read, what reading them does. */
const stateOf = (drive: SpeedRow): string => !drive.total_s ? scanText(drive.scan).text
  : clockOf(drive.total_s) + ' long · ' + (drive.blocks ? plural(drive.blocks, 'block') : 'no block')
    + (drive.works ? ' · ' + plural(drive.works, 'roadworks', 'roadworks') : '');
const marked = (drive: SpeedRow): boolean => drive.blocks > 0 || drive.works > 0;
const hrefOf = (drive: SpeedRow): string => '/speed-review?drive=' + encodeURIComponent(drive.route);

// While logs are read, the list again every 2 s (never under an opened drive).
const reading = computed(() => (drives.value ?? []).some((drive) => drive.scan.state === 'queued' || drive.scan.state === 'scanning'));
let timer: number | undefined;
function schedule(): void {
  window.clearTimeout(timer);
  if (reading.value && tabDrive.value === null) timer = window.setTimeout(() => void load().then(schedule), 2000);
}
const reload = (): Promise<void> => load().then(schedule);
// Back on the list: the blocks and the roadworks may have changed.
watch(tabDrive, (route) => { if (route === null) void reload(); else window.clearTimeout(timer); });
onMounted(() => { if (tabDrive.value === null) void reload(); });
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
        time to a time, with that speed or left out of the training; mark the roadworks there too. Nothing here changes the car.</p>
      <p v-if="engaged" class="import-why" id="sr-engaged">Only drives with sunnypilot off (sunnylink › Toggles › Enable
        sunnypilot): there the speed is all yours. {{ plural(engaged, 'drive') }} with sunnypilot on {{ engaged === 1 ? 'is' : 'are' }} not listed.</p>
      <p v-if="error" class="empty">{{ error }}</p>
      <p v-else-if="drives === null" class="empty">Loading…</p>
      <ul v-else class="rt-list">
        <li v-for="drive in rows" :key="drive.route" class="rt-drive dr-drive">
          <a class="rt-open" :href="hrefOf(drive)">
            <div class="rt-name">
              <strong>{{ drive.title ?? 'Places unknown' }} <b v-if="drive.dashcam" class="dashcam-badge">DASHCAM MODE</b></strong>
              <span><template v-if="drive.when">{{ drive.when_certain ? '' : '≈ ' }}{{ drive.when }}</template>{{
                drive.minutes !== null ? ' · ' + drive.minutes + ' min' : '' }}</span>
            </div>
            <p class="dr-drive-state" :class="{'dr-muted': !marked(drive)}">{{ stateOf(drive) }}</p>
            <p class="rt-shares"><span class="rt-frames dr-route">{{ drive.route }}</span></p>
            <span class="data-chev rt-go" aria-hidden="true">›</span>
          </a>
        </li>
      </ul>
    </section>
  </template>
</template>
