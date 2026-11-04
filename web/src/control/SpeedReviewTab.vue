<script setup lang="ts">
// Data › Speed review, where a person says the speed the car should hold, for the brain to learn from (speed_review.py):
// nothing here changes the car. Every drive, the newest first; a tap opens it ("/speed-review?drive=<route>",
// SpeedReviewDrive.vue). Drive review stays what it is: a place to look.
import {computed, onMounted, ref, shallowRef, watch} from 'vue';
import {api, messageOf} from '../common/api';
import {plural} from '../common/format';
import {clockOf, scanText} from './drive-review';
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
const rows = computed(() => [...(drives.value ?? [])].sort((a, b) => startOf(b).localeCompare(startOf(a))));
const progressOf = (drive: SpeedRow): string => !drive.total_s ? scanText(drive.scan).text : !drive.examined_s ? 'Not started.'
  : clockOf(drive.examined_s) + ' of ' + clockOf(drive.total_s) + ' examined.';
const stateOf = (drive: SpeedRow): string => (drive.first && drive.examined_s < drive.total_s ? 'To review first · ' : '') + progressOf(drive);
const shareOf = (drive: SpeedRow): string => (drive.total_s ? drive.examined_s / drive.total_s * 100 : 0) + '%';
const hrefOf = (drive: SpeedRow): string => '/speed-review?drive=' + encodeURIComponent(drive.route);

// Back on the list: the shares may have changed.
watch(tabDrive, (route) => { if (route === null) void load(); });
onMounted(() => { if (tabDrive.value === null) void load(); });
</script>

<template>
  <SpeedReviewDrive v-if="tabDrive" :key="tabDrive" :route="tabDrive" :read-only="readOnly"/>
  <template v-else>
    <SectionHead tab="speedreview"/>
    <section class="card sr-drives" aria-labelledby="sr-drives-title">
      <div class="ds-head">
        <h2 id="sr-drives-title">Drives</h2>
        <span v-if="drives" class="ds-when">{{ plural(rows.length, 'drive') }}</span>
      </div>
      <p class="import-why">Say the speed the car should hold along a drive: the limit the comma's vision held is proposed, you confirm or correct
        it. The brain network learns from these labels; nothing here changes the car.</p>
      <p v-if="error" class="empty">{{ error }}</p>
      <p v-else-if="drives === null" class="empty">Loading…</p>
      <ul v-else class="rt-list">
        <li v-for="drive in rows" :key="drive.route" class="rt-drive dr-drive">
          <a class="rt-open" :href="hrefOf(drive)">
            <div class="rt-name">
              <strong>{{ drive.title ?? 'Places unknown' }}</strong>
              <span><template v-if="drive.when">{{ drive.when_certain ? '' : '≈ ' }}{{ drive.when }}</template>{{
                drive.minutes !== null ? ' · ' + drive.minutes + ' min' : '' }}</span>
            </div>
            <p class="dr-drive-state" :class="{'dr-muted': !drive.examined_s}">{{ stateOf(drive) }}</p>
            <div v-if="drive.examined_s" class="sr-bar" aria-hidden="true"><i :style="{width: shareOf(drive)}"></i></div>
            <p class="rt-shares"><span class="rt-frames dr-route">{{ drive.route }}</span></p>
            <span class="data-chev rt-go" aria-hidden="true">›</span>
          </a>
        </li>
      </ul>
    </section>
  </template>
</template>
