<script setup lang="ts">
// Read the logs, at the top of Data › Speed review: reads the comma's logs (rlog, qlog) of every drive not read yet
// (POST /api/drive-review/scan-all: no networks, CPU only), what the proposals there, Sign distance, Models › Drive and the
// brain's training read. The drives and the state of their logs are the list's (SpeedReviewTab.vue, the same rows as
// GET /api/speed-review/drives); the list reads them again every 2 s while logs are read. A single drive is read from
// its player (Search the logs).
import {computed, ref} from 'vue';
import {messageOf, post} from '../common/api';
import {plural} from '../common/format';
import {toast} from '../composables/toast';
import type {DriveRow} from './drive-review';

const props = defineProps<{drives: DriveRow[] | null; error: string; readOnly: boolean}>();
const emit = defineEmits<{changed: []}>();

const rows = computed(() => props.drives ?? []);
const busy = computed(() => rows.value.some((drive) => drive.scan.state === 'queued' || drive.scan.state === 'scanning'));
/** Drives whose logs wait to be read, by the server's one rule (DriveReviewStore.needs_scan, the same as Sign distance):
 * not read, out of date, a reading that stopped, logs not imported yet or readable again; never one before the first
 * day read. */
const toRead = computed(() => rows.value.filter((drive) => !drive.before_review && drive.needs_scan).length);
const read = computed(() => rows.value.filter((drive) => drive.scan.state === 'done' || drive.scan.state === 'no_episodes').length);
const withoutLogs = computed(() => rows.value.filter((drive) => drive.scan.state === 'no_logs' && !drive.logs_found).length);
/** Left out by the first day read (DriveReviewStore.review_from): their player still reads them one by one. */
const early = computed(() => rows.value.filter((drive) => drive.before_review && drive.needs_scan).length);

const starting = ref(false);
async function readAll(): Promise<void> {
  starting.value = true;
  try {
    const answer = await post<{queued: string[]}>('/api/drive-review/scan-all');
    toast(answer.queued.length ? 'Reading the logs of ' + plural(answer.queued.length, 'drive') + '.' : 'Nothing new to read.');
    emit('changed');
  } catch (problem) {
    toast(messageOf(problem), true);
  } finally {
    starting.value = false;
  }
}
</script>

<template>
  <section class="card import-card" id="read-logs" aria-labelledby="rl-title">
    <h2 id="rl-title" class="sr-only">The comma's logs</h2>
    <div class="import-row">
      <span class="import-icon" :class="{fresh: toRead > 0 || busy}" aria-hidden="true">
        <span v-if="busy || (!drives && !error)" class="dr-spin"></span><template v-else-if="toRead || !drives">!</template><template v-else>✓</template></span>
      <div class="import-what">
        <strong>{{ !drives ? (error ? 'The drives could not be read' : 'Reading the drives…') : busy ? 'Reading the logs…'
          : toRead ? 'Logs of ' + plural(toRead, 'drive') + ' to read' : 'The logs of every drive are read' }}</strong>
        <span>Reads the comma's logs (rlog, qlog) of the drives: the limits it had, the car's speed and what openpilot did, for the
          answers of Speed review, Sign distance, Models › Drive and the brain's training. No networks: only the logs, on the CPU.</span>
      </div>
      <button type="button" class="btn" :class="{primary: toRead > 0}" id="read-logs-all" :disabled="readOnly || starting || busy || !toRead"
              @click="readAll">Read the logs</button>
    </div>
    <p v-if="drives" class="import-why" id="rl-facts">{{ plural(read, 'drive') }} read<template v-if="withoutLogs"> ·
      {{ plural(withoutLogs, 'drive') }} without logs on the server</template><template v-if="early"> ·
      {{ plural(early, 'older drive') }} left out (each is read from its player)</template>.</p>
  </section>
</template>
