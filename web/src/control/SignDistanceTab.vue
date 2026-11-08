<script setup lang="ts">
// Data › Sign distance: how late the car's countdown to the sign ahead reaches zero (sign-distance.ts). First what the
// marks say so far: the median of late_s over every sign marked (the server's figure, never worked out here) and, as a
// second figure, in metres; then the drives: those with a sign to compare first, a tap opens one
// ("/sign-distance?drive=<route>&sign=<key>", SignDistanceDrive.vue), then the others with what keeps them out (logs
// not read yet, unreadable, no video yet, a build too old, no sign seen, every sign left out). Nothing here writes:
// the marks are made on a drive, by a person signed in.
import {computed, onBeforeUnmount, onMounted, ref, shallowRef, watch} from 'vue';
import {api, messageOf} from '../common/api';
import {plural} from '../common/format';
import SectionHead from './SectionHead.vue';
import SignDistanceDrive from './SignDistanceDrive.vue';
import {API, type Drive, driveHref, EXCLUDED_LABEL, type Excluded, lateSentence, signedMetres, signedSeconds, statusTone,
  type Summary} from './sign-distance';
import {tabDrive} from './tabs';

/** null until the server's status is in: then neither the mark buttons nor the read-only note are shown. */
defineProps<{readOnly: boolean | null}>();

const drives = shallowRef<Drive[] | null>(null);
const summary = shallowRef<Summary | null>(null);
const error = ref('');
async function load(): Promise<void> {
  try {
    const [list, figures] = await Promise.all([api<{drives: Drive[]}>(API + '/drives'), api<{summary: Summary}>(API + '/summary')]);
    if (!Array.isArray(list?.drives) || !figures?.summary || !figures.summary.late_s || !figures.summary.late_m) {
      throw new Error('The server sent figures this page cannot read.');
    }
    drives.value = list.drives;
    summary.value = figures.summary;
    error.value = '';
  } catch (problem) {
    error.value = messageOf(problem);
  }
}

/** A drive with a sign to compare: the others (every sign left out included) go below, with what keeps them out. */
const comparable = (drive: Drive): boolean => drive.status === 'signs' && drive.counts.included > 0;
const withSigns = computed(() => (drives.value ?? []).filter(comparable));
const others = computed(() => (drives.value ?? []).filter((drive) => !comparable(drive)));
const reading = computed(() => (drives.value ?? []).some((drive) => drive.status === 'reading'));
const lateOf = computed(() => new Map((summary.value?.by_drive ?? []).map((item) => [item.route, item.late_s])));
const toMarkTotal = computed(() => withSigns.value.reduce((sum, drive) => sum + Math.max(0, drive.counts.included
  - drive.counts.zero - drive.counts.cant_tell), 0));
const excludedText = computed(() => Object.entries(summary.value?.excluded ?? {})
  .map(([reason, count]) => count + ' ' + EXCLUDED_LABEL[reason as Excluded]).join(', '));
/** The drives kept out, by state: "3 drives not read yet · 2 drives with no sign seen". */
const STATUS_SHORT: Partial<Record<Drive['status'], string>> = {
  reading: 'being read', unread: 'not read yet', unreadable: 'with logs this server cannot open', no_logs: 'without logs',
  build_too_old: 'driven with a build too old', no_sign_seen: 'with no sign seen', no_video: 'with logs only, no video yet',
  signs: 'with every sign left out',
};
const othersText = computed(() => {
  const counts = new Map<Drive['status'], number>();
  for (const drive of others.value) counts.set(drive.status, (counts.get(drive.status) ?? 0) + 1);
  return [...counts].map(([status, count]) => plural(count, 'drive') + ' ' + (STATUS_SHORT[status] ?? status)).join(' · ');
});
const nameOf = (drive: Drive): string => drive.title ?? 'Places unknown';
const toMarkOf = (drive: Drive): number => Math.max(0, drive.counts.included - drive.counts.zero - drive.counts.cant_tell);
function signsText(drive: Drive): string {
  const counts = drive.counts;
  const parts = [plural(counts.included, 'sign') + ' to compare'];
  const marked = counts.zero + counts.cant_tell;
  parts.push(marked ? marked + ' marked' : 'none marked yet');
  return parts.join(' · ');
}
/** What keeps a drive below: its state, or for a drive whose signs are all left out, why they are. */
const otherText = (drive: Drive): string => drive.message ?? (leftOutText(drive) || 'No sign to compare.');
const otherTone = (drive: Drive): string => drive.status === 'signs' ? 'muted' : statusTone(drive.status);
function leftOutText(drive: Drive): string {
  const parts = Object.entries(drive.counts.excluded).map(([reason, count]) => count + ' ' + EXCLUDED_LABEL[reason as Excluded]);
  return parts.length ? plural(Object.values(drive.counts.excluded).reduce((a, b) => a + (b ?? 0), 0), 'sign') + ' left out (' + parts.join(', ') + ')' : '';
}

// The list reloads every 2 s while a drive's logs are being read (Read the logs on Speed review, or a drive's player).
let timer = 0;
function schedule(): void {
  window.clearTimeout(timer);
  if (reading.value && !tabDrive.value) timer = window.setTimeout(() => void load().then(schedule), 2000);
}
onMounted(() => {
  if (!tabDrive.value) void load().then(schedule);
});
watch(tabDrive, (now) => {
  if (!now) void load().then(schedule);
});
onBeforeUnmount(() => window.clearTimeout(timer));
</script>

<template>
  <SignDistanceDrive v-if="tabDrive" :route="tabDrive" :read-only="readOnly"/>
  <template v-else>
    <SectionHead tab="signdistance"/>

    <section class="card sd-summary" aria-labelledby="sd-summary-title">
      <div class="ds-head">
        <h2 id="sd-summary-title">How late vision's zero comes</h2>
        <span v-if="summary" class="ds-when">{{ plural(summary.zero, 'sign') }} marked</span>
      </div>
      <p class="sd-purpose">The car counts down the distance to the sign ahead. On each sign you mark the frame where it is
        about to leave the picture: your zero. The figure is how far vision's zero comes from yours, from the camera and the car's
        speed only, never the map.</p>
      <p v-if="error" class="empty">{{ error }}</p>
      <p v-else-if="!summary" class="empty">Loading…</p>
      <template v-else>
        <div v-if="summary.late_s.median !== null" class="sd-figure" id="sd-median">
          <strong>{{ lateSentence(summary.late_s.median) }}</strong>
          <span>median of {{ plural(summary.late_s.count, 'sign') }}<template v-if="summary.late_s.count > 1"> · from
            {{ signedSeconds(summary.late_s.min) }} to {{ signedSeconds(summary.late_s.max) }}</template></span>
        </div>
        <p v-else class="sd-figure-empty">No zero marked yet: open a drive below and mark its signs.</p>
        <p v-if="summary.late_m.median !== null" class="sd-second">In metres: {{ signedMetres(summary.late_m.median) }} median of
          {{ plural(summary.late_m.count, 'sign') }}. {{ summary.late_m.note }}</p>
        <p class="sd-facts">
          <span>{{ plural(summary.included, 'sign') }} to compare</span>
          <span v-if="toMarkTotal">{{ toMarkTotal }} to mark</span>
          <span v-if="summary.cant_tell">{{ summary.cant_tell }} can't tell</span>
          <span v-if="excludedText">left out: {{ excludedText }}</span>
          <span v-if="summary.degraded_marks" class="sd-warn">{{ plural(summary.degraded_marks, 'mark') }} out of the medians (video and logs
            not in sync to the frame)</span>
          <span v-if="summary.orphans" class="sd-warn">{{ plural(summary.orphans, 'mark') }} whose sign is no longer found</span>
        </p>
      </template>
    </section>

    <section class="card sd-drives" aria-labelledby="sd-drives-title">
      <div class="ds-head">
        <h2 id="sd-drives-title">Drives with signs</h2>
        <span v-if="drives" class="ds-when">{{ plural(withSigns.length, 'drive') }}</span>
      </div>
      <p v-if="error" class="empty">{{ error }}</p>
      <p v-else-if="drives === null" class="empty">Loading…</p>
      <p v-else-if="!withSigns.length" class="empty">No drive with a sign to compare yet. Vision's distance is in the logs of drives
        made with a build from 31 October 2026 on; their logs are read by Read the logs in Speed review.</p>
      <ul v-else class="rt-list" id="sd-drive-list">
        <li v-for="drive in withSigns" :key="drive.route" class="rt-drive sd-drive">
          <a class="rt-open" :href="driveHref(drive.route)">
            <div class="rt-name">
              <strong>{{ nameOf(drive) }}</strong>
              <span v-if="drive.when">{{ drive.when_certain ? '' : '≈ ' }}{{ drive.when }}</span>
            </div>
            <p class="dr-drive-state" :class="toMarkOf(drive) ? 'dr-todo' : 'dr-ok'">{{ signsText(drive) }}<template
              v-if="!toMarkOf(drive) && drive.counts.included > 0"> · all marked</template></p>
            <p class="rt-shares">
              <span v-if="lateOf.get(drive.route)?.median != null" class="sd-late">median {{ signedSeconds(lateOf.get(drive.route)!.median) }}</span>
              <span v-if="leftOutText(drive)">{{ leftOutText(drive) }}</span>
              <span v-if="drive.counts.orphans" class="sd-warn">{{ plural(drive.counts.orphans, 'mark') }} without its sign</span>
              <span v-if="!drive.unit_known">unit not in the logs: metres</span>
              <span v-else-if="drive.unit === 'ft'">screen in feet</span>
              <span class="rt-frames dr-route">{{ drive.route }}</span>
            </p>
            <span class="data-chev rt-go" aria-hidden="true">›</span>
          </a>
        </li>
      </ul>
    </section>

    <section v-if="others.length" class="card sd-drives" aria-labelledby="sd-others-title">
      <div class="ds-head">
        <h2 id="sd-others-title">Other drives</h2>
        <span class="ds-when">{{ plural(others.length, 'drive') }}</span>
      </div>
      <p class="sd-facts"><span>{{ othersText }}</span></p>
      <details class="sd-others">
        <summary>Show them</summary>
        <ul class="rt-list">
          <li v-for="drive in others" :key="drive.route" class="rt-drive sd-drive" :data-status="drive.status">
            <div class="rt-name">
              <strong>{{ nameOf(drive) }}</strong>
              <span v-if="drive.when">{{ drive.when_certain ? '' : '≈ ' }}{{ drive.when }}</span>
            </div>
            <p class="dr-drive-state" :class="'dr-' + otherTone(drive)">{{ otherText(drive) }}<template v-if="drive.status === 'unread'">
              <a :href="'/speed-review?drive=' + encodeURIComponent(drive.route)">Read them in Speed review ›</a></template><template
              v-else-if="drive.status === 'no_video'"> <a href="/prepare">Drives ›</a></template><template
              v-else-if="drive.status === 'signs'"> <a :href="driveHref(drive.route)">Open ›</a></template></p>
            <ul v-if="drive.status === 'unreadable' && drive.unreadable_files.length" class="sd-files">
              <li v-for="file in drive.unreadable_files" :key="file.segment + file.kind">minute {{ file.segment }} {{ file.kind }}: {{ file.error }}</li>
              <li class="sd-files-note">Their permissions are fixed on the server; this page never changes them.</li>
            </ul>
            <p class="rt-shares">
              <span v-if="drive.build?.commit">build {{ drive.build.commit.slice(0, 9) }}</span>
              <span class="rt-frames dr-route">{{ drive.route }}</span>
            </p>
          </li>
        </ul>
      </details>
    </section>
  </template>
</template>
