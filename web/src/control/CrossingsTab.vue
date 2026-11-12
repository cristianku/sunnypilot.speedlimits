<script setup lang="ts">
// Crossings: the vehicles that cross our path, as a person decides them. A detector on the server finds the vehicles of
// the brain's drives and a rule proposes those that seem to cross (nnslr_tools.road_users); here a person says yes or no
// to each, and only those verdicts teach the brain's users_head vehicles: a parked car, the car ahead and oncoming
// traffic never do. The drives with proposals, those still to review first; a tap opens one ("/crossings?drive=<route>",
// CrossingsDrive.vue). Nothing here writes.
import {computed, onMounted, ref, shallowRef, watch} from 'vue';
import {api, messageOf} from '../common/api';
import {plural} from '../common/format';
import CrossingsDrive from './CrossingsDrive.vue';
import {API, type CrossingDrive, driveHref} from './crossings';
import SectionHead from './SectionHead.vue';
import {tabDrive} from './tabs';

defineProps<{readOnly: boolean | null}>();

const drives = shallowRef<CrossingDrive[] | null>(null);
const error = ref('');
async function load(): Promise<void> {
  try {
    const answer = await api<{drives: CrossingDrive[]}>(API + '/drives');
    if (!Array.isArray(answer?.drives)) throw new Error('The server sent drives this page cannot read.');
    drives.value = answer.drives;
    error.value = '';
  } catch (problem) {
    error.value = messageOf(problem);
  }
}
const open = (drive: CrossingDrive): number => drive.proposed - drive.reviewed;
const sorted = computed(() => [...(drives.value ?? [])].sort((a, b) => Number(open(b) > 0) - Number(open(a) > 0)
  || b.route.localeCompare(a.route)));
const totals = computed(() => (drives.value ?? []).reduce((sum, drive) => ({proposed: sum.proposed + drive.proposed,
  reviewed: sum.reviewed + drive.reviewed, crossing: sum.crossing + drive.crossing}), {proposed: 0, reviewed: 0, crossing: 0}));

onMounted(() => {
  if (!tabDrive.value) void load();
});
watch(tabDrive, (now) => {
  if (!now) void load();
});
</script>

<template>
  <CrossingsDrive v-if="tabDrive" :route="tabDrive" :read-only="readOnly"/>
  <template v-else>
    <SectionHead tab="crossings"/>
    <section class="card sd-drives" aria-labelledby="cx-drives-title">
      <div class="ds-head">
        <h2 id="cx-drives-title">Vehicles that cross your path</h2>
        <span v-if="drives" class="ds-when">{{ totals.reviewed }} of {{ plural(totals.proposed, 'vehicle') }} reviewed</span>
      </div>
      <p class="sd-purpose">A detector on the server finds the vehicles in your drives and proposes those that seem to cross your
        path. You say yes or no: only your yes teaches the brain to slow down for a vehicle; a parked car, the car ahead
        or oncoming traffic never does.</p>
      <p v-if="error" class="empty">{{ error }}</p>
      <p v-else-if="drives === null" class="empty">Loading…</p>
      <p v-else-if="!drives.length" class="empty">No drive read yet: the detector reads the brain's drives on the V100s
        (nnslr road-users), then its proposals show here.</p>
      <ul v-else class="rt-list" id="cx-drive-list">
        <li v-for="drive in sorted" :key="drive.route" class="rt-drive sd-drive">
          <a class="rt-open" :href="driveHref(drive.route)">
            <div class="rt-name">
              <strong class="dr-route">{{ drive.route }}</strong>
            </div>
            <p class="dr-drive-state" :class="open(drive) ? 'dr-todo' : 'dr-ok'">{{ plural(drive.proposed, 'vehicle') }} proposed ·
              {{ open(drive) ? open(drive) + ' to review' : drive.proposed ? 'all reviewed' : 'nothing to review' }}</p>
            <p class="rt-shares">
              <span v-if="drive.crossing">{{ drive.crossing }} crossing</span>
              <span v-if="drive.yaw === null">no yaw in the logs</span>
            </p>
            <span class="data-chev rt-go" aria-hidden="true">›</span>
          </a>
        </li>
      </ul>
    </section>
  </template>
</template>
