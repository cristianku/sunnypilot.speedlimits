<script setup lang="ts">
// Models › Drive: what the comma's screen would show along a recorded drive with a set of networks, before that set goes
// on the car. A drive is replayed through the car's own vision loop (nnslr_tools.simulation), and opened
// ("/drive?drive=<route>", DriveSimulation.vue) its video plays with the comma's two squares on it: the speed to hold
// and the sign read. A drive where all of it is right is certified; a new set is then replayed on the certified drives
// and compared with them frame by frame: the regression test. Here, the list: the set chosen, one choice for the whole
// tab (On the car: the release the comma downloads; In use: the versions in use on Models), with its networks; the
// certified drives against that set, and Replay the certified drives (the panel action "simulate",
// helpers/simulate_drive.sh, on the GPU) with its job while it runs, a replay started from a terminal too; then every
// drive, the newest first, with what it has for the set chosen. Nothing here changes the car, the labels or the networks
// in use.
import {computed, onMounted, ref, shallowRef, watch} from 'vue';
import {api, messageOf} from '../common/api';
import {plural, sentence} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Job, PanelStatus} from '../common/panel-types';
import DriveSimulation from './DriveSimulation.vue';
import JobProgress from './JobProgress.vue';
import {jobOf, terminalActivities} from './jobs';
import SectionHead from './SectionHead.vue';
import Segmented from './Segmented.vue';
import {changeLines, changeName, chosenSet, identical, networksOf, regressionShort, regressionText, SET_KEYS, SET_LABEL, type SimulationList,
  type SimulationRow} from './simulation';
import {tabDrive} from './tabs';
import TerminalActivity from './TerminalActivity.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions]; stop: [job: Job];
  details: [job: Job]}>();
const SETS = SET_KEYS.map((key) => ({key, label: SET_LABEL[key]}));

const payload = shallowRef<SimulationList | null>(null);
const error = ref('');
async function load(): Promise<void> {
  try {
    const answer = await api<SimulationList>('/api/simulation/drives');
    // A cut answer (the page left while it came) reads as {message} in api(): never a list without its drives.
    if (!answer || !Array.isArray(answer.drives)) throw new Error('The server sent a list this page cannot read.');
    payload.value = answer;
    error.value = '';
  } catch (problem) {
    error.value = messageOf(problem);
  }
}

const state = computed((): ActionState | null => props.status?.actions?.simulate ?? null);
const readOnly = computed(() => props.status?.actions_enabled === false);
const job = computed(() => jobOf(props.status, ['simulate']));
const running = computed(() => job.value?.state === 'running' ? job.value : null);
/** A replay started from a terminal (helpers/simulate_drive.sh, nnslr simulate): shown as the site's own, without Stop. */
const terminal = computed(() => terminalActivities(props.status, ['train']).filter((activity) => activity.label === 'drive replay'));
const working = computed(() => running.value !== null || terminal.value.length > 0);
// A replay that ends: what it computed.
watch(working, (now, before) => { if (before && !now && tabDrive.value === null) void load(); });
// Back on the list: the drive may have been replayed, certified or removed.
watch(tabDrive, (route) => { if (route === null) void load(); });
onMounted(() => { if (tabDrive.value === null) void load(); });

// -- the set chosen ---------------------------------------------------------------------------------------------------------
const setState = computed(() => payload.value?.sets[chosenSet.value]);
const networks = computed(() => networksOf(setState.value));

// -- the drives -------------------------------------------------------------------------------------------------------------
// By date and time ("when" ends in the time): the number of a drive is not its date, the comma's count started again.
const startOf = (drive: SimulationRow): string => (drive.day ?? '') + (drive.when ?? '').slice(-5);
const drives = computed(() => [...(payload.value?.drives ?? [])].sort((a, b) => startOf(b).localeCompare(startOf(a))));
const hrefOf = (drive: SimulationRow): string => '/drive?drive=' + encodeURIComponent(drive.route);
/** What a drive has for the set chosen: its replay next to the certified one, or only that it was replayed. */
function hasOf(drive: SimulationRow): {text: string; tone: string; title: string; stale: string; staleTitle: string} | null {
  const result = drive.sets[chosenSet.value];
  if (!result) return null;
  const found = drive.certified ? drive.regression[chosenSet.value] : undefined;
  // Out of date: an orange badge naming what changed since the replay, the versions in its tooltip.
  const names = [...new Set((result.changed ?? []).map(changeName))];
  const stale = result.current ? '' : 'Out of date' + (names.length ? ': ' + names.join(', ') : '');
  const staleTitle = result.current ? '' : ['Replayed before these changes; Compute, in the drive, replays it.',
    ...changeLines(result.changed)].join('\n');
  if (!found) return {text: 'Replayed', tone: 'dr-muted', title: '', stale, staleTitle};
  return {text: regressionShort(found), tone: identical(found) ? 'dr-ok' : 'dr-bad', title: regressionText(found), stale, staleTitle};
}
const rows = computed(() => drives.value.map((drive) => ({drive, has: hasOf(drive)})));

// -- the regression test ----------------------------------------------------------------------------------------------------
const certified = computed(() => drives.value.filter((drive) => drive.certified));
/** The certified drives against the set chosen as it is today: a replay with other networks, car code or settings than today's is not its. */
const test = computed(() => {
  let same = 0;
  let differ = 0;
  let open = 0;
  for (const drive of certified.value) {
    const found = drive.sets[chosenSet.value]?.current ? drive.regression[chosenSet.value] : undefined;
    if (!found) open += 1;
    else if (identical(found)) same += 1;
    else differ += 1;
  }
  return {same, differ, open};
});
const testText = computed(() => 'With ' + SET_LABEL[chosenSet.value] + ': ' + [test.value.same ? test.value.same + ' identical' : '',
  test.value.differ ? test.value.differ + (test.value.differ === 1 ? ' differs' : ' differ') : '',
  test.value.open ? test.value.open + ' not replayed yet' : ''].filter(Boolean).join(' · ') + '.');
/** Why the replay cannot start now, in the server's words; '' when it can. */
const blocked = computed((): string => {
  if (props.status && !state.value) return 'This server cannot replay a drive yet.';
  if (state.value && !state.value.allowed) return sentence(state.value.reason);
  // The release of the car's networks is the clone's: there is nothing a replay could make ready.
  if (chosenSet.value === 'car' && setState.value && !setState.value.ready) return sentence(setState.value.reason);
  return test.value.open ? '' : 'Every certified drive is replayed with these networks.';
});
function replay(): void {
  if (state.value) emit('run', 'simulate', state.value, undefined, {set: chosenSet.value, certified: '1'});
}
</script>

<template>
  <DriveSimulation v-if="tabDrive" :key="tabDrive" :route="tabDrive" :status="status"
                   @run="(name, action, start, options) => emit('run', name, action, start, options)" @stop="emit('stop', $event)"
                   @details="emit('details', $event)"/>
  <template v-else>
    <SectionHead tab="drive"/>
    <section class="card dv-networks" aria-labelledby="dv-networks-title">
      <div class="ds-head">
        <h2 id="dv-networks-title">Networks</h2>
        <Segmented v-model="chosenSet" :options="SETS" label="The networks a drive is replayed with"/>
      </div>
      <template v-if="payload">
        <dl v-if="networks.length" class="dv-nets" id="dv-nets">
          <template v-for="item in networks" :key="item.label"><dt>{{ item.label }}</dt><dd>{{ item.id }}</dd></template>
        </dl>
        <p v-if="setState && !setState.ready && setState.reason" class="import-why" id="dv-set-reason">{{ sentence(setState.reason) }}</p>
        <p v-else-if="setState?.name" class="import-why">{{ setState.name }}</p>
      </template>
      <!-- Without a certified drive the regression test has no card: a replay started in a drive shows here. -->
      <div v-if="working && !certified.length" class="dv-job">
        <JobProgress v-if="running" :job="running" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
        <TerminalActivity v-for="activity in terminal" :key="activity.pid + activity.label" :activity="activity" :preparation="null"/>
      </div>
    </section>

    <section v-if="certified.length" class="card dv-test" aria-labelledby="dv-test-title">
      <div class="ds-head">
        <h2 id="dv-test-title">Regression test</h2>
        <span class="ds-when">{{ plural(certified.length, 'certified drive') }}</span>
      </div>
      <div v-if="working" class="dv-job">
        <JobProgress v-if="running" :job="running" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
        <TerminalActivity v-for="activity in terminal" :key="activity.pid + activity.label" :activity="activity" :preparation="null"/>
      </div>
      <div v-else class="dv-run">
        <p class="import-why" id="dv-test-text">{{ testText }}{{ blocked && !readOnly ? ' ' + blocked : '' }}</p>
        <button v-if="!readOnly" type="button" class="btn primary" id="dv-replay" :disabled="!state?.allowed || blocked !== ''"
                title="Replays every certified drive not replayed with these networks yet, on the GPU, and compares each with what was certified"
                @click="replay">Replay the certified drives</button>
      </div>
      <p v-if="!working && job?.state === 'failed'" class="import-last">The last replay failed ·
        <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>
    </section>

    <section class="card sr-drives" aria-labelledby="dv-drives-title">
      <div class="ds-head">
        <h2 id="dv-drives-title">Drives</h2>
        <span v-if="payload" class="ds-when">{{ plural(drives.length, 'drive') }}</span>
      </div>
      <p class="import-why">Open a drive and press play: on its video the comma’s screen, as the car would draw it with the networks
        chosen. Certify a drive where all of it is right: the next networks are replayed on the certified drives and compared
        with them.</p>
      <p v-if="error" class="empty" id="dv-error">{{ error }}</p>
      <p v-else-if="payload === null" class="empty">Loading…</p>
      <p v-else-if="!rows.length" class="empty">No drive on the server yet.</p>
      <ul v-else class="rt-list">
        <li v-for="{drive, has} in rows" :key="drive.route" class="rt-drive dr-drive">
          <a class="rt-open" :href="hrefOf(drive)">
            <div class="rt-name">
              <strong>{{ drive.title ?? 'Places unknown' }} <b v-if="drive.dashcam" class="dashcam-badge">DASHCAM MODE</b></strong>
              <span><template v-if="drive.when">{{ drive.when_certain === false ? '≈ ' : '' }}{{ drive.when }}</template>{{
                drive.minutes !== null ? ' · ' + drive.minutes + ' min' : '' }}</span>
            </div>
            <p class="dv-has">
              <span class="dr-route">{{ drive.route }}</span>
              <span v-if="drive.certified || has" class="dv-state">
                <b v-if="drive.certified" class="dv-badge">Certified</b>
                <span v-if="has" :class="has.tone" :title="has.title || undefined">{{ has.text }}</span>
                <b v-if="has?.stale" class="dv-badge dv-stale-badge" :title="has.staleTitle">{{ has.stale }}</b>
              </span>
            </p>
            <span class="data-chev rt-go" aria-hidden="true">›</span>
          </a>
        </li>
      </ul>
    </section>
  </template>
</template>
