<script setup lang="ts">
// The signs on every frame on Models (nnslr_tools.sign_tracks): the cutter and the classifier in use, as Sign review runs
// them, on every decoded frame of each drive's video, for the boxes Speed review draws over the video at 20 frames a
// second (derived/sign_tracks/<route>.json). Find them (helpers/sign_tracks.sh, the panel action "sign_tracks") on every
// drive or on the drives chosen, a dry run on the CPU first if wanted (a few frames of one drive, checked against Sign
// review, apart); its job while it runs; then the drives from GET /api/panel/sign-tracks: read by the networks in use, by
// older ones, stopped half way, or not yet. The buttons follow the server's action state (allowed, and its reason when
// not); a real run also needs a free GPU, one of the two (jobs.ts noFreeCard: the server and the helper check it), a dry
// run does not. Hidden on a server without the action.
import {computed, onMounted, ref, shallowRef, watch} from 'vue';
import {api} from '../common/api';
import {isoWhen, plural, sentence} from '../common/format';
import type {ActionOptions, ActionName, ActionState, Job, PanelStatus, SignTracksDrive, SignTracksPayload} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import JobProgress from './JobProgress.vue';
import {gpusBusyText, JOB_STATE, jobOf, noFreeCard, startedWhen, terminalActivities} from './jobs';
import TerminalActivity from './TerminalActivity.vue';

const props = defineProps<{status: PanelStatus | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions];
  stop: [job: Job]; details: [job: Job]}>();

const state = computed((): ActionState | null => props.status?.actions?.sign_tracks ?? null);
/** A server from before the action: no card. */
const shown = computed(() => state.value !== null);
const job = computed(() => jobOf(props.status, ['sign_tracks']));
const bigJob = computed(() => job.value && (job.value.state === 'running' || job.value.state === 'failed') ? job.value : null);
const activities = computed(() => terminalActivities(props.status, ['train']).filter((activity) => activity.label === 'sign tracks'));
const idle = computed(() => !bigJob.value && !activities.value.length);
/** No GPU free (each card held by a job or another program): only a dry run can start (start() refuses the others). */
const gpuBusy = computed(() => noFreeCard(props.status?.system));

const payload = shallowRef<SignTracksPayload | null>(null);
async function load(): Promise<void> {
  try {
    payload.value = await api<SignTracksPayload>('/api/panel/sign-tracks');
  } catch {
    // Not on this server yet (it comes with a restart), or not readable now: the card shows no drives.
    payload.value = null;
  }
}
onMounted(() => void load());
const working = computed(() => job.value?.state === 'running' || activities.value.length > 0);
watch(working, (now, before) => {
  if (before && !now) void load();
});

const drives = computed((): SignTracksDrive[] => payload.value?.drives ?? []);
const done = computed(() => drives.value.filter((drive) => drive.state === 'done').length);
const toDo = computed(() => drives.value.filter((drive) => drive.state !== 'done' && drive.video).length);
/** The drives chosen; none: every drive. */
const chosen = ref<string[]>([]);
const choosing = ref(false);
const options = (dry: boolean): ActionOptions | undefined => {
  const picked = chosen.value.length ? {routes: chosen.value.join(',')} : {};
  return dry || chosen.value.length ? {...picked, ...(dry ? {dry: 'on'} : {})} : undefined;
};
function start(dry: boolean): void {
  if (state.value) emit('run', 'sign_tracks', state.value, undefined, options(dry));
}

/** The route as it is written (the drive's name is not capitalized like a head's). */
const ROUTE_STYLE = {textTransform: 'none', fontFamily: 'var(--font-data)'} as const;
const STATE_TEXT: Record<SignTracksDrive['state'], string> = {
  none: 'not read yet', done: 'read by the networks in use', old: 'read by older networks or on other minutes',
  partial: 'stopped half way',
};
const stateText = (drive: SignTracksDrive): string => !drive.video ? 'its video must be made again first (Drive view)'
  : drive.state === 'partial' ? STATE_TEXT.partial + ': ' + drive.done_minutes + ' of ' + plural(drive.minutes, 'minute')
    : STATE_TEXT[drive.state];
const driveFacts = (drive: SignTracksDrive): string => drive.made_at
  ? isoWhen(drive.made_at) + ' · ' + plural(drive.frames, 'frame') + ' · ' + plural(drive.boxes, 'box', 'boxes')
  : plural(drive.minutes, 'minute');
const networksText = computed(() => {
  const networks = payload.value?.networks_in_use;
  return networks ? 'With the cutter ' + networks.cutter?.id + ' and the classifier ' + networks.classifier?.id + '.'
    : 'No cutter or classifier in use: choose them in Models first.';
});
</script>

<template>
  <section v-if="shown" class="card import-card labeler-card" id="sign-tracks" aria-labelledby="st-title">
    <h2 id="st-title" class="rl-title">Signs on every frame</h2>
    <p class="rl-what">The cutter and the classifier in use, as Sign review runs them, on every frame of each drive's video:
      the boxes Speed review draws over the video at 20 frames a second. A drive read by the same networks on the same
      minutes is left as it is.</p>
    <JobProgress v-if="bigJob" :job="bigJob" @details="emit('details', $event)" @stop="emit('stop', $event)"/>
    <TerminalActivity v-for="activity in activities" :key="activity.pid + activity.label" :activity="activity" :preparation="null"/>
    <div v-if="idle" class="import-row">
      <span class="import-icon" :class="{fresh: !done}" aria-hidden="true"><Icon name="routes"/></span>
      <div class="import-what">
        <strong>{{ payload ? done + ' of ' + plural(drives.length, 'drive') + ' read' : 'Drives not known yet' }}</strong>
        <span>{{ payload ? networksText : '' }} On the GPU, about 50 frames a second{{ toDo ? '; ' + plural(toDo, 'drive')
          + ' to read' : '' }}.</span>
      </div>
      <div class="rl-buttons">
        <button type="button" class="btn" id="sign-tracks-dry" :disabled="!state?.allowed" @click="start(true)">Dry run</button>
        <button type="button" class="btn primary" id="sign-tracks-run" :disabled="!state?.allowed || gpuBusy"
                @click="start(false)">{{ chosen.length ? 'Run on ' + plural(chosen.length, 'drive') : 'Run on every drive' }}</button>
      </div>
    </div>
    <p v-if="idle && state && !state.allowed" class="import-why">{{ sentence(state.reason) }}</p>
    <p v-else-if="idle && gpuBusy" class="import-why warn" id="sign-tracks-gpu">{{ gpusBusyText(status?.system) }}. Only a dry
      run on the CPU can start now.</p>
    <p v-if="job && !bigJob" class="import-last">Last run {{ startedWhen(job) }} ·
      {{ job.state === 'interrupted' ? 'stopped' : JOB_STATE[job.state][0] }} ·
      <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>

    <label v-if="drives.length && idle" class="rl-marks">
      <input type="checkbox" v-model="choosing" @change="choosing || (chosen = [])"> Only the drives I choose
      (the dry run takes the first)</label>
    <ul v-if="drives.length && (choosing || !idle)" class="rl-heads" id="sign-tracks-drives">
      <li v-for="drive in drives" :key="drive.route" class="rl-head" :class="{writes: drive.state === 'done'}"
          :data-route="drive.route">
        <label class="rl-head-name" :style="ROUTE_STYLE"><input v-if="idle" type="checkbox" :value="drive.route" v-model="chosen"
          :disabled="!drive.video"> {{ drive.route }}</label>
        <span class="rl-head-num">{{ driveFacts(drive) }}</span>
        <span class="rl-head-writes">{{ stateText(drive) }}</span>
      </li>
    </ul>
  </section>
</template>
