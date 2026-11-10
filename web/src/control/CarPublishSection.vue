<script setup lang="ts">
// Publish › Publish: the networks in use put into the comma's software in one go (scripts/send_to_car.sh): the SignNet
// in use, the classifier in use, the applicability network in use and the brain in use are prepared as ONNX and checked, published in
// speed-limits-models, the sunnypilot speed-limits branch moves to them and psa-torque-sunny-testing is built again;
// then the comma's Check for updates finds them. Each network in use is shown next to the one the comma's software has
// (models/on-car.json), and the job's steps while it runs.
import {computed} from 'vue';
import {isoWhen} from '../common/format';
import type {ActionName, ActionOptions, ActionState, Job, ModelRole, ModelVersion, ModelsPayload, PanelStatus} from '../common/panel-types';
import {askConfirm} from '../composables/confirm';
import JobProgress from './JobProgress.vue';
import {JOB_STATE, jobOf, startedWhen} from './jobs';
import {publishQuestion, rebuildQuestion} from './models';

const props = defineProps<{status: PanelStatus | null; models: ModelsPayload | null}>();
const emit = defineEmits<{run: [name: ActionName, state: ActionState, start?: undefined, options?: ActionOptions];
  stop: [job: Job]; details: [job: Job]}>();

/** The networks the comma runs, in the order they work. */
const CAR_ROLES: readonly ModelRole[] = ['cutter', 'classifier', 'decider', 'brain'];
const versionsOf = (role: ModelRole): ModelVersion[] => props.models?.roles.find((item) => item.role === role)?.versions ?? [];
const inUseOf = (role: ModelRole): ModelVersion | null => versionsOf(role).find((version) => version.in_use) ?? null;
const whenOf = (version: ModelVersion): string => version.trained_at ? isoWhen(version.trained_at) : version.run;

/** A line per network: the version in use, and whether the comma's software has it (with the time of the one there:
 * two versions often have the same name). */
const rows = computed(() => CAR_ROLES.map((role) => {
  const label = props.models?.roles.find((item) => item.role === role)?.label ?? role;
  const used = inUseOf(role);
  const car = versionsOf(role).find((version) => version.on_car) ?? null;
  const same = used !== null && car !== null && used.id === car.id;
  const usedText = used ? used.label + ', ' + whenOf(used)
    : role === 'brain' ? 'None in use: the comma runs the rules of the signs'
      : role === 'decider' ? 'None in use: the comma takes every sign, of another road too' : 'None in use';
  const older = car !== null && used?.trained_at && car.trained_at && Date.parse(car.trained_at) < Date.parse(used.trained_at);
  const carText = same ? 'In the branch: this one' : car ? 'In the branch: ' + (older ? 'an older one, ' : 'another one, ') + whenOf(car)
    : used ? 'Not in the branch yet' : 'None in the branch either';
  const differs = !same && (used !== null || car !== null);
  return {role, label, usedText, carText, tone: same ? 'same' : differs ? 'other' : 'none', differs};
}));

/** The SignNet in use, when the comma can run it (finished, at the car's band and grid). */
const signnet = computed((): ModelVersion | null => {
  const used = inUseOf('cutter');
  return used?.role === 'cutter' && used.kind === 'signnet' && used.fits_the_car !== false && used.finished ? used : null;
});
const pending = computed(() => Boolean(props.models?.car_build_pending));
const differing = computed(() => rows.value.filter((row) => row.differs).length);

/** The last publication to the car: running, or failed with something still owed, in full; one done is a line. A failure
 * with nothing owed was finished afterwards, from a terminal: this page only knows the jobs of its own button. */
const job = computed(() => jobOf(props.status, ['send_to_car']));
const running = computed(() => job.value?.state === 'running');
const owed = computed(() => pending.value || differing.value > 0);
const bigJob = computed(() => job.value && (running.value || (job.value.state === 'failed' && owed.value)) ? job.value : null);

/** The state in one line: the comma's software has the networks in use, or not yet and why. */
const headline = computed((): {tone: 'ok' | 'warn' | ''; text: string} => {
  const models = props.models;
  if (models === null) return {tone: '', text: ''};
  if (models.on_car_problem) return {tone: 'warn', text: 'What is on the car cannot be read: ' + models.on_car_problem + '.'};
  if (signnet.value === null) return {tone: 'warn', text: 'The cutter in use is not a finished SignNet the comma can run: choose one in Versions.'};
  if (pending.value) return {tone: 'warn', text: 'The last publication did not finish building psa-torque-sunny-testing: the comma finds nothing new yet.'};
  if (differing.value === 0) return {tone: 'ok', text: '✓ The branch has the networks in use.'};
  return {tone: 'warn', text: differing.value === rows.value.length ? 'The branch has none of the networks in use.'
    : 'The branch does not have ' + (differing.value === 1 ? 'one' : differing.value) + ' of the networks in use yet.'};
});
/** Where the comma's networks come from: "speed-limits-models f2ea8f5, 2026-10-29". */
const published = computed((): string => {
  const car = props.models?.on_car;
  if (!car) return '';
  return [(car.repository?.split('/').pop() ?? 'speed-limits-models') + (car.commit ? ' ' + car.commit.slice(0, 7) : ''),
    car.version ?? ''].filter(Boolean).join(', ');
});

const state = computed(() => props.status?.actions.send_to_car ?? null);
/** The button is always there with a SignNet the comma can run: Publish, Finish publishing, or Build again when nothing is owed. */
const publishable = computed(() => signnet.value !== null && !running.value);
const buttonText = computed(() => differing.value > 0 ? 'Publish' : pending.value ? 'Finish publishing' : 'Build again');
/** Why Publish is off, under it. */
const blocked = computed((): string => {
  if (props.status !== null && !props.status.actions_enabled) return 'Read-only: the buttons are turned off on this server.';
  if (state.value === null) return 'Publishing to the car is not available on this server.';
  return state.value.allowed ? '' : state.value.reason.charAt(0).toUpperCase() + state.value.reason.slice(1) + '.';
});

/** Asks, naming the SignNet, the classifier, the applicability network and the brain in use, then starts the job. */
async function publish(): Promise<void> {
  const version = signnet.value;
  const current = state.value;
  if (version === null || current === null || blocked.value) return;
  if (!await askConfirm(owed.value ? publishQuestion(version, inUseOf('classifier'), inUseOf('brain'), inUseOf('decider'))
    : rebuildQuestion())) return;
  emit('run', 'send_to_car', current, undefined, owed.value ? {signnet: version.run} : {signnet: version.run, rebuild: 'on'});
}
</script>

<template>
  <section class="card publish car-publish" id="car-publish" aria-labelledby="car-publish-title">
    <h2 id="car-publish-title">Publish</h2>
    <p class="note">The networks in use into the comma’s software, in one go: the package is prepared and checked, published
      in speed-limits-models, and psa-torque-sunny-testing is built with it. A few minutes; then, on the comma: Settings,
      Software, Check for updates, and install.</p>
    <p v-if="headline.text" class="car-state" :class="headline.tone" id="car-state">{{ headline.text }}</p>
    <ul v-if="models" class="car-rows" id="car-rows">
      <li v-for="row in rows" :key="row.role" class="car-row" :class="row.tone" :data-role="row.role">
        <span class="car-role">{{ row.label }}</span>
        <span class="car-what"><b>{{ row.usedText }}</b><small>{{ row.carText }}</small></span>
      </li>
    </ul>
    <div class="train-slot">
      <JobProgress v-if="bigJob" :job="bigJob" :heading="bigJob.run ? 'SignNet ' + bigJob.run : undefined"
                   @details="emit('details', $event)" @stop="emit('stop', $event)"/>
      <div v-if="publishable" class="car-go">
        <button class="btn primary" type="button" id="car-publish-button" :disabled="blocked !== ''"
                :aria-describedby="blocked ? 'car-publish-reason' : undefined" @click="publish">{{ buttonText }}</button>
        <p v-if="blocked" class="act-reason" id="car-publish-reason">{{ blocked }}</p>
      </div>
      <p v-if="job && !bigJob" class="last-line" :class="{'pub-done': job.state === 'ok'}" id="car-last"><template
        v-if="job.state === 'ok'">✓ Branch psa-torque-sunny-testing ready, {{ job.ended_at ? isoWhen(job.ended_at) : startedWhen(job) }}: on
        the comma, Check for updates and install</template><template
        v-else>Last: {{ job.label }}, {{ startedWhen(job) }} · {{ job.state === 'interrupted' ? 'stopped' : job.state === 'failed' ? 'failed, finished later' : JOB_STATE[job.state][0] }}</template> ·
        <button type="button" class="link-button" @click="emit('details', job)">Details</button></p>
    </div>
    <p v-if="models?.comma_trial" class="note warn-text" id="car-trial">A trial is on the comma now (SignNet
      {{ models.comma_trial.signnet.run }}): its next software update replaces it.</p>
    <p v-if="published" class="note" id="car-published">The branch has the networks of {{ published }}.</p>
  </section>
</template>
