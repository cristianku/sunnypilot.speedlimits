<script setup lang="ts">
// Crossings, a drive opened ("/crossings?drive=<route>&event=<id>"): one proposed vehicle at a time. The drive's video
// (the drive player's MP4, nnslr_tools.drive_video) from 1 s before the event to half a second after it, looping, with
// the tracked vehicle's box and its path drawn over it (the boxes of the readings, 4 a second, the box between two of
// them moved along). The person says whether it crosses our path (Y) or not (N): the verdict is saved at once, as in
// Sign review, and the next vehicle still to review opens. Space plays the event again, ← → the vehicle before or after.
// Only a person signed in gives a verdict: on a read-only server, or signed out, the buttons are not there.
import {computed, onBeforeUnmount, onMounted, ref, shallowRef, useTemplateRef, watch} from 'vue';
import {api, ApiError, messageOf, post} from '../common/api';
import {plural} from '../common/format';
import {toast} from '../composables/toast';
import {acrossText, API, CLASS_WORDS, type CrossingDrive, type CrossingEvent, type CrossingView, placeText, toReview,
  type Verdict, type VerdictName, videoTime} from './crossings';
import type {VideoState} from './drive-review';
import {goTo, showEvent, tabEvent} from './tabs';

/** readOnly is null until the server's status is in (no buttons and no note yet). */
const props = defineProps<{route: string; readOnly: boolean | null}>();

// -- the drive and its events -----------------------------------------------------------------------------------------
const drive = shallowRef<CrossingView | null>(null);
const error = ref('');
const events = computed(() => drive.value?.events ?? []);
const eventId = ref<string | null>(null);
const event = computed((): CrossingEvent | null => events.value.find((item) => item.id === eventId.value) ?? null);
const position = computed(() => events.value.findIndex((item) => item.id === eventId.value));

async function loadDrive(keep: string | null = null): Promise<void> {
  const route = props.route;
  try {
    const answer = await api<CrossingView>(API + '/drives/' + encodeURIComponent(route));
    if (route !== props.route) return;
    if (!answer || !Array.isArray(answer.events) || typeof answer.revision !== 'number') {
      throw new Error('The server sent a drive this page cannot read.');
    }
    drive.value = answer;
    error.value = '';
    const wanted = keep ?? tabEvent.value;
    const found = answer.events.find((item) => item.id === wanted) ?? answer.events.find(toReview) ?? answer.events[0];
    if (found) openEvent(found.id);
  } catch (problem) {
    if (route === props.route) error.value = messageOf(problem);
  }
}

function openEvent(id: string): void {
  eventId.value = id;
  showEvent(id);
}
watch(tabEvent, (id) => {
  if (id && id !== eventId.value && events.value.some((item) => item.id === id)) eventId.value = id;
});

// -- the video ----------------------------------------------------------------------------------------------------------
const video = shallowRef<VideoState | null>(null);
const videoError = ref('');
let videoTimer = 0;
async function loadVideo(): Promise<void> {
  window.clearTimeout(videoTimer);
  const route = props.route;
  try {
    const answer = await api<VideoState>('/api/panel/drive-video/' + encodeURIComponent(route) + '?start=1');
    if (route !== props.route) return;
    video.value = answer;
    videoError.value = '';
    if (answer.state === 'waiting' || answer.state === 'making') videoTimer = window.setTimeout(() => void loadVideo(), 2000);
  } catch (problem) {
    if (route === props.route) videoError.value = messageOf(problem);
  }
}
const ready = computed(() => video.value?.state === 'ready' && Boolean(video.value.url) && Boolean(video.value.index));
const player = useTemplateRef<HTMLVideoElement>('player');
const failed = ref(false);

/** The event's boxes at their time in the video, in order. */
const track = computed(() => {
  const index = video.value?.index;
  return (event.value?.boxes ?? []).flatMap((item) => {
    const at = videoTime(index, item);
    return at === null ? [] : [{at, box: item.box}];
  });
});
/** The loop: 1 s before the first box to half a second after the last. */
const span = computed(() => {
  const list = track.value;
  if (!list.length) return null;
  return {from: Math.max(0, list[0]!.at - 1), to: list[list.length - 1]!.at + 0.5};
});
const now = ref(0);
const slow = ref(false);

function restart(): void {
  const element = player.value;
  const window_ = span.value;
  if (!element || !window_) return;
  element.currentTime = window_.from;
  element.playbackRate = slow.value ? 0.5 : 1;
  void element.play().catch(() => undefined);
}
watch([span, player], () => restart());
watch(slow, () => {
  if (player.value) player.value.playbackRate = slow.value ? 0.5 : 1;
});

let frame = 0;
function tick(): void {
  const element = player.value;
  const window_ = span.value;
  if (element && window_) {
    if (!element.seeking && (element.currentTime >= window_.to || element.currentTime < window_.from - 0.3)) {
      element.currentTime = window_.from;
      if (element.paused) void element.play().catch(() => undefined);
    }
    now.value = element.currentTime;
  }
  frame = window.requestAnimationFrame(tick);
}

/** The box at the time shown: between two readings it moves from one to the other; none before the first or after
 * the last (a reading's quarter second either side). */
const shownBox = computed((): [number, number, number, number] | null => {
  const list = track.value;
  const t = now.value;
  if (!list.length || t < list[0]!.at - 0.125 || t > list[list.length - 1]!.at + 0.125) return null;
  const after = list.findIndex((item) => item.at >= t);
  if (after <= 0) return (after === 0 ? list[0]! : list[list.length - 1]!).box;
  const a = list[after - 1]!;
  const b = list[after]!;
  const share = b.at > a.at ? (t - a.at) / (b.at - a.at) : 0;
  return a.box.map((value, n) => value + (b.box[n]! - value) * share) as [number, number, number, number];
});
/** The path of the vehicle: the foot of its box at each reading. */
const path = computed(() => track.value.map(({box}) => ((box[0] + box[2]) / 2).toFixed(1) + ',' + box[3].toFixed(1)).join(' '));
const frameSize = computed(() => drive.value?.frame_size ?? [1344, 760]);
const progress = computed(() => {
  const window_ = span.value;
  return window_ ? Math.max(0, Math.min(1, (now.value - window_.from) / (window_.to - window_.from))) : 0;
});

// -- verdicts -----------------------------------------------------------------------------------------------------------
const saving = ref(false);
const refused = ref('');
const writable = computed(() => props.readOnly === false && !refused.value);
const readOnlyNote = computed(() => refused.value || (props.readOnly === true ? 'This server is read-only: the verdicts are shown, not changed.' : ''));

function applyVerdict(id: string, verdict: Verdict | null, revision: number): void {
  const current = drive.value;
  if (!current) return;
  drive.value = {...current, revision, events: current.events.map((item) => item.id === id ? {...item, verdict} : item)};
}

async function send(verdict: VerdictName | 'clear'): Promise<void> {
  const item = event.value;
  const current = drive.value;
  if (!item || !current || saving.value || !writable.value) return;
  saving.value = true;
  try {
    const answer = await post<{revision: number; event: string; verdict: Verdict | null}>(
      API + '/drives/' + encodeURIComponent(props.route) + '/verdicts',
      {event: item.id, verdict, base_revision: current.revision});
    applyVerdict(answer.event, answer.verdict, answer.revision);
    if (verdict === 'clear') toast('Vehicle ' + (position.value + 1) + ' back to review.');
    else nextToReview();
  } catch (problem) {
    if (problem instanceof ApiError && problem.status === 409) {
      toast(messageOf(problem), true);
      void loadDrive(item.id);
    } else {
      if (problem instanceof ApiError && problem.status === 403) refused.value = messageOf(problem);
      toast(messageOf(problem), true);
    }
  } finally {
    saving.value = false;
  }
}

const remaining = computed(() => events.value.filter(toReview).length);
const crossingCount = computed(() => events.value.filter((item) => item.verdict?.verdict === 'crossing').length);
/** The next vehicle still to review after the one shown (from the start again); none: the drive is done. */
function nextToReview(): void {
  const list = events.value;
  const here = position.value;
  const after = [...list.slice(here + 1), ...list.slice(0, Math.max(0, here))];
  const next = after.find(toReview);
  if (next) openEvent(next.id);
  else toast(remaining.value ? 'Saved.' : 'Every vehicle of this drive is reviewed.');
}
function step(by: -1 | 1): void {
  const to = events.value[position.value + by];
  if (to) openEvent(to.id);
}

const nextDriveBusy = ref(false);
async function nextDrive(): Promise<void> {
  nextDriveBusy.value = true;
  try {
    const answer = await api<{drives: CrossingDrive[]}>(API + '/drives');
    const list = answer.drives.filter((item) => item.route !== props.route && item.reviewed < item.proposed);
    const after = list.find((item) => item.route < props.route) ?? list[0];
    if (after) goTo('crossings', '', after.route, null);
    else toast('Every vehicle of every drive is reviewed.');
  } catch (problem) {
    toast(messageOf(problem), true);
  } finally {
    nextDriveBusy.value = false;
  }
}
function back(): void {
  if ((history.state as {fromList?: boolean} | null)?.fromList) history.back();
  else goTo('crossings');
}

// -- the keys -------------------------------------------------------------------------------------------------------------
function onKey(keyEvent: KeyboardEvent): void {
  if (keyEvent.altKey || keyEvent.ctrlKey || keyEvent.metaKey || !event.value) return;
  const target = keyEvent.target instanceof HTMLElement ? keyEvent.target : null;
  const tag = target?.tagName ?? '';
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  const key = keyEvent.key;
  const lower = key.toLowerCase();
  if (key === 'ArrowLeft' || key === 'ArrowRight') step(key === 'ArrowLeft' ? -1 : 1);
  else if (key === ' ') restart();
  else if (lower === 'y') void send('crossing');
  else if (lower === 'n') void send('not_crossing');
  else if (lower === 's') slow.value = !slow.value;
  else return;
  keyEvent.preventDefault();
}
onMounted(() => {
  document.addEventListener('keydown', onKey);
  frame = window.requestAnimationFrame(tick);
  void loadDrive();
  void loadVideo();
});
onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKey);
  window.cancelAnimationFrame(frame);
  window.clearTimeout(videoTimer);
});
watch(() => props.route, () => {
  drive.value = null;
  video.value = null;
  eventId.value = null;
  void loadDrive();
  void loadVideo();
});

// -- words ----------------------------------------------------------------------------------------------------------------
const heading = computed(() => position.value >= 0 ? 'Vehicle ' + (position.value + 1) + ' of ' + events.value.length : '');
const verdictWhen = computed(() => {
  const said = event.value?.verdict;
  if (!said) return '';
  return [said.reviewer ? 'by ' + said.reviewer : null, said.at ? said.at.replace('T', ' ').slice(0, 16) : null].filter(Boolean).join(', ');
});
const videoNote = computed(() => {
  const state = video.value;
  if (videoError.value) return videoError.value;
  if (!state) return 'Loading the video…';
  if (state.state === 'waiting' || state.state === 'making') return 'Making the drive\'s video… ' + Math.round((state.progress ?? 0) * 100) + '%';
  if (state.state === 'failed' || state.state === 'missing') return 'The video of this drive cannot be shown' + (state.reason ? ': ' + state.reason : '.');
  if (failed.value) return 'The video of this drive cannot be played.';
  if (event.value && !track.value.length) return 'The video has not the minute of this vehicle.';
  return '';
});
</script>

<template>
  <section class="card rt-view sd-view cx-view">
    <div class="rt-view-head">
      <button type="button" class="link-button rt-back" @click="back"><span aria-hidden="true">‹</span> Crossings</button>
      <div class="rt-view-name">
        <h2>{{ route }}</h2>
        <span v-if="drive">{{ plural(events.length, 'vehicle') }} proposed · {{ remaining ? remaining + ' to review' : 'all reviewed' }}<template
          v-if="crossingCount"> · {{ crossingCount }} crossing</template></span>
      </div>
    </div>
    <p v-if="error" class="empty">{{ error }}</p>
    <p v-else-if="!drive" class="empty">Loading…</p>
    <p v-else-if="!events.length" class="empty">The rule proposed no crossing vehicle on this drive.</p>
    <template v-else>
      <div v-if="!drive.current" class="dr-banner dr-todo" id="cx-stale">
        <p>These proposals were made from older readings of the detector: nnslr road-users makes them again.</p>
      </div>
      <nav class="sd-signs" aria-label="Vehicles of this drive">
        <button v-for="(item, n) in events" :key="item.id" type="button" class="sd-sign-chip"
                :class="item.verdict?.verdict === 'crossing' ? 'zero' : item.verdict ? 'cant' : 'todo'"
                :aria-current="item.id === eventId ? 'true' : undefined"
                :title="'Vehicle ' + (n + 1) + ': ' + (item.verdict?.verdict === 'crossing' ? 'crosses' : item.verdict ? 'does not cross' : 'to review')"
                @click="openEvent(item.id)">{{ n + 1 }}</button>
      </nav>

      <template v-if="event">
        <div class="sd-sign-head">
          <h3 id="cx-title" :data-event="event.id">{{ heading }}</h3>
          <span>{{ CLASS_WORDS[event.class ?? ''] ?? event.kind }} · {{ acrossText(event.score) }} · {{ placeText(event.from) }}</span>
        </div>

        <div class="sd-stage cx-stage" :class="{empty: !ready || Boolean(videoNote)}"
             :style="{'--sd-aspect': frameSize[0] + ' / ' + frameSize[1]}">
          <video v-if="ready" ref="player" :src="video!.url" muted playsinline preload="auto"
                 @loadedmetadata="restart" @error="failed = true" @playing="failed = false"></video>
          <svg v-if="ready && track.length" class="cx-over" :viewBox="'0 0 ' + frameSize[0] + ' ' + frameSize[1]"
               preserveAspectRatio="none" aria-hidden="true">
            <polyline v-if="track.length > 1" class="cx-path" :points="path"/>
            <rect v-if="shownBox" class="cx-box" :x="shownBox[0]" :y="shownBox[1]" :width="shownBox[2] - shownBox[0]"
                  :height="shownBox[3] - shownBox[1]" rx="4"/>
          </svg>
          <span v-if="videoNote" class="dr-stage-note">{{ videoNote }}</span>
          <span class="cx-loop" :style="{width: (progress * 100).toFixed(1) + '%'}" aria-hidden="true"></span>
        </div>
        <div class="sd-stage-bar">
          <button type="button" class="btn small" title="Play it again (Space)" @click="restart">Again <kbd class="sd-kbd">Space</kbd></button>
          <button type="button" class="btn small" :aria-pressed="slow" title="Half speed (S)" @click="slow = !slow">½× <kbd class="sd-kbd">S</kbd></button>
          <span v-if="event.verdict" class="sd-chip-mine" :class="{cant: event.verdict.verdict !== 'crossing'}">{{
            event.verdict.verdict === 'crossing' ? 'Crosses my path' : 'Does not cross' }}</span>
        </div>

        <div v-if="writable" class="sd-actions" role="group" aria-label="Your verdict">
          <button type="button" class="btn primary" id="cx-yes" :disabled="saving" @click="send('crossing')">Crosses my path
            <kbd class="sd-kbd">Y</kbd></button>
          <button type="button" class="btn" id="cx-no" :disabled="saving" @click="send('not_crossing')">Does not cross
            <kbd class="sd-kbd">N</kbd></button>
          <button v-if="event.verdict" type="button" class="btn" id="cx-clear" :disabled="saving" @click="send('clear')">Clear</button>
        </div>
        <p v-else-if="readOnlyNote" class="dr-hint sd-readonly">{{ readOnlyNote }}</p>
        <p v-if="verdictWhen" class="dr-hint">{{ verdictWhen }}</p>

        <div class="sd-next">
          <div class="sd-step-signs" role="group" aria-label="Vehicle before or after">
            <button type="button" class="btn" :disabled="position <= 0" @click="step(-1)"><span aria-hidden="true">‹</span> Before
              <kbd class="sd-kbd">←</kbd></button>
            <button type="button" class="btn" :disabled="position >= events.length - 1" @click="step(1)">After <span
              aria-hidden="true">›</span> <kbd class="sd-kbd">→</kbd></button>
          </div>
          <template v-if="!remaining">
            <span class="sd-done">Every vehicle of this drive is reviewed.</span>
            <button type="button" class="btn" :disabled="nextDriveBusy" @click="nextDrive">Next drive ›</button>
          </template>
          <span v-else class="dr-hint">{{ remaining }} to review</span>
          <a class="link-button sd-car" :href="'/speed-review?drive=' + encodeURIComponent(route)">What the car did here ›</a>
        </div>
        <p class="dr-hint sd-keys">Keys:<template v-if="writable"> Y crosses my path, N does not,</template> Space again, S half speed,
          ← → the vehicle before or after. Only vehicles that cross your path count: a parked car, the car ahead or oncoming
          traffic does not.</p>
      </template>
    </template>
  </section>
</template>
