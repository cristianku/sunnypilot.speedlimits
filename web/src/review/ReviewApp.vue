<script setup lang="ts">
// The sign review: one route's frames, the boxes the model proposed, the reviewer's confirmation of each
// frame. Served by the review server (every confirmation saved in its database) or written by write_review as
// a static index.html that works offline (payload "server": false, confirmations kept in this browser, JSONL
// export). The state lives in store.ts; the plain modules of web/src/review change it (navigation, edits,
// confirm, server, export) and the components below draw it.
import {computed, nextTick, onBeforeUnmount, onMounted, watchEffect} from 'vue';
import {aiPhase, loadAiState} from './ai';
import {watchAiLive} from './ai-live';
import {watchBusy} from './busy';
import {useCarReading} from './car';
import {loadCarLogs} from './car-moment';
import CarMoment from './CarMoment.vue';
import {ensureCurrentFrame, row, status, work} from './current';
import {unsavedKeys, workOf} from './frames';
import Inspector from './Inspector.vue';
import MobileBar from './MobileBar.vue';
import {pickStartFrame, startAtWork} from './navigation';
import {scrollTimelineToCurrent} from './scroll';
import {loadServer, retryUnsaved} from './server';
import Stage from './Stage.vue';
import StageBar from './StageBar.vue';
import {initStore, report, restore, rowOf, state} from './store';
import Timeline from './Timeline.vue';
import type {Payload} from './types';
import {useKeyboard} from './use-keyboard';

const props = defineProps<{payload: Partial<Payload>}>();

initStore(props.payload);
restore();
state.currentKey = pickStartFrame();
// As when a frame is opened: its first box selected, so the keys (speed, road) work at once.
const startRow = state.currentKey === null ? null : rowOf(state.currentKey);
if (startRow !== null && workOf(startRow).detections.length > 0) state.selected = 0;
if (report.apiBase) state.server = 'loading';
ensureCurrentFrame();
useKeyboard();
useCarReading();
// The drive at work has none until its first proposals come (review/busy.ts adds them).
const hasFrames = computed(() => report.rows.length > 0);

// State for whoever drives the page (an agent with Playwright): attributes only, the layout does not change.
watchEffect(() => {
  const shown = row.value;
  if (shown === null) return;
  const key = shown.frame_key;
  const s = state.edits[key];
  const data = document.body.dataset;
  data['frameKey'] = key;
  data['frameStatus'] = status.value;
  data['frameReviewerKind'] = s && s.server_confirmed ? (s.reviewer_kind || 'human') : '';
  data['server'] = state.server;
  data['unsaved'] = String(unsavedKeys().length);
  data['saving'] = state.inFlight.size > 0 ? 'true' : 'false';
  data['saveCount'] = String(state.saveCount);
  // The AI suggestion of the frame: none, asking, ready or failed.
  data['ai'] = aiPhase(key);
  if (state.lastSave) {
    data['lastSaveKey'] = state.lastSave.key;
    data['lastSaveResult'] = state.lastSave.result;
    data['lastSaveRevision'] = String(state.lastSave.revision);
  }
}, {flush: 'post'});

const retryLater = (): void => void retryUnsaved(false);
let timer: number | undefined;
onMounted(() => {
  void nextTick(scrollTimelineToCurrent);
  if (!report.apiBase) return;
  watchBusy();
  void loadAiState();
  void loadCarLogs();
  // A confirmed frame gets its boxes from the server: then its first box is selected too.
  void loadServer().then(() => {
    startAtWork();
    watchAiLive();
    const shown = row.value;
    if (shown !== null && state.selected < 0 && workOf(shown).detections.length > 0) state.selected = 0;
  });
  timer = window.setInterval(retryLater, 30000);
  window.addEventListener('online', retryLater);
});
onBeforeUnmount(() => {
  window.clearInterval(timer);
  window.removeEventListener('online', retryLater);
});
</script>

<template>
  <div class="page">
    <div id="stateBox" class="state-box" :hidden="hasFrames">
      <template v-if="!hasFrames && report.atWork">
        <h2>{{ state.busy?.proposer ?? 'Propose again' }} is starting on this drive</h2>
        <p>Its first proposals appear here in a minute.</p>
      </template>
      <template v-else-if="!hasFrames">
        <h2>This report has no frames</h2>
        <p>Choose another route above, or run the pre-annotation again.</p>
      </template>
    </div>
    <div id="workspace" class="workspace" :class="{'tl-open': state.timelineOpen}" :hidden="!hasFrames">
      <template v-if="row && work">
        <div class="stage-col">
          <CarMoment :row="row" :work="work"/>
          <Stage :row="row" :work="work"/>
          <StageBar :row="row"/>
          <Timeline/>
        </div>
        <Inspector :row="row" :work="work"/>
      </template>
    </div>
  </div>
  <MobileBar :row="row" :work="work"/>
</template>
