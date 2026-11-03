<script setup lang="ts">
// The bar under the photo: position in the route, which frames to show (ReviewFilters.vue), state tag, frame data,
// Car networks (T) and Ask AI (I), whose answers appear under the bar (CarSection.vue, AiSection.vue), drawing and
// removing boxes, the second reading shown or not.
import {computed} from 'vue';
import Icon from '../components/Icon.vue';
import {ai, aiBlocker, aiPhase, askAi} from './ai';
import {car, carAvailable, toggleCar} from './car';
import {position, selectedBox, status} from './current';
import {releaseFocus} from './dom';
import {removeSelected, toggleDrawing, toggleHints} from './edits';
import {thousands} from '../common/format';
import {aiConfirmed, framePlace, isBetweenFrame, proposalsOf, reasonsOf, statusLabel} from './frames';
import {stepSecond} from './navigation';
import {frameOfSecond} from './seconds';
import {stepFrame} from './video';
import {hintsOf, report, state} from './store';
import ReviewFilters from './ReviewFilters.vue';
import type {FrameRow} from './types';

const props = defineProps<{row: FrameRow}>();

const pillClass = computed(() => {
  const key = props.row.frame_key;
  return 'pill state ' + status.value + (aiConfirmed(key) ? ' ai' : '')
    + (status.value === 'todo' && reasonsOf(key).length > 0 ? ' sus' : '');
});

const frameData = computed(() => {
  const row = props.row;
  const proposals = proposalsOf(row).length;
  // "4:22 into the drive · frame 7/20": where the second has the frames of its video.
  const place = frameOfSecond(row);
  let info = framePlace(row) + (place === null ? '' : ' · frame ' + place[0] + '/' + place[1]);
  info += ' · ' + proposals + (proposals === 1 ? ' proposal' : ' proposals');
  if (report.routeHasHints) info += ' · ' + hintsOf(row.frame_key).length + ' in the second reading';
  return info;
});

// "Segment 3, image 12"; a frame between the seconds: "Segment 3, frame 245 of the video".
const frameTitle = computed(() => {
  const row = props.row;
  if (!isBetweenFrame(row) || typeof row.decoded_frame_index !== 'number') {
    return 'Segment ' + row.segment_index + ', image ' + row.output_index;
  }
  return 'Segment ' + row.segment_index + ', frame ' + row.decoded_frame_index + ' of the video';
});

const aiAt = computed(() => aiPhase(props.row.frame_key));
// No AI reviewer chosen in Settings: the button leads there instead of never working.
const aiUnset = computed(() => ai.state !== null && ai.state.provider === null && aiAt.value === 'none');
const aiBlock = computed(() => aiBlocker());
const aiLabel = computed(() => (aiAt.value === 'asking' ? 'Asking…' : aiAt.value === 'none' ? 'Ask AI' : 'Ask again'));
const aiTitle = computed(() => {
  if (aiBlock.value !== '') return aiBlock.value;
  const provider = ai.state?.provider;
  return (provider ? provider.name + ' · ' + provider.model + ': ' : '') + 'ask the AI about this frame (I). Nothing is saved.';
});

function hintsSwitched(event: Event): void {
  if ((event.target as HTMLInputElement).checked !== state.showHints) toggleHints();
}
</script>

<template>
  <div class="stage-bar">
    <div class="nav">
      <button type="button" class="btn icon second" id="prevSecond" aria-label="A second back" title="A second back, a whole cell (Shift ←)"
              :disabled="!position.canPrevSecond" @click="stepSecond(-1)"><span class="nav-glyph">«</span><kbd class="nav-key">⇧←</kbd></button>
      <button type="button" class="btn icon" id="prev" aria-label="Previous frame" title="Previous frame (←) · a second back: Shift ←"
              :disabled="!position.canPrev" @click="stepFrame(-1)"><span class="nav-glyph">‹</span><kbd class="nav-key">←</kbd></button>
      <span class="pos" id="pos">{{ thousands(position.index + 1) }} / {{ thousands(position.length) }}</span>
      <button type="button" class="btn icon" id="next" aria-label="Next frame" title="Next frame (→) · a second on: Shift →"
              :disabled="!position.canNext" @click="stepFrame(1)"><span class="nav-glyph">›</span><kbd class="nav-key">→</kbd></button>
      <button type="button" class="btn icon second" id="nextSecond" aria-label="A second on" title="A second on, a whole cell (Shift →)"
              :disabled="!position.canNextSecond" @click="stepSecond(1)"><span class="nav-glyph">»</span><kbd class="nav-key">⇧→</kbd></button>
    </div>
    <ReviewFilters/>
    <span id="statePill" :class="pillClass">{{ statusLabel(row.frame_key) }}</span>
    <span class="data" id="frameData" :title="frameTitle">{{ frameData }}</span>
    <div v-if="report.apiBase" class="stage-helpers">
      <button v-if="carAvailable()" type="button" class="btn car-ask" id="carToggle" :aria-pressed="car.shown"
              title="What the car's cutter and classifier read on this frame (T). Nothing is saved."
              @click="toggleCar(); releaseFocus($event)">Car networks<kbd class="keycap" aria-hidden="true">T</kbd></button>
      <a v-if="aiUnset" class="btn ai-ask" id="aiSetup" href="/account#ai-reviewer" title="No AI reviewer yet: choose one in Settings"><Icon name="ai"/>Set up AI review</a>
      <button v-else type="button" class="btn ai-ask" id="askAi" :disabled="aiBlock !== '' || aiAt === 'asking'" :title="aiTitle"
              @click="askAi(); releaseFocus($event)"><Icon name="ai"/>{{ aiLabel }}<kbd class="keycap" aria-hidden="true">I</kbd></button>
    </div>
    <div class="stage-tools">
      <button type="button" class="btn" :class="{on: state.drawing}" id="draw" :disabled="report.readOnly" @click="toggleDrawing">{{ state.drawing ? 'Cancel' : '+ Box' }}</button>
      <button type="button" class="btn danger" id="remove" :disabled="selectedBox === null" @click="removeSelected">Delete</button>
    </div>
    <label class="switch" id="hintToggle" :hidden="!report.routeHasHints"><input type="checkbox" id="showHints" :checked="state.showHints" @change="hintsSwitched"><span class="track" aria-hidden="true"></span>Second reading</label>
  </div>
</template>
