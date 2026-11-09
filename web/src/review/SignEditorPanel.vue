<script setup lang="ts">
// Sign review's editor at the side of Speed review (Signs): the bar of the frame (‹ › frame by frame, « » a second, the
// filter with AI trial, Car networks, Ask AI, + Box, Delete), the answers of the car's networks and of the AI under it
// (CarSection.vue, AiSection.vue) and the inspector (Propose again or the AI at work, the count, who reviews, the boxes,
// the chips of family, value, type and hours, the zoom of the box, Apply previous, Confirm), the same components as the
// page of Sign review (ReviewApp.vue). Its keys are Sign review's (use-keyboard.ts) while it is on screen; the frame is
// the one the player stands on (SpeedReviewDrive.vue, embed.ts). The photo with its boxes is SignEditorStage.vue, over
// the player's picture.
import {onBeforeUnmount, onMounted, watchEffect} from 'vue';
import AiSection from './AiSection.vue';
import {useCarReading} from './car';
import CarSection from './CarSection.vue';
import {row, work} from './current';
import {keepSaving} from './embed';
import {currentRow, unsavedKeys, workOf} from './frames';
import Inspector from './Inspector.vue';
import StageBar from './StageBar.vue';
import {state} from './store';
import {useKeyboard} from './use-keyboard';

useKeyboard();
useCarReading();
// The frame on screen has its working copy, and the selection points at one of its boxes (current.ts
// ensureCurrentFrame, without opening the first frame of the drive: the player says which frame is open).
watchEffect(() => {
  const shown = currentRow();
  if (shown === null) return;
  const copy = workOf(shown);
  if (state.selected >= copy.detections.length) state.selected = -1;
});
let stop: (() => void) | null = null;
onMounted(() => { stop = keepSaving(); });
onBeforeUnmount(() => stop?.());
</script>

<template>
  <div class="sign-editor sr-editor-side" :data-frame-key="row?.frame_key ?? ''" :data-server="state.server"
       :data-save-count="state.saveCount" :data-unsaved="unsavedKeys().length"
       :data-last-save="state.lastSave ? state.lastSave.key + ' ' + state.lastSave.result : ''">
    <template v-if="row && work">
      <StageBar :row="row"/>
      <CarSection :row="row" :work="work"/>
      <AiSection :row="row"/>
      <Inspector :row="row" :work="work"/>
    </template>
  </div>
</template>
