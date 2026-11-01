<script setup lang="ts">
// The phone's bottom bar: a second back, the frame before, Confirm (saying what it saves), the frame after, a
// second on, and the message that must not be missed (a save that failed, a warning).
import {computed, useTemplateRef} from 'vue';
import {position, status} from './current';
import {confirmFrame} from './confirm';
import {confirmLabels, statusLine} from './messages';
import {stepSecond} from './navigation';
import {stepFrame} from './video';
import {report} from './store';
import type {FrameEdit, FrameRow} from './types';
import {useHeightVar} from './use-layout';

const props = defineProps<{row: FrameRow | null; work: FrameEdit | null}>();
const bar = useTemplateRef<HTMLElement>('bar');
useHeightVar(bar, '--mobilebar-h');

const labels = computed(() => (props.row && props.work
  ? confirmLabels(props.row.frame_key, props.work, status.value) : {label: 'Confirm', mobileLabel: 'Confirm', already: false}));
const message = computed(() => (props.row ? statusLine(props.row.frame_key, status.value).mobile : ''));
</script>

<template>
  <nav ref="bar" class="mobilebar" id="mobilebar" aria-label="Navigation" :hidden="row === null">
    <p class="mobile-msg" id="mobileMsg" role="alert" :hidden="message === ''">{{ message }}</p>
    <div class="mobile-buttons">
      <button type="button" class="btn second" id="mPrevSecond" aria-label="A second back" :disabled="!position.canPrevSecond"
              @click="stepSecond(-1)">«</button>
      <button type="button" class="btn" id="mPrev" aria-label="Previous frame" :disabled="!position.canPrev"
              @click="stepFrame(-1)">‹</button>
      <button type="button" class="btn primary confirm" :class="{already: labels.already}" id="mConfirm" data-action="confirm"
              :disabled="status === 'saving' || report.readOnly" :data-save-state="status" @click="confirmFrame"><span class="btn-label" id="mConfirmLabel">{{ report.readOnly ? 'Read only' : labels.mobileLabel }}</span><kbd class="keycap" aria-hidden="true">↵</kbd></button>
      <button type="button" class="btn" id="mNext" aria-label="Next frame"
              :disabled="!position.canNext" @click="stepFrame(1)">›</button>
      <button type="button" class="btn second" id="mNextSecond" aria-label="A second on" :disabled="!position.canNextSecond"
              @click="stepSecond(1)">»</button>
    </div>
  </nav>
</template>
