<script setup lang="ts">
// The confirm block at the bottom of the inspector: warnings (server, browser storage, frames not saved), the
// save line of the frame with Discard changes / Withdraw confirmation, Confirm and the keyboard shortcuts.
import {computed} from 'vue';
import {signStyleOf} from './catalog';
import {status} from './current';
import {countryOf} from './frames';
import {confirmFrame, discardChanges, withdrawFrame} from './confirm';
import {releaseFocus} from './dom';
import {confirmLabels, pageNotices, statusLine} from './messages';
import {retryUnsaved} from './server';
import {report, state, toggleKeys} from './store';
import type {FrameEdit, FrameRow} from './types';

const props = defineProps<{row: FrameRow; work: FrameEdit}>();
const notices = computed(() => pageNotices());
// In the US a speed is two keys, in mph.
const usSigns = computed(() => signStyleOf(countryOf(props.row, props.work)) === 'us');
const line = computed(() => statusLine(props.row.frame_key, status.value));
const labels = computed(() => confirmLabels(props.row.frame_key, props.work, status.value));
// Nothing to say under the panel: on a phone that section folds away instead of leaving an empty block.
const quiet = computed(() => line.value.text === '' && notices.value.length === 0 && !line.value.discard
  && !line.value.withdraw);
</script>

<template>
  <section class="confirm-section" :class="{quiet}">
    <div class="notices" id="notices">
      <div v-for="(notice, i) in notices" :key="i" :class="notice.error ? 'notice error' : 'notice'" role="alert">
        <span>{{ notice.text }}</span>
        <button v-if="notice.retry" type="button" class="btn small" @click="retryUnsaved(true); releaseFocus($event)">Retry saving</button>
      </div>
    </div>
    <div class="confirm-area">
      <div class="status-line">
        <p :class="line.cls ? 'save-msg ' + line.cls : 'save-msg'" id="saveMsg" role="status" :title="line.text">{{ line.text }}</p>
        <div class="status-actions">
          <button type="button" class="link" id="discard" :hidden="!line.discard" @click="discardChanges">Discard changes</button>
          <button type="button" class="link danger" id="withdraw" :hidden="!line.withdraw" :data-armed="line.armed"
                  @click="withdrawFrame">{{ line.armed ? 'Yes, withdraw' : 'Withdraw confirmation' }}</button>
        </div>
      </div>
      <button type="button" class="btn primary confirm" :class="{already: labels.already}" id="confirm"
              data-action="confirm" :disabled="status === 'saving' || report.readOnly" :data-save-state="status"
              @click="confirmFrame"><span class="btn-label" id="confirmLabel">{{ report.readOnly ? 'Read only' : labels.label }}</span><kbd class="keycap" aria-hidden="true">↵</kbd></button>
      <button type="button" class="link keys-toggle" id="keysToggle" :aria-expanded="state.showKeys"
              @click="toggleKeys(); releaseFocus($event)">{{ state.showKeys ? 'Hide the keys' : 'Keys' }} <kbd>?</kbd></button>
      <p v-if="state.showKeys" class="keys" id="keys" aria-label="Keyboard shortcuts"><span><kbd>←</kbd> <kbd>→</kbd> frame</span> <span><kbd>⇧</kbd> <kbd>←</kbd> <kbd>→</kbd> second</span> <span><kbd>↑</kbd> <kbd>↓</kbd> box</span> <span v-if="usSigns">speed in mph: <kbd>3</kbd> <kbd>5</kbd> = 35, <kbd>6</kbd> <kbd>5</kbd> = 65, <kbd>3</kbd> alone = 30, <kbd>0</kbd> = ?</span><span v-else>speed: <kbd>2</kbd> = 20 … <kbd>9</kbd> = 90, <kbd>1</kbd> = 100, <kbd>1</kbd> <kbd>2</kbd> = 120, <kbd>0</kbd> = ?</span> <span><kbd>O</kbd> <kbd>X</kbd> <kbd>U</kbd> road</span> <span><kbd>S</kbd> other sign, ours</span> <span><kbd>V</kbd> on a vehicle</span> <span><kbd>L</kbd> variable limit</span> <span><kbd>G</kbd> give way</span> <span><kbd>P</kbd> stop</span> <span><kbd>R</kbd> roundabout</span> <span><kbd>E</kbd> something else</span> <span><kbd>M</kbd> main road</span> <span><kbd>W</kbd> person</span> <span><kbd>Y</kbd> bicycle</span> <span><kbd>K</kbd> car</span> <span><kbd>J</kbd> road users complete</span> <span><kbd>Enter</kbd> confirm</span> <span><kbd>B</kbd> new box</span> <span :hidden="!report.apiBase"><kbd>C</kbd> fit every box</span> <span :hidden="!report.apiBase"><kbd>⇧</kbd> <kbd>C</kbd> fit with AI</span> <span :hidden="!report.apiBase"><kbd>A</kbd> apply previous</span> <span :hidden="!report.apiBase"><kbd>I</kbd> ask AI</span> <span :hidden="!report.apiBase"><kbd>T</kbd> car networks</span> <span><kbd>Del</kbd> delete</span> <span><kbd>D</kbd> delete all boxes</span> <span id="keyHints" :hidden="!report.routeHasHints"><kbd>H</kbd> second reading</span> <span><kbd>Esc</kbd> cancel</span> <span><kbd>?</kbd> hide the keys</span></p>
    </div>
  </section>
</template>
