<script setup lang="ts">
// "AI reviewer" under the bar of the photo, only once Ask AI (its button in the bar, key I) was pressed on this frame:
// the model chosen in Settings at work, or its answer box by box (Keep, Correct, Add, Delete, Not judged), drawn on
// the photo in its own colour; Apply makes it the frame's edits (saved only by Confirm, as the reviewer's own review),
// Dismiss drops it. A frame of the route's "Try without saving" opens with that suggestion, marked as such. Hidden
// offline.
import {computed} from 'vue';
import {isoWhen} from '../common/format';
import {useNow} from '../composables/now';
import {ai, aiPhase, applyAi, dismissAi, VERDICT} from './ai';
import {describe} from './catalog';
import {releaseFocus} from './dom';
import {report} from './store';
import type {FrameRow} from './types';

const props = defineProps<{row: FrameRow}>();
const now = useNow();
const ask = computed(() => ai.asks[props.row.frame_key] ?? null);
const phase = computed(() => aiPhase(props.row.frame_key));
const provider = computed(() => ai.state?.provider ?? null);
const waited = computed(() => (ask.value?.phase === 'asking' ? Math.max(0, Math.floor((now.value - ask.value.since) / 1000)) : 0));
const suggestion = computed(() => (ask.value?.phase === 'ready' ? ask.value.suggestion : null));

interface Item {
  key: string;
  /** The box's number once applied; for a deletion, its number on the frame now. */
  number: string;
  label: string;
  kind: string;
  what: string;
  notes: string;
}

const items = computed((): Item[] => {
  const value = suggestion.value;
  if (value === null) return [];
  const kept = value.detections.map((d, i): Item => {
    const [label, kind] = VERDICT[d.ai_action];
    return {key: 'box-' + i, number: String(i + 1), label, kind, what: describe(d), notes: d.notes ?? ''};
  });
  // A trial numbered the proposals in the report's order, not the page's: no box number then.
  const numbered = !ask.value?.trial;
  const deleted = value.deleted.map((d): Item => {
    const [label, kind] = VERDICT.delete;
    const what = describe({bbox_xyxy: d.bbox_xyxy, family: d.family, value_kph: d.value_kph});
    return {key: 'delete-' + d.number, number: '✕', label, kind, what: numbered ? 'Box ' + d.number + ': ' + what : what,
      notes: d.notes};
  });
  return [...kept, ...deleted];
});
/** "From “Try without saving” · 25 Oct, 10:01" over a suggestion of the trial. */
const trialText = computed(() => {
  const trial = ask.value?.trial;
  if (!trial) return '';
  const when = isoWhen(trial.at);
  return 'From “Try without saving”' + (when ? ' · ' + when : '');
});
</script>

<template>
  <section v-if="report.apiBase && phase !== 'none'" class="ai-section stage-extra" id="aiSection" :data-ai-phase="phase"
           aria-labelledby="aiTitle">
    <div class="section-head">
      <h2 id="aiTitle">AI reviewer</h2>
      <span class="count ai-model" id="aiModel" :title="provider?.reviewer_name">{{ provider ? provider.name + ' · ' + provider.model : '' }}</span>
    </div>
    <p v-if="phase === 'asking'" class="ai-note ai-wait" id="aiWait" role="status"><span class="ai-spin" aria-hidden="true"></span>Asking {{ provider?.model ?? 'the AI' }}… {{ waited }} s</p>
    <template v-else-if="phase === 'failed'">
      <p class="ai-note ai-error" id="aiError" role="alert">{{ ask?.error }}</p>
      <div class="ai-actions"><button type="button" class="btn small" id="aiDismiss" @click="dismissAi(); releaseFocus($event)">Dismiss</button></div>
    </template>
    <template v-else-if="suggestion">
      <p v-if="trialText" class="ai-note" id="aiTrial">{{ trialText }}</p>
      <p class="ai-msg" id="aiMessage">{{ suggestion.message }}</p>
      <div class="ai-actions">
        <button type="button" class="btn small primary" id="aiApply" title="The suggestion becomes the boxes of this frame: check them, then Confirm"
                @click="applyAi(); releaseFocus($event)">Apply</button>
        <button type="button" class="btn small" id="aiDismiss" @click="dismissAi(); releaseFocus($event)">Dismiss</button>
      </div>
      <ul v-if="items.length > 0" class="ai-list" id="aiList">
        <li v-for="item in items" :key="item.key" class="ai-row" :data-verdict="item.kind">
          <span class="n">{{ item.number }}</span>
          <span :class="'pill ' + item.kind">{{ item.label }}</span>
          <span class="what" :title="item.what">{{ item.what }}</span>
          <span v-if="item.notes" class="ai-notes">{{ item.notes }}</span>
        </li>
      </ul>
      <p v-else class="ai-note">No sign: applying leaves the frame with no boxes.</p>
      <p v-if="suggestion.comment" class="ai-note ai-comment">“{{ suggestion.comment }}”</p>
      <p v-for="(warning, i) in suggestion.warnings" :key="i" class="ai-note ai-warn">{{ warning }}</p>
    </template>
  </section>
</template>
