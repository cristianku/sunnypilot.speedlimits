<script setup lang="ts">
// A person's verdict on an episode of a drive (DriveReviewDrive.vue), or on any stretch of it: the limit was wrong, it
// was right and the driver acted for another reason, or it cannot be told (with why). A wrong limit needs the right
// one: its value, where it starts and until when it holds (until the next one when the end is left open), each set on
// its own; the limit can be "unknown" (nothing, earlier signs included, establishes it: not the same as a sign
// missing from the picture). Apart from the limit: the number the sign showed when it differs, the kind of road (never a
// limit), and whether the person is certain. Saved with the revision read: a change made meanwhile elsewhere is
// refused (409) and the drive is read again.
import {computed, ref, watch} from 'vue';
import {ApiError, messageOf, post} from '../common/api';
import {askConfirm} from '../composables/confirm';
import {toast} from '../composables/toast';
import {type Annotation, CHANGES, clockOf, type Episode, type Interval, OTHER_REASONS, QUICK_LIMITS, UNDETERMINABLE_REASONS,
  type Verdict, VERDICTS} from './drive-review';
import type {DraftInterval} from './DriveReviewLanes.vue';
import {ROAD_TYPES} from './roads';

const props = defineProps<{route: string; episode: Episode | null; annotation: Annotation | null; cursor: number; t0: number;
  readOnly: boolean}>();
const emit = defineEmits<{saved: [annotation: Annotation]; reload: []; seek: [seconds: number]; draft: [intervals: DraftInterval[]]}>();

interface Row {key: number; from: number; to: number | null; unknown: boolean; value: number | null; change: string | null;
  signRead: number | null; scene: string | null; certain: boolean}
let keys = 0;
const verdict = ref<Verdict | null>(null);
const reason = ref<string | null>(null);
const undeterminable = ref<string | null>(null);
const note = ref('');
const rows = ref<Row[]>([]);
const dirty = ref(false);
const saving = ref(false);

const seconds = (ns: number): number => (ns - props.t0) / 1e9;
const nanos = (s: number): number => Math.round(props.t0 + s * 1e9);
function load(): void {
  const payload = props.annotation?.payload;
  verdict.value = payload?.verdict ?? null;
  reason.value = payload?.reason ?? null;
  undeterminable.value = payload?.undeterminable_reason ?? null;
  note.value = payload?.note ?? '';
  rows.value = (payload?.intervals ?? []).map((item: Interval): Row => ({key: ++keys, from: seconds(item.start_ns),
    to: item.end_ns === null ? null : seconds(item.end_ns), unknown: item.limit.state === 'unknown', value: item.limit.value_kph,
    change: item.change, signRead: item.sign_read_kph, scene: item.scene, certain: item.certain}));
  dirty.value = false;
}
watch(() => [props.annotation?.id, props.annotation?.revision, props.episode?.id], load, {immediate: true});
watch([verdict, reason, undeterminable, note, rows], () => { dirty.value = true; }, {deep: true, flush: 'sync'});
watch(rows, () => emit('draft', rows.value.map((row) => ({from: row.from, to: row.to,
  text: row.unknown ? '?' : row.value === null ? '…' : String(row.value)}))), {deep: true, immediate: true});

function addRow(): void {
  const at = props.cursor;
  rows.value = [...rows.value, {key: ++keys, from: at, to: null, unknown: false, value: null, change: 'sign', signRead: null,
    scene: null, certain: true}].sort((a, b) => a.from - b.from);
  if (verdict.value === null) verdict.value = 'wrong_limit';
}
function removeRow(key: number): void {
  rows.value = rows.value.filter((row) => row.key !== key);
}
function setStart(row: Row): void {
  row.from = props.cursor;
  if (row.to !== null && row.to <= row.from) row.to = null;
  rows.value = [...rows.value].sort((a, b) => a.from - b.from);
}
function setEnd(row: Row): void {
  if (props.cursor > row.from) row.to = props.cursor;
  else toast('The end must come after the start: move further on first.', true);
}
const numberOf = (text: string): number | null => {
  const value = Number(text);
  return text.trim() === '' || !Number.isFinite(value) ? null : Math.round(value);
};

/** What stops a save, in words; empty when it can be saved. */
const problem = computed((): string => {
  if (verdict.value === null) return 'Choose a verdict.';
  if (verdict.value === 'undeterminable' && !undeterminable.value) return 'Say why it cannot be told.';
  if (verdict.value === 'wrong_limit' && !rows.value.length) return 'A wrong limit needs the right one: add it from where it starts.';
  for (const [index, row] of rows.value.entries()) {
    if (!row.unknown && (row.value === null || row.value < 5 || row.value > 200)) return 'Give the limit of every stretch (5–200 km/h), or mark it unknown.';
    if (row.signRead !== null && (row.signRead < 5 || row.signRead > 200)) return 'The number on the sign must be 5–200 km/h.';
    const next = rows.value[index + 1];
    if (next && row.to !== null && row.to > next.from) return 'Two stretches overlap: end the first before the next starts.';
  }
  return '';
});

async function save(): Promise<void> {
  if (problem.value || saving.value) return;
  saving.value = true;
  const body = {
    route: props.route, episode_id: props.annotation?.episode_id ?? props.episode?.id ?? null,
    ...(props.annotation ? {id: props.annotation.id, base_revision: props.annotation.revision} : {}),
    verdict: verdict.value, reason: verdict.value === 'correct_limit' ? reason.value : null,
    undeterminable_reason: verdict.value === 'undeterminable' ? undeterminable.value : null, note: note.value.trim() || null,
    intervals: rows.value.map((row) => ({start_ns: nanos(row.from), end_ns: row.to === null ? null : nanos(row.to),
      limit: {state: row.unknown ? 'unknown' : 'known', value_kph: row.unknown ? null : row.value}, change: row.change,
      sign_read_kph: row.signRead, scene: row.scene, certain: row.certain, note: null})),
  };
  try {
    const answer = await post<{annotation: Annotation}>('/api/drive-review/annotations', body);
    toast('Saved.');
    emit('saved', answer.annotation);
  } catch (error) {
    toast(messageOf(error), true);
    if (error instanceof ApiError && (error.status === 409 || error.status === 404)) emit('reload');
  } finally {
    saving.value = false;
  }
}
async function remove(): Promise<void> {
  const current = props.annotation;
  if (!current) return;
  const ok = await askConfirm({title: 'Delete this verdict?', text: 'Its history stays on the server; the episode is back to review.',
    action: 'Delete', danger: true});
  if (!ok) return;
  try {
    await post('/api/drive-review/annotations/' + current.id + '/delete', {route: props.route, base_revision: current.revision});
    toast('Deleted.');
  } catch (error) {
    toast(messageOf(error), true);
  }
  emit('reload');
}
</script>

<template>
  <section class="dr-note" aria-labelledby="dr-note-title">
    <div class="dr-note-head">
      <h3 id="dr-note-title">{{ episode ? 'Your verdict' : 'A note on this stretch' }}</h3>
      <span v-if="annotation" class="dr-note-who">{{ annotation.author }} · revision {{ annotation.revision }}</span>
    </div>
    <div class="dr-verdicts" role="radiogroup" aria-label="Verdict">
      <button v-for="[key, label] in VERDICTS" :key="key" type="button" role="radio" class="btn small" :class="'dr-v-' + key"
              :aria-checked="verdict === key" :disabled="readOnly" @click="verdict = key">{{ label }}</button>
    </div>
    <div v-if="verdict === 'correct_limit'" class="dr-reasons" role="radiogroup" aria-label="Why the driver acted">
      <span class="dr-field-name">Why the driver acted</span>
      <button v-for="[key, label] in OTHER_REASONS" :key="key" type="button" role="radio" class="dr-choice" :aria-checked="reason === key"
              :disabled="readOnly" @click="reason = reason === key ? null : key">{{ label }}</button>
    </div>
    <div v-if="verdict === 'undeterminable'" class="dr-reasons" role="radiogroup" aria-label="Why it cannot be told">
      <span class="dr-field-name">Why it cannot be told</span>
      <button v-for="[key, label] in UNDETERMINABLE_REASONS" :key="key" type="button" role="radio" class="dr-choice"
              :aria-checked="undeterminable === key" :disabled="readOnly" @click="undeterminable = key">{{ label }}</button>
    </div>

    <div class="dr-rows">
      <div class="dr-rows-head">
        <span class="dr-field-name">The applicable limit</span>
        <button type="button" class="btn small" :disabled="readOnly" id="dr-add-limit" @click="addRow">Add a limit from here ({{ clockOf(cursor) }})</button>
      </div>
      <p v-if="!rows.length" class="dr-hint">Nothing set. A wrong limit needs the right one: go to where it starts, then Add a limit from here.</p>
      <div v-for="row in rows" :key="row.key" class="dr-row">
        <div class="dr-row-limit">
          <label class="dr-kph"><span class="sr-only">Limit in km/h</span>
            <input type="number" inputmode="numeric" min="5" max="200" step="5" placeholder="km/h" :disabled="readOnly || row.unknown"
                   :value="row.value ?? ''" @change="row.value = numberOf(($event.target as HTMLInputElement).value)"/></label>
          <div class="dr-quick">
            <button v-for="value in QUICK_LIMITS" :key="value" type="button" class="dr-choice" :aria-pressed="!row.unknown && row.value === value"
                    :disabled="readOnly" @click="row.unknown = false; row.value = value">{{ value }}</button>
            <button type="button" class="dr-choice" :aria-pressed="row.unknown" :disabled="readOnly"
                    title="Nothing, earlier signs included, establishes the limit here" @click="row.unknown = !row.unknown">Unknown</button>
          </div>
        </div>
        <div class="dr-row-times">
          <span>From <button type="button" class="link-button" @click="emit('seek', row.from)">{{ clockOf(row.from) }}</button>
            <button type="button" class="btn small" :disabled="readOnly" @click="setStart(row)">Here</button></span>
          <span>until <button v-if="row.to !== null" type="button" class="link-button" @click="emit('seek', row.to)">{{ clockOf(row.to) }}</button>
            <template v-else>the next limit</template>
            <button type="button" class="btn small" :disabled="readOnly" @click="setEnd(row)">Here</button>
            <button v-if="row.to !== null" type="button" class="btn small" :disabled="readOnly" @click="row.to = null">Open</button></span>
        </div>
        <div class="dr-row-more">
          <label class="wp-field"><span>Set by</span>
            <select v-model="row.change" :disabled="readOnly"><option :value="null">–</option>
              <option v-for="[key, label] in CHANGES" :key="key" :value="key">{{ label }}</option></select></label>
          <label class="wp-field"><span>Sign showed</span>
            <input type="number" inputmode="numeric" min="5" max="200" step="5" placeholder="same" class="dr-small-input" :disabled="readOnly"
                   :value="row.signRead ?? ''" @change="row.signRead = numberOf(($event.target as HTMLInputElement).value)"/></label>
          <label class="wp-field"><span>Road</span>
            <select v-model="row.scene" :disabled="readOnly"><option :value="null">–</option>
              <option v-for="[key, label] in ROAD_TYPES.filter(([name]) => name !== 'unknown')" :key="key" :value="key">{{ label }}</option></select></label>
          <label class="dr-check"><input v-model="row.certain" type="checkbox" :disabled="readOnly"/> Certain</label>
          <button type="button" class="link-button dr-remove" :disabled="readOnly" @click="removeRow(row.key)">Remove</button>
        </div>
      </div>
    </div>

    <label class="dr-note-text"><span class="dr-field-name">Note</span>
      <textarea v-model="note" rows="2" maxlength="1000" :disabled="readOnly" placeholder="What you saw (optional)"></textarea></label>
    <div class="dr-note-actions">
      <button type="button" class="btn primary" id="dr-save" :disabled="readOnly || saving || Boolean(problem) || (!dirty && Boolean(annotation))"
              @click="save">{{ annotation ? 'Save changes' : 'Save' }}</button>
      <button v-if="annotation" type="button" class="btn" :disabled="readOnly" @click="remove">Delete</button>
      <span v-if="problem && verdict !== null" class="dr-hint">{{ problem }}</span>
    </div>
  </section>
</template>
