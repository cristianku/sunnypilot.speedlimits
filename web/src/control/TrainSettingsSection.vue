<script setup lang="ts">
// The settings of a training of Train (Cutter, Classifier, Applicability, Brain): its epochs, patience, batch, learning
// rate, the processes reading the frames, the classifier's false boxes; GET/POST /api/panel/train-settings
// (nnslr_tools.train_settings). Saved here, the next training of that kind uses them (Panel.start sets the variables
// of its script); the one running keeps its own, and the job's chips say what each one used. What the car needs (the
// cutter's band, grid, channels, fuse) is shown fixed. The look of On the car (CarSettingsSection), folded in a row.
import {computed, onMounted, reactive, ref, shallowRef} from 'vue';
import {api, messageOf, post} from '../common/api';
import {isoWhen} from '../common/format';
import type {TrainSetting, TrainSettingsAction, TrainSettingsPayload} from '../common/panel-types';
import {toast} from '../composables/toast';

const props = defineProps<{training: TrainSettingsAction; running: boolean}>();
const payload = shallowRef<TrainSettingsPayload | null>(null);
const error = ref('');
const saving = ref(false);
/** The values typed, by training and key ("train_cutter.epochs"). */
const draft = reactive<Record<string, string | number>>({});
const field = (key: string): string => props.training + '.' + key;

const current = computed(() => payload.value?.trainings[props.training] ?? null);
const settings = computed(() => current.value?.settings ?? []);
const readOnly = computed(() => !payload.value?.can_edit);
/** The network inside a sentence. */
const networkWords = computed(() => current.value?.label.toLowerCase() ?? 'network');

/** The value typed, NaN when not a number (v-model of a number input gives a number once typed). */
function typed(setting: TrainSetting): number {
  const text = String(draft[field(setting.key)] ?? '').trim();
  return text === '' ? NaN : Number(text);
}
const whole = (setting: TrainSetting): boolean => Number.isInteger(setting.step) && setting.step >= 1;

function fill(value: TrainSettingsPayload): void {
  payload.value = value;
  for (const [action, training] of Object.entries(value.trainings)) {
    for (const setting of training.settings) draft[action + '.' + setting.key] = String(setting.value);
  }
}
async function refresh(): Promise<void> {
  try {
    fill(await api<TrainSettingsPayload>('/api/panel/train-settings'));
    error.value = '';
  } catch (problem) {
    error.value = messageOf(problem);
  }
}
onMounted(refresh);

const boundsOf = (setting: TrainSetting): string => setting.min + ' to ' + setting.max + (setting.unit ? ' ' + setting.unit : '');
/** A typed value that is not a number in its bounds, or ''. */
function problemOf(setting: TrainSetting): string {
  const value = typed(setting);
  if (!Number.isFinite(value)) return 'A number';
  if (whole(setting) && !Number.isInteger(value)) return 'A whole number';
  if (value < setting.min || value > setting.max) return boundsOf(setting);
  return '';
}
const changed = computed(() => settings.value.filter((setting) => typed(setting) !== setting.value));
const invalid = computed(() => settings.value.some((setting) => problemOf(setting) !== ''));
const allDefault = computed(() => settings.value.every((setting) => setting.value === setting.default));
/** The row folded: the settings changed from the default, or that they are all the default. */
const summary = computed(() => {
  const set = settings.value.filter((setting) => setting.value !== setting.default);
  return set.length ? set.map((setting) => setting.label.toLowerCase() + ' ' + setting.value).join(' · ') : 'All default';
});

async function save(values: Record<string, number | null>): Promise<void> {
  saving.value = true;
  try {
    fill(await post<TrainSettingsPayload>('/api/panel/train-settings', {training: props.training, values}));
    toast('Saved: the next training of the ' + networkWords.value + ' uses them');
  } catch (problem) {
    toast(messageOf(problem), true);
  } finally {
    saving.value = false;
  }
}
function saveChanged(): void {
  void save(Object.fromEntries(changed.value.map((setting) => [setting.key, typed(setting)])));
}
function resetAll(): void {
  void save(Object.fromEntries(settings.value.map((setting) => [setting.key, null])));
}
</script>

<template>
  <details v-if="current" class="latest-row tset" :id="'train-settings-' + training">
    <summary><span class="latest-what"><span class="latest-name">Settings</span>
      <span class="latest-text">{{ summary }}</span></span><span class="latest-chev" aria-hidden="true">›</span></summary>
    <div class="tset-body">
      <p class="note">Used by the next training of the {{ networkWords }}<template v-if="running">: the one
        running keeps its own</template>. Each job says the values it used.</p>
      <label v-for="setting in settings" :key="setting.key" class="cs-row" :data-setting="setting.key">
        <span class="cs-text"><b>{{ setting.label }}</b><small>{{ setting.help }}<template
          v-if="setting.value !== setting.default"> Default {{ setting.default }}.</template></small></span>
        <span class="cs-input">
          <!-- Decimals get a wider field: 0.00001 fits on a phone. -->
          <input type="number" :min="setting.min" :max="setting.max" :step="setting.step"
                 :class="{wide: !whole(setting)}" :inputmode="whole(setting) ? 'numeric' : 'decimal'" v-model="draft[field(setting.key)]"
                 :disabled="readOnly || saving" :aria-invalid="problemOf(setting) !== ''" :title="boundsOf(setting)">
          <small>{{ setting.unit }}</small>
        </span>
      </label>
      <template v-if="current.fixed.length">
        <h3 class="ds-title">Fixed for the car</h3>
        <div v-for="item in current.fixed" :key="item.label" class="cs-row tset-fixed">
          <span class="cs-text"><b>{{ item.label }}</b><small>{{ item.help }}</small></span>
          <span class="tset-value">{{ item.value }}</span>
        </div>
      </template>
      <div class="cs-foot">
        <span class="cs-state">{{ readOnly ? 'Read-only on this server.' : allDefault ? 'The scripts’ defaults.' : 'Changed from the defaults.' }}<template
          v-if="current.at"> Changed {{ isoWhen(current.at) }}{{ current.by ? ' by ' + current.by : '' }}.</template></span>
        <button type="button" class="link-button" :disabled="readOnly || saving || allDefault" @click="resetAll">All to default</button>
        <button type="button" class="btn primary small" :id="'train-settings-save-' + training"
                :disabled="readOnly || saving || invalid || changed.length === 0" @click="saveChanged">Save</button>
      </div>
    </div>
  </details>
  <p v-else-if="error" class="latest-none">Settings not loaded: {{ error }}</p>
</template>
