<script setup lang="ts">
// On the car (Models tab): the settings of the comma's vision limit (sunnypilot speed_limits/vision.py), how sure a
// sign or a road type must be and how long it must be seen; GET/POST /api/panel/car-settings (nnslr_tools.car_settings).
// Saved here, they reach the comma only with the next Publish or Try on the comma, in speed_limits.json.
import {computed, onMounted, reactive, ref, shallowRef} from 'vue';
import {api, messageOf, post} from '../common/api';
import {isoWhen} from '../common/format';
import {toast} from '../composables/toast';

interface CarSetting {
  key: string; label: string; help: string; default: number; min: number; max: number; step: number; unit: string;
  group: 'signs' | 'road' | 'distance'; value: number; on_car: number | null;
}
interface CarSettingsPayload {settings: CarSetting[]; by: string | null; at: string | null; sent: boolean}

const props = defineProps<{readOnly: boolean}>();
const payload = shallowRef<CarSettingsPayload | null>(null);
const error = ref('');
const saving = ref(false);
/** The values typed, as text, by key. */
const draft = reactive<Record<string, string | number>>({});
const GROUPS = [['signs', 'Signs'], ['road', 'Scene'], ['distance', 'Sign distance']] as const;

/** A probability (unit "sure") is shown and typed in %, the rest as they are. */
const percent = (setting: CarSetting): boolean => setting.unit === 'sure';
const shown = (setting: CarSetting, value: number): number => percent(setting) ? Math.round(value * 1000) / 10 : value;
const unitOf = (setting: CarSetting): string => percent(setting) ? '% sure' : setting.unit;
/** The value typed, as the server takes it (a probability back from %), NaN when not a number. */
function typed(setting: CarSetting): number {
  // v-model of a number input gives a number once typed.
  const text = String(draft[setting.key] ?? '').trim();
  const value = text === '' ? NaN : Number(text);
  return percent(setting) ? Math.round(value * 10) / 1000 : value;
}

function fill(value: CarSettingsPayload): void {
  payload.value = value;
  for (const setting of value.settings) draft[setting.key] = String(shown(setting, setting.value));
}
async function refresh(): Promise<void> {
  try {
    fill(await api<CarSettingsPayload>('/api/panel/car-settings'));
    error.value = '';
  } catch (problem) {
    error.value = messageOf(problem);
  }
}
onMounted(refresh);

/** A typed value that is not a number in its bounds, or ''. */
function problemOf(setting: CarSetting): string {
  const value = typed(setting);
  if (!Number.isFinite(value)) return 'A number';
  if (value < setting.min || value > setting.max) return boundsOf(setting);
  return '';
}
const boundsOf = (setting: CarSetting): string => shown(setting, setting.min) + ' to ' + shown(setting, setting.max) + ' ' + unitOf(setting);
const changed = computed(() => (payload.value?.settings ?? []).filter((setting) => typed(setting) !== setting.value));
const invalid = computed(() => (payload.value?.settings ?? []).some((setting) => problemOf(setting) !== ''));

async function save(values: Record<string, number | null>): Promise<void> {
  saving.value = true;
  try {
    fill(await post<CarSettingsPayload>('/api/panel/car-settings', {values}));
    toast('Saved: they reach the comma with the next Publish');
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
  void save(Object.fromEntries((payload.value?.settings ?? []).map((setting) => [setting.key, null])));
}
const allDefault = computed(() => (payload.value?.settings ?? []).every((setting) => setting.value === setting.default));
</script>

<template>
  <section class="card car-settings" id="car-settings" aria-labelledby="car-settings-title">
    <h2 id="car-settings-title">On the car</h2>
    <p class="note">How the comma turns what its camera sees into a limit. Saved here, they reach the comma with the next
      <strong>Publish</strong> (Publish tab, or Try on the comma), never on their own.</p>
    <template v-if="payload">
      <div v-for="[group, title] in GROUPS" :key="group" class="cs-group">
        <h3 class="ds-title">{{ title }}</h3>
        <label v-for="setting in payload.settings.filter((item) => item.group === group)" :key="setting.key"
               class="cs-row" :data-setting="setting.key">
          <span class="cs-text"><b>{{ setting.label }}</b><small>{{ setting.help }}<template
            v-if="setting.value !== setting.default"> Default {{ shown(setting, setting.default) }}.</template><template
            v-if="setting.on_car !== null && setting.on_car !== setting.value"> On the car now: {{ shown(setting, setting.on_car) }}.</template></small></span>
          <span class="cs-input">
            <input type="number" :min="shown(setting, setting.min)" :max="shown(setting, setting.max)"
                   :step="percent(setting) ? 0.1 : setting.step" inputmode="decimal" v-model="draft[setting.key]"
                   :disabled="readOnly || saving" :aria-invalid="problemOf(setting) !== ''" :title="boundsOf(setting)">
            <small>{{ unitOf(setting) }}</small>
          </span>
        </label>
      </div>
      <div class="cs-foot">
        <span class="cs-state">{{ payload.sent ? 'The comma has these.' : 'Not on the comma yet: Publish puts them there.' }}<template
          v-if="payload.at"> Changed {{ isoWhen(payload.at) }}{{ payload.by ? ' by ' + payload.by : '' }}.</template></span>
        <button type="button" class="link-button" :disabled="props.readOnly || saving || allDefault" @click="resetAll">All to default</button>
        <button type="button" class="btn primary small" id="car-settings-save"
                :disabled="props.readOnly || saving || invalid || changed.length === 0" @click="saveChanged">Save</button>
      </div>
    </template>
    <p v-if="error" class="note">Not updated: {{ error }}</p>
  </section>
</template>
