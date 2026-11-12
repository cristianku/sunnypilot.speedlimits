<script setup lang="ts">
// The editor of the selected box: where it comes from and Fit box (its zoom is inside the photo, ZoomInset.vue), the
// labels used last, road, speed signs, other value, type and kinds of other signs, details.
import {computed, ref, watch} from 'vue';
import {BOOLEAN_FIELDS, FAMILIES, PANELS, PLATE_HOURS, QUICK_VALUES, ROADS, SELECT_FIELDS, SIGN_KIND_GROUPS, US_VALUES,
  familyHasValue, signStyleOf, type BooleanField, type SelectField} from './catalog';
import {aiBlocker, canFitLocally, fitBox, fitLocal, fitting, localFitting} from './ai';
import {addedBy, originalOf} from './boxes';
import {releaseFocus} from './dom';
import {applyLabel, setFamily, setField, setKind, setOnVehicle, setOtherOurs, setValue, squareSelected, togglePainted, toggleVariable} from './edits';
import {countryOf, workOf} from './frames';
import {vFill} from './fill';
import {labelKey, labelOf, recentLabels} from './recent';
import {report, state} from './store';
import type {Detection, FrameRow} from './types';

const props = defineProps<{row: FrameRow; detection: Detection | null}>();

// The labels given last on this drive, one tap each (recent.ts); pressed, the one the box has.
const recent = computed(() => (props.detection === null ? [] : recentLabels(props.row, state.selected)));
const labelNow = computed(() => (props.detection === null ? '' : labelKey(labelOf(props.detection))));

const usual = FAMILIES.filter((family) => !family.minor);
const minor = FAMILIES.filter((family) => family.minor);
const hasValueField = computed(() => props.detection !== null && familyHasValue(props.detection.family));
const noValue = computed(() => hasValueField.value
  && (props.detection?.value_kph === null || props.detection?.value_kph === undefined));
const road = computed(() => props.detection?.applicability || 'unresolved');
const panel = computed(() => props.detection?.supplementary_panel ?? 'unknown');
const variable = computed(() => props.detection?.family === 'variable_display');
// In the US the signs are the white SPEED LIMIT rectangles, in mph and every 5; elsewhere the red discs of km/h.
const usSigns = computed(() => signStyleOf(countryOf(props.row, workOf(props.row))) === 'us');
const quickValues = computed(() => usSigns.value ? US_VALUES : QUICK_VALUES);
// From and to of a plate of hours. The box takes them when both are chosen: until then they wait here.
const asksHours = computed(() => panel.value === 'time' && props.detection?.family === 'maximum_speed');
const halfHours = ref<[string, string]>(['', '']);
watch(() => [props.row.frame_key, state.selected], () => { halfHours.value = ['', '']; });
const hours = computed<[string, string]>(() => {
  const [from = '', to = ''] = props.detection?.plate_hours?.split('-') ?? [];
  return props.detection?.plate_hours ? [from, to] : halfHours.value;
});
// O X U, G P R E M and the road users W Y K on a keyboard (use-keyboard.ts).
const ROAD_KEYS: Record<string, string> = {own_road: 'O', other_road: 'X', unresolved: 'U'};
const KIND_KEYS: Record<string, string> = {give_way: 'G', stop: 'P', roundabout: 'R', other: 'E', main_road: 'M',
  road_user_person: 'W', road_user_bicycle: 'Y', road_user_car: 'K'};
const canFit = computed(() => props.detection !== null && aiBlocker() === '');
const fittingThis = computed(() => fitting.key === props.row.frame_key && fitting.index === state.selected);
const canFitHere = computed(() => props.detection !== null && canFitLocally());
const fittingHere = computed(() => localFitting.key === props.row.frame_key && localFitting.index === state.selected);
// Where the box comes from: the model's proposal, else who added it (the reviewer by hand, or an AI reviewer).
const evidence = computed(() => {
  const d = props.detection;
  if (d === null) return '';
  const original = originalOf(props.row, d);
  const label = d.model_label || original?.model_label;
  if (label) return 'Model proposal: ' + label;
  if (original !== null) return 'Model proposal.';
  return 'Box ' + addedBy(workOf(props.row), d) + '.';
});

const SELECT_CHOICES: Record<SelectField, [string, string][]> = {
  visibility: [['unknown', 'To assess'], ['clear', 'Clear'], ['partial', 'Partial'], ['poor', 'Poor']],
  occlusion: [['unknown', 'To assess'], ['none', 'None'], ['partial', 'Partial'], ['heavy', 'Heavy']],
  supplementary_panel: [['unknown', 'To assess'], ['none', 'None'], ['time', 'Hours'], ['wet', 'Wet'], ['present', 'Other'],
    ['unresolved', 'Unresolved']],
};
const SELECT_LABELS: Record<SelectField, string> = {
  visibility: 'Visibility', occlusion: 'Occlusion', supplementary_panel: 'Panel below',
};
const BOOLEAN_LABELS: Record<BooleanField, string> = {temporary: 'Temporary', variable_display: 'Variable speed limit'};

function selectValue(key: SelectField): string {
  return props.detection?.[key] ?? 'unknown';
}
function booleanValue(key: BooleanField): string {
  const value = props.detection?.[key];
  return value === true ? 'true' : value === false ? 'false' : 'unknown';
}

function valueChanged(event: Event): void {
  const raw = (event.target as HTMLInputElement).value.trim();
  if (raw === '') {
    setValue(null);
    return;
  }
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 5 || value > 250) {
    state.valueError = true;
    return;
  }
  setValue(value);
}
function selectChanged(key: SelectField, event: Event): void {
  setField(key, (event.target as HTMLSelectElement).value);
}
function hoursChanged(side: 0 | 1, event: Event): void {
  const chosen: [string, string] = [...hours.value];
  chosen[side] = (event.target as HTMLSelectElement).value;
  halfHours.value = chosen;
  const [from, to] = chosen;
  const value = from !== '' && to !== '' && from !== to ? from + '-' + to : null;
  if ((props.detection?.plate_hours ?? null) !== value) setField('plate_hours', value);
  releaseFocus(event);
}
function booleanChanged(key: BooleanField, event: Event): void {
  const value = (event.target as HTMLSelectElement).value;
  setField(key, value === 'true' ? true : value === 'false' ? false : null);
}
</script>

<template>
  <section id="editor" :hidden="detection === null">
    <div class="section-head"><h2 id="editorTitle">Box {{ state.selected + 1 }}</h2></div>
    <div class="zoom-foot">
      <p class="evidence" id="evidence">{{ evidence }}</p>
      <div class="fit-chips">
        <!-- Square on its centre (Q); Shift while drawing or dragging a corner draws a square. -->
        <button type="button" class="chip fit" id="squareBox" :disabled="report.readOnly"
                title="Make this box a square on its centre; hold Shift while drawing or dragging a corner for a square"
                @click="squareSelected(); releaseFocus($event)">Square<kbd>Q</kbd></button>
        <!-- This rough box made tight on the server, in a fraction of a second; C fits every box of the frame. -->
        <button v-if="canFitHere" type="button" class="chip fit" id="fitLocal" :disabled="localFitting.key !== null"
                title="This box; C fits every box of the frame"
                @click="fitLocal('box'); releaseFocus($event)">{{ fittingHere ? 'Fitting…' : 'Fit box' }}</button>
        <!-- The same by the AI, shown only the picture around it (Shift+C): slower, for the boxes Fit box misses. -->
        <button v-if="canFit" type="button" class="chip fit ai" id="fitBox" :disabled="fitting.key !== null"
                @click="fitBox(true); releaseFocus($event)">{{ fittingThis ? 'Fitting…' : 'Fit with AI' }}<kbd>⇧C</kbd></button>
      </div>
    </div>
    <!-- The labels given last on this drive: one tap puts the whole label on this box (its place stays). -->
    <div v-if="recent.length" class="recent-row" id="recentLabels" role="group" aria-label="Labels used last on this drive">
      <span class="recent-label">Recent</span>
      <button v-for="item in recent" :key="item.key" type="button" class="chip recent" :aria-pressed="item.key === labelNow"
              :title="item.title" @click="applyLabel(item.label); releaseFocus($event)"><span v-if="item.disc !== null"
              class="mini-sign">{{ item.disc }}</span>{{ item.text }}</button>
    </div>
    <!-- Asked for nearly every box: right under the box's heading. -->
    <p class="field-label">Road</p>
    <div class="segmented" id="roads" role="group" aria-label="Which road it applies to">
      <button v-for="item in ROADS" :key="item.id" type="button" :aria-pressed="road === item.id"
              :data-applicability="item.id" @click="setField('applicability', item.id); releaseFocus($event)">{{ item.label }}<kbd class="key-hint">{{ ROAD_KEYS[item.id] }}</kbd></button>
    </div>
    <!-- Most boxes: an other sign for our road, in one tap (S); a truck's sticker, not a sign (V). -->
    <div class="quick-row">
      <!-- An action, never lit: Road and Type above and below say what the box is. -->
      <button type="button" class="chip quick" id="otherOurs" @click="setOtherOurs(); releaseFocus($event)">Other sign · our road<kbd>S</kbd></button>
      <button type="button" class="chip quick" id="onVehicle" title="A sign-like disc or panel on a vehicle, such as a truck's top-speed sticker: not a sign"
              @click="setOnVehicle(); releaseFocus($event)">On a vehicle<kbd>V</kbd></button>
    </div>
    <!-- The signs are always there: a number makes the box a speed limit, "?" means the number cannot be read. -->
    <div :class="['signs', {led: variable, us: usSigns}]" id="quickValues" role="group" :aria-label="usSigns ? 'Speed in mph' : 'Speed'">
      <template v-if="usSigns">
        <button v-for="value in quickValues" :key="value" type="button" class="sign us"
                :aria-pressed="detection?.value_kph === value" :data-value="value" :aria-label="value + ' mph written on sign'"
                @click="setValue(value); releaseFocus($event)"><span class="us-top" aria-hidden="true">SPEED<br>LIMIT</span><span class="us-num">{{ value }}</span></button>
      </template>
      <template v-else>
        <button v-for="value in quickValues" :key="value" type="button" :class="value >= 100 ? 'sign wide' : 'sign'"
                :aria-pressed="detection?.value_kph === value" :data-value="value" :aria-label="value + ' written on sign'"
                @click="setValue(value); releaseFocus($event)">{{ value }}</button>
      </template>
      <button type="button" :class="usSigns ? 'sign us unknown' : 'sign unknown'" :aria-pressed="noValue" data-value="unknown"
              title="Number not readable" aria-label="Number not readable" @click="setValue(null); releaseFocus($event)"><template
              v-if="usSigns"><span class="us-top" aria-hidden="true">SPEED<br>LIMIT</span><span class="us-num">?</span></template><template
              v-else>?</template></button>
    </div>
    <!-- The lit display over the road (L): lit, the numbers above look like it; the number stays either way. -->
    <div class="quick-row">
      <button type="button" class="chip quick variable" id="variable" :aria-pressed="variable"
              @click="toggleVariable(); releaseFocus($event)">Variable speed limit<kbd>L</kbd></button>
    </div>
    <!-- The panel under the sign: with None and Other the box stays on the disc, with Hours it takes disc and plate
         (a sign of its own for the classifier). A second tap takes the choice back. -->
    <p class="field-label">Panel below</p>
    <div class="segmented" id="panels" role="group" aria-label="Panel below the sign">
      <button v-for="item in PANELS" :key="item.id" type="button" :aria-pressed="panel === item.id" :data-panel="item.id"
              @click="setField('supplementary_panel', panel === item.id ? 'unknown' : item.id); releaseFocus($event)">{{ item.label }}</button>
    </div>
    <p v-if="panel === 'time' || panel === 'wet'" class="effect-note" id="plateNote">The box takes the disc and the plate.</p>
    <!-- The hours written on the plate; empty when they cannot be read, or the plate has days or two ranges. -->
    <div v-if="asksHours" id="plateHours" class="value-line plate-hours">
      <label for="hoursFrom">From</label>
      <select id="hoursFrom" aria-label="Plate valid from" :value="hours[0]" @change="hoursChanged(0, $event)">
        <option value="">–</option>
        <option v-for="hour in PLATE_HOURS" :key="hour" :value="String(hour)">{{ hour }}</option>
      </select>
      <label for="hoursTo">to</label>
      <select id="hoursTo" aria-label="Plate valid to" :value="hours[1]" @change="hoursChanged(1, $event)">
        <option value="">–</option>
        <option v-for="hour in PLATE_HOURS" :key="hour" :value="String(hour)">{{ hour === 0 ? '0 / 24' : hour }}</option>
      </select>
    </div>
    <div id="valueBlock" class="value-line" :hidden="!hasValueField">
      <label for="value">Other value</label>
      <input v-fill="detection?.value_kph === null || detection?.value_kph === undefined ? '' : String(detection.value_kph)"
             type="number" id="value" min="5" max="250" step="1" inputmode="numeric" placeholder="–" @change="valueChanged">
      <select aria-label="Sign unit" :value="detection?.sign_unit ?? 'unknown'"
              @change="setField('sign_unit', ($event.target as HTMLSelectElement).value as 'km/h' | 'mph' | 'unknown')">
        <option value="unknown">Unit not recorded</option>
        <option value="km/h">km/h</option>
        <option value="mph">mph</option>
      </select>
      <span v-if="state.typing" class="typing" role="status">typing {{ state.typing }}…</span>
    </div>
    <p class="field-error" id="valueError" :hidden="!state.valueError">Type a whole number from 5 to 250, or leave it empty if it cannot be read.</p>
    <p class="field-label">Type</p>
    <div class="chips" id="families" role="group" aria-label="Sign type">
      <button v-for="family in usual" :key="family.id" type="button" class="chip"
              :aria-pressed="detection?.family === family.id" :data-family="family.id"
              @click="setFamily(family.id); releaseFocus($event)">{{ family.label }}</button>
    </div>
    <p class="effect-note">✕ in the list removes a box (a duplicate, a box on nothing); <strong>Not a sign</strong> keeps it, to teach the model what to ignore.</p>
    <div class="chips minor" id="familiesMore" role="group" aria-label="Other types">
      <button v-for="family in minor" :key="family.id" type="button" class="chip"
              :aria-pressed="detection?.family === family.id" :data-family="family.id"
              :title="family.id === 'not_a_sign' ? 'Stays as a negative example; to remove a wrong box use ✕' : undefined"
              @click="setFamily(family.id); releaseFocus($event)">{{ family.label }}</button>
    </div>
    <!-- Other signs by kind: a button makes the box an "Other sign" of that kind. -->
    <div class="kinds" id="signKinds">
      <div v-for="group in SIGN_KIND_GROUPS" :key="group.label" class="kind-group" role="group" :aria-label="group.label">
        <span class="kind-group-label">{{ group.label }}</span>
        <button v-for="kind in group.kinds" :key="kind.id" type="button" class="chip kind"
                :aria-pressed="detection?.family === 'other_sign' && detection?.sign_type === kind.id"
                :data-kind="kind.id" :title="kind.label + (KIND_KEYS[kind.id] ? ' (' + KIND_KEYS[kind.id] + ')' : '')"
                @click="setKind(kind.id); releaseFocus($event)">{{ kind.chip }}<kbd v-if="KIND_KEYS[kind.id]" class="key-hint">{{ KIND_KEYS[kind.id] }}</kbd></button>
      </div>
      <!-- On or off, with any kind: the children's triangle with "SCHULE" is Children, painted. -->
      <div class="kind-group" role="group" aria-label="Where">
        <span class="kind-group-label">Where</span>
        <button type="button" class="chip kind" id="painted"
                :aria-pressed="detection?.family === 'other_sign' && detection?.painted === true"
                @click="togglePainted(); releaseFocus($event)">Painted on the road</button>
      </div>
    </div>
    <details>
      <summary>Sign details</summary>
      <div class="grid2">
        <label v-for="key in SELECT_FIELDS" :key="key">{{ SELECT_LABELS[key] }}
          <select :id="key" v-fill="selectValue(key)" @change="selectChanged(key, $event)">
            <option v-for="[value, text] in SELECT_CHOICES[key]" :key="value" :value="value">{{ text }}</option>
          </select>
        </label>
        <label v-for="key in BOOLEAN_FIELDS" :key="key">{{ BOOLEAN_LABELS[key] }}
          <select :id="key" v-fill="booleanValue(key)" @change="booleanChanged(key, $event)">
            <option value="unknown">Not determined</option>
            <option value="true">Yes</option>
            <option value="false">No</option>
          </select>
        </label>
        <label>Notes
          <input v-fill="detection?.notes ?? ''" type="text" id="notes" autocomplete="off"
                 @change="setField('notes', ($event.target as HTMLInputElement).value)">
        </label>
      </div>
    </details>
  </section>
</template>
