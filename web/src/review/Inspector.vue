<script setup lang="ts">
// The inspector, docked on the right on a computer and under the photo on a phone: the drive, the boxes of the
// frame, the second reading, the AI reviewer, the editor of the selected one, "More", the confirm block.
import {computed} from 'vue';
import AiSection from './AiSection.vue';
import BoxEditor from './BoxEditor.vue';
import BoxList from './BoxList.vue';
import AiLiveBanner from './AiLiveBanner.vue';
import BusyBanner from './BusyBanner.vue';
import {boxCountText} from './boxes';
import {applyPrevious, carryState} from './carry';
import {carMomentOf} from './car-moment';
import ConfirmArea from './ConfirmArea.vue';
import RoutePick from './RoutePick.vue';
import {releaseFocus} from './dom';
import Icon from '../components/Icon.vue';
import {selectedBox} from './current';
import {countryOf, reasonsOf, segmentSecond} from './frames';
import OtherFields from './OtherFields.vue';
import SecondReading from './SecondReading.vue';
import {report, state} from './store';
import type {FrameEdit, FrameRow} from './types';

const props = defineProps<{row: FrameRow; work: FrameEdit}>();
const reasons = computed(() => reasonsOf(props.row.frame_key));
const count = computed(() => props.work.detections.length);
const carry = computed(() => carryState(props.row));
// The previous frame has a confirmed sign and this one nothing: carrying it over is the likely next move.
const carryHint = computed(() => carry.value.enabled
  && reasons.value.some((reason) => reason.rule === 'd' && reason.text.startsWith('the previous frame')));
const country = computed(() => countryOf(props.row, props.work));
/** The flag of a country code: its two letters as regional indicator symbols ("CH" → 🇨🇭). */
function flagOf(code: string): string {
  return [...code.toUpperCase()].map((letter) => String.fromCodePoint(0x1f1a5 + letter.charCodeAt(0))).join('');
}
// From the GPS, shown only: the reviewer does not choose it (REST and MCP can still send another one).
const countryText = computed(() => {
  const code = country.value;
  if (!code) return props.row.gps === undefined ? 'Country not known yet: GPS not read for this route' : 'Country not known: no GPS position';
  const source = props.row.country_from === 'nearest_fix' ? 'from the GPS of a nearby frame' : 'from the GPS';
  return (report.countries[code] ?? code) + ' · ' + source;
});
// On the server: what the car did at this moment (its limit, the map's, the vision's), in Drive review.
const driveHref = computed(() => report.apiBase === null ? null : '/drive-review?drive=' + encodeURIComponent(report.routeId)
  + '&segment=' + props.row.segment_index + '&second=' + Math.round((segmentSecond(props.row) ?? 0) * 10) / 10);
// What the comma had at this frame, from its logs: the map's limit, the vision memory's, the one the car used.
const moment = computed(() => carMomentOf(props.row));
</script>

<template>
  <aside class="panel" :class="{editing: selectedBox !== null}" aria-label="Review of the frame">
    <section>
      <RoutePick/>
      <BusyBanner/>
      <AiLiveBanner/>
      <div class="section-head"><h2>Signs</h2><span class="count" id="boxCount">{{ boxCountText(count) }}</span>
        <!-- Off, it still answers a press: the reason why appears on the status line. -->
        <button v-if="carry.shown" type="button" class="btn small carry" id="applyPrevious" :aria-disabled="!carry.enabled"
                :title="carry.title" @click="applyPrevious(); releaseFocus($event)"><Icon name="previous"/>Apply previous</button></div>
      <p class="country-line" id="country" :data-country="country ?? ''"><span v-if="country" class="flag" aria-hidden="true">{{ flagOf(country) }}</span>{{ countryText }}</p>
      <dl v-if="moment" class="car-moment" id="carMoment" :class="{differ: moment.differ}">
        <div><dt>Map</dt><dd id="carMap">{{ moment.map }}<span v-if="moment.ahead" class="ahead"> · {{ moment.ahead }}</span></dd></div>
        <div><dt>Vision memory</dt><dd id="carVision">{{ moment.vision }}</dd></div>
        <div><dt>Car</dt><dd id="carLimit">{{ moment.car }}</dd></div>
        <p v-if="moment.differ" class="differ-note">Vision ≠ map here</p>
      </dl>
      <p v-if="driveHref" class="drive-link"><a id="driveReviewLink" :href="driveHref">What the car did here, in Drive review →</a></p>
      <p class="reasons" id="reasons" :hidden="reasons.length === 0"><template v-if="reasons.length > 0"><strong>Suspicious: </strong>{{ reasons.map((reason) => String(reason.text)).join('; ') }}<button v-if="carryHint" type="button" class="btn small carry-hint" @click="applyPrevious(); releaseFocus($event)">Apply previous <kbd class="keycap">A</kbd></button></template></p>
      <p class="filter-note" id="filterNote" :hidden="state.filterNote === ''">{{ state.filterNote }}</p>
      <BoxList :row="row" :work="work"/>
      <p class="pick-hint" id="pickHint" :hidden="count === 0 || state.selected >= 0">Select a box to correct its value, type and road.</p>
      <p class="empty-note" id="noBoxes" :hidden="count > 0"><svg viewBox="0 0 34 34" aria-hidden="true"><circle cx="17" cy="17" r="14" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="4 3.3"/><path d="M12 17h10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg><span>No boxes. If there is no speed sign, confirm “no sign”: this example is needed too.</span></p>
    </section>
    <!-- Right under the list: what another model saw is compared with the boxes, before any of them is edited. -->
    <SecondReading :row="row" :work="work"/>
    <AiSection :row="row"/>
    <BoxEditor :row="row" :detection="selectedBox"/>
    <OtherFields :row="row" :work="work"/>
    <ConfirmArea :row="row" :work="work"/>
  </aside>
</template>
