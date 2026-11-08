<script setup lang="ts">
// The inspector, docked on the right on a computer and under the photo on a phone: the drive, the boxes of the
// frame, the second reading, the editor of the selected one, "More", the confirm block. What the car's networks read
// and the AI reviewer are under the photo (StageBar.vue), so the editor comes right under the boxes.
import {computed, onBeforeUnmount, watch} from 'vue';
import {isGuest} from '../common/guest';
import {setReviewSwitch} from '../common/review-switch';
import BoxEditor from './BoxEditor.vue';
import BoxList from './BoxList.vue';
import AiLiveBanner from './AiLiveBanner.vue';
import BusyBanner from './BusyBanner.vue';
import {boxCountText} from './boxes';
import {applyPrevious, carryState} from './carry';
import ConfirmArea from './ConfirmArea.vue';
import RoutePick from './RoutePick.vue';
import {releaseFocus} from './dom';
import Icon from '../components/Icon.vue';
import {selectedBox} from './current';
import {reasonsOf, segmentSecond} from './frames';
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
// On the server: what the car did at this moment (its limit, the map's, the vision's), in the drive's player of Speed review.
const driveHref = computed(() => report.apiBase === null || isGuest() ? null : '/speed-review?drive=' + encodeURIComponent(report.routeId)
  + '&segment=' + props.row.segment_index + '&second=' + Math.round((segmentSecond(props.row) ?? 0) * 10) / 10);
// The same address in the top bar's switch, always at the same place (common/review-switch.ts).
watch(driveHref, (href) => setReviewSwitch(href === null ? null : 'signs', href), {immediate: true});
onBeforeUnmount(() => setReviewSwitch(null, null));
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
      <p v-if="driveHref" class="drive-link"><a id="driveReviewLink" :href="driveHref">What the car did here, in Speed review →</a></p>
      <p class="reasons" id="reasons" :hidden="reasons.length === 0"><template v-if="reasons.length > 0"><strong>Suspicious: </strong>{{ reasons.map((reason) => String(reason.text)).join('; ') }}<button v-if="carryHint" type="button" class="btn small carry-hint" @click="applyPrevious(); releaseFocus($event)">Apply previous <kbd class="keycap">A</kbd></button></template></p>
      <p class="filter-note" id="filterNote" :hidden="state.filterNote === ''">{{ state.filterNote }}</p>
      <BoxList :row="row" :work="work"/>
      <p class="pick-hint" id="pickHint" :hidden="count === 0 || state.selected >= 0">Select a box to correct its value, type and road.</p>
      <p class="empty-note" id="noBoxes" :hidden="count > 0"><svg viewBox="0 0 34 34" aria-hidden="true"><circle cx="17" cy="17" r="14" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="4 3.3"/><path d="M12 17h10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>No boxes. If there is no speed sign, confirm “no sign”: this example is needed too.</p>
    </section>
    <!-- Right under the list: what another model saw is compared with the boxes, before any of them is edited. -->
    <SecondReading :row="row" :work="work"/>
    <BoxEditor :row="row" :detection="selectedBox"/>
    <OtherFields :row="row" :work="work"/>
    <ConfirmArea :row="row" :work="work"/>
  </aside>
</template>
