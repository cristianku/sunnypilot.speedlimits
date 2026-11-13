<script setup lang="ts">
// The inspector, docked on the right on a computer and under the photo on a phone: the drive, the boxes of the
// frame, the second reading, the editor of the selected one, "More", the confirm block. What the car's networks read
// and the AI reviewer are under the photo (StageBar.vue), so the editor comes right under the boxes.
import {computed} from 'vue';
import {isGuest} from '../common/guest';
import BoxEditor from './BoxEditor.vue';
import BoxList from './BoxList.vue';
import AiLiveBanner from './AiLiveBanner.vue';
import BusyBanner from './BusyBanner.vue';
import {boxCountText} from './boxes';
import {applyPrevious, carried, carryState} from './carry';
import ConfirmArea from './ConfirmArea.vue';
import RoutePick from './RoutePick.vue';
import ReviewToolbar from './ReviewToolbar.vue';
import {releaseFocus} from './dom';
import Icon from '../components/Icon.vue';
import {selectedBox} from './current';
import {toggleUsersComplete} from './edits';
import {frameStatus, reasonsOf, segmentSecond} from './frames';
import OtherFields from './OtherFields.vue';
import SecondReading from './SecondReading.vue';
import {report, state} from './store';
import type {FrameEdit, FrameRow} from './types';

const props = defineProps<{row: FrameRow; work: FrameEdit}>();
const reasons = computed(() => reasonsOf(props.row.frame_key));
const count = computed(() => props.work.detections.length);
const carry = computed(() => carryState(props.row));
// The frame's tick for UserNet (the road-user detector): its tooltip is the whole rule.
const USERS_COMPLETE_TITLE = 'Road users complete (J): every person on foot and two-wheeler from 24 px tall and every '
  + 'vehicle from 40 px tall has its box on this frame, in pixels of the full image, whether it matters for the speed '
  + 'or not. Saved with the boxes; Apply previous never carries it.';
// The previous frame has a confirmed sign and this one nothing: carrying it over is the likely next move.
const carryHint = computed(() => carry.value.enabled
  && reasons.value.some((reason) => reason.rule === 'd' && reason.text.startsWith('the previous frame')));
// Who reviewed the frame, at the top with a coloured square: a person (green), an AI (violet: Review with AI, REST,
// MCP), or nobody yet (grey). The confirmed copy on the server, else this browser's confirmation.
const reviewedBy = computed((): {kind: 'human' | 'ai' | 'none'; text: string; title: string} => {
  const base = props.work.base;
  const who = base?.confirmed ? base : frameStatus(props.row.frame_key) === 'done' ? props.work : null;
  if (!who || !who.reviewer) return {kind: 'none', text: 'Not reviewed yet', title: 'Nobody has confirmed this frame yet'};
  const ai = who.reviewer_kind === 'ai';
  const name = who.reviewer.replace(/^ai\//, '');
  const via = who.channel === 'mcp' ? ' · MCP' : who.channel === 'rest' ? ' · REST' : '';
  return {kind: ai ? 'ai' : 'human', text: (ai ? 'AI · ' : '') + name + via,
    title: (ai ? 'Reviewed by an AI: ' : 'Reviewed by a person: ') + who.reviewer
      + (who.review_timestamp ? ', ' + who.review_timestamp.replace('T', ' ').slice(0, 16) : '')};
});
// On the server: what the car did at this moment (its limit, the map's, the vision's), in the drive's player of Speed review.
const driveHref = computed(() => report.apiBase === null || isGuest() || report.embedded ? null : '/speed-review?drive=' + encodeURIComponent(report.routeId)
  + '&segment=' + props.row.segment_index + '&second=' + Math.round((segmentSecond(props.row) ?? 0) * 10) / 10);
</script>

<template>
  <aside class="panel" :class="{editing: selectedBox !== null}" aria-label="Review of the frame">
    <section>
      <RoutePick v-if="!report.embedded"/>
      <ReviewToolbar v-else/>
      <BusyBanner/>
      <AiLiveBanner/>
      <p class="reviewed-by" :class="'rb-' + reviewedBy.kind" id="reviewedBy" :title="reviewedBy.title"><span class="rb-square"
        aria-hidden="true"></span><template v-if="reviewedBy.kind !== 'none'">Reviewed by </template><b>{{ reviewedBy.text }}</b></p>
      <div class="section-head"><h2>Signs</h2><span class="count" id="boxCount">{{ boxCountText(count) }}</span>
        <!-- Off, it still answers a press: the reason why appears on the status line. Once it has carried the boxes,
             a second press saves and goes on, as Enter. -->
        <button v-if="carry.shown" type="button" class="btn small carry" id="applyPrevious" :aria-disabled="!carry.enabled"
                :title="carry.title" @click="applyPrevious(); releaseFocus($event)"><Icon name="previous"/>{{ carried === row.frame_key ? 'Save and next' : 'Apply previous' }}</button></div>
      <p v-if="driveHref" class="drive-link"><a id="driveReviewLink" :href="driveHref">What the car did here, in Speed review →</a></p>
      <p class="reasons" id="reasons" :hidden="reasons.length === 0"><template v-if="reasons.length > 0"><strong>Suspicious: </strong>{{ reasons.map((reason) => String(reason.text)).join('; ') }}<button v-if="carryHint" type="button" class="btn small carry-hint" @click="applyPrevious(); releaseFocus($event)">Apply previous <kbd class="keycap">A</kbd></button></template></p>
      <p class="filter-note" id="filterNote" :hidden="state.filterNote === ''">{{ state.filterNote }}</p>
      <BoxList :row="row" :work="work"/>
      <button type="button" class="chip users-complete" id="usersComplete" :aria-pressed="work.users_complete === true"
              :disabled="report.readOnly" :title="USERS_COMPLETE_TITLE"
              @click="toggleUsersComplete(); releaseFocus($event)">{{ work.users_complete === true ? '✓ ' : '' }}Road users complete<kbd class="key-hint">J</kbd></button>
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
