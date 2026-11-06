<script setup lang="ts">
// The site of a reviewer from outside (nnslr_tools.review_access): "My reviews", the drives given to them with a link for
// each kind of work, and the pages of Speed review, Sign distance and Roads for one of those drives; Sign review is its
// own page (/route/<id>/). No status of the server here, no tab of the administrator: every address without a drive shows
// the list, and the server answers only for the work given.
import {computed, onMounted, ref, shallowRef} from 'vue';
import {api, messageOf, post} from '../common/api';
import {plural} from '../common/format';
import {askConfirm} from '../composables/confirm';
import {toast} from '../composables/toast';
import type {GivenDrive, MyWork} from './reviewers';
import RoadDrive from './RoadDrive.vue';
import SignDistanceDrive from './SignDistanceDrive.vue';
import SpeedReviewDrive from './SpeedReviewDrive.vue';
import {currentTab, tabDrive, useTabRouting} from './tabs';

useTabRouting();
const mine = shallowRef<MyWork | null>(null);
const error = ref('');
onMounted(async () => {
  try {
    mine.value = await api<MyWork>('/api/reviewer/work');
  } catch (problem) {
    error.value = messageOf(problem);
  }
});
/** Until the list is in nothing can be saved: a drive opened from a link waits for it. */
const readOnly = computed(() => mine.value?.read_only ?? true);
const minutesOf = (minutes: number | null): string => minutes === null ? '' : plural(minutes, 'minute');
async function complete(drive: GivenDrive): Promise<void> {
  const ok = await askConfirm({title: 'Review completed?', action: 'Completed',
    text: 'Your work on this drive is collected and the drive leaves your list within a few minutes.'});
  if (!ok) return;
  try {
    mine.value = await post<MyWork>('/api/reviewer/done', {route: drive.route});
  } catch (problem) {
    toast(messageOf(problem), true);
  }
}
</script>

<template>
  <main class="page" :data-tab="currentTab">
    <SpeedReviewDrive v-if="currentTab === 'speedreview' && tabDrive" :key="tabDrive" :route="tabDrive" :read-only="readOnly"/>
    <SignDistanceDrive v-else-if="currentTab === 'signdistance' && tabDrive" :route="tabDrive" :read-only="readOnly"/>
    <RoadDrive v-else-if="currentTab === 'roads' && tabDrive" :route="tabDrive" :read-only="readOnly"/>
    <template v-else>
      <header class="section-head"><h1 class="head-title">My reviews</h1></header>
      <section class="card gr-work" aria-label="The drives given to you">
        <p v-if="error" class="empty">{{ error }}</p>
        <p v-else-if="!mine" class="empty">Loading…</p>
        <p v-else-if="!mine.drives.length" class="empty" id="gr-none">Nothing is given to you yet. When a drive is assigned to you it
          appears here.</p>
        <ul v-else class="gr-list" id="gr-drives">
          <li v-for="drive in mine.drives" :key="drive.route" class="gr-drive">
            <div class="gr-name">
              <strong>{{ drive.when ?? 'Drive' }}</strong>
              <span v-if="drive.minutes !== null">{{ minutesOf(drive.minutes) }}</span>
            </div>
            <div class="gr-links">
              <a v-for="item in drive.work" :key="item.kind" class="btn small" :href="item.href">{{ item.label }}</a>
              <span v-if="drive.completed" class="gr-done">Completed</span>
              <button v-else-if="mine.can_complete" type="button" class="btn small primary" @click="complete(drive)">Review
                completed</button>
            </div>
          </li>
        </ul>
      </section>
    </template>
  </main>
</template>
