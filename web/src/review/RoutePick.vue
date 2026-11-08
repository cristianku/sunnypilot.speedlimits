<script setup lang="ts">
// The drive under review, the menu to open another one, its count and who reviews (ReviewToolbar.vue), at the top
// of the inspector (above the country of the frame): its name on two lines ("Basel → Mumpf", then "Wed 21 Oct 2026 ·
// 13:55 · 38 min", review.route_links), so the time is never cut; the route's code only in the title; under them
// DASHCAM MODE on a dashcam drive (sunnypilot off in all of it, review.dashcam_of). The menu is the real select,
// invisible over those lines.
import {computed} from 'vue';
import ReviewToolbar from './ReviewToolbar.vue';
import {report} from './store';

/** The two lines of the drive on screen; offline, or for a drive nothing is known of, its code. */
const face = computed(() => {
  const link = report.routes.find((route) => route.route_id === report.routeId);
  const title = link?.title || link?.name || report.routeId || 'Local report';
  const when = link?.title ? [link.when, link.minutes ? link.minutes + ' min' : null].filter(Boolean).join(' · ') : '';
  return {title, when, full: Boolean(link?.full_definition), dashcam: link?.dashcam === true};
});

/** The routes of the server, this one chosen; offline only this report. */
const options = computed(() => {
  const list = report.routes.map((route) => ({
    value: String(route.url || ''), text: String(route.name || route.route_id || ''), selected: route.route_id === report.routeId,
  }));
  if (!list.some((option) => option.selected)) {
    list.unshift({value: '', text: report.routeId ? report.routeId : 'Local report', selected: true});
  }
  return list;
});

function changeRoute(event: Event): void {
  const value = (event.target as HTMLSelectElement).value;
  if (value.startsWith('/route/')) window.location.assign(value);
}
</script>

<template>
  <div class="route-pick" :title="report.routeId">
    <label for="route" class="route-pick-label">Drive</label>
    <div class="route-face">
      <span class="route-face-text" aria-hidden="true"><span class="route-title">{{ face.title }}</span><span
        v-if="face.when || face.full" class="route-when">{{ face.when }}<template v-if="face.full"><template
          v-if="face.when"> · </template><span title="Recorded at full definition, 2688 × 1520">HD</span></template></span><span
        v-if="face.dashcam" class="route-badges"><b class="dashcam-badge">DASHCAM MODE</b></span></span>
      <select id="route" :disabled="report.routes.length <= 1" @change="changeRoute">
        <option v-for="option in options" :key="option.value + option.text" :value="option.value"
                :selected="option.selected">{{ option.text }}</option>
      </select>
    </div>
    <ReviewToolbar/>
  </div>
</template>
