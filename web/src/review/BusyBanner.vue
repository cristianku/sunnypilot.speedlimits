<script setup lang="ts">
// At the top of the inspector of a read-only drive (review/busy.ts): Propose again works on it and how far it is,
// the drive and the frames being proposed now; on the drive at work, back to the newest frame when another is open
// (on the newest the page follows by itself); a reload once the drive is done.
import {computed} from 'vue';
import {behind, busyBanner, showNewest} from './busy';
import {reloadSignEditor} from './embed';
import {report} from './store';

const banner = computed(() => busyBanner());
/** The page again; the editor on the Signs side of Speed review reads its drive again, on the frame open. */
function reload(): void {
  if (report.embedded) void reloadSignEditor();
  else window.location.reload();
}
</script>

<template>
  <div v-if="banner" class="busy-banner" id="busyBanner">
    <div class="busy-head">
      <strong>{{ banner.title }}</strong>
      <span v-if="banner.percent !== null" class="busy-pct">{{ banner.percent }}%</span>
    </div>
    <div v-if="banner.percent !== null" class="busy-bar" role="progressbar" aria-label="This drive"
         :aria-valuenow="banner.percent" aria-valuemin="0" aria-valuemax="100">
      <span :style="{width: banner.percent + '%'}"></span>
    </div>
    <p>{{ banner.text }}</p>
    <div v-if="banner.step" class="busy-step">
      <p>{{ banner.step.text }} <span class="nowrap">{{ banner.step.range }}</span></p>
      <div class="busy-step-row">
        <div class="busy-bar thin" :class="{over: banner.step.over}" role="progressbar" aria-label="These frames" :aria-valuenow="banner.step.percent"
             aria-valuemin="0" aria-valuemax="100"><span :style="{width: banner.step.percent + '%'}"></span></div>
        <span class="busy-step-pct">{{ banner.step.percent }}%</span>
      </div>
    </div>
    <p v-if="report.atWork && !banner.reload" class="busy-note">New frames appear here as they are proposed.
      Read only until all are done.</p>
    <button v-if="banner.reload" type="button" class="btn small" @click="reload">{{ banner.reload }}</button>
    <button v-else-if="behind()" type="button" class="busy-newest" @click="showNewest">
      Show the newest frame<span class="chev" aria-hidden="true">›</span>
    </button>
  </div>
</template>
