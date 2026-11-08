<script setup lang="ts">
// Speed review ⇄ Sign review of the drive shown, at the same moment, in the top bar before the comma: the page shown
// lit, the other a link (common/review-switch.ts). Nothing on the other pages.
import {reviewSwitch} from '../common/review-switch';
</script>

<template>
  <nav v-if="reviewSwitch.current && reviewSwitch.other" class="review-switch" aria-label="Speed review or Sign review">
    <a v-if="reviewSwitch.current === 'signs'" class="rs-item" id="review-switch-speed" :href="reviewSwitch.other"
       title="Speed review of this drive at this moment">Speed</a>
    <span v-else class="rs-item rs-on" aria-current="page">Speed</span>
    <a v-if="reviewSwitch.current === 'speed'" class="rs-item" id="review-switch-signs" :href="reviewSwitch.other"
       title="Sign review of this drive at this moment">Signs</a>
    <span v-else class="rs-item rs-on" aria-current="page">Signs</span>
  </nav>
</template>
