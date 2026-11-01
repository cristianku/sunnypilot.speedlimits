<script setup lang="ts">
// The bar above the photo: the frame's country and what the comma had at that moment (map and the limit ahead, the
// vision memory and how far it saw the sign, the car's limit and where it came from: car-moment.ts). Above the photo,
// never on it, so it hides nothing of the image.
import {computed} from 'vue';
import {carLogs, carMomentOf} from './car-moment';
import {commaReadText, countryOf, isReadFrame} from './frames';
import {report} from './store';
import type {FrameEdit, FrameRow} from './types';

const props = defineProps<{row: FrameRow; work: FrameEdit}>();

const moment = computed(() => carMomentOf(props.row));
// "The comma read 20 here": the frame added at a sign the comma's vision read.
const readText = computed(() => {
  const text = commaReadText(props.row);
  return text.charAt(0).toUpperCase() + text.slice(1);
});
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
</script>

<template>
  <div class="car-moment" id="carMoment" :class="{differ: moment?.differ}">
    <span class="country" id="country" :data-country="country ?? ''" :title="countryText" :aria-label="countryText">{{ country ? flagOf(country) : '?' }}</span>
    <span v-if="isReadFrame(row)" class="read-here" id="commaRead">{{ readText }}</span>
    <template v-if="moment">
      <span><b>Map</b> <span id="carMap">{{ moment.map }}</span><span v-if="moment.ahead" class="ahead"> → {{ moment.ahead }}</span></span>
      <span><b>Vision</b> <span id="carVision">{{ moment.vision }}</span><span v-if="moment.visionDistance" class="ahead" id="carVisionDist"> sign in {{ moment.visionDistance }}</span></span>
      <span><b>Car</b> <span id="carLimit">{{ moment.car }}</span><span v-if="moment.source" class="ahead"> from {{ moment.source }}</span></span>
    </template>
    <span v-else-if="carLogs.phase === 'none' || carLogs.phase === 'ready'" class="ahead">No car logs for this frame</span>
  </div>
</template>
