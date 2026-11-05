<script setup lang="ts">
// "Car networks" under the bar of the photo, only while its button in the bar (key T) is on: the car's cutter and
// classifier on the frame on screen (car.ts), and on each frame shown after it until the button is pressed again.
// One row per box the car finds: what the classifier read, how sure, what the car does with it, and "Use" to make it
// a box of the frame. Nothing is saved. Hidden offline.
import {computed} from 'vue';
import {car, carAskOf, carAvailable, carDetection, carSignUsed, useCarSign, type CarSign} from './car';
import {describe, holdsPlate} from './catalog';
import {releaseFocus} from './dom';
import type {FrameEdit, FrameRow} from './types';

const props = defineProps<{row: FrameRow; work: FrameEdit}>();
const ask = computed(() => carAskOf(props.row.frame_key));
const reading = computed(() => ask.value?.reading ?? null);

/** What the car does with a box, as a pill. */
function useText(sign: CarSign, minimum: number): [string, string] {
  switch (sign.use) {
    case 'speed': return sign.speed === 0 ? ['Ends the limit', 'ok'] : ['Limit ' + Math.round(sign.speed ?? 0), 'ok'];
    case 'road': return ['Road type', 'rev'];
    case 'board': return ['Board, road type', 'rev'];
    case 'nothing': return ['No limit', 'muted'];
    default: return ['Dropped: under ' + Math.round(minimum * 100) + '%', 'warn'];
  }
}

const rows = computed(() => {
  const value = reading.value;
  if (value === null) return [];
  return value.signs.map((sign, i) => {
    const d = carDetection(sign);
    const [use, kind] = useText(sign, value.release.min_confidence);
    const sure = Math.round(sign.confidence * 100);
    return {
      index: i, what: describe({...d, value_kph: null}) + (d.value_kph === null ? '' : ' · ' + d.value_kph)
        + (holdsPlate(d) ? ', with hours' : ''),
      sure: sure + '%', sureKind: sign.confidence >= 0.9 ? 'high' : sign.confidence >= value.release.min_confidence ? 'medium' : 'low',
      use, kind, sub: 'box ' + Math.round(sign.score * 100) + '%' + (i === value.surest ? ' · the frame’s reading' : ''),
      used: carSignUsed(props.work, sign), sign,
    };
  });
});
const releaseText = computed(() => {
  const value = reading.value;
  if (value === null) return '';
  return value.release.name + ' · ' + value.ms + ' ms';
});
const releaseTitle = computed(() => {
  const r = reading.value?.release;
  if (!r) return '';
  return 'cutter ' + (r.cutter ?? '?') + '\nclassifier ' + (r.classifier ?? '?') + '\nmodels ' + (r.models_commit ?? '?')
    + ' · sunnypilot ' + (r.sunnypilot_commit ?? '?');
});
</script>

<template>
  <section v-if="carAvailable() && car.shown" class="car-section stage-extra" id="carSection" :data-car="ask?.phase ?? 'reading'"
           aria-labelledby="carTitle">
    <div class="section-head">
      <h2 id="carTitle">Car networks</h2>
      <span v-if="reading" class="count car-release" id="carRelease" :title="releaseTitle">{{ releaseText }}</span>
    </div>
    <p v-if="!ask || ask.phase === 'reading'" class="ai-note ai-wait" role="status"><span class="ai-spin car-spin" aria-hidden="true"></span>Reading the frame…</p>
    <p v-else-if="ask.phase === 'failed'" class="ai-note ai-error" id="carError" role="alert">{{ ask.error }}</p>
    <template v-else-if="reading">
      <div v-if="rows.length > 0" class="hint-list" id="carList">
        <div v-for="item in rows" :key="item.index" class="hint-row car-row" :data-use="item.sign.use">
          <span class="n">{{ item.index + 1 }}</span>
          <div class="hint-text">
            <div class="hint-head">
              <span class="hint-what">{{ item.what }}</span>
              <span :class="'conf ' + item.sureKind">{{ item.sure }}</span>
            </div>
            <div class="hint-head">
              <span :class="'pill ' + item.kind">{{ item.use }}</span>
              <span class="hint-sub">{{ item.sub }}</span>
            </div>
          </div>
          <button v-if="item.used" type="button" class="btn small tint" disabled>Already used</button>
          <button v-else type="button" class="btn small tint" :data-car-index="item.index"
                  :aria-label="'Use box ' + (item.index + 1) + ' of the car: ' + item.what"
                  @click="useCarSign(row, item.sign); releaseFocus($event)">Use</button>
        </div>
      </div>
      <p v-else class="ai-note" id="carNone">The car's networks find no sign here.</p>
      <p class="hint-stats">The car's code on the recorded video: close to the car, not bit for bit. One frame alone is not a limit.</p>
    </template>
  </section>
</template>
