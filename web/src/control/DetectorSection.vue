<script setup lang="ts">
// The detector on the test: the reviewed signs it found and missed and the boxes it drew where there is no sign;
// recall by the size of the sign (the width of the reviewed box in the frame), on every frame or only on those a
// person confirmed; and route by route.
import {computed, ref} from 'vue';
import {thousands} from '../common/format';
import type {DetectorEvaluation} from '../common/panel-types';
import {signsHref} from '../common/sign-link';
import {pct, plain} from './evaluate';
import Segmented from './Segmented.vue';

const props = defineProps<{detector: DetectorEvaluation}>();
type Scope = 'all' | 'human';
const SCOPES: {key: Scope; label: string}[] = [{key: 'all', label: 'All frames'}, {key: 'human', label: 'Human-reviewed'}];
const scope = ref<Scope>('all');
const sizes = computed(() => scope.value === 'human' && props.detector.sizes_human ? props.detector.sizes_human : props.detector.sizes);
const routePad = (route: string): string => /^0+(?=[0-9a-f]+--)/.exec(route)?.[0] ?? '';
</script>

<template>
  <section class="card eval-card" id="detector-eval" aria-labelledby="detector-eval-title">
    <div class="ds-head">
      <h2 id="detector-eval-title">Detector<span class="eval-what">finds the signs in the frame</span></h2>
      <span class="ds-when">score ≥ {{ detector.score_threshold ?? '–' }} · overlap (IoU) ≥ {{ detector.iou_threshold ?? '–' }}</span>
    </div>
    <div class="ds-figures">
      <div class="ds-figure"><span class="ds-value" id="detector-recall">{{ pct(detector.recall) }}</span><span class="ds-unit">recall</span></div>
      <div class="ds-figure"><span class="ds-value" id="detector-precision">{{ pct(detector.precision) }}</span><span class="ds-unit">precision</span></div>
      <div class="ds-figure"><span class="ds-value">{{ pct(detector.f1) }}</span><span class="ds-unit">F1</span></div>
    </div>
    <ul class="box-counts" aria-label="Boxes on the test">
      <li class="found"><strong>{{ thousands(detector.found) }}</strong>found<span>of {{ thousands(detector.truth_boxes) }} reviewed signs</span></li>
      <li class="missed"><strong>{{ thousands(detector.missed) }}</strong>missed<span>signs it did not find</span></li>
      <li class="false"><strong>{{ thousands(detector.false_boxes) }}</strong>false boxes<span>{{ detector.false_per_image === null ? '' : plain(detector.false_per_image) + ' per frame, ' }}on {{
        thousands(detector.images ?? 0) }} frames</span></li>
    </ul>
    <p v-if="detector.human" class="note">On the {{ thousands(detector.human.images ?? 0) }} frames a person confirmed: recall {{ pct(detector.human.recall) }}
      ({{ thousands(detector.human.found) }} of {{ thousands(detector.human.truth_boxes) }}), precision {{ pct(detector.human.precision) }},
      {{ thousands(detector.human.false_boxes) }} false boxes.</p>

    <div class="ds-block">
      <h3 class="ds-title">Recall by sign size<span class="ds-when">width of the reviewed box in the frame</span></h3>
      <Segmented v-if="detector.sizes_human" v-model="scope" :options="SCOPES" label="Frames counted"/>
      <ul v-if="sizes" class="barlist sizes" id="detector-sizes" aria-label="Recall by sign size">
        <li v-for="row in sizes" :key="row.key" :data-size="row.key"
            :title="row.label + ': ' + row.found + ' found of ' + row.truth + ', ' + row.false_boxes + ' false boxes of this width'">
          <span class="bl-label">{{ row.label }}</span>
          <span class="bl-track" aria-hidden="true"><span :style="{width: Math.max((row.recall ?? 0) * 100, row.truth ? 0.6 : 0) + '%'}"></span></span>
          <span class="bl-value">{{ row.truth ? pct(row.recall) : '–' }}<small>{{ thousands(row.found) }}/{{ thousands(row.truth) }}</small></span>
          <span class="bl-extra">{{ row.false_boxes ? thousands(row.false_boxes) + ' false' : '' }}</span>
        </li>
      </ul>
      <p v-else class="empty">The report has no boxes: the recall by size cannot be computed.</p>
    </div>

    <div v-if="detector.routes.length" class="ds-block">
      <h3 class="ds-title">By route</h3>
      <div class="table-wrap"><table class="eval-table ds-routes">
        <thead><tr><th>Route</th><th class="num">Frames</th><th class="num">Signs</th><th class="num">Found</th><th class="num">Missed</th>
          <th class="num">False</th><th class="num">Recall</th><th class="num">Precision</th></tr></thead>
        <tbody>
          <tr v-for="row in detector.routes" :key="row.route">
            <td class="route"><a :href="signsHref(row.route)" :title="row.route"><span v-if="routePad(row.route)" class="rid-pad">{{
              routePad(row.route) }}</span>{{ row.route.slice(routePad(row.route).length) }}</a></td>
            <td class="num">{{ thousands(row.images ?? 0) }}</td><td class="num">{{ thousands(row.truth_boxes) }}</td>
            <td class="num">{{ thousands(row.found) }}</td><td class="num">{{ thousands(row.missed) }}</td><td class="num">{{ thousands(row.false_boxes) }}</td>
            <td class="num">{{ pct(row.recall) }}</td><td class="num">{{ pct(row.precision) }}</td>
          </tr>
        </tbody>
      </table></div>
    </div>
  </section>
</template>
