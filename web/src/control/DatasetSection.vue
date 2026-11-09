<script setup lang="ts">
// What the gold dataset holds (status.dataset): frames, objects and routes; the objects by sign family; who
// confirmed them (people or an AI reviewer); the split into train, validation, test and the buffer around the
// test (by place), as one stacked bar; the last import; and route by route.
import {computed} from 'vue';
import {isoWhen, plural, thousands} from '../common/format';
import type {DatasetSummary} from '../common/panel-types';
import {signsHref} from '../common/sign-link';

const props = defineProps<{dataset: DatasetSummary | null; loading: boolean}>();

// The split in the order the training uses it, each a slot of the reference categorical palette (control.html
// --split-1..5, validated for both themes); train and validation are the blue and orange of the charts. The
// test is made of places, and the buffer (the places next to it) is neither trained on nor measured. Partitions
// of older split files (held-out routes, hard negatives) are added after these when a split has them.
const SPLITS: [key: string, label: string][] = [['train', 'Train'], ['validation', 'Validation'],
  ['test', 'Test (places kept apart)'], ['buffer', 'Buffer (next to the test)']];

const percentOf = (part: number, whole: number): number => whole ? part * 100 / whole : 0;
const percentText = (part: number, whole: number): string => {
  const value = percentOf(part, whole);
  return value > 0 && value < 1 ? '<1%' : Math.round(value) + '%';
};

const families = computed(() => {
  const value = props.dataset;
  if (value === null) return [];
  const top = Math.max(1, ...value.families.map((family) => family.count));
  return value.families.map((family) => ({...family, width: family.count * 100 / top, share: percentText(family.count, value.objects)}));
});
const reviewers = computed(() => {
  const value = props.dataset;
  if (value === null) return [];
  const top = Math.max(1, value.human, value.ai);
  return [{key: 'human', label: 'People', count: value.human}, {key: 'ai', label: 'AI reviewers', count: value.ai}]
    .map((row) => ({...row, width: row.count * 100 / top, share: percentText(row.count, value.human + value.ai)}));
});
const split = computed(() => {
  const value = props.dataset?.split ?? null;
  if (value === null) return null;
  const known = new Set(SPLITS.map(([key]) => key));
  const parts = [...SPLITS, ...Object.keys(value.counts).filter((key) => !known.has(key)).map((key): [string, string] => [key, key])]
    .map(([key, label], index) => ({key, label, count: value.counts[key] ?? 0, slot: index < SPLITS.length ? index + 1 : 0}));
  const total = parts.reduce((sum, part) => sum + part.count, 0);
  return {parts: parts.map((part) => ({...part, share: percentText(part.count, total), width: percentOf(part.count, total)})),
    total, made: isoWhen(value.made_at), warnings: value.warnings};
});
/** Why the split counts fewer objects than the dataset: the dataset changed after it (the sha256 the split recorded is
 * not the dataset's; by the counts for an older split without one), or the proposals a person rejected, left out. */
const splitNote = computed((): string => {
  const value = props.dataset;
  if (value === null || value.split === null || split.value === null) return '';
  const rejected = value.split.rejected ?? 0;
  const recorded = value.split.dataset_sha256 ?? null;
  const changed = recorded !== null && value.sha256 ? recorded !== value.sha256 : split.value.total + rejected !== value.objects;
  if (changed) return 'The dataset has changed since this split: the next training splits it again.';
  return rejected ? 'The split leaves out ' + plural(rejected, 'rejected proposal') + '.' : '';
});
const routes = computed(() => {
  const value = props.dataset;
  if (value === null) return [];
  return value.per_route.map((row) => ({...row, pad: /^0+(?=[0-9a-f]+--)/.exec(row.route)?.[0] ?? ''}));
});
const totals = computed(() => routes.value.reduce((sum, row) => ({frames: sum.frames + row.frames, objects: sum.objects + row.objects,
  speed: sum.speed + row.speed_signs, human: sum.human + row.human, ai: sum.ai + row.ai}),
{frames: 0, objects: 0, speed: 0, human: 0, ai: 0}));
const imported = computed(() => props.dataset?.last_import ?? null);
</script>

<template>
  <section class="card dataset" id="dataset" aria-labelledby="dataset-title">
    <div class="ds-head">
      <h2 id="dataset-title">In the dataset</h2>
      <span v-if="dataset?.changed_at" class="ds-when">changed {{ isoWhen(dataset.changed_at) }}</span>
    </div>

    <p v-if="dataset === null" class="empty">{{ loading ? 'Loading…' : 'Nothing in the dataset yet: confirm some frames, then import the reviews.' }}</p>
    <template v-else>
      <div class="ds-figures">
        <div class="ds-figure"><span class="ds-value" id="ds-frames">{{ thousands(dataset.frames) }}</span><span class="ds-unit">frames</span></div>
        <div class="ds-figure"><span class="ds-value" id="ds-objects">{{ thousands(dataset.objects) }}</span><span class="ds-unit">objects</span></div>
        <div class="ds-figure"><span class="ds-value">{{ thousands(dataset.routes) }}</span><span class="ds-unit">{{ dataset.routes === 1 ? 'route' : 'routes' }}</span></div>
      </div>

      <div class="ds-grid">
        <div class="ds-block">
          <h3 class="ds-title">Objects by sign family</h3>
          <ul class="barlist" aria-label="Objects by sign family">
            <li v-for="family in families" :key="family.key" :data-family="family.key" :title="family.label + ': ' + thousands(family.count) + ' objects (' + family.share + ')'">
              <span class="bl-label">{{ family.label }}</span>
              <span class="bl-track" aria-hidden="true"><span :style="{width: Math.max(family.width, 0.6) + '%'}"></span></span>
              <span class="bl-value">{{ thousands(family.count) }}</span>
            </li>
          </ul>
        </div>
        <div class="ds-block">
          <h3 class="ds-title">Who confirmed them</h3>
          <ul class="barlist share" aria-label="Objects confirmed by people and by AI reviewers">
            <li v-for="row in reviewers" :key="row.key" :data-reviewer="row.key" :title="row.label + ': ' + thousands(row.count) + ' objects (' + row.share + ')'">
              <span class="bl-label">{{ row.label }}</span>
              <span class="bl-track" aria-hidden="true"><span :style="{width: Math.max(row.width, 0.6) + '%'}"></span></span>
              <span class="bl-value">{{ thousands(row.count) }}<small>{{ row.share }}</small></span>
            </li>
          </ul>
          <h3 class="ds-title">Last import</h3>
          <div v-if="imported" class="ds-import" id="ds-import">
            <p><template v-if="imported.at">{{ isoWhen(imported.at) }}: </template><strong>{{ plural(imported.new_frames, 'new frame') }}</strong>,
              {{ thousands(imported.replaced_frames ?? 0) }} updated, {{ thousands(imported.removed_frames ?? 0) }} removed<template
                v-if="imported.already">, {{ thousands(imported.already) }} unchanged</template>.</p>
            <p v-if="imported.changed_after_import > 0" class="ds-warn" id="ds-changed">
              {{ plural(imported.changed_after_import, 'confirmed frame was', 'confirmed frames were') }} changed after their import and
              not applied: the dataset only adds new frames, it never rewrites one it already holds.</p>
          </div>
          <p v-else class="empty">No import yet.</p>
        </div>
      </div>

      <div class="ds-block">
        <h3 class="ds-title">Split<span v-if="split?.made" class="ds-when">made {{ split.made }}</span></h3>
        <template v-if="split">
          <div class="split-bar" role="img" :aria-label="'Split: ' + split.parts.map((part) => part.label + ' ' + thousands(part.count)).join(', ')">
            <span v-for="part in split.parts.filter((item) => item.count > 0)" :key="part.key" :class="'split-' + part.slot"
                  :style="{flexGrow: part.count}" :title="part.label + ': ' + thousands(part.count) + ' objects (' + part.share + ')'"></span>
          </div>
          <ul class="split-legend">
            <li v-for="part in split.parts" :key="part.key" :data-split="part.key"><span class="split-key" :class="'split-' + part.slot" aria-hidden="true"></span>{{
              part.label }}<strong>{{ thousands(part.count) }}</strong><span class="split-share">{{ part.share }}</span></li>
          </ul>
          <p class="note">Training learns from <em>train</em>, picks its best epoch on <em>validation</em> and is measured on <em>test</em>; whole
            routes held out and hard negatives (things that look like signs) check that it generalises.<template
              v-if="splitNote"> {{ splitNote }}</template></p>
          <p v-for="(warning, index) in split.warnings" :key="index" class="note ds-note">{{ warning }}</p>
        </template>
        <p v-else class="empty">No split yet: the next training makes it.</p>
      </div>

      <div class="ds-block">
        <h3 class="ds-title">By route</h3>
        <div class="table-wrap"><table class="ds-routes">
          <thead><tr><th>Route</th><th class="num">Frames</th><th class="num">Objects</th><th class="num" title="Maximum speed, variable display, zone, end of limit">Speed signs</th>
            <th class="num">People</th><th class="num">AI</th></tr></thead>
          <tbody>
            <tr v-for="row in routes" :key="row.route">
              <td class="route"><a :href="signsHref(row.route)" :title="row.route"><span v-if="row.pad" class="rid-pad">{{ row.pad }}</span>{{
                row.route.slice(row.pad.length) }}</a></td>
              <td class="num">{{ thousands(row.frames) }}</td><td class="num">{{ thousands(row.objects) }}</td>
              <td class="num">{{ thousands(row.speed_signs) }}</td><td class="num">{{ thousands(row.human) }}</td><td class="num">{{ thousands(row.ai) }}</td>
            </tr>
          </tbody>
          <tfoot v-if="routes.length > 1"><tr><th scope="row">Total</th><td class="num">{{ thousands(totals.frames) }}</td><td class="num">{{ thousands(totals.objects) }}</td>
            <td class="num">{{ thousands(totals.speed) }}</td><td class="num">{{ thousands(totals.human) }}</td><td class="num">{{ thousands(totals.ai) }}</td></tr></tfoot>
        </table></div>
      </div>
    </template>
  </section>
</template>
