<script setup lang="ts">
// The Mapillary switch of a training (mapillary-choices.ts), in Train › Classifier, Train › Cutter and their steps of
// Train › Pipeline: our signs, with Mapillary, or Mapillary first; then where the photos were taken and a share of them.
// idPrefix names its elements (mapillary: the classifier's; cutter-mapillary: the cutter's), noteId the note under the
// card that says what the share takes.
import {thousands} from '../common/format';
import type {PanelStatus} from '../common/panel-types';
import {MAPILLARY_OPTIONS, type MapillarySwitch} from './mapillary-choices';
import RegionPicker from './RegionPicker.vue';
import Segmented from './Segmented.vue';

withDefaults(defineProps<{choices: MapillarySwitch; status: PanelStatus | null; label: string; idPrefix?: string;
  noteId?: string}>(), {idPrefix: 'mapillary', noteId: 'train-mapillary-note'});
defineEmits<{locate: []}>();
</script>

<template>
  <div v-if="choices.mapillaryState?.available" class="seg-row">
    <Segmented v-model="choices.mapillary" class="mly-seg" :options="MAPILLARY_OPTIONS" :label="label"/>
  </div>
  <div v-if="choices.mapillaryState?.available && choices.withMapillary" class="mly-choice" :id="'train-' + idPrefix + '-choice'">
    <RegionPicker v-model="choices.region" :regions="choices.mapillaryRegions" :locate="status?.actions.mapillary_locations ?? null"
                  :locating="choices.locating" :id-prefix="idPrefix" @locate="$emit('locate')"/>
    <label class="picker mly-share" :class="{bad: choices.share === null}">
      <span class="picker-label">Share</span>
      <span class="mly-share-field"><input v-model="choices.shareText" :id="idPrefix + '-share'" type="text" inputmode="decimal"
             maxlength="6" autocomplete="off" :aria-describedby="noteId">%</span>
      <span class="picker-sub">{{ choices.share === null ? 'from 0.1 to 100' : 'about ' + thousands(choices.mapillarySigns) + ' signs' }}</span>
    </label>
  </div>
</template>
