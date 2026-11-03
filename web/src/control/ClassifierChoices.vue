<script setup lang="ts">
// The choices of the classifier's next training (classifier-choices.ts), in Train › Classifier and in the classifier's
// step of Train › Pipeline: its signs, the places and the share of the Mapillary signs (MapillaryChoices.vue), where
// it starts.
import type {PanelStatus} from '../common/panel-types';
import PickerSheet from '../components/PickerSheet.vue';
import type {ClassifierChoices} from './classifier-choices';
import MapillaryChoices from './MapillaryChoices.vue';

defineProps<{choices: ClassifierChoices; status: PanelStatus | null}>();
defineEmits<{locate: []}>();
</script>

<template>
  <MapillaryChoices :choices="choices" :status="status" label="Signs the classifier learns from" @locate="$emit('locate')"/>
  <div v-if="choices.mapillary !== 'pretrain'" class="cls-start">
    <PickerSheet v-model="choices.startKey" :options="choices.startOptions" label="Start from" id="train-classifier-start"/>
  </div>
  <p v-if="choices.mapillary === 'pretrain'" class="latest-none" id="train-start-note">Start: ImageNet, then its own Mapillary
    stage; another classifier cannot be the start of “Mapillary, then ours”.</p>
  <p v-else-if="choices.startVersion && choices.startSplit(choices.startVersion) === 'another'" class="latest-none warn-text" id="train-start-note">Trained
    on another split: it may have seen some of today's test signs, so the test of the new one is not fair.</p>
  <p v-else-if="choices.startVersion && choices.startSplit(choices.startVersion) === 'unknown'" class="latest-none warn-text" id="train-start-note">Its
    split is not recorded: it may have seen some of today's test signs, so the test of the new one may not be fair.</p>
  <p v-else-if="choices.startVersion" class="latest-none" id="train-start-note">It goes on from that classifier's weights, learnt on
    today's split: the test stays fair.</p>
</template>
