<script setup lang="ts">
// The networks that read the brain's signs (sign-networks.ts) in one row, in Train › Brain and in its step of
// Train › Pipeline: the cutter, the classifier and the applicability network, those in use first; in a pipeline, none
// for a part it trains before the brain (trained, #71).
import PickerSheet from '../components/PickerSheet.vue';
import {SIGN_PARTS, type SignNetworksChoice, type SignPart} from './sign-networks';

withDefaults(defineProps<{choice: SignNetworksChoice; id: string; trained?: readonly SignPart[]}>(), {trained: () => []});
</script>

<template>
  <div v-if="SIGN_PARTS.some((item) => !trained.includes(item.part))" class="mly-choice sn-row" :id="id + '-signs'" role="group" aria-label="The networks that read the signs the brain learns from">
    <template v-for="item in SIGN_PARTS" :key="item.part">
      <PickerSheet v-if="choice.options[item.part].length && !trained.includes(item.part)" v-model="choice.keys[item.part]" :options="choice.options[item.part]"
                   :label="'Signs: ' + item.label" :id="id + '-' + item.part"/>
    </template>
  </div>
  <p class="latest-none" :class="{'warn-text': choice.noteOf(trained).warn}" :id="id + '-signs-note'">{{ choice.noteOf(trained).text }}</p>
</template>
