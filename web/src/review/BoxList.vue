<script setup lang="ts">
// "Signs": the boxes of the frame, what each one is, where it comes from, what the reviewer changed, the
// boxes that are the same sign twice. A click opens the editor of the box, ✕ removes it.
import {computed} from 'vue';
import {boxItems} from './boxes';
import {positionChecked} from './carry';
import {releaseFocus} from './dom';
import {removeBox, selectBox} from './edits';
import {state} from './store';
import type {FrameEdit, FrameRow} from './types';

const props = defineProps<{row: FrameRow; work: FrameEdit}>();
const items = computed(() => boxItems(props.row, props.work));
// The road of each box, as on the photo: OURS stands out.
const ROAD_TAGS = {ours: 'OURS', another: 'another road', unclear: 'road ?'} as const;
</script>

<template>
  <div class="box-list" id="boxList">
    <div v-for="item in items" :key="item.index"
         :class="['box-row', {selected: item.index === state.selected, check: item.check, ai: item.ai}]"
         :data-index="item.index" :data-detection-id="item.detectionId"
         :data-duplicate-of="item.duplicateOf === null ? undefined : item.duplicateOf">
      <button type="button" class="box-main" :aria-pressed="item.index === state.selected" @click="selectBox(item.index); releaseFocus($event)"><span class="n">{{ item.index + 1 }}</span><span class="what"><span class="what-text">{{ item.what }}</span><span :class="['road-tag', item.road]"><svg v-if="item.road === 'ours'" viewBox="0 0 12 12" aria-hidden="true"><path d="M1 11.5L4.6 1M11 11.5L7.4 1M6 11V7.6"/></svg>{{ ROAD_TAGS[item.road] }}</span></span><span class="src">{{ item.source.text }}<span v-if="item.source.changed !== null" class="changed">{{ item.source.changed }}</span><span v-if="item.duplicateOf !== null" class="dup">{{ ' · same sign as ' + (item.duplicateOf + 1) }}</span></span></button>
      <button v-if="item.check" type="button" class="btn small box-here" :title="'The sign is here: box ' + (item.index + 1) + ' checked'"
              :aria-label="'Position of box ' + (item.index + 1) + ' checked'" @click="positionChecked(item.index); releaseFocus($event)">Here</button>
      <button type="button" class="box-del" :title="'Delete box ' + (item.index + 1)"
              :aria-label="'Delete box ' + (item.index + 1)" @click="removeBox(item.index); releaseFocus($event)">✕</button>
    </div>
  </div>
</template>
