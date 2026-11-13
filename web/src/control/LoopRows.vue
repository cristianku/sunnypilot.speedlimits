<script setup lang="ts">
// The loop of the work on the Overview (loop.ts): Drives, Review, Train, Car, numbered in their order, each a link to
// its page with how much waits there and its one button. Work running for a row takes the place of its button (its Stop
// is in the Running card above); a row whose work is done shows a tick. The first row with work waiting is marked.
// Train also has a link to Training › Pipeline under its button, there while its import runs too.
import {computed} from 'vue';
import {sentence} from '../common/format';
import type {ActionName, ActionState, PanelStatus} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import {ACTION_ICON} from './jobs';
import {loopRows} from './loop';

const props = defineProps<{status: PanelStatus | null}>();
defineEmits<{run: [name: ActionName, state: ActionState]}>();

const rows = computed(() => loopRows(props.status));
</script>

<template>
  <nav class="loop" aria-label="The loop of the work">
    <template v-if="rows">
      <section v-for="(row, index) in rows" :key="row.key" class="loop-row" :class="['k-' + row.kind, {waiting: row.waiting, primary: row.primary}]"
               :id="'loop-' + row.key" :aria-labelledby="'loop-' + row.key + '-label'">
        <a class="loop-head" :href="row.href" :id="'loop-' + row.key + '-label'"><span class="loop-num">{{ index + 1 }}</span>{{ row.label }}</a>
        <div class="loop-big">{{ row.big }}</div>
        <p v-if="row.note || row.more" class="loop-note">{{ row.note }} <a v-if="row.more" :href="row.more.href">{{ row.more.label }}</a></p>
        <p v-if="row.warn" class="loop-note warn-text">{{ row.warn }}</p>
        <p v-if="row.chips?.length" class="loop-chips" :id="'loop-' + row.key + '-chips'">
          <a v-for="chip in row.chips" :key="chip.href" :href="chip.href" :class="{ai: chip.ai}"><i v-if="chip.ai" class="sr-ai-square" aria-hidden="true"></i>{{ chip.label }}</a></p>
        <div class="loop-act">
          <span v-if="row.running" class="loop-running" role="status">{{ row.running }}</span>
          <template v-else-if="row.action && 'name' in row.action">
            <button class="btn" :class="{primary: row.primary}" type="button" :disabled="!row.action.state.allowed"
                    @click="$emit('run', row.action.name, row.action.state)"><Icon :name="ACTION_ICON[row.action.name]"/>{{ row.action.label }}</button>
            <p v-if="!row.action.state.allowed && row.action.state.reason" class="loop-note">{{ sentence(row.action.state.reason) }}</p>
          </template>
          <a v-else-if="row.action" class="btn" :class="{primary: row.primary}" :href="row.action.href"><Icon :name="row.action.icon"/>{{ row.action.label }}</a>
          <span v-else-if="row.ok && !row.link" class="loop-ok" aria-label="Done">✓</span>
          <a v-if="row.link" class="btn" :href="row.link.href" :id="'loop-' + row.key + '-link'"><Icon :name="row.link.icon"/>{{ row.link.label }}</a>
        </div>
      </section>
    </template>
    <p v-else class="loop-note">…</p>
  </nav>
</template>
