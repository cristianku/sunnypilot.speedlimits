<script setup lang="ts">
// The actions of a tab, one card each: the whole card is the button (its icon, its label, what it does in one
// sentence) and, when it cannot start now, why (the reason line, tied to the button for screen readers). tip: a line more
// in each button's tooltip, after what it does (the two GPUs, on the trainings).
import {computed} from 'vue';
import {sentence} from '../common/format';
import type {ActionName, ActionState, PanelStatus} from '../common/panel-types';
import Icon from '../components/Icon.vue';
import {ACTION_ICON} from './jobs';

const props = defineProps<{items: [name: ActionName, text: string][]; actions: PanelStatus['actions'] | null;
  tip?: string | undefined}>();
defineEmits<{run: [name: ActionName, state: ActionState]}>();

const rows = computed(() => {
  const actions = props.actions;
  return props.items.map(([name, text]) => {
    const state = actions?.[name] ?? null;
    const reason = state && !state.allowed && state.reason ? sentence(state.reason) : '';
    return {name, text, state, reason, title: props.tip ? text + '\n' + props.tip : text};
  });
});
</script>

<template>
  <ul class="acts">
    <li v-for="row in rows" :key="row.name" class="act" :class="{blocked: row.state !== null && !row.state.allowed}" :data-action="row.name">
      <button class="act-card" type="button" :id="'run-' + row.name" :title="row.title" :disabled="row.state === null || !row.state.allowed"
              :aria-labelledby="'act-label-' + row.name"
              :aria-describedby="'act-text-' + row.name + (row.reason ? ' act-reason-' + row.name : '')"
              @click="row.state && $emit('run', row.name, row.state)">
        <span class="act-icon"><Icon :name="ACTION_ICON[row.name]"/></span>
        <span class="act-body">
          <span class="act-label" :id="'act-label-' + row.name">{{ row.state?.label ?? '…' }}</span>
          <span class="act-text" :id="'act-text-' + row.name">{{ row.text }}</span>
          <span v-if="row.reason" class="act-reason" :id="'act-reason-' + row.name">{{ row.reason }}</span>
        </span>
        <span class="act-go" aria-hidden="true"></span>
      </button>
    </li>
  </ul>
</template>
