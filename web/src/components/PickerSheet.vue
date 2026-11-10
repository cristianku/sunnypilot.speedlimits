<script setup lang="ts">
// A choice of one among many, as on iOS: a compact button (what is chosen: its name, under it its details, ›) opens a
// sheet from the bottom with every option as a row (name, details, a badge such as "In use"), a ✓ on the chosen one; a
// tap chooses and closes. Instead of a native select, whose list a phone draws in its own style with the long texts
// wrapped. The look is shared.css's .sheet and .picker.
import {computed, ref, useTemplateRef} from 'vue';
import {useSheet} from '../composables/sheet';

export interface PickerOption {
  key: string;
  title: string;
  /** A shorter title for the button ("19:30" under the label "Classifier"); the sheet shows the title. */
  short?: string | undefined;
  sub?: string | undefined;
  badge?: string | undefined;
}

const props = defineProps<{options: PickerOption[]; label: string; id?: string}>();
const model = defineModel<string>({required: true});
const open = ref(false);
const panel = useTemplateRef<HTMLElement>('panel');
const close = (): void => { open.value = false; };
useSheet(open, panel, close);
const chosen = computed(() => props.options.find((option) => option.key === model.value) ?? props.options[0] ?? null);
function choose(key: string): void {
  model.value = key;
  close();
}
</script>

<template>
  <button type="button" class="picker" :id="id" :disabled="options.length < 2" :aria-label="label + ': ' + (chosen?.title ?? '')"
          aria-haspopup="dialog" @click="open = true">
    <span class="picker-label">{{ label }}</span>
    <span class="picker-title"><span class="picker-text">{{ chosen?.short ?? chosen?.title ?? '–' }}</span><span v-if="chosen?.badge" class="picker-badge">{{ chosen.badge }}</span></span>
    <span v-if="chosen?.sub" class="picker-sub">{{ chosen.sub }}</span>
    <span v-if="options.length > 1" class="picker-chev" aria-hidden="true">›</span>
  </button>
  <Teleport to="body">
    <div class="sheet-backdrop" :class="{open}" @click="close"></div>
    <section ref="panel" class="sheet picker-sheet" :class="{open}" role="dialog" aria-modal="true" :aria-label="label"
             :aria-hidden="!open" :inert="!open">
      <div class="sheet-grabber" aria-hidden="true"></div>
      <div class="picker-head"><h2>{{ label }}</h2>
        <button class="picker-done" type="button" @click="close">Done</button></div>
      <div class="sheet-body">
        <ul class="picker-list" role="listbox" :aria-label="label">
          <li v-for="option in options" :key="option.key">
            <button type="button" class="picker-row" role="option" :aria-selected="option.key === model" @click="choose(option.key)">
              <span class="picker-row-main"><span class="picker-row-title">{{ option.title }}<span v-if="option.badge"
                class="picker-badge">{{ option.badge }}</span></span>
                <span v-if="option.sub" class="picker-row-sub">{{ option.sub }}</span></span>
              <span class="picker-check" aria-hidden="true">{{ option.key === model ? '✓' : '' }}</span>
            </button>
          </li>
        </ul>
      </div>
    </section>
  </Teleport>
</template>
