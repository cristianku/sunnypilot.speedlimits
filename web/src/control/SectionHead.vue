<script setup lang="ts">
// The head of a page: its section's title, large as on iOS, and under it the section's pages as a segmented control
// (Data: Drives, Review, Drive review, Speed review, Roads, Training dataset; Models: Versions, Results, Publish), the page shown marked and a
// dot on a page with work running; on a phone the control scrolls sideways, the page shown brought into view. pages.sub_nav
// draws the same on the route list.
import {computed, nextTick, onMounted, useTemplateRef, watch} from 'vue';
import {busyPaths} from '../composables/busy';
import {SECTIONS, sectionOf, type ControlTab} from './tabs';

const props = defineProps<{tab: ControlTab}>();
const section = computed(() => SECTIONS[sectionOf(props.tab)]);
const nav = useTemplateRef<HTMLElement>('nav');
function reveal(): void {
  const box = nav.value;
  const link = box?.querySelector<HTMLElement>('[aria-current="page"]');
  if (!box || !link || box.scrollWidth <= box.clientWidth) return;
  const left = link.getBoundingClientRect().left - box.getBoundingClientRect().left + box.scrollLeft;
  box.scrollLeft = Math.max(0, left - (box.clientWidth - link.offsetWidth) / 2);
}
onMounted(reveal);
watch(() => props.tab, () => void nextTick(reveal));
</script>

<template>
  <header class="section-head">
    <h1 class="head-title">{{ section.title }}</h1>
    <nav v-if="section.pages.length" ref="nav" class="subnav" :aria-label="section.title">
      <a v-for="page in section.pages" :key="page.key" :href="page.href" :aria-current="page.key === tab ? 'page' : undefined"
         :data-busy="busyPaths.has(page.href) ? 'true' : undefined">{{ page.label }}</a>
    </nav>
  </header>
</template>
