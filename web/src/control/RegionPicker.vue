<script setup lang="ts">
// Where the Mapillary photos of a training come from (the classifier's crops, the cutter's windows): a button showing the places chosen, and under it a menu that
// opens downwards (on a phone, a sheet from the bottom) with All the world, the continents and the countries (each with its signs), several at a time. The
// positions come from helpers/mapillary_locations.sh: until they are fetched the menu says so and offers to fetch them.
// v-model: the keys chosen (continents: europe, north_america...; countries: CH, DE...); none is All the world. idPrefix
// names its elements (mapillary-region...: the classifier's; another for a second picker on the same page).
import {computed, nextTick, onBeforeUnmount, ref, watch} from 'vue';
import {sentence, thousands} from '../common/format';
import type {ActionState, MapillaryRegions} from '../common/panel-types';
import {CONTINENT_NAMES, regionCrops} from './mapillary';

const props = withDefaults(defineProps<{regions: MapillaryRegions | null; locate: ActionState | null; locating: boolean;
  idPrefix?: string}>(), {idPrefix: 'mapillary'});
const emit = defineEmits<{locate: []}>();
const chosen = defineModel<string[]>({required: true});

const open = ref(false);
const query = ref('');
const root = ref<HTMLElement | null>(null);
const fetched = computed(() => Boolean(props.regions?.fetched));

const nameOf = (key: string): string => CONTINENT_NAMES[key]
  ?? props.regions?.countries.find((item) => item.code === key)?.name ?? key;
const title = computed(() => {
  if (!chosen.value.length) return 'All the world';
  if (chosen.value.length <= 2) return chosen.value.map(nameOf).join(', ');
  return thousands(chosen.value.length) + ' places';
});
const signs = computed(() => regionCrops(props.regions, chosen.value));
const sub = computed(() => {
  if (!props.regions) return '';
  if (!fetched.value) return 'Positions not fetched yet';
  return thousands(signs.value) + ' signs' + (chosen.value.length > 2 ? ' · ' + chosen.value.map(nameOf).join(', ') : '');
});

/** The countries by continent, the continents with most signs first; only those matching the search when there is one. */
const groups = computed(() => {
  const regions = props.regions;
  if (!regions) return [];
  const wanted = query.value.trim().toLowerCase();
  const order = regions.continents.map((item) => item.key);
  const byContinent = new Map<string, MapillaryRegions['countries']>();
  for (const country of regions.countries) {
    if (wanted && !country.name.toLowerCase().includes(wanted) && country.code.toLowerCase() !== wanted) continue;
    const key = country.continent ?? 'other';
    byContinent.set(key, [...(byContinent.get(key) ?? []), country]);
  }
  return [...byContinent.entries()]
    .sort(([a], [b]) => (order.indexOf(a) === -1 ? 99 : order.indexOf(a)) - (order.indexOf(b) === -1 ? 99 : order.indexOf(b)))
    .map(([key, countries]) => ({key, name: CONTINENT_NAMES[key] ?? 'Elsewhere', countries}));
});

/** A continent chosen takes its countries' place; a country chosen leaves its continent. */
function toggle(key: string, continent: string | null = null): void {
  const next = new Set(chosen.value);
  if (next.has(key)) next.delete(key);
  else {
    next.add(key);
    if (key in CONTINENT_NAMES) {
      for (const country of props.regions?.countries ?? []) if (country.continent === key) next.delete(country.code);
    } else if (continent) next.delete(continent);
  }
  chosen.value = [...next];
}

function close(): void { open.value = false; }
function onDocument(event: Event): void {
  if (root.value && !root.value.contains(event.target as Node)) close();
}
function onKey(event: KeyboardEvent): void {
  if (event.key === 'Escape' && open.value) {
    close();
    root.value?.querySelector<HTMLButtonElement>('.rp-button')?.focus();
  }
}
watch(open, async (value) => {
  if (value) {
    document.addEventListener('pointerdown', onDocument);
    document.addEventListener('keydown', onKey);
    await nextTick();
    root.value?.querySelector<HTMLElement>('.rp-menu .picker-row, .rp-menu .btn')?.focus({preventScroll: true});
    // Under the button it may open past the window's edge: bring it in, as little as needed (on a phone it is a sheet
    // fixed at the bottom, always in view).
    const menu = root.value?.querySelector<HTMLElement>('.rp-menu');
    if (menu && getComputedStyle(menu).position !== 'fixed') menu.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  } else {
    document.removeEventListener('pointerdown', onDocument);
    document.removeEventListener('keydown', onKey);
    query.value = '';
  }
});
onBeforeUnmount(() => {
  document.removeEventListener('pointerdown', onDocument);
  document.removeEventListener('keydown', onKey);
});
</script>

<template>
  <div ref="root" class="rp">
    <button type="button" class="picker rp-button" :id="idPrefix + '-region'" :aria-expanded="open" :aria-controls="idPrefix + '-region-menu'"
            @click="open = !open">
      <span class="picker-label">Where</span>
      <span class="picker-title"><span class="rp-title">{{ title }}</span></span>
      <span class="picker-sub">{{ sub }}</span>
      <span class="picker-chev" aria-hidden="true">›</span>
    </button>
    <div v-if="open" class="rp-backdrop" aria-hidden="true" @click="close"></div>
    <div v-if="open" class="rp-menu" :id="idPrefix + '-region-menu'" role="group" aria-label="Where the photos were taken">
      <div class="rp-head">
        <span class="rp-head-title">Where the photos were taken</span>
        <button type="button" class="picker-done" @click="close">Done</button>
      </div>
      <div class="rp-scroll">
        <ul v-if="!query" class="picker-list">
          <li><button type="button" class="picker-row" role="menuitemcheckbox" :aria-checked="!chosen.length" @click="chosen = []">
            <span class="picker-row-main"><span class="picker-row-title">All the world</span>
              <span class="picker-row-sub">{{ thousands(regions?.crops ?? 0) }} signs</span></span>
            <span class="picker-check" aria-hidden="true">{{ chosen.length ? '' : '✓' }}</span></button></li>
        </ul>
        <div v-if="!fetched" class="rp-empty" :id="idPrefix + '-region-empty'">
          <p class="rp-empty-title">Positions not fetched yet</p>
          <p class="rp-empty-text">To choose a continent or a country, first find where each Mapillary photo was taken.
            It asks Mapillary over the internet and takes a few minutes.</p>
          <button type="button" class="btn primary rp-locate" :id="idPrefix + '-locate'" :disabled="locating || !locate?.allowed"
                  @click="emit('locate')">{{ locating ? 'Locating the photos…' : 'Locate the photos' }}</button>
          <p v-if="!locating && locate && !locate.allowed && locate.reason" class="rp-reason">{{ sentence(locate.reason) }}</p>
        </div>
        <template v-else>
          <input v-model="query" class="rp-search" type="search" placeholder="Search a country" aria-label="Search a country"
                 autocomplete="off" spellcheck="false">
          <template v-if="!query">
            <p class="sheet-section">Continents</p>
            <ul class="picker-list">
              <li v-for="item in regions?.continents ?? []" :key="item.key">
                <button type="button" class="picker-row" role="menuitemcheckbox" :aria-checked="chosen.includes(item.key)"
                        :data-region="item.key" @click="toggle(item.key)">
                  <span class="picker-row-main"><span class="picker-row-title">{{ item.name }}</span>
                    <span class="picker-row-sub">{{ thousands(item.crops) }} signs</span></span>
                  <span class="picker-check" aria-hidden="true">{{ chosen.includes(item.key) ? '✓' : '' }}</span></button>
              </li>
            </ul>
          </template>
          <template v-for="group in groups" :key="group.key">
            <p class="sheet-section">{{ query ? group.name : 'Countries of ' + group.name }}</p>
            <ul class="picker-list">
              <li v-for="item in group.countries" :key="item.code">
                <button type="button" class="picker-row" role="menuitemcheckbox" :aria-checked="chosen.includes(item.code)"
                        :data-region="item.code" @click="toggle(item.code, item.continent)">
                  <span class="picker-row-main"><span class="picker-row-title">{{ item.name }}</span>
                    <span class="picker-row-sub">{{ thousands(item.crops) }} signs<span v-if="item.continent && chosen.includes(item.continent)">
                      · in {{ group.name }}</span></span></span>
                  <span class="picker-check" aria-hidden="true">{{ chosen.includes(item.code) ? '✓' : '' }}</span></button>
              </li>
            </ul>
          </template>
          <p v-if="query && !groups.length" class="rp-foot">No country with Mapillary signs matches “{{ query }}”.</p>
          <p class="rp-foot">{{ thousands(regions?.located_crops ?? 0) }} of {{ thousands(regions?.crops ?? 0) }} signs have a
            place; the others come in only with All the world.<template v-if="locating"> Locating the photos…</template><template
            v-else-if="locate?.allowed && (regions?.located_crops ?? 0) < (regions?.crops ?? 0)"> <button type="button"
            class="link-button" :id="idPrefix + '-locate-again'" @click="emit('locate')">Locate the rest</button></template></p>
        </template>
      </div>
    </div>
  </div>
</template>
