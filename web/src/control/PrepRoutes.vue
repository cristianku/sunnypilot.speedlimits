<script setup lang="ts">
// The routes a running preparation is working on: the drive, its phase and how far it is (for Propose again every
// route of the batch, waiting, proposing or ready; the time left is the card's, once, above them). A drive with a
// report opens its review page: read only while Propose again works on it (the new proposals made so far).
import {thousands} from '../common/format';
import type {RouteProgress} from '../common/panel-types';
import {signsHref} from '../common/sign-link';
import Bar from '../components/Bar.vue';

defineProps<{routes: RouteProgress[]}>();
// The server says the phase in English; these are the codes of an older server.
const PHASE: Record<string, string> = {frame: 'extracting frames', cartelli: 'proposing signs', pronta: 'ready',
  'da collegare': 'to link'};
// The phases of a drive that has a report to open.
const OPENS = new Set(['ready', 'pronta', 'waiting', 'proposing again']);
</script>

<template>
  <ul v-if="routes.length" class="prep-routes">
    <li v-for="route in routes" :key="route.route">
      <component :is="OPENS.has(route.phase) ? 'a' : 'div'" class="prep-drive"
                 :href="OPENS.has(route.phase) ? signsHref(route.route) : undefined">
        <span class="prep-route">{{ route.title || route.route.replace(/^0+(?=[0-9a-f]+--)/, '') }}</span>
        <span class="prep-phase">{{ PHASE[route.phase] ?? route.phase }}<template v-if="route.total && route.phase !== 'waiting'"> · {{ thousands(route.done) }}
          of {{ thousands(route.total) }} {{ route.unit }}</template><span v-if="OPENS.has(route.phase)" class="prep-chev" aria-hidden="true">›</span></span>
        <Bar :percent="route.percent" :kind="route.percent >= 100 ? 'ok' : 'run'"/>
      </component>
    </li>
  </ul>
</template>
