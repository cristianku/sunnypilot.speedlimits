<script setup lang="ts">
// Train › Brain: what the brain gets and says on the car, and what only its training has (brain-card.ts), with the
// driver's inputs among the inputs when the choice of the next training has them on.
import {computed} from 'vue';
import {BRAIN_ANSWERS, BRAIN_GROUPS, BRAIN_OUT, BRAIN_OUTPUTS, BRAIN_PICTURE} from './brain-card';

const props = defineProps<{driver: boolean}>();
const given = computed(() => BRAIN_GROUPS.filter((group) => group.use === 'car' || (group.use === 'driver' && props.driver)));
const notGiven = computed(() => BRAIN_GROUPS.filter((group) => !given.value.includes(group)));
</script>

<template>
  <div class="brain-io" id="train-brain-io">
    <section class="brain-io-part" id="train-brain-io-car">
      <h4>On the car: what it gets, 4 times a second</h4>
      <ul>
        <li v-for="fact in BRAIN_PICTURE" :key="fact.name"><code>{{ fact.name }}</code> {{ fact.what }}</li>
        <li v-for="group in given" :key="group.name"><code>{{ group.name }}</code> <span class="muted">{{ group.size }}</span> {{ group.what }}</li>
      </ul>
      <p class="brain-io-not">Not given: <template v-for="(group, index) in notGiven" :key="group.name">{{ index ? ', ' : '' }}<code>{{ group.name }}</code>
        ({{ group.use === 'driver' ? 'Driver’s inputs off' : 'no brain learns it' }})</template>, and nothing of the map.</p>
    </section>
    <section class="brain-io-part" id="train-brain-io-says">
      <h4>On the car: what a brain trained today says</h4>
      <ul>
        <li v-for="fact in BRAIN_OUTPUTS" :key="fact.name"><code>{{ fact.name }}</code> {{ fact.what }}</li>
      </ul>
    </section>
    <section class="brain-io-part" id="train-brain-io-answers">
      <h4>Only in training: the answers</h4>
      <ul>
        <li v-for="fact in BRAIN_ANSWERS" :key="fact.name"><code>{{ fact.name }}</code> {{ fact.what }}</li>
      </ul>
      <p class="brain-io-not">The answers are never inputs. works_head, road_head and type_head stay out of the car’s network:
        they only teach h in training.</p>
    </section>
    <section class="brain-io-part" id="train-brain-io-out">
      <h4>Only in training: seconds that answer nothing</h4>
      <ul>
        <li v-for="fact in BRAIN_OUT" :key="fact.name"><code>{{ fact.name }}</code> {{ fact.what }}</li>
      </ul>
    </section>
  </div>
</template>
