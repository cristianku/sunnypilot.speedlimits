<script setup lang="ts">
// The photo of Sign review's editor over the picture of Speed review's player (SignEditorPanel.vue is its side): the
// frame of Sign review the player stands on, with its boxes to select, move, resize and draw by mouse or finger
// (Stage.vue, pointer.ts). A click on it is its own: it never starts the player.
import {row, work} from './current';
import Stage from './Stage.vue';
</script>

<template>
  <div class="sign-editor sr-editor-stage" @click.stop>
    <Stage v-if="row && work" :row="row" :work="work"/>
  </div>
</template>
