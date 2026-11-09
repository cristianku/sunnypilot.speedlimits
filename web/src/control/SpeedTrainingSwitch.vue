<script setup lang="ts">
// Speed review › a drive › "OK for brain training" (speed_review.save_training): a person who went through the whole drive
// says all its signs were marked right, and the brain learns it, whole (brain_targets.training_routes); a drive without
// it stays out of the brain training whole, never in pieces, since the brain has a memory. Under the switch, who set it
// last and when. Only the administrator sets it: a guest sees it, never changes it, and neither does the access token
// alone (the server refuses both). It is off, with the reason, on a drive whose logs are not read or with sunnypilot on.
import {computed, ref} from 'vue';
import {messageOf, post} from '../common/api';
import {isoWhen} from '../common/format';
import {isGuest} from '../common/guest';
import {toast} from '../composables/toast';
import type {SpeedDrive} from './speed-review';

const props = defineProps<{drive: SpeedDrive; readOnly: boolean}>();
/** changed: the drive as the server has it after the switch; failed: read it again. */
const emit = defineEmits<{changed: [drive: SpeedDrive]; failed: []}>();

const on = computed(() => props.drive.training_ok?.ok === true);
/** Why the switch cannot change here, or ''. Turning it off is never kept from the administrator. */
const reason = computed((): string => {
  if (isGuest()) return 'Only the administrator marks a drive OK for training.';
  if (props.readOnly) return 'The panel is read-only on this server.';
  if (on.value) return '';
  if (!props.drive.span_ns) return 'Its logs are not read yet: the brain cannot learn from it.';
  if (props.drive.dashcam === false) return 'sunnypilot was on in this drive: the brain never learns from it.';
  return '';
});
const who = computed((): string => {
  const set = props.drive.training_ok;
  if (!set) return '';
  return (set.ok ? 'Marked OK by ' : 'Unmarked by ') + set.reviewer + (isoWhen(set.at) ? ' · ' + isoWhen(set.at) : '');
});

const saving = ref(false);
async function toggle(event: Event): Promise<void> {
  const box = event.target as HTMLInputElement;
  const ok = box.checked;
  // The switch shows what the server has, never what was tapped, until it answers.
  box.checked = on.value;
  if (reason.value || saving.value) return;
  saving.value = true;
  try {
    emit('changed', await post<SpeedDrive>('/api/speed-review/drives/' + encodeURIComponent(props.drive.route) + '/training',
      {ok}));
    toast(ok ? 'Marked OK for brain training: the brain learns this drive whole.'
      : 'Unmarked: this drive stays out of the brain training.');
  } catch (problem) {
    toast(messageOf(problem), true);
    emit('failed');
  } finally {
    saving.value = false;
  }
}
</script>

<template>
  <div class="sr-training" id="sr-training" role="group" aria-labelledby="sr-training-title">
    <label class="sr-switch" :class="{'sr-switch-off': !!reason}">
      <span id="sr-training-title" class="sr-switch-label">OK for brain training</span>
      <input type="checkbox" role="switch" id="sr-training-ok" :checked="on" :aria-checked="on"
             :disabled="!!reason || saving" @change="toggle">
      <span class="sr-switch-track" aria-hidden="true"></span>
    </label>
    <p v-if="who" class="dr-hint" id="sr-training-who">{{ who }}</p>
    <p class="dr-hint" id="sr-training-why">{{ reason || (on
      ? 'The brain learns this drive whole. Turn it off if a sign of it is wrong.'
      : 'Turn it on once you went through the whole drive and every sign is marked right. Until then the whole drive stays out of the brain training.') }}</p>
  </div>
</template>
