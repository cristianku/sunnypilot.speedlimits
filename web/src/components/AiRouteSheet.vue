<script setup lang="ts">
// "Review with AI" on the Signs side of Speed review (control/SpeedReviewDrive.vue): the button of a drive
// (button[data-ai-route], with data-ai-name and data-ai-left) opens this sheet. It reads the AI reviewer of Settings (GET /api/ai/state) and starts the panel job
// ai_review on the drive (POST /api/panel/run/ai_review): "Try without saving" suggests the boxes of 20 frames,
// which the Signs side of the drive shows (its "AI trial" filter); "Review and save" saves what the AI says, after
// a question; what it saves is final (since 13 Nov 2026, #79): an AI review that goes into training at the next import. Only a reviewer that passed its Test in Settings can do either.
// What: All signs (every frame left) or Traffic lights (scope lights: only the frames where the car comes to a traffic
// light of the map, #79); each frame is reviewed whole either way.
import {computed, onBeforeUnmount, onMounted, ref, useTemplateRef} from 'vue';
import {api, messageOf, post} from '../common/api';
import type {AiState} from '../common/ai-types';
import type {StartedJob} from '../common/panel-types';
import {signsHref} from '../common/sign-link';
import {askConfirm} from '../composables/confirm';
import {useSheet} from '../composables/sheet';
import Icon from './Icon.vue';

const TRY_LIMIT = '20';

interface Drive {
  route: string;
  name: string;
  left: number;
}

const drive = ref<Drive | null>(null);
const ai = ref<AiState | null>(null);
const loadError = ref('');
const error = ref('');
const busy = ref(false);
const started = ref<{label: string; saving: boolean} | null>(null);
type Scope = 'signs' | 'lights';
const SCOPES: readonly {key: Scope; label: string}[] = [{key: 'signs', label: 'All signs'}, {key: 'lights', label: 'Traffic lights'}];
const scope = ref<Scope>('signs');
const panel = useTemplateRef<HTMLElement>('panel');
const open = computed(() => drive.value !== null);
const close = (): void => { drive.value = null; };
useSheet(open, panel, close);

/** What the saves are recorded as: ai/<provider>/<model>. */
const recordedAs = computed(() => {
  const name = ai.value?.provider?.reviewer_name ?? '';
  return name.startsWith('ai/') ? name : 'ai/' + name;
});
const blocker = computed(() => {
  if (ai.value === null) return '';
  if (!ai.value.available || ai.value.can_ask === false) return ai.value.reason || 'Choose an AI reviewer in Settings.';
  return '';
});
const leftText = computed(() => {
  const left = drive.value?.left ?? 0;
  return left === 1 ? '1 frame left to review' : left.toLocaleString('en-US') + ' frames left to review';
});

async function show(target: HTMLElement): Promise<void> {
  const route = target.dataset['aiRoute'];
  if (!route) return;
  drive.value = {route, name: target.dataset['aiName'] || route, left: Number(target.dataset['aiLeft'] || 0)};
  ai.value = null;
  loadError.value = '';
  error.value = '';
  started.value = null;
  scope.value = 'signs';
  try {
    ai.value = await api<AiState>('/api/ai/state');
  } catch (problem) {
    loadError.value = messageOf(problem);
  }
}

async function start(save: boolean): Promise<void> {
  const value = drive.value;
  if (value === null || busy.value) return;
  error.value = '';
  if (save) {
    const ok = await askConfirm({title: 'Review and save with the AI?', action: 'Review and save', danger: true,
      text: 'What it saves is recorded as ' + recordedAs.value + ' and is final: it goes into training at the next import '
        + 'as an AI review, with nothing to confirm. You can still correct a frame on the Signs side. Frames a person '
        + 'confirmed are never touched.'});
    if (!ok || drive.value !== value) return;
  }
  busy.value = true;
  try {
    const body: Record<string, string> = save ? {route: value.route, save: 'yes'} : {route: value.route, save: 'no', limit: TRY_LIMIT};
    if (scope.value === 'lights') body['scope'] = 'lights';
    const job = await post<StartedJob>('/api/panel/run/ai_review', body);
    if (drive.value === value) started.value = {label: job.label, saving: save};
  } catch (problem) {
    if (drive.value === value) error.value = messageOf(problem);
  } finally {
    busy.value = false;
  }
}

function onClick(event: MouseEvent): void {
  const target = event.target instanceof Element ? event.target.closest<HTMLElement>('button[data-ai-route]') : null;
  // "Stop the AI" on the drive the AI reviews is the page script's (pages/site.ts).
  if (target === null || target.dataset['aiStop']) return;
  event.preventDefault();
  void show(target);
}
onMounted(() => document.addEventListener('click', onClick));
onBeforeUnmount(() => document.removeEventListener('click', onClick));
</script>

<template>
  <div class="sheet-backdrop" :class="{open}" @click="close"></div>
  <section ref="panel" class="sheet ai-sheet" :class="{open}" role="dialog" aria-modal="true" aria-labelledby="ai-route-title"
           :aria-hidden="!open" :inert="!open" id="ai-route-sheet">
    <div class="sheet-grabber" aria-hidden="true"></div>
    <div class="sheet-head">
      <div class="grow"><h2 id="ai-route-title">Review with AI</h2>
        <p class="sheet-sub">{{ drive?.name }}<template v-if="drive"> · {{ leftText }}</template></p></div>
      <button class="btn small" type="button" @click="close">Close</button>
    </div>
    <div class="sheet-body">
      <p v-if="ai === null && !loadError" class="ai-help">Reading the AI reviewer…</p>
      <p v-else-if="loadError" class="auth-error" role="alert">{{ loadError }}</p>
      <template v-else-if="started">
        <p class="ai-started" role="status"><span class="pill run">{{ started.saving ? 'Reviewing' : 'Trying' }}</span>
          <span>Started: follow it on the <a href="/">Overview</a>.</span></p>
        <p class="ai-help">{{ started.saving ? 'Each frame the AI reviews completely is saved as ' + recordedAs + '; the others stay for you.'
          : 'Nothing is saved. When it is done, open the drive: the frames the AI tried are under “AI trial”, ready to apply or dismiss.' }}</p>
        <a v-if="!started.saving && drive" class="btn primary" id="ai-route-open" :href="signsHref(drive.route)">Open the drive</a>
      </template>
      <template v-else-if="blocker">
        <p class="ai-help" role="status">{{ blocker }}</p>
        <a class="btn primary" href="/account#ai-reviewer">Open Settings</a>
      </template>
      <template v-else-if="ai?.provider">
        <ul class="ios-list ai-who">
          <li class="ai-prov"><span class="ai-prov-main"><span class="ai-prov-name">{{ ai.provider.name }}</span>
            <span class="ai-prov-sub">{{ ai.provider.model }} · saves as <code>{{ recordedAs }}</code></span></span></li>
        </ul>
        <div class="segmented ai-scope" role="radiogroup" aria-label="What the AI reviews" id="ai-route-scope">
          <button v-for="option in SCOPES" :key="option.key" type="button" role="radio" :aria-checked="scope === option.key"
                  :title="option.key === 'lights' ? 'Only the frames where the car comes to a traffic light of the map; each is still reviewed whole' : 'Every frame left to review'"
                  @click="scope = option.key">{{ option.label }}</button>
        </div>
        <div class="ai-choices">
          <button type="button" class="ai-choice" id="ai-route-try" :disabled="busy" @click="start(false)">
            <span class="ai-choice-icon"><Icon name="ai"/></span>
            <span class="ai-choice-body"><strong>Try without saving</strong>
              <span>The AI suggests the boxes of {{ TRY_LIMIT }} frames left to review{{ scope === 'lights' ? ' that come to a traffic light' : '' }}. You see them on the drive's Signs side; nothing is saved.</span></span>
          </button>
          <button type="button" class="ai-choice save" id="ai-route-save" :disabled="busy" @click="start(true)">
            <span class="ai-choice-icon"><Icon name="ai"/></span>
            <span class="ai-choice-body"><strong>Review and save</strong>
              <span>{{ scope === 'lights' ? 'Every frame left to review where the car comes to a traffic light of the map.' : 'Every frame left to review.' }} What the AI saves is final and goes into training as an AI review.</span></span>
          </button>
        </div>
      </template>
      <p class="auth-error" role="alert">{{ error }}</p>
    </div>
  </section>
</template>
