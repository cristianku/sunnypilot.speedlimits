<script setup lang="ts">
// The Account menu at the right of every top bar: who is signed in, the appearance (dark, light,
// automatic), Settings (the password, the signed-in browsers, the AI reviewer) and sign out. A panel under the icon, a sheet
// across the width on a phone. On a phone it is also the site's menu (the ☰ instead of the person, as shops do): the
// pages of the tab row first (read from nav.tabs, hidden there), the account at the bottom.
import {computed, nextTick, onBeforeUnmount, onMounted, ref, useTemplateRef} from 'vue';
import {api, messageOf, post} from '../common/api';
import type {AuthState} from '../common/auth-types';
import {isGuest} from '../common/guest';
import {onThemeChange, setTheme, storedTheme, type Theme} from '../common/theme';
import {busyPaths, linkBusy} from '../composables/busy';
import {askConfirm} from '../composables/confirm';

const props = defineProps<{current?: boolean}>();
const guest = isGuest();
const THEMES: [Theme, string][] = [['dark', 'Dark'], ['light', 'Light'], ['auto', 'Automatic']];

/** The links of the tab row of this page (pages.SITE_TABS), for the phone menu. */
const links = [...document.querySelectorAll<HTMLAnchorElement>('nav.tabs a')];
/** The page marked as current, read again at every opening: the control pages switch tabs without a reload (tabs.ts). */
const currentHref = ref('');
/** The pages of the phone menu: the current one marked, a dot on those with work running (composables/busy.ts). */
const pages = computed(() => links.map((link) => ({href: link.getAttribute('href') ?? '/', label: link.textContent ?? '',
  current: link.getAttribute('href') === currentHref.value, busy: linkBusy(link, busyPaths.value)})));
function readCurrent(): void {
  currentHref.value = links.find((link) => link.getAttribute('aria-current') === 'page')?.getAttribute('href') ?? '';
}
const open = ref(false);
/** A phone: the menu is the drawer (shared.css, max-width 560px). */
const narrow = window.matchMedia('(max-width: 560px)');
const phone = ref(narrow.matches);
narrow.addEventListener('change', () => { phone.value = narrow.matches; });
const state = ref<AuthState | null>(null);
const error = ref('');
const theme = ref<Theme>(storedTheme());
const button = useTemplateRef<HTMLButtonElement>('button');
const pop = useTemplateRef<HTMLDivElement>('pop');
onThemeChange(() => { theme.value = storedTheme(); });

async function show(): Promise<void> {
  readCurrent();
  open.value = true;
  await nextTick();
  pop.value?.querySelector<HTMLElement>('[aria-checked="true"]')?.focus();
  try {
    state.value = await api<AuthState>('/api/auth/state');
  } catch (problem) {
    error.value = messageOf(problem);
  }
}

function close(refocus: boolean): void {
  if (!open.value) return;
  open.value = false;
  if (refocus) button.value?.focus();
}

async function signOut(everywhere: boolean): Promise<void> {
  if (everywhere) {
    // The menu closes first: the dialog gives the focus back to the Account button.
    close(true);
    const ok = await askConfirm({title: 'Sign out everywhere?', action: 'Sign out everywhere', danger: true,
      text: 'Every browser signed in to this site, this one included, will have to sign in again.'});
    if (!ok) return;
  }
  try {
    await post(everywhere ? '/api/auth/logout-all' : '/api/auth/logout');
  } catch {
    // Already signed out.
  }
  location.replace('/');
}

function onDocumentClick(event: MouseEvent): void {
  const target = event.target as Node;
  if (!pop.value?.contains(target) && !button.value?.contains(target)) close(false);
}
function onKey(event: KeyboardEvent): void {
  if (event.key === 'Escape' && open.value) {
    event.stopPropagation();
    close(true);
  }
}
onMounted(() => {
  document.addEventListener('click', onDocumentClick);
  document.addEventListener('keydown', onKey, true);
});
onBeforeUnmount(() => {
  document.removeEventListener('click', onDocumentClick);
  document.removeEventListener('keydown', onKey, true);
});

// The arrows move the Appearance choice, as in any group of radio buttons; they never reach the page behind.
const THEME_STEPS: Record<string, number> = {ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1};
function themeKey(event: KeyboardEvent): void {
  const step = THEME_STEPS[event.key];
  if (step === undefined) return;
  event.preventDefault();
  event.stopPropagation();
  const index = THEMES.findIndex(([value]) => value === theme.value);
  const next = THEMES[(index + step + THEMES.length) % THEMES.length];
  if (next === undefined) return;
  setTheme(next[0]);
  theme.value = next[0];
  void nextTick(() => pop.value?.querySelector<HTMLElement>('[aria-checked="true"]')?.focus());
}

const sessionsText = (value: AuthState): string =>
  (value.sessions ?? 1) > 1 ? 'Signed in on ' + value.sessions + ' browsers' : 'Signed in on this browser';
</script>

<template>
  <button ref="button" class="icon-btn account-btn" :class="{current: props.current}" type="button" id="account-btn"
          aria-haspopup="dialog" :aria-expanded="open" aria-controls="account-pop" aria-label="Menu, account and settings"
          title="Account" @click="open ? close(false) : show()">
    <svg v-if="pages.length" class="i-menu" viewBox="0 0 20 20" aria-hidden="true">
      <path d="M3.5 6h13M3.5 10h13M3.5 14h13" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
    </svg>
    <svg class="i-person" :class="{alone: !pages.length}" viewBox="0 0 20 20" aria-hidden="true">
      <circle cx="10" cy="7" r="3.4" fill="none" stroke="currentColor" stroke-width="1.8"/>
      <path d="M3.6 17.2c.9-3.1 3.4-4.8 6.4-4.8s5.5 1.7 6.4 4.8" fill="none" stroke="currentColor" stroke-width="1.8"
            stroke-linecap="round"/>
    </svg>
  </button>
  <!-- On a phone the drawer leaves the top bar (its blur would hold a fixed panel inside the bar). -->
  <Teleport to="body" :disabled="!phone">
  <div v-if="open" class="pop-backdrop" aria-hidden="true" @click="close(false)"></div>
  <div v-show="open" ref="pop" class="account-pop" id="account-pop" role="dialog" aria-label="Account and settings">
    <button class="pop-close" type="button" aria-label="Close the menu" @click="close(true)">
      <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5L5 15" fill="none" stroke="currentColor" stroke-width="1.8"
        stroke-linecap="round"/></svg></button>
    <nav v-if="pages.length" class="pop-nav" aria-label="Pages">
      <!-- A tap on a page closes the menu: the control pages switch tabs without a reload (tabs.ts). -->
      <a v-for="page in pages" :key="page.href" class="pop-page" :href="page.href" :aria-current="page.current ? 'page' : undefined"
         :data-busy="page.busy ? 'true' : undefined" @click="close(false)">
        {{ page.label }}<span aria-hidden="true">›</span></a>
    </nav>
    <div class="pop-head">
      <span class="pop-avatar" aria-hidden="true">{{ (state?.user ?? '?').slice(0, 1).toUpperCase() }}</span>
      <div class="pop-who">
        <template v-if="state === null">
          <strong>Account</strong><span>{{ error || 'Loading…' }}</span>
        </template>
        <template v-else-if="state.local">
          <strong>This computer</strong><span>Site on 127.0.0.1, no sign-in</span>
        </template>
        <template v-else>
          <strong>{{ state.user }}</strong><span>{{ state.role === 'reviewer' ? 'Reviewer' : sessionsText(state) }}</span>
        </template>
      </div>
    </div>
    <div class="pop-section">
      <div class="pop-label">Appearance</div>
      <div class="segmented" role="radiogroup" aria-label="Appearance" @keydown="themeKey">
        <button v-for="[value, label] in THEMES" :key="value" type="button" role="radio" :tabindex="theme === value ? 0 : -1"
                :aria-checked="theme === value" @click="setTheme(value)">{{ label }}</button>
      </div>
    </div>
    <div class="pop-section pop-items">
      <a v-if="!guest" class="pop-item" href="/account" id="account-settings">Settings</a>
      <a v-if="state?.sign_out" class="pop-item" :href="state.sign_out">Sign out</a>
      <template v-else-if="state !== null && !state.local">
        <button class="pop-item" type="button" @click="signOut(false)">Sign out</button>
        <button class="pop-item danger" type="button" @click="signOut(true)">Sign out everywhere</button>
      </template>
    </div>
  </div>
  </Teleport>
</template>
