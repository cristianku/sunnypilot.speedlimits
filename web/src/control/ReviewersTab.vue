<script setup lang="ts">
// Data › Reviewers: who reviews from outside and on what. The people are the user names of the sign-in in front of the site
// (they join this list by themselves at their first visit, or are added here first); a tap on a kind of work of a drive
// gives it to the person shown, a second tap takes it back, at once. A person sees only what is given here: the server
// checks it on every request (nnslr_tools.review_access).
import {computed, onMounted, ref, shallowRef} from 'vue';
import {api, messageOf, post} from '../common/api';
import {plural} from '../common/format';
import {askConfirm} from '../composables/confirm';
import {toast} from '../composables/toast';
import type {AssignableDrive, Person, ReviewersView} from './reviewers';
import SectionHead from './SectionHead.vue';
import Segmented from './Segmented.vue';

const view = shallowRef<ReviewersView | null>(null);
const error = ref('');
const shown = ref<string | null>(null);
const newName = ref('');
const saving = ref(false);
type Filter = 'all' | 'given';
const filter = ref<Filter>('all');
const FILTERS: readonly {key: Filter; label: string}[] = [{key: 'all', label: 'All drives'}, {key: 'given', label: 'Given'}];

async function load(): Promise<void> {
  try {
    view.value = await api<ReviewersView>('/api/reviewers');
    error.value = '';
    if (shown.value === null || !view.value.people.some((person) => person.name === shown.value)) {
      shown.value = view.value.people[0]?.name ?? null;
    }
  } catch (problem) {
    error.value = messageOf(problem);
  }
}
onMounted(load);

const canEdit = computed(() => view.value?.can_edit === true);
const given = computed(() => shown.value === null ? {} : view.value?.work[shown.value] ?? {});
const givenCount = computed(() => Object.keys(given.value).length);
const drives = computed(() => (view.value?.drives ?? []).filter((drive) => filter.value === 'all' || given.value[drive.route]));
const hasKind = (drive: AssignableDrive, key: string): boolean => (given.value[drive.route] ?? []).includes(key);
const workOf = (person: Person): number => Object.keys(view.value?.work[person.name] ?? {}).length;
/** What the person did last, in words: the first visit is what proves the name is the right one. */
const seenText = (person: Person | undefined): string => {
  if (!person) return '';
  return person.last_seen ? 'Last seen ' + person.last_seen.slice(0, 16).replace('T', ' ') + ' UTC' : 'Not seen yet';
};
const person = computed(() => view.value?.people.find((item) => item.name === shown.value));

/** A change on the server, then its answer: the page shows what the server holds. */
async function change(address: string, body: Record<string, unknown>): Promise<boolean> {
  saving.value = true;
  try {
    const answer = await post<Pick<ReviewersView, 'people' | 'work'>>(address, body);
    if (view.value) view.value = {...view.value, people: answer.people, work: answer.work};
    return true;
  } catch (problem) {
    toast(messageOf(problem), true);
    return false;
  } finally {
    saving.value = false;
  }
}
async function add(): Promise<void> {
  const name = newName.value.trim();
  if (!name) return;
  if (await change('/api/reviewers/people', {name})) {
    newName.value = '';
    shown.value = name;
  }
}
async function remove(): Promise<void> {
  const name = shown.value;
  if (name === null) return;
  const ok = await askConfirm({title: 'Remove ' + name + '?', action: 'Remove', danger: true,
    text: 'All the work given to this person is taken back at once. What they already saved stays. Their account is closed in the sign-in, not here.'});
  if (!ok) return;
  if (await change('/api/reviewers/people/remove', {name})) shown.value = view.value?.people[0]?.name ?? null;
}
function toggle(drive: AssignableDrive, key: string): void {
  if (shown.value === null) return;
  void change('/api/reviewers/assign', {user: shown.value, route: drive.route, kind: key, on: !hasKind(drive, key)});
}
</script>

<template>
  <SectionHead tab="reviewers"/>
  <p v-if="error" class="banner">{{ error }}</p>
  <p v-else-if="!view" class="empty">Loading…</p>
  <template v-else>
    <p v-if="!view.open" class="banner" id="rw-closed">The sign-in from outside is not switched on yet: you can prepare the people and their
      work, nobody can come in.</p>
    <section class="card rw-people" aria-labelledby="rw-people-title">
      <div class="ds-head">
        <h2 id="rw-people-title">People</h2>
        <span class="ds-when">{{ plural(view.people.length, 'person', 'people') }}</span>
      </div>
      <p class="note">People from outside work only on what you give them here, and never see the place of a drive, the
        training or the car. What they save goes into the dataset under their name, as a person's review.</p>
      <p v-if="!view.people.length" class="empty">Nobody yet: add the user name of their account.</p>
      <div v-else class="rw-names" role="radiogroup" aria-label="The person shown">
        <button v-for="item in view.people" :key="item.name" type="button" role="radio" class="rw-name" :aria-checked="item.name === shown"
                @click="shown = item.name">{{ item.name }}<small>{{ plural(workOf(item), 'drive') }}</small></button>
      </div>
      <form v-if="canEdit" class="rw-add" @submit.prevent="add">
        <input v-model="newName" type="text" id="rw-new-name" maxlength="64" autocomplete="off" autocapitalize="none" spellcheck="false"
               placeholder="User name" aria-label="User name of the new person">
        <button type="submit" class="btn" :disabled="saving || !newName.trim()">Add</button>
      </form>
    </section>

    <section v-if="shown !== null" class="card rw-work" aria-labelledby="rw-work-title">
      <div class="ds-head">
        <h2 id="rw-work-title">Work of {{ shown }}</h2>
        <span class="ds-when">{{ plural(givenCount, 'drive') }} given · {{ seenText(person) }}</span>
      </div>
      <div class="rw-tools">
        <Segmented v-model="filter" :options="FILTERS" label="Which drives"/>
        <button v-if="canEdit" type="button" class="btn small danger" id="rw-remove" :disabled="saving" @click="remove">Remove {{ shown }}</button>
      </div>
      <p v-if="!drives.length" class="empty">{{ filter === 'given' ? 'Nothing given yet: tap a kind of work under All drives.' : 'No drive to give yet.' }}</p>
      <ul v-else class="rw-list" id="rw-drives">
        <li v-for="drive in drives" :key="drive.route" class="rw-drive">
          <div class="rw-drive-name">
            <strong>{{ drive.title ?? 'Places unknown' }}</strong>
            <span>{{ [drive.when, drive.minutes === null ? null : plural(drive.minutes, 'minute')].filter(Boolean).join(' · ') }}</span>
          </div>
          <div class="rw-kinds">
            <button v-for="kind in view.kinds" :key="kind.key" type="button" class="rw-kind" :aria-pressed="hasKind(drive, kind.key)"
                    :disabled="!canEdit || saving || !drive.kinds.includes(kind.key) && !hasKind(drive, kind.key)"
                    :title="drive.kinds.includes(kind.key) ? undefined : 'This drive has nothing for this work yet'"
                    @click="toggle(drive, kind.key)">{{ kind.label }}</button>
          </div>
        </li>
      </ul>
    </section>
  </template>
</template>
