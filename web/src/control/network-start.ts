// Where the next training of the cutter, of applicability or of the brain starts, kept once for its section and for its
// step of Train › Pipeline (the same choice in both): a finished version of its role in Models, whose weights it goes on
// from (the one in use, until another is chosen), or its usual start (random weights for SignNet, ImageNet for
// applicability and the brain; the choice while none is in use). An applicability run is one choice: each of its
// networks goes on from its own of that run. A brain is one seed: its three seeds go on from it; it learnt on the
// brain's frozen split, and the server refuses one stepped otherwise (nnslr_tools.brain_train.brain_start). One that
// took your pedals and the car's speed is a start too: the new one leaves those inputs out.
import {computed, reactive, ref, type Ref} from 'vue';
import {isoWhen, plural} from '../common/format';
import type {ActionOptions, ModelRole, ModelVersion, PanelStatus} from '../common/panel-types';
import type {PickerOption} from '../components/PickerSheet.vue';

export type StartRole = 'cutter' | 'decider' | 'brain';
type Split = 'same' | 'another' | 'unknown';
const SPLIT_WORDS: Record<Split, string> = {same: 'learnt on today’s split', another: 'learnt on another split',
  unknown: 'its split not recorded'};
/** The usual start of each, first in the list: its test stays fair. */
const FRESH: Record<StartRole, string> = {cutter: 'Random weights', decider: 'ImageNet', brain: 'ImageNet'};
const NAME: Record<StartRole, string> = {cutter: 'SignNet', decider: 'Applicability', brain: 'Brain'};
const POOL_WORDS = {grid: 'a grid', mean: 'a mean'} as const;
const POOL_TITLE = {grid: 'Grid', mean: 'Mean'} as const;
export const FRESH_START = 'fresh';

export function useNetworkStart(role: StartRole, status: Ref<PanelStatus | null>,
  versionsOf: (role: ModelRole) => readonly ModelVersion[] | undefined, pool?: () => string) {
  /** A finished version that can be a start: a SignNet the comma runs (its band and grid), a whole applicability
   * training (not a smoke run), a brain stepped as today's (four times a second, its memory's size recorded: not one of
   * before 8 Nov 2026). */
  const startable = (version: ModelVersion): boolean => version.role === role && version.finished
    && (version.role !== 'cutter' || (version.kind === 'signnet' && version.fits_the_car === true))
    && (version.role !== 'decider' || !version.smoke_limit)
    && (version.role !== 'brain' || (version.hidden ?? null) !== null);
  /** One choice per run (the one in use there, else its newest), the run in use first, then the newest. */
  const starts = computed(() => {
    const byRun = new Map<string, ModelVersion[]>();
    for (const version of versionsOf(role) ?? []) {
      if (startable(version)) byRun.set(version.run, [...byRun.get(version.run) ?? [], version]);
    }
    return [...byRun.values()].map((items) => ({version: items.find((item) => item.in_use) ?? items[0]!,
      count: items.length, inUse: items.some((item) => item.in_use)}))
      .sort((a, b) => Number(b.inUse) - Number(a.inUse));
  });
  /** Its kind and when: "SignNet 896 × 288 29 Oct 14:02", "Brain Grid, seed 1 10 Nov 01:11". */
  const titleOf = (version: ModelVersion): string => (role === 'decider' ? NAME[role]
    : version.role === 'brain' ? 'Brain ' + POOL_TITLE[version.pool ?? 'grid'] + ', seed ' + (version.seed ?? '?')
      : version.label) + ' ' + (version.trained_at ? isoWhen(version.trained_at) : version.run);
  /** A brain that took your pedals and the car's speed too (model_roles.DRIVER_GROUPS). */
  const takesTheDriver = (version: ModelVersion): boolean => version.role === 'brain'
    && (version.groups ?? []).some((group) => group === 'pedals' || group === 'speed');
  /** Whether it learnt on the split the new one is measured on. */
  const currentSplit = computed(() => status.value?.classifier_options?.split_sha256 ?? null);
  const splitOf = (version: ModelVersion): Split | null => {
    if (version.role !== 'cutter' && version.role !== 'decider') return null;
    return !version.split_sha256 || !currentSplit.value ? 'unknown' : version.split_sha256 === currentSplit.value ? 'same' : 'another';
  };
  const options = computed((): PickerOption[] => [
    {key: FRESH_START, title: FRESH[role], sub: 'a fresh start: always a fair test'},
    ...starts.value.map(({version, count, inUse}) => {
      const split = splitOf(version);
      const off = version.role === 'brain' ? version.test?.mean_error_kph ?? null : null;
      return {key: version.id, title: titleOf(version), short: titleOf(version), badge: inUse ? 'In use' : undefined,
        sub: [split ? SPLIT_WORDS[split] : '', role === 'decider' ? plural(count, 'network') : '',
          takesTheDriver(version) ? 'with your pedals and speed' : '',
          off === null ? '' : off.toFixed(1) + ' km/h off on the test'].filter(Boolean).join(' · ')};
    }),
  ]);
  /** The choice made here (null: none yet, the one in use). */
  const picked = ref<string | null>(null);
  const key = computed({
    get: (): string => picked.value === FRESH_START || starts.value.some((item) => item.version.id === picked.value)
      ? picked.value! : starts.value.find((item) => item.inUse)?.version.id ?? FRESH_START,
    set: (value: string) => { picked.value = value; },
  });
  const version = computed(() => starts.value.find((item) => item.version.id === key.value)?.version ?? null);
  /** A line under the choice: what it goes on from, and whether the new one's test stays fair. */
  const note = computed((): {text: string; warn: boolean} | null => {
    if (version.value === null) return null;
    if (version.value.role === 'brain') {
      const theirs = version.value.pool;
      if (theirs && pool && pool() !== theirs) return {text: 'That brain sees the picture as ' + POOL_WORDS[theirs] + ': choose '
        + (theirs === 'grid' ? 'Grid' : 'Mean') + ' in Picture, or the training stops at its first check.', warn: true};
      return {text: 'Its three seeds go on from that brain’s weights, learnt on the brain’s frozen split: the test stays fair. '
        + 'An input it did not have starts at zero' + (takesTheDriver(version.value)
          ? '; your pedals and the car’s speed, which it took, are left out.' : '.'), warn: false};
    }
    const split = splitOf(version.value);
    const what = role === 'cutter' ? 'It goes on from that SignNet’s weights'
      : role === 'decider' ? 'Each version goes on from its own of that run (one it lacks from ImageNet)'
        : 'Each network goes on from its own of that run (one it lacks from ImageNet)';
    if (split === 'another') return {text: 'Trained on another split: it may have seen some of today’s test frames, so the test '
      + 'of the new one is not fair.', warn: true};
    if (split === 'unknown') return {text: 'Its split is not recorded: it may have seen some of today’s test frames, so the test '
      + 'of the new one may not be fair.', warn: true};
    if (split === 'same') return {text: what + ', learnt on today’s split: the test stays fair.', warn: false};
    return {text: what + '. If that run was tested on other drives, Models says the new one’s test is not fair.', warn: false};
  });
  /** The option of its training (ActionOptions.start). */
  const option = computed((): ActionOptions => ({start: version.value?.id ?? FRESH_START}));
  /** What it starts from, in words, after the training's own: "", or ", going on from SignNet 896 × 288 29 Oct 14:02". */
  const words = computed(() => version.value ? ', going on from ' + titleOf(version.value) : '');
  return reactive({options, key, version, note, option, words});
}
export type NetworkStartChoice = ReturnType<typeof useNetworkStart>;
