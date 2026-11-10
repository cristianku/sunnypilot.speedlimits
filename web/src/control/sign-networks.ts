// The networks that read the signs the next brain learns from (#31), kept once for Train › Brain and for its step of
// Train › Pipeline (the same choice in both): the cutter (a SignNet the comma runs), the classifier and the applicability
// network, which leaves out the signs of another road as the car does. Those in use until others are chosen; the same
// three read the brain's inputs (as the car, nnslr_tools.road_type_signs) and the next limit it learns
// (nnslr_tools.brain_signs), and its result records them (Models).
import {computed, reactive, ref} from 'vue';
import {isoWhen} from '../common/format';
import type {ActionOptions, ModelRole, ModelVersion} from '../common/panel-types';
import type {PickerOption} from '../components/PickerSheet.vue';

export type SignPart = 'cutter' | 'classifier' | 'decider';
export const SIGN_PARTS: {part: SignPart; label: string; option: 'sign_cutter' | 'sign_classifier' | 'sign_decider'}[] = [
  {part: 'cutter', label: 'Cutter', option: 'sign_cutter'},
  {part: 'classifier', label: 'Classifier', option: 'sign_classifier'},
  {part: 'decider', label: 'Applicability', option: 'sign_decider'},
];

export function useSignNetworks(versionsOf: (role: ModelRole) => readonly ModelVersion[] | undefined) {
  /** A finished version that can read the signs as the comma does: a SignNet of its band and grid, a classifier with
   * “not a sign” (not an old Reader), a whole applicability training (not a smoke run). */
  const usable = (version: ModelVersion): boolean => version.finished
    && (version.role !== 'cutter' || (version.kind === 'signnet' && version.fits_the_car === true))
    && (version.role !== 'classifier' || version.part === 'classifier')
    && (version.role !== 'decider' || !version.smoke_limit);
  /** Each part's versions, the one in use first. */
  const versions = (part: SignPart): ModelVersion[] => [...versionsOf(part) ?? []].filter(usable)
    .sort((a, b) => Number(b.in_use) - Number(a.in_use));
  /** "SignNet 896 × 288 29 Oct 14:02", "Applicability, Crop + geometry 10 Nov 01:11". */
  const titleOf = (version: ModelVersion): string => (version.role === 'decider' ? 'Applicability, ' + version.label
    : version.label) + ' ' + (version.trained_at ? isoWhen(version.trained_at) : version.run);
  const options = computed(() => Object.fromEntries(SIGN_PARTS.map(({part}) => [part, versions(part).map(
    (version): PickerOption => ({key: version.id, title: titleOf(version), short: titleOf(version),
      badge: version.in_use ? 'In use' : undefined, sub: version.run}))])) as Record<SignPart, PickerOption[]>);
  /** The choice made here (none yet: the one in use, else the newest). */
  const picked = ref<Partial<Record<SignPart, string>>>({});
  const keyOf = (part: SignPart): string => {
    const found = versions(part);
    return found.some((version) => version.id === picked.value[part]) ? picked.value[part]!
      : found.find((version) => version.in_use)?.id ?? found[0]?.id ?? '';
  };
  const keys = reactive({
    cutter: computed({get: () => keyOf('cutter'), set: (value: string) => { picked.value = {...picked.value, cutter: value}; }}),
    classifier: computed({get: () => keyOf('classifier'), set: (value: string) => { picked.value = {...picked.value, classifier: value}; }}),
    decider: computed({get: () => keyOf('decider'), set: (value: string) => { picked.value = {...picked.value, decider: value}; }}),
  });
  const versionOf = (part: SignPart): ModelVersion | null => versions(part).find((version) => version.id === keys[part]) ?? null;
  /** The parts with none to choose: the training is refused without them (no applicability, no brain). */
  const missing = computed(() => SIGN_PARTS.filter(({part}) => versionOf(part) === null).map(({label}) => label.toLowerCase()));
  /** The options of its training, all three named (a network put in use meanwhile changes nothing). */
  const option = computed((): ActionOptions => Object.fromEntries(SIGN_PARTS.flatMap(({part, option: key}) => {
    const version = versionOf(part);
    return version ? [[key, version.id]] : [];
  })));
  /** The choice in words, after the training's own: ", its signs read with SignNet 896 × 288 29 Oct 14:02 (in use), ...". */
  const words = computed(() => {
    const named = SIGN_PARTS.flatMap(({part}) => {
      const version = versionOf(part);
      return version ? [titleOf(version) + (version.in_use ? ' (in use)' : '')] : [];
    });
    return named.length ? ', its signs read with ' + named.join(', ') : '';
  });
  /** The line under the choice. */
  const note = computed((): {text: string; warn: boolean} => missing.value.length
    ? {text: 'No ' + missing.value.join(', no ') + ' to read the signs with: train one, or the brain cannot start.', warn: true}
    : {text: 'The signs the brain learns from are read as the car reads them: the applicability network leaves out those of '
      + 'another road (an exit, a side road). The same three for its inputs and for the next limit; a drive not read by '
      + 'these three is read again first.', warn: false});
  return reactive({options, keys, missing, option, words, note});
}
export type SignNetworksChoice = ReturnType<typeof useSignNetworks>;
