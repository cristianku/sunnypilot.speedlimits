// The choices of the classifier's next training, kept once for Train › Classifier and Train › Pipeline (the same
// choices in both): its signs (ours, with Mapillary in train, or Mapillary first then ours), the places and the share
// of the Mapillary signs, and where it starts (ImageNet or a finished classifier of Models).
import {computed, reactive, ref, type Ref} from 'vue';
import {isoWhen, thousands} from '../common/format';
import type {ActionOptions, ClassifierModel, ModelRole, PanelStatus} from '../common/panel-types';
import type {PickerOption} from '../components/PickerSheet.vue';
import {useMapillaryChoice} from './mapillary-choices';

export {MAPILLARY_OPTIONS, type MapillaryChoice} from './mapillary-choices';
type Split = 'same' | 'another' | 'unknown';
const SPLIT_WORDS: Record<Split, string> = {same: 'learnt on today’s split', another: 'learnt on another split',
  unknown: 'its split not recorded'};

export function useClassifierChoices(status: Ref<PanelStatus | null>,
  versionsOf: (role: ModelRole) => readonly {role: string}[] | undefined) {
  /** Whether the Mapillary crops are prepared (status.classifier_options), and which of them: the places their photos
   * were taken (none chosen: all the world) and a share of them, in percent (with Mapillary in train the server then
   * keeps at most 300 of each kind, its pretraining takes every one). */
  const choice = useMapillaryChoice(status, computed(() => status.value?.classifier_options?.mapillary ?? null),
    computed(() => status.value?.classifier_options?.mapillary_regions ?? null));
  const {mapillary, mapillarySigns, mapillaryOptions} = choice;
  /** Where the classifier starts: a finished classifier of Models with “not a sign”, whose weights it goes on learning
   * from (the one in use first, and the choice until another is made), or ImageNet (its test stays fair; the choice
   * while none is in use). Not with “Mapillary, then ours”, which starts from its own Mapillary stage. A start that
   * learnt another split may have seen some of today's test signs. */
  const startVersions = computed((): ClassifierModel[] => ((versionsOf('classifier') ?? []) as ClassifierModel[])
    .filter((version) => version.role === 'classifier' && version.part === 'classifier'
      && version.class_scheme !== undefined && version.finished)
    .sort((a, b) => Number(b.in_use) - Number(a.in_use)));
  const currentSplit = computed(() => status.value?.classifier_options?.split_sha256 ?? null);
  const startSplit = (version: ClassifierModel): Split => !version.split_sha256 || !currentSplit.value
    ? 'unknown' : version.split_sha256 === currentSplit.value ? 'same' : 'another';
  const startOptions = computed((): PickerOption[] => [
    {key: 'imagenet', title: 'ImageNet', sub: 'a fresh start: always a fair test'},
    ...startVersions.value.map((version) => ({key: version.id, title: version.label + ' ' + isoWhen(version.trained_at),
      short: version.label + ' ' + isoWhen(version.trained_at), badge: version.in_use ? 'In use' : undefined,
      sub: SPLIT_WORDS[startSplit(version)]})),
  ]);
  /** The start chosen here (null: none yet, the one in use). */
  const pickedStart = ref<string | null>(null);
  const startKey = computed({
    get: (): string => pickedStart.value === 'imagenet' || startVersions.value.some((version) => version.id === pickedStart.value)
      ? pickedStart.value! : startVersions.value.find((version) => version.in_use)?.id ?? 'imagenet',
    set: (value: string) => { pickedStart.value = value; },
  });
  const startVersion = computed(() => mapillary.value === 'pretrain' ? null
    : startVersions.value.find((version) => version.id === startKey.value) ?? null);
  /** The options of “Train the classifier” (ActionOptions). */
  const options = computed((): ActionOptions => mapillary.value === 'pretrain' ? mapillaryOptions.value
    : {...mapillaryOptions.value, start: startVersion.value?.id ?? 'imagenet'});
  /** What it learns from, in words: after "A new classifier, taught on the signs and on the cutter’s mistakes". */
  const words = computed(() => (mapillary.value === 'on' ? ', with about ' + thousands(mapillarySigns.value) + ' signs of Mapillary' : '')
    + (mapillary.value === 'pretrain' ? ', first on about ' + thousands(mapillarySigns.value) + ' signs of Mapillary alone, '
      + 'then on ours from there' : '')
    + (startVersion.value ? ', starting from ' + startVersion.value.label + ' (' + isoWhen(startVersion.value.trained_at) + ')' : ''));
  return reactive({...choice, startOptions, startKey, startVersion, startSplit, options, words});
}
export type ClassifierChoices = ReturnType<typeof useClassifierChoices>;
