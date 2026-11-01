// How the next training of the scene network reads the signs of the drives, kept once for Train › Scene and for its step
// of Train › Pipeline (the same choice in both): with the car's classifier (the one of the sunnypilot clone, the
// default) or with a finished classifier with “not a sign” of Models before it is on the car (one that reads the green
// boards towards a motorway); and its board input, the seconds since a green board was confirmed (never a limit): with
// it when that classifier read the boards on enough drives, always (refused when it read too few), or never.
import {computed, reactive, ref} from 'vue';
import {isoWhen} from '../common/format';
import type {ActionOptions, ClassifierModel, ModelRole} from '../common/panel-types';
import type {PickerOption} from '../components/PickerSheet.vue';

export type BoardsChoice = 'auto' | 'on' | 'off';
export const BOARDS_OPTIONS: {key: BoardsChoice; label: string}[] = [{key: 'auto', label: 'When read'},
  {key: 'on', label: 'Always'}, {key: 'off', label: 'Never'}];
export const CAR_CLASSIFIER = 'car';

export function useRoadSigns(versionsOf: (role: ModelRole) => readonly {role: string}[] | undefined) {
  const versions = computed((): ClassifierModel[] => ((versionsOf('classifier') ?? []) as ClassifierModel[])
    .filter((version) => version.role === 'classifier' && version.part === 'classifier'
      && version.class_scheme !== undefined && version.finished));
  const titleOf = (version: ClassifierModel): string => version.label + ' ' + isoWhen(version.trained_at);
  const classifierOptions = computed((): PickerOption[] => [
    {key: CAR_CLASSIFIER, title: 'The car’s classifier', sub: 'the signs the comma confirms today'},
    ...versions.value.map((version) => ({key: version.id, title: titleOf(version), short: titleOf(version),
      badge: version.in_use ? 'In use' : undefined, sub: 'the drives are read again with it, once'})),
  ]);
  const picked = ref<string>(CAR_CLASSIFIER);
  const classifier = computed({
    get: (): string => versions.value.some((version) => version.id === picked.value) ? picked.value : CAR_CLASSIFIER,
    set: (value: string) => { picked.value = value; },
  });
  const version = computed(() => versions.value.find((item) => item.id === classifier.value) ?? null);
  const boards = ref<BoardsChoice>('auto');
  /** A line under the choices: what the board input is, and when it is on. */
  const note = computed(() => 'Green boards: the seconds since the classifier confirmed a green board towards a motorway, '
    + 'an input of the scene network and never a limit. '
    + (boards.value === 'auto' ? 'On when it read them on 3 drives or more.'
      : boards.value === 'on' ? 'Refused when it read them on fewer than 3 drives.' : 'Off: the scene network as before.'));
  /** The options of its training (ActionOptions.classifier, ActionOptions.boards). */
  const option = computed((): ActionOptions => ({classifier: classifier.value, boards: boards.value}));
  /** What it reads the signs with, in words, after the training's own: "", or ", reading the signs with Classifier …". */
  const words = computed(() => (version.value ? ', reading the signs with ' + titleOf(version.value) : '')
    + (boards.value === 'on' ? ', with the green boards' : boards.value === 'off' ? ', without the green boards' : ''));
  return reactive({classifierOptions, classifier, version, boards, note, option, words});
}
export type RoadSignsChoice = ReturnType<typeof useRoadSigns>;
