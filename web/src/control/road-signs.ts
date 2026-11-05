// How the next training of the scene network reads the signs of the drives, kept once for Train › Scene and for its step
// of Train › Pipeline (the same choice in both): with the car's classifier (the one of the sunnypilot clone, the
// default) or with a finished classifier with “not a sign” of Models before it is on the car (one that reads the green
// boards towards a motorway). Its board input, the seconds since a green board was confirmed (never a limit), has no
// choice here: the server gives it when that classifier read the boards on enough drives.
import {computed, reactive, ref} from 'vue';
import {isoWhen} from '../common/format';
import type {ActionOptions, ClassifierModel, ModelRole} from '../common/panel-types';
import type {PickerOption} from '../components/PickerSheet.vue';

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
  /** The option of its training (ActionOptions.classifier). */
  const option = computed((): ActionOptions => ({classifier: classifier.value}));
  /** What it reads the signs with, in words, after the training's own: "", or ", reading the signs with Classifier …". */
  const words = computed(() => version.value ? ', reading the signs with ' + titleOf(version.value) : '');
  return reactive({classifierOptions, classifier, version, option, words});
}
export type RoadSignsChoice = ReturnType<typeof useRoadSigns>;
