// The Mapillary switch of the cutter's next training, kept once for Train › Cutter and Train › Pipeline (the same
// choice in both): our frames only, the Mapillary windows added to train, or the windows alone first then ours
// (scripts/run_signnet.sh, NNSLR_CUTTER_MAPILLARY=1|pretrain), with the places and the share of their photos; and the
// cut of those windows (helpers/mapillary_windows.sh, the action mapillary_windows).
import {computed, reactive, type Ref} from 'vue';
import {thousands} from '../common/format';
import type {PanelStatus} from '../common/panel-types';
import {jobOf, terminalActivities} from './jobs';
import {useMapillaryChoice} from './mapillary-choices';

export function useCutterChoices(status: Ref<PanelStatus | null>) {
  /** Whether the windows are cut (status.cutter_options.mapillary), and the speed signs of their photos by place. */
  const choice = useMapillaryChoice(status, computed(() => status.value?.cutter_options?.mapillary ?? null),
    computed(() => status.value?.cutter_options?.mapillary_regions ?? null));
  const {mapillary, mapillarySigns, mapillaryOptions} = choice;
  /** The cut of the windows: its job from here, or the command started in a terminal. */
  const cutJob = computed(() => jobOf(status.value, ['mapillary_windows']));
  const cutting = computed(() => cutJob.value?.state === 'running'
    || terminalActivities(status.value, ['mapillary']).some((activity) => activity.label === 'Mapillary windows'));
  /** What it learns from, in words: after "A new SignNet, the cutter the comma runs, for every sign". */
  const words = computed(() => mapillary.value === 'on'
    ? ', with up to ' + thousands(mapillarySigns.value) + ' speed signs of Mapillary in train'
    : mapillary.value === 'pretrain' ? ', first on the Mapillary windows alone (about ' + thousands(mapillarySigns.value)
      + ' speed signs), then on our drives from there' : '');
  /** About how long it takes on the GPU: the windows are frames more, and in two stages a training more. */
  const minutes = computed(() => mapillary.value === 'on' ? 'About two hours' : mapillary.value === 'pretrain'
    ? 'About three hours' : 'About 80 minutes');
  return reactive({...choice, cutJob, cutting, options: mapillaryOptions, words, minutes});
}
export type CutterChoices = ReturnType<typeof useCutterChoices>;
