// The Mapillary switch of a training, the same for the classifier (its crops, helpers/mapillary.sh) and for the cutter
// (its windows, helpers/mapillary_windows.sh): our signs only, with Mapillary in train, or Mapillary first then ours;
// the places the Mapillary photos were taken (the positions of helpers/mapillary_locations.sh) and a share of them.
import {computed, ref, type Ref, type UnwrapNestedRefs} from 'vue';
import type {ActionOptions, MapillaryRegions, PanelStatus} from '../common/panel-types';
import {jobOf, terminalActivities} from './jobs';
import {regionCrops, shareOf} from './mapillary';

export type MapillaryChoice = 'off' | 'on' | 'pretrain';
export const MAPILLARY_OPTIONS: {key: MapillaryChoice; label: string}[] = [{key: 'off', label: 'Our signs'},
  {key: 'on', label: 'With Mapillary'}, {key: 'pretrain', label: 'Mapillary, then ours'}];

/** The switch, its region and its share: state says whether the Mapillary signs are ready (off until they are),
 * regions their signs by place. The refs are spread into the reactive choices of the training. */
export function useMapillaryChoice<State extends {available: boolean}>(status: Readonly<Ref<PanelStatus | null>>,
  state: Readonly<Ref<State | null>>, regions: Readonly<Ref<MapillaryRegions | null>>) {
  const picked = ref<MapillaryChoice | null>(null);
  const mapillary = computed({
    get: (): MapillaryChoice => state.value?.available ? picked.value ?? 'off' : 'off',
    set: (value: MapillaryChoice) => { picked.value = value; },
  });
  const withMapillary = computed(() => mapillary.value !== 'off');
  /** The places chosen (none: all the world) and the share, in percent. */
  const region = ref<string[]>([]);
  const shareText = ref('100');
  const share = computed(() => shareOf(shareText.value));
  const locateJob = computed(() => jobOf(status.value, ['mapillary_locations']));
  const locating = computed(() => locateJob.value?.state === 'running'
    || terminalActivities(status.value, ['mapillary']).some((activity) => activity.label === 'Mapillary positions'));
  /** About how many Mapillary signs join: those of the places chosen, times the share. */
  const mapillarySigns = computed(() => Math.round(regionCrops(regions.value, region.value) * (share.value ?? 0) / 100));
  const mapillaryOptions = computed((): ActionOptions => withMapillary.value
    ? {mapillary: mapillary.value, region: region.value.length ? region.value.join(',') : 'all', share: String(share.value ?? shareText.value)}
    : {mapillary: 'off'});
  /** Why it cannot start with these choices ('' when it can): a share that is not a percentage. */
  const invalid = computed(() => withMapillary.value && share.value === null
    ? 'the share of Mapillary is a percentage, more than 0 and at most 100' : '');
  return {mapillaryState: state, mapillary, withMapillary, mapillaryRegions: regions, region, shareText, share, locateJob,
    locating, mapillarySigns, mapillaryOptions, invalid};
}
/** The switch as MapillaryChoices.vue shows it: in the reactive choices of the classifier or of the cutter. */
export type MapillarySwitch = UnwrapNestedRefs<ReturnType<typeof useMapillaryChoice<{available: boolean}>>>;
