// What a box can be (families, values, roads) and what the second reading says, with the English words the
// page shows for them (the same as the dataset section of the Train tab, nnslr_tools.panel.FAMILY_LABELS).
import type {Detection, Hint} from './types';

export interface Family {
  id: string;
  label: string;
  hasValue: boolean;
  /** A shorter name in the list of boxes, where the number and the road tag must fit beside it. */
  brief?: string;
  /** Shown smaller, under the usual choices. */
  minor: boolean;
}

// The first five are the usual choices; the minor ones are shown smaller under them.
export const FAMILIES: readonly Family[] = [
  {id: 'maximum_speed', label: 'Speed limit', hasValue: true, minor: false},
  // The crossed number of an end of limit goes in the notes: the box has no value (review rules).
  {id: 'cancellation', label: 'End of limit', hasValue: false, minor: false},
  {id: 'unreadable', label: 'Unreadable', hasValue: false, minor: false},
  {id: 'variable_display', label: 'Variable speed limit', brief: 'Variable limit', hasValue: true, minor: false},
  {id: 'zone', label: 'Zone', hasValue: true, minor: false},
  {id: 'road_marking_candidate', label: 'Road marking', hasValue: true, minor: true},
  {id: 'other_sign', label: 'Other sign', hasValue: false, minor: true},
  {id: 'not_a_sign', label: 'Not a sign', hasValue: false, minor: true},
];
export const QUICK_VALUES: readonly number[] = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 120];
/** The limits of a US speed sign (MUTCD R2-1), in mph: every 5 from 15 to 80. */
export const US_VALUES: readonly number[] = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80];

/** How the speed signs look where a frame was taken: 'us', the white SPEED LIMIT rectangle in mph; 'disc', the red
 * ring of km/h everywhere else. */
export type SignStyle = 'us' | 'disc';
export function signStyleOf(country: string | null): SignStyle {
  return country === 'US' ? 'us' : 'disc';
}
/** The one kind of a "Not a sign" box (nnslr_tools.sign_types.NOT_A_SIGN_TYPES): a sign-like disc or panel on a
 * vehicle, such as the top-speed sticker on the back of a truck. */
export const ON_VEHICLE = 'on_vehicle';

/** Drops a kind the box's family cannot have: one left from an "Other sign" (a direction, say) whose number was then
 * typed, which made it a speed limit. The server refuses such a box, so it goes before every save too. */
export function dropStaleKind(d: Detection): void {
  if (d.family !== 'other_sign') delete d.painted;
  if (d.family !== 'maximum_speed' || d.supplementary_panel !== 'time') delete d.plate_hours;
  if (d.sign_type == null) return;
  if (d.family === 'other_sign' || (d.family === 'not_a_sign' && d.sign_type === ON_VEHICLE)) return;
  delete d.sign_type;
}

export interface SignKind {
  id: string;
  /** In the list of the boxes: "Danger: children". */
  label: string;
  /** On its button, under the name of its group: "Children". */
  chip: string;
  /** On the photo. */
  short: string;
}

// What an "Other sign" is (sign_type; the same ids as nnslr_tools.sign_types.SIGN_TYPES), in groups of buttons
// under "Type". Some give a limit without a number (town entry, motorway), some carry numbers that are no speed
// (weight, height): the model has to know them apart.
export const SIGN_KIND_GROUPS: readonly {label: string; kinds: readonly SignKind[]}[] = [
  {label: 'Danger', kinds: [
    {id: 'pedestrian_crossing', label: 'Pedestrian crossing', chip: 'Pedestrian crossing', short: 'crossing'},
    {id: 'children', label: 'Danger: children', chip: 'Children', short: 'children'},
    {id: 'cyclists', label: 'Danger: cyclists', chip: 'Cyclists', short: 'cyclists'},
    {id: 'traffic_queue', label: 'Danger: traffic queue', chip: 'Traffic queue', short: 'queue'},
    {id: 'roadworks', label: 'Roadworks', chip: 'Roadworks', short: 'works'},
    {id: 'danger_other', label: 'Other danger', chip: 'Other danger', short: 'danger'},
  ]},
  {label: 'Priority', kinds: [
    {id: 'give_way', label: 'Give way', chip: 'Give way', short: 'give way'},
    {id: 'stop', label: 'Stop', chip: 'Stop', short: 'stop'},
    {id: 'priority_to_right', label: 'Junction: priority to the right', chip: 'Priority to the right', short: 'right first'},
    {id: 'main_road', label: 'Main road', chip: 'Main road', short: 'main road'},
    {id: 'main_road_end', label: 'End of main road', chip: 'End of main road', short: 'main road end'},
  ]},
  {label: 'Bans and other numbers', kinds: [
    {id: 'no_overtaking', label: 'No overtaking', chip: 'No overtaking', short: 'no overtaking'},
    {id: 'no_overtaking_end', label: 'End of no overtaking', chip: 'End of no overtaking', short: 'overtaking end'},
    {id: 'no_entry', label: 'No entry', chip: 'No entry', short: 'no entry'},
    {id: 'no_vehicles', label: 'No vehicles (red ring, white disc)', chip: 'No vehicles', short: 'no vehicles'},
    {id: 'minimum_speed', label: 'Minimum speed (blue)', chip: 'Minimum speed', short: 'minimum'},
    {id: 'weight_limit', label: 'Weight limit', chip: 'Weight', short: 'weight'},
    {id: 'height_limit', label: 'Height limit', chip: 'Height', short: 'height'},
    {id: 'width_limit', label: 'Width limit', chip: 'Width', short: 'width'},
  ]},
  {label: 'Mandatory', kinds: [
    {id: 'keep_right', label: 'Keep right', chip: 'Keep right ↘', short: 'keep right'},
    {id: 'keep_left', label: 'Keep left', chip: '↙ Keep left', short: 'keep left'},
    {id: 'ahead_only', label: 'Straight ahead only', chip: '↑ Straight ahead', short: 'ahead'},
  ]},
  {label: 'Roads and places', kinds: [
    {id: 'town_entry', label: 'Town entry', chip: 'Town entry', short: 'town'},
    {id: 'town_exit', label: 'Town exit', chip: 'Town exit', short: 'town end'},
    {id: 'motorway_start', label: 'Motorway', chip: 'Motorway', short: 'motorway'},
    {id: 'motorway_end', label: 'End of motorway', chip: 'End of motorway', short: 'motorway end'},
    {id: 'expressway_start', label: 'Expressway', chip: 'Expressway', short: 'expressway'},
    {id: 'expressway_end', label: 'End of expressway', chip: 'End of expressway', short: 'expressway end'},
    {id: 'roundabout', label: 'Roundabout', chip: 'Roundabout', short: 'roundabout'},
    {id: 'one_way', label: 'One way', chip: 'One way', short: 'one way'},
  ]},
  {label: 'Directions', kinds: [
    {id: 'direction_left', label: 'Direction ←', chip: '← Left', short: '←'},
    {id: 'direction_right', label: 'Direction →', chip: 'Right →', short: '→'},
    {id: 'direction_straight', label: 'Direction ↑', chip: '↑ Straight', short: '↑'},
    {id: 'direction_several', label: 'Directions, several', chip: 'Several', short: 'directions'},
    {id: 'motorway_direction', label: 'Motorway direction (green)', chip: 'Motorway direction', short: 'motorway dir.'},
    {id: 'motorway_exit', label: 'Motorway exit (gore arrow)', chip: 'Motorway exit', short: 'exit'},
  ]},
  // By what is lit in this picture; a flashing amber is amber on one frame and off on the next.
  {label: 'Traffic light', kinds: [
    {id: 'traffic_light_red', label: 'Traffic light: red (also red with amber)', chip: 'Red', short: 'light red'},
    {id: 'traffic_light_amber', label: 'Traffic light: amber', chip: 'Amber', short: 'light amber'},
    {id: 'traffic_light_green', label: 'Traffic light: green', chip: 'Green', short: 'light green'},
    {id: 'traffic_light_off', label: 'Traffic light: off (no lamp lit)', chip: 'Off', short: 'light off'},
  ]},
  {label: 'Other', kinds: [{id: 'other', label: 'Something else', chip: 'Something else', short: 'other'}]},
];

export function signKindOf(id: string | null | undefined): SignKind | null {
  if (!id) return null;
  for (const group of SIGN_KIND_GROUPS) {
    const found = group.kinds.find((kind) => kind.id === id);
    if (found) return found;
  }
  return null;
}
export const ROADS: readonly {id: string; label: string}[] = [
  {id: 'own_road', label: 'Ours'},
  {id: 'other_road', label: 'Another'},
  {id: 'unresolved', label: 'Unclear'},
];

// The panel under the sign, in one tap. Hours: a limit valid only at some hours, such as the German 22-6. Wet: a limit
// valid only on a wet road, such as the German "bei Nässe".
export const PANELS: readonly {id: string; label: string}[] = [
  {id: 'none', label: 'None'},
  {id: 'time', label: 'Hours'},
  {id: 'wet', label: 'Wet'},
  {id: 'present', label: 'Other'},
];

/** A box around a sign and its plate (of hours, or for wet roads): taller than the sign on purpose, so no fit may bring it
 * to the disc. */
export function holdsPlate(d: Detection): boolean {
  return d.supplementary_panel === 'time' || d.supplementary_panel === 'wet';
}

/** The whole hours a plate can say, from and to (0 is also the 24 of "20-24"). */
export const PLATE_HOURS: readonly number[] = Array.from({length: 24}, (_, hour) => hour);

// The selects and the text field of "Sign details" (their ids are the field names).
export const SELECT_FIELDS = ['visibility', 'occlusion', 'supplementary_panel'] as const;
export const BOOLEAN_FIELDS = ['temporary', 'variable_display'] as const;
export type SelectField = typeof SELECT_FIELDS[number];
export type BooleanField = typeof BOOLEAN_FIELDS[number];

interface HintType {
  id: string;
  label: string;
  short: string;
}
const UNSURE: HintType = {id: 'unsure', label: 'Unsure', short: '?'};
const HINT_TYPES: readonly HintType[] = [
  {id: 'maximum_speed', label: 'Speed limit', short: 'limit'},
  {id: 'cancellation', label: 'End of limit', short: 'end'},
  {id: 'zone', label: 'Zone', short: 'zone'},
  {id: 'zone_end', label: 'End of zone', short: 'zone end'},
  {id: 'variable_display', label: 'Variable speed limit', short: 'variable'},
  {id: 'road_marking', label: 'Road marking', short: 'road'},
  UNSURE,
];
const CONFIDENCES: ReadonlyMap<string, string> = new Map([['high', 'sure'], ['medium', 'medium'], ['low', 'low']]);

/** A speed limit with its number: limit, zone, variable display. */
export function isSpeedFamily(family: string): boolean {
  return family === 'maximum_speed' || family === 'zone' || family === 'variable_display';
}

export function familyOf(id: string): Family | null {
  return FAMILIES.find((family) => family.id === id) ?? null;
}

export function familyHasValue(id: string): boolean {
  return familyOf(id)?.hasValue ?? false;
}

export function hasValue(d: {value_kph?: number | null | undefined}): boolean {
  return d.value_kph !== null && d.value_kph !== undefined;
}

/** The full description in the panel: "Speed limit · 60 km/h", "Variable limit · 80 km/h", "Town entry". */
export function describe(d: Detection): string {
  const kind = d.family === 'other_sign' ? signKindOf(d.sign_type) : null;
  const painted = d.family === 'other_sign' && d.painted === true ? ', painted on the road' : '';
  if (kind !== null) return kind.label + painted;
  if (painted) return 'Other sign' + painted;
  if (d.family === 'not_a_sign' && d.sign_type === ON_VEHICLE) return 'Not a sign: on a vehicle';
  const family = familyOf(d.family);
  let text = family ? family.brief ?? family.label : String(d.family || 'No type');
  if (hasValue(d)) text += ' · ' + d.value_kph + (d.sign_unit === 'mph' || d.sign_unit === 'km/h' ? ' ' + d.sign_unit : ' (unit not recorded)');
  if (d.plate_hours) text += ' · hours ' + d.plate_hours;
  if (d.supplementary_panel === 'wet') text += ' · when wet';
  return text;
}

// Short label on the photo: the list in the panel has the full description.
export function shortName(d: Detection): string {
  const known = hasValue(d);
  const v = known && d.sign_unit === 'mph' ? d.value_kph + ' mph' : d.value_kph;
  switch (d.family) {
    case 'maximum_speed': return known ? String(v) : 'limit ?';
    case 'zone': return known ? 'zone ' + v : 'zone';
    case 'variable_display': return known ? 'variable ' + v : 'variable';
    case 'road_marking_candidate': return known ? 'road ' + v : 'road';
    case 'cancellation': return known ? 'end ' + v : 'end';
    case 'other_sign': return (d.painted === true ? 'painted ' : '') + (signKindOf(d.sign_type)?.short ?? 'other');
    case 'not_a_sign': return d.sign_type === ON_VEHICLE ? 'on a vehicle' : 'no';
    default: return '?';
  }
}

/** How the id of a box the car's networks found starts (car.ts). */
export const CAR_BOX = 'car-';

/** The colour of a box on the photo (overlay tokens of shared.css). */
export function boxColor(d: Detection): string {
  if (d.detection_id?.startsWith(CAR_BOX)) return 'var(--ov-car)';
  if (isSpeedFamily(d.family)) return 'var(--ov-speed)';
  if (d.family === 'cancellation') return 'var(--ov-cancel)';
  if (d.family === 'not_a_sign') return 'var(--ov-none)';
  return 'var(--ov-other)';
}

export function hintTypeOf(id: string): HintType {
  return HINT_TYPES.find((type) => type.id === id) ?? UNSURE;
}

export function confidenceLabel(id: string): string {
  return CONFIDENCES.get(id) ?? String(id);
}

export function sideLabel(side: string): string {
  if (side === 'left') return 'on the left';
  if (side === 'right') return 'on the right';
  if (side === 'center' || side === 'centre') return 'in the middle';
  if (side === 'overhead') return 'over the road';
  return side;
}

// Short label drawn next to a hint box, e.g. "60?": its colour says it is the second opinion.
export function hintLabel(h: Hint): string {
  const type = hintTypeOf(h.type);
  let what = type.short;
  if (hasValue(h)) {
    what = h.type === 'maximum_speed' || h.type === 'unsure' ? String(h.value_kph) : type.short + ' ' + h.value_kph;
  } else if (h.type === 'unsure') {
    what = 'unsure';
  }
  return what + '?';
}

// The family a hint becomes when the reviewer presses "Use".
export function familyForHint(type: string): string {
  switch (type) {
    case 'maximum_speed': return 'maximum_speed';
    case 'cancellation': return 'cancellation';
    case 'zone': return 'zone';
    case 'zone_end': return 'cancellation';
    case 'variable_display': return 'variable_display';
    case 'road_marking': return 'road_marking_candidate';
    default: return 'unreadable';
  }
}
