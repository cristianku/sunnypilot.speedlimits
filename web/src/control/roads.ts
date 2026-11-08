// The kinds of road with their words (the drive player shows the scene network's).
import type {RoadType} from '../common/panel-types';

/** The kinds of road in the order of the legend, each with its colour (control.html .rt-town ...). */
export const ROAD_TYPES: [key: RoadType | 'unknown', label: string][] = [['town', 'Town'], ['zone30', 'Zone 30'],
  ['rural', 'Rural'], ['expressway', 'Expressway'], ['motorway', 'Motorway'], ['ramp', 'Ramp'], ['merge', 'Merge lane'],
  ['parking', 'Parking'], ['garage', 'Garage'], ['unknown', 'Unknown']];
export const ROAD_LABEL = Object.fromEntries(ROAD_TYPES) as Record<RoadType | 'unknown', string>;
