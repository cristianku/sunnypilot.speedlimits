// Crossings: the vehicles the crossing rule proposed on the brain's drives (nnslr_tools.road_users), and a person's
// verdict on each, crosses my path or not. Only the verdicts teach the brain vehicles: the rule only proposes. The drives
// (GET /api/road-users/drives), a drive's events with their verdicts (GET /api/road-users/drives/<route>), a verdict
// (POST .../verdicts {event, verdict, base_revision}). The boxes are in the road frame's pixels (frame_size, 1344 x 760),
// one at each reading of the event's track (4 a second); a reading is frame `frame` of minute `segment` of the 20 fps
// video, found in the drive's MP4 at the start of its minute in the video's index plus frame / 20.
import type {VideoState} from './drive-review';

export const API = '/api/road-users';

export interface CrossingDrive {
  route: string;
  proposed: number;
  reviewed: number;
  crossing: number;
  made?: string | null;
  yaw?: string | null;
}

export interface Place {
  key: string;
  segment: number;
  frame: number;
  seconds: number;
  video_seconds: number | null;
}

export type VerdictName = 'crossing' | 'not_crossing';

export interface Verdict {
  verdict: VerdictName;
  reviewer: string;
  at: string;
}

export interface CrossingEvent {
  id: string;
  kind: string;
  class: string | null;
  /** The highest speed across our path the rule measured, in m/s (a scale: the class's typical height sets it). */
  score: number;
  from: Place;
  to: Place;
  boxes: (Place & {box: [number, number, number, number]})[];
  verdict: Verdict | null;
}

export interface CrossingView {
  route: string;
  revision: number;
  current: boolean;
  frame_size: [number, number];
  fps: number;
  readings_per_second: number;
  yaw?: string | null;
  made?: string | null;
  unjudged_readings: number;
  events: CrossingEvent[];
}

export function driveHref(route: string, event: string | null = null): string {
  return '/crossings?drive=' + encodeURIComponent(route) + (event ? '&event=' + encodeURIComponent(event) : '');
}

/** The seconds of a reading in the drive's MP4, from the video's own index; null for a minute the video has not. */
export function videoTime(index: VideoState['index'] | undefined, place: Pick<Place, 'segment' | 'frame'>): number | null {
  const minute = index?.segments.find((item) => item.segment === place.segment);
  return minute ? minute.start + place.frame / (index?.fps || 20) : null;
}

export const CLASS_WORDS: Record<string, string> = {car: 'Car', bus: 'Bus', truck: 'Truck', person: 'Person',
  bicycle: 'Bicycle', motorcycle: 'Motorcycle'};

/** "minute 3, 12.4 s" of a reading. */
export function placeText(place: Pick<Place, 'segment' | 'frame'>): string {
  return 'minute ' + place.segment + ', ' + (place.frame / 20).toFixed(1) + ' s';
}

/** The speed across in km/h, rounded to 5 (the rule's figure is only a scale). */
export function acrossText(mps: number): string {
  return '≈ ' + Math.max(5, Math.round(mps * 3.6 / 5) * 5) + ' km/h across';
}

export const toReview = (event: CrossingEvent): boolean => event.verdict === null;
