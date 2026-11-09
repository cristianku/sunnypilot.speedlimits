// The review at a glance, from the routes of the status: how many frames are confirmed, the routes in
// progress and to start, and the route "Resume review" opens (the one in progress, else the next to start).
import type {PanelStatus, RouteRow} from '../common/panel-types';
import {signsHref} from '../common/sign-link';

export interface ReviewSummary {
  /** Frames of the reports, and how many of them are confirmed. */
  frames: number;
  confirmed: number;
  percent: number;
  /** Routes still to review (state 'revisionare'): with some confirmed frames, and with none yet. */
  inProgress: number;
  toStart: number;
  /** Routes with a report, every frame confirmed. */
  reviewed: number;
  /** The route Resume review opens, or null when nothing is left to review. */
  resume: {route: string; href: string; left: number; started: boolean} | null;
}

/** The signs of a drive to review: the Signs side of its Speed review, which opens on its first frame still to review. */
export const reviewHref = (route: string): string => signsHref(route);

export function reviewSummary(status: PanelStatus | null): ReviewSummary {
  const routes = status?.routes ?? [];
  const reported = routes.filter((row) => row.run_id);
  const frames = reported.reduce((total, row) => total + row.report_frames, 0);
  const confirmed = reported.reduce((total, row) => total + Math.min(row.confirmed, row.report_frames), 0);
  const toReview = routes.filter((row) => row.state === 'revisionare');
  const inProgress = toReview.filter((row) => row.confirmed > 0);
  const toStart = toReview.filter((row) => row.confirmed === 0);
  const next: RouteRow | undefined = inProgress[0] ?? toStart[0];
  return {
    frames, confirmed, percent: frames ? confirmed * 100 / frames : 0,
    inProgress: inProgress.length, toStart: toStart.length, reviewed: reported.length - toReview.length,
    resume: next === undefined ? null : {route: next.route, href: reviewHref(next.route),
      left: Math.max(0, next.report_frames - next.confirmed), started: next.confirmed > 0},
  };
}
