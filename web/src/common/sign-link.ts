// Where the signs of a drive are reviewed: the Signs side of its Speed review (control/SpeedReviewDrive.vue), the one
// review page since the page of Sign review was retired (9 Nov 2026; its old address /route/<id>/ leads there too,
// pages.signs_address says the same on the server). At a moment, second `second` of segment `segment` (the player
// seeks there, control/DriveReviewDrive.vue seekLinked), and on a frame of Sign review, its key "<segment>/<image>" (the
// editor opens that very frame, also one between the seconds).

/** A moment of a drive as the links carry it: second N of a segment's video. */
export interface SignMoment {
  segment: number;
  second: number;
}

/** The Signs side of Speed review of a drive, at a moment and on a frame when given. */
export function signsHref(route: string, at: SignMoment | null = null, frame: string | null = null): string {
  let address = '/speed-review?drive=' + encodeURIComponent(route) + '&side=signs';
  if (at !== null) address += '&segment=' + at.segment + '&second=' + Math.max(0, Math.round(at.second * 1000) / 1000);
  if (frame !== null) address += '&frame=' + encodeURIComponent(frame).replace('%2F', '/');
  return address;
}

/** What the address of the page asks of the Signs side: the side itself (side=signs) and a frame (frame=<key>). */
export function askedSigns(search: string = window.location.search): {side: boolean; frame: string | null} {
  const query = new URLSearchParams(search);
  const frame = query.get('frame');
  return {side: query.get('side') === 'signs', frame: frame !== null && /^[0-9]{1,9}\/[0-9]{1,9}$/.test(frame) ? frame : null};
}
