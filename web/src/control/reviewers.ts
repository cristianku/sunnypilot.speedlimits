// The answers of GET /api/reviewers (Data › Reviewers, the administrator) and GET /api/reviewer/work (My reviews, a
// reviewer from outside). The rules are on the server (nnslr_tools.review_access).

/** Who may be given work: a user name of the sign-in in front of the site. */
export interface Person {
  name: string;
  added: string | null;
  last_seen: string | null;
}
export interface WorkKind {
  key: string;
  label: string;
}
/** A drive and the kinds of work it can give today. */
export interface AssignableDrive {
  route: string;
  title: string | null;
  when: string | null;
  minutes: number | null;
  kinds: string[];
}
/** What came back from a drive on the review machine. */
export interface BoxSaved {
  frames: number;
  stores: number;
  conflicts: number;
  refused: number;
}
export interface BoxDrive {
  route: string;
  sent_at?: string;
  left_at?: string;
  saved?: BoxSaved;
  leaving_at?: string;
  removing?: boolean;
  problem?: string;
  completed?: {person: string; kind: string; at: string}[];
}
/** The machine of the outside reviewers (nnslr_tools.review_box.view). */
export interface BoxView {
  linked: boolean;
  last: {at: string; ok: boolean; seconds: number; problems: string[]} | null;
  drives: BoxDrive[];
  finished: BoxDrive[];
}
export interface ReviewersView {
  /** The sign-in from outside is set on this server: without it the people are ready, nobody comes in. */
  open: boolean;
  can_edit: boolean;
  kinds: WorkKind[];
  people: Person[];
  /** By person, by drive: the kinds given. */
  work: Record<string, Record<string, string[]>>;
  drives: AssignableDrive[];
  box: BoxView;
}

/** A kind of work on a drive of a reviewer, with the page that opens it. */
export interface GivenWork {
  kind: string;
  label: string;
  href: string;
}
export interface GivenDrive {
  route: string;
  when: string | null;
  minutes: number | null;
  /** The person said it is finished: the work goes away at the next copy back. */
  completed: boolean;
  work: GivenWork[];
}
export interface MyWork {
  user: string | null;
  read_only: boolean;
  /** On the review machine only: the list offers "Review completed". */
  can_complete: boolean;
  drives: GivenDrive[];
}
