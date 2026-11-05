// Who proposes the signs of a preparation, in words, wherever a preparation starts or shows (Prepare, the Overview's
// Drives card), and the networks in use (its Car card): the proposer's name and, for Our networks, the cutter,
// classifier and decider it runs, by their run.
import type {Proposer} from '../common/panel-types';

export const PROPOSER_LABELS: Record<Proposer, string> = {vietnam: 'Vietnamese YOLO', ours: 'Our networks', qwen: 'Qwen3-VL'};
const PART_NAMES: Record<string, string> = {cutter: 'cutter', classifier: 'classifier', decider: 'applicability', road_type: 'scene',
  brain: 'brain'};
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** One network in use by its id ("cutter:2026-10-30_1046-cutter/signnet"): "cutter SignNet, 30 Oct 10:46". The time
 * is the one of its run's name, as the server wrote it. */
export function networkText(part: string, id: string): string {
  const rest = id.includes(':') ? id.slice(id.indexOf(':') + 1) : id;
  const [run = rest, ...path] = rest.split('/');
  const time = /^(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})/.exec(run);
  const when = time ? Number(time[3]) + ' ' + MONTHS[Number(time[2]) - 1] + ' ' + time[4] + ':' + time[5] : run;
  const last = path[path.length - 1] ?? '';
  const what = part === 'cutter' ? (last === 'signnet' ? 'SignNet' : last.replace('x', ' × '))
    : part === 'decider' && last ? last.replace('_', ' + ') : '';
  return (PART_NAMES[part] ?? part) + (what ? ' ' + what : '') + ', ' + when;
}

/** The networks of Our networks, one after the other: "cutter SignNet, 30 Oct 10:46 · classifier, 30 Oct 11:22 · …". */
export function networksText(networks: Record<string, string> | null | undefined): string {
  return Object.entries(networks ?? {}).map(([part, id]) => networkText(part, id)).join(' · ');
}

/** "Our networks: cutter SignNet, 30 Oct 10:46 · …", or the proposer's name alone. */
export function proposerText(proposer: Proposer, networks?: Record<string, string> | null): string {
  const parts = proposer === 'ours' ? networksText(networks) : '';
  return PROPOSER_LABELS[proposer] + (parts ? ': ' + parts : '');
}
