// Train › Brain as a flow of blocks (BrainFlow.vue): only what the next training uses, by its switches (BrainChoices): what it
// gives the brain, the network, what it says, and the answers it is compared with. A block's detail is made of the facts of
// brain-card.ts, by the names of the code. The brain on the car is drawn in Models (model-flow.ts).
import {BRAIN_ANSWERS, BRAIN_GROUPS, BRAIN_OUT, BRAIN_OUTPUTS, BRAIN_PICTURE, type BrainFact} from './brain-card';

/** input: given in training; net: the network; car: an output of the car's network; train: only in training (a head that
 * only teaches h, and the answers). */
export type FlowKind = 'input' | 'net' | 'car' | 'train';

export interface FlowBlock {
  id: string;
  title: string;
  /** One short line under the title. */
  sub: string;
  kind: FlowKind;
  /** 0 what it gets, 1 the network, 2 what it says, 3 the answers of the training. */
  column: number;
  /** In columns 2 and 3, the row the block sits on (from 1) and how many it takes: an answer beside what it answers. */
  row?: number;
  rows?: number;
  facts: BrainFact[];
  /** A line under the facts. */
  note?: string;
  /** The blocks this one's arrows go to; with compares, a line without arrows to what it is compared with. */
  to?: string[];
  /** Only compared with the blocks of to (an answer of the training): nothing flows from it. */
  compares?: boolean;
}

/** What the next training gives and learns (BrainChoices): the driver's inputs, and the three heads only the training has. */
export interface FlowChoices {
  driver: boolean;
  works: boolean;
  road: boolean;
  types: boolean;
}

export const FLOW_COLUMNS = ['What it gets, 4 times a second', 'The network', 'What it says',
  'The answers it is compared with'];
/** The row of the line "only in training" in column 2, between the heads of the car and those only the training has. */
export const DIVIDER_ROW = 4;

const named = (facts: BrainFact[], name: string): BrainFact => facts.find((fact) => fact.name === name)!;
const groups = (...names: string[]): BrainFact[] => names.map((name) => {
  const group = BRAIN_GROUPS.find((item) => item.name === name)!;
  return {name: group.name + ' · ' + group.size, what: group.what};
});

/** The blocks of the next training: only what it gives and learns (BrainChoices). What it leaves out is not drawn: the
 * driver's inputs with Driver's inputs off, a head whose switch is off and its answer, openpilot engaged and the map, which no
 * training gives. The car's view of the brain in use is in Models (model-flow.ts). */
export function flowBlocks(learns: FlowChoices): FlowBlock[] {
  const onlyTraining = 'Only in training: it stays out of the car’s network and only teaches h.';
  const blocks: FlowBlock[] = [
    {id: 'camera', title: 'Road camera', sub: '256 × 144', kind: 'input', column: 0, facts: [named(BRAIN_PICTURE, 'picture')],
      to: ['features']},
    {id: 'signs', title: 'Signs it read', sub: 'signs · plate · hours · wet', kind: 'input', column: 0,
      facts: groups('signs', 'plate', 'plate_hours', 'plate_wet'),
      note: 'What our sign networks read on the video, as the car reads them, mistakes included: never the signs you confirmed '
        + 'in Sign review. A drive not read yet has them unknown throughout.', to: ['memory']},
    {id: 'place', title: 'The car and the place', sub: 'hour · car ahead · country · sign distance', kind: 'input', column: 0,
      facts: groups('hour', 'lead', 'country', 'sign_distance'), to: ['memory']},
  ];
  if (learns.driver) {
    blocks.push({id: 'driver', title: 'Your pedals and speed', sub: 'Driver’s inputs on', kind: 'input', column: 0,
      facts: groups('pedals', 'speed'),
      note: 'Off by default: the answer is your own speed, so with the car’s speed in, the network can learn to say it back. '
        + 'On here to compare.', to: ['memory']});
  }
  // The heads only the training has, each with its answer on its row, after the line "only in training".
  const helpers = [
    {on: learns.works, id: 'works', title: 'Roadworks', answer: 'Roadworks on the road', where: 'marked in Speed review'},
    {on: learns.road, id: 'road', title: 'Kind of road', answer: 'Kinds of road', where: 'marked in Speed review'},
    {on: learns.types, id: 'type', title: 'Road type', answer: 'Road types', where: 'the map’s, corrected in Speed review'},
  ].filter((helper) => helper.on);
  const heads = ['speed', 'next', 'metres', ...helpers.map((helper) => helper.id)];
  blocks.push(
    {id: 'features', title: 'MobileNetV3-small', sub: 'not trained: picture → features', kind: 'net', column: 1,
      facts: [{name: 'features', what: 'what the frozen MobileNetV3-small sees in the picture, the same on the car'}], to: ['memory']},
    {id: 'memory', title: 'Memory (GRU)', sub: 'h, one step every ¼ s', kind: 'net', column: 1, facts: [named(BRAIN_PICTURE, 'h')],
      note: 'Everything on the left goes into it at every step, and every head on the right reads h. In training each head is '
        + 'compared with its answer, and the error goes back through the head into h: nothing flows from an answer into the network.',
      to: heads},
    {id: 'speed', title: 'Speed to hold', sub: '1–200 km/h', kind: 'car', column: 2, row: 1, facts: [BRAIN_OUTPUTS[0]!]},
    {id: 'next', title: 'Next limit', sub: 'or none', kind: 'car', column: 2, row: 2, facts: [BRAIN_OUTPUTS[1]!]},
    {id: 'metres', title: 'Metres to it', sub: 'to the next sign', kind: 'car', column: 2, row: 3, facts: [BRAIN_OUTPUTS[2]!]},
    {id: 'answer-speed', title: 'Your speed', sub: 'every second you drove', kind: 'train', column: 3, row: 1,
      facts: [named(BRAIN_ANSWERS, 'speed')], note: 'Only the drives with sunnypilot off: dashcam drives.', to: ['speed'],
      compares: true},
    {id: 'answer-signs', title: 'Signs read', sub: 'our networks · your Sign review over them', kind: 'train', column: 3, row: 2,
      rows: 2, facts: [named(BRAIN_ANSWERS, 'next, metres')],
      note: 'Each training first reads with the networks in use every drive they have not read; where you confirmed a frame, '
        + 'or boxed a sign, yours wins. Their mistakes are answers too, fewer as they get better. Never an input: the car is given what its '
        + 'own reader reads.',
      to: ['next', 'metres'], compares: true},
  );
  helpers.forEach((helper, index) => {
    const row = DIVIDER_ROW + 1 + index;
    const fact = named(BRAIN_ANSWERS, helper.id + '_head');
    blocks.push(
      {id: helper.id, title: helper.title, sub: helper.id + '_head', kind: 'train', column: 2, row, facts: [fact], note: onlyTraining},
      {id: 'answer-' + helper.id, title: helper.answer, sub: helper.where, kind: 'train', column: 3, row,
        facts: [{name: fact.name, what: fact.what}], to: [helper.id], compares: true,
        ...(helper.id === 'type' ? {note: 'The map only teaches here, in training: the car never gets it.'} : {})},
    );
  });
  const out = BRAIN_OUT.filter((reason) => learns.types || reason.name !== 'no_road_type');
  blocks.push({id: 'out', title: 'Seconds that answer nothing', sub: out.length + ' reasons', kind: 'train', column: 3,
    row: helpers.length ? DIVIDER_ROW + 1 + helpers.length : DIVIDER_ROW, facts: out,
    note: 'They still go through the memory, as on the car.'});
  return blocks;
}
