// Train › Brain as a flow of blocks (BrainFlow.vue): what the car gives the brain, the network, what it says on the car,
// and apart what only its training has. A block's detail is made of the facts of brain-card.ts, by the names of the code.
import {BRAIN_ANSWERS, BRAIN_GROUPS, BRAIN_OUT, BRAIN_OUTPUTS, BRAIN_PICTURE, type BrainFact} from './brain-card';

/** input and car: on the car; net: the network; train: only in training; off: never given. */
export type FlowKind = 'input' | 'net' | 'car' | 'train' | 'off';

export interface FlowBlock {
  id: string;
  title: string;
  /** One short line under the title. */
  sub: string;
  kind: FlowKind;
  /** 0 what it gets, 1 the network, 2 what it says, 3 the answers of the training. */
  column: number;
  facts: BrainFact[];
  /** A line under the facts. */
  note?: string;
  /** The blocks this one's arrows go to. */
  to?: string[];
}

export const FLOW_COLUMNS = ['What it gets, 4 times a second', 'The network', 'What it says', 'Only in training: the answers'];

const named = (facts: BrainFact[], name: string): BrainFact => facts.find((fact) => fact.name === name)!;
const groups = (...names: string[]): BrainFact[] => names.map((name) => {
  const group = BRAIN_GROUPS.find((item) => item.name === name)!;
  return {name: group.name + ' · ' + group.size, what: group.what};
});

/** The blocks; driver says whether the next training gives the driver's own inputs (Driver's inputs on). */
export function flowBlocks(driver: boolean): FlowBlock[] {
  return [
    {id: 'camera', title: 'Road camera', sub: '256 × 144', kind: 'input', column: 0, facts: [named(BRAIN_PICTURE, 'picture')],
      to: ['features']},
    {id: 'signs', title: 'Signs it read', sub: 'signs · plate · hours · wet', kind: 'input', column: 0,
      facts: groups('signs', 'plate', 'plate_hours', 'plate_wet'),
      note: 'What our sign networks read on the video, as the car reads them, mistakes included: never the signs you confirmed '
        + 'in Sign review. A drive not read yet has them unknown throughout.', to: ['memory']},
    {id: 'place', title: 'The car and the place', sub: 'hour · car ahead · country · sign distance', kind: 'input', column: 0,
      facts: groups('hour', 'lead', 'country', 'sign_distance'), to: ['memory']},
    {id: 'driver', title: 'Your pedals and speed', sub: driver ? 'given: Driver’s inputs on' : 'not given: Driver’s inputs off',
      kind: driver ? 'input' : 'off', column: 0, facts: groups('pedals', 'speed'),
      note: 'Off by default: with them the network could learn to follow the car instead of the road.', to: ['memory']},
    {id: 'never', title: 'Never given', sub: 'openpilot on/off · the map', kind: 'off', column: 0,
      facts: [...groups('engaged'), {name: 'the map', what: 'nothing of the map: its limit, its road types and its roads stay out'}],
      note: 'engaged is made on the car, and no brain learns it.'},
    {id: 'features', title: 'MobileNetV3-small', sub: 'not trained: picture → features', kind: 'net', column: 1,
      facts: [{name: 'features', what: 'what the frozen MobileNetV3-small sees in the picture, the same on the car'}], to: ['memory']},
    {id: 'memory', title: 'Memory (GRU)', sub: 'h, one step every ¼ s', kind: 'net', column: 1, facts: [named(BRAIN_PICTURE, 'h')],
      note: 'Everything above goes into it at every step; every head below reads h.',
      to: ['speed', 'next', 'metres', 'works', 'road', 'type']},
    {id: 'speed', title: 'Speed to hold', sub: '1–200 km/h', kind: 'car', column: 2, facts: [BRAIN_OUTPUTS[0]!]},
    {id: 'next', title: 'Next limit', sub: 'or none', kind: 'car', column: 2, facts: [BRAIN_OUTPUTS[1]!]},
    {id: 'metres', title: 'Metres to it', sub: 'to the next sign', kind: 'car', column: 2, facts: [BRAIN_OUTPUTS[2]!]},
    {id: 'works', title: 'Roadworks', sub: 'works_head', kind: 'train', column: 2, facts: [named(BRAIN_ANSWERS, 'works_head')],
      note: 'Only in training: it stays out of the car’s network and only teaches h.'},
    {id: 'road', title: 'Kind of road', sub: 'road_head', kind: 'train', column: 2, facts: [named(BRAIN_ANSWERS, 'road_head')],
      note: 'Only in training: it stays out of the car’s network and only teaches h.'},
    {id: 'type', title: 'Road type', sub: 'type_head', kind: 'train', column: 2, facts: [named(BRAIN_ANSWERS, 'type_head')],
      note: 'Only in training: it stays out of the car’s network and only teaches h.'},
    {id: 'answer-speed', title: 'Your speed', sub: 'every second you drove', kind: 'train', column: 3,
      facts: [named(BRAIN_ANSWERS, 'speed')], note: 'Only the drives with sunnypilot off: dashcam drives.', to: ['speed']},
    {id: 'answer-signs', title: 'Signs you confirmed', sub: 'Sign review', kind: 'train', column: 3,
      facts: [named(BRAIN_ANSWERS, 'next, metres')],
      note: 'Only where you went through the frames in Sign review: anywhere else next and metres answer nothing.',
      to: ['next', 'metres']},
    {id: 'answer-works', title: 'Your roadworks', sub: 'Speed review', kind: 'train', column: 3,
      facts: [{name: 'works_head', what: named(BRAIN_ANSWERS, 'works_head').what}], to: ['works']},
    {id: 'answer-road', title: 'Your kinds of road', sub: 'Speed review', kind: 'train', column: 3,
      facts: [{name: 'road_head', what: named(BRAIN_ANSWERS, 'road_head').what}], to: ['road']},
    {id: 'answer-type', title: 'Road types', sub: 'the map’s, with your changes', kind: 'train', column: 3,
      facts: [{name: 'type_head', what: named(BRAIN_ANSWERS, 'type_head').what}],
      note: 'The map only teaches here, in training: the car never gets it.', to: ['type']},
    {id: 'out', title: 'Seconds that answer nothing', sub: BRAIN_OUT.length + ' reasons', kind: 'train', column: 3, facts: BRAIN_OUT,
      note: 'They still go through the memory, as on the car.'},
  ];
}
