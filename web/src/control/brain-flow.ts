// The blocks of a network drawn as a flow (FlowChart.vue): the network in use in Models (model-flow.ts). Train › Brain draws
// the next training on its own (BrainFlow.vue).
import type {BrainFact} from './brain-card';

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
