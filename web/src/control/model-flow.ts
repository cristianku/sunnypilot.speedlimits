// Models: a network as the car runs it, as a flow of blocks (ModelFlow.vue, drawn by FlowChart): what it gets, the network,
// what it says and where that goes. Only the car's side: the heads only the training has are dropped at the export and are
// not drawn (Train › Brain draws the next training, brain-flow.ts). The facts are those of the ONNX contracts the comma
// reads (speed_limits.json of speed-limits-models) and of the version's result.json (model_roles.brain_digest).
import type {BrainModel, ModelVersion} from '../common/panel-types';
import {BRAIN_GROUPS, BRAIN_PICTURE} from './brain-card';
import type {FlowBlock, FlowKind} from './brain-flow';

export const MODEL_COLUMNS = ['What it gets', 'The network', 'What it says'];
export const MODEL_LEGEND: {kind: FlowKind; label: string}[] = [{kind: 'input', label: 'Input'}, {kind: 'car', label: 'Output'}];

const SIGNNET: FlowBlock[] = [
  {id: 'band', title: 'Road camera, top band', sub: '896 × 288', kind: 'input', column: 0,
    facts: [{name: 'images', what: 'the road frame scaled to 896 wide, its top 288 rows, RGB 0..1, a few times a second'}],
    to: ['signnet']},
  {id: 'signnet', title: 'SignNet', sub: 'stride 8', kind: 'net', column: 1,
    facts: [{name: 'out', what: 'a map of 112 × 36 cells, one every 8 pixels: heat, log width, log height, offset x and y'}],
    to: ['boxes']},
  {id: 'boxes', title: 'Boxes of signs', sub: 'best 5, score 0.3 or more', kind: 'car', column: 2,
    facts: [{name: 'box', what: 'a sign at each local peak of heat: its centre and size in the frame, and its score'}],
    note: 'Each box goes to the classifier. It must miss nothing: the classifier drops the false ones.'},
];

const CLASSIFIER: FlowBlock[] = [
  {id: 'crop', title: 'Each box, cut out', sub: '160 × 160', kind: 'input', column: 0,
    facts: [{name: 'images', what: 'the box of SignNet cut from the road frame, ImageNet normalised; with Read Signs at Full '
      + 'Resolution, from the full frame of camerad'}], to: ['classifier']},
  {id: 'classifier', title: 'MobileNetV3-small', sub: 'one box at a time', kind: 'net', column: 1,
    facts: [{name: 'classes', what: 'one score for each class it knows'}], to: ['class']},
  {id: 'class', title: 'What the box is', sub: 'a speed · end · zone · display · other · not a sign', kind: 'car', column: 2,
    facts: [{name: 'maximum_speed:N', what: 'a limit, 10 to 120; maximum_speed_hours:N one with its hours'},
      {name: 'cancellation', what: 'the end of a limit'},
      {name: 'zone:30 · road_marking:30 · variable_display:N', what: 'a zone, a number on the road, an LED display'},
      {name: 'other_sign:…', what: 'another sign by its type (give way, town entry, motorway…)'},
      {name: 'not_a_sign · unreadable', what: 'a false box of SignNet, a sign it cannot read'}],
    note: 'The signs read go into the brain (its input signs), never straight to the limit.'},
];

const DECIDER: FlowBlock[] = [
  {id: 'sign', title: 'The sign', sub: 'crop 96 · its box', kind: 'input', column: 0,
    facts: [{name: 'crop', what: 'the sign cut out, 96 × 96'}, {name: 'box_geometry', what: 'where the box is in the frame and its size'}],
    to: ['decider']},
  {id: 'scene', title: 'The road and the car', sub: 'frame 384 × 216 · speed', kind: 'input', column: 0,
    facts: [{name: 'scene', what: 'the whole frame, 384 × 216'}, {name: 'car_speed', what: 'the car’s speed'}], to: ['decider']},
  {id: 'decider', title: 'MobileNetV3-small', sub: 'sign and scene together', kind: 'net', column: 1,
    facts: [{name: 'context_full', what: 'one network reads the crop, the box and the scene'}], to: ['own', 'reading']},
  {id: 'own', title: 'Whose sign', sub: 'own road · other road · can’t tell', kind: 'car', column: 2,
    facts: [{name: 'ownership', what: 'own_road, other_road, unresolved'}],
    note: 'It runs only on this server: it says whom a sign is for in Prepare and in the signs the brain learns from. '
      + 'It never goes to the car.'},
  {id: 'reading', title: 'The sign read', sub: 'its class', kind: 'car', column: 2,
    facts: [{name: 'classes', what: 'the same classes as the classifier'}]},
];

const group = (name: string) => {
  const found = BRAIN_GROUPS.find((item) => item.name === name);
  return {name: name + (found ? ' · ' + found.size : ''), what: found?.what ?? name};
};

/** The brain as the car steps it: its inputs are the groups it was trained with. */
function brainBlocks(version: BrainModel): FlowBlock[] {
  const groups = version.groups ?? [];
  const has = (...names: string[]) => names.filter((name) => groups.includes(name));
  // A server not restarted since 8 Nov 2026 sends neither: a brain of before, 1 Hz, a class every 5 km/h.
  const kph = version.kph ?? {from: 5, to: 200, step: 5};
  const rate = version.rate_hz ?? 1;
  const every = rate === 1 ? 'once a second' : rate + ' times a second';
  const inputs: FlowBlock[] = [
    {id: 'camera', title: 'Road camera', sub: '256 × 144', kind: 'input', column: 0, facts: [BRAIN_PICTURE[0]!], to: ['features']},
  ];
  const sets: [string, string, string[]][] = [
    ['signs', 'Signs read', has('signs', 'plate', 'plate_hours', 'plate_wet')],
    ['place', 'The car and the place', has('hour', 'lead', 'country', 'sign_distance')],
    ['driver', 'Your pedals and speed', has('pedals', 'speed')],
  ];
  for (const [id, title, names] of sets) {
    if (!names.length) continue;
    inputs.push({id, title, sub: names.join(' · '), kind: 'input', column: 0, facts: names.map(group), to: ['memory'],
      ...(id === 'signs' ? {note: 'What SignNet and the classifier read on this car, mistakes included.'
        + (version.plates ? '' : ' This version did not learn the plates: the car gives it plate unknown and no hours.')} : {})});
  }
  const outputs: FlowBlock[] = [
    {id: 'speed', title: 'Speed to hold', sub: `${kph.from ?? '?'}–${kph.to ?? '?'} km/h, every ${kph.step ?? '?'}` + (version.ahead ? '' : ' · no next limit'), kind: 'car',
      column: 2, facts: [{name: 'speed', what: 'one score for each speed; the car takes the highest'}],
      note: 'It is the vision limit, speedLimit of speedVisionSP, for Smartspeed.' + (version.ahead ? '' : ' This version says '
        + 'nothing else: no next limit and no metres to it (it was trained before those heads, 8 Nov 2026), and with a brain the '
        + 'car no longer runs the rules’ LimitAhead, so speedLimitAhead stays 0.')},
  ];
  if (version.ahead) {
    outputs.push(
      {id: 'next', title: 'Next limit', sub: 'or none', kind: 'car', column: 2,
        facts: [{name: 'next', what: 'the limit of the sign ahead, speedLimitAhead'}]},
      {id: 'metres', title: 'Metres to it', sub: 'counted down between steps', kind: 'car', column: 2,
        facts: [{name: 'metres', what: 'speedLimitAheadDistance, less the metres driven since the step'}]},
    );
  }
  return [
    ...inputs,
    {id: 'features', title: 'MobileNetV3-small', sub: 'picture → features', kind: 'net', column: 1,
      facts: [{name: 'features', what: 'what it sees in the picture'}], to: ['memory']},
    {id: 'memory', title: 'Memory (GRU)', sub: `h ${version.hidden ?? 128}, one step ${every}`, kind: 'net', column: 1,
      facts: [BRAIN_PICTURE[1]!], note: 'The car gives h back at the next step (hidden_out → hidden).',
      to: outputs.map((block) => block.id)},
    ...outputs,
  ];
}

/** The blocks of a version as the car (or for applicability, this server) runs it; null for one with no diagram. */
export function modelBlocks(version: ModelVersion): FlowBlock[] | null {
  if (version.role === 'brain') return brainBlocks(version);
  if (version.role === 'cutter') return version.kind === 'signnet' ? SIGNNET : null;
  if (version.role === 'classifier') return CLASSIFIER;
  if (version.role === 'decider') return DECIDER;
  return null;
}
