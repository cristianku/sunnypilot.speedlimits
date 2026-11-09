// Train › Brain: what the brain gets on the car and what only its training has, by the names of the code
// (sunnypilot/speed_limits/brain_inputs.py INPUT_SIZES, nnslr_tools.brain_net, brain_targets, brain_train). The groups are
// kept equal to INPUT_SIZES, with what each is for a training of today, by tests/review/test_brain_card.py.

/** A name of the code and what it is. */
export interface BrainFact {
  name: string;
  what: string;
}

/** One group of the frame's inputs (INPUT_SIZES): its size, and whether a training of today gives it to the network:
 * car (always, brain_train.ROAD_INPUTS), driver (only with Driver's inputs on, brain_targets.DRIVER_INPUTS), never (made on
 * the car, no brain learns it: brain_targets.LEFT_OUT). */
export interface BrainGroup extends BrainFact {
  size: number;
  use: 'car' | 'driver' | 'never';
}

/** What every step gets on the car beside the groups. */
export const BRAIN_PICTURE: BrainFact[] = [
  {name: 'picture', what: 'the road camera at 256 × 144 (vision.BrainInput), through MobileNetV3-small (not trained) to its features'},
  {name: 'h', what: 'the memory (GRU) of every step before; it starts again from zeros after 5 s without a frame'},
];

export const BRAIN_GROUPS: BrainGroup[] = [
  {name: 'signs', size: 11, use: 'car', what: 'the limit confirmed (vision.SpeedLimitMemory), from a display, a zone or a marking on the road, '
    + 'the age of its last reading, an end confirmed since, read in the last second; the last single reading and its age'},
  {name: 'plate', size: 4, use: 'car', what: 'the plate under that number: none, time, other, unknown'},
  {name: 'plate_hours', size: 5, use: 'car', what: 'the hours written on that plate, from and to'},
  {name: 'plate_wet', size: 1, use: 'car', what: 'that plate says “wet road only”'},
  {name: 'hour', size: 3, use: 'car', what: 'the local hour, on a circle'},
  {name: 'lead', size: 3, use: 'car', what: 'metres and speed of the car ahead (radarState.leadOne)'},
  {name: 'country', size: 7, use: 'car', what: 'the last country of the GPS: CH, DE, FR, IT, AT, US, none'},
  {name: 'sign_distance', size: 2, use: 'car', what: 'the metres to the round sign ahead (vision.SignDistance), from its box and vEgo'},
  {name: 'motion', size: 8, use: 'car', what: 'the car’s turn signals (on in the last second, seconds since last on, left and right), '
    + 'the steering angle and the yaw rate (carState, deviceMotion)'},
  {name: 'pedals', size: 3, use: 'driver', what: 'gas and brake pressed in the last second (carState)'},
  {name: 'speed', size: 2, use: 'driver', what: 'the car’s own speed, vEgo'},
  {name: 'engaged', size: 7, use: 'never', what: 'openpilot on or off (selfdriveState.enabled)'},
];

/** What it says on the car (the heads the car's network keeps). */
export const BRAIN_OUTPUTS: BrainFact[] = [
  {name: 'speed = speed_head(h) + trust(h) · number', what: 'the speed to hold, a class from 1 to 200 km/h; number is the limit confirmed of signs'},
  {name: 'next = next_head(h) + trust_next(h) · number', what: 'the next limit, or none'},
  {name: 'metres = metres_head(h) + metres_gate(h) · sign', what: 'the metres to it; sign is sign_distance'},
];

/** The answers its training compares with what it says: never inputs, none of them reaches the car. */
export const BRAIN_ANSWERS: BrainFact[] = [
  {name: 'speed', what: 'your dashboard speed, driver_speed (vEgoCluster, or vEgo + the car’s offset), on every second of the dashcam '
    + 'drives; the kph of your blocks of Speed review where you set one'},
  {name: 'next, metres', what: 'from the signs our networks in use (cutter, classifier, applicability) read on every picture of the '
    + 'drive, yours of Sign review over them where you confirmed the frames; passed where Sign distance reaches zero'},
  {name: 'works_head', what: 'the roadworks on the road, where they are marked in Speed review (Roadworks on)'},
  {name: 'road_head', what: 'the kinds of road (narrow, country, gravel, potholes…), where they are marked in Speed review (Kinds of road on)'},
  {name: 'type_head', what: 'the road type of the map (Read the map in Data › Prepare), corrected in Speed review where it is wrong (Road types on)'},
];

/** The seconds that answer nothing (brain_targets.OUT_REASONS): they still go through the memory, as on the car. */
export const BRAIN_OUT: BrainFact[] = [
  {name: 'no_speed', what: 'the logs have no speed'},
  {name: 'stopped', what: 'below 1 km/h'},
  {name: 'cruise', what: 'openpilot or the car’s cruise held the speed (never on a dashcam drive)'},
  {name: 'left_out', what: 'a block of yours with no speed'},
  {name: 'boundary', what: '2 s around the jump at the edge of a block'},
  {name: 'no_frame', what: 'no picture'},
  {name: 'no_road_type', what: 'no road type, with Road types on'},
];
