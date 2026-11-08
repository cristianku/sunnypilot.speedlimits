# Request for advice: the brain does not learn the next speed limit

You are a senior ML engineer with experience in driver-assistance perception and sequence models trained on small, real
driving datasets. Read this whole page and the code it points at in this repository
(`github.com/cristianku/sunnypilot.speedlimits`, branch `main`), then answer the question at the end. I want concrete,
ranked advice, not a survey.

## 1. The project, from the beginning

- The car is a Peugeot 3008 with a comma 4 running sunnypilot (an openpilot fork). The road camera gives frames at
  20 Hz; our networks run on the comma's own GPU.
- The goal is the speed limit from vision only, for sunnypilot's Smartspeed / Speed Limit Assist. **The vision side
  never reads the map.** Map and vision are weighed only later by sunnypilot's resolver, which is outside this question.
  OpenStreetMap is used only offline, for training labels such as road types.
- The pipeline on the car has three steps.
  1. **SignNet** finds the boxes of signs in the top band of the frame (896 × 288).
  2. **A classifier** (MobileNetV3-small on 160 × 160 crops) reads each box: a limit 10–120, end of limit, zone,
     LED display, other sign, or not a sign.
  3. **The brain** is one recurrent network and the only decider. It outputs what the car does. The signs that were
     read are inputs to it (hints), never the limit itself.
- The car code is in `cristianku/sunnypilot`, branch `speed-limits`, under `sunnypilot/speed_limits/`:
  - `brain.py` steps the ONNX model;
  - `brain_inputs.py` builds the input rows, and the same file is used to build them for training;
  - `speedvisiond.py` publishes the outputs: `speedLimit`, `speedLimitAhead` and `speedLimitAheadDistance`. The
    distance is counted down by odometry between steps.

## 2. The brain

The code is `src/nnslr_tools/brain_net.py`; training is in `brain_train.py` and the targets in `brain_targets.py`.

- **Rate:** one step at 4 Hz. Training uses chunks of 64 s (256 steps), and the GRU state is carried over on the car.
- **Picture:** the road frame at 256 × 144 goes through a **frozen** ImageNet MobileNetV3-small. Its features are
  576 × 8 × 5 cells. A 1 × 1 convolution reduces them to 32 channels × 40 cells, and a linear layer reduces that to 128
  features (LayerNorm, Hardswish, dropout 0.2). Batch norm is frozen.
- **Other inputs**, concatenated with the picture features into a GRU with hidden size 128:
  - `signs` (11 values), described in detail below;
  - `plate`, `plate_hours` and `plate_wet`: the sub-plates of the last sign;
  - `hour`, `lead` (the lead car) and `country`;
  - `sign_distance` (2 values): the metres to the sign ahead, from the box geometry and the car's speed, and a known
    flag.
- **The `signs` row:**
  - [0] the confirmed number / 200, and [1] known. Confirmed means the same reading for 1.5 s, the rule of
    `vision.SpeedLimitMemory`;
  - [2] display, [3] zone, [4] marking, [5] log age, [6] ended, and [7] read in the last 1 s;
  - [8] the last single reading / 200, [9] known, and [10] its age.
- **Heads:**
  - `speed = speed_head(h) + trust(h) * onehot(number in force)` gives the speed to hold, over 200 classes
    (1–200 km/h). The target is the driver's dashboard speed, as a Gaussian bell with σ 2 km/h.
  - `next = next_head(h) + trust_next(h) * [onehot(confirmed number), 0]` gives the next limit: 200 classes plus a
    "none" class.
  - `metres = metres_head(h) + metres_gate(h) * sign_distance` gives the metres to the next limit.
  - There are also small auxiliary heads: works, road kind and road type.
- **The number is never guessed.** It comes in through the `signs` input and is added straight to its class. The brain
  only has to decide **whether** the number it holds is still ahead (the next limit) or already in force, and **how
  far** it is.
- **Loss:** cross entropy for speed, plus `0.5 × cross entropy(next)` and `1.0 × smooth L1(metres)`, each on its own
  frames. Since today the next loss is balanced: all frames with a number ahead together weigh as much as all frames
  with "none". AdamW, lr 1e-3, 40 epochs, about 25 s an epoch on one GPU. Three seeds.

## 3. The next-limit target (`brain_targets.ahead_from_signs`)

- A frame gets a number only within **100 m before the next sign that changes the limit**. The sign's boxes must be
  covered by readings, and the limit in force must be certain.
- A frame is "none" only when no sign is within 130 m and the whole 130 m ahead was read.
- Everything in between is out of the loss.
- The labels come from the reviewed signs: boxes confirmed by a person, or read by the networks where nobody reviewed.

## 4. The data

- There are 27 dashcam drives, all from one car, mostly in Switzerland.
- At 4 Hz, the frames that have a speed answer are:
  - train 61,119;
  - validation 10,778;
  - test 46,092;
  - buffer 7,785.
- The **events** (the start of a run of frames with a number ahead, one per limit change) are:
  - **train 120**;
  - validation 30;
  - test 158.
- The frames with a number ahead are:
  - train 2,840;
  - validation 1,034;
  - test 3,833.
- `sign_distance` is known on 10 % of all frames, and on 55 % of the frames with a number ahead.

## 5. What we measured

The speed head works: its mean error is about 10–11 km/h on validation and test. It beats "follow the sign"
(26 km/h on validation) and a constant (62 km/h). The next head does not work.

- **Old seed 0, without balancing:**
  - it was right on 87 % of next-limit frames overall, almost all of them "none";
  - on limit changes it was right about 5 % of the time;
  - false "ahead" 1.1 %.
- **Diagnosis on the test frames with a number ahead, by what the `signs` input holds at that frame:**

  | signs input at that frame | frames | old s0 right | balanced s0 right | what it said otherwise |
  |---|---|---|---|---|
  | confirmed number == the next limit | 515 | 79 | 108 | "none" on 432 → 380 |
  | only a single reading == next | 56 | 3 | 2 | mostly "none" |
  | the next sign not read yet | 410 | 15 | 21 | "none" |

  So on 515 frames it already **holds the right number in its input**, and it still says "none" on about 75 % of them.
  At that point the number is read but the sign is not passed yet. It cannot tell "a sign ahead, read" from "a sign
  just passed, now in force".
- **Balanced seed 0 (today):**
  - changes read right went from 14 % to 19 %;
  - false "ahead" went from 1.1 % to 2.2 %;
  - the metres error is about 30 m;
  - the speed is unchanged.
- **Validation, balanced seed 0:**
  - changes right 5 %;
  - changes with the number already read: 12 % right;
  - false ahead 3 %;
  - metres error 22 m.

## 6. Hypotheses (Cristian's first three, then ours)

1. **Too little data.** There are only 120 training events, from one car.
2. **The resolution is too low.** At 256 × 144 a sign 100 m ahead is a few pixels, so the picture cannot show "the sign
   is still in front of us".
3. **The network is not adequate.** It does not see what it must see: a frozen ImageNet backbone, 40 cells, and a
   128-wide GRU at 4 Hz.
4. **Ambiguity between the target and the input.**
   - The target starts at 100 m, often before the sign is read, and 410 of 981 frames have nothing read yet.
   - Nothing in the input says "this reading belongs to a sign still ahead", apart from `sign_distance` (missing on
     45 % of those frames), the age fields and the picture.
   - Read-but-not-passed and passed look alike in the `signs` row.

## 7. The question

Give me your **top 3 approaches, ranked**, to make the next limit and its metres work on this car. For each, say:

- which hypothesis it addresses;
- what to change: the inputs, the targets, the architecture, the loss or the data;
- what you expect it to gain;
- how to check it with the numbers above.

Respect these constraints:

- vision and car signals only, never the map;
- it runs on the comma 4 GPU at 4 Hz;
- the sign reader (SignNet + classifier) already exists and can give more per-sign data. Examples: box size and
  position, a track id, or "box left the frame";
- the data cannot grow quickly: about 30 drives.

Another constraint is that the limit logic is learned by a network, not hand-coded rules on the car. Within that, tell
me plainly whether the next limit belongs in the brain, or in a small network of its own fed with the sign tracks.
Such tracks could include the box leaving the frame, the box's growth rate and the car's odometry. And tell me which
of the four hypotheses the numbers above point to.
