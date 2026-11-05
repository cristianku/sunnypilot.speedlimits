# The speed-limit pipeline: cutter → classifier → decider

A plan, started on 24 October 2026 from Codex's "teacher and student" proposal, Cristian's scheme of three separate networks, and the real data on the server. The first runs of that day are measured below.

![The pipeline](neural-network.png)

## Where things run

**All training happens on the server**, on the V100. The comma never trains anything.

- **The networks for the car** are trained on the server from the reviews, then copied to the car. They run on **Chestnut**, comma's external GPU, not on the comma four's own chip (decided on 24 October 2026): the chip would need the frame shrunk until distant signs disappear, while Chestnut takes the full 1344 × 760 frame. Chestnut also runs openpilot's driving model, which always comes first: the room left next to it is still to be measured. The cutter is the exception: SignNet was made for the comma's own GPU (step 1 below). In the car they learn nothing and see only the past, since there are no future frames while driving.
- **The same networks prepare new drives on the server** (Prepare, the "Our networks" proposer). There a drive can also be looked at with its later frames, its GPS and the map: that extra step (the teacher below) makes better proposals than the car can, without needing a different network.

## Three networks, each trained and measured on its own

1. **Cut: the cutter finds every box that could be a sign.**
   - **SignNet plays this role** (since 30 October 2026, see the decisions), the same network the comma runs. It looks at the band at the top of the frame of the main road camera (the top 432 of the 760 rows of the comma four's 1344 × 760, scaled to 896 × 288; the top 416 rows alone hold 97% of the reviewed signs), and gives a heat map of sign centres with each sign's size (stride 8, `nnslr train-signnet`, the **Train the cutter** button). The car keeps the best 5 boxes of a frame that score 0.3 or more. On the server the SignNet in use gives the car's best 5 boxes a frame, from score 0.1 (the car keeps those of 0.3 or more), to Prepare's Our networks, to the mined false boxes and to the chain, in frame pixels and with no size filter; the chain is judged at 0.3, as the car reads it.
   - It must miss nothing. False boxes are fine, because the classifier drops them, so it can be a small network.
   - It is measured on **signs missed, by size**, and on how many boxes per frame it passes on; for SignNet, on the signs among the boxes the car keeps (score 0.3 or more, at most 5 a frame), and among the best 5 boxes of a frame at any score.
   - A found box matches a reviewed box at IoU 0.5 (greedy by score, one to one). For the boxes the car keeps there is also a **lenient count** (`car_centre` in `result.json`, shown first on Train and Models with the IoU 0.5 count in small): a box also matches when its centre is inside the reviewed box and its width and height are each half to twice the sign's. A far 10 px sign boxed 2 px tight on each side has IoU 0.36, a miss and a false box at IoU 0.5 though it sits on the sign; a huge box around a small sign still does not match. `signnet.remeasure_car_centre` adds it to an older run's `result.json` without training.
   - It also cuts the other traffic signs (`--families all`, the default; `speed` is the speed signs only), so that the classifier sees them and says what they are. Its speed-sign recall is still reported alone (`recall_by_family`).
   - The one on the car (`2026-10-28_1028-signnet-twothirds-s8-f1`, 63 k parameters) finds, at the car's score of 0.3, 40% of the test signs (152 of 381) and 58% of the speed signs (120 of 208), with 0.22 false boxes a frame, measured on the split of 26 October; among its best 5 boxes at any score it has 55% (209) and 62% (130), with 4.8 false boxes a frame. A sign below the band (a road marking, a low sign close by) is not found, as on the car.
   - Before it, an SSDlite detector on the whole frame played this role (the old Detector, kept to compare: see the results). The very first one found only 6% of the test signs.
2. **Classify: the classifier says what each box is.**
   - It takes the box cut from the full-resolution frame, never from the scaled-down one.
   - Its answers are a speed value, end of limit, zone, variable display, "?" (a limit it cannot read), another sign with its kind (town entry, direction, give way…), or **not a sign**.
   - Today's reader plays this role. It still lacks "not a sign": the 2,431 model boxes that reviewers removed, plus the cutter's own mistakes, teach it.
   - `scripts/run_classifier.sh` (the **Train the classifier** button) trains it with those classes (`nnslr train --classes v2`): the removed proposals that overlap no kept box, the boxes marked "not a sign" and up to 3 × the signs of false boxes mined with the cutter in use (`nnslr mine-negatives`; with a SignNet, among the car's best 5 boxes a frame, from score 0.1), darker crops, jittered boxes and a quarter of the speed signs drawn as an end of limit, labelled `cancellation` (`synthetic_signs`: the real ones are too few). It then measures it alone and the whole chain on the test (`nnslr evaluate-chain`: signs found **and** read right, false boxes per frame that survive), next to the old reader. First results below.
   - Frames an AI reviewed before the other-sign rule have their other signs without a box. There, the other-sign proposals the AI removed and the cutter's boxes scoring 0.3 or more are not used as "not a sign" (see the results).
   - It is measured on right answers per sign.
3. **Decide: the decider says whom each sign applies to.**
   - It takes the list of recognised signs with their position and size, the scene as a small image, and the car's speed and steering. The signs of a frame look at each other, as with two 80s on one gantry.
   - It answers ours, another or abstain, with how sure it is.
   - Experiment B supports it: position and speed raised the "Another" signs found from 59% to 77%, while the full image made the network memorise places.
4. **Over time**, with the past only: the same sign is followed across frames, as "Apply previous" does (the tracks of 323 signs are already computed with `nnslr tracks`). The reading improves as the car gets closer, and the sign is either ahead or passed.
5. **The limit now** is rules and explicit state, not a network, so the reason for a limit is always visible.
   - The country comes from the GPS, and its rules interpret what the networks read: a town entry in Poland means 50.
   - The logic is written and tested here, in `speed_vision_core`, and replayed offline on the recorded routes.
   - Only then does it go into sunnypilot, as a separate change: the networks run on Chestnut through openpilot's model process (`modeld` owns Chestnut: no second process may start it), and Speed Limit Assist takes it as a third source next to the OSM map and the car's CAN, which the 3008 does not have. At first the limit is only shown on the screen.

## Why the cutter needs pixels

History (24 October 2026): the old Detector. SignNet keeps the pixels another way: it reads only the band where the signs stand, at two thirds of the full width (896 of 1344 px).

The cutter has to find a sign before anything can cut it out, and the median sign is only about 20 px wide in the frame:

| Detector input | A 20 px sign becomes | Cost compared with 320 |
|---|---|---|
| 320 × 320 (today, squashed) | about 5 × 8 px | 1× |
| 640 × 360 | about 10 px | about 2.3× |
| 960 × 544 | about 14 px | about 5× |
| 1344 × 760 (full) | 20 px | about 10× |

The cutter only answers "could this be a sign?" and may leave false boxes, so it can stay small. There are three ways to keep it cheap:
- give it the full resolution only on the band where signs appear;
- run it a few times a second instead of 20;
- once a sign is found, follow it with the tracker and crop it at full resolution.

`scripts/run_detector_sizes.sh` measures each size: signs missed by size at low score thresholds, boxes per frame and time per frame. It trains the old Detector, kept to compare: a Detector can no longer be put in use.

## First results (24 October 2026)

Measured on the test areas of route 0000012d, which no training sees. 35% of those frames are at dusk; only 7% of the training frames are dark.

**Detector (run `2026-10-24_1452`, input 320 × 320)**

| Sign width | Found |
|---|---|
| Under 40 px | 0 of 68 |
| 40 px and wider | 5 of 12 |
| All | 6%, with 529 false boxes |

Its validation loss was best at epoch 8 of 20: after that it memorised the training images. Training now stops after 5 epochs without improvement.

**Old Detector comparison (`scripts/run_detector_sizes.sh 2026-10-24_1452`)**

All three sizes use small default boxes (the smallest is 0.02 of the image side), 10% of the sign-free frames per epoch, and the best epoch chosen by validation recall. Scores are at the cutter's low threshold (0.1), on the 80 test signs.

| Input | Signs found | < 12 px | 12–20 px | 20–40 px | ≥ 40 px | Hand-reviewed | Boxes per frame |
|---|---|---|---|---|---|---|---|
| 320 × 320, default boxes 0.2 (the run above) | 6% | 0% | 0% | 0% | 42% | | |
| 320 × 320 | 38% | 11% | 19% | 58% | 92% | 48% | 80 |
| 640 × 360 | 62% | 17% | 61% | 89% | 92% | 78% | 237 |
| **960 × 544** | **79%** | **44%** | **81%** | **100%** | 92% | **96%** | **42** |

- **960 × 544 was the cutter in use that day**, then the 1344 × 760 Detector; since 30 October 2026 the cutter is SignNet. The small default boxes alone take 320 × 320 from 6% to 38%, and the larger input takes it to 79%.
- **Cost:** on the V100, 11.1, 11.3 and 13.3 ms per frame, which says little about the comma; 960 × 544 has about 5× the pixels of 320 × 320.
- **Every box goes to the classifier, and almost all are junk.** Dropping them is the classifier's job.
- **Still hard:** signs under 12 px. The next steps are the full-resolution band and more drives.

**Reader (same run)**

| | Right answers |
|---|---|
| All test signs | 71% |
| Signs reviewed by hand | 60% |
| Speed limit 60 | 81% |
| Speed limit, number unreadable | 91% |
| Other signs | 57% |

Other signs were mostly taken for variable displays: the AI reviewer used to leave them out of the boxes, while people box them. The rule now asks for them too, with their kind.

**Classifier (run `2026-10-24_1750-classifier`, cutter 960 × 544 at score 0.1, test)**

| | New classifier | Old reader |
|---|---|---|
| Speed signs found **and** read right | 61% (49 of 80) | 66% (53 of 80) |
| … of those a person confirmed | 70% (16 of 23) | 65% (15 of 23) |
| False boxes per frame left after the classifier | **0.26** (from 41.5) | 41.5 (it has no "not a sign") |
| Other signs read right | **0 of 54** | |

- **It drops 99.4% of the cutter's false boxes,** which is its job. The ones left are mostly read as "speed, number unreadable" or "variable display".
- **It called every other sign "not a sign".** The training had 34 other signs against 5,906 "not a sign" crops, and many of those crops were real signs:
  - before the rule to box other signs, the AI reviewer removed the other-sign proposals: 1,339 of the 1,434 removed proposals of train were proposed as other signs, mostly pedestrian crossings, give way, no parking and direction signs;
  - about 15 of the 40 highest-scoring cutter boxes mined in those frames were real signs: no entry, no overtaking, roadworks, end of motorway.
- **Fixed:** in those frames, removed other-sign proposals and mined boxes scoring 0.3 or more are no longer "not a sign" (`BOXES_OTHER_SIGNS`, `mine-negatives --sign-score`). Run `2026-10-24_1835-classifier` (cutter 1344 × 760, score 0.1, test) measures the fix: other signs read right went from 0 to **18 of 54**; speed signs found and read right **66%** (53 of 80; old reader 68%); the cutter's 210 false boxes per frame drop to **0.49**. In train, 650 likely signs were left out of the mined boxes.
- Those 1,339 removed proposals are other signs waiting for their kind: reviewing them would give the classifier its first real set of other signs.

**Proposers of Prepare (`helpers/measure_proposer.sh`, 25 October 2026, the 604 test frames)**

| Proposer | Speed signs proposed with the right number | Speed signs found | All signs found | False proposals per frame | Seconds per frame |
|---|---|---|---|---|---|
| Vietnamese YOLO (Hugging Face) | 2 of 80 | 27 of 80 | 45 of 134 | 0.08 | 0.5 |
| Our cutter + classifier | **51 of 80** | **60 of 80** | 66 of 134 | 0.45 | 0.8 |

The Vietnamese model, which proposed the signs of every route so far, finds a third of the test speed signs and reads almost none right: reviewers drew and labelled the rest by hand. Our networks propose most of them with the right number, at the cost of about one false proposal every two frames to delete.

**Experiment B (run `2026-10-24_expB`, on the reviewed boxes)**

| Variant | Reading on validation | "Another" found on validation | Reading on test |
|---|---|---|---|
| crop | 85% | 59% | 62% |
| crop + position and speed | 82% | 77% | 63% |
| crop + position + full image | 69% | 59% | 46% |

The full image reached 99.7% on training: it learned the places. The test has no "Another" sign yet, so "whom it applies to" can only be measured on validation, which has 17 cases.

## Cameras and devices

- The comma four records two road cameras, both 1344 × 760: the main road camera (`fcamera.hevc`) and the wide-angle one (`ecamera.hevc`). We use the main one, because its narrower view gives each sign more pixels. The wide one may help at the moment of passing, when a sign at the side has left the main view.
- The comma 3X has three cameras (main road, wide road, driver), and its road cameras record 1928 × 1208 with another sensor and lens. Networks trained on comma four frames do not work on a 3X by themselves. Three things make them work:
  - every frame is converted into a common "standard camera" from the device's calibration, so a sign at 100 m has the same size on every device, as openpilot does for its driving model;
  - training uses variations of size and colour;
  - drives recorded with a 3X are added when available.

## The teacher

The teacher has bigger versions of the three networks, at full resolution everywhere. It also sees the later frames, the GPS and the map. It is used in three separate ways:
- **new labels**, which a person checks on the site;
- **distillation**: the networks for the car also learn how sure the teacher is;
- **starting weights**.

The test stays human: route 0000012d, reviewed by hand, split by place, never trained on.

## The plan

| Step | What | Why |
|---|---|---|
| 1. Cutter | Compare input sizes and the full-resolution band; judge on signs missed (by size) at low thresholds and boxes per frame. Done: SignNet on the band, the network the comma runs, is the one cutter (30 Oct 2026) | No sign is cut out if the cutter misses it |
| 2. Classifier | Add "not a sign" (removed proposals, cutter mistakes) and the kinds of other signs as classes; darker images in training; a button to train it alone | It drops the cutter's false boxes, and the test has many dark frames and other signs |
| 3. Decider | A small network over the classifier's answers, positions, speed and a small scene image; collect "Another" cases (ramps, parallel roads, exits), proposed from GPS and map | Position helped in experiment B; the full image memorised places; 125 "Another" against 1463 "Ours" |
| 4. Site | Every network by role (cutter, classifier, decider): versions, numbers, a training button, the version in use; the whole chain measured end to end | Each network is replaced on its own, and the site must show what is in use |
| 5. Time and state | Tracks, "passed or not" and the limit state on whole routes; time with a wrong limit on the screen measured | The limit must stay valid when the sign is out of sight |
| 6. Chestnut | Export of the networks for the car; latency and memory measured on Chestnut next to the driving model (Cristian installs it) | The driving model comes first |

Data comes before all of these. Route 0000012d is reviewed by hand as the test, drives on new roads and at night are needed, and the split is by place: 75–100% of every route runs on the same roads as another route.

## Second opinion (strata)

strata, a local model, agrees on these points:
- a human test set first;
- crops at full resolution;
- a finder at 480–640 px followed by full-resolution crops;
- an explicit limit state that says "unknown";
- 1 frame per second is too sparse for tracking.

It added two points, both taken: "Unclear" is taught as abstain rather than dropped, and 125 "Another" are too few for a network alone, so that case also needs geometry, rules and targeted collection. It also wanted experiment B only after the detector; B was run first anyway, because it uses the reviewed boxes and so measures context without the detector's errors.

## Decisions taken (24 October 2026)

- **Three networks** (cutter, classifier, decider), each trained, measured and replaced on its own, as Cristian proposed. The cutter favours missing nothing; the classifier drops what is not a sign; the decider works on the recognised signs and their positions.
- **Test.** Route 0000012d is reviewed entirely by hand: about 800 frames with proposals and 200+ signs, on motorway and in town. The final test also needs drives on new roads.
- **Split by place.** Areas of about 300 m. Every frame in an area goes to the same partition, whatever route it comes from. The test areas are those of the test route, and frames of any route next to them are kept out of training (the buffer). The validation has its own buffer too since 28 Oct 2026 (policy `place-v4`): two compact stretches of other roads, each started where it adds the most speed-limit values, where before its areas were scattered and shared their signs with the training. strata asked for 250–500 m areas with a 100–200 m buffer.
- **"Ours".** The sign is meant for the traffic on the carriageway we drive on, in our direction, whatever the lane. **"Another"** is an exit ramp, a side road, a parallel road or the opposite direction. The decider answers which applies; the label only fixes what the answer means. A lane label comes later, if gantries with different values per lane show up.
- **"Unclear"** is taught as "abstain", not dropped.
- **Other traffic signs** get a box with their kind, from people and from the AI reviewer alike.
- **On a vehicle** (25 October 2026): a sign-like disc or panel on a vehicle, such as the top-speed sticker on the back of a truck, is boxed as "Not a sign" with the kind `on_vehicle` (key V). Cut to its box it is the same pixels as a sign, so the classifier does not learn from it (it would learn to reject signs); the decider, which sees the scene, learns it as not ours (`other_road`).
- **Painted on the road** (28 October 2026): an other sign painted on the road, such as the children's triangle with "SCHULE" in front of a school, keeps its kind (Children) and is marked "Painted on the road" on the review page (`painted: true`, other signs only). The networks learn it from the picture; the flag keeps these cases countable apart. A painted number is not this: it is a road marking.
- **Road markings are speed limits** (26 October 2026): a number painted on the road ("30", "30 ZONE") counts as a speed sign. It is often the only one a driver gets, for example when leaving a parking space inside a 30 zone, past its entry sign. The cutter learns to box it (both "Every sign" and "Speed signs only"), the classifier reads it (`road_marking:30`) and the decider says whose road it is on. Older reviews left many painted numbers without a box, so until they are boxed, those frames teach the cutter that the paint is background. SignNet, the cutter since 30 October 2026, reads only the band at the top of the frame, so it does not find most painted numbers, as on the car.
- **Published together, on Hugging Face** (25 October 2026): the three networks in use go out as one package (ONNX, contracts, model card, `networks.json`) in the catalog `speed_networks.json` of `cristianku/sunnypilot-speedlimits-models`, so that other people can use them; a car will download from that catalog, as sunnypilot's model picker does from its own. Prepared and published from the Models tab (`scripts/publish_networks.sh`).
- **One cutter: SignNet** (30 October 2026): the cutter is SignNet, the network the comma runs (band 896 × 288, stride 8, fuse 1 × 1, widths 16,24,32,48), and no longer an SSDlite Detector on the server next to SignNet on the car. Why: the classifier learns the false boxes of the cutter, so they must be the false boxes of the cutter the car runs; and there is one network to train and measure. **Train the cutter** (`scripts/run_signnet.sh`, about 80 minutes on the GPU) trains a SignNet with the car's setup; **Use this one** takes only a finished SignNet at 896 × 288 and stride 8, the only one the car's `vision.py` runs, and **Send to the car** and **Try on the comma now** refuse any other. Prepare's Our networks, the mined false boxes and `evaluate-chain` use the SignNet in use, with the car's best 5 boxes a frame, from score 0.1 (the car keeps those of 0.3 or more), no size filter; `evaluate-chain` measures at cutter scores 0.1, 0.2 and 0.3 and judges at 0.3, the car's floor. The old Detectors (`scripts/run_detector_sizes.sh`, the whole pipeline's detector) are kept to compare, and cannot be put in use. The cost: signs below the band (road markings, low signs close by) are not found, as on the car, and on the split of 26 October the SignNet on the car finds fewer test signs among its best 5 boxes at any score than the 1344 × 760 Detector did with its best 5 (55% against 61% of all signs, 26% against 29% of those under 12 px); at the car's 0.3 it finds 40% of all signs (152 of 381) with 0.22 false boxes a frame. Every SignNet so far was trained on the split of 26 October: a new **Train the cutter** is needed before one is put in use.
- **Our networks propose only whole** (25 October 2026): Prepare offers "Our networks" only when the cutter, the classifier and the decider are all in use (`preannotate.ours_missing`). The decider sets the road (ours, another, or unresolved when it abstains) of every speed sign proposed, with the car's speed from `nnslr gps` when the drive has it (`_chain_proposer`); the other signs stay unresolved, since it learnt ownership only on the speed signs.

## The road type with memory (29 October 2026)

**Why.** Mobileye, and today's sunnypilot, keep a sign's limit until another sign ends it. On a Swiss motorway on-ramp a 60 sign is never ended by a sign: the motorway's 120 starts where the ramp merges. The same holds for the end of a town (80 or 100 after the town's 50), for an expressway (100) and for a car park. The car needs the kind of road it is on, and the moment it changes, from the camera. One frame is often not enough (the first metres of a motorway look like a rural road), so the network (**Scene** on the site) has a memory of the seconds before.

**The network** (`src/nnslr_tools/road_type_net.py`, `nnslr train-road-type`):
- one step a second: the whole road frame, resized to 256 × 144, and the car's speed;
- encoder: torchvision's MobileNetV3-small with ImageNet weights, pooled, then 128 numbers (Linear, Hardswish, Dropout);
- the speed as two numbers, `min(max(v, 0), 60) / 30` and 1 when known (GPS speed in training, `carState.vEgo` on the car);
- memory: a GRU of 128; output: the probability of each of `road_types.ROAD_TYPES` (town, rural, expressway, motorway, ramp, parking, zone30, garage, merge), in that order and with those names (zone30, a 30 zone, garage, a car park underground or indoors, and merge, the acceleration lane after an entry ramp, added last on 30 Oct 2026: a checkpoint trained before them keeps its six to eight, on the car its parking gives the garage's 10 km/h and its ramp's limit ends by the older rule);
- the version without memory, to compare: the same encoder and speed with a small head per frame (`--memory none`), also measured with a causal 5-second majority vote over its answers, the cheapest memory;
- size: 1,101,478 parameters (1,018,406 without memory), about 84 MFLOPs a step, 3.9 ms a step on the V100 at batch 1. At 1 Hz that is a small fraction of Chestnut, next to the driving model. Measured on the comma 4 (30 Oct 2026): the resize of the road frame to 256 × 144 on the CPU 263 ms, the network on the GPU 26 ms, once a second; the resize is the part to make faster.

**Labels and weights** (`src/nnslr_tools/road_type_data.py`). `nnslr road-types` only proposes a road type for each frame. Since 4 Nov 2026 (`LABEL_POLICY`, `confirmed-only`) a frame has a label only when a person confirmed or corrected its type in Data › Roads (`road_types.confirmed`: `reviewed` true and a valid type), and those decisions outlive a new search of the map. The weight of a frame in the loss:
- a person's type: 1;
- everything else: 0. That is a proposal of the map, however sure, a ramp from its OSM link (`motorway_link`, `trunk_link`), an acceleration lane, a frame without a road type and "Can't tell". It is out of the loss and out of every measure, on validation and test too, but still goes through the memory;
- on top, the loss weighs each class by 1 / √(its frequency).

The windows are cut around confirmed frames. A proposal keeps one use, as an input: the `map` sign stands at the start of each entry ramp, a person's type over the map's (`map_label`), so a ramp corrected to motorway places no sign. It is never an answer. The split counts confirmed frames only (`road-type-place-v3`): the validation areas start on them and grow through driven areas without them, so they move as the reviews grow. With too few confirmed frames in a partition (`missing_labels`: 2,000 in train, 300 in validation and test, 30 each of town, rural, motorway and ramp in all three) a training stops before writing anything and says what is missing.

Everything trained before that date, the results of this section included, learned from the map's proposals as well: sure frames and OSM ramps at weight 1, other unsure frames at 0.3, and it was measured on them. Those numbers are not comparable with later ones, and this rule does not correct those models. Starting from one (`--init-from`) brings along what it learned, and the log says so.

**Split** (`road-type-place-v1`, split sha256 `a1681d1cef42…`). The drives pass the same roads again and again, so it is by place, like the signs' split: areas of about 300 m. Test is every area of the held-out test drive of `data/splits/latest.json` (`0000012d--6552a45378`, the same drive as the signs' test), so that drive is measured as it was driven; the areas next to it are a buffer. Validation is two compact clusters of other areas (about 15 % of the frames) with their own buffer. A sequence is a run of frames of one drive, without a gap, in one partition, of at least 30 seconds: the network is trained and measured on whole stretches, never on lone frames.

| Partition | Sequences | Labelled frames | Town | Rural | Motorway | Ramp | Parking |
|---|---|---|---|---|---|---|---|
| Train | 18 | 10,031 | 2,595 | 2,171 | 3,902 | 365 | 178 |
| Validation | 11 | 1,814 | 328 | 509 | 783 | 82 | 0 |
| Test | 17 | 6,646 | 1,757 | 937 | 2,840 | 327 | 0 |

There is no expressway frame anywhere (the drives have none on the map), and no parking in validation or test. The changes of road type (10 s of each side): ramp → motorway 2 in train, 1 in validation, 4 in test; motorway → ramp 2, 1, 2; town entry 13, 3, 11; town exit 11, 3, 10. Merges (a ramp of 3 s or more, then the motorway): 5 in train, 1 in validation, 7 in test.

Unsure frames, by class (to confirm first in Data › Roads): town 343, rural 1,352, motorway 397, ramp 832, parking 13.

**Training.** Windows of 48 seconds, 35 % of them around a change of road type; half of the batches start from the memory of the 32 seconds before the window, as on the car, the others from zeros. Per sequence the same random colour and framing (no mirror: traffic keeps right), a second dropped now and then, the speed ±5 % and unknown in one sequence in ten. AdamW, the best epoch by the validation macro F1, run over whole sequences with their memory. `scripts/run_road_type.sh` does both networks, the comparison and the export in about 15 minutes on the V100.

**First results** (run `2026-10-29_0937-road-type`, test, weight-1 frames, 5,029 seconds):

| | Macro F1 | Accuracy | Town | Rural | Motorway | Ramp |
|---|---|---|---|---|---|---|
| With memory (GRU, epoch 37) | **0.584** | **0.790** | **0.932** | 0.576 | **0.827** | 0.000 |
| Without memory (epoch 30) | 0.548 | 0.749 | 0.831 | 0.553 | 0.801 | 0.006 |
| Without memory, 5 s majority | 0.571 | | 0.846 | 0.612 | 0.824 | 0.000 |

| Change (test) | With memory: found, median delay | Without memory |
|---|---|---|
| Ramp → motorway (4) | 4, −7.5 s | 3, −10 s |
| Motorway → ramp (2) | 0 | 0 |
| Town entry (11) | 11, +3 s | 11, 0 s |
| Town exit (10) | 10, +1.5 s | 9, −2 s |
| Other (8) | 3, +7 s | 4, +1.5 s |

- **False switches** inside one road type, per hour: with memory 58 raw, 33 after the car's 3-step debounce; without memory 483 and 75; with the 5 s majority 136 and 80. The memory is what makes the answer steady.
- **After a merge**, the first 60 s on the motorway after a ramp, the share said motorway: with memory **91.5 %** (90.9 % debounced), without memory 79.9 % (81.0 %), with the majority 81.6 % (83.0 %), over 7 merges and 364 s of test. Without memory most of the misses are "town" (57 s); with memory 22 s rural and 9 s town. Validation has a single merge (60 s, all right for every version). **These are few merges**: the trend is clear, the number is not yet a measure.
- **Ramp is never recognised** (F1 0): of the 327 ramp seconds of test it says motorway for 180, rural for 82, town for 62. There are about 12 ramps in train, all with unsure labels. So the "ramp → motorway" change is found early (−7.5 s): the network says motorway while still on the ramp.
- **Motorway taken as rural**: 652 of the 2,840 motorway seconds of test, mostly in two stretches (0000012e segments 33–45, 350 s; 0000012d, 231 s).

**On the car** (the ONNX step, `nnslr export-road-type`; `road-type/export/road-type.onnx` and `road-type.onnx.contract.json` in the run). One step: `image` (1, 3, 144, 256), `speed` (1, 2), `hidden` (1, 128) in; `probabilities` (1, 6) and `hidden_out` (1, 128) out. The GRU is written as matrix products, sigmoid and tanh, with no ONNX GRU node; the ImageNet normalisation is inside. Checked against PyTorch with onnxruntime: largest error 6e-7 on random inputs, and the same answer on all 120 seconds of a real drive run step by step. The input, from sunnypilot's `speed_limits/vision.py`:
1. `y, uv = vision.nv12_planes(buf.data, buf.width, buf.height, buf.stride, buf.uv_offset)` from `VISION_STREAM_ROAD`, the whole 1344 × 760 frame;
2. `planes = vision.yuv_to_rgb(y, uv, 0, buf.height, 0, buf.width)` (BT.601, as the dataset PNGs);
3. `planes = vision.resize(planes, 256, 144)` (PIL bilinear: bit-exact with the training cache);
4. `image = planes[None].astype(np.float32) / 255.0`, RGB, NCHW, no mean or std;
5. `speed = [[min(max(vEgo, 0), 60) / 30, 1.0]]`, or `[[0, 0]]` when unknown;
6. `hidden`: zeros at the start of the drive and after more than 5 s without a step, then the previous `hidden_out`;
7. one step a second (the frames of the training are 1 s apart; a faster rate would make the memory faster too);
8. `road_type = classes[argmax(probabilities)]`, taken only after the same class 3 steps in a row.

**What is missing.**
- Ramps: confirmed ramp frames in Data › Roads, and more drives with on- and off-ramps. Until the network says "ramp", the car must not lift a ramp's limit on "motorway" alone: it says motorway on the ramp itself.
- Expressways: no frame yet. Parking: no held-out frame.
- More merges in validation and test, to make the after-merge number a measure.
- The stretches of motorway taken as rural, to look at (the frames, or the map's labels there).
- The car: nothing uses it. It needs `SpeedLimitMemory` to take the road type, the timing on Chestnut, and a decision on how the network and the map combine.

### Second version: the motorway sign, and when the ramp's limit ends (29 October 2026)

**Why.** What matters on the car is one moment: when the 60 of an entry ramp ends and the motorway's 120 comes back. The comma's rule (`vision.SpeedLimitMemory` in sunnypilot) is:
- a number read on a ramp (the road type says ramp) or within 45 s of a confirmed motorway or expressway sign is the ramp's limit;
- it ends once the road type says motorway or expressway 8 steps in a row.

In Switzerland the motorway sign (4.01) stands where the entry ramp starts. So the network does not need to tell the ramp itself on an entry. It must say "on the carriageway" only after the merge, and keep saying it, with its memory, when the first frames of the motorway look like a town or a rural road. Ramp recognition is still needed for interchanges, which have no 4.01; it comes second. No rule on the speed.

**The sign input** (`src/nnslr_tools/road_type_signs.py`). One more input of two numbers a step:
- `[min(t, 120) / 60, 1]`, where t is the seconds since a motorway or expressway sign was confirmed;
- `[2, 0]` when no such sign was confirmed in the last 120 s.

On the car t is `now - SpeedLimitMemory._road_sign_at`. The network grows to 1,102,246 parameters; the cost per step is unchanged (84 MFLOPs, 4.0 ms on the V100).

**Where the signs come from.** There are two sources (`--sign-source`):
- **The car's own code** (`nnslr road-type-signs`, step 4 of the script). It runs the car's `vision.py`, SignNet and classifier from the sunnypilot clone, read only, over `fcamera.hevc` at 4 frames a second (the rate speedvisiond really reads at), with the settings of the car's release and speedvisiond's confirmation (2 frames within 1.5 s). On all 12 drives it read 99,000 frames, 15,587 of them with a sign read sure enough, and confirmed **no motorway sign at all**. At the three reviewed 4.01 signs, the classifier gives `motorway_start` at most 0.022 on every box: it reads the sign as a generic other sign, or the cutter leaves it out. The classifier has seen only 8 reviewed frames of 4.01. So today the car would give this input nothing. The readings are cached in `data/derived/road_type_signs/`; on the GPU (NVDEC and onnxruntime-gpu, the same readings as the CPU's) 20 minutes of drive take about 30 s, on the CPU about 16 minutes for all on 30 processes.
- **The map** (the default, used for training and as "the car reading every 4.01"). The sign is placed at the first second of every OSM ramp that leads onto the motorway from another road. A ramp from the motorway itself is an interchange and gets no sign. This gives 8 signs on 7 of the 12 drives: only 2 fall in train, 1 in validation and 5 in test areas. Of those 5, 3 are on merges that lie whole in test. The three reviewed 4.01 signs were seen from 7 s before to 6 s after that second, so in training the time moves by up to ±4 s. In a quarter of the windows the sign is missed, and in 5 % of the windows without one a sign is invented, as a misreading.

**The board input** (1 November 2026). The green boards towards a motorway or expressway (`other_sign:motorway_direction`: places, arrows, the motorway pictogram on a direction board) stand at the start of the entry ramp in Switzerland, where the limit is not yet 120. So a board never sets a limit on the car: it is one more input of the scene network, the seconds since the last board was confirmed, encoded as the sign input (`[min(t, 120) / 60, 1]`, else `[2, 0]`), after the motion input.
- On the car `SpeedLimitMemory` confirms a board as any sign (2 pictures within 1.5 s; `BOARD_SIGNS` in `vision.py`), keeps only its time (`board_age(now)`) and drops it from the confirmed signs: the limit, the road and the ramp never read it. `RoadTypeReader.step` gives the ONNX step its `board` input when the network has one.
- Offline `nnslr road-type-signs` runs the same memory and writes the boards' times beside the motorway signs (`boards` in `data/derived/road_type_signs/<route>.json`). `--classifier` reads the drives with another classifier's `reader-best.pt`, exported once to ONNX (in `data/derived/road_type_signs/classifiers/<sha>/`), before it is on the car: the car's classifier has no board class yet. Its sha is part of the readings' fingerprint.
- `nnslr train-road-type` takes the input on its own when every drive was read with the boards and boards were confirmed on at least 3 drives (`MIN_BOARD_DRIVES`); `--boards` asks for it (refused below that), `--no-boards` leaves it out. In training the boards are dropped, invented and moved in time as the motorway sign is.
- `--init-from` a network without the input: its first layers get zero columns for the board, so it starts exactly where it was.
- The export names the input `board` (after `motion`); its contract says `boards` and `board_signs`. `nnslr send-to-car` refuses a scene network with the input beside a classifier without the class (a check of the class name only).
- The classifier of 1 November (`2026-11-01_1812-classifier`, 9 train crops of the boards) reads 1 of 93 boards on validation and 0 of 16 on test: not enough for the input yet. More boards are being labelled.

**Training changes.** A quarter of the windows are cut around a merge. The ramp and the first 60 s of carriageway after it weigh 3 times in the loss. Everything else is as in the first version, and the best epoch is still chosen by validation macro F1.

**The comma's rule, measured** (`road_type_metrics.comma_release`, `nnslr release-road-type` for any checkpoint). Every ramp → motorway merge of validation and test is measured:
- each drive is run whole from zeros, as on the car, and the answer goes through the 3-step debounce;
- the ramp's number is read at the ramp's third second;
- it is held when the road type says ramp then, or a sign was confirmed in the 45 s before;
- it is released at the 8th motorway or expressway answer in a row.

Each merge has one of four outcomes:
- **on the ramp**: released before the merge, so a false 120 on the ramp;
- **in time**: released within 60 s after the merge;
- **late**: not released by 60 s after the merge;
- **not held**: the ramp's number is kept as an ordinary limit, until another sign ends it, which is today's stuck 60.

Since 31 October 2026 a number read off the carriageways that neither the ramp nor the sign holds is held "off" (see "A limit read before the motorway" below) and gets one of the first three outcomes too; "not held" is left for a number read on the carriageway or the lane.

There are two scenarios: the map's signs, and the signs the car's code confirms today.

**Results** (run `2026-10-29_1231-road-type`; v1 is `2026-10-29_0937-road-type`, measured again with `release-road-type`). Seconds are relative to the merge.

| Merge | Before | Ramp | v1 | v2 without the sign | v2 with the sign |
|---|---|---|---|---|---|
| validation 00000128 17/20 → 17/34 | rural | 14 s | +4 | +17 | +2 |
| test 0000012d 6/20 → 7/0 | rural | 40 s | −21 on the ramp | −17 on the ramp | −29 on the ramp |
| test 0000012d 25/30 → 25/38 | town | 8 s | +19 | +37 | +21 |
| test 0000012e 25/54 → 26/22 | rural | 28 s | +5 | late (town for 80 s) | +8 |
| test, 4 interchanges (0000012d 18/6, 18/14, 29/37, 31/14) | motorway | 4–36 s | not held | not held | not held |

| Test, 7 merges, map signs | On the ramp | In time (median) | Late | Not held | Macro F1 | After a merge |
|---|---|---|---|---|---|---|
| v1 | 1 | 2 (+12 s) | 0 | 4 | 0.584 | 91.5 % |
| v2 without the sign (epoch 44) | 1 | 1 (+37 s) | 1 | 4 | 0.517 | 71.4 % |
| v2 with the sign (epoch 54) | 1 | 2 (+14.5 s) | 0 | 4 | 0.561 | 90.7 % |

- **Without memory** (with the sign, epoch 24): on the ramp 1 (−4 s), in time 1 (+19 s), late 1, not held 4. Macro F1 0.441, after a merge 59.9 %.

The same trainings with two more seeds (test, map signs; trained beside the run, not part of it). "Motorway called ramp" counts the seconds of motorway said ramp. A motorway limit read then would end after 8 s, wrongly.

| Version, seed | On the ramp | In time (median) | Late | Not held | Ramp F1 | Motorway called ramp |
|---|---|---|---|---|---|---|
| v1 | 1 | 2 (+12 s) | 0 | 4 | 0.000 | 46 s |
| with the sign, seed 0 (the run) | 1 | 2 (+14.5 s) | 0 | 4 | 0.000 | 2 s |
| with the sign, seed 1 | 1 | 1 (+24 s) | 1 | 4 | 0.065 | 95 s |
| with the sign, seed 2 | 0 | 2 (+35 s) | 1 | 4 | 0.083 | 183 s |
| without the sign, seed 0 (the run) | 1 | 1 (+37 s) | 1 | 4 | 0.005 | 57 s |
| without the sign, seed 1 | 1 | 4 (+17.5 s) | 1 | 1 | 0.120 | 575 s |
| without the sign, seed 2 | 1 | 1 (+20 s) | 1 | 4 | 0.040 | 73 s |

- **The three entry ramps with a sign, over the three seeds** (9 merges): with the sign, 5 in time, 2 on the ramp and 2 late; without it, 3, 3 and 3. The sign helps a little, and only where it stands.
- **Seed 1 without the sign** stopped at epoch 7, before it learnt the ramps by heart. It says ramp often enough to hold 3 of the 4 interchanges, all in time. But it calls 575 s of motorway a ramp, which is the opposite error.

- **With the car's signs of today**, every merge of every version is "not held", 1 in validation and 7 in test: no sign is confirmed, and no version ever says ramp.
- **The sign helps the memory, not the moment.** Against v2 without it, the sign keeps "motorway" after the merge. On 0000012e the version without the sign says town for 80 s, while the one with it releases at +8 s. Against v1, the sign changes nothing on the release. On the long ramp it even makes things worse: it says motorway from the sign on, for the whole 40 s, which is a two-lane ramp with crash barriers on both sides.
- **The ramp is still not recognised.** The run's version scores F1 0.000 on test and on validation, where none of the 82 ramp seconds is said ramp; over the seeds the best is 0.12. Of the 319 ramp seconds of test, the version with the sign says 202 motorway, 72 town and 45 rural. Train has about 12 ramps and 2 entries with a sign, and the training loss falls to 0.01: it learns them by heart. So the 4 interchanges of test stay "not held" in every version.
- False switches per hour on test (raw, debounced): v1 58 and 33, v2 with the sign 80 and 48, without it 119 and 52.

**On the car** (`road-type/export/road-type.onnx` and its contract in the run). The step has one more input:
- `sign` (1, 2), next to `image`, `speed` and `hidden`;
- the contract's `car_input.steps` say how to make it: `age = now - memory._road_sign_at`, then `sign = [[min(age, 120) / 60, 1.0]]` when the age is at most 120 s, else `[[2.0, 0.0]]`.

Checked against PyTorch with onnxruntime: largest error 5e-7 on random inputs. Over 60 s of a real drive (00000125 from 4/30, with the sign for 30 of them) the largest error is 4e-6, and the answer is the same in all 60 s.

**What is missing.**
- **The car must read 4.01.** Without it the sign input is empty, and the rule holds a ramp's number only when the network says ramp, which it almost never does. The classifier needs reviewed 4.01 frames, many more than 8, before the sign input means anything on the car.
- **Ramps, for interchanges.** Ramps need more drives with on-ramps, off-ramps and interchanges. With 12 ramps in train the class does not generalise, whatever the weights or the oversampling.
- **Merges and signs.** There are 8 merges in validation and test, and only 4 with a sign. Train has 2 entries with a sign. One merge moves a whole row of the table, and seeds of the same training differ by as much as the versions do. The numbers above are a trend, not a measure.
- **Long entry ramps.** On the 40 s two-lane entry ramp of the test, 7 of the 8 networks above release the 120 on the ramp, between 4 and 29 s before the merge. The sign's age alone cannot place the merge, because entry ramps here last from 8 to 40 s.

### Third version: the junction head (29 October 2026)

**Why.** In Poland, and in Switzerland by OSStr art. 16 cpv. 2, a speed sign holds at most to the end of the next intersection. Poland has no end sign: the car must know by itself that it crossed an intersection, so that the limit goes back to the general one. The map gives the labels; the camera, with the network's memory over the seconds before, gives the answer on the car. The comma reads nothing new from the map.

**The labels, from the map** (`src/nnslr_tools/junctions.py`, `nnslr junctions --all`, step 3 of the script; `data/derived/junctions/<route>.json`). For every frame of every drive: `in_junction` (the car is inside an at-grade intersection of public roads) and `passed` (the first frame after it left one). From the GPS track and the road the car is on (`map_context.our_road`):
- an intersection is a node of the map where at least three arms of public roads meet: trunk, primary, secondary, tertiary, unclassified, residential, living_street and their `_link`. A node where only motorways, expressways and their ramps meet is the diverge or merge of an interchange, never an intersection;
- service roads, tracks, paths, footways, cycleways, pedestrian ways, steps, bridleways and roads with `access` or `motor_vehicle` private (or no, agricultural, forestry, customers, delivery, permit, emergency) are not roads here: the law leaves out driveways and the exits of internal and unpaved roads;
- two arms of one road within 50° of each other, one of them one-way, are one arm: the start of a dual carriageway, a lane that splits and joins again, a roundabout's bypass;
- a roundabout is one intersection, from where the car enters it to where it leaves it. Nodes closer than 25 m of road, or joined by a turning lane of at most 80 m, are one intersection (a dual carriageway crossed, a staggered crossing), at most 60 m across;
- a bridge or a tunnel shares no node with our road, so it is never an intersection by itself. The car passes an intersection when the track comes within 12 m of a node of it and the road it is on, within 3 s, is one of its roads; a passage near a node of another road (a bridge above, a road beside) is kept apart as "maybe" for the review page.

**The area** of an intersection along the track: from a radius before its first node to a radius after its last one. The radius is half the width of the widest road (its `lanes` × 3.25 m, or 2 lanes, 1 for a one-way road or a ramp) plus 4 m for the stop line and the corner, kept between 10 and 20 m: 10 m for two two-lane roads, 10.5 m with a four-lane road, 13.75 m with six. The distance along the track is the integral of the GPS speed (a car waiting at a light does not move; its GPS position does), and each node is placed at the track's closest point to it. A frame is `in_junction` when its time is inside the area; `passed` is the first frame after the area, within 1.5 s.

| Drive | km | Intersections | Roundabouts | Passed | Maybe (another road) | Frames inside |
|---|---|---|---|---|---|---|
| 0000011e | 9.3 | 14 | 4 | 14 | 0 | 60 |
| 0000011f | 9.6 | 12 | 3 | 12 | 1 | 56 |
| 00000122 | 9.0 | 12 | 4 | 11 | 0 | 70 |
| 00000123 | 10.8 | 15 | 3 | 15 | 1 | 173 |
| 00000125 | 39.2 | 20 | 0 | 20 | 3 | 113 |
| 00000126 | 36.2 | 27 | 4 | 27 | 5 | 175 |
| 00000128 | 17.7 | 43 | 4 | 42 | 1 | 168 |
| 0000012a | 40.8 | 29 | 0 | 29 | 8 | 179 |
| 0000012c | 40.6 | 25 | 0 | 24 | 6 | 62 |
| 0000012d (test) | 40.2 | 21 | 0 | 21 | 7 | 115 |
| 0000012e | 39.7 | 26 | 0 | 26 | 7 | 68 |
| 00000147 | 5.4 | 18 | 0 | 17 | 0 | 91 |
| **All** | **298.5** | **262** | **22** | **258** | **39** | **1,330** |

Four passages have no `passed`: the GPS stops inside them. Every passage has at least one frame inside.

**Checked by eye** before training: 24 intersections on sheets of their frames (before, inside, passed) with the map around them. The first 12 had 3 wrong: a lane that merges back, and the bypass of a roundabout counted as two intersections. The rule of the one-way arms within 50° fixes them; the second 12 were all right. Two things the labels do on purpose: a car queued at a red light within the radius is "inside" (`frames_inside` up to 111 s), and a car that takes a roundabout's bypass passes no intersection. A test of the labeller then found that the end of a track segment, clamped, counted as a touch of the node and widened areas by up to 12 m a side; version 2 places each node at the closest point only (1,799 frames inside became 1,330).

**1 Hz.** The derived frames are one a second, so 2 Hz would need the 394,000 frames of the drives decoded again from HEVC. It is also not needed for the labels: at 50 km/h a two-lane crossing (20 m) lasts 1.4 s, and every passage of the 12 drives has at least one frame inside. The road type's memory runs at 1 Hz on the car, and a faster step would make its memory faster too. So the junction head stays at 1 Hz, and 2 Hz was not measured.

**The network.** The same encoder and memory, one more head: `Linear(128 → 2)` on the GRU's state, the sigmoid of `in_junction` and `passed`. Its loss is a binary cross-entropy beside the road type's, positives weighed by √(negatives / positives); the `passed` output is not taught 1 s around a labelled "passed", since the map's timing is ±1 s. A fifth of the train windows are cut around a "passed". The best epoch is chosen by the mean of the road-type macro F1 and the junction F1 on validation. The version without memory has the same head per frame (128 → 128 → 2). Frames without a GPS fix have no junction label and are not taught.

**The measure** (`road_type_metrics.junction_metrics`; each drive of test run whole from zeros, as on the car). A "passed" event is when the `passed` probability rises to the threshold (decode `passed`), or when the `in_junction` probability falls below it (decode `exit`), at most one event every 3 s. An event matches a labelled "passed" within 2 or 4 s, one to one. An event matched to nothing within 4 s is a **false "passed"**, the dangerous error since it lifts a limit; it is counted per hour and per 10 km, in town and outside (by the frame's road type). Events in the 4 s after a passage whose exit was not seen are not judged. The car's decode and threshold are those with the best F1 at 2 s on validation.

**Results** (run `2026-10-29_1412-road-type-junction`). Test is the areas of the held-out drive. It holds 83 intersections passed in 1.59 h and 92.5 km: 0.49 h and 11.7 km in town, 1.10 h and 80.8 km outside. Validation has only 28, in 0.50 h, with 6 minutes of town. Each version runs at its operating point, chosen on validation. "Needed" counts the intersections of the test that come within 300 m after a number sign the car's own code confirmed (the replay of `SpeedLimitMemory` over the sign readings of step 4). Those are the ones where, in Poland, the limit would end. 35 of the 83 are needed.

| Test, at the operating point | Rule | Recall 2 s | Recall 4 s | F1 2 s | False "passed" /h | /10 km | Town /h, /10 km | Outside /h, /10 km | Needed caught (2 s, 4 s) |
|---|---|---|---|---|---|---|---|---|---|
| With memory and the sign (epoch 14) | exit, in < 0.3 | 30.1 % | 31.3 % | 0.281 | 43.4 | 7.5 | 91.4, 38.6 | 21.8, 3.0 | 11, 12 of 35 |
| With memory, no sign (epoch 16) | passed ≥ 0.3 | 15.7 % | 20.5 % | 0.182 | 27.0 | 4.7 | 75.6, 31.9 | 5.4, 0.7 | 7, 8 of 35 |
| Without memory (epoch 37) | exit, in < 0.3 | 26.5 % | 34.9 % | 0.216 | 57.8 | 10.0 | 138.5, 58.3 | 21.8, 3.0 | 9, 12 of 35 |

The network with memory and the sign at three thresholds (test):

| Rule | Recall 2 s | Recall 4 s | False /h | /10 km | Town /h | Outside /h |
|---|---|---|---|---|---|---|
| passed ≥ 0.3 | 38.6 % | 43.4 % | 49.0 | 8.4 | 115.8 | 19.1 |
| passed ≥ 0.5 | 26.5 % | 28.9 % | 32.1 | 5.5 | 75.2 | 12.7 |
| passed ≥ 0.7 | 21.7 % | 24.1 % | 21.4 | 3.7 | 52.8 | 7.3 |
| exit, in < 0.3 (the operating point) | 30.1 % | 31.3 % | 43.4 | 7.5 | 91.4 | 21.8 |
| exit, in < 0.5 | 25.3 % | 27.7 % | 37.7 | 6.5 | 85.3 | 16.4 |
| exit, in < 0.7 | 18.1 % | 21.7 % | 28.3 | 4.9 | 67.0 | 10.9 |

- **It is not usable on the car.** At the operating point it finds 3 intersections in 10 and says "passed" falsely 43 times an hour: 91 an hour in town, more than one a minute. Each false "passed" would lift a limit. The strictest threshold still makes 21 false events an hour, and finds a fifth of the intersections.
- **The memory helps a little.** At the same rule (exit, in < 0.3), with memory: 30 % recall at 2 s and 43 false an hour. Without memory: 27 % and 58 false. Inside the intersection (frames, at 0.5), F1 is 0.24 with memory and 0.18 without. The memory makes the moment sharper, not the finding: within 4 s the version without memory finds more (35 % against 31 %).
- **Validation says much more than test**: F1 0.62 and 24 false an hour on validation, 0.28 and 43 on test. Validation has 28 intersections and 6 minutes of town, so the choice of epoch and threshold rests on very little.
- **The false events were not looked at yet.** Some may be real intersections the rules leave out (a car park exit, a service road, a farm track), which look like one from the camera. That would make the measure stricter than the law, but the car still could not tell them apart.

**The road type changed too.** The same encoder and memory now serve two heads, and the best epoch is chosen on the mean of both F1. Against the second version (`2026-10-29_1231-road-type`, with the sign; test):

| | Macro F1 | Ramp F1 | After a merge | Ramp limit ends in time (of 7) | Never held | False switches /h (raw, debounced) | Other roads called ramp |
|---|---|---|---|---|---|---|---|
| v2, with the sign | 0.561 | 0.000 | 90.7 % | 2 | 4 | 80, 48 | 16 s |
| v3, with the sign and the junctions | 0.623 | 0.105 | 62.4 % | 3 | 3 | 171, 69 | 202 s |
| v3, no sign | 0.633 | 0.076 | 79.1 % | 1 | 4 | 157, 66 | 159 s |
| v3, without memory | 0.590 | 0.030 | 85.4 % | 1 | 4 | 498, 70 | 69 s |

The third version says ramp sometimes, for 29 of the 319 ramp seconds of test, and holds one interchange in time. But it also calls 202 s of other roads a ramp, its answer switches twice as often, and after a merge it is right 62 % of the time instead of 91 %. A first run of this version, killed before it finished and trained with the first labeller's wider areas, chose a later epoch: after a merge 98.6 %, macro F1 0.684, junction F1 0.214 with 25 false an hour. The epoch the junction head picks moves the road type a lot. **For the road type, the second version stays the one to use.**

**On the car.** The ONNX step has one more output, after the road type's, which keep their names and order: `probabilities` (1, 6), `hidden_out` (1, 128), `junction` (1, 2) = sigmoid of `in_junction` and `passed`. The contract has `outputs.junction` (names, the decode, the operating point chosen on validation), `step.junctions` and `car_input.junction_decode`, the line the car runs. Checked against PyTorch with onnxruntime: largest error 5e-6 on random inputs, 9e-6 over 60 s of a real drive with the same answer in all 60. The network has 1,102,504 parameters (258 more); a step still costs 84 MFLOPs.

**Corrections in Data › Roads.** The review page of a drive shows the map's intersections as dots under the strip: filled for an intersection, an orange ring for one to check (the GPS stops inside, the track passes more than 8 m from it, the car stays more than 30 s inside, or a "maybe" on another road), a gray ring for one dropped. Next to review stops on the rings. Under the road-type buttons, "Junction here" and "Not a junction" (and Undo) correct the picture shown. The corrections are appended to `annotations/junctions/<route>.jsonl` (the last line wins) and are never touched by a new map run; the training labels are the map's with the corrections over them. Today 56 are to check: 39 "maybe" and 17 of the map's intersections.

**What is missing.**
- **Look at the false "passed"**: the 69 of test at the operating point, on the frames. Are they driveways and car park exits (the rules leave them out, the camera does not), stretches the map lacks, or plain mistakes? That decides between more rules, more data and another input.
- **More intersections in validation**: 28, and 6 minutes of town, are not enough to choose an epoch or a threshold.
- **The epoch for two heads**: choose the road type's epoch by its own measure, and train or fine-tune the junction head apart, so that one does not spoil the other. Done in the fourth version.
- **A person's pass over the 56 to check** in Data › Roads, then train again.
- **Poland**: there is no Polish drive yet. The head is trained on Swiss intersections, which look alike but are not the same.
- **The car**: nothing uses it, and it should not until the false "passed" are rare. `SpeedLimitMemory` would end a sign's number at a "passed" only where the law says so (Poland; in Switzerland, some signs).

### Fourth version: the car's own motion (29 October 2026)

**Why.** A car that turns at a crossing has passed it: the gyro knows that better than the camera. So the network now also gets the car's own motion, next to the speed and the sign, to tell when the car turned at a junction. Only the car's own signals: the camera's side never reads the map (`AGENTS.md`). The map only makes the labels, offline.

**The inputs** (`src/nnslr_tools/car_motion.py`, `nnslr car-motion --all`, step 4 of the script; `data/derived/car_motion/<route>.json`). From each drive's `rlog.zst`, one row per 1 Hz frame, the samples of the last second (t − 1 s, t]:

| Input | From (cereal) | Each second | Scale |
|---|---|---|---|
| `heading_change` | `livePose.angularVelocityDevice.z` (named `deviceMotion` in this sunnypilot; the device z points down, so yaw = −z) when `.valid`; else `carState.yawRate` | the mean yaw rate in deg/s, i.e. the heading change over the step in degrees (left positive) | / 30, clipped to ±3 |
| `yaw_max` | the same | the largest \|yaw rate\| | / 30, 0 to 3 |
| `yaw_known` | | 1 with at least 10 samples | |
| `steer` | `carState.steeringAngleDeg` | the mean (left positive) | / 180, clipped to ±4 |
| `steer_known` | | 1 with at least 10 samples | |
| `left_blinker`, `right_blinker` | `carState.leftBlinker`, `rightBlinker` | 1 when on at any sample of the last 2 s | |
| `blinker_known` | | 1 with at least 10 samples | |

An unknown input is 0 with its flag 0. All 12 drives have it: the yaw from the device 100 % of the seconds (99.9 % of the frames), `carState` 100 %. This car fills `carState.yawRate` with 0, so the fallback was never used. The speed input is now `carState.vEgo` (the GPS speed where it is missing), with the same scaling: `[min(max(v, 0), 60) / 30, 1]`.

**The dropout.** As for the sign: in 20 % of the train windows all the motion is unknown (a car or a log without it), in 10 % one part (the yaw, or the `carState` signals).

**The selection** (item 3 of the brief: the junction head must not drag the road type down). The road type's epoch is chosen on its own measure on validation, `road_score` = macro F1 + 0.1 × the first minute after a merge right (debounced) − 0.0005 × false switches an hour (debounced); the junction head never chooses it, nor stops the training. Then the junction head gets a stage of its own (`train_junction_stage`, up to 30 epochs, half the windows around a "passed"): everything the road type uses is frozen, and the head is trained with a GRU of its own, copied from the road GRU. It is kept only when it beats the shared head of the joint training on validation (F1 at 2 s). With memory and motion: road epoch 43 of 58 (validation road score 0.756, macro F1 0.675), and the junction head kept its own memory (validation F1 0.552 against 0.516 shared, stage epoch 4). Without the motion: road epoch 19 of 34, shared head kept (0.567 against 0.561). Without memory: epoch 42 of 57, its head retrained by the stage (0.344 against 0.298).

**Results** (run `2026-10-29_1533-road-type-motion`, the same split and the same test as the third version: 83 intersections passed in 1.59 h and 92.5 km, 0.49 h and 11.7 km in town, 1.10 h and 80.8 km outside). Each version at its operating point, chosen on validation:

| Test, at the operating point | Rule | Recall 2 s | Recall 4 s | F1 2 s | False "passed" /h | /10 km | Town /h, /10 km | Outside /h, /10 km | Needed caught (2 s, 4 s) |
|---|---|---|---|---|---|---|---|---|---|
| v3, with memory and the sign | exit, in < 0.3 | 30.1 % | 31.3 % | 0.281 | 43.4 | 7.5 | 91.4, 38.6 | 21.8, 3.0 | 11, 12 of 35 |
| v4, with memory, the sign and the motion | passed ≥ 0.5 | 22.9 % | 24.1 % | 0.279 | 20.7 | 3.6 | 47.0, 19.8 | 9.1, 1.2 | 6, 7 of 35 |
| v4, the same without the motion | exit, in < 0.3 | 32.5 % | 38.6 % | 0.289 | 45.2 | 7.8 | 110.6, 46.5 | 16.3, 2.2 | 9, 10 of 35 |
| v4, without memory (sign and motion) | exit, in < 0.7 | 13.3 % | 22.9 % | 0.135 | 38.3 | 6.6 | 99.9, 42.0 | 10.9, 1.5 | 4, 8 of 35 |

At the same rule (test):

| Rule | v3 | v4, with the motion | v4, without the motion |
|---|---|---|---|
| passed ≥ 0.3: recall 2 s, 4 s; false /h (town, outside) | 38.6 %, 43.4 %; 49.0 (115.8, 19.1) | 28.9 %, 30.1 %; 24.5 (55.2, 10.9) | 30.1 %, 39.8 %; 46.5 (122.9, 12.7) |
| passed ≥ 0.5 | 26.5 %, 28.9 %; 32.1 (75.2, 12.7) | 22.9 %, 24.1 %; 20.7 (47.0, 9.1) | 26.5 %, 32.5 %; 36.5 (94.3, 10.9) |
| passed ≥ 0.7 | 21.7 %, 24.1 %; 21.4 (52.8, 7.3) | 16.9 %, 18.1 %; 14.5 (32.7, 6.4) | 22.9 %, 26.5 %; 27.0 (67.6, 9.1) |

**The same network with its motion unknown** (`test_motion_unknown` in `result.json`, the last block of `comparison.txt`): passed ≥ 0.5, recall 21.7 % instead of 22.9 %, 20.1 false an hour instead of 20.7 (town 45.0 instead of 47.0); macro F1 0.661 instead of 0.664, the first minute after a merge 96.2 % both. Without memory, the same: 38.3 false an hour both.

- **The fourth network makes half the false "passed"**: in town 47 an hour at its operating point against 91 (v3) and 111 (the same run without the motion); at passed ≥ 0.3, 24.5 an hour (town 55) against 46.5 (town 123) without the motion, with about the same recall within 2 s (29 % against 30 %). It also finds fewer: 23 % at its operating point, and 6 of the 35 intersections that followed a number sign, against 11 for v3.
- **But the motion is not what cuts them.** The network barely uses it: with the motion unknown it gives the same numbers. The difference with the run without the motion comes from the rest of its training (its own junction memory, another epoch) and from chance; with one run each, it cannot be split further.
- **Why the turn tells little here**: 75 of the 83 intersections of test were crossed straight (the car turned 60° or more in 8, 30° or more in 15; in train 36 of 138, in validation 5 of 28). "No turn" says nothing against a false "passed", and the turned ones the camera already sees: the network with the motion caught 5 of the 8 turned (6 within 4 s), the one without it 6.
- **Still not usable on the car**: 20 false "passed" an hour, 47 in town.

**The car's turn alone, without the network** (`road_type_metrics.turns`, `turn_events`; `turn_rule` in `result.json`). Geometry only, never a speed threshold: a run of seconds turning the same way at 2 deg/s or more; its heading change is the sum of the yaw over the run, its radius the path driven over the run (the speed integrated) divided by the heading change in radians. The event is the second after the run, when the heading changed by at least 60° on a radius of at most 40 m. On the same test:

| Turn | Radius | Events | Recall 2 s, 4 s | False /h (town, outside) |
|---|---|---|---|---|
| ≥ 45° | ≤ 40 m | 19 | 3.6 %, 4.8 % | 9.4 (28.5, 0.9) |
| ≥ 60° | ≤ 25 m | 7 | 0, 0 | 4.4 (14.3, 0) |
| **≥ 60°** | **≤ 40 m** | **13** | **3.6 %, 4.8 %** | **5.7 (16.3, 0.9)**, 1.0 /10 km |
| ≥ 60° | ≤ 60 m | 19 | 4.8 %, 7.2 % | 8.2 (18.4, 3.6) |
| ≥ 90° | ≤ 40 m | 9 | 1.2 %, 2.4 % | 4.4 (12.2, 0.9) |

The rule finds 3 of the 83 intersections within 2 s (4 within 4 s) and says "passed" falsely 9 times in 1.59 h, 8 of them in town. It can see at most the 8 turned crossings, and finds 3 or 4 of those; why it misses the others (the timing of its event, the radius, a roundabout's turns both ways) was not looked at. **What the network adds** are the straight crossings, which no turn can see: 15 of the 75 within 2 s (17 within 4 s) with the motion, 23 without it, 20 for v3. The rule's 9 false events were not looked at yet (a turn into a car park or a driveway would be one).

**The road type** is the best so far (test):

| | Macro F1 | Ramp F1 | After a merge (raw, debounced) | Ramp limit ends in time (of 7) | Released on the ramp | Never held | False switches /h (raw, debounced) | Other roads called ramp |
|---|---|---|---|---|---|---|---|---|
| v2, with the sign | 0.561 | 0.000 | 90.7 %, 90.9 % | 2 | 1 | 4 | 80, 48 | 16 s |
| v3, with the sign and the junctions | 0.623 | 0.105 | 62.4 %, 61.3 % | 3 | 1 | 3 | 171, 69 | 202 s |
| **v4, with the sign and the motion** | **0.664** | **0.127** | **97.0 %, 96.2 %** | 1 | 2 | 4 | 88, 43 | 36 s |
| v4, without the motion | 0.647 | 0.069 | 90.9 %, 91.5 % | 1 | 2 | 4 | 81, 37 | 182 s |
| v4, without memory | 0.577 | 0.055 | 71.2 %, 74.7 % | 1 (and 1 late) | 1 | 4 | 589, 80 | 144 s |

Its own selection fixed what the junction head had spoilt in v3: after a merge 96 % instead of 61 %, 43 false switches an hour instead of 69 (v2: 48), 36 s of other roads called ramp instead of 202. Town F1 0.880, rural 0.737, motorway 0.912; town entries 11 of 11 at 2 s, exits 9 of 10 at 1 s. The motion matters little here too (macro F1 0.661 with it unknown). **The comma's rule on the ramp's limit is not better**: on the 7 merges of test, 1 ends in time and 2 too early on the ramp (v2: 2 and 1, v3: 3 and 1). Seven merges are too few to choose on, and the ramp is still mostly missed (F1 0.13). With the signs the car's code confirms today (0 motorway signs on the 12 drives, read again after `/root/sunnypilot` changed), v4 holds one ramp limit by its own ramp answer and releases it on the ramp; v3 held one in time.

**On the car.** The ONNX step has the motion as one more input and, since the junction head has its own memory, one more input and output for it: inputs `image` (1, 3, 144, 256), `speed` (1, 2), `sign` (1, 2), `motion` (1, 8), `hidden` (1, 128), `junction_hidden` (1, 128); outputs `probabilities` (1, 6), `hidden_out` (1, 128), `junction` (1, 2), `junction_hidden_out` (1, 128). The car feeds each hidden output back as the next step's hidden input, both zeros at the start of a drive (and after more than 5 s without a step). The contract has `car_input.motion`: the field names and order, the cereal fields, the summary (step 1 s, at least 10 samples, the scales and clips, the blinkers' 2 s), and `never: a map field`; `car_input.steps` has the lines `speedvisiond` would run (subscribe to `livePose`/`deviceMotion` at 20 Hz and `carState` at 100 Hz, summarise the last second); `car_input.junction_decode`: passed ≥ 0.5, rising, at most one event in 3 s. Checked with onnxruntime: largest error 1.0e-6 on random inputs, 1.1e-5 over 60 s of a real drive (the motion known all 60 s, the sign 30 s), the same answer in all 60. 1,209,256 parameters: 103,680 for the junction head's own GRU, 3,072 for the motion in the road GRU; a step still costs 84 MFLOPs. Nothing on the car reads it yet: `/root/sunnypilot` was not touched.

**What is missing.**
- **Make the turn count, or drop it.** The network learnt to do without it. To try: the turn of the last 5 to 10 s straight into the junction head, more turned crossings (most are in town), or a check by eye of whether the false "passed" happen while the car turns at all. If not, the motion can go and the input stays 2 + 2.
- **Look at the false "passed"**: 33 of v4 on test at its operating point, and the 9 of the turn rule.
- **The ramp's limit**: 7 merges in test, 1 in validation; more motorway entries are needed to choose between v2, v3 and v4 on the comma's rule.
- **More intersections in validation** (28), **Poland**, and **the car** (as in the third version).

### The acceleration lane (30 October 2026)

**Why.** On the comma the 60 of an entry ramp gave way to the motorway's 120 a median 13 s after the merge: 8 s of "motorway" in a row (`MERGE_SECONDS`), the 3 steps of the debounce and the network's own lag. The driver needs the 120 earlier, where the acceleration lane starts, to speed up and join the traffic; the map gives it only at the end of the ramp.

**The class `merge`.** A ninth road type, added last so older checkpoints keep their indices (`--init-from` maps the head by name). Its labels come from the map, offline (`road_types.merge_lanes`): on a ramp that runs straight into a motorway or an expressway (`motorway_link` or `trunk_link`, then the carriageway with no gap), the last `MERGE_LEAD_M` = 50 m of the ramp and the first `MERGE_LANE_M` = 200 m of the carriageway, measured with the car's GPS speed. In Switzerland the map's merge node is at the end of the gore, where the lane starts (checked by eye on 0000012d). The reason is "acceleration lane (…)" with the reason of the frame under it; the frames stay as sure as before. An exit is not a merge. A missing second ends the lane there. The lane's frames on the ramp are its "lead" (`road_types` writes `"lead": true`).

**The slips (review of 31 Oct).** 14 of the first 31 lanes were the GPS on a link beside the carriageway for 4-8 s as the car passed a junction: motorway, "ramp", motorway. A ramp shorter than `MIN_LINK_RUN` = 10 frames between two stretches of the same motorway or expressway is now smoothed into the carriageway (the links from one carriageway to another lasted 12-36 s). After the fix: 238 frames in 17 lanes (one of 47 s, slow traffic).

**The measures.** For the measures of the road (changes, after a merge, switches) a merge is folded into the carriageway that follows it and its lead into the ramp (`road_type_metrics.fold_labels`, `fold_merge`), so the old versions stay comparable; the F1 of each class stays raw. The car's debounce comes before the fold, and a change is found only on `CONFIRM_SECONDS` of one raw class in a row (merge, merge, motorway is no run for the car). `comma_release` follows the car: the first debounced "merge" releases the ramp's limit only when a carriageway sign holds it, else the lane counts towards the 8 s of carriageway; a sign no longer holds a number read after a merge since that sign.

**On the car** (`vision.SpeedLimitMemory`, sunnypilot branch `speed-limits`). With the ramp's limit held and the network saying merge, the general limit of the carriageway comes back at once, as a sign's, when the carriageway is known from its sign (a motorway or expressway sign confirmed within `RAMP_SECONDS` before the ramp's number). Without such a sign the lane only counts towards `MERGE_SECONDS`, and the carriageway the network names after it gives the limit. A merge without a ramp's limit held changes nothing. A number read on the lane, or after a ramp's limit was released, is the carriageway's even when its sign is near (`_joined_at`); the sign's kind is forgotten at a sign of another road and once the network has seen a carriageway for `ROAD_TYPE_SECONDS` since the sign (not in the first seconds, when the debounced road type is still the one before the sign), so a later ramp to the other kind is not given the first one's limit. `road_type_metrics.comma_release` follows the same rules, with the release of the ramp before as the join. The lane's seconds count for the carriageway's dwell (`ROAD_TYPE_SECONDS`). The map is still never read on the car.

**First training with the class (31 October 2026).** Run `2026-10-31_0007-road-type` had a validation without a single motorway frame: with eight classes the rare ones weighed as much as the motorway in the choice of the validation areas, and the buffer's cost decided. Its epoch was chosen blind to the motorway, and on test it said rural on the motorways (recall 0.11; 0.92 on its own training drives: it had learnt the drives). The split now never leaves out a class of a tenth of the frames (`road_type_data.MAIN_CLASS_SHARE`, policy `road-type-place-v2`, split `d3fb3f48a194`): validation 3,431 frames, 1,061 of them motorway, 155 ramp, 19 merge. The test is unchanged (40 sequences, 11,840 frames), so run `2026-10-31_0046-road-type` is measured against the version in use (`2026-10-30_1216-road-type`, measured again on today's data with `nnslr release-road-type`):

| test | in use (1216, with motion) | with memory and motion | with memory, no motion | without memory |
|---|---|---|---|---|
| accuracy | 0.789 | 0.742 | **0.811** | 0.648 |
| macro F1 | **0.433** | 0.391 | 0.429 | 0.350 |
| false switches/h (debounced) | 101.7 (44.5) | 70.6 (35.3) | **39.5 (24.9)** | 703.7 (94.1) |
| after a merge | 0.767 | 0.830 | **0.850** | 0.744 |
| F1 motorway / rural / town | 0.903 / 0.703 / 0.801 | 0.837 / 0.632 / 0.821 | **0.918 / 0.748 / 0.823** | 0.733 / 0.595 / 0.720 |
| F1 ramp / parking / 30 zone | **0.183 / 0.833** / 0.037 | 0.033 / 0.720 / 0.087 | 0.039 / 0.748 / 0.154 | 0.051 / 0.504 / 0.195 |
| ramp's limit (map signs, 9 merges) | on the ramp 2, in time 1, late 1, never held 5 | the same | on the ramp 2, in time 2, never held 5 | in time 3, late 1, never held 5 |

No version ever says merge (F1 0): the training has 48 merge frames. The version in use stays: only the one with memory and motion goes on the car (`car_release.ROAD_TYPE_ON_CAR`, the only ONNX step the training exports) and it is worse; the one without motion is better on most measures but loses the ramp (11 of 527 frames) and some of the car parks.

**A link right after an entry (31 October 2026).** Past a ramp's limit a number is the carriageway's, so the sign of an interchange's link, read within `RAMP_SECONDS` of the entry's motorway sign but before the debounced road type says ramp, stays on the next motorway: too low, until the next sign. At `74a0778ab` the car took it as the ramp's, but it took the carriageway's own 100 there as the ramp's too and gave 120. A rule that took the number back as the ramp's when the road type says ramp within 6 s of it was tried and dropped. The version in use never says ramp on a link between two carriageways (9 in test, ramp probability at most 0.05), so the rule changed none of the 9 test merges; and a false ramp on the carriageway right after its own number gave the general limit, too high. Neither case happens before the car confirms motorway signs, which it has not done on any drive so far.

**Still to do.** More motorway entries for the class merge; check the lanes of the map by eye in Data › Roads; let the version without the car's motion go on the car (its ONNX step exported, chosen on the Models tab).

### A limit read before the motorway (31 October 2026)

**Why.** On a drive the network missed a motorway entry and the car the end of a 40: it held the 60 read before, in town, onto the motorway, where only the next sign would end it.

**The rule** (`vision.SpeedLimitMemory`, `CARRIAGEWAY_LAG`). A number is remembered with the time it was read when the road type was neither motorway, expressway nor merge (`_number_at`). It ends, and the carriageway's general limit comes back as a sign's, once the debounced road type has said motorway or expressway for `ROAD_TYPE_SECONDS` (10 s), if that road type began at least `CARRIAGEWAY_LAG` (5 s, a setting in Models › On the car, `carriageway_lag`, 0 to 30 s) after the number: a number read in the last seconds before may already be the carriageway's own, since the network lags the road by a few seconds. A sign of a road, the end of a limit, a car park or the ramp's own release forget it; a number read on the carriageway or the lane is never ended this way. Roadworks are no exception in the code: their limit is read on the motorway itself, so it holds. The map is not read.

**The measure.** `comma_release` applies the same rule to every merge, whichever rule comes first; a number neither the ramp nor the sign holds but read off the carriageways is held `off` (`held_by_off` in the summary, "off the carriageway" in `release_text`).

### Leaving a car park (31 October 2026)

**Why.** Out of a car park the comma kept its limit (10 km/h) until the network had seen a road for `ROAD_TYPE_SECONDS` (10 s) after the debounce: 10 to 13 s on the street, the car held back while it should speed up.

**The rule.** A car park now ends after `PARKING_EXIT_SECONDS` (3 s) of a road with a general limit, a setting of its own in Models › On the car (`parking_exit_seconds`, 0 to 30 s); `ROAD_TYPE_SECONDS` still sets how long a new road type must last before its limit follows it elsewhere.

**The measure.** The car's own memory (`vision.SpeedLimitMemory`, no signs) on the answers of four scene networks on their validation and test drives, the debounce of the car included. With the version on the car (1216) the limit goes 3 and 6 s after the two labelled exits, against 10 and 13 s before; no network left a car park while the car was still inside at 3 s, where the run 0007 did once at 0 to 2 s. Some exits of the 0046 networks take 7 to 17 s even at 0 s: there the network itself still says car park on the street. Only 11 exits over the four measures, so the next drives out of car parks should be watched.

## The brain (4 November 2026)

**What it is for.** One network with memory that says the speed to hold, from the road camera and what was read, and generalises to roads it never saw: the end of roadworks with no end sign, a limit the reader got wrong, a road type no rule covers. It never reads the map. It does not exist yet as a trained network: this section is the road from a person's labels to its weights.

**From a label to the weights.** A label in the database changes nothing by itself:

```
Speed review  ->  answers on the frames  ->  errors of the network  ->  weights of the shared part
speed_review.sqlite   brain_targets.py        brain_net.brain_loss       brain_train.py
```

**The network** (`src/nnslr_tools/brain_net.py`): the road-type network's encoder, frozen on its ImageNet weights, its features through a Linear and a LayerNorm, with the frame's inputs, a GRU, then two heads on the same memory: `speed_head`, the only output on the car, a class of 5 km/h; `works_head`, are we inside roadworks, only in training. `car_weights` leaves it out. A mark is an answer, never an input, and no code goes from works to speed: "works ended" and "120" stay two separate answers.

```
loss = cross entropy(speed, his speed) + 0.3 * binary cross entropy(works, his marks)
```

Each term only on the frames with an answer of its own. The works error goes back through the GRU into the layer on the picture's features, the part the speed uses too.

**The answers** (`src/nnslr_tools/brain_targets.py`):

| Frame | Speed | Roadworks |
| --- | --- | --- |
| a person gave a speed | that class | |
| not examined, can't tell, before `known_ns` | out | |
| within 2 s of a change of speed, or of a works tap | out | out |
| inside a mark with an end | | 1 |
| from the drive's start to "checked to here", outside the marks | | 0 |
| anywhere else, and from the start of a mark with no end | | out |

A stretch with no mark is not verified, it is not "no works".

**Inputs.** Groups apart, all from what comes at or before the frame: `signs` (the number confirmed by the car's own rule, its kind and age, an end confirmed since, and the last single reading), `plate` (the panel below the number confirmed: unknown until the reader gives it), `hour` (local, from the GPS clock), `lead`, `country` (the last one a fix gave), and the driver's two, `pedals` and `speed`. All but the driver's are made by one file, `sunnypilot/speed_limits/brain_inputs.py` of the branch `speed-limits`, which the server loads by its path, so that what the network learns on is what it gets on the car; the checkpoint keeps its hash. Limits of some hours are not a rule in code: the network gets number, plate, hour and country, and decides.

**The hour in training.** Every labelled drive is by day, so the real hour would only say which drive a frame is from: every epoch a drive takes a clock moved by a random number of hours, one for all its frames, and validation and test keep the real one. A drive with a plate of hours on frames of train keeps its real clock. Such a drive, when his label is above the number on the plate, also goes through the epoch once more with its clock moved into the night (22 to 6) and the number as the answer: the same pictures, only in training, nothing on the car (`brain_targets.epoch_drives`). A label under the number has another reason, roadworks for one, and no copy. A label equal to the number by day takes the whole passage out, since that plate is no 22-6. Limits: a plate of other hours driven outside them (7-17 at 19:00) gets a wrong night answer, and the pictures are of the day. Real drives after 22:00 are added when they happen. The reader gives no plate yet, so today there is no copy: `train-brain` logs how many frames have a night answer.

**The driver's inputs.** Gas, brake and the car's speed stay out by default (`--driver` adds them): the gas is pressed on 96 % of the labelled frames of `00000168`, where he overrode a limit stuck too low, and the car's speed is what the network itself decides on the car. Measured on the first 4,251 labelled frames: own speed is within 10 km/h of the label on 34 % of them, so the risk is the pedal more than the speed.

**The split.** The frozen one, `data/splits/speed-brain-split.json`: on the place, never computed again. A later drive goes by the area of each frame. A works site with a frame in test or validation, 60 s around it included, has no frame in train, nor has another site on those areas. A picture of test or validation runs in no training step and in no memory of one: the memory starts again after it. Buffer frames only keep the memory going, under no_grad. What does cross is a number read there, as an input, as on the car.

**Sequences.** Whole drives in order, the state carried from piece to piece, so that approach, crossing and exit are one sequence. The memory starts again after 5 s without a frame, as on the car, and all the pieces of a drive get one look of the augmentation in an epoch, so that the state does not jump from piece to piece. Batch norm is frozen: its statistics would let later frames reach earlier ones.

**What the training refuses.** A drive with no reading of the signs or read by another reader than the car's of today, since an unread drive looks like a drive with no sign; a drive with no frame; a validation with no speed answers. When train has only marks and no stretch checked, the works task leaves the loss of train (it would learn to say yes everywhere) and stays in the measure. The checkpoint says what it came from: inputs, pool, split, reader.

**The measure** (`src/nnslr_tools/brain_train.py`), on places excluded from training: speed right and error in km/h; the same on frames with no gas, where the driver gives nothing away; the same where the car's speed is 20 km/h or more from the label, and how many of those frames the network puts nearer the car's speed, because the video shows how fast the car goes and leaving the inputs out is not enough; the same inside the marks; precision and recall of the works output; per site, how long after his end tap the speed reaches what he labelled, and how much of the mark was above his label. Every epoch also shows the speeds the network said and what two simple answers score on validation, one constant speed and the number in force on the frames that have one (`follow_the_sign`, with the network's own error on those same frames): the network has to beat both, or it adds nothing to the car's rule. Each result is also measured with the memory started cold where a stretch out of train begins (`validation_cold`, `test_cold`), since a memory warmed up on train frames can know the road. The comparison is the same training with and without `--no-works`, and with `--pool grid`, which keeps where things are in the frame (the mean over the 8 x 5 map loses most of it, and an occupied lane is a matter of where).

**The first run (5 Nov 2026, `data/runs/2026-11-05_0720-brain`) learned nothing.** It said 120 on all 9,522 frames: the 3.75 km/h off on validation is what a constant scores there, and test, 58.4 km/h off, is worse than a constant 50 (11.7). Its error stayed at 1.6, the entropy of the labels: it had learned only how often each speed comes. The encoder had moved: with batch norm frozen, one sequence a step and Adam, nothing brought its values back, and they grew 250 times over ImageNet's. The GRU saturated (its state moved 0.0002 across frames) and taking the signs away changed the logits by 0.0001. Hence the frozen encoder and the LayerNorm, in the network above; the road-type network, at the same learning rate with its batch norm moving, stays bounded. The LayerNorm is a new layer for the car's export: the parity with tinygrad and onnxruntime has to be run again before any work on the car.

**The second run (5 Nov 2026, `data/runs/2026-11-05_0845-brain`, frozen encoder and LayerNorm) learned the roads, not the signs.** Train reached 92 % right, 1.6 km/h off, error 0.08 (best epoch 17, stopped at 27). Test, 218 frames nearly all of one drive: 50 % right, 8.2 km/h off, against 11.7 for a constant 50. On the 149 test frames with a number in force the sign is right on 97 %, 0.13 km/h off, the network on 66 %, 3.3 km/h off. Taking the signs away changes nothing on those frames, and with a grey picture it says 120 everywhere: the speed comes from the picture. On train the number equals the label on only 44 % of the frames that have one, because the drives were chosen for the limit's errors, so the number looks unreliable, while the picture, on four drives, gives the label almost exactly. Validation cannot tell: it is one stretch of one drive, the network says 120 on all of it, and the epoch it picks means little. Roadworks taught nothing: train has no answer on them yet.

**What the comma's limit scores.** On the first 4,251 labelled frames `vis_limit` equals the label on 48 %, 20.6 km/h off on average. The labelled drives were chosen for their errors, so this is a starting point, not the limit's overall score.
