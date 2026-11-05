# Full-resolution sign reading: our view

5 November 2026. This says what we make of Astra's [design assessment](2026-11-05-full-resolution-sign-reader-design.md) and how recording and sign reading stay apart.

**Update, same day.** Cristian decided: build it, no measure first, the road decides. The verdict, the order of work and the second file below are the view of that morning and no longer hold. What is built: one `fcamera.hevc` at 2688 × 1520 with the switch on (no second file), the reader's crop from the full frame, and the server's side ([the guide](GUIDE.md#drives-at-full-definition)). What is left: `camerad`. A correction: the ISP does not write the road's full frame anywhere today. The road camera goes through the IFE, which gives only the 1344 × 760; the full output that exists is the cabin's, through the BPS.

## Verdict

Full resolution pays for plates such as "22-6". For plain speed limits it is not proven yet. The test on a parked comma is worth doing, but not first.

- **Plain limits.** A sign grows before the car passes it, and the limit holds from the sign onwards. Reading it 20 m earlier at 120 km/h gains 0.6 s. What counts is reading it at least once, not reading it from far away.
- **The missing number.** The 54 % is per frame, at about one frame a second, while the comma looks at the same sign many more times. We do not know yet how many signs the chain loses altogether. That number decides.
- **Plates.** Here resolution really matters: a digit of a "22-6" is 5 px tall at 20 m today.
- **Cost.** A `camerad` and `loggerd` change carried through every merge, buffer ownership, a second video file, a second cutter window and a new training of the reader. We want the numbers before paying for that.

## What Astra says

Astra says it is worth an experiment, not yet a permanent camera change. The sums and the code citations we checked hold.

- **Gain.** By day, 1.3-1.7 times the reading distance on signs already found, which is Astra's estimate and not a measurement. A 60 cm sign goes from about 34 m to 44-58 m. At night perhaps nothing. A "22-6" might be readable at 20 m by day and marginal at 30 m.
- **Limit.** With today's cutter the chain rises at most from 54 % to 70 %, because 30 % of signs are never found. Signs 40 px and wider already stop at 67 %, so resolution is not the whole problem.
- **First step.** A parked comma, 10-20 pairs of images from the same exposure, no models and no training. Stop there if the gain stays under 10-15 %. It needs a change to `camerad`, which Cristian runs on the comma.
- **Proportions.** The first test keeps the stretched 160 x 160. Padding is tried afterwards, in a training of its own.

## Three things that touch us now

1. **Our mistake.** The "with a plate" picture in the brief is a Swiss 60 with a double arrow, not the German 30 with "22-6". It proves nothing about time plates.
2. **Night copies.** A `time` plate does not tell 22-6 from 6-22, yet our night copies assume night. None exists yet, so no harm so far.
3. **The brain.** In training it gets readings from a reader that has probably seen those roads, so they are better than on new roads. That may be why it corrected none of the 42 wrong signs. This is not measured. The fix is readings from a reader trained without those roads, which can run here on the V100.

## Order of work

1. Measure on the server how many signs are lost, per sign and per width, cutter apart from reading.
2. Give the brain readings from a reader that has not seen those roads. It is today's limit and needs no comma.
3. Run the parked-comma test, which is small and says whether those pixels really exist.

Recording and the cutter come only after that.

## What is saved and what is read are two separate things

Cristian's point: what we save to disk for training is one question, what the reader crops from is another. Production goes back to normal logs, the reader does not have to change.

| Who | Gets | Changes? |
|---|---|---|
| Driving model | 1344 x 760 | No |
| Cutter | 1344 x 760, its 896 x 288 band | No, unless a second window is added later |
| Brain | 1344 x 760 reduced to 256 x 144, plus the reader's result | No |
| Reader | Only the pixels of the box, cut in memory from 2688 x 1520 | Yes, never saved |
| Disk | `fcamera.hevc` as today, at 2688 x 1520 when the switch is on | Only with the switch on |

**The camera side.** (Wrong in the first version, see the update above: the road's full frame needs the BPS.) What the reader adds is memory, about 140-170 MB by Astra's sums, and the copy of each crop. Neither is measured on the device.

**The real cost is disk and encoder, and only with the switch on.** At 10 fps it adds 2.7-4.5 GB an hour and 40.9 Mpixel/s to encode; at 20 fps, 4.5-9 GB an hour and 81.7 Mpixel/s. With the switch off the cost is zero. The encoder's spare capacity is not measured.

**The Developer switch fits the code.**

- The panel is `selfdrive/ui/sunnypilot/layouts/settings/developer.py`.
- `RecordFront` in `system/loggerd/loggerd.h` already turns one camera's recording on and off, read when the process starts.
- Each recording has its own output size: `qcamera` is 526 x 330, scaled by the encoder. So a definition in between looks possible, though it is not tested from a full-size input.
- `encoderd` skips no frames today, so 10 fps needs a small piece of code.

**What to keep in mind.**

- The switch changes the disk, not the reader. In production the reader still gets sharp crops, so it must be trained on sharp crops, and only drives recorded with the switch on give those. Drives recorded with it off still serve the brain and the low-resolution crops.
- The file on disk is compressed and the crop on the car is not, so the bitrate has to be tested on signs.
- All of this holds only if the full-size output is valid.
