# Drive review: what the comma's logs can tell

The drive review (Data › Drive review, `nnslr drive-review`) reads the comma's own logs of a drive, joins them to the
drive's video and looks for episodes worth a look. This page says, for each thing the review needs, whether the log
**has** it, whether it can be **derived** from what the log has, or whether it is **absent**, and what the comma's
telemetry would need to change to fill the gaps. The code is `src/nnslr_tools/drive_extract.py` (what is read),
`drive_logs.py` (files, merge, sync), `drive_episodes.py` (the scan) and `drive_review.py` (the store and the API).

Checked on the logs of the runtime branch (`speed-limits` of cristianku/sunnypilot, schema pinned by commit and
SHA-256 in `drive_logs.py`) and on two real drives of the PSA Peugeot 3008 with a comma 4 (`mici`):
`00000154--fa7e8f49aa` (20 minutes, openpilot in dashcam mode, vision and map on) and `00000141--153e33df1f`
(4 minutes, openpilot engaged, no speedVisionSP in its log). The tests use made-up logs only
(`tests/review/test_drive_review.py`, with a reduced schema in `tests/review/fixtures/drive_schema/`, both marked
SYNTHETIC).

## Where the logs come from

| Source | How | Notes |
|---|---|---|
| The comma's download | `helpers/fetch_comma_segments.sh` already copies `rlog.zst` next to every `fcamera.hevc`; found in `raw/routes/<route>/<segment>/` and in the shared folder | nothing new connects to the comma |
| The portal | **Send a folder of logs** / **Send log files** (comma connect names `<route>--<segment>--rlog.zst`) | kept unchanged in `raw/drive_logs/`, named by SHA-256; the same file twice is kept once |
| qlog | accepted, ranked after an rlog of the same minute | decimated: see below |

Every file is recorded with its drive, segment, kind, SHA-256, source, size and coverage (span and messages per
service). A drive without any log keeps its video review and shows *Logs not available: import the logs to look for
the episodes of the recorded drive.*

## What the review needs, and what the log gives

| Need | Status | From |
|---|---|---|
| Time base | **has** | `logMonoTime` (ns, monotonic, runs on across the minutes of a drive). Never file times or names |
| Video frame ↔ log time | **has** | `roadEncodeIdx`, on this car `narrowRoadEncodeIdx` (20 Hz): `segmentId` and `timestampSof` of every frame. A linear fit gives the precision: *frame* (within half a frame), *degraded* (holes in the index, or a video with another frame count), *no video*, *no index* |
| Date and hour | **derived**, only with a GPS fix | `gpsLocationExternal.unixTimestampMillis`. Without a fix the date is marked uncertain |
| Position | **has** with a fix | `gpsLocationExternal` (about 5 Hz, kept at 1 Hz) |
| Speed | **has** | `carState.vEgo` (100 Hz) |
| The comma's vision limit | **has** | `speedVisionSP.speedLimit` (about 4 Hz): the limit the memory holds and the screen shows, not the sign of one frame. 0 means "not sure" and is treated as missing |
| A sign confirmed | **has** | `speedVisionSP.signRead` marks the moment the memory took a sign; the value is the limit after it |
| Scene (road type) | **has** | `speedVisionSP.roadType` |
| The vision network in use | **has** | `speedVisionSP.modelName`: shown with each episode and kept with each verdict (`vision_model` in the export), so a label says which network was wrong. Drives before the first day reviewed (30 Oct 2026 by default, **Drives from** on the page) are left out: an older network is not worth judging |
| Map limit | **has**, needs a GPS fix | `liveMapDataSP.speedLimit` with `speedLimitValid`, the next limit ahead and its distance, `roadName` (1 Hz). Not valid → missing, never a disagreement |
| Limit the car used | **has** | `longitudinalPlanSP.speedLimit.resolver` (`speedLimit`, `source` none/car/map/vision, `speedLimitFinal`) |
| Speed Limit Assist | **has** | `longitudinalPlanSP.speedLimit.assist` (`state`, `active`), `longitudinalPlanSource`, `vTarget` |
| openpilot engaged | **has** | `selfdriveState` (`state`, `enabled`, `active`, `alertType`), `selfdriveStateSP.mads`, `carControl.latActive`/`longActive`, `carParams.dashcamOnly` |
| Set speed | **has** | `carState.vCruise` (km/h, 255 none), `cruiseState.speed` |
| Accelerator, brake | **has** | `carState.gasPressed`, `brakePressed` |
| Wheel held against openpilot | **derived** | `carState.steeringPressed` held 1 s or more while openpilot steered; shorter is a touch (a hint only) |
| Disengagement and its cause | **derived** | `selfdriveState` leaving engaged, with `onroadEvents` around it (`userDisable`, `softDisable`, `immediateDisable`, `buttonCancel`...) and the pedals |
| Set speed changed by hand | **derived on the PSA** | `carState.buttonEvents` is empty in the logs so far. A change of `vCruise` while ICBM was `inactive` for the whole second around it is the driver's (*Set speed by hand*); with ICBM in another state but no `sendButton`, or no ICBM in the log, it stays *changed by hand or by the car: the log does not say*. opendbc now fills `buttonEvents` on the PSA (below): logs recorded with it have the press itself |
| Set speed changed by ICBM | **has** | `selfdriveStateSP.intelligentCruiseButtonManagement.sendButton` |
| Why the vision limit changed | **absent** | the log has the new limit and `signRead`, not the reason (a sign, the end of a ramp, the scene, a timeout) |
| Why the vision limit holds | **has**, from 31 Oct 2026 builds | `speedVisionSP.speedLimitReason` (sign, end sign, road sign, ramp end, carriageway, car park, road type, zone 30) and `speedLimitState` (limit, none, unknown): the strip under the **Reads** lane and **Why** at the time shown. An older build leaves them unset: hatched, not in the log |
| Raw sign reads (class, value, confidence) | **has**, from 31 Oct 2026 builds | `speedVisionSP.rawSign*`: the last sign read the memory could take, at any confidence, and whether the limit was set from it: the ticks of the **Reads** lane (solid: taken, hollow: dropped, too unsure or overruled) and **Last sign read**. Its box is not logged; the server's networks on the extracted frames are shown in their own lane |
| The distance to the sign ahead | **has**, from 31 Oct 2026 builds | `speedVisionSP.signDistance` (m, what the screen showed) and `signSize` (px): **sign ahead in** at the time shown |
| The map as the car had it | **absent** | only its limit is logged, not the map version or the way; the "map today" lane is the local OSM of today, kept apart |
| Speed Limit Source setting | **not read** | the review shows the resolver's source instead |
| The driver's reason | **absent** | only the review can say it; the driver's speed is a hint, never a limit |

On the two real drives: `00000154` has vision, map and resolver for the whole 20 minutes but `dashcamOnly` is set,
so openpilot never drove and only the vision-against-map channel can find something (7 disagreements). `00000141`
was engaged (5 brakes, 5 disengagements by the driver) but its log has no `speedVisionSP` at all, so no vision limit; its 9 set
speed changes had no `buttonEvents`; with ICBM idle around them they now count as the driver's.

## Limits of the reading

- **qlog**: a decimated copy of the rlog. A short pedal press or a brief change can fall between its messages, and its
  encode index is sparser; the scan uses the rlog whenever there is one, and says which file it read.
- **Holes**: a missing minute, or a service that goes quiet (more than 0.5 s for carState, 1.5 s for the plan, 3 s
  for vision and map), ends its values with *missing*; nothing is carried over a hole. A minute whose clock goes back
  is left out and reported.
- **The vision ahead of the map**: the vision reads a sign tens of metres before the car reaches it; the map changes
  only where its limit begins (at the sign or past it) and at its next update. A disagreement that began with a new
  vision limit and ended within 10 s with the map on that same limit is no episode (counted as `vision_ahead`).
- **Missing is not wrong**: no vision, vision not sure (0), or no valid map limit is shown hatched and never counts as
  a disagreement.
- **Episodes** are clues for a person, not labels. A rule or a model may only propose; only a person's verdict is a
  target in `nnslr drive-review export`, and only when it is certain and has an interval.

## Telemetry changes that would help

None of these is made by the drive review: each is a change to the comma's software, for a later decision.

1. **Cruise buttons on the PSA**: done in opendbc `048c22ae` (`psa-torque-sunny-testing` of cristianku/opendbc), in
   the car's logs once sunnypilot's test branch is rebuilt with it. The stalk is wired to the BSI and is on none of
   the comma's buses (every bit of buses 0, 1 and 2 checked against 14 presses of `00000141`); the press is read back
   from what the BSI applied to `HS2_DAT_MDD_CMD_452`: +/- (one press per tap, one per 5 km/h step while held), set
   (255 to the current speed), resume (255 to another setpoint, a heuristic), cancel (to 255 without the brake; with
   the brake it is `brakePressed`, not a button) and main (the stalk wheel: off, limiter, cruise). On the car it only
   adds alerts: Speed Limit Assist with openpilot longitudinal ignores the buttons, MADS reads the main switch from
   `cruiseState.available`, and the set speed still comes from the car (`pcmCruiseSpeed`).
2. **Why the vision limit changed**: done in the 31 Oct 2026 builds (`speedLimitReason` and `rawSign*`, see above).
3. **Unknown versus not sure**: done in the 31 Oct 2026 builds (`speedLimitState`).
4. **The map the car used**: the map data version and the way (or segment) of `liveMapDataSP`, so "the map then" can
   be compared with "the map today".
5. **The Speed Limit Source setting** and the offset in use, in the log (or read from the params of `initData`), so the
   resolver's choice can be explained.
6. **A date without GPS**: read the log's wall clock (the `clocks` service) when there is no fix, if the comma's clock
   was set.
7. **A driver's flag**: the comma's bookmark pressed while driving as a third channel ("the limit is wrong here").
