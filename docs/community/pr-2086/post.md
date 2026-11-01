# Speed Limit Assist: fixing advance adaptation to lower map limits — PR #2086

**TL;DR**: when the map knows that a lower limit is coming, sunnypilot is meant to start slowing down *before* the change point. On master this never happens, because of a clock mix-up. [PR #2086](https://github.com/sunnypilot/sunnypilot/pull/2086) fixes it and makes the early switch hold. The PR is open, not merged yet. The numbers below come from replaying the resolver on my logged drives, not from road tests of the fixed code.

## What should happen

mapd publishes the next limit and its distance (the "50 in 220 m" on the screen). When the next limit is lower than your speed, the resolver should switch to it once the distance is shorter than the braking distance at 1 m/s² (`LIMIT_ADAPT_ACC`), so the car starts slowing down before the sign.

## What happens on master

The resolver mixes two clocks (there is a `FIXME-SP` above that code):

```python
time.monotonic() - gps_data.unixTimestampMillis * 1e-3   # about -1.79e9 s on the device
```

Seconds since boot minus Unix time. The distance driven since the GPS fix comes out at about −1.79 billion seconds × speed, so the distance to the limit ahead never gets shorter than the braking distance: **the early switch never happens**, and braking only starts at the change point. The 10 s "map data too old" check never fires either. The unit tests mocked the GPS time with `time.monotonic()`, which hid the bug.

A second issue shows up once the clock is right: the braking distance came from the car's speed, so when the car slowed down a bit faster than the curve, the resolver went back to the higher limit until the change point.

## The fix (2 small commits)

1. One clock: the GPS age comes from `sm.recv_time`, and the distance driven is counted from when `liveMapDataSP` arrived (mapd measures the distance when it sends it, at 1 Hz).
2. The braking distance comes from `max(v_ego, current limit)`, so the switch holds once the car has slowed down.

New tests cover the early switch beyond and within the braking distance, with the distance driven since the map data, and holding once the car is slower. The new cases fail on master.

## Replayed on real drives

I replayed 5 of my logged drives through the resolver (policy Map Data only), covering 31 drops of the map's limit. The replay feeds the logged speed and map data to the resolver: it shows when the resolver switches, not how the car would have driven.

![Replay of 5 logged drives](https://raw.githubusercontent.com/cristianku/sunnypilot.speedlimits/main/docs/community/pr-2086/replay.png)

- **master**: 0 early switches.
- **Clock fix only**: 9 early switches, and 6 of them went back up to the higher limit before the change point.
- **Both fixes (the PR)**: 27 of 31, from 4 to 14 s and up to 300 m before the change. One went back up, when mapd itself stopped announcing the limit ahead. In the other 4, mapd never announced the lower limit in advance, so there was nothing to switch to.

## What the car does with it

The resolver plans the switch for 1 m/s², but the longitudinal planner slows down for a lower cruise speed at most at 0.5 m/s² (`A_CRUISE_MIN`). So the car starts braking earlier and is already slower at the sign, but it still reaches the new limit after the change point, just much sooner than on master. A simple simulation, not a logged drive, of an 80 → 50 change:

![80 to 50: master and PR #2086](https://raw.githubusercontent.com/cristianku/sunnypilot.speedlimits/main/docs/community/pr-2086/sim.png)

Speed Limit Assist's own target acceleration is not wired to the planner yet (a `TODO-SP` in `speed_limit_assist.py`); this PR does not change that.

## What it does not change

- **Higher limit ahead** (50 → 60): no early switch, by design. You do not speed up before the sign.
- **The car's own limit and the other sources**: untouched. Only the map part of the resolver changes.

Feedback and logs are welcome, especially from anyone driving with Speed Limit Assist and the map as a source. The PR is [#2086](https://github.com/sunnypilot/sunnypilot/pull/2086).
