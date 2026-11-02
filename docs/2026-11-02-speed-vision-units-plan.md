# International sign units: preparation plan

Spec: the user's requirements in this session. The referenced
`2026-11-02-paperclip-speed-vision-units.md` was absent at inspection time.

Coordination update: the user explicitly authorized direct runtime integration
in `/root/sunnypilot` on `speed-limits`, based on `8d37cf946`. The main preference
applies to `sunnypilot.speedlimits`. The initial patch-only restriction is superseded.
Preserve shared changes and create no branch/worktree. Runtime code and its tests
are now integrated in that checkout. No commits, pushes, imports, training,
installs, restarts, deployment or GPU jobs. Border/policy release prerequisites
remain open.

Subsequent source-publication authorization: the user requested commits and pushes
for the two project branches. This supersedes the earlier no-commit/no-push
restriction only; installation, training, imports, restart and deployment remain
outside this work. The remote proposal document was recovered and read from
main before publishing, while preserving its commit and all session changes.

## Decisions

- Class numbers mean the digits on the sign. Normalize new observations before
  confirmation; memory and comparisons use m/s, publication is an identity.
- Country from offline polygons and live GPS only. Unit lookup is a whitelist,
  independent of implicit legal rules. Unknown blocks numbers, not cancellation.
- A confirmed country change invalidates old jurisdiction's memory and pending
  observations; missing GPS only blocks new numbers and holds the physical limit.
- GPS age, accuracy, border margin and confirmation time have no validated values.
  Require an explicit experimental policy; without one country is unknown. Replay
  experiments must not turn their parameter choices into production defaults.
- Preserve legacy physical constants (zone, garage, parking and km/h legal table).
  Do not infer GB/US implicit limits. Do not lower an explicit town-speed sign just
  because it equals the town default.
- Keep schema ordinals 0..11; append original number, unit, country. rawSignSpeed
  stays physical km/h, zero when its unit is unknown.
- Dataset `value_kph` is a legacy name for the written number. Keep old rows and
  class names readable; add optional `sign_value` and `sign_unit`, mirror the
  written digits in legacy `value_kph` for canonical rows and require aliases to
  agree, never convert class 25 into 40. No data migration is executed.

## Tasks

- [x] Runtime patch: `openpilot/sunnypilot/speed_limits/{country.py,vision.py,
  speedvisiond.py}`, `openpilot/cereal/custom.capnp`, and
  `speed_limit_resolver.py`. New tests exercise normalization, unknown unit,
  boundary changes, implicit rules, parking, and log compatibility.
- [x] Dataset: `annotations.py`, `training.py`, `exporting.py`, annotation schema,
  and review field preservation; verify old/new rows and model metadata on CPU.
- [x] Integration: run the real patched resolver and SLA methods with isolated
  platform dependencies. Match the planner's current-or-last validity expression;
  test both vision policies, no map fallback in Vision Only, map fallback in mixed,
  invalidation after a previous target, and display changes with offsets disabled.
- [x] Inventory/replay: inspect local video/log files and GPS caches, replay
  observed country sequences separately from synthetic unit transitions. Record
  provenance, missing coverage and experiment choices without private coordinates.
- [x] Verification/report: local focused tests, patch applicability, original
  checkout preservation, no inference/device claims.

## Review focus

Nonfinite/future/duplicate GPS, ambiguous borders and polygon holes; cancellation
with unknown unit; two countries with the same physical candidate; explicit 50
versus town default; old resolver targets surviving a zero vision output.

## Progress and rulings

Initial inspection: speedlimits main clean at 450ba3e; runtime clean at 8d37cf946;
distill master at 2231dc0 (ahead 4); perception modifications outside scope.
Cached frame GPS: CH, DE, FR only; no observed km/h↔mph transition.

Preparation complete, with release prerequisites explicitly left open in the report.
The initial patch-only constraint was superseded by the user's coordination
correction: runtime work belongs directly on `sunnypilot:speed-limits`; main
applies to speedlimits. Integration now has 123 passing tests in that checkout.
Remaining rulings: preserve canonical numeric aliases for old consumers; no
production GPS defaults; no new retention timeout; clear explicit jurisdiction
memory on a confirmed country change.
Independent code review found alias/carry-over, unit-table coupling and corrupt-gzip
issues, corrected with regression tests. A final author pass found malformed GPS
timestamps could retain a fresh country; the new regression now blocks them.
No release/device readiness is claimed. See the report for test restrictions and
the mistakenly executed synthetic CPU training test.
