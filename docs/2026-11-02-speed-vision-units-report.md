# International units: preparation and evidence, 2 November 2026

The runtime implementation is now **integrated directly
in `/root/sunnypilot` on `speed-limits`**, after the user's explicit authorization.
Integration was verified against base `8d37cf946`; speedlimits remains on main. This supersedes the initial
patch-only preparation. It is **not an installed or validated feature**.
No branch/worktree, commit, push, installation, restart, publication, deployment
or device operation was performed. Perception's pre-existing changes were left
alone. Distill was read only. No prohibited worker or Serverone-ai was used.

The requested `docs/2026-11-02-paperclip-speed-vision-units.md` and the two
untracked prototypes named in the request were absent from their stated paths.
The specification used here is the user's detailed request and the current
sources/documentation; no previous conversation was assumed.

Source-publication update: the user subsequently requested commits/pushes for both
project branches. This does not authorize deployment, production GPS policies,
model publication or device changes. The remote proposal document was recovered
and read from `main@1f4f710` before publication; it was preserved by a fast-forward
of the local main branch. Its requirements agree with the implementation scope
and the validation gaps recorded here. The original absence described above
refers to the initial local inspection.
The source whitespace check excludes the archived unified-diff artifact itself:
its blank context lines intentionally contain the patch-format space. The patch
is applied and exercised by the preparation tests; it has not been stripped or
rewritten to satisfy a source-file whitespace rule.

## Runtime integration update

Branch and Git status were checked before writing: runtime was clean on
`speed-limits@8d37cf946`. The prepared patch was checked and applied without
conflicts. Existing speedlimits work and perception changes were preserved;
distill was not modified. No branch/worktree was created.

Runtime files integrated:

- `openpilot/sunnypilot/speed_limits/vision.py`, `speedvisiond.py`, new `country.py`;
- `openpilot/cereal/custom.capnp`;
- `openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_resolver.py`;
- existing vision/schema tests and new
  `openpilot/sunnypilot/speed_limits/tests/international/` (`conftest.py`,
  `test_memory_units.py`, `test_country.py`, `test_chain.py`, `test_daemon.py`);
- `openpilot/sunnypilot/speed_limits/README.md` documents behavior and release gaps.

The integration includes style/type/comment corrections after re-reading the
patch: explicit optional country type; physical versus original units documented;
strict ring zips; lint-clean native test sources. No border file, model weights,
manifest setting or production GPS default was added. With the current absence
of policy/borders, a future manual daemon start would block new numeric readings.
No daemon or service was started or restarted here.

Tests ran **against the actual checkout**, not a temporary patched source copy:
baseline **86 passed**, integrated vision/international suite **123 passed**.
Python was 3.12.15, within the runtime's supported range. Existing NumPy and Cap'n
Proto binaries came from the Python 3.12 environment; only pure Python pytest
dependencies were reused from the speedlimits environment, with plugin autoload
disabled. No installation was performed. Ruff passes for all touched runtime
Python/test files; whitespace checks pass.

The native resolver/SLA suites were also attempted at collection. They stop with
two environment errors: missing `msgq` and missing
`openpilot/common/libparams_c.so`. Therefore **full native resolver/SLA/planner
validation is still missing**. The new chain tests use the actual checkout's
source class bodies with platform imports isolated, and simulate daemon I/O.
They do not run native IPC or model inference.
The integrated chain checks also cover missing vision with expired map GPS, and
preservation of the existing history behavior in car-only/map-only policies.

Current integration evidence: `runtime-baseline.txt`, `runtime-integrated-tests.txt`,
`runtime-lint.txt`, `runtime-native-sla-collection.txt` under
`/root/work/speed-vision-units-2026-11-02/`. No new inference, training, dataset
import, restart, publication, deployment or device operation occurred in this
integration phase. The accidental synthetic training test described below belongs
to the earlier preparation phase.

The real GPS replay and experimental parameter comparisons below remain earlier
preparation evidence. They are not promoted to validated policy or border data
by this source integration. A further independent review was requested, but the
reviewer could not run because its usage quota was exhausted; this integration
received an author review, not a new completed independent review.

## Prepared changes

- `patches/speed-limits-unit-normalization/runtime.patch`: `vision.py`,
  `speedvisiond.py`, new `country.py`, `cereal/custom.capnp`, resolver, and native
  `test_vision.py` / `test_speed_vision_sp.py`. Pinned base: `8d37cf946`.
- An experimental offline border asset and its provenance/readiness limitations
  in `patches/speed-limits-unit-normalization/README.md`.
- Dataset/review: `src/nnslr_tools/annotations.py`, `carry_over.py`, `training.py`,
  `exporting.py`, `review_db.py`, `review_service.py`, annotation JSON schema.
- Frontend sources: `web/src/review/{types.ts,edits.ts,catalog.ts,BoxEditor.vue}`.
  They preserve original digits while editing and allow an independent unit.
  The served bundles were **not replaced**; the live portal has not been updated.
- Tests: `tests/units/`, `tests/dataset/test_annotations.py`,
  `tests/review/test_carry_over.py`, `tests/review/test_review_api.py`.
- Read-only replay: `scripts/replay_speed_vision_country.py`.
- Documentation: preparation plan, this report, `docs/DATASET.md`, `docs/GUIDE.md`.

## Runtime decisions

The classifier reads unit-free digits. Every new numeric observation is converted
before confirmation: km/h divides by 3.6; mph multiplies by 0.44704. Memory,
candidate comparison and publication all use m/s. Publication is an identity;
the resolver applies no second conversion to vision's limit. `speedLimit` remains
m/s, or zero when unavailable.

The unit whitelist is independent of the implicit-law table. FR and the existing
European metric countries are km/h; GB/US are mph. Unsupported country, missing
policy, invalid/stale GPS or ambiguous borders gives unknown. No CH/km/h fallback.
Numeric observations are blocked during unknown intervals; the previous physical
limit holds and cancellation is still processed. No new state-retention timeout
was invented. A confirmed change of jurisdiction clears pending confirmation and
explicit memory and re-evaluates implicit rules; parking retains its physical
limit. This is the preparation's explicit state-policy choice, not device-validated
behavior.

Legacy implicit, zone, garage and parking constants keep their documented physical
km/h values, converted to m/s independently of sign units. Existing legal tables
were preserved, not re-certified. GB/US have no implicit defaults here. A number
equal to the town default remains an explicit signed number; the scene cannot
turn that explicit 50 into an inferred zone 30.

Cap'n Proto ordinals 0..11 stay intact; append `rawSignValue @12`, `rawSignUnit @13`
and `rawSignCountry @14`. `rawSignSpeed @7` remains physical km/h, zero when the
unit is unknown. The original country/unit are kept with the raw reading and do
not change retrospectively. Old/new message round-trips were tested.

The planner currently accepts `speed_limit_valid or speed_limit_last_valid`.
Without the patch, publishing zero after 22.352 m/s leaves SLA targeting 22.352.
The resolver patch clears both previous values when a vision policy has no
current allowed solution. In mixed mode a valid current map fallback wins; Vision
Only never takes map or car. Tests execute the actual resolver/SLA classes with
platform imports isolated, plus speedvisiond.main with simulated I/O. They are
code tests, not a running native messaging/planner/device stack.

Changing display units preserves the physical vision limit. The chain test uses
offsets disabled. Existing fixed-offset configuration still interprets its amount
in the user's selected display unit; this preparation does not redesign offsets.

## Annotation/model compatibility

`value_kph` was a misleading legacy name for classifier digits. New optional
`sign_value` and `sign_unit` separate these facts. Canonical writers mirror the
written digits into `value_kph` for existing consumers; conflicting aliases and
canonical rows with a missing mirror are rejected. Input reviews can supply only
`sign_value`; saving normalizes the alias. Old rows are unchanged and need no
rewrite. Carry-over preserves both number and unit and removes stale metadata.

25 mph is **class 25**, not class 40. The export adds written-number/unit-source
metadata without changing class ordering. However the current installed contract
has `maximum_speed:20` and `:30`, and **no `maximum_speed:25`**. Tests of 25 mph use
supplied/simulated readings; there is no evidence of real recognition of 25.
No actual reviews or canonical dataset were migrated/imported.
The existing Mapillary adapter also selects round European `g1` numeric signs;
it excludes the American/Canadian rectangular designs. That filter was inspected
and left unchanged. Unit normalization does not establish visual recognition of
those designs.

## Local data inventory

Counts are file entries, including explicit links between repositories; they
must not be added as independent recordings. No route IDs or coordinates are
included in this tracked report.

| Location | Available entries |
| --- | --- |
| `/root/comma-videos` → `/mnt/comma-raw/comma-videos` | 1,939 HEVC, 1,185 rlog files |
| speedlimits `data/raw/routes` | 43 routes, 1,939 HEVC, 1,163 rlog entries |
| distill `data` | 299 segments, 598 HEVC, 299 rlogs; existing localizer outputs |
| speedlimits cached frame GPS | 42 routes, 63,338 rows |
| speedlimits cached drive-log extracts | 1,227 cache files/versions; all select external GPS |
| annotations | 23,033 rows: CH 22,133; DE 833; FR 34; unknown 33 |

Cached frame countries: CH 43,113; DE 1,374; FR 1,490; unknown 17,361.
Position provenance: localizer 34,377; GPS 24,527; absent 4,434. A country label
can be absent even when a position exists. There are six candidate multi-country
routes: two CH→DE, two DE→CH, one CH→FR, one FR→CH. These are offline country
proposals, not independently reviewed legal-border ground truth. No GB/US or
km/h↔mph crossing was found in the inspected GPS/annotation sources; the incoming
folder contains no additional recordings. Raw logs outside cached coverage have
not all been reclassified, so this is an observed-coverage statement, not proof
that every unprocessed video stays in those three countries.

## Real GPS replay, experimental policies

Read 18 existing local rlogs around the candidate crossings: **8,363 external GPS
messages**. The replay uses log monotonic time, event validity, hasFix, horizontal
accuracy, offline polygon holes, border margin and confirmation. It performs no
model inference and writes no annotations. Extracted accuracy quantiles: median
1.759 m, p95 2.925 m, p99 5.285 m; sample gap median 0.1024 s, p99 0.2160 s.
These describe selected windows, not worst-case receiver accuracy/latency.

All experiments explicitly used maximum GPS age 2 s, maximum reported accuracy
20 m and maximum confirmation-sample gap 1 s. They are **experimental choices**:
20 m exceeds these windows' p99, 1 s exceeds their ordinary sampling interval,
and 2 s gives a freshness boundary to exercise. Neither arrival latency nor long
outages were validated by these quantiles. Unit tests cover stale/missing/future,
duplicate, malformed and oscillating fixes separately.

| Borders / margin / confirmation | Unknown samples | Confirmed country changes |
| --- | ---: | ---: |
| Natural Earth world 1:50m / 100 m / 1 s | 1,948 (23.29%) | 0 |
| Local detailed GISCO / 25 m / 0.5 s | 2,165 (25.89%) | 6 |
| Local detailed GISCO / 100 m / 1 s | 3,429 (41.00%) | 6 |
| Local detailed GISCO / 250 m / 2 s | 7,196 (86.05%) | 1 |

The coarse asset can report a stable wrong country near the Rhine; fewer unknown
samples do not mean better classification. The detailed source reproduces the
candidate crossings, but the test has no independent border truth. The margin
and confirmation were varied together, so their effects are confounded. These
results motivate further source validation and separate parameter sweeps, not
production defaults. Missing policy/borders leaves numeric vision unavailable.

The public-domain world asset is bundled only for reproducibility of the failed
comparison. The finer GISCO asset remains private due to its non-commercial
terms; it has not been promoted into a distributable runtime resource.

Re-run, from speedlimits, with an explicitly supplied policy JSON:

```sh
.venv/bin/python scripts/replay_speed_vision_country.py --help
.venv/bin/python scripts/replay_speed_vision_country.py \
  --data-root data \
  --border-source data/derived/gps/borders.json.gz \
  --policy /root/work/speed-vision-units-2026-11-02/experimental-policy.json
```

Policies, replay JSON (including source/patch hashes), test outputs and the ordinary
candidate source directory are in `/root/work/speed-vision-units-2026-11-02/`.

## Verification and limits

- **43 new unit/chain/daemon/schema tests pass**, including the wrapper that runs
  the **86 migrated native vision tests**. The final focused regression selection
  reports **157 passed, 7 deselected**. Results are in `cpu-tests.txt` and `native-tests.txt`
  under the work folder above. Native msgq/Params are unavailable here: platform
  dependencies are isolated, while resolver/SLA algorithms remain the real source.
  Inland FR/GB/US coordinate fixtures additionally exercise packaged polygons →
  confirmed country → unit → physical limit; these are synthetic coordinate tests,
  not GB/US drive replay.
  Reproduce the new tests with `.venv/bin/python -m pytest tests/units -o addopts='' -q`.
- Focused annotation, training-plan, classifier-label, export, carry-over, review
  DB/import and direct review-service regressions pass. Training jobs and native
  HTTP tests are excluded from the final selection.
- A broader attempted run had **3 failures and 19 fixture errors** because the
  sandbox forbids creation of HTTP sockets (`PermissionError: Operation not
  permitted`). Failure names: `test_api_on_the_lan_server_uses_the_same_token`,
  `test_review_site_api_and_core_work_without_the_mcp_sdk`,
  `test_concurrent_http_saves`. HTTP-facing fixtures in carry-over/API were likewise
  blocked. The service save/normalization/progress path was then tested directly
  against a temporary database; no live-site save was made.
- TypeScript check passes; the review page compiles to a separate work-folder
  bundle. Served JS is intentionally untouched. No browser/device proof is claimed.
- Patch applicability and `git diff --check` pass. The whole default suite was
  not completed: it includes training/HTTP operations outside this final selection.
- An initial overly broad test selection mistakenly ran the existing
  `test_tiny_cpu_classifier_chain`, including synthetic CPU mini-training
  (`best_epoch=1`) and a temporary checkpoint in `/tmp/pytest-of-root/pytest-4/`.
  That selection was interrupted and excluded afterwards. This violated the
  requested training restriction; it used fixture data only, no actual dataset,
  video or remote worker, and published nothing.

Remaining release prerequisites: an acceptable distributable border source;
independent crossing truth; separately justified GPS/state policies; real mph
recordings including GB/US and unit transitions; trained/evaluated numeric classes
such as 25; full native resolver/planner/SLA and device tests; frontend bundle
promotion only when authorized. No real model inference or device testing was
performed in this session.
