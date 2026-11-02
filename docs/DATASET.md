<!-- [reviewed-dataset] - START -->
# Canonical reviewed dataset v1

The executable contract is `nnslr_tools.annotations`; the portable structural
schema is [annotation.schema.json](../schemas/annotation.schema.json).
The CLI uses stdlib only. Dataset truth stays outside the runtime contract.

## Where the raw files live

Raw comma files (`fcamera.hevc`, `ecamera.hevc`, `rlog.zst`, ...) are stored
once, outside the clone, in the shared folder `COMMA_VIDEOS_DIR` (default:
`comma-videos` next to the clone, e.g. `/root/comma-videos` on the training
container). Its layout is the comma one, `<route>--<segment>/<file>`, and
`openpilot.distill` uses the same folder.

`<data-root>/raw/routes/<route>/<segment>/<file>` only holds absolute symlinks
to those files. `nnslr sync-routes` downloads into the shared folder and
creates the links; `nnslr link-videos` links segments that are already there
without network access and never writes into the shared folder. In this
project `sync-routes` is the only writer of the shared folder;
`openpilot.distill`'s `helpers/fetch_comma_segments.sh` writes the same folder.
The two commands are the only writers of the links.

`sync-routes` never overwrites a file in the shared folder. If it reports
`content differs` for a shared file (for example a copy cut off by an
interrupted distill fetch), rerun distill's fetch or delete that shared file by
hand, then rerun `sync-routes`. An old regular file in `raw/routes` is moved
into the shared folder only after it has been checked against the comma.

Readers (`route-manifest`, `extract-frames --route`, ...) accept such a link
only when it is the file itself, inside `raw/routes/`, and it resolves into
`COMMA_VIDEOS_DIR`; any other link out of the data root is still rejected.
Readers take the shared folder from `COMMA_VIDEOS_DIR` (or its default), never
from `--videos-dir`: after `sync-routes`/`link-videos --videos-dir <dir>`, also
`export COMMA_VIDEOS_DIR=<dir>`, otherwise the readers reject the links (both
commands print this warning).
Derived files (frames, alignment, annotations, datasets, splits, runs) are
always written under the data root, never into the shared folder. Paths in
manifests and frame provenance stay paths inside the data root
(`raw/routes/<route>/<segment>/fcamera.hevc`), and hashes are those of the
shared file.

## Records and provenance

Each JSONL record is one reviewed object, or an explicitly confirmed negative
frame (`sign_family=not_a_sign`, `bbox_xyxy=null`, `value_state=not_applicable`).
Coordinates refer to the original PNG, not the review preview. Fields are
required, unknown fields rejected, and absent evidence is represented by null
or a declared unknown enum, never zero speed or an invented timestamp.

`annotation_id` is a stable hash of route, segment, camera, output index and
object identity; the source decoded index takes precedence when known.
`output_index` and `decoded_frame_index` are different fields.
Legacy image-only records may have no decoded index; they retain the actual
image SHA-256 and remain usable only for tasks not requiring capture timing.
A decoded index requires a source-video hash. Exact timing additionally needs
a positive capture timestamp, source-log hash and explicit SOF/EOF provenance.

Families: maximum_speed, cancellation, zone, variable_display, other_sign,
not_a_sign, unreadable, road_marking. The last two preserve unreadable-family
and reviewed asphalt evidence; a road marking is not silently promoted to a
vertical regulatory sign. Values are read/unreadable/unknown/not_applicable;
a read value is an integer in the existing numeric domain 5..250. This numeric
domain is a plausibility constraint, not a claim about local road law.
`value_kph` is the historical name for the **written number**, not a converted
physical speed. Optional `sign_value` records those same digits and `sign_unit`
records `km/h`, `mph` or `unknown` (null/missing: not recorded). The canonical
writer mirrors `sign_value` into `value_kph` for old consumers; conflicting aliases
or a canonical row with `sign_value` but a null alias are rejected. Old rows need
no rewrite. A 25 mph sign has numeric class `maximum_speed:25`, never `:40`.
The reader export declares `number_semantics=written_sign_value` and
`unit_source=runtime_country`; this adds metadata without reordering classes.
See [the preparation report](2026-11-02-speed-vision-units-report.md) for runtime,
replay and remaining validation limits. No existing data has been migrated.
Applicability is own_road/other_road/unresolved. Panel presence, visibility,
occlusion, temporary status and variable-display status remain independent.

Review states are pending/accepted/corrected/rejected. Pending data is invalid
for this dataset. Accepted/corrected records are eligible reviewed labels;
rejected proposals remain audit records and never become positive labels or
negatives automatically. Only explicit whole-frame confirmation with zero
retained detections creates a negative frame. Deleted proposals are retained as
rejected records alongside it. The importer compares the reviewed proposal with
its original to distinguish acceptance from correction. Exact original JSONL
inputs are archived by SHA-256 under `annotations/imports/`.

Every imported object preserves reviewer, timezone-aware review timestamp,
notes, proposal source/score, original proposal/model and source-review hash.
These fields record the review assertion; the tool does not authenticate the
reviewer's identity. `provenance.label_basis` says who confirmed the frame:
`human_review` (a person) or `ai_review` (an AI reviewer through the REST API or
MCP: reviewer `ai/<name>`, `provenance.reviewer_kind: ai` and
`provenance.review_channel` `rest`, `mcp` or `web`). Both are confirmed reviews
and both go into gold; an AI confirmation needs no second, human approval.
Automatic `model_review` labels stay training-candidate only. A browser export
is not dataset truth until import passes.
Legacy browser confirmations lacking reviewer/time must be reconfirmed.

## Validation

`validate-dataset` checks schema/types/enums, finite values, numeric domain,
pixel bounds, image existence/hash/geometry, complete PNG chunks/checksums/scanlines,
duplicate identities, contradictory whole-frame negative/object labels, review state,
provenance and consistent monotonic capture evidence. It reports errors as JSON
and exits nonzero. No corrupted input is silently coerced into a valid label.
Missing capture timing remains unresolved and is counted separately from exact
timing; label validation does not close the T2 synchronization gate.

## Frozen splits

The split builder unions full routes with shared site IDs, encounter IDs and
image hashes before assignment. Empty/unknown grouping labels do not join
unrelated routes. Supply the same `site_group` in the review UI for repeated
visits to the same physical site, and `encounter_id` for shared encounters.
Missing physical-site labels are reported: route isolation alone cannot prove
that two trips never pass the same sign.

`build-splits` assigns three partitions (policy `train-validation-test-v2`):
the training learns from `train`, picks its best epoch on `validation` and is
measured on `test`. At least three connected groups are needed. `validation`
and `test` each get whole groups adding up to about 15% of the accepted and
corrected rows, at least one group each; every other group is `train`, so every
reviewed route is used. Reviewed negatives (`not_a_sign`) stay with their group,
in whatever partition. The test takes first the groups with the most frames
confirmed by a person (`provenance.label_basis` `human_review`), skipping a
group more than twice the remaining size, so it is as trustworthy as the
reviews allow; `--test-route ROUTE` (repeatable) pins routes and their groups to
`test` first. The seed defaults to zero and decides ties and the validation
order; sorting plus hashing makes repeated invocations reproducible. Rejected
audit records are excluded. Requested isolation is never weakened to force a
ratio when grouping constraints leave too few independent samples. The split
file records `policy`, `test_routes`, `counts`, `human_reviewed_counts` (rows
confirmed by a person per partition) and the `routes` of every partition.

Split files of the first policy, with five partitions (`route-held-out` and
`hard-negative` as well), stay valid: `validate-dataset --splits` still accepts
those names, but no command builds them any more and nothing read those two
partitions.

Split filenames are hashes of their canonical contents; changing a dataset
requires a new split. Existing named split files cannot be overwritten with a
different definition. Validation with `--splits` detects stale dataset hashes,
missing/extra members, unknown split names and group leakage.

## Current gate

Synthetic import/validation/split tests demonstrate tooling, not a trained Swiss
model or a human-reviewed Swiss corpus. Human annotation, independent visual
synchronization measurements, dataset coverage and a frozen model I/O contract
are still required before the baseline training milestone.
<!-- [reviewed-dataset] - END -->

<!-- [model-review] - START -->
## Automatic review and candidate data

`--dataset-kind training-candidate` explicitly permits model-reviewed rows in
`import-annotations`, `validate-dataset` and `build-splits`. Its default output
is `<data-root>/datasets/training_candidate_dataset/objects.jsonl`; gold remains
the default mode and accepts human review only. Gold validation rejects model
rows even when imported directly as canonical annotations. Candidate split
metadata cannot be presented as gold, and the same route/site/image-disjoint
train/validation/test partitions are still required.

Human canonical rows retain schema version 1. Model rows use version 2 and
require `provenance.label_basis=model_review`, a source-review SHA256,
`review_model`, immutable `review_model_version`, `review_model_sha256`, finite
`confidence` in [0,1], a matching `reviewer=codex/<review_model>`, an aware review
timestamp and verified image hash. Reviewed-frame imports carry model fields
at the top level; the importer transfers them to canonical provenance.
Pending/unresolved reviews cannot be imported. Model confidence is a model
assertion, not calibrated accuracy or human ground truth.

Run the automatic reviewer **on the data host**, with an already-running local
llama.cpp vision server and verified model metadata:

```sh
python -m nnslr_tools.model_review /path/to/preannotations.jsonl \
  --data-root /path/to/data \
  --output /path/to/data/review/run-name \
  --model-metadata /path/to/review-model.json \
  --server-metadata /path/to/review-server.json
```

`--limit 3` produces an explicitly sampled pilot. Without a limit all supplied
frames are reviewed. The endpoint is loopback-only; no image goes to a remote
API. Pillow comes from the existing optional `preannotate` dependency profile.
The worker does not download models, launch servers, access devices or start
training. Server metadata binds PID/argv/port to the actual process and the
model/projector files; model metadata includes model ID, pinned revision,
asset filenames/SHA256s, directory and bundle SHA256 (SHA256 of the sorted,
compact JSON asset-hash dictionary).

The current input is the existing bootstrap preannotator, **not a trained
NNSLR predictor**. Every audit row marks this comparison scope and records
`nnslr.available=false`. Thus the resulting discrepancy counts are not NNSLR
precision/recall measurements. Ordinary `other_sign` proposals are outside
the speed comparison vocabulary but remain in the original evidence.

Pass 1 sees the full frame without prediction labels. Pass 2 sees a full frame
plus an unlabeled proposal crop, or four overlapping tiles covering the frame.
This is a different visual method using the same reviewer, so its errors are
correlated. Both raw responses, transforms, prompts, times and model identity
are retained. Multiple objects, conflicting confident answers and unreadable
values remain unresolved. Only confirmed negatives and single objects whose
proposal crop is explicitly associated with the answer become candidates.
False negatives with no verified box stay audit-only. No coordinate is invented.

Each run keeps `run.json`, immutable input/model/prompt/source binding,
per-frame checkpoints, and `final/report/{audit.jsonl,report.html,run.json}`
with category manifests. `final/candidate_reviews.jsonl`, a separate canonical
candidate dataset, `validation.json` and `splits.json` record downstream
results. Rerunning the same command resumes completed frames; a changed
binding requires a new output directory. Missing inputs or failed model calls
stop the run with an explicit UNAVAILABLE checkpoint, which is retried on
resume. Failed/insufficient splits are reported without randomizing adjacent
frames. No model review closes missing capture-time provenance, the human
gold-set gate, Swiss coverage or the unimplemented training/evaluation stages.
<!-- [model-review] - END -->

## Revisione via web

Start the review server on the training LXC (openpilot-distill, 10.10.10.213):

```sh
cd ~/sunnypilot.speedlimits && .venv/bin/nnslr review --host 0.0.0.0
```

It prints one address per LAN interface, for example
`http://10.10.10.213:8765/`; open it in the browser on the Mac and sign in with
the administrator's name and password. The first visit creates the administrator
with the token as setup code (`src/nnslr_tools/review_auth.py`); the session is
an HttpOnly, SameSite=Strict cookie. Without a session (or the token in an
`X-Review-Token` header) every page shows the sign-in page and every image and
API call answers 401. The token is new at every start unless `NNSLR_REVIEW_TOKEN`
is set (at least 16 letters, digits, `-` or `_`).
Without `--host` the server listens on 127.0.0.1 only and needs no token
(`ssh -L 8765:127.0.0.1:8765 openpilot-distill` from the Mac). The start page
lists every route with a pre-annotation report: frames, frames with proposals
and confirmed frames; `--route R` prints the address of that route directly.

Every "Conferma" is checked with the import-annotations rules and
saved at once in `<data root>/annotations/review.sqlite`: `frame_reviews` holds
the current review per route, report run and frame, `frame_review_history` every
save (append-only). A refused frame shows the reason on the page; a frame the
server did not receive stays marked "non salvato" and is retried. The reports
themselves are never modified, and "Scarica revisioni JSONL" remains as a backup.

Import the confirmed frames into the canonical dataset on the LXC:

```sh
.venv/bin/nnslr import-annotations --from-db
.venv/bin/nnslr import-annotations --from-db --route <route>
.venv/bin/nnslr import-annotations --from-db --dry-run
```

Validation, the source archive under `annotations/imports/` and the output file
are the same as for a JSONL import. Only the confirmed frames of each route's
current report are taken, confirmed by a person or by an AI reviewer (REST API,
MCP); AI frames keep their provenance and are counted in
`imported_ai_frame_count`. Withdrawn confirmations and unreviewed proposals are
never taken. Confirming a frame never imports or trains by itself. Frames already in the dataset with the same boxes and
labels are skipped, so the command can be repeated after every review session.
The dataset follows the current review of each frame of the current reports.
The comparison is per frame: a frame already in the dataset whose review changed
after its import (a box added, deleted, drawn again or relabelled, or the review
reclassified as an AI review) replaces all its old rows (same frame identity),
counted in `replaced_frame_count` and listed under `changed_after_import`; a
frame imported and later withdrawn leaves the dataset, counted in
`removed_withdrawn_frame_count` and listed under `removed_withdrawn`. The rows
taken out are archived exactly as they were in the content-addressed
`annotations/replaced/<sha256>.jsonl` (`replaced_archive` in the report). The
validation runs on the whole new dataset and nothing is written when it fails;
an import with only removals still writes the dataset. Frames that the current
reports do not review again stay as they are, and so do the rows of a withdrawn
frame that came from an older pre-annotation run of the route.
`--dry-run` (with `--from-db`) reports what would be new, replaced and removed,
validation included, and writes nothing: no dataset and no archive. A save
carries at most 100 boxes, and the reviewer name at most 100 printable characters.
