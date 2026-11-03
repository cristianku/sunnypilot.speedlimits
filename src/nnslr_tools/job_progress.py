"""How far an action of the panel is, from its log: the steps, the current one, a percent when it is known.

The helpers write a line per phase (``== a. ...`` in helpers/3_train.sh, ``== 4/10 Train detector`` in
scripts/run_v100_baseline.sh, ``== 3/7 Mine the cutter's false boxes ==`` in scripts/run_classifier.sh,
``== 3/3 Train SignNet ==`` in scripts/run_signnet.sh (``== 3/4 Train the detector at 960x544 ==`` in the old
scripts/run_detector_sizes.sh), ``== b+c. ...`` in helpers/1_prepare.sh, ``== d. Map roads near the drives`` in
helpers/road_types.sh) and the training one line per epoch (``detector epoch 3/20 ...``, ``classifier epoch 3/20 ...``,
``pretrain epoch 3/20 ...`` for the first stage of a two-stage classifier, nnslr_tools.training; ``signnet epoch 3/40
...``, nnslr_tools.signnet). Nothing here guesses a time: a percent comes only from
steps and epochs actually written, or from the preparation's own count of finished routes.
"""
from __future__ import annotations

import re

# Steps of the import and of the training, with their weight in the whole (they add up to 100): the
# detector and the reader are most of a training.
TRAIN_STEPS = (
    ('import', 'Import the reviews', 2),
    ('split', 'Check the dataset and fix the split', 2),
    ('gpu_env', 'Set up the GPU environment', 6),
    ('checks', 'Checks and GPU smoke test', 3),
    ('detector', 'Train the detector', 40),
    ('reader', 'Train the reader', 30),
    ('evaluate', 'Evaluate on unseen routes', 7),
    ('export', 'Hard examples and ONNX export', 6),
    ('bundle', 'Package and verify the bundle', 4),
)
DATASET_STEPS = TRAIN_STEPS[:2]
# scripts/run_classifier.sh: the false boxes of the cutter in use mined, then the classifier with "not a sign" trained
# on them, measured alone and in the chain (cutter -> classifier). Its training is most of it.
CLASSIFIER_STEPS = (
    ('checks', 'Checks and GPU smoke test', 4),
    ('mine', 'Mine the cutter’s false boxes', 14),
    ('classifier', 'Train the classifier', 60),
    ('evaluate', 'Evaluate the classifier alone', 7),
    ('chain', 'Evaluate the chain end to end', 15),
)
# The same in two stages (NNSLR_CLASSIFIER_MAPILLARY=pretrain, PRETRAINING): the Mapillary crops alone, then ours from
# them, a step each with half the training's weight.
CLASSIFIER_TWO_STAGE_STEPS = (
    *CLASSIFIER_STEPS[:2],
    ('pretrain', 'Learn the Mapillary signs', 30),
    ('classifier', 'Train on our signs', 30),
    *CLASSIFIER_STEPS[3:],
)
# scripts/run_signnet.sh: the cutter, a SignNet with the car's setup, then its best epoch measured on validation and
# test ("signnet measure the best epoch ..."), until "SignNet complete: <run>".
SIGNNET_STEPS = (
    ('checks', 'Checks and GPU smoke test', 5),
    ('signnet', 'Train SignNet', 85),
    ('evaluate', 'Measure on validation and test', 10),
)
# The old cutter, scripts/run_detector_sizes.sh with one size: a detector trained as the cutter, then measured on test
# and validation (the jobs of the history before SignNet; chosen by the log, see is_signnet).
CUTTER_STEPS = (
    ('checks', 'Checks and GPU smoke test', 5),
    ('detector', 'Train the cutter', 80),
    ('evaluate', 'Evaluate on test and validation', 15),
)
# scripts/run_experiment_b.sh: the decider's three variants, one after the other (their epoch lines name them).
DECIDER_STEPS = (
    ('checks', 'Checks and GPU smoke test', 4),
    ('experiment_b_crop', 'The crop alone', 32),
    ('experiment_b_crop_geom', 'Crop, position and speed', 32),
    ('experiment_b_full', 'Crop, position and the scene', 32),
)
# scripts/run_road_type.sh: the junctions of the drives, the motorway signs and the car's own motion read from its logs,
# the road-type network with memory (and its junction head), the sign input and the car's motion, the same without the
# car's motion, the same without memory, then the comparison and the ONNX step (their epoch lines name them:
# road_type_gru, road_type_gru_nomotion, road_type_frame; road_type_gru_nosign in the runs before 29 Oct 2026).
ROAD_TYPE_STEPS = (
    ('checks', 'Checks and GPU smoke test', 4),
    ('frames', 'Find the junctions, resize the frames', 4),
    ('signs', "Read the motorway signs and the car's motion", 8),
    ('road_type_gru', 'Train with memory, the sign and the motion', 30),
    ('road_type_gru_nomotion', 'Train with memory, no motion', 27),
    ('road_type_frame', 'Train without memory', 23),
    ('export', 'Compare and export the ONNX step', 4),
)
PREPARE_STEPS = (('link', 'Link the videos', 5), ('frames', 'Extract the frames and propose the signs', 95))
# scripts/publish_networks.sh: "== 1/4 Export the cutter" ... (prepare), "== 1/3 Check the package" ... (upload).
NETWORKS_PREPARE_STEPS = (('cutter', 'Export the cutter', 40), ('classifier', 'Export the classifier', 20),
                          ('decider', 'Export applicability', 30), ('package', 'Model card and checks', 10))
NETWORKS_PUBLISH_STEPS = (('check', 'Check the package', 10), ('catalog', 'Read the catalog', 10),
                          ('upload', 'Upload to Hugging Face', 80))
# scripts/send_to_car.sh: "== 1/5 Export ..." to "== 5/5 Build ..."; the build of the testing branch takes the minutes.
CAR_STEPS = (('export', 'Prepare the package: the networks in use to ONNX, checked', 10), ('publish', 'Publish it in speed-limits-models', 5),
             ('record', 'Record them as on the car', 2), ('move', 'Move the sunnypilot speed-limits branch', 3),
             ('build', 'Build psa-torque-sunny-testing', 80))
# scripts/try_on_comma.sh: "== 1/5 Check the comma" to "== 5/5 Wait ..."; the comma compiles them in a few minutes.
TRY_STEPS = (('check', 'Check the comma', 3), ('export', 'Export the SignNet and the classifier', 15),
             ('verify', "Check them with the comma's code", 15), ('copy', 'Copy them onto the comma', 7),
             ('prepare', 'Wait for the comma to prepare them', 60))
# scripts/try_on_comma.sh --restore: "== 1/2 Check the comma", "== 2/2 Put back the build's networks".
RESTORE_STEPS = (('check', 'Check the comma', 50), ('restore', "Put back the build's networks", 50))
FETCH_STEPS = (('fetch', 'Download the drives from the comma', 100),)
# helpers/road_types.sh: openpilot.distill's localizer on the segments new to it (about 20 s of a core each, 6 side by
# side: some 20 minutes the first time) and the GPS of the drives prepared without it or localized since, the maps the
# drives need (seconds, a minute or two for each map downloaded), the car's speed from the logs (a minute for each new
# drive), the map roads near the drives (minutes when the map is read again), the road type of every frame, the
# intersections every drive passed.
ROAD_TYPES_STEPS = (('gps', 'Read the GPS of the drives', 10), ('maps', 'Download the maps the drives need', 5),
                    ('speed', 'Read the speed of the drives', 10),
                    ('map_roads', 'Read the map roads near the drives', 50),
                    ('road_types', 'Find the road type of every picture', 18),
                    ('junctions', 'Find the junctions the car drove through', 7))
UPDATE_STEPS = (('update', 'Update the code and restart the site', 100),)
# [ai-reviewer] - START
# One step each: nnslr ai-review and nnslr measure-ai-reviewer report their frames (nnslr_tools.progress).
AI_REVIEW_STEPS = (('review', 'Review the frames with the AI', 100),)
AI_MEASURE_STEPS = (('measure', 'Measure the AI reviewer', 100),)
# [ai-reviewer] - END
STEPS = {'fetch': FETCH_STEPS, 'prepare': PREPARE_STEPS, 'dataset': DATASET_STEPS, 'train': TRAIN_STEPS,
         'train_continue': TRAIN_STEPS, 'train_resume': TRAIN_STEPS, 'train_classifier': CLASSIFIER_STEPS,
         'train_cutter': CUTTER_STEPS, 'train_decider': DECIDER_STEPS, 'train_road_type': ROAD_TYPE_STEPS,
         'update': UPDATE_STEPS, 'ai_review': AI_REVIEW_STEPS,
         'measure_ai_reviewer': AI_MEASURE_STEPS, 'networks_prepare': NETWORKS_PREPARE_STEPS,
         'networks_publish': NETWORKS_PUBLISH_STEPS, 'road_types': ROAD_TYPES_STEPS, 'send_to_car': CAR_STEPS,
         'try_on_comma': TRY_STEPS, 'comma_restore': RESTORE_STEPS}
# The actions whose percent is the weighted sum of their steps (the AI's: its frames done, from its reports).
WEIGHTED = ('train', 'train_continue', 'train_resume', 'train_classifier', 'train_cutter', 'train_decider',
            'train_road_type', 'dataset', 'ai_review',
            'measure_ai_reviewer', 'networks_prepare', 'networks_publish', 'road_types', 'send_to_car',
            'try_on_comma', 'comma_restore')

# The sub-steps of scripts/run_v100_baseline.sh ("== 4/10 ...") and the step they belong to.
PIPELINE = {1: 'checks', 2: 'checks', 3: 'checks', 4: 'detector', 5: 'reader', 6: 'evaluate', 7: 'export',
            8: 'export', 9: 'bundle', 10: 'bundle'}
# The sub-steps of scripts/run_classifier.sh ("== 3/7 ..."): the comparison with the old reader closes the chain.
CLASSIFIER_PIPELINE = {1: 'checks', 2: 'checks', 3: 'mine', 4: 'classifier', 5: 'evaluate', 6: 'chain', 7: 'chain'}
# scripts/run_detector_sizes.sh numbers its phases out of 2 + 2 x the sizes: they are known by their titles. The checks
# are always its first two.
CUTTER_PIPELINE = {1: 'checks', 2: 'checks'}
CUTTER_TITLES = ((re.compile(r'Validate the training'), 'checks'), (re.compile(r'Explicit CUDA smoke test'), 'checks'),
                 (re.compile(r'Train the detector at '), 'detector'), (re.compile(r'Evaluate the detector at '), 'evaluate'))
# Which numbered sub-steps an action's log has: (how many, their steps; None: known by their titles, TITLES). Every
# other action: run_v100_baseline.sh.
# scripts/run_experiment_b.sh ("== 3/5 Train variant crop ==").
DECIDER_PIPELINE = {1: 'checks', 2: 'checks', 3: 'experiment_b_crop', 4: 'experiment_b_crop_geom', 5: 'experiment_b_full'}
SUBSTEPS = {'train_classifier': (7, CLASSIFIER_PIPELINE), 'train_cutter': (None, CUTTER_PIPELINE),
            'train_decider': (5, DECIDER_PIPELINE),
            'train_road_type': (8, {1: 'checks', 2: 'checks', 3: 'frames', 4: 'signs', 5: 'road_type_gru',
                                    6: 'road_type_gru_nomotion', 7: 'road_type_frame', 8: 'export'}),
            'networks_prepare': (4, {1: 'cutter', 2: 'classifier', 3: 'decider', 4: 'package'}),
            'networks_publish': (3, {1: 'check', 2: 'catalog', 3: 'upload'}),
            'send_to_car': (5, {1: 'export', 2: 'publish', 3: 'record', 4: 'move', 5: 'build'}),
            'try_on_comma': (5, {1: 'check', 2: 'export', 3: 'verify', 4: 'copy', 5: 'prepare'}),
            'comma_restore': (2, {1: 'check', 2: 'restore'})}
TITLES = {'train_cutter': CUTTER_TITLES}
# scripts/run_signnet.sh ("== 3/3 Train SignNet ==") and the lines that close its training.
SIGNNET_PIPELINE = {1: 'checks', 2: 'checks', 3: 'signnet'}
SIGNNET_LOG = re.compile(r'^(?:== scripts/run_signnet\.sh |== \d+/\d+ Train SignNet\b|signnet epoch \d)', re.M)
SIGNNET_MEASURE = re.compile(r'^signnet measure the best epoch\b')
SIGNNET_COMPLETE = re.compile(r'^SignNet complete: ')
MARKERS = (
    (re.compile(r'^== a\+b\. '), 'gpu_env', ('import', 'split')),
    (re.compile(r'^== a\. (?:importing|importo)'), 'import', ()),
    (re.compile(r'^== b\. '), 'split', ()),
    (re.compile(r'^== c\. '), 'gpu_env', ()),
    (re.compile(r'^== a\. (?:linking|collego)'), 'link', ()),
    (re.compile(r'^== b\+c\. '), 'frames', ()),
    # helpers/road_types.sh, by the words after its letters (a marker counts only for an action with that step: the
    # "== b. " of the import above is not one of its steps): the letters of a log from before the maps (3 Nov 2026),
    # or from before the speed, are one or two earlier.
    (re.compile(r'^== a\. GPS of the drives'), 'gps', ()),
    (re.compile(r'^== b\. Maps the drives need'), 'maps', ()),
    (re.compile(r'^== [bc]\. Speed of the drives'), 'speed', ()),
    (re.compile(r'^== [bcd]\. Map roads'), 'map_roads', ()),
    (re.compile(r'^== [cde]\. Road type'), 'road_types', ()),
    (re.compile(r'^== [def]\. Junctions of the drives'), 'junctions', ()),
)
SUBSTEP = re.compile(r'^== (\d+)/(\d+) ')
EPOCH = re.compile(r'^(signnet|detector|reader|classifier|pretrain|experiment_b_crop_geom|experiment_b_crop|experiment_b_full|'
                   r'road_type_gru_nosign|road_type_gru_nomotion|road_type_gru|road_type_frame) epoch (\d+)/(\d+)\b')
# "classifier stops early after epoch 12: ...": the training ends before its planned epochs.
EARLY = re.compile(r'^(detector|reader|classifier|pretrain|road_type_gru_nosign|road_type_gru_nomotion|road_type_gru|'
                   r'road_type_frame) stops early after epoch (\d+)\b')
# The values of an epoch line: "train_loss 0.5000 val_loss 0.6000 ... 40s" (the detector also writes
# val_recall, its validation recall at score 0.1, which chooses its best epoch; the classifier with "not a sign"
# val_sign_accuracy, the sign crops read right, and val_rejection, the "not a sign" crops it rejects: their mean,
# the balanced accuracy, chooses its best epoch; the road-type network val_macro_f1, its F1 over the road types,
# val_road_score, its own measure that chooses its best epoch (road_type_net.road_score), and val_junction_f1, its
# junction head; SignNet val_recall and val_false_per_frame in the 5 best boxes of a frame, as the car keeps them;
# experiment B val_ownership_macro_f1, whose sign it is, and val_other_road_recall, a "-" when it has no case).
METRIC = re.compile(r'\b(train_loss|val_loss|train_accuracy|val_accuracy|val_recall|val_sign_accuracy|val_rejection|'
                    r'val_balanced_accuracy|val_macro_f1|val_false_per_frame|val_road_score|val_junction_f1|'
                    r'val_ownership_macro_f1|val_other_road_recall) (-?[0-9.]+(?:e-?[0-9]+)?)\b')
SECONDS = re.compile(r'\b([0-9]+)s$')
FETCHED = re.compile(r'^== (\S+)--(\d+) -> ')
# helpers/fetch_comma_segments.sh, a route and its segments (in Italian in the older logs).
ON_COMMA = re.compile(r'^(\S+) (?:on the comma|sul comma): ')
# Its first line, every drive it goes through; then a line per drive that starts with its name, and the segments it
# downloads ("Download: 35 36").
COMMA_ROUTES = re.compile(r'^Routes found on the comma:(.*)$')
ROUTE_LINE = re.compile(r'^(\S+?)(?:--\d+)?(?::| on the comma)')
TO_DOWNLOAD = re.compile(r'Download:((?: \d+)*)$')
# A proposer measured (helpers/measure_proposer.sh): its worker writes a line per frame, "Preannotating 280/604: ...".
PREANNOTATING = re.compile(r'^Preannotating (\d+)/(\d+)\b')
# The last line the panel's wrapper writes (nnslr_tools.panel.WRAPPER; in Italian in the older logs).
FINISHED = re.compile(r'^== (?:finished with code|finito con codice) ')
# A line of JSON printed in the log (the report of the CUDA smoke test, ...): a brace, a bracket, a quoted key
# or a bare value. Never the last line shown: a lone "}" says nothing. A frame counter ("[12/60] 3/27: 2 signs ...",
# nnslr ai-review) is no JSON.
JSON_LINE = re.compile(r'^(?!\[\d+/\d+\] )(?:[{}\[\]"]|(?:-?\d[\d.eE+-]*|true|false|null),?$)')


# Lines a library prints after its error, pointing to its documentation: they say nothing of what went wrong.
HINT_LINE = re.compile(r'^(Search for `|For debugging consider passing|Compile with `TORCH_USE_CUDA_DSA`)')
# The failures seen in the logs, each with what the page says instead of the exit code and the paths (the log, with
# them, stays in Details). The first that matches the end of the log wins.
FAILURES = (
    (re.compile(r'CUDA out of memory|CUDA error: out of memory|OutOfMemoryError|OOM on device'),
     'The GPU ran out of memory: another job or program was using it. Free the GPU, then try again.'),
    (re.compile(r'No space left on device'), 'The disk is full: free some space, then try again.'),
    (re.compile(r'creazione di \.venv-gpu fallita|\.venv-gpu.*failed'),
     'Setting up the GPU programs failed. Details has the log.'),
    (re.compile(r'servono almeno \d+ route|at least \d+ (?:independent )?(?:reviewed )?(?:routes|drives)'),
     'More reviewed drives are needed before the dataset can be split.'),
    (re.compile(r'modifiche locali|local changes'),
     'The code on this server has changes of its own: the update leaves them alone.'),
    (re.compile(r'Connection refused|Connection timed out|No route to host|Could not resolve hostname|'
                r'Permission denied \(publickey'),
     'The comma did not answer, or refused this server: is it on, and on the same network?'),
)
# How far back from the end of a failed log a known failure is looked for.
FAILURE_LINES = 60


def last_line(text: str) -> str:
    """The last line that says something: not the start and end markers of the wrapper, not JSON, not a library's
    hint to its documentation, not empty."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if (line and not FINISHED.match(line) and not JSON_LINE.match(line) and not HINT_LINE.match(line)
                and not line.startswith(('== helpers/', '== scripts/'))):
            return line[:300]
    return ''


def failure_text(text: str) -> str:
    """What went wrong in one plain sentence, when the end of the log shows a known failure (FAILURES); ''
    otherwise."""
    tail = '\n'.join(text.splitlines()[-FAILURE_LINES:])
    return next((sentence for pattern, sentence in FAILURES if pattern.search(tail)), '')


def duration_text(seconds: float | None) -> str:
    """"about 2 h 50 min", "about 12 min", "under a minute"; '' when unknown."""
    if seconds is None:
        return ''
    minutes = round(seconds / 60)
    if minutes < 1:
        return 'under a minute'
    return 'about ' + (f'{minutes // 60} h {minutes % 60} min' if minutes >= 60 else f'{minutes} min')


def again_detail(preparation: dict) -> str:
    """Propose again in one line: "Qwen3-VL · 0 of 4 drives ready" (panel.Panel._again_progress). The drive at
    work and its frames are in the list of drives under it, the time left once, as the job's 'seconds_left'."""
    parts = [f"{preparation.get('routes_done', 0)} of {preparation.get('routes_total', 0)} drives ready"]
    head = preparation.get('proposer')
    return ((head + ' · ') if head else '') + ' · '.join(parts)


def progress(action: str, state: str, text: str, preparation: dict | None = None) -> dict:
    """{'steps': [{'key', 'label', 'state': done|current|todo|failed}], 'percent': 0-100 or None,
    'detail': str, 'last_line': str, 'step_progress': how far the current step is from the log alone (see
    _inside_step) or None, 'curves': {'detector'|'reader'|'classifier'|'signnet': {'epochs': total, ('stopped_early':
    epoch,) 'points': [{'epoch', 'train_loss', 'val_loss', ('train_accuracy', 'val_accuracy',) ('val_recall',)
    ('val_false_per_frame',) ('val_sign_accuracy', 'val_rejection', 'val_balanced_accuracy',) ('val_macro_f1',
    'val_road_score', 'val_junction_f1',) ('val_ownership_macro_f1', 'val_other_road_recall', 'val_selection_score',)
    'seconds'}]}}} (also the road type's networks and experiment B's variants, by the names of their epoch lines) for an
    action in state running, ok, failed or interrupted; with 'seconds_left' (the server's time left, None while
    unknown) for Propose again running, 'signnet' (is_signnet) for the cutter's training and 'failure' (failure_text,
    '' when unknown) for a failed job."""
    two_stages = action == 'train_classifier' and PRETRAINING.search(text) is not None
    signnet = is_signnet(action, text)
    steps = steps_of(action, two_stages, signnet)
    keys = [key for key, _, _ in steps]
    current = keys[0] if keys else ''
    done: set[str] = set()
    epoch: tuple[str, int, int] | None = None
    # The numbered sub-steps of the action's script, and the last one written: its number and its title.
    count, pipeline = (3, SIGNNET_PIPELINE) if signnet else SUBSTEPS.get(action, (10, PIPELINE))
    complete = False
    substep: tuple[int, str] | None = None
    # The networks that stopped early, and after which epoch.
    stopped: dict[str, int] = {}
    fetched = 0
    route = ''
    # The fetch: the drives on the comma, the one it is at, its segments to download and those reached.
    comma_routes: list[str] = []
    at_route = -1
    segments: tuple[int, int] = (0, 0)
    # A proposer measured: the frames proposed so far and all of them.
    proposed: tuple[int, int] | None = None
    # The curves of the training: per network, one point per epoch.
    curves: dict[str, dict] = {}
    for raw in text.splitlines():
        line = raw.strip()
        for pattern, step, skipped in MARKERS:
            if pattern.match(line) and step in keys:
                done.update(skipped)
                current = step
        if signnet and SIGNNET_MEASURE.match(line):
            current = 'evaluate'
        if signnet and SIGNNET_COMPLETE.match(line):
            complete = True
        match = SUBSTEP.match(line)
        if match:
            number, title = int(match.group(1)), line[match.end():].strip(' =')
            if count is None:
                step = next((step for pattern, step in TITLES.get(action, ()) if pattern.match(title)), None)
            else:
                step = pipeline.get(number) if int(match.group(2)) == count else None
            if step in keys:
                current = step
                substep = (number, title)
        match = EPOCH.match(line)
        if match:
            epoch = (match.group(1), int(match.group(2)), int(match.group(3)))
            point: dict = {'epoch': epoch[1]}
            point.update((name, float(value)) for name, value in METRIC.findall(line))
            if 'val_sign_accuracy' in point and 'val_rejection' in point and 'val_balanced_accuracy' not in point:
                # The measure that chooses the classifier's best epoch: the mean of the two.
                point['val_balanced_accuracy'] = (point['val_sign_accuracy'] + point['val_rejection']) / 2
            if epoch[0].startswith('experiment_b') and 'val_accuracy' in point:
                # The measure that chooses experiment B's best epoch (context_experiment.selection_score): the mean of the
                # reading accuracy and the ownership macro F1, a missing one ("-") counting 0.
                point['val_selection_score'] = (point['val_accuracy'] + point.get('val_ownership_macro_f1', 0.0)) / 2
            seconds = SECONDS.search(line)
            if seconds:
                point['seconds'] = int(seconds.group(1))
            curve = curves.setdefault(epoch[0], {'epochs': epoch[2], 'points': []})
            curve['epochs'] = epoch[2]
            # A resumed run writes its epochs again: the last line of an epoch wins.
            curve['points'] = [p for p in curve['points'] if p['epoch'] != epoch[1]] + [point]
            if stopped.pop(epoch[0], None) is not None:
                curve.pop('stopped_early', None)
        match = EARLY.match(line)
        if match:
            stopped[match.group(1)] = int(match.group(2))
            if match.group(1) in curves:
                curves[match.group(1)]['stopped_early'] = int(match.group(2))
        match = COMMA_ROUTES.match(line)
        if match:
            comma_routes = match.group(1).split()
        match = ROUTE_LINE.match(line) if comma_routes else None
        if match and match.group(1) in comma_routes:
            index = comma_routes.index(match.group(1))
            if index != at_route:
                at_route, segments = index, (0, 0)
        if FETCHED.match(line):
            fetched += 1
            segments = (segments[0] + 1, segments[1])
        match = ON_COMMA.match(line)
        if match:
            route = match.group(1)
            wanted = TO_DOWNLOAD.search(line)
            segments = (0, len(wanted.group(1).split()) if wanted else 0)
        match = PREANNOTATING.match(line)
        if match and int(match.group(2)):
            proposed = (int(match.group(1)), int(match.group(2)))
    # Step 4 of two stages is the pretraining until the script starts the second ("-- the second stage").
    if two_stages and current == 'classifier' and not SECOND_STAGE.search(text):
        current = 'pretrain'
    if current in keys:
        done.update(keys[:keys.index(current)])
    if complete:
        # SignNet wrote its result: every step is over, also in a log read from a terminal.
        done.update(keys)

    inside = _inside_step(current, epoch, substep, pipeline, stopped) if current in keys and not complete else None
    finished_ok = state == 'ok'
    rows = []
    for key, label, _ in steps:
        if finished_ok or key in done:
            step_state = 'done'
        elif key == current:
            step_state = 'current' if state == 'running' else 'failed' if state == 'failed' else 'todo'
        else:
            step_state = 'todo'
        rows.append({'key': key, 'label': label, 'state': step_state})

    percent: float | None = None
    detail = ''
    if finished_ok:
        percent = 100.0
    elif action in WEIGHTED:
        total = sum(weight for _, _, weight in steps)
        reached = sum(weight for key, _, weight in steps if key in done)
        if inside is not None:
            weight = dict((key, weight) for key, _, weight in steps)[current]
            reached += weight * inside['done'] / inside['total']
        percent = round(reached * 100 / total, 1)
    elif action in ('prepare', 'prepare_again') and preparation is not None and preparation.get('percent') is not None:
        percent = float(preparation['percent'])
    elif action == 'measure_proposer' and proposed is not None:
        percent = round(min(proposed) * 100 / proposed[1], 1)
    elif action == 'fetch' and comma_routes:
        inside = min(segments[0] / segments[1], 1.0) if segments[1] else 0.0
        percent = round((max(at_route, 0) + inside) * 100 / len(comma_routes), 1)

    if epoch is not None and epoch[0] == current and state == 'running' and epoch[0] in stopped:
        detail = NET_NAMES.get(epoch[0], epoch[0]) + ' · stopped early after epoch ' + str(stopped[epoch[0]])
    elif epoch is not None and epoch[0] == current and state == 'running':
        # A variant of the decider by its step's label ("Crop, position and speed"), a network by its name.
        name = dict((key, label) for key, label, _ in steps).get(epoch[0], epoch[0]) \
            if epoch[0].startswith(('experiment_b', 'road_type')) else NET_NAMES.get(epoch[0], epoch[0])
        detail = name + ' · epoch ' + str(epoch[1]) + ' of ' + str(epoch[2])
    elif action == 'prepare_again' and preparation is not None and preparation.get('again'):
        detail = again_detail(preparation)
    elif action == 'prepare' and preparation is not None and preparation.get('routes_total'):
        detail = str(preparation['routes_done']) + ' of ' + str(preparation['routes_total']) + ' drives ready'
    elif action == 'fetch' and (fetched or route):
        detail = (str(fetched) + (' segment downloaded' if fetched == 1 else ' segments downloaded')
                  + (' · drive ' + str(at_route + 1) + ' of ' + str(len(comma_routes))
                     if comma_routes and at_route >= 0 and state == 'running' else '')
                  + (' · ' + route if route and state == 'running' else ''))
    elif action == 'measure_proposer' and proposed is not None and state == 'running':
        detail = f'{proposed[0]:,} of {proposed[1]:,} frames'
    result = {'steps': rows, 'percent': percent, 'detail': detail, 'last_line': last_line(text),
              'step_progress': inside if state == 'running' and current in keys else None, 'curves': curves}
    if action == 'train_classifier':
        # The chain step runs evaluate-chain several times (test, validation, then test with the old reader), each
        # counting its frames from 0: with_step_progress counts the runs done, not to go back from 100% to 0.
        runs_done = len(CHAIN_RUN_DONE.findall(text))
        runs_total = CHAIN_RUNS - (1 if 'no old reader' in text else 0)
        result['chain_runs'] = {'done': min(runs_done, runs_total), 'total': runs_total}
        result['two_stages'] = two_stages
    if action == 'train_cutter':
        # Which steps the log has (SIGNNET_STEPS, or the old detector's CUTTER_STEPS), for with_step_progress.
        result['signnet'] = signnet
    preparing = action in ('prepare', 'prepare_again') and preparation is not None and state == 'running'
    if preparing:
        # Its drives are listed with their phase under the card: the raw line of the helper says it again.
        result['last_line'] = ''
    if preparing and preparation.get('again'):
        # The time left of Propose again, from the pace of the drive at work: the only one the card shows.
        result['seconds_left'] = preparation.get('seconds_left')
    if state == 'failed':
        result['failure'] = failure_text(text)
    return result


def _inside_step(current: str, epoch: tuple[str, int, int] | None, substep: tuple[int, str] | None,
                 pipeline: dict[int, str] = PIPELINE, stopped: dict[str, int] | None = None) -> dict | None:
    """{'done', 'total', 'percent', 'text'} of the current step from its log: the epochs written by a training
    (all of them once it stopped early), or which of its numbered sub-steps it is in (the checks of
    scripts/run_v100_baseline.sh are three, the export and the bundle two: the second of three is at 33%). None
    when the log says nothing more."""
    if epoch is not None and epoch[0] == current and stopped and current in stopped:
        return {'done': 1, 'total': 1, 'percent': 100.0, 'text': 'stopped early after epoch ' + str(stopped[current])}
    if epoch is not None and epoch[0] == current and epoch[2]:
        done, total = min(epoch[1], epoch[2]), epoch[2]
        return {'done': done, 'total': total, 'percent': round(done * 100 / total, 1),
                'text': 'epoch ' + str(epoch[1]) + ' of ' + str(total)}
    parts = sorted(number for number, step in pipeline.items() if step == current)
    if substep is None or substep[0] not in parts or len(parts) < 2:
        return None
    done = parts.index(substep[0])
    return {'done': done, 'total': len(parts), 'percent': round(done * 100 / len(parts), 1),
            'text': substep[1] + ' · ' + str(done + 1) + ' of ' + str(len(parts))}


# scripts/run_classifier.sh: the runs of nnslr evaluate-chain in its chain step (test and validation, then test with
# the old reader), and the line each writes when it is done or skipped.
CHAIN_RUNS = 3
CHAIN_RUN_DONE = re.compile(r'^(?:evaluated the chain on |skip evaluated chain )', re.M)
# A classifier run in two stages (scripts/run_classifier.sh with NNSLR_CLASSIFIER_MAPILLARY=pretrain) says so in its
# head ("pretraining:  --classes v2 ... --mapillary-only"): the pretraining on the Mapillary crops alone ("pretrain epoch
# 3/30 ..."), then the training on ours from it ("classifier epoch ..."), a step each (CLASSIFIER_TWO_STAGE_STEPS).
PRETRAINING = re.compile(r'^pretraining:\s', re.MULTILINE)
SECOND_STAGE = re.compile(r'^-- the second stage', re.MULTILINE)
# A network by its name in the line under the progress bar ("Mapillary · epoch 3 of 30").
NET_NAMES = {'pretrain': 'Mapillary', 'signnet': 'SignNet'}


def is_signnet(action: str, text: str) -> bool:
    """Whether a cutter training's log is a SignNet's (scripts/run_signnet.sh, its step, its epochs): else it is the old
    detector's of scripts/run_detector_sizes.sh, with its own steps."""
    return action == 'train_cutter' and SIGNNET_LOG.search(text) is not None


def steps_of(action: str, two_stages: bool = False, signnet: bool = False) -> tuple:
    """The steps of an action: a classifier in two stages has its pretraining as a step of its own; the cutter is a
    SignNet, or the old detector."""
    if action == 'train_cutter' and signnet:
        return SIGNNET_STEPS
    return CLASSIFIER_TWO_STAGE_STEPS if action == 'train_classifier' and two_stages else STEPS.get(action, ())

# The tasks that report their own progress (nnslr_tools.progress) and the step they belong to.
TASK_STEPS = {'import': 'import', 'validate': 'split', 'split': 'split', 'detector': 'detector',
              'reader': 'reader', 'evaluate': 'evaluate', 'hard_examples': 'export', 'export': 'export',
              # scripts/run_classifier.sh: the mining of the cutter's false boxes, the classifier, the chain.
              'mine': 'mine', 'classifier': 'classifier', 'chain': 'chain',
              # Its first stage on the Mapillary crops alone (NNSLR_CLASSIFIER_MAPILLARY=pretrain).
              'pretrain': 'pretrain',
              # [ai-reviewer] - START
              # nnslr ai-review and nnslr measure-ai-reviewer: a frame at a time.
              'ai_review': 'review', 'ai_measure': 'measure'}
              # [ai-reviewer] - END


def with_step_progress(result: dict, action: str, record: dict | None, setup_line: str = '',
                       download: dict | None = None) -> dict:
    """Adds 'step_progress' ({'done', 'total', 'percent', 'text'}) for the current step of a running job:
    from the last report of a command (when it belongs to that step) or, while the GPU environment is
    made, the last line of its log; otherwise what the log alone says (progress(): epochs, sub-steps). With a
    count, the job's percent counts it too."""
    current = next((step['key'] for step in result['steps'] if step['state'] == 'current'), None)
    if current is None:
        return result
    step_progress = result.get('step_progress')
    if record is not None and TASK_STEPS.get(record.get('task')) == current and record.get('total'):
        done, total = int(record['done']), int(record['total'])
        share = min(done, total) / total
        text = str(record.get('text') or '')
        runs = result.get('chain_runs') if current == 'chain' else None
        if runs and runs['total'] > 1:
            # One run of several: the runs done count, then this one's frames.
            share = min(runs['done'] + share, runs['total']) / runs['total']
            text = text + ' · run ' + str(min(runs['done'] + 1, runs['total'])) + ' of ' + str(runs['total'])
        step_progress = {'done': done, 'total': total, 'percent': round(share * 100, 1), 'text': text}
        steps = steps_of(action, bool(result.get('two_stages')), bool(result.get('signnet')))
        if action in WEIGHTED and steps:
            weights = {key: weight for key, _, weight in steps}
            reached = sum(weights[step['key']] for step in result['steps'] if step['state'] == 'done')
            result = {**result, 'percent': round((reached + weights[current] * share) * 100
                                                 / sum(weights.values()), 1)}
    elif current == 'gpu_env' and download is not None and download['written']:
        # The file pip is downloading: the GPU environment has several, of sizes known only when they start.
        written, total = download['written'], download['total']
        step_progress = {'done': round(written / 1e6), 'total': round(total / 1e6),
                         'percent': round(min(written, total) * 100 / total, 1),
                         'text': f"Downloading {download['name']} · {written / 1e6:.0f} of {total / 1e6:.0f} MB"}
    elif current == 'gpu_env' and setup_line:
        step_progress = {'done': 0, 'total': 0, 'percent': None, 'text': setup_line[:200]}
    return {**result, 'step_progress': step_progress}
