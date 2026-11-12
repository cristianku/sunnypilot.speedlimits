# [control-panel] - START
"""Control panel of the review site: the state of every route, the machine, and buttons that run
the easy helpers (helpers/fetch_comma_segments.sh, helpers/1_prepare.sh, helpers/3_train.sh,
helpers/update.sh; scripts/run_signnet.sh for the cutter and scripts/run_classifier.sh for the
classifier with "not a sign"; helpers/ai_review.sh and helpers/measure_ai_reviewer.sh for the AI reviewer of
Settings) in the background with a live log.

Only the fixed actions of ACTIONS can run, never a command taken from a request. They are the
helpers of the checkout this package runs from (editable install); without them (wheel
install) the panel is read-only. NNSLR_PANEL_ACTIONS=0 also makes it read-only, e.g. for a
review service that other people use. The output of every action goes to
<data-root>/logs/panel/<job id>.log, and its last line is the exit code, so the history
survives a restart of the server.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import sqlite3
import stat
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from nnslr_tools.car_release import CAR_SCRIPT, brain_refused, build_pending, car_missing
from nnslr_tools.comma_trial import TRY_SCRIPT, CommaProbe, read_trial, trial_missing
from nnslr_tools.job_progress import last_line, progress, with_step_progress
from nnslr_tools.model_roles import (BRAIN_WEIGHTS, CAR_NETWORKS, NOT_SELECTABLE, SIGNNET_PART, ModelChoiceError,
                                     ModelIndex)
from nnslr_tools.runs import RunsIndex, WatchError
from nnslr_tools.usernet import BDD_DIR
from nnslr_tools.usernet import LOG as USERNET_LOG
from nnslr_tools.usernet import WEIGHTS as USERNET_WEIGHTS

REPO = Path(__file__).resolve().parents[2]
ACTIONS_ENV = 'NNSLR_PANEL_ACTIONS'
PREPARE_JOBS_ENV = 'NNSLR_PANEL_PREPARE_JOBS'
LOG_CHUNK_BYTES = 64 * 1024
# The progress of a job is read from at most the last 4 MB of its log.
PROGRESS_BYTES = 4 * 1024 * 1024
# Written by helpers/1_prepare.sh when it starts: the routes of that run, for the progress.
BATCH_FILE = 'logs/prepare/batch.json'
HISTORY = 12
# A download announced by pip in the GPU setup log.
DOWNLOADING = re.compile(r'Downloading (\S+?)(?:\.whl)? \(([0-9.]+) (kB|MB|GB)\)')
RUN_NAME = re.compile(r'[A-Za-z0-9._-]+')
STABLE_CHECK_SECONDS = 600
JOB_ID = re.compile(r'[0-9]{8}-[0-9]{6}-[a-z_]+')
# The last line of a log; "finito con codice" in the logs written before it was in English.
FINISHED = re.compile(r'== (?:finished with code|finito con codice) (\d+)\s*$')
# Runs the helper and appends its exit code as the last line of the log.
WRAPPER = '"$@"; code=$?; echo; echo "== finished with code $code"; exit $code'
# The niceness of the jobs: the site (nice 0) gets the CPU first when a job keeps every core busy.
JOB_NICE = 10
# The wrapper of the actions started before the footer was in English: still running ones are still ours (_group).
OLD_WRAPPERS = ('"$@"; code=$?; echo; echo "== finito con codice $code"; exit $code',)
# A training run's name, as helpers/3_train.sh writes it: "== d. training <run> (<mode>) -> ...".
RUN_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}')
RUN_LINE = re.compile(r'^== d\. training (\S+) \(', re.M)
# The classifier's and the cutter's trainings: the panel names their run after the command ("== scripts/run_classifier.sh
# <run>", "== scripts/run_signnet.sh <run>", the first line of the log), and the script says it again near its top
# ("run:          <run>").
CLASSIFIER_SCRIPT = 'scripts/run_classifier.sh'
# Where "Train the classifier" starts ({'start': ...}): ImageNet (the default), or a finished classifier with "not a
# sign" of the Models tab, by its id (its reader-best.pt becomes NNSLR_CLASSIFIER_INIT_FROM).
CLASSIFIER_IMAGENET = 'imagenet'
CLASSIFIER_ID = re.compile(r'classifier:([A-Za-z0-9][A-Za-z0-9._-]{0,99})/classifier')
# Where the cutter's, the applicability's and the brain's trainings start ({'start': ...}): their default (NETWORK_FRESH:
# random weights for SignNet, ImageNet for the others), or a finished version of their role in Models, by its id; its
# script gets the earlier run's weights or folder (the variable here), where each applicability variant goes on from its
# own and every seed of the brain from the one brain.
NETWORK_FRESH = 'fresh'
NETWORK_STARTS = {'train_cutter': ('cutter', 'NNSLR_CUTTER_INIT_FROM'), 'train_decider': ('decider', 'NNSLR_DECIDER_INIT_FROM'),
                  'train_brain': ('brain', 'NNSLR_BRAIN_INIT_FROM'), 'train_users': ('users', 'NNSLR_USERNET_INIT_FROM')}
# A split file's sha256 (splits/latest.json).
SPLIT_SHA = re.compile(r'[0-9a-f]{64}')
# The cutter is SignNet, the network the comma runs, trained with the car's setup (896 x 288, stride 8). The jobs of
# the old cutter, a detector of scripts/run_detector_sizes.sh NAME SIZE, still name their run (RUN_LINES).
CUTTER_SCRIPT = 'scripts/run_signnet.sh'
# The decider: experiment B, its three variants (the crop alone; with position and speed; with the scene), on a run of
# its own (start() adds <time>-decider).
DECIDER_SCRIPT = 'scripts/run_experiment_b.sh'
# The brain, the network that says the speed to hold: its three seeds on runs of their own (start() adds <time>-brain, the
# script -s0, -s1, -s2), or seed 0 alone. What it learns: {'works': 'on'|'off', 'road', 'types', 'pool': 'grid'|'mean',
# 'seeds': 'all'|'one'}, the variables
# of the script, the first of each the usual one. The driver's inputs are gone since 10 Nov 2026 (brain_targets.LEFT_OUT):
# 'driver' is still taken as 'off', the only value, so that a page loaded before can start a training.
BRAIN_SCRIPT = 'scripts/run_brain.sh'
# UserNet, the detector of the road users (nnslr_tools.usernet), on a run of its own (start() adds <time>-users). Its
# option {'stage': 'bdd' | 'drives'}: BDD100K's pictures or the road users boxed in Sign review (USERNET_STAGES, the
# script's NNSLR_USERNET_STAGE); where it starts is a finished UserNet of Models (NETWORK_STARTS), or random weights.
USERNET_SCRIPT = 'scripts/run_usernet.sh'
USERNET_STAGES = ('bdd', 'drives')
BRAIN_WORKS = {'on': '1', 'off': '0'}
BRAIN_ROAD = {'on': '1', 'off': '0'}
BRAIN_TYPES = {'on': '1', 'off': '0'}
# Train › Brain's Road users (option B, #75, 12 Nov 2026): the brain also learns where the road users are and how big,
# a head only the training has (brain_net.users_head; nnslr road-users reads them first); off: --no-users.
BRAIN_USERS = {'on': '1', 'off': '0'}
# Train › Brain's Drives: every dashcam drive (none given), or the drives of an earlier brain run, by its NAME (its
# routes.json, in runs/NAME/ or in a seed's runs/NAME-sN/): two trainings on the same drives, compared fairly in Models.
BRAIN_SEED_RUN = re.compile(r'(.+)-s[0-9]')
BRAIN_POOLS = ('grid', 'mean')
# Train › Brain's Seeds: the three (NAME-s0, -s1, -s2, to compare in Models) or seed 0 alone, a third of the time.
BRAIN_SEEDS = {'all': '0 1 2', 'one': '0'}
# Models › Regression: the models he chose along the certified drives, only those still missing (nnslr_tools.regression).
REGRESSION_SCRIPT = 'scripts/run_regression.sh'
# Models › Drive (nnslr_tools.simulation): a recorded drive replayed through the car's own vision loop with a set of
# networks, for the comma's screen over its video and the regression test on the certified drives. Always on a V100 (the
# video is decoded and the networks run there): it has no run on the CPU. Its options: {'set': 'car' | 'in-use'} the
# networks, and one of {'route': R} a drive, {'certified': '1'} every certified drive.
SIMULATE_SCRIPT = 'helpers/simulate_drive.sh'
SIMULATE_OPTIONS = ('set', 'certified')
# The map's limit of every frame (Read the map, on Data › Speed review): the GPS of the drives, the maps they need, the
# map roads near them and the way each frame is on.
ROAD_TYPES_SCRIPT = 'helpers/road_types.sh'
# The road labeler (docs/ROAD_LABELER.md), an offline network that labels Speed review, never on the car: trained and
# run along the drives with the GPU of .venv-gpu (Models › Versions, the Road labeler card). Its options: {'dry': 'on'}
# a dry run on the CPU; Prepare Speed review takes the run it uses ({'labeler': run with road-labeler.pt}) and writes
# everything its heads earned (the marks of the kinds of road and of the roadworks, the layer of road types), or with
# {'marks': 'off'} only the chances, nothing into Speed review.
ROAD_LABELER_SCRIPT = 'helpers/road_labeler.sh'
ROAD_LABELER_NNSLR = '.venv-gpu/bin/nnslr'
ROAD_LABELER_ACTIONS = ('road_labeler_train', 'road_labeler_predict')
ROAD_LABELER_OPTIONS = ('dry', 'labeler', 'marks')
# A drive's id.
ROUTE_ID = re.compile(r'^[0-9a-f]{8}--[0-9a-f]{10}$')
# Fit box of the review page (key C, nnslr_tools.box_fit) trained again on the boxes people drew (Models, the Fit box
# card): the two rankers (colour trees, edge MLPs), then the box refiner, a small CNN with the PyTorch of .venv-gpu, all on
# the CPU; refiner.onnx is replaced only when it beats the rankers on routes left out. Its options: {'dry': 'on'} only
# counts the boxes; {'refiner': 'off'} the rankers alone (--no-refiner). Its summaries: BOX_FIT_SUMMARIES of the data root.
BOX_FIT_SCRIPT = 'scripts/train_box_fit.sh'
BOX_FIT_PYTHON = '.venv-gpu/bin/python'
BOX_FIT_OPTIONS = ('refiner',)
BOX_FIT_SUMMARIES = Path('evaluations') / 'box_fit'
# Where the photos of the Mapillary Traffic Sign Dataset were taken (Train › Classifier, its region): asked of
# Mapillary with the token of .env, written into the data root (nnslr mapillary-locations).
LOCATIONS_SCRIPT = 'helpers/mapillary_locations.sh'
# The MTSD photos cut into windows the size of the car's band (Train › Cutter, its Mapillary switch): read from the
# TrueNAS share, written into the data root (nnslr mapillary-windows).
WINDOWS_SCRIPT = 'helpers/mapillary_windows.sh'
# Our three networks in use as one package (ONNX, model card), and that package on Hugging Face.
NETWORKS_SCRIPT = 'scripts/publish_networks.sh'
NETWORKS_REPO = os.environ.get('HF_REPO', 'cristianku/sunnypilot-speedlimits-models')
NETWORK_PACKAGES_SHOWN = 5
# The drives' table of the status (helpers/route_status.route_rows) stats every video and counts the lines of every
# frames.jsonl on the share: about 9 s on 1 800 segments. On the site it is read in a thread (Panel.background_rows) and
# read again once it is older than this, or once a job began or ended since.
ROUTE_ROWS_FRESH_S = 20.0
RUN_ARGUMENT_LINE = re.compile(r'^run:\s+(\S+)\s*$', re.M)
RUN_LINES = {
    'train_classifier': (re.compile(r'^== scripts/run_classifier\.sh (\S+)\s*$', re.M), RUN_ARGUMENT_LINE),
    'train_cutter': (re.compile(r'^== scripts/run_signnet\.sh (\S+)\s*$', re.M),
                     re.compile(r'^== scripts/run_detector_sizes\.sh (\S+)(?: \S+)*\s*$', re.M), RUN_ARGUMENT_LINE),
    'train_decider': (re.compile(r'^== scripts/run_experiment_b\.sh (\S+)\s*$', re.M), RUN_ARGUMENT_LINE),
    'train_brain': (re.compile(r'^== scripts/run_brain\.sh (\S+)\s*$', re.M), RUN_ARGUMENT_LINE),
    'train_users': (re.compile(r'^== scripts/run_usernet\.sh (\S+)\s*$', re.M), RUN_ARGUMENT_LINE),
    # helpers/road_labeler.sh: "run:          <name>" (the run trained, or the run predict uses).
    'road_labeler_train': (RUN_ARGUMENT_LINE,),
    'road_labeler_predict': (RUN_ARGUMENT_LINE,),
    # The SignNet being published ("== Send to the car: SignNet <run>, classifier <id>", scripts/send_to_car.sh).
    'send_to_car': (re.compile(r'^== Send to the car: SignNet ([^\s,]+)', re.MULTILINE),),
    'try_on_comma': (re.compile(r'^== Try on the comma: SignNet ([^\s,]+)', re.MULTILINE),),
}
# Train › Pipeline: our networks trained one after the other, in this order, each a job of its own with the choices of
# its own button (start()); the next one starts when one ends well (advance_pipeline), each on the networks the pipeline
# has just trained: the classifier on its SignNet, the brain's signs read with its cutter, classifier and applicability
# network (#71). Once all ended well, every network it trained is put in use (_put_in_use). The plan and how far it is:
# PIPELINE_FILE in the panel's logs.
PIPELINE_ORDER = ('dataset', 'train_cutter', 'train_classifier', 'train_decider', 'train_brain')
PIPELINE_FILE = 'pipeline.json'
# How often the pipeline looks whether its training ended, besides the moment its process ends (also after a restart
# of this server, which loses that process).
PIPELINE_WATCH_SECONDS = 20
# A network a pipeline trains before the step that uses it, while the pipeline is only checked: unknown until then (the
# classifier's cutter, the networks that read the brain's signs).
PIPELINE_NETWORK = 'pipeline'
# The role of the network each training of a pipeline makes.
PIPELINE_ROLES = {'train_cutter': 'cutter', 'train_classifier': 'classifier', 'train_decider': 'decider',
                  'train_brain': 'brain'}
# The applicability network a pipeline takes of the three versions its training makes: the variant of the one in use,
# this one when none is.
PIPELINE_DECIDER_VARIANT = 'full'
# Each network in the line of a pipeline that put it in use.
PIPELINE_WORDS = {'cutter': 'cutter', 'classifier': 'classifier', 'decider': 'applicability', 'brain': 'brain'}
# The actions that take a SignNet of the Models tab ({'signnet': run}).
SIGNNET_ACTIONS = ('send_to_car', 'try_on_comma')
# The cutter's training (scripts/run_signnet.sh NAME): the car's band, always, and the boxes it learns to find
# (NNSLR_CUTTER_FAMILIES): every sign (the speed signs and the other signs) or the speed signs only.
CUTTER_BAND = '896x288'
CUTTER_FAMILIES = ('all', 'speed')
CUTTER_FAMILY = 'all'
# The trainings with a Mapillary switch and the variable of their script (NNSLR_..._MAPILLARY: 1 or pretrain; its
# _REGION and _SHARE): the classifier's crops (helpers/mapillary.sh), the cutter's windows (helpers/mapillary_windows.sh).
MAPILLARY_ENV = {'train_classifier': 'NNSLR_CLASSIFIER_MAPILLARY', 'train_cutter': 'NNSLR_CUTTER_MAPILLARY'}
# The networks that read the signs a brain learns from (sign_networks, #31): its option, the Models role it is a version
# of, the variable of scripts/run_brain.sh. None given: those in use.
BRAIN_SIGN_NETWORKS = {'sign_cutter': ('cutter', 'NNSLR_BRAIN_CUTTER'),
                       'sign_classifier': ('classifier', 'NNSLR_BRAIN_CLASSIFIER'),
                       'sign_decider': ('decider', 'NNSLR_BRAIN_DECIDER')}
# The options an action takes, besides a training's start and the run to resume (checked by Panel.start).
# The GPU memory Qwen3-VL-8B takes in float16 (scripts/setup_vlm.sh: 17.5 GB), with a margin for one frame.
QWEN_GPU_MB = 18 * 1024
ACTION_OPTIONS = {'train_cutter': ('families', 'start', 'mapillary', 'region', 'share'), 'train_decider': ('start',),
                  'train_brain': ('works', 'road', 'types', 'driver', 'pool', 'seeds', 'start', 'sign_cutter',
                                  'sign_classifier', 'sign_decider', 'users', 'routes_from'),
                  'train_users': ('stage', 'start'),
                  'regression': ('openpilot', 'brains'),
                  'simulate': ('route', 'set', 'certified'),
                  'road_labeler_train': ('dry',), 'road_labeler_predict': ('dry', 'labeler', 'marks'),
                  'box_fit_train': ('dry', 'refiner'),
                  'train_classifier': ('mapillary', 'region', 'share', 'start', 'cutter'),
                  'prepare': ('proposer',), 'prepare_again': ('proposer', 'route'),
                  'measure_proposer': ('proposer',), 'networks_publish': ('package',), 'send_to_car': ('signnet', 'rebuild'),
                  'try_on_comma': ('signnet',),
                  'ai_review': ('route', 'provider', 'save', 'limit'), 'measure_ai_reviewer': ('provider', 'limit')}
# [ai-reviewer] - START
# The AI actions and their options (strings; checked by Panel._ai_arguments). The provider is an id of Settings
# (nnslr_tools.llm_providers): its key never goes into a command line, an environment or a log, the job reads it.
AI_ACTIONS = ('ai_review', 'measure_ai_reviewer')
AI_OPTIONS = ('route', 'provider', 'save', 'limit')
PROVIDER_ID = re.compile(r'[a-z][a-z0-9-]{0,31}')
MAX_AI_LIMIT = 5000
# [ai-reviewer] - END
# Who proposes the signs of a route (nnslr preannotate --proposer): the choice of the Prepare tab, kept on the server
# (PROPOSER_FILE in the data root) once chosen; before that, the one measured best (Panel.proposer_choice).
PROPOSERS = ('vietnam', 'ours', 'qwen')
PROPOSER_FILE = 'web/proposer.json'
# The proposer when none is measured yet: the helpers' own default (helpers/1_prepare.sh).
DEFAULT_PROPOSER = 'vietnam'
# The actions that take a proposer, and the two that prepare routes (their progress is the preparation's).
PROPOSER_ACTIONS = ('prepare', 'prepare_again', 'measure_proposer')
# The log of Review with AI (helpers/ai_review.sh, nnslr ai-review): its first line names the drive and whether it
# saves, then a line per frame done, "[37/1390] 2/71: 3 signs (3 added) · 15.2 s · saved".
AI_LIVE_ROUTE = re.compile(r'^Reviewing the route ([0-9a-f]{8}--[0-9a-f]{10}) with the AI, (saving|a trial)', re.M)
AI_LIVE_STEP = re.compile(r'^\[(\d+)/(\d+)\] (\d+/\d+): ', re.M)
PREPARING = ('prepare', 'prepare_again')
# A proposer in the log of a job started before the panel kept its options (<job id>.json): "Measuring the proposer
# qwen" (helpers/measure_proposer.sh), "proposing the signs (again) (qwen)" (helpers/1_prepare.sh).
PROPOSER_LINE = re.compile(r'(?:Measuring the proposer |(?:proposing the signs(?: again)?|propongo di nuovo i cartelli) \()'
                           r'(vietnam|ours|qwen)\b')


def _run_of(log: Path, action: str = 'train') -> str | None:
    """The run a training job wrote to, from the start of its log (the line comes after the import; for the
    classifier's training, the first line)."""
    try:
        with log.open('rb') as handle:
            head = handle.read(65536).decode('utf-8', 'replace')
    except OSError:
        return None
    for pattern in RUN_LINES.get(action, (RUN_LINE,)):
        match = pattern.search(head)
        if match and RUN_NAME.fullmatch(match.group(1)):
            return match.group(1)
    return None

# The trainings whose log starts with how they were launched (scripts/run_signnet.sh, run_classifier.sh,
# run_experiment_b.sh, and the old scripts/run_detector_sizes.sh): "band:         896x288", "sizes:        1344x760",
# "settings:     --epochs 20 --batch-size 8 ... --pretrained".
SETTING_LINE = re.compile(r'^([a-z][a-z ]{1,20}):\s+(.+)$')
FAMILY_TEXT = {'all': 'every sign', 'speed': 'speed signs only'}
USERNET_STAGE_WORDS = {'bdd': 'BDD100K', 'drives': 'your drives'}
# The "negatives:" line of scripts/run_classifier.sh: the score, per frame and factor of its mined false boxes.
NEGATIVES_LINE = re.compile(r'score >= (\S+) .*at most (\S+) per frame and (\S+) x')
NEGATIVES_DEFAULT = ('0.1', '3', '3')


def _settings_of(log: Path, action: str) -> list[str]:
    """How a training of the cutter, the classifier, the decider, the brain or UserNet was launched, as short chips from
    the head of its log: ["896 × 288", "every sign", "stride 8", "40 epochs", "batch 16", "learning rate 0.003"],
    ["1344 × 760", "speed signs only", "20 epochs", "batch 8", "pretrained"], ["BDD100K", "30 epochs", "batch 32", ...];
    [] for other jobs."""
    if action not in ('train_cutter', 'train_classifier', 'train_decider', 'train_brain', 'train_users'):
        return []
    try:
        with log.open('rb') as handle:
            head = handle.read(8192).decode('utf-8', 'replace')
    except OSError:
        return []
    fields: dict[str, str] = {}
    for line in head.splitlines()[1:40]:
        if line.startswith('== ') and fields:
            break
        match = SETTING_LINE.match(line)
        if match:
            fields.setdefault(match.group(1).strip(), match.group(2).strip())
    flags = _flags(fields.get('settings', ''))
    # A classifier in two stages (run_classifier.sh, NNSLR_CLASSIFIER_MAPILLARY=pretrain): its Mapillary crops are
    # chosen in the first ("pretraining:  --classes v2 ... --mapillary-only"), the second starts from it.
    pretraining = _flags(fields.get('pretraining', ''))
    if pretraining:
        flags = {**{key: value for key, value in pretraining.items() if key.startswith('--mapillary')}, **flags}
    chips = []
    # UserNet's stage (scripts/run_usernet.sh, "settings:     --stage bdd ...").
    if flags.get('--stage'):
        chips.append(USERNET_STAGE_WORDS.get(flags['--stage'], flags['--stage']))
    if fields.get('band'):
        chips.append(fields['band'].replace('x', ' × '))
    elif flags.get('--band-width'):
        chips.append('band ' + flags['--band-width'])
    if fields.get('sizes'):
        chips += [size.replace('x', ' × ') for size in fields['sizes'].split()]
    # The detector's flag (--cutter-families) and SignNet's (--families).
    families = flags.get('--cutter-families') or flags.get('--families')
    if families:
        chips.append(FAMILY_TEXT.get(families, families))
    if flags.get('--stride'):
        chips.append('stride ' + flags['--stride'])
    if flags.get('--classes'):
        chips.append('classes ' + flags['--classes'])
    if pretraining:
        chips.append('Mapillary first, then ours')
    elif '--mapillary' in flags:
        chips.append('Mapillary signs')
    if flags.get('--mapillary-region'):
        chips.append('Mapillary ' + _region_words(flags['--mapillary-region']))
    if flags.get('--mapillary-share'):
        chips.append('Mapillary ' + flags['--mapillary-share'] + ' %')
    # The cutter's cap of the windows added to ours (NNSLR_CUTTER_MAPILLARY_RATIO).
    if flags.get('--mapillary-ratio'):
        chips.append('Mapillary ' + flags['--mapillary-ratio'] + ' × ours')
    # The first stage's epochs (Train › Classifier › Settings), when not the same as the second's.
    if pretraining.get('--epochs') and pretraining['--epochs'] != flags.get('--epochs'):
        chips.append('Mapillary ' + pretraining['--epochs'] + ' epochs')
    epochs = flags.get('--epochs') or fields.get('epochs')
    if epochs:
        chips.append(epochs + ' epochs')
    if flags.get('--patience'):
        chips.append('patience ' + flags['--patience'])
    # The brain's choices (scripts/run_brain.sh), when not the usual ones. The driver's inputs, of the runs of before
    # 10 Nov 2026 only: --driver turned them on, and a run of before 8 Nov that left them out said --no-driver.
    if '--no-works' in flags:
        chips.append('no roadworks')
    if '--no-road' in flags:
        chips.append('no kinds of road')
    if '--no-road-types' in flags:
        chips.append('no road types')
    if '--no-users' in flags:
        chips.append('no road users')
    if '--driver' in flags:
        chips.append('driver’s inputs')
    elif '--no-driver' in flags:
        chips.append('no driver’s inputs')
    if flags.get('--pool') == 'mean':
        chips.append('picture as a mean')
    if flags.get('--batch-size') or flags.get('--batch'):
        chips.append('batch ' + (flags.get('--batch-size') or flags['--batch']))
    if flags.get('--learning-rate'):
        chips.append('learning rate ' + flags['--learning-rate'])
    # The classifier's and the road type's flag; SignNet's own line ("workers:      16").
    workers = flags.get('--num-workers') or fields.get('workers')
    if workers:
        chips.append(workers + ' workers')
    # The classifier's false boxes ("negatives:    cutter boxes of score >= 0.1 ..., at most 3 per frame and 3 x the
    # sign crops per partition"), when not the defaults of its script.
    negatives = NEGATIVES_LINE.search(fields.get('negatives', ''))
    if negatives and negatives.groups() != NEGATIVES_DEFAULT:
        chips.append('false boxes ≥ {}, {} a frame, {} × signs'.format(*negatives.groups()))
    if flags.get('--anchor-min-ratio'):
        chips.append('default box ' + flags['--anchor-min-ratio'])
    if flags.get('--negative-frame-fraction'):
        chips.append('sign-free ' + flags['--negative-frame-fraction'])
    if fields.get('cutter'):
        chips.append('cutter ' + fields['cutter'])
    if '--pretrained' in flags or '--pretrained' in pretraining:
        chips.append('pretrained')
    elif flags.get('--init-from') and not pretraining:
        chips.append('from ' + _start_words(flags['--init-from']))
    elif pretraining.get('--init-from'):
        # A cutter's first stage on the Mapillary windows can go on from a SignNet (NNSLR_CUTTER_INIT_FROM).
        chips.append('from ' + _start_words(pretraining['--init-from']))
    elif fields.get('init'):
        chips.append('from ' + fields['init'])
    if fields.get('dataset kind'):
        chips.append(fields['dataset kind'] + ' dataset')
    return chips


def _flags(settings: str) -> dict[str, str]:
    """The flags of a settings line and their values: {'--epochs': '20', '--pretrained': ''}."""
    words = settings.split()
    return {word: (words[index + 1] if index + 1 < len(words) and not words[index + 1].startswith('--') else '')
            for index, word in enumerate(words) if word.startswith('--')}


def _start_words(path: str) -> str:
    """The start of a classifier, a SignNet, a brain or a UserNet (--init-from) in words: the run of a
    .../runs/<run>/classifier/reader-best.pt, of a .../runs/<run>/signnet-best.pt, .../runs/<run>/brain-last.pt or
    .../runs/<run>/usernet-best.pt."""
    parts = Path(path).parts
    if len(parts) >= 3 and parts[-2] in ('classifier', 'pretrain', 'reader'):
        return parts[-3]
    return parts[-2] if len(parts) >= 2 and parts[-1] in ('signnet-best.pt', 'brain-last.pt', USERNET_WEIGHTS) \
        else Path(path).name


def _region_words(region: str) -> str:
    """A region of the Mapillary crops in words: "Europe", "CH, DE, IT"."""
    from nnslr_tools.gps import CONTINENTS
    return ', '.join(CONTINENTS.get(part, part) for part in region.split(','))


ACTIONS = {
    # Every route still on the comma, over ssh with this machine's key (Host comma in ~/.ssh/config).
    'fetch': {'label': 'Download from the comma', 'group': 'fetch', 'command': ['helpers/fetch_comma_segments.sh']},
    'prepare': {'label': 'Prepare new drives', 'group': 'prepare', 'command': ['helpers/1_prepare.sh']},
    # The frames still to confirm of every route, proposed again (after choosing another proposer); the confirmed
    # frames stay (nnslr preannotate --keep-reviews).
    'prepare_again': {'label': 'Propose again', 'group': 'prepare', 'command': ['helpers/1_prepare.sh', '--again']},
    # The map's limit of every frame of the prepared drives, from their GPS and the map, read offline (Data › Speed
    # review); in the prepare group: it reads the reports a preparation writes.
    'road_types': {'label': 'Read the map', 'group': 'prepare', 'command': [ROAD_TYPES_SCRIPT]},
    # One round with the machine of the outside reviewers (nnslr_tools.review_box); a timer does the same every 5 minutes.
    'review_box': {'label': 'Update the review machine', 'group': 'review_box', 'command': ['helpers/review_box.sh']},
    # Where each Mapillary photo was taken, for the region of "Train the classifier": network only, a few minutes.
    'mapillary_locations': {'label': 'Locate the Mapillary photos', 'group': 'mapillary', 'command': [LOCATIONS_SCRIPT]},
    # The Mapillary photos cut into windows for "Train the cutter": the CPU for a while, the share read once; the
    # windows already cut are kept, windows.jsonl is replaced whole at the end.
    'mapillary_windows': {'label': 'Cut the Mapillary windows', 'group': 'mapillary', 'command': [WINDOWS_SCRIPT]},
    'dataset': {'label': 'Import sign reviews', 'group': 'train',
                'command': ['helpers/3_train.sh', '--dataset-only']},
    # The whole pipeline: import, split, the detector and the reader, evaluation, export, bundle.
    'train': {'label': 'Whole pipeline', 'group': 'train', 'command': ['helpers/3_train.sh']},
    'train_continue': {'label': 'Train from the stable model', 'group': 'train',
                       'command': ['helpers/3_train.sh', '--continue']},
    # The steps a stopped training still misses, on the same run ({'run': name}: helpers/3_train.sh --resume).
    'train_resume': {'label': 'Resume the training', 'group': 'train', 'command': ['helpers/3_train.sh']},
    # The classifier with "not a sign": the false boxes of the cutter in use mined, the classifier trained on them,
    # measured alone and in the chain, on a run of its own (its name is added by start(): <time>-classifier).
    'train_classifier': {'label': 'Train the classifier', 'group': 'train', 'command': [CLASSIFIER_SCRIPT]},
    # The cutter: a SignNet with the car's setup (896 x 288, stride 8), trained and measured on validation and test, on
    # a run of its own (start() adds <time>-cutter; {'families'}).
    'train_cutter': {'label': 'Train the cutter', 'group': 'train', 'command': [CUTTER_SCRIPT]},
    'train_decider': {'label': 'Train applicability', 'group': 'train', 'command': [DECIDER_SCRIPT]},
    'train_brain': {'label': 'Train the brain', 'group': 'train', 'command': [BRAIN_SCRIPT]},
    # UserNet, the road users' detector ({'stage', 'start'}), on a run of its own (start() adds <time>-users).
    'train_users': {'label': 'Train UserNet', 'group': 'train', 'command': [USERNET_SCRIPT]},
    'regression': {'label': 'Run the regression', 'group': 'train', 'command': [REGRESSION_SCRIPT]},
    # A drive, or every certified drive, replayed with a set of networks for Models › Drive (SIMULATE_SCRIPT; start()
    # adds --set and --route or --certified). In the train group: the GPU.
    'simulate': {'label': 'Replay a drive', 'group': 'train', 'command': [SIMULATE_SCRIPT]},
    # The road labeler (ROAD_LABELER_SCRIPT): trained on the GPU on the corrections of Speed review, and run along every
    # drive to label Speed review ({'labeler': run}): its base layer. In the train group: the GPU.
    'road_labeler_train': {'label': 'Train the road labeler', 'group': 'train', 'command': [ROAD_LABELER_SCRIPT, 'train']},
    'road_labeler_predict': {'label': 'Prepare Speed review', 'group': 'train',
                             'command': [ROAD_LABELER_SCRIPT, 'predict']},
    # Fit box trained again (BOX_FIT_SCRIPT): the CPU only, about an hour and a quarter with the refiner. A group of its
    # own: never two at once, but beside a training on the GPU.
    'box_fit_train': {'label': 'Train Fit box', 'group': 'box_fit', 'command': [BOX_FIT_SCRIPT]},
    # The cutter, classifier and decider in use exported as one package (start() names it); nothing is uploaded.
    'networks_prepare': {'label': 'Prepare the package', 'group': 'publish', 'command': [NETWORKS_SCRIPT]},
    # A package prepared here ({'package': name}) uploaded to Hugging Face, public: other people download it.
    'networks_publish': {'label': 'Publish on Hugging Face', 'group': 'publish', 'command': [NETWORKS_SCRIPT, '--upload']},
    # Publish (Publish tab): the SignNet in use ({'signnet': run}), the classifier and the road type in use put into the
    # comma's software: exported, published in cristianku/speed-limits-models, recorded on the car, the sunnypilot
    # speed-limits branch moved to them and psa-torque-sunny-testing built again (nnslr_tools.car_release). Then the
    # comma updates from its settings.
    'send_to_car': {'label': 'Publish to the car', 'group': 'publish', 'command': [CAR_SCRIPT]},
    # The same two networks copied straight onto the comma over ssh, checked first with the comma's own code, until
    # its next software update (nnslr_tools.comma_trial): no publication, no build. And the build's networks back.
    'try_on_comma': {'label': 'Try on the comma now', 'group': 'publish', 'command': [TRY_SCRIPT]},
    'comma_restore': {'label': "Back to the build's networks", 'group': 'publish', 'command': [TRY_SCRIPT, '--restore']},
    # A proposer of Prepare measured on the test frames (the Proposers view of Evaluate); in the train group: the GPU.
    'measure_proposer': {'label': 'Measure a proposer', 'group': 'train', 'command': ['helpers/measure_proposer.sh']},
    # Qwen3-VL as the third proposer of Prepare: its packages, its weights (17.5 GB, what is already there is kept after
    # its sha256 is checked) and a test on the GPU; in the train group: the GPU.
    'setup_vlm': {'label': 'Install Qwen3-VL', 'group': 'train', 'command': ['scripts/setup_vlm.sh']},
    # git pull + restart of this server (KillMode=process: the action itself survives the restart).
    # --no-link: the link with the token must not land in the log.
    'update': {'label': 'Update the site', 'group': 'update', 'command': ['helpers/update.sh', '--no-link']},
    # [ai-reviewer] - START
    # The frames of a route nobody confirmed, reviewed by an AI provider of Settings: a trial ({'save': 'no'}, the
    # suggestions go to data/ai_reviews/<route>/) or saved as AI reviews ({'save': 'yes'}: final, the next import
    # trains on them). Frames a person confirmed are never touched.
    'ai_review': {'label': 'Review a route with AI', 'group': 'ai', 'command': ['helpers/ai_review.sh']},
    # An AI provider measured on the frames people reviewed (the AI reviewers view of Evaluate); nothing is saved.
    'measure_ai_reviewer': {'label': 'Measure an AI reviewer', 'group': 'ai',
                            'command': ['helpers/measure_ai_reviewer.sh']},
    # [ai-reviewer] - END
}
# The same work started from a terminal: (group, label, fragment of the command line).
ACTIVITIES = (
    # Also the copy in openpilot.distill: both write the shared folder of the comma videos.
    ('fetch', 'download from the comma', 'fetch_comma_segments.sh'),
    ('prepare', 'preparation', 'helpers/1_prepare.sh'),
    ('prepare', 'frame extraction', 'nnslr extract-frames'),
    ('prepare', 'pre-annotation', 'nnslr preannotate'),
    ('prepare', 'map reading', ROAD_TYPES_SCRIPT),
    ('mapillary', 'Mapillary positions', 'nnslr mapillary-locations'),
    ('mapillary', 'Mapillary windows', 'nnslr mapillary-windows'),
    ('train', 'dataset and training', 'helpers/3_train.sh'),
    ('train', 'GPU training', 'run_v100_baseline.sh'),
    ('train', 'classifier training', 'run_classifier.sh'),
    ('train', 'cutter training', 'run_signnet.sh'),
    ('train', 'cutter training', 'nnslr train-signnet'),
    ('train', 'cutter training', 'run_detector_sizes.sh'),
    ('train', 'brain training', 'run_brain.sh'),
    ('train', 'brain training', 'nnslr train-brain'),
    ('train', 'UserNet training', 'run_usernet.sh'),
    ('train', 'UserNet training', 'nnslr train-usernet'),
    ('train', 'regression', 'run_regression.sh'),
    ('train', 'drive replay', SIMULATE_SCRIPT),
    ('train', 'drive replay', 'nnslr simulate'),
    ('train', 'road labeler', ROAD_LABELER_SCRIPT),
    ('train', 'road labeler', 'nnslr road-labeler-train'),
    ('train', 'road labeler', 'nnslr road-labeler-predict'),
    ('box_fit', 'Fit box training', BOX_FIT_SCRIPT),
    ('box_fit', 'Fit box training', 'nnslr train-box-fit'),
    ('box_fit', 'Fit box training', 'nnslr_tools.box_fit.refiner_train'),
    ('update', 'code update', 'helpers/update.sh'),
    ('publish', 'package of the networks', 'publish_networks.sh'),
    ('publish', 'publish to the car', 'send_to_car.sh'),
    ('publish', 'networks on the comma', 'try_on_comma.sh'),
    # [ai-reviewer] - START
    ('ai', 'AI review', 'nnslr ai-review'),
    ('ai', 'AI reviewer measure', 'nnslr measure-ai-reviewer'),
    # [ai-reviewer] - END
    # Shown only: the driving model of openpilot.distill shares the GPU (the training refuses a busy GPU).
    ('other', 'openpilot.distill training', 'supervised.train'),
)
# The groups of this checkout's own helpers (see Panel._ours).
OWN_GROUPS = ('prepare', 'train', 'update', 'ai', 'publish', 'mapillary', 'box_fit')
GROUP_NAMES = {'fetch': 'a download from the comma', 'prepare': 'a preparation',
               'train': 'an import or a training', 'update': 'a code update', 'ai': 'an AI review',
               'publish': 'a package of the networks', 'mapillary': 'a search for the Mapillary positions',
               'box_fit': 'a training of Fit box'}
# What runs, in the reason of an action that must wait ("A proposer measurement is running"): the job of the site by
# its action, the work from a terminal by its label (ACTIVITIES); GROUP_NAMES when neither is known.
RUNNING_NAMES = {'fetch': 'a download from the comma', 'prepare': 'a preparation', 'prepare_again': 'a preparation',
                 'measure_proposer': 'a proposer measurement', 'setup_vlm': 'the install of Qwen3-VL',
                 'dataset': 'an import of the reviews', 'train': 'a training', 'train_continue': 'a training',
                 'train_resume': 'a training', 'train_classifier': 'a classifier training',
                 'train_cutter': 'a cutter training', 'train_decider': 'a decider training',
                 'train_brain': 'a brain training', 'train_users': 'a UserNet training', 'regression': 'a regression run',
                 'simulate': 'a replay of a drive',
                 'road_labeler_train': 'a road labeler training', 'road_labeler_predict': 'a road labeler run',
                 'box_fit_train': 'a training of Fit box',
                 'update': 'a code update', 'review_box': 'a copy with the review machine',
                 'networks_prepare': 'the preparation of a package', 'networks_publish': 'a publication',
                 'send_to_car': 'a publication to the car', 'try_on_comma': 'a trial on the comma',
                 'comma_restore': "a return to the build's networks",
                 'ai_review': 'an AI review',
                 'measure_ai_reviewer': 'an AI reviewer measurement', 'road_types': 'a reading of the map',
                 'mapillary_locations': 'a search for the Mapillary positions',
                 'mapillary_windows': 'the cut of the Mapillary windows'}
ACTIVITY_NAMES = {'download from the comma': 'a download from the comma', 'preparation': 'a preparation',
                  'frame extraction': 'a preparation', 'pre-annotation': 'a preparation',
                  'dataset and training': 'an import or a training', 'GPU training': 'a training',
                  'classifier training': 'a classifier training', 'cutter training': 'a cutter training',
                  'brain training': 'a brain training', 'UserNet training': 'a UserNet training',
                  'regression': 'a regression run',
                  'drive replay': 'a replay of a drive',
                  'road labeler': 'a road labeler run',
                  'Fit box training': 'a training of Fit box',
                  'code update': 'a code update', 'AI review': 'an AI review',
                  'package of the networks': 'a package of the networks', 'publish to the car': 'a publication to the car',
                  'networks on the comma': 'a copy onto the comma',
                  'AI reviewer measure': 'an AI reviewer measurement', 'map reading': 'a reading of the map',
                  'Mapillary positions': 'a search for the Mapillary positions',
                  'Mapillary windows': 'the cut of the Mapillary windows',
                  'openpilot.distill training': 'openpilot.distill training'}
# Groups that never run together with a group, besides itself. The download and the preparation
# both write the links of data/raw/routes; the update rewrites the helpers under a running bash
# and restarts this server; a preparation replaces the report an AI review is working on. The search for the
# Mapillary positions writes a file of its own only (a classifier started meanwhile records the positions it read). The
# training of Fit box runs on the CPU beside the others, never under an update (it rewrites the models in the checkout).
CONFLICTS = {'fetch': ('prepare', 'update'), 'prepare': ('fetch', 'update', 'ai'), 'train': ('update',),
             'update': ('fetch', 'prepare', 'train', 'ai', 'publish', 'mapillary', 'review_box', 'box_fit'),
             'ai': ('prepare', 'update'), 'publish': ('update',), 'mapillary': ('update',), 'review_box': ('update',),
             'box_fit': ('update',)}
# The two V100s (issue #29): one GPU job per card, so two trainings can run at once. The groups whose actions may run
# side by side, pair by pair (_pair_apart), instead of one at a time.
PAIRED_GROUPS = ('train',)
# Never beside any other action of the train group, either way: the import and the whole pipeline rewrite the dataset
# and the split the others read, and the install of Qwen3-VL tests it on the GPU.
TRAIN_ALONE = ('dataset', 'train', 'train_continue', 'train_resume', 'setup_vlm')
# Never two of the same: each writes files of its own kind that a second one would write too (the brain's labels, the
# regression's results, the replays of the drives and the export of the versions in use, the road labeler's marks, the
# proposers' measures).
TRAIN_ONCE = ('train_brain', 'regression', 'simulate', 'road_labeler_train', 'road_labeler_predict', 'measure_proposer')
# Never together: the regression runs the brains along the drives the brain learns from, and the road labeler's run
# writes the labels of Speed review the brain, the regression and a new road labeler read.
TRAIN_APART = (frozenset(('train_brain', 'regression')), frozenset(('train_brain', 'road_labeler_predict')),
               frozenset(('regression', 'road_labeler_predict')),
               frozenset(('road_labeler_train', 'road_labeler_predict')))
# The work of the train group started from a terminal, by its label (ACTIVITIES), as the action it is for these rules.
TERMINAL_KINDS = {'cutter training': 'train_cutter', 'brain training': 'train_brain', 'UserNet training': 'train_users',
                  'classifier training': 'train_classifier', 'regression': 'regression',
                  'drive replay': 'simulate',
                  'road labeler': 'road_labeler_train',
                  'dataset and training': 'train', 'GPU training': 'train'}
# The actions that take a card of their own, the free one of lowest index: none starts while every card is busy.
EXCLUSIVE_GPU = ('train', 'train_continue', 'train_resume', 'train_classifier', 'train_cutter', 'train_decider',
                 'train_brain', 'train_users', 'regression', 'measure_proposer', 'setup_vlm', 'road_labeler_train',
                 'road_labeler_predict', 'box_fit_train')
# The actions that share a card: the one with the most free memory, refused only when it has less than they need. The
# replay of a drive takes about 1 GB of a V100 (NVDEC and the networks under onnxruntime), never the CPU.
SHARED_GPU = {'simulate': 2 * 1024}
# The trainings that take a free card when there is one, and otherwise share the one with the most room with other
# trainings of this list and the replays (issue #41): a V100 holds several, each about 1 GB today (the cutter at batch
# 16, the classifier on its crops), the memory each may take at most (MB) reserved on its card while it starts.
# Their pair rules stay (_pair_apart); the others keep a card of their own.
# Fit box (#65): its trees, MLPs and small refiner take well under 2 GB.
SHARED_TRAINING_MB = {'train_classifier': 4 * 1024, 'train_cutter': 4 * 1024, 'train_decider': 4 * 1024,
                      'box_fit_train': 2 * 1024}
# Of those, the ones whose dry run ({'dry': 'on'}, --dry-run) runs on the CPU and takes no card.
DRY_ON_CPU = ('road_labeler_train', 'road_labeler_predict', 'box_fit_train')
# The proposers of Prepare that run on the GPU (every one today): a preparation with one of them takes the card with the
# most free memory, shared with what else runs there.
GPU_PROPOSERS = ('vietnam', 'ours', 'qwen')
# The lock of openpilot.distill's GPU work (its helpers and its training take it): its holder is on a card too.
GPU_LOCK = '/run/lock/v100-gpu.lock'
# A card used by a program that is neither a job of the panel nor work it knows from a terminal (another container,
# llama.cpp): nvidia-smi shows its compute processes with the pids of the host only.
UNNAMED = 'a program the server cannot name'
# How long a search of a job's cards through its open files holds when it found none (the job not yet on the GPU), and
# when it found them.
SCAN_UNKNOWN_SECONDS = 10
SCAN_FOUND_SECONDS = 600
# The fetch reads the comma logs (trip length; reverse gear in Distill) with openpilot's LogReader; the
# .venv of this repository has it only with the [log-reader] extra and the schema checkouts.
FETCH_PYTHON_ENV = 'COMMA_FETCH_PYTHON'
DISTILL_PYTHON = 'openpilot.distill/.venv/bin/python'


class PanelError(Exception):
    def __init__(self, status: int, reason: str, message: str):
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


class Panel:
    def __init__(self, root: Path, *, repo: Path = REPO, actions_enabled: bool | None = None,
                 background_rows: bool = False):
        self.root = root
        self.repo = repo
        self.logs = root / 'logs' / 'panel'
        if actions_enabled is None:
            actions_enabled = os.environ.get(ACTIONS_ENV, '1') != '0'
        self.actions_enabled = actions_enabled
        # Reentrant: start() holds it while _running_groups() takes it again.
        self._lock = threading.RLock()
        self._processes = {}
        # The run names handed out by new_run_name, so two starts in the same minute never get the same one.
        self._reserved_runs: set[str] = set()
        # The pid each job wrote when it started (<job id>.pid), read once: by job id.
        self._pid_files: dict[str, int] = {}
        # The cards found through the open files of a job or of work from a terminal: by key, (when, cards).
        self._scanned: dict[tuple, tuple[float, list[int] | None]] = {}
        # The activities and GPU holders of the status being read in this thread, read once for it (_read_status).
        self._reading = threading.local()
        self._status_module = None
        # Progress of each job, by job id: (log size, progress), so a log is read again only when it grew.
        self._progress: dict[str, tuple[int, str, dict]] = {}
        self._dataset: tuple[tuple, dict | None] | None = None
        # (time of the last check, published?): None until the first answer.
        self._stable: tuple[float, bool | None] = (-float('inf'), None)
        self._stable_checking = False
        # One reading of the status at a time, and when it began: the polls of the pages that waited for it share the
        # next one (status()).
        self._status_lock = threading.Lock()
        self._status_last: tuple[float, dict] = (-float('inf'), {})
        # The drives' table read in a thread (_route_rows), the site's: (when its reading began, rows, the jobs running
        # then). Elsewhere (the tests, a terminal) every status reads it.
        self.background_rows = background_rows
        self._rows_lock = threading.Lock()
        self._rows_last: tuple[float, list[dict], frozenset] | None = None
        self._rows_reading = False
        # The training runs of <root>/runs, for the Evaluate tab (nnslr_tools.runs).
        self.runs_index = RunsIndex(root)
        # The same runs by role (cutter, classifier, decider) and the model in use, for the Models tab.
        self.models_index = ModelIndex(self.runs_index)
        # Whether the comma answers, for "Try on the comma now" (a TCP probe in a thread, at most once a minute).
        self.comma_probe = CommaProbe()
        # The thread that starts the next training of a pipeline (_watch_pipeline): again after a restart of this server.
        self._pipeline_watch: threading.Thread | None = None
        plan = _read_json(self.logs / PIPELINE_FILE)
        if self.actions_enabled and plan is not None and plan.get('state') == 'running':
            self._watch_pipeline()

    # -- helpers of the checkout -------------------------------------------------------------
    def available(self) -> bool:
        return all((self.repo / name).is_file() for name in
                   ('helpers/route_status.py', 'helpers/1_prepare.sh', 'helpers/3_train.sh'))

    def fetch_python(self) -> str | None:
        """The Python the fetch reads the comma logs with: COMMA_FETCH_PYTHON, else the .venv of
        openpilot.distill next to this checkout. '' when the fetch finds its own (NNSLR_OPENPILOT_ROOT
        set for the [log-reader] extra), None when there is none."""
        if os.environ.get(FETCH_PYTHON_ENV):
            return os.environ[FETCH_PYTHON_ENV]
        distill = self.repo.parent / DISTILL_PYTHON
        if os.access(distill, os.X_OK):
            return str(distill)
        if os.environ.get('NNSLR_OPENPILOT_ROOT'):
            return ''
        return None

    def _route_status(self):
        if self._status_module is None:
            spec = importlib.util.spec_from_file_location('nnslr_panel_route_status',
                                                          self.repo / 'helpers/route_status.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._status_module = module
        return self._status_module

    # -- what is running -----------------------------------------------------------------------
    def activities(self) -> list[dict]:
        """Helpers and nnslr steps running on this machine, also those started from a terminal: one for each label and
        process session, 'pid' the first of its processes. 'job': the job of the panel whose session it is (the job's
        wrapper leads it), else None. 'gpus': the cards it holds, as for a job (jobs()): for work of the train group or
        the openpilot.distill training started from a terminal the cards its processes have open, None when none is
        found (it holds every card then); [] for the rest."""
        found: dict[tuple, dict] = {}
        matched: dict[tuple, list[int]] = {}
        proc = Path('/proc')
        if not proc.is_dir():
            return []
        own = os.getpid()
        for entry in proc.iterdir():
            if not entry.name.isdigit() or int(entry.name) == own:
                continue
            try:
                command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
            except OSError:
                continue
            session = None
            for group, label, fragment in ACTIVITIES:
                if fragment in command and self._ours(entry, group):
                    if session is None:
                        session = _session_of(int(entry.name))
                    if session is None:
                        # Gone between its command line and its stat (a DataLoader worker of a job, re-made each
                        # epoch): no session to tie it to a job, and no card files, so it would hold every card.
                        break
                    key = (label, session)
                    found.setdefault(key, {'group': group, 'label': label, 'pid': int(entry.name)})
                    matched.setdefault(key, []).append(int(entry.name))
        if not found:
            return []
        jobs = self._jobs_now()
        by_session = {info['pid']: job_id for job_id, info in jobs.items()}
        for key, item in found.items():
            job = by_session.get(key[1]) if key[1] is not None else None
            item['job'] = job
            if job is not None:
                item['gpus'] = jobs[job]['gpus']
            elif item['group'] in ('train', 'other'):
                item['gpus'] = self._scan_cards(('session',) + key, matched[key])
            else:
                item['gpus'] = []
        return list(found.values())

    def _jobs_now(self) -> dict[str, dict]:
        """The jobs of the panel running now, by id: {'action', 'pid' (its wrapper, the leader of its session), 'gpus'
        (_job_gpus)}. Also those started before a restart of this server (their wrapper still alive, _group); when a pid
        was used again, by the newest. Read once for the status being read in this thread."""
        cached = getattr(self._reading, 'jobs', None)
        if cached is not None:
            return cached
        alive = {}
        with self._lock:
            for job_id, process in self._processes.items():
                if process.poll() is None:
                    alive[job_id] = process.pid
        if self.logs.is_dir():
            for path in self.logs.glob('*.pid'):
                job_id = path.stem
                if job_id in alive or not JOB_ID.fullmatch(job_id) or job_id.split('-', 2)[2] not in ACTIONS:
                    continue
                pid = self._pid_files.get(job_id)
                if pid is None:
                    try:
                        pid = int(path.read_text().strip())
                    except (OSError, ValueError):
                        continue
                    self._pid_files[job_id] = pid
                if os.path.exists(f'/proc/{pid}') and self._group(job_id) == pid:
                    alive[job_id] = pid
        newest = {pid: job_id for job_id, pid in sorted(alive.items())}
        jobs = {job_id: {'action': job_id.split('-', 2)[2], 'pid': pid,
                         'gpus': self._job_gpus(job_id, running=True, pid=pid)}
                for pid, job_id in newest.items()}
        if getattr(self._reading, 'active', False):
            self._reading.jobs = jobs
        return jobs

    def _job_gpus(self, job_id: str, *, running: bool, pid: int | None = None) -> list[int] | None:
        """The cards of a job, by nvidia-smi's index: those start() gave it (<job id>.json, 'gpus'), and while it runs
        those a brain borrowed for its readings of the signs ('borrowed', borrow_cards); for a job started before that,
        [] when it uses no GPU, else the cards its processes have open while it runs; None when they are not known (it
        holds every card)."""
        record = _read_json(self.logs / (job_id + '.json')) or {}
        if 'gpus' in record:
            gpus = record['gpus']
            if gpus is None or not (isinstance(gpus, list) and all(isinstance(i, int) for i in gpus)):
                return None
            borrowed = record.get('borrowed') if running else None
            if isinstance(borrowed, list) and all(isinstance(i, int) for i in borrowed):
                return list(dict.fromkeys([*gpus, *borrowed]))
            return gpus
        if not self._uses_gpu(job_id):
            return []
        if not running or pid is None:
            return None
        return self._scan_cards(('job', job_id), [pid])

    def _uses_gpu(self, job_id: str) -> bool:
        """Whether a job uses the GPU, from its action, the first line of its log (a dry run on the CPU) and its
        proposer."""
        action = job_id.split('-', 2)[2]
        path = self.logs / (job_id + '.log')
        if action in DRY_ON_CPU:
            try:
                with path.open('rb') as log:
                    first = log.readline(4096).decode('utf-8', 'replace')
            except OSError:
                first = ''
            return '--dry-run' not in first.split()
        if action in PREPARING:
            # None: the helpers' default proposer, on the GPU.
            proposer = _proposer_of(path)
            return proposer is None or proposer in GPU_PROPOSERS
        return action in EXCLUSIVE_GPU or action in SHARED_GPU

    def _scan_cards(self, key: tuple, pids: list[int]) -> list[int] | None:
        """_tree_cards of the processes, kept a while by key: SCAN_FOUND_SECONDS when some cards are found,
        SCAN_UNKNOWN_SECONDS when none or every one is (the job may not be on its card yet, only checking it)."""
        now = time.monotonic()
        cached = self._scanned.get(key)
        if cached is not None and now - cached[0] < cached[2]:
            return cached[1]
        cards = _tree_cards(pids)
        some = cards is not None and not set(_card_indexes().values()) <= set(cards)
        if len(self._scanned) > 256:
            self._scanned.clear()
        self._scanned[key] = (now, cards, SCAN_FOUND_SECONDS if some else SCAN_UNKNOWN_SECONDS)
        return cards

    def _gpu_holders(self, activities: list[dict]) -> list[dict]:
        """What holds the GPUs now: [{'job': id or None, 'name': "a cutter training", 'label': "A cutter training",
        'gpus': the cards (None: every card), 'action': the action, or the kind of work from a terminal}]. The jobs of
        the panel that use a GPU, the work of the train group and the openpilot.distill training from a terminal, and
        the holder of openpilot.distill's GPU lock when it is none of those. Read once for the status being read in
        this thread."""
        cached = getattr(self._reading, 'holders', None)
        if cached is not None and cached[0] is activities:
            return cached[1]
        jobs = self._jobs_now()
        holders = []
        for job_id, info in sorted(jobs.items()):
            if info['gpus'] != []:
                holders.append({'job': job_id, 'name': RUNNING_NAMES.get(info['action'], 'a job of the panel'),
                                'gpus': info['gpus'], 'action': info['action']})
        sessions = {info['pid'] for info in jobs.values()}
        for item in activities:
            if item.get('job') is not None or item['group'] not in ('train', 'other'):
                continue
            sessions.add(_session_of(item['pid']))
            gpus = item.get('gpus')
            if gpus == []:
                continue
            holders.append({'job': None, 'name': ACTIVITY_NAMES.get(item['label'], GROUP_NAMES.get(item['group'], UNNAMED)),
                            'gpus': gpus, 'action': TERMINAL_KINDS.get(item['label'], 'train')
                            if item['group'] == 'train' else None})
        for pid in _gpu_lock_holders():
            session = _session_of(pid)
            if session is None or session in sessions:
                continue
            sessions.add(session)
            holders.append({'job': None, 'name': ACTIVITY_NAMES['openpilot.distill training'],
                            'gpus': self._scan_cards(('lock', session, pid), [pid]), 'action': None})
        for holder in holders:
            name = holder['name']
            holder['label'] = name if name.startswith('openpilot') else name[0].upper() + name[1:]
        if getattr(self._reading, 'activities', None) is activities:
            self._reading.holders = (activities, holders)
        return holders

    def _gpu_states(self, system: dict, activities: list[dict]) -> tuple[list[dict], list[dict]]:
        """The cards of a system() reading with their holders and whether each is free (_card_states), and the
        holders."""
        holders = self._gpu_holders(activities)
        return _card_states(_cards_of(system), holders), holders

    def _train_clash(self, action: str, activities: list[dict]) -> str:
        """Why an action of the train group cannot run beside the ones running now (_pair_apart), in words: "A cutter
        training is running"; '' when it can."""
        jobs = self._jobs_now()
        for job_id, info in sorted(jobs.items()):
            if ACTIONS[info['action']]['group'] == 'train' and _pair_apart(action, info['action']):
                name = RUNNING_NAMES.get(info['action'], GROUP_NAMES['train'])
                return name[0].upper() + name[1:] + ' is running'
        for item in activities:
            if item['group'] != 'train' or item.get('job') is not None:
                continue
            if _pair_apart(action, TERMINAL_KINDS.get(item['label'], 'train')):
                name = ACTIVITY_NAMES.get(item['label'], GROUP_NAMES['train'])
                return name[0].upper() + name[1:] + ' is running'
        return ''

    def _ours(self, process: Path, group: str) -> bool:
        """The helpers of this checkout run from it (the panel starts them there, the guide says so for a
        terminal): the same helper of another checkout is not ours. A download and other programs (the
        copy in openpilot.distill, its training) count wherever they run."""
        if group not in OWN_GROUPS:
            return True
        try:
            cwd = Path(os.readlink(process / 'cwd'))
        except OSError:
            return True
        return cwd.is_relative_to(self.repo.resolve())

    def _running_groups(self, activities: list[dict]) -> set[str]:
        groups = {item['group'] for item in activities}
        with self._lock:
            for job_id, process in self._processes.items():
                if process.poll() is None:
                    groups.add(ACTIONS[job_id.split('-', 2)[2]]['group'])
        return groups

    def _busy_reason(self, group: str, activities: list[dict]) -> str:
        """Why an action waits for `group`, in words: "A proposer measurement is running" (the running job of the site
        in that group, else the work from a terminal, else the group)."""
        name = None
        with self._lock:
            for job_id, process in self._processes.items():
                action = job_id.split('-', 2)[2]
                if process.poll() is None and ACTIONS[action]['group'] == group:
                    name = RUNNING_NAMES.get(action)
                    break
        if name is None:
            name = next((ACTIVITY_NAMES[item['label']] for item in activities
                         if item['group'] == group and item['label'] in ACTIVITY_NAMES), None)
        name = name or GROUP_NAMES[group]
        return name[0].upper() + name[1:] + ' is running'

    def running(self) -> dict:
        """What runs now, for the dots of the busy tabs on the pages outside the panel (GET /api/panel/system): the
        running jobs of the site (id, action, label, state, gpus) and the work started from a terminal."""
        activities = self.activities()
        jobs = [{key: job[key] for key in ('id', 'action', 'label', 'state', 'gpus')}
                for job in self.jobs(activities) if job['state'] == 'running']
        return {'jobs': jobs, 'activities': activities}

    def jobs(self, activities: list[dict] | None = None) -> list[dict]:
        """The last actions started from the panel, newest first."""
        if activities is None:
            activities = self.activities()
        if not self.logs.is_dir():
            return []
        busy = {item['group'] for item in activities}
        names = sorted((p for p in self.logs.glob('*.log') if JOB_ID.fullmatch(p.stem)), reverse=True)
        now = self._jobs_now()
        result = []
        for path in names[:HISTORY]:
            job_id = path.stem
            action = job_id.split('-', 2)[2]
            if action not in ACTIONS:
                continue
            with self._lock:
                process = self._processes.get(job_id)
            code = _exit_code(path)
            if process is not None and process.poll() is None:
                state = 'running'
            elif code is not None:
                state = 'ok' if code == 0 else 'failed'
            elif process is None and (self._group(job_id) is not None or (
                    ACTIONS[action]['group'] in busy and not (self.logs / (job_id + '.pid')).exists())):
                # Started before a restart of this server and still going: its wrapper is still alive (the pid the
                # panel wrote when it started it), or, for a job from before those pids, its group is busy. A job
                # with a pid that is gone was interrupted, whatever else of its group runs now: a measure that died
                # at 416 of 604 frames showed as running for hours while a cutter trained.
                state = 'running'
            else:
                state = 'interrupted'
            started = datetime.strptime(job_id[:15], '%Y%m%d-%H%M%S')
            info = path.stat()
            # Who proposes (a preparation) or is measured: "Measure Qwen3-VL" rather than "Measure a proposer".
            proposer = _proposer_of(path) if action in PROPOSER_ACTIONS else None
            label = ACTIONS[action]['label']
            if action == 'measure_proposer' and proposer is not None:
                from nnslr_tools.preannotate import PROPOSER_NAMES
                label = 'Measure ' + PROPOSER_NAMES.get(proposer, proposer)
            result.append({'id': job_id, 'action': action, 'label': label,
                           'run': _run_of(path, action) if ACTIONS[action]['group'] == 'train'
                           or action in SIGNNET_ACTIONS else None,
                           'proposer': proposer,
                           'networks': _networks_of(self.root, path) if proposer == 'ours' else None,
                           'settings': _settings_of(path, action),
                           'started': started.strftime('%Y-%m-%d %H:%M:%S'),
                           # The same moment with the server's offset, for the elapsed time in the page.
                           'started_at': started.astimezone().isoformat(timespec='seconds'),
                           # When it ended: the last write of its log (the wrapper's exit code, or where it stopped).
                           'ended_at': None if state == 'running'
                           else datetime.fromtimestamp(info.st_mtime).astimezone().isoformat(timespec='seconds'),
                           'state': state, 'code': code,
                           'stoppable': state == 'running' and self._group(job_id) is not None,
                           'size': info.st_size,
                           # The cards it holds (nvidia-smi's index): [] without a GPU, None when not known (every
                           # card).
                           'gpus': now[job_id]['gpus'] if job_id in now
                           else self._job_gpus(job_id, running=state == 'running', pid=self._group(job_id))})
        return result

    # -- the machine ---------------------------------------------------------------------------
    def comma(self) -> dict:
        """The comma pill of the top bar: reachable or not (nnslr_tools.comma_link), and whether a download from it
        runs now (a job of the panel or helpers/fetch_comma_segments.sh from a terminal), and what a download must
        wait for (a preparation running: Download new drives is off until it ends)."""
        from nnslr_tools import comma_link
        return {**comma_link.state(self.root), **self._download_state()}

    def test_comma(self) -> dict:
        """The comma checked now (its Test button)."""
        from nnslr_tools import comma_link
        return {**comma_link.check_now(self.root), **self._download_state()}

    def _download_state(self) -> dict:
        """'downloading', 'waits_for': what keeps a download from starting ("a preparation"), None when nothing,
        and 'download_progress': {percent, detail} of the download of the panel running now, else None."""
        activities = self.activities()
        running = self._running_groups(activities)
        other = _conflicting_group('fetch', running) if 'fetch' not in running else None
        found = None
        if 'fetch' in running:
            job = next((job for job in self.jobs(activities) if job['action'] == 'fetch' and job['state'] == 'running'),
                       None)
            if job is not None:
                result = self.job_progress(job, None)
                found = {'percent': result['percent'], 'detail': result['detail']}
        return {'downloading': 'fetch' in running, 'waits_for': GROUP_NAMES.get(other, other) if other else None,
                'download_progress': found}

    def set_comma_address(self, body) -> dict:
        """The address of the comma from its panel ({'address': '192.168.1.20'}; '' for the ssh host "comma")."""
        from nnslr_tools import comma_link
        address = body.get('address') if isinstance(body, dict) else None
        if not isinstance(address, str):
            raise PanelError(400, 'bad_address', 'The address is a string.')
        try:
            comma_link.set_address(self.root, address)
        except comma_link.AddressError as exc:
            raise PanelError(400, 'bad_address', str(exc)) from None
        return self.comma()

    def system(self) -> dict:
        """The machine, for the Stato page (as distill-web's Status): each GPU, CPU per core, RAM, disk.

        'gpus': each card on its own (_gpu_cards), by nvidia-smi's index. 'gpu': what the pages read before there were
        two cards, kept for a page loaded before an update: the first card's numbers and the processes of all of them.
        Each card also says whether it is 'free' and its 'holders': [{'job': the job of the panel or None, 'label': "A
        cutter training", "openpilot.distill training", "A program the server cannot name"}] (_gpu_holders)."""
        info = {'gpu': None, 'gpus': [], 'disk': None, 'cpus': os.cpu_count(), 'load': None, 'loads': None,
                'cpu': _cpu_usage(), 'memory': _memory()}
        try:
            loads = os.getloadavg()
            info['load'] = round(loads[0], 1)
            info['loads'] = [round(value, 2) for value in loads]
        except OSError:
            pass
        try:
            usage = shutil.disk_usage(self.root)
            info['disk'] = {'free_gb': round(usage.free / 1e9, 1), 'total_gb': round(usage.total / 1e9, 1)}
        except OSError:
            pass
        cards = _gpu_cards()
        info['gpus'] = cards
        if cards:
            activities = getattr(self._reading, 'activities', None)
            states = _card_states(cards, self._gpu_holders(self.activities() if activities is None else activities))
            info['gpus'] = [{**card, 'free': state['free'],
                             'holders': [{'job': holder['job'], 'label': holder['label']} for holder in state['holders']]}
                            for card, state in zip(cards, states)]
            every = [process for card in cards for process in card['process_list']]
            # 'processes' stays a count: the training actions refuse a busy GPU with it.
            info['gpu'] = {**{key: cards[0][key] for key in ('name', 'utilization', 'memory_used_mb', 'memory_total_mb',
                                                             'temperature', 'power_w')},
                           'processes': len(every), 'process_list': every}
        return info

    def ai_live(self, jobs: list[dict] | None = None) -> dict | None:
        """The Review with AI at work (one at a time, group ai), for the route list and the drive's page, from its
        log: the drive, whether it saves, how many frames of how many it has done, the frame it did last and about
        how long is left (from its pace so far). None when none runs."""
        if jobs is None:
            jobs = self.jobs()
        job = next((item for item in jobs if item['action'] == 'ai_review' and item['state'] == 'running'), None)
        if job is None:
            return None
        path = self.logs / (job['id'] + '.log')
        try:
            with path.open('rb') as handle:
                head = handle.read(4096).decode('utf-8', 'replace')
                handle.seek(max(0, path.stat().st_size - 16384))
                tail = handle.read().decode('utf-8', 'replace')
        except OSError:
            return None
        found = AI_LIVE_ROUTE.search(head)
        if found is None:
            return None
        steps = AI_LIVE_STEP.findall(tail)
        done, total, frame = (int(steps[-1][0]), int(steps[-1][1]), steps[-1][2]) if steps else (0, None, None)
        started = datetime.strptime(job['id'][:15], '%Y%m%d-%H%M%S').timestamp()
        left = None
        if done and total:
            left = round((time.time() - started) / done * (total - done))
        return {'job': job['id'], 'route': found.group(1), 'saving': found.group(2) == 'saving', 'done': done,
                'total': total, 'frame_key': frame, 'seconds_left': left}

    # -- everything for the page ---------------------------------------------------------------
    def status(self) -> dict:
        """Everything for the page, one reading at a time: a call waiting for another takes the next reading begun after
        it was made, so the polls of the pages share it. On a busy machine a reading can take longer than the pages
        wait between polls; read side by side they would pile up, each slowing the others."""
        asked = time.monotonic()
        with self._status_lock:
            began, payload = self._status_last
            if began > asked:
                return payload
            began = time.monotonic()
            payload = self._read_status()
            self._status_last = (began, payload)
            return payload

    def _read_status(self) -> dict:
        # The running jobs, the activities and the holders of the GPUs read once for this status (_jobs_now,
        # _gpu_holders), however many parts of it ask.
        self._reading.active = True
        try:
            return self._read_status_once()
        finally:
            self._reading.__dict__.clear()

    def _read_status_once(self) -> dict:
        activities = self.activities()
        self._reading.activities = activities
        system = self.system()
        payload = {'available': self.available(), 'actions_enabled': self.actions_enabled and self.available(),
                   'routes': [], 'next_step': '', 'next_action': None, 'reviewed_routes': 0, 'min_routes': 3,
                   'activities': activities, 'jobs': self.jobs(activities), 'system': system, 'actions': {},
                   'preparation': None}
        payload['ai_live'] = self.ai_live(payload['jobs'])
        if payload['available']:
            module = self._route_status()
            rows = self._route_rows(module, frozenset(job['id'] for job in payload['jobs'] if job['state'] == 'running'))
            payload['routes'] = rows
            payload['preparation'] = self.preparation(rows, activities)
            payload['next_step'] = module.next_step(rows)
            payload['min_routes'] = module.MIN_REVIEWED_ROUTES
            payload['reviewed_routes'] = sum(1 for row in rows if row['confirmed'])
            payload['next_action'] = _next_action(payload['next_step'])
            # The prepared drives without road types: the Overview asks for them once nothing waits to be prepared.
            from nnslr_tools.road_types import drives_without
            payload['road_types_missing'] = drives_without(self.root)
        # Who can propose the signs on this server (the choice of the Prepare tab), and why not.
        from nnslr_tools.preannotate import ours_missing, vlm_missing
        # qwen: why it is not installed; qwen_gpu: why it cannot load on the GPU now (start() refuses it then); ours:
        # why Our networks are not whole yet; chosen: the choice kept on the Prepare tab (None before one: then the
        # best measured, proposers()); networks: what Our networks run, the parts in use.
        # The card with the most free memory: the one a preparation would take.
        payload['proposers'] = {'qwen': vlm_missing(self.root),
                                'qwen_gpu': _qwen_gpu_reason(_best_card(_cards_of(system))) or None,
                                'ours': ours_missing(self.root), 'chosen': self._kept_proposer(),
                                'networks': _active_parts(self.root)}
        # The packages of our networks prepared here, newest first, and where they are published (Models tab).
        from nnslr_tools import hf_account
        from nnslr_tools.networks_package import packages
        payload['networks'] = {'packages': packages(self.root)[:NETWORK_PACKAGES_SHOWN], 'repo': NETWORKS_REPO,
                               'account': hf_account.public_state(self.root)}
        # [ai-reviewer] - START
        # The AI reviewer of Settings (a read of its file, no network): whether a route can be reviewed with AI.
        payload['ai_reviewer'], ai_providers = self._ai_settings()
        payload['actions'] = self._action_states(payload, activities, ai_providers=ai_providers)
        # [ai-reviewer] - END
        # The choices of "Train the cutter", for its buttons.
        # Its Mapillary switch: whether the windows are cut (helpers/mapillary_windows.sh), and the regions their photos
        # can be chosen by (the positions of helpers/mapillary_locations.sh, the same as the classifier's).
        from nnslr_tools.mtsd_windows import LICENCE, windows_regions_state, windows_state
        payload['cutter_options'] = {'families': list(CUTTER_FAMILIES), 'family': CUTTER_FAMILY, 'band': CUTTER_BAND,
                                     'mapillary': {**windows_state(self.root), 'licence': LICENCE},
                                     'mapillary_regions': windows_regions_state(self.root, self.repo)}
        # The Mapillary switch of "Train the classifier": whether its crops are prepared (helpers/mapillary.sh), and
        # the regions its photos can be chosen by (helpers/mapillary_locations.sh).
        from nnslr_tools.mapillary import crops_state, regions_state
        # And the split the next one learns on (splits/latest.json): a classifier to start from that learnt on another
        # one may have seen some of its test signs ("Start from"; the same for a SignNet or an applicability variant).
        payload['classifier_options'] = {'mapillary': crops_state(self.root),
                                         'mapillary_regions': regions_state(self.root, self.repo),
                                         'split_sha256': current_split(self.root)}
        payload['dataset'] = self.dataset_summary()
        # The Car row of the Overview: what the car should run and what its software has, without the Models index.
        payload['car'] = self.car_summary()
        payload['training_starts'] = self.training_starts()
        payload['pipeline'] = self.pipeline_view()
        payload['next_step_text'], payload['all_ready'] = _next_step_text(payload)
        for job in payload['jobs']:
            job['progress'] = self.job_progress(job, payload['preparation'])
        self._terminal_training(payload)
        return payload

    def _route_rows(self, module, running: frozenset) -> list[dict]:
        """The drives' table (route_rows). With background_rows only the first after a start is read here: a status
        then shows the last one read, and begins a new reading in a thread when it is older than ROUTE_ROWS_FRESH_S or
        the running jobs changed (a preparation adds frames, an import confirms them). The GPU, the jobs and the
        buttons of a status never wait for it."""
        if not self.background_rows:
            return module.route_rows(self.root, module.videos_dir())
        with self._rows_lock:
            last = self._rows_last
            again = last is not None and not self._rows_reading and (
                time.monotonic() - last[0] > ROUTE_ROWS_FRESH_S or last[2] != running)
            if again:
                self._rows_reading = True
        if last is None:
            return self._read_route_rows(module, running)
        if again:
            threading.Thread(target=self._read_route_rows, args=(module, running), name='route-rows',
                             daemon=True).start()
        return last[1]

    def _read_route_rows(self, module, running: frozenset) -> list[dict]:
        began = time.monotonic()
        try:
            rows = module.route_rows(self.root, module.videos_dir())
            with self._rows_lock:
                self._rows_last = (began, rows, running)
            return rows
        finally:
            with self._rows_lock:
                self._rows_reading = False

    def _terminal_training(self, payload: dict) -> None:
        """A training started from a terminal (helpers/3_train.sh) is no job of the panel: its progress comes
        from the log the helper writes itself, the newest data/logs/train-<run>.log; the classifier's
        (scripts/run_classifier.sh) from the newest data/runs/<run>/classifier.log; the cutter's
        (scripts/run_signnet.sh) from the newest data/runs/<run>/signnet.log, or the train.log of a SignNet trained by
        hand (a run with signnet-best.pt); the old detector's (scripts/run_detector_sizes.sh) from the newest
        data/runs/<run>/detector-sizes/<size>.log, which holds the training of a size alone. The cutter's logs are read
        as its training step (its epochs: the checks before it are over).

        Each activity on its own, also while a training of the panel runs beside it (one on each card): the activities
        of the panel's jobs have their job's progress, and the logs of the runs of the running jobs are not theirs."""
        trainings = [item for item in payload['activities'] if item['group'] == 'train' and item.get('job') is None]
        if not trainings:
            return
        # The runs of the panel's trainings at work: their logs (runs/<run>/..., logs/train-<run>.log) are theirs.
        taken = {job['run'] for job in payload['jobs'] if job['state'] == 'running' and job.get('run')}
        # The road labeler and the replay of a drive started from a terminal have no log of
        # their own to read: no progress, and not the whole pipeline's either.
        terminal = {'classifier training': 'train_classifier', 'cutter training': 'train_cutter',
                    'brain training': 'train_brain', 'UserNet training': 'train_users', 'road labeler': 'road_labeler',
                    'drive replay': 'simulate'}
        patterns = (('train', ('logs/train-*.log',)), ('train_classifier', ('runs/*/classifier.log',)),
                    ('train_cutter', ('runs/*/signnet.log', 'runs/*/train.log', 'runs/*/detector-sizes/*.log')),
                    ('train_brain', ('runs/*/brain.log',)), ('train_users', ('runs/*/' + USERNET_LOG,)))
        for action, globs in patterns:
            items = [item for item in trainings if terminal.get(item['label'], 'train') == action]
            found = [path for pattern in globs for path in self.root.glob(pattern) if items]
            # A train.log of a run is a SignNet's only next to its weights.
            found = [path for path in found if path.name != 'train.log' or (path.parent / 'signnet-best.pt').is_file()]
            found = [path for path in found if not _log_of_runs(self.root, path, taken)]
            logs = sorted(found, key=lambda path: path.stat().st_mtime)
            if not logs:
                continue
            size = logs[-1].stat().st_size
            with logs[-1].open('rb') as log:
                log.seek(max(0, size - PROGRESS_BYTES))
                text = log.read().decode('utf-8', 'replace')
            if action == 'train_cutter' and logs[-1].parent.name == 'detector-sizes':
                text = '== 3/4 Train the detector at ' + logs[-1].stem + ' ==\n' + text
            elif action == 'train_cutter':
                text = '== 3/3 Train SignNet ==\n' + text
            elif action == 'train_users':
                # The tail of usernet.log may have lost the script's step lines: its epochs are step 3's.
                text = '== 3/3 Train UserNet ==\n' + text
            result = progress(action, 'running', text)
            for item in items:
                item['progress'] = result

    def job_progress(self, job: dict, preparation: dict | None) -> dict:
        """Steps, percent and last line of a job, from its log (nnslr_tools.job_progress)."""
        key = (job['size'], job['state'])
        cached = self._progress.get(job['id'])
        # A preparation's percent changes without its log growing: never cached while it runs.
        if cached is not None and cached[:2] == key and not (job['action'] in PREPARING and job['state'] == 'running'):
            result = cached[2]
        else:
            result = self._read_progress(job, preparation)
            self._progress[job['id']] = (key[0], key[1], result)
        # The progress inside the current step changes without the log growing: never from the cache.
        if job['state'] == 'running':
            result = self._with_step_progress(result, job)
        return result

    def _read_progress(self, job: dict, preparation: dict | None) -> dict:
        try:
            with (self.logs / (job['id'] + '.log')).open('rb') as log:
                log.seek(max(0, job['size'] - PROGRESS_BYTES))
                text = log.read().decode('utf-8', 'replace')
        except OSError:
            text = ''
        return progress(job['action'], job['state'], text, preparation if job['action'] in PREPARING else None)

    def _with_step_progress(self, result: dict, job: dict) -> dict:
        """The progress inside the current step: the last report of a command (written by nnslr_tools.progress) if it
        came after the job started, or what the GPU setup is doing. Each job has its file, logs/progress-<job id>.json
        (NNSLR_PROGRESS_FILE, so two jobs at once never read each other's); a job started before that (its record
        without 'own_progress') the shared logs/progress.json."""
        action = job['action']
        own = (_read_json(self.logs / (job['id'] + '.json')) or {}).get('own_progress')
        record = _read_json(self.root / 'logs' / (f'progress-{job["id"]}.json' if own else 'progress.json'))
        try:
            started = datetime.fromisoformat(job['started_at']).timestamp()
        except ValueError:
            started = 0.0
        if record is not None and not (isinstance(record.get('at'), (int, float)) and record['at'] >= started):
            record = None
        setup = ''
        download = None
        try:
            with (self.root / 'logs' / 'setup-gpu.log').open('rb') as log:
                log.seek(max(0, log.seek(0, 2) - 8192))
                tail = log.read().decode('utf-8', 'replace')
            setup = last_line(tail)
            download = self._download(tail)
        except OSError:
            pass
        return with_step_progress(result, action, record, setup, download)

    def _download(self, setup_log: str) -> dict | None:
        """The file pip is downloading for the GPU environment: its name and size from the log
        ("Downloading torch-....whl (906.4 MB)"), and how much is already written, from the size of the
        temporary file pip has open (seen through /proc, whatever /tmp the job has)."""
        found = DOWNLOADING.findall(setup_log)
        if not found:
            return None
        name, size, unit = found[-1]
        total = float(size) * {'kB': 1e3, 'MB': 1e6, 'GB': 1e9}[unit]
        written = 0
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                command = (entry / 'cmdline').read_bytes()
                if b'pip' not in command or b'install' not in command:
                    continue
                for handle in (entry / 'fd').iterdir():
                    target = os.readlink(handle)
                    if '/tmp' in target and not target.endswith(('.py', '.pyc', '.so')):
                        info = os.stat(handle)
                        if stat.S_ISREG(info.st_mode) and info.st_size <= total * 1.05:
                            written = max(written, info.st_size)
            except OSError:
                continue
        return {'name': name.split('-')[0], 'written': written, 'total': total}

    def _action_states(self, payload: dict, activities: list[dict], *, ai_providers: int = 0) -> dict:
        running = self._running_groups(activities)
        rows = payload['routes']
        # Each card with what holds it: an action that takes a card waits while none is free (gpu_reason says what
        # holds them); the install of Qwen3-VL reads the card with the most free memory.
        cards, holders = self._gpu_states(payload['system'], activities)
        gpu_reason = _gpu_busy(cards, holders)
        best = _best_card(cards)
        to_prepare = sum(1 for row in rows if row['state'] in ('collegare', 'estrarre', 'preannotare'))
        confirmed = sum(row['confirmed'] for row in rows)
        states = {}
        for name, action in ACTIONS.items():
            reason = ''
            conflict = _conflicting_group(action['group'], running)
            if not payload['actions_enabled']:
                reason = 'the panel is read-only'
            elif conflict is not None:
                reason = self._busy_reason(conflict, activities)
            elif action['group'] in PAIRED_GROUPS and (clash := self._train_clash(name, activities)):
                reason = clash
            elif name == 'update' and not (self.repo / 'helpers/update.sh').is_file():
                reason = 'helpers/update.sh is missing in this installation'
            elif name == 'update':
                pass
            elif name == 'fetch' and not (self.repo / 'helpers/fetch_comma_segments.sh').is_file():
                reason = 'helpers/fetch_comma_segments.sh is missing in this installation'
            elif name == 'fetch' and self.fetch_python() is None:
                reason = 'the reader of the comma logs is missing (see docs/GUIDE.md)'
            elif name == 'fetch':
                pass
            elif name == 'prepare' and not to_prepare:
                reason = 'no new drive to prepare'
            elif name == 'road_types' and not (self.repo / ROAD_TYPES_SCRIPT).is_file():
                reason = ROAD_TYPES_SCRIPT + ' is missing in this installation'
            elif name == 'road_types' and not any(row.get('run_id') for row in rows):
                reason = 'no prepared drive'
            elif name == 'road_types':
                # It reads the GPS and the map: it needs no confirmed frame.
                pass
            elif name == 'mapillary_locations':
                # Network only (no GPU, no confirmed frame): the script and the token of .env.
                reason = self._locations_missing()
            elif name == 'mapillary_windows' and not (self.repo / WINDOWS_SCRIPT).is_file():
                # The CPU and the share only; whether the share is mounted the script says (no stat of it here: a share
                # that hangs would hang the status).
                reason = WINDOWS_SCRIPT + ' is missing in this installation'
            elif name == 'mapillary_windows':
                # Photos of Mapillary, not our drives: it needs no confirmed frame.
                pass
            elif name == 'train_classifier' and not (self.repo / CLASSIFIER_SCRIPT).is_file():
                reason = CLASSIFIER_SCRIPT + ' is missing in this installation'
            elif name == 'train_cutter' and not (self.repo / CUTTER_SCRIPT).is_file():
                reason = CUTTER_SCRIPT + ' is missing in this installation'
            elif name == 'train_decider' and not (self.repo / DECIDER_SCRIPT).is_file():
                reason = DECIDER_SCRIPT + ' is missing in this installation'
            elif name == 'train_brain' and not (self.repo / BRAIN_SCRIPT).is_file():
                reason = BRAIN_SCRIPT + ' is missing in this installation'
            elif name == 'train_brain' and (missing := self._brain_drives_missing()):
                reason = missing
            elif name == 'train_brain' and gpu_reason:
                reason = gpu_reason
            elif name == 'train_brain':
                # Its labels are his own speed on every dashcam drive with its logs read, each whole (the blocks of
                # Speed review correct it): no frame needs a confirmation.
                pass
            elif name in ROAD_LABELER_ACTIONS and (missing := self._road_labeler_missing(name)):
                reason = missing
            elif name in ROAD_LABELER_ACTIONS:
                # Its answers are the corrections of Speed review and the map: it needs no confirmed sign. The GPU is
                # checked when it starts (start() and the helper), so a dry run on the CPU stays possible while every
                # card is busy.
                pass
            elif name == 'box_fit_train' and not (self.repo / BOX_FIT_SCRIPT).is_file():
                reason = BOX_FIT_SCRIPT + ' is missing in this installation'
            elif name == 'box_fit_train':
                # On a V100 (#65): a free card, or one shared with the other trainings of SHARED_TRAINING_MB, checked
                # when it starts (start()), so a dry run on the CPU stays possible while every card is busy; the
                # candidates are cut on the CPU first. The script counts the boxes people drew and stops when they are
                # too few, and start() refuses a real run without .venv-gpu.
                pass
            elif name == 'train_users' and not (self.repo / USERNET_SCRIPT).is_file():
                reason = USERNET_SCRIPT + ' is missing in this installation'
            elif name == 'train_users' and gpu_reason:
                reason = gpu_reason
            elif name == 'train_users':
                # BDD100K's pictures or the road users boxed in Sign review: its script's dry run checks the stage's
                # data when it starts (start() refuses BDD's stage without its pictures); it needs no confirmed sign.
                pass
            elif name == 'regression' and not (self.repo / REGRESSION_SCRIPT).is_file():
                reason = REGRESSION_SCRIPT + ' is missing in this installation'
            elif name == 'regression' and gpu_reason:
                reason = gpu_reason
            elif name == 'regression':
                # What it needs (a certified drive, a model chosen) is checked by Models › Regression and by start().
                pass
            elif name == 'simulate' and not (self.repo / SIMULATE_SCRIPT).is_file():
                reason = SIMULATE_SCRIPT + ' is missing in this installation'
            elif name == 'simulate' and (room := _room_reason(best, SHARED_GPU[name], 'A replay')):
                # The replay decodes the video and runs the networks on a V100, never on the CPU: it shares the card
                # with the most free memory, and waits only while none has the room.
                reason = room
            elif name == 'simulate':
                # Its drive and its set of networks are checked by Models › Drive and by start(); it reads the drives'
                # videos and logs, so it needs no confirmed frame.
                pass
            elif name in ('networks_prepare', 'networks_publish') and not (self.repo / NETWORKS_SCRIPT).is_file():
                reason = NETWORKS_SCRIPT + ' is missing in this installation'
            elif name == 'networks_prepare':
                # Our networks only whole (the same rule as Prepare's proposer).
                reason = payload.get('proposers', {}).get('ours') or ''
            elif name == 'networks_publish' and not payload.get('networks', {}).get('packages'):
                reason = 'no package prepared yet'
            elif name == 'networks_publish' and not (payload.get('networks', {}).get('account') or {}).get('connected'):
                reason = 'connect Hugging Face first: Settings, Hugging Face'
            elif name == 'send_to_car':
                # The SignNet is chosen on the Models tab (start() checks it); the classifier is the one in use, and
                # the brain in use never one that takes the driver's inputs.
                reason = (car_missing(self.repo) or self.models_index.in_use('classifier')[1]
                          or brain_refused(self.models_index))
            elif name in ('try_on_comma', 'comma_restore'):
                # Whether the comma answers is known from a probe of a minute ago at most (never an ssh per page);
                # the job checks the rest on the comma and says why it stops.
                reason = trial_missing(self.repo) or (
                    self.models_index.in_use('classifier')[1] if name == 'try_on_comma' else '') or (
                    'the comma does not answer on the network: it answers while the car is on or just parked'
                    if self.comma_probe.state() is False else '')
            # [ai-reviewer] - START
            elif name in AI_ACTIONS and not (self.repo / action['command'][0]).is_file():
                reason = action['command'][0] + ' is missing in this installation'
            elif name == 'ai_review' and not payload.get('ai_reviewer', {}).get('available'):
                # A reviewer chosen but not passing its Test is named in the state, with why it cannot review.
                reason = ('the AI reviewer has not passed its Test: test it in Settings'
                          if payload.get('ai_reviewer', {}).get('provider') else 'no AI reviewer: choose one in Settings')
            elif name == 'ai_review' and not any(row.get('run_id') for row in rows):
                reason = 'no prepared drive'
            elif name == 'ai_review':
                # It reviews the frames nobody confirmed: it needs no confirmed frame.
                pass
            elif name == 'measure_ai_reviewer' and not ai_providers:
                reason = 'no AI provider: add one in Settings'
            # [ai-reviewer] - END
            elif name != 'prepare' and not confirmed:
                reason = 'no confirmed frame'
            elif name.startswith('train') and payload['reviewed_routes'] < payload['min_routes']:
                reason = (f'{payload["min_routes"]} reviewed drives are needed, '
                          f'now {payload["reviewed_routes"]}')
            elif name.startswith('train') and gpu_reason and _shared_card(cards, name) is None:
                reason = gpu_reason
            elif name in ('measure_proposer', 'setup_vlm') and gpu_reason:
                # Each takes a card of its own, as a training.
                reason = gpu_reason
            elif (name == 'setup_vlm' and best is not None and best.get('memory_used_mb') and best.get('memory_total_mb')
                  and best['memory_used_mb'] > best['memory_total_mb'] / 2):
                # Its last step loads Qwen3-VL on the GPU (about 17 GB): it would only run out of memory.
                reason = 'the GPU memory is taken by another program: free it first'
            elif name == 'train_classifier':
                # It mines the false boxes of the cutter in use (models/active.json): there must be one.
                reason = self.models_index.in_use('cutter')[1]
            states[name] = {'label': action['label'], 'allowed': not reason, 'reason': reason}
        return states

    def _road_labeler_missing(self, name: str) -> str:
        """Why the road labeler cannot be trained or run now, or '': its helper and .venv-gpu, a drive with a finished
        scan and its frames, and for Prepare Speed review a trained run."""
        if not (self.repo / ROAD_LABELER_SCRIPT).is_file():
            return ROAD_LABELER_SCRIPT + ' is missing in this installation'
        if not (self.repo / ROAD_LABELER_NNSLR).is_file():
            return 'no .venv-gpu: install it with scripts/setup_v100_training_env.sh'
        from nnslr_tools.road_labeler_data import labeler_routes
        try:
            if not labeler_routes(self.root):
                return 'no drive with its logs read and its frames'
        except Exception:  # the store being written: start() and the helper say it
            pass
        if name == 'road_labeler_predict':
            from nnslr_tools import labeler_runs
            if not labeler_runs.trained_runs(self.root):
                return 'no road labeler trained yet'
        return ''

    def _road_labeler_arguments(self, action: str, options: dict) -> list[str]:
        """The arguments of helpers/road_labeler.sh from the options (strings, checked by start()): --dry-run, and
        for Prepare Speed review the run (--run-dir), every drive (--all) and --no-marks (nothing written into Speed
        review). A real run, not a dry one, is refused while no card is free (the helper checks its card again, with its
        memory)."""
        dry = options.get('dry')
        if dry not in (None, 'on'):
            raise PanelError(400, 'bad_option', 'A dry run is "on", or none.')
        arguments = ['--dry-run'] if dry else []
        if action == 'road_labeler_predict':
            from nnslr_tools import labeler_runs
            name = options.get('labeler')
            if not labeler_runs.trained(self.root, name):
                raise PanelError(400, 'bad_option', 'Unknown road labeler run: choose a trained one in Models.')
            marks = options.get('marks')
            if marks not in (None, 'on', 'off'):
                raise PanelError(400, 'bad_option', 'The marks are on or off.')
            arguments += ['--run-dir=' + str(self.root / 'runs' / name), '--all']
            if marks == 'off':
                arguments.append('--no-marks')
        if not dry:
            self._refuse_busy_gpu()
        return arguments

    def _refuse_busy_gpu(self) -> None:
        """A real run of the road labeler, refused while no card is free: "The GPU is
        busy with another process: wait for it to finish, or start a dry run on the CPU." """
        cards, holders = self._gpu_states(self.system(), self.activities())
        reason = _gpu_busy(cards, holders)
        if reason:
            raise PanelError(409, 'gpu_taken', reason[0].upper() + reason[1:] + ': wait for it to finish, or start a '
                             'dry run on the CPU.')

    def _box_fit_arguments(self, options: dict) -> list[str]:
        """The arguments of scripts/train_box_fit.sh from the options (strings, checked by start()): --dry-run, and
        --no-refiner for {'refiner': 'off'}. A real run trains on the GPU with the Python of .venv-gpu (#65)."""
        dry = options.get('dry')
        if dry not in (None, 'on'):
            raise PanelError(400, 'bad_option', 'A dry run is "on", or none.')
        refiner = options.get('refiner')
        if refiner not in (None, 'on', 'off'):
            raise PanelError(400, 'bad_option', 'The refiner is on or off.')
        arguments = ['--dry-run'] if dry else []
        if refiner == 'off':
            arguments.append('--no-refiner')
        if not dry and not (self.repo / BOX_FIT_PYTHON).is_file():
            raise PanelError(409, 'not_available', 'No .venv-gpu: Fit box trains on the GPU with it; install it with '
                             'scripts/setup_v100_training_env.sh.')
        return arguments

    def box_fit(self) -> dict:
        """The Fit box card of Models (GET /api/panel/box-fit): the last summary of each part, as
        scripts/train_box_fit.sh keeps them in evaluations/box_fit of the data root, or None. The rankers
        (<time>.json): the boxes, the routes and the mean IoU with the person's box on routes left out, the rough box's
        and the fit's. The refiner (refiner-<time>.json): the rough box's, the rankers' and the refiner's mean IoU on the
        full frames of the routes left out, in all and on the small signs, and whether refiner.onnx was written. A file
        that cannot be read is skipped."""
        folder = self.root / BOX_FIT_SUMMARIES

        def newest(pattern: str) -> tuple[str, dict] | None:
            for path in sorted(folder.glob(pattern), reverse=True):
                data = _read_json(path)
                if isinstance(data, dict):
                    return path.name, data
            return None

        def number(value):
            return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

        def when(name: str) -> str | None:
            match = re.search(r'(\d{8}T\d{6}Z)\.json$', name)
            if not match:
                return None
            return datetime.strptime(match.group(1), '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).isoformat()

        rankers = None
        found = newest('[0-9]*.json')
        if found is not None:
            name, data = found
            overall = (data.get('cross_validation') or {}).get('overall') or {}
            rankers = {'file': name, 'at': when(name), 'boxes': number(data.get('boxes')),
                       'routes': number(data.get('routes')), 'folds': number(data.get('folds')),
                       'rough': number(overall.get('rough_mean_iou')), 'fit': number(overall.get('mean_iou')),
                       'written': bool(data.get('written'))}
        refiner = None
        found = newest('refiner-*.json')
        if found is not None:
            name, data = found
            left_out = data.get('left_out') or {}

            def measure(group: str) -> dict | None:
                values = left_out.get(group)
                if not isinstance(values, dict):
                    return None
                return {key: number(values.get(key)) for key in ('cases', 'rough', 'rankers', 'refiner')}

            refiner = {'file': name, 'at': when(name), 'boxes': number(data.get('train_boxes')),
                       'held_out_boxes': number(data.get('held_out_boxes')), 'best_epoch': number(data.get('best_epoch')),
                       'epochs': number(data.get('epochs')), 'minutes': number(data.get('minutes')),
                       'all': measure('all'), 'small': measure('size:small'),
                       'wins': bool(data.get('refiner_wins')), 'written': bool(data.get('written'))}
        return {'rankers': rankers, 'refiner': refiner, 'folder': str(BOX_FIT_SUMMARIES),
                'refiner_model': (self.repo / 'src' / 'nnslr_tools' / 'box_fit' / 'refiner.onnx').is_file()}

    def _locations_missing(self) -> str:
        """Why the Mapillary positions cannot be fetched now, or ''."""
        from nnslr_tools.mapillary import token_ready
        if not (self.repo / LOCATIONS_SCRIPT).is_file():
            return LOCATIONS_SCRIPT + ' is missing in this installation'
        if not token_ready(self.repo):
            return "no Mapillary token: add MAPILLARY_TOKEN to the server's .env"
        return ''

    def _brain_drives_missing(self) -> str:
        """Why the brain has no drive to learn from now, or '': a drive is one with sunnypilot off (a dashcam drive), its
        logs read to the end and its frames (brain_targets.training_routes, the one rule of the training; what they give:
        brain_labels). A drive is labelled by being driven, so no block or mark of Speed review is needed."""
        try:
            # It loads the car's brain_inputs.py from the sunnypilot clone (once a process).
            from nnslr_tools import brain_targets
            left_out: dict[str, str] = {}
            found = brain_targets.training_routes(self.root, left_out)
        except (ImportError, OSError, ValueError, sqlite3.Error) as exc:
            return str(exc) or 'the drives of the brain cannot be read'
        if found:
            return ''
        if left_out:
            return (f'no drive to learn from: sunnypilot was on in every drive with its logs read ({len(left_out)}); '
                    'drive with sunnylink › Toggles › Enable sunnypilot off')
        return 'no drive to learn from yet: prepare one, then Read the logs on Data › Speed review'

    def brain_labels(self) -> dict:
        """Train › Brain, what a training gets today from every drive it learns from (training_routes: every dashcam
        drive, each whole; training: brain_targets.training_counts, their number and hours): his speed with the blocks of
        Speed review and the roadworks, as
        brain_targets.summary, with the drives the next training reads again first (to_read: not read by the networks in
        use as the car reads them, applicability included (sign_networks: every drive while those three are not exported
        yet), or without the brain's pictures) and
        the names of the kinds of road (road_labels), and the earlier brain runs whose drives the next one can take
        (same_drives: brain_route_runs). Without the inputs, which it does not show: about 10 s over every
        drive, asked when the section opens, never by the status."""
        from nnslr_tools.drive_review import DriveReviewStore
        store = DriveReviewStore(self.root)
        try:
            # It loads the car's brain_inputs.py from the sunnypilot clone.
            from nnslr_tools import brain_targets, sign_networks
            used = sign_networks.in_use(self.root)
            folder = sign_networks.exported(self.root, used) if all(used.get(part) for part in sign_networks.PARTS) \
                else None
            # As the training will have them, once it has read again what it reads first.
            found = brain_targets.summary(brain_targets.dataset(self.root, lambda route: store.merged(route, keep=False),
                                                                log=lambda text: None, rate=brain_targets.RATE_HZ,
                                                                need_pictures=False, inputs=False, types=True,
                                                                networks={**used, 'models': folder} if folder else None))
            labels = brain_targets.road_labels(self.root)
            training = brain_targets.training_counts(self.root)
        except (ImportError, OSError, ValueError, sqlite3.Error) as exc:
            return {'problem': str(exc), 'same_drives': self.brain_route_runs()}
        return {**found, 'road_labels': labels, 'training': training, 'same_drives': self.brain_route_runs(),
                'to_read': sum(folder is None or not row['reader_current'] or not row.get('pictures', True)
                               for row in found['drives'])}

    # -- the dataset and the starting points of a training ----------------------------------------
    def dataset_summary(self) -> dict | None:
        """What the gold dataset holds (frames, objects by family, human/AI, per route), its split and the
        last import. Read again only when one of the files changed."""
        objects = self.root / 'annotations' / 'objects.jsonl'
        files = [objects, self.root / 'logs' / 'import.json', self.root / 'splits' / 'latest.json']
        key = tuple(path.stat().st_mtime_ns if path.is_file() else 0 for path in files)
        if self._dataset is not None and self._dataset[0] == key:
            return self._dataset[1]
        summary = _dataset_summary(self.root) if objects.is_file() else None
        self._dataset = (key, summary)
        return summary

    def training_starts(self) -> list[dict]:
        """Where a training can start: ImageNet weights (the default), random weights, the stable model
        on Hugging Face (when one is published), or a finished run of this server."""
        starts = [
            {'key': 'imagenet', 'label': 'Pre-trained on ImageNet', 'default': True, 'available': True, 'reason': '',
             'description': 'MobileNetV3 that already knows shapes, edges and circles from a million photos: '
                            'it learns the Swiss signs fastest.'},
            {'key': 'random', 'label': 'Random weights', 'default': False, 'available': True, 'reason': '',
             'description': 'A network that knows nothing: it learns everything from the reviewed frames. '
                            'Slower, and weaker with few frames.'},
        ]
        stable = self._stable_published()
        starts.append({'key': 'stable', 'label': 'The stable model', 'default': False, 'available': stable is True,
                       'reason': '' if stable is True else ('Checking Hugging Face…' if stable is None else
                                                             'No stable model published on Hugging Face yet'),
                       'description': 'Continues from the last model published as stable on Hugging Face.'})
        for run in self._finished_runs():
            starts.append({'key': 'run:' + run, 'label': 'Run ' + run, 'default': False, 'available': True,
                           'reason': '', 'description': 'Continues from the best weights of this earlier '
                                                        'training on this server.'})
        return starts

    def _finished_runs(self) -> list[str]:
        """Runs of this server with the best weights of both networks, newest first."""
        runs = self.root / 'runs'
        if not runs.is_dir():
            return []
        found = [entry for entry in runs.iterdir() if RUN_NAME.fullmatch(entry.name)
                 and (entry / 'detector' / 'detector-best.pt').is_file() and (entry / 'reader' / 'reader-best.pt').is_file()]
        return [entry.name for entry in sorted(found, key=lambda entry: entry.stat().st_mtime, reverse=True)]

    def _stable_published(self) -> bool | None:
        """True when training/stable.json exists on Hugging Face; None while not checked yet. Asked at most
        every 10 minutes, in the background, so a status call never waits for the network."""
        with self._lock:
            checked_at, published = self._stable
            if time.monotonic() - checked_at > STABLE_CHECK_SECONDS and not self._stable_checking:
                self._stable_checking = True
                threading.Thread(target=self._check_stable, daemon=True).start()
        return published

    def _check_stable(self) -> None:
        repo = os.environ.get('NNSLR_HF_REPO', 'cristianku/sunnypilot-speedlimits-models')
        url = 'https://huggingface.co/' + repo + '/resolve/main/training/stable.json'
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=10) as answer:
                published = answer.status == 200
        except (OSError, ValueError):
            # 401/404 (no repository or no stable.json) and no network alike: not available.
            published = False
        with self._lock:
            self._stable = (time.monotonic(), published)
            self._stable_checking = False

    # -- the training runs, for the Evaluate tab ------------------------------------------------------
    def runs(self) -> dict:
        """Every run under <root>/runs with its state and headline numbers, newest first."""
        return self.runs_index.summaries()

    def run(self, name: str) -> dict:
        """One run: its evaluation per label and per sign size, the dataset and split it used, its curves."""
        detail = self.runs_index.detail(name)
        if detail is None:
            raise PanelError(404, 'unknown_run', 'No such training run.')
        return detail

    def frames(self, name: str, model, partition, route=None, previews=None) -> dict:
        """The frames of one route of an evaluation of a run, with the boxes over each, for the Watch panel
        (nnslr_tools.runs.RunsIndex.frames)."""
        from nnslr_tools.drives import drive_of
        try:
            answer = self.runs_index.frames(name, model, partition, route, previews)
        except WatchError as exc:
            raise PanelError(exc.status, exc.reason, exc.message) from exc
        # Each route as a drive ("Basel → Mumpf", when it was driven), for the Drive view of Results.
        for item in answer.get('routes', []):
            drive = drive_of(self.root, item['route']) or {}
            item['title'], item['when'] = drive.get('title'), drive.get('when')
        return answer

    # -- the networks by role, for the Models tab -----------------------------------------------------
    def proposers(self) -> dict:
        """The latest measure of each proposer of Prepare (helpers/measure_proposer.sh), and why one cannot run."""
        from nnslr_tools.preannotate import PROPOSERS, ours_missing, vlm_missing
        from nnslr_tools.proposer_eval import latest_reports
        reports = latest_reports(self.root)
        unavailable = {'qwen': vlm_missing(self.root), 'ours': ours_missing(self.root)}
        best = _best_proposer(reports, {key for key, why in unavailable.items() if why})
        return {'proposers': [{'key': key, 'label': label, 'unavailable': unavailable.get(key)}
                              for key, label in PROPOSERS.items()], 'reports': reports,
                # The choice of Prepare: kept on the server once made (one that cannot run stays chosen, and Prepare
                # says why with its buttons off), else the best measured that can run.
                'chosen': self._kept_proposer() or best or DEFAULT_PROPOSER, 'best': best,
                # What Our networks propose with now: the cutter, classifier and decider in use (Models), by id.
                'networks': _active_parts(self.root)}

    def _kept_proposer(self) -> str | None:
        """The proposer chosen on the Prepare tab (PROPOSER_FILE), or None before a choice."""
        kept = (_read_json(self.root / PROPOSER_FILE) or {}).get('proposer')
        return kept if kept in PROPOSERS else None

    def set_proposer(self, body) -> dict:
        """Keeps the choice of Prepare's Found by ({'proposer': 'ours'}) for every browser; proposers() with it. A
        read-only panel refuses it, as it refuses the actions."""
        if not self.actions_enabled:
            raise PanelError(403, 'actions_disabled', 'The panel is read-only on this server.')
        proposer = body.get('proposer') if isinstance(body, dict) else None
        if proposer not in PROPOSERS:
            raise PanelError(400, 'bad_option', 'Unknown proposer: ' + ', '.join(PROPOSERS) + '.')
        path = self.root / PROPOSER_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name('.' + path.name + '.tmp')
        temporary.write_text(json.dumps({'proposer': proposer}) + '\n', encoding='utf-8')
        os.replace(temporary, path)
        return self.proposers()

    # [ai-reviewer] - START
    def ai_reviewers(self) -> dict:
        """The AI providers of Settings (never a key) with the latest measure of each (helpers/measure_ai_reviewer.sh),
        for the AI reviewers view of the Evaluate tab."""
        from nnslr_tools import ai_review, llm_providers
        return {**llm_providers.public_settings(self.root), 'reports': ai_review.latest_reports(self.root)}

    def _ai_settings(self) -> tuple[dict, int]:
        """The state of the AI reviewer (llm_providers.ai_state) and how many providers there are. A providers file
        that cannot be read makes the AI unavailable, never the whole panel."""
        from nnslr_tools import llm_providers
        from nnslr_tools.review_db import ReviewRejected
        try:
            return llm_providers.ai_state(self.root), len(llm_providers.public_settings(self.root)['providers'])
        except ReviewRejected as exc:
            return {'available': False, 'reason': exc.message, 'provider': None}, 0

    def _ai_arguments(self, action: str, options: dict) -> list[str]:
        """The arguments of an AI action from its options, all strings: the route (ai_review only, one with a
        prepared report), --provider=<id of Settings> (the reviewer when none is given), --save (ai_review with
        save 'yes'; 'no' is the default) and --limit=<frames, 1 to MAX_AI_LIMIT>. Never a key: the job reads the
        providers file itself."""
        from nnslr_tools import llm_providers
        arguments = []
        if action == 'ai_review':
            route = options.get('route')
            if not route:
                raise PanelError(400, 'bad_option', 'Choose the route to review.')
            from nnslr_tools.preannotate import PreannotationError
            from nnslr_tools.review import latest_report
            try:
                latest_report(self.root, route)
            except (PreannotationError, ValueError, OSError):
                raise PanelError(400, 'bad_option', 'Unknown route, or a route without a prepared report.') from None
            arguments.append(route)
        settings = llm_providers.public_settings(self.root)
        provider = options.get('provider')
        if provider is None:
            provider = settings['reviewer']
            if provider is None:
                raise PanelError(409, 'no_ai_reviewer', 'Choose an AI reviewer in Settings.')
        elif not PROVIDER_ID.fullmatch(provider) or provider not in {item['id'] for item in settings['providers']}:
            raise PanelError(400, 'bad_option', 'Unknown AI provider: add it in Settings.')
        arguments.append('--provider=' + provider)
        save = options.get('save', 'no')
        if save not in ('yes', 'no'):
            raise PanelError(400, 'bad_option', 'Save is yes or no.')
        if save == 'yes':
            # Its saves are final: only a provider that passed its last Test (also checked by the job itself).
            refusal = llm_providers.review_refusal(llm_providers.provider(self.root, provider))
            if refusal:
                raise PanelError(409, 'ai_not_tested', refusal)
            arguments.append('--save')
        limit = options.get('limit')
        if limit is not None:
            if not re.fullmatch(r'[0-9]{1,4}', limit) or not 1 <= int(limit) <= MAX_AI_LIMIT:
                raise PanelError(400, 'bad_option', f'The limit is a number of frames from 1 to {MAX_AI_LIMIT}.')
            arguments.append('--limit=' + str(int(limit)))
        return arguments
    # [ai-reviewer] - END

    def models(self) -> dict:
        """Every trained cutter, classifier and decider with its numbers, newest first, and the one in use; and
        whether a build of the testing branch is still owed after a Publish to the car (publishing again builds it)."""
        return {**self.models_index.index(), 'car_build_pending': build_pending(self.root),
                'comma_trial': read_trial(self.root)}

    def car_summary(self) -> dict:
        """The networks of the car at a glance, from two small files: the ones in use (models/active.json) and the ones
        the comma's software has (models/on-car.json; None before the first Publish), by id, the reason one of them
        cannot be read, and whether a build of the testing branch is still owed."""
        active, problem = self.models_index.read_active()
        on_car, car_problem = self.models_index.read_on_car()
        roles = tuple(CAR_NETWORKS.values())
        in_use = {role: (active.get(role) or {}).get('id') for role in roles}
        return {'in_use': in_use,
                'on_car': None if on_car is None else {role: on_car['ids'].get(role) for role in roles},
                'problem': problem or car_problem, 'build_pending': build_pending(self.root)}

    def set_active_model(self, body, *, who: str | None, via: str) -> dict:
        """The model in use for a role ({"role", "id"}), written to <root>/models/active.json. A read-only panel
        refuses it, as it refuses the actions."""
        if not self.actions_enabled:
            raise PanelError(403, 'actions_disabled', 'The panel is read-only on this server.')
        if not isinstance(body, dict):
            raise PanelError(400, 'invalid_json', 'A JSON object is required.')
        try:
            return self.models_index.set_active(body.get('role'), body.get('id'), who=who, via=via)
        except ModelChoiceError as exc:
            raise PanelError(exc.status, exc.reason, exc.message) from exc

    # -- progress of the preparation -----------------------------------------------------------
    def preparation(self, rows: list[dict], activities: list[dict]) -> dict | None:
        """How far the running preparation is, from what it already wrote; None when none runs.

        A route counts half for the extraction (segments with frames.jsonl out of the linked ones)
        and half for the pre-annotation (frames already in the report being written out of the
        extracted ones). The overall percent is over the routes that helpers/1_prepare.sh listed in
        BATCH_FILE when it started; without that file (a run of an older 1_prepare.sh) only the
        routes being worked on are shown.
        """
        running = False
        for item in activities:
            # The map reading shares the group (one at a time with a preparation) but prepare no drive.
            if item['group'] == 'prepare' and item['label'] != 'map reading':
                running = True
        if not running:
            return None
        by_route = {}
        for row in rows:
            by_route[row['route']] = row
        result = {'percent': None, 'routes_done': 0, 'routes_total': 0, 'working': []}
        again = self._again_progress(by_route)
        if again is not None:
            return again
        batch = self._batch_routes()
        if batch is not None:
            total = 0
            for route in batch:
                row = by_route.get(route)
                if row is None:
                    continue
                progress = self._route_progress(row)
                total = total + progress['percent']
                result['routes_total'] = result['routes_total'] + 1
                if progress['percent'] == 100:
                    result['routes_done'] = result['routes_done'] + 1
            if result['routes_total']:
                result['percent'] = total // result['routes_total']
        for route in sorted(self._working_routes()):
            row = by_route.get(route)
            if row is not None:
                result['working'].append(self._route_progress(row))
        return result

    def _again_progress(self, by_route: dict) -> dict | None:
        """Propose again (helpers/1_prepare.sh --again): every route of the batch already has a report, the new
        proposals are written aside (.pending-<run>) and replace it when the route is done. Per route: waiting, its
        frames proposed so far, or ready (a report newer than the batch); the percent over the frames of the batch,
        the time left from the pace of the route at work, and who proposes. None for a batch of new routes."""
        try:
            batch = json.loads((self.root / 'logs/prepare/batch.json').read_text(encoding='utf-8'))
            started = datetime.strptime(batch['started'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
            routes = [route for route in batch['routes'] if isinstance(route, str)]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if not routes or any(by_route.get(route, {}).get('run_id') is None for route in routes):
            return None
        from nnslr_tools.drives import drive_of
        from nnslr_tools.preannotate import PROPOSER_NAMES
        now = datetime.now(timezone.utc)
        rows, done_frames, all_frames, pace = [], 0, 0, None
        for route in routes:
            row = by_route[route]
            base = self.root / 'derived/preannotations' / route
            total = row.get('frames') or row.get('report_frames') or 0
            pending = sorted(base.glob('.pending-*'))
            fresh = pending[-1] if pending and datetime.fromtimestamp(pending[-1].stat().st_mtime, timezone.utc) >= started else None
            if fresh is not None:
                done = min(_line_count(fresh / 'preannotations.jsonl'), total)
                since = (now - datetime.fromtimestamp(fresh.stat().st_ctime, timezone.utc)).total_seconds()
                if done and since > 0:
                    pace = since / done
                phase, percent = 'proposing again', (min(99, 100 * done // total) if total else 0)
            elif str(row['run_id']) >= started.strftime('%Y%m%dT%H%M%SZ'):
                done, phase, percent = total, 'ready', 100
            else:
                done, phase, percent = 0, 'waiting', 0
            done_frames, all_frames = done_frames + done, all_frames + total
            rows.append({'route': route, 'title': (drive_of(self.root, route) or {}).get('title'), 'phase': phase,
                         'done': done, 'total': total, 'unit': 'frames', 'percent': percent})
        left = round(pace * (all_frames - done_frames)) if pace else None
        for item in rows:
            if item['phase'] == 'proposing again' and pace:
                item['seconds_left'] = round(pace * (item['total'] - item['done']))
        proposer = self._again_proposer()
        return {'percent': round(100 * done_frames / all_frames, 1) if all_frames else None,
                'routes_done': sum(1 for item in rows if item['phase'] == 'ready'), 'routes_total': len(rows),
                'working': rows, 'again': True, 'seconds_left': left,
                'proposer': PROPOSER_NAMES.get(proposer, proposer) if proposer else None}

    def _again_proposer(self) -> str | None:
        """Who proposes again: the "(qwen)" of the newest prepare_again job's log."""
        logs = sorted(self.logs.glob('*-prepare_again.log')) if self.logs.is_dir() else []
        if not logs:
            return None
        found = re.findall(r'(?:proposing the signs again|propongo di nuovo i cartelli) \((\w+)\)',
                           logs[-1].read_text(encoding='utf-8', errors='replace'))
        return found[-1] if found else None

    def _route_progress(self, row: dict) -> dict:
        route = row['route']
        if row['run_id'] is not None:
            return {'route': route, 'phase': 'ready', 'done': row['report_frames'],
                    'total': row['report_frames'], 'unit': 'frames', 'percent': 100}
        if row['linked'] == 0:
            return {'route': route, 'phase': 'to link', 'done': 0, 'total': 0, 'unit': 'segments', 'percent': 0}
        if row['extracted'] < row['linked']:
            return {'route': route, 'phase': 'extracting frames', 'done': row['extracted'], 'total': row['linked'],
                    'unit': 'segments', 'percent': 50 * row['extracted'] // row['linked']}
        done = self._preannotated_frames(route)
        total = row['frames']
        percent = 50
        if total:
            percent = 50 + 50 * min(done, total) // total
        # The report counts only once it is moved in place (run_id above).
        if percent > 99:
            percent = 99
        return {'route': route, 'phase': 'proposing signs', 'done': done, 'total': total, 'unit': 'frames', 'percent': percent}

    def _batch_routes(self) -> list[str] | None:
        try:
            routes = json.loads((self.root / BATCH_FILE).read_text()).get('routes')
        except (OSError, ValueError, AttributeError):
            return None
        if not isinstance(routes, list):
            return None
        result = []
        for route in routes:
            if isinstance(route, str):
                result.append(route)
        return result

    def _preannotated_frames(self, route: str) -> int:
        """Frames in the report the pre-annotation is writing (the newest .pending-<run id>)."""
        base = self.root / 'derived/preannotations' / route
        if not base.is_dir():
            return 0
        pending = sorted(base.glob('.pending-*'))
        if not pending:
            return 0
        return _line_count(pending[-1] / 'preannotations.jsonl')

    def _working_routes(self) -> set[str]:
        """Routes an nnslr extract-frames or nnslr preannotate is working on right now."""
        found = set()
        proc = Path('/proc')
        if not proc.is_dir():
            return found
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                arguments = (entry / 'cmdline').read_bytes().decode('utf-8', 'replace').split('\0')
            except OSError:
                continue
            if 'extract-frames' not in arguments and 'preannotate' not in arguments:
                continue
            if '--route' not in arguments:
                continue
            position = arguments.index('--route')
            if position + 1 < len(arguments):
                found.add(arguments[position + 1])
        return found

    # -- actions -------------------------------------------------------------------------------
    def _mapillary_choice(self, action: str, options: dict, mapillary: bool) -> dict:
        """The environment of the region and share of the Mapillary crops of the classifier, or of the windows of the
        cutter ({'region': 'europe,CH', 'share': '5'}): only with the Mapillary signs on; a region other than all needs
        the positions."""
        from nnslr_tools.mapillary import FULL_SHARE, LOCATIONS_FILE, check_share, region_text
        region, share = options.get('region'), options.get('share')
        if region is None and share is None:
            return {}
        if action not in MAPILLARY_ENV:
            raise PanelError(400, 'bad_option', 'Only "Train the cutter" and "Train the classifier" take a region and '
                             'a share of Mapillary.')
        if not isinstance(region or '', str) or not isinstance(share or '', str):
            raise PanelError(400, 'bad_option', 'The region and the share are strings.')
        try:
            region, share = region_text(region), check_share(share)
        except ValueError as error:
            words = str(error).split(': ', 1)[-1]  # without its code (mapillary_region: ...)
            raise PanelError(400, 'bad_option', 'Mapillary: ' + words + '.') from None
        if region == 'all' and share == FULL_SHARE:
            return {}
        if not mapillary:
            raise PanelError(400, 'bad_option', 'The region and the share choose Mapillary signs: turn Mapillary on.')
        if region != 'all' and not (self.root / LOCATIONS_FILE).is_file():
            raise PanelError(409, 'not_available', 'The Mapillary positions are not fetched yet: '
                             'locate the Mapillary photos first.')
        env = {}
        if region != 'all':
            env[MAPILLARY_ENV[action] + '_REGION'] = region
        if share != FULL_SHARE:
            env[MAPILLARY_ENV[action] + '_SHARE'] = f'{share:g}'
        return env

    def _classifier_cutter(self, cutter: object) -> str:
        """NNSLR_CUTTER of "Train the classifier": a finished SignNet by its id ('cutter:<run>/signnet', the one a
        pipeline has just trained), its weights and its result in its run."""
        from nnslr_tools.chain import SIGNNET_FILE, SIGNNET_REFERENCE
        match = SIGNNET_REFERENCE.fullmatch(cutter) if isinstance(cutter, str) else None
        if match is None:
            raise PanelError(400, 'bad_option', 'The cutter is a SignNet of Models, by its id.')
        run = self.root / 'runs' / match.group(1)
        if not (run / SIGNNET_FILE).is_file() or not (run / 'result.json').is_file():
            raise PanelError(409, 'not_available', 'That SignNet has not finished its training on this server.')
        return cutter

    def _classifier_start(self, start: object) -> dict:
        """The environment of where "Train the classifier" starts: nothing for ImageNet (the default), or
        NNSLR_CLASSIFIER_INIT_FROM, the reader-best.pt of a finished classifier with "not a sign" of the Models tab
        given by its id ('classifier:<run>/classifier'); with the two stages, their Mapillary stage starts from it."""
        if start is None or start == CLASSIFIER_IMAGENET:
            return {}
        if not isinstance(start, str):
            raise PanelError(400, 'bad_option', 'The start is a string: imagenet or the id of a classifier.')
        entry = next((entry for entry in self.models_index.artifacts(('classifier',))['classifier']
                      if entry['id'] == start), None)
        if entry is None:
            raise PanelError(400, 'bad_option', 'Unknown classifier to start from: choose one of Models.')
        if not CLASSIFIER_ID.fullmatch(start) or entry.get('part') != 'classifier':
            raise PanelError(400, 'bad_option', 'An old reader cannot be a start: only a classifier with “not a sign”.')
        if not entry['finished']:
            raise PanelError(400, 'bad_option', 'That classifier has not finished its training: choose a finished one.')
        from nnslr_tools.chain import resolve_model
        try:
            weights = resolve_model(start, 'classifier', self.root)['path']
        except ValueError:
            raise PanelError(400, 'bad_option', 'The weights of that classifier are missing on this server.') from None
        return {'NNSLR_CLASSIFIER_INIT_FROM': str(weights)}

    def _regression_models(self) -> tuple[list[dict], list[dict]]:
        """The driving models and the brains Models › Regression offers."""
        from nnslr_tools import regression
        from nnslr_tools.localized import distill_root
        active, _problem = self.models_index.read_active()
        chosen = (active.get('brain') or {}).get('id')
        versions = [{**entry, 'in_use': entry['id'] == chosen} for entry in self.models_index.artifacts(('brain',))['brain']]
        return regression.driving_models(distill_root()), regression.brain_models(versions)

    def _regression_choices(self, options: dict) -> dict:
        """The variables of scripts/run_regression.sh: what he chose, each of them known; a run needs a drive and a model."""
        from nnslr_tools import regression
        if not regression.certified(self.root):
            raise PanelError(409, 'not_available', 'Certify a drive first.')
        models, brains = self._regression_models()
        try:
            driving = regression.checked_choice(options.get('openpilot', ''), {model['key'] for model in models}, 'driving model')
            chosen = regression.checked_choice(options.get('brains', ''), {brain['key'] for brain in brains}, 'brain')
        except regression.RegressionError as problem:
            raise PanelError(problem.status, problem.reason, problem.message) from problem
        if not driving and not chosen:
            raise PanelError(400, 'bad_option', 'Choose a model first.')
        return {'NNSLR_REGRESSION_OPENPILOT': ','.join(driving), 'NNSLR_REGRESSION_BRAINS': ','.join(chosen)}

    def regression(self) -> dict:
        """GET /api/regression/routes: every drive, the certified ones with their state, and the models to choose."""
        from nnslr_tools import drive_logs, drive_review, regression, speed_review
        from nnslr_tools.localized import distill_root
        store, drives = speed_review.store_for(self.root), drive_review.store_for(self.root)
        models, brains = self._regression_models()
        rows = regression.route_rows(
            self.root, speed_review.drives_list(store, drives), drive_logs.videos_dir(), distill_root(),
            [model['key'] for model in models], [brain['key'] for brain in brains], regression.frozen_split(self.root),
            store.blocks, lambda route: (drives.summary(route) or {}).get('t0_ns'))
        return {'ok': True, 'routes': rows, 'models': models, 'brains': brains, 'horizons': list(regression.HORIZONS_S),
                'horizon': regression.DEFAULT_HORIZON_S}

    def regression_route(self, route: str, horizon: int, brain: str | None) -> dict:
        """GET /api/regression/routes/<route>: the curves of a drive."""
        from nnslr_tools import drive_review, regression
        models, brains = self._regression_models()
        return {**regression.route_view(self.root, route, drive_review.store_for(self.root).merged(route), models, brains,
                                        horizon=horizon, brain=brain),
                'certified': regression.certified(self.root).get(route)}

    def regression_certify(self, route: str, on: bool, who: str) -> dict:
        """POST /api/regression/routes/<route>/certify: a drive in or out of the certified ones."""
        from nnslr_tools import drive_review, regression
        if on:
            regression.certify(self.root, route, who, drive_review.all_routes(self.root))
        else:
            regression.uncertify(self.root, route)
        return self.regression()

    # -- Models › Drive ---------------------------------------------------------------------------
    def _again_routes(self, value: str) -> list[str]:
        """The drives of Propose again for {'route': 'R'} or 'R1,R2,…' (issue #42), each with a report, in the order
        given, once: the helper's arguments after --again, which redo only those (their confirmed frames kept)."""
        from nnslr_tools.review import list_reports
        routes = list(dict.fromkeys(part.strip() for part in value.split(',') if part.strip()))
        if not routes:
            raise PanelError(400, 'bad_option', 'Propose again of a drive takes its id.')
        known = {route for route, _directory in list_reports(self.root)}
        for route in routes:
            if not ROUTE_ID.match(route) or route not in known:
                raise PanelError(400, 'bad_option', 'No proposals to redo for this drive: ' + route[:40] + '.')
        return routes

    def _simulate_arguments(self, options: dict) -> list[str]:
        """The arguments of helpers/simulate_drive.sh from the options (strings, checked by start()): --set=<car|in-use>,
        then --route=<drive> for {'route': R}, a drive the portal has, or --certified for {'certified': '1'}, every
        certified drive: exactly one of the two. No dry run: the replay is always on a V100, and start() takes its card
        (refused while none is free)."""
        from nnslr_tools import drive_review, simulation
        chosen = options.get('set')
        if chosen not in simulation.SETS:
            raise PanelError(400, 'bad_option', 'Choose the networks of the replay: ' + ' or '.join(simulation.SETS) + '.')
        route, certified = options.get('route'), options.get('certified')
        if certified not in (None, '1'):
            raise PanelError(400, 'bad_option', 'Every certified drive is "1", or none.')
        if (route is None) == (certified is None):
            raise PanelError(400, 'bad_option', 'Replay one drive (route) or every certified drive (certified): one of '
                             'the two.')
        if certified:
            if not simulation.certified(self.root):
                raise PanelError(409, 'not_available', 'Certify a drive first.')
            return ['--set=' + chosen, '--certified']
        if not ROUTE_ID.match(route) or route not in drive_review.all_routes(self.root):
            raise PanelError(400, 'bad_option', 'Unknown drive: ' + route[:40] + '.')
        return ['--set=' + chosen, '--route=' + route]

    def _simulation_sets(self) -> dict:
        """The two sets of networks as Models › Drive shows them (simulation.set_states): the release on the car and the
        export of the versions in use, which is ready only while it is of the versions in use on Models now (by the
        car's names of the networks, CAR_NETWORKS)."""
        from nnslr_tools import simulation
        active, _problem = self.models_index.read_active()
        in_use = {network: (active.get(role) or {}).get('id') for network, role in CAR_NETWORKS.items()}
        return simulation.set_states(self.root, in_use=in_use)

    def simulation(self) -> dict:
        """GET /api/simulation/drives: the sets of networks and every drive (the rows of Speed review's list), each
        with what it has: its replays by set, its certification and how each replay stands next to the certified one.
        The sets and the car's code are read once for the whole list."""
        from nnslr_tools import drive_review, simulation, speed_review
        states, code = self._simulation_sets(), simulation.code_hashes()
        rows = []
        for drive in speed_review.drives_list(speed_review.store_for(self.root), drive_review.store_for(self.root)):
            found = simulation.summary(self.root, drive['route'], states, code)
            rows.append({**drive, **{key: found[key] for key in ('sets', 'certified', 'regression')}})
        return {'ok': True, 'sets': states, 'drives': rows}

    def simulation_drive(self, route: str, set_key: str) -> dict:
        """GET /api/simulation/drives/<route>?set=<car|in-use|certified>: the replay of a drive with a set as the page
        draws it (view: None when that set was never replayed on it; 'certified': the certified replay), and, for a
        set next to a certified drive, where the two differ (differences). On the clock of the drive's logs as the
        timeline has it (the last scan's t0_ns): SimulationError 409 no_logs before Read the logs."""
        from nnslr_tools import drive_review, simulation
        if set_key != 'certified' and set_key not in simulation.SETS:
            raise simulation.SimulationError(400, 'bad_set', 'The set is ' + ', '.join(simulation.SETS) + ' or certified.')
        t0_ns = (drive_review.store_for(self.root).summary(route) or {}).get('t0_ns')
        if t0_ns is None:
            raise simulation.SimulationError(409, 'no_logs', 'The logs of this drive are not read yet: Read the logs on '
                                             'Data › Speed review first, the replay is drawn on their clock.')
        states = self._simulation_sets()
        was = simulation.reference(self.root, route)
        record = was if set_key == 'certified' else simulation.read(self.root, route, set_key)
        differences = None
        if set_key != 'certified' and was is not None and record is not None:
            differences = simulation.differences(simulation.compare(was, record), t0_ns)
        return {'ok': True, 'route': route, 'sets': states, 'summary': simulation.summary(self.root, route, states),
                'view': None if record is None else simulation.view(record, t0_ns), 'differences': differences}

    def simulation_certify(self, route: str, on: bool, set_key: str | None, who: str, made_at: str | None = None) -> dict:
        """POST /api/simulation/drives/<route>/certify: the drive's replay with a set becomes its reference
        (SimulationError 409 no_simulation without one; 409 changed when it is no longer the one of made_at, the replay
        the person looked at), or the drive leaves the certified ones."""
        from nnslr_tools import simulation
        if on:
            simulation.certify(self.root, route, set_key, who, made_at)
        else:
            simulation.uncertify(self.root, route)
        return {'ok': True, 'summary': simulation.summary(self.root, route, self._simulation_sets())}

    def _brain_choices(self, options: dict, check_only: bool = False) -> dict:
        """The variables of scripts/run_brain.sh from the options of Train › Brain ({'works', 'road', 'types', 'pool',
        'seeds'}), the usual ones when not given: the roadworks, the kinds of road and the road types on, the picture as a
        grid, the three seeds.
        'driver' is only 'off' (a page loaded before 10 Nov 2026 sends it): the brain never takes his pedals nor the
        car's speed."""
        if options.get('driver', 'off') != 'off':
            raise PanelError(400, 'bad_option', 'The brain no longer takes the driver’s inputs: with the car’s speed in, '
                                                'it followed your speed. Reload the page.')
        works, road, types = options.get('works', 'on'), options.get('road', 'on'), options.get('types', 'on')
        pool, seeds, users = options.get('pool', BRAIN_POOLS[0]), options.get('seeds', 'all'), options.get('users', 'on')
        if not all(isinstance(value, str) for value in (works, road, types, pool, seeds, users)) \
                or works not in BRAIN_WORKS or road not in BRAIN_ROAD or types not in BRAIN_TYPES \
                or pool not in BRAIN_POOLS or seeds not in BRAIN_SEEDS or users not in BRAIN_USERS:
            raise PanelError(400, 'bad_option', 'Roadworks, the kinds of road, the road types and the road users are on '
                                                'or off, the picture mean or grid, the seeds all or one.')
        found = {'NNSLR_BRAIN_WORKS': BRAIN_WORKS[works], 'NNSLR_BRAIN_ROAD': BRAIN_ROAD[road],
                 'NNSLR_BRAIN_TYPES': BRAIN_TYPES[types], 'NNSLR_BRAIN_POOL': pool,
                 'NNSLR_BRAIN_SEEDS': BRAIN_SEEDS[seeds], 'NNSLR_BRAIN_USERS': BRAIN_USERS[users],
                 **self._brain_sign_networks(options, check_only)}
        routes_from = options.get('routes_from')
        if routes_from not in (None, ''):
            if not isinstance(routes_from, str) \
                    or routes_from not in {run['run'] for run in self.brain_route_runs()}:
                raise PanelError(400, 'bad_option', 'Same drives as an earlier brain: choose one of the list (a brain '
                                                    'run that kept its drives, routes.json).')
            found['NNSLR_BRAIN_ROUTES_FROM'] = routes_from
        return found

    def brain_route_runs(self) -> list[dict]:
        """The earlier brain runs a training can take the drives of (Train › Brain's Drives, scripts/run_brain.sh's
        NNSLR_BRAIN_ROUTES_FROM), newest first: one per run NAME, its seeds runs/NAME-s0, -s1, -s2 counted as that run,
        each with a routes.json (brain_train.ROUTES, the drives it took; the brains before 12 Nov 2026 kept none and are
        not here): {run: NAME, drives: how many, at: when its routes.json was written}."""
        runs = self.root / 'runs'
        found: dict[str, dict] = {}
        if not runs.is_dir():
            return []
        for entry in runs.iterdir():
            routes = entry / 'routes.json'
            if not RUN_NAME.fullmatch(entry.name) or not routes.is_file():
                continue
            seed = BRAIN_SEED_RUN.fullmatch(entry.name)
            name = seed.group(1) if seed else entry.name
            try:
                drives = json.loads(routes.read_text(encoding='utf-8'))
                at = routes.stat().st_mtime
            except (OSError, ValueError):
                continue
            if not isinstance(drives, list) or not drives:
                continue
            # The run's own routes.json first (the one scripts/run_brain.sh reads), else a seed's.
            if name not in found or entry.name == name:
                found[name] = {'run': name, 'drives': len(drives), 'at': at}
        return sorted(found.values(), key=lambda run: run['at'], reverse=True)

    def _brain_sign_networks(self, options: dict, check_only: bool = False) -> dict:
        """The variables of the networks that read the brain's signs (BRAIN_SIGN_NETWORKS): a finished version of
        Models for each one chosen (a SignNet the comma runs, a classifier, an applicability network that is not a smoke
        run), nothing for one not chosen (scripts/run_brain.sh takes the one in use). Refused when no applicability
        network is in use nor chosen: the brain would learn from the signs of other roads, which the car leaves out.
        check_only: a pipeline only checked, whose trainings before the brain make some of them (PIPELINE_NETWORK)."""
        from nnslr_tools.preannotate import networks_in_use
        found = {}
        for key, (role, variable) in BRAIN_SIGN_NETWORKS.items():
            chosen = options.get(key)
            if chosen is None:
                continue
            if check_only and chosen == PIPELINE_NETWORK:
                found[variable] = chosen
                continue
            entry = next((entry for entry in self.models_index.artifacts((role,))[role] if entry['id'] == chosen), None)
            if entry is None:
                raise PanelError(400, 'bad_option', 'Unknown network to read the signs with: choose one of Models.')
            if not entry['finished']:
                raise PanelError(400, 'bad_option', 'That network has not finished its training: choose a finished one.')
            if role == 'cutter' and (entry.get('kind') != 'signnet' or not entry.get('fits_the_car')):
                raise PanelError(400, 'bad_option', 'The comma reads its signs with a SignNet of its band and grid.')
            if role == 'classifier' and entry.get('part') != 'classifier':
                raise PanelError(400, 'bad_option', 'The comma reads its signs with a classifier with “not a sign”.')
            if role == 'decider' and entry.get('smoke_limit'):
                raise PanelError(400, 'bad_option', 'A smoke run learnt on a few frames: choose a whole applicability '
                                                    'training.')
            found[variable] = chosen
        used = networks_in_use(self.root)
        missing = [word for key, word in (('cutter', 'cutter'), ('classifier', 'classifier'),
                                          ('decider', 'applicability network'))
                   if BRAIN_SIGN_NETWORKS['sign_' + key][1] not in found and key not in used]
        if missing:
            raise PanelError(409, 'no_sign_networks', 'The brain reads its signs as the car does: no '
                             + ', '.join(missing) + ' is in use or chosen (Models).')
        return found

    def _network_start(self, action: str, start: object) -> dict:
        """The environment of where the cutter's, the applicability's, the brain's or UserNet's training starts (one of
        NETWORK_STARTS): nothing for its default (NETWORK_FRESH), or the variable of a finished version of its role in
        Models given by its id: the signnet-best.pt of a SignNet of the comma's band and grid, the experiment-b folder of
        an applicability run, the brain-last.pt of a brain (one that took the driver's pedals and speed too: the new
        one leaves those inputs out; its script's dry run checks the rest: stepped as the new one, on the same split),
        the usernet-best.pt of a UserNet (its training checks that it is the same network)."""
        if start is None or start == NETWORK_FRESH:
            return {}
        if not isinstance(start, str):
            raise PanelError(400, 'bad_option', 'The start is a string: fresh, or the id of a version of Models.')
        role, variable = NETWORK_STARTS[action]
        entry = next((entry for entry in self.models_index.artifacts((role,))[role] if entry['id'] == start), None)
        if entry is None or not RUN_NAME.fullmatch(entry['run']):
            raise PanelError(400, 'bad_option', 'Unknown network to start from: choose one of Models.')
        if not entry['finished']:
            raise PanelError(400, 'bad_option', 'That network has not finished its training: choose a finished one.')
        if entry.get('smoke_limit'):
            raise PanelError(400, 'bad_option', 'A smoke run learnt on a few frames: choose a whole training.')
        run = self.root / 'runs' / entry['run']
        if role == 'cutter':
            if entry.get('kind') != 'signnet' or not entry.get('fits_the_car'):
                raise PanelError(400, 'bad_option', 'Only a SignNet of the comma’s band and grid can be a start.')
            from nnslr_tools.chain import SIGNNET_FILE
            return {variable: str(run / SIGNNET_FILE)}
        if role == 'brain':
            return {variable: str(run / BRAIN_WEIGHTS)}
        if role == 'users':
            return {variable: str(run / USERNET_WEIGHTS)}
        return {variable: str(run / entry['part'].split('/')[0])}

    def start(self, action: str, options: dict | None = None, *, check_only: bool = False) -> dict | None:
        """Starts an action (check_only: only checks it and its options, for a pipeline, and returns None). The only options are the starting point of a training ({'start': key} of
        training_starts()), turned into a flag or the environment of helpers/3_train.sh, the run to resume, and the
        cutter's families (one of CUTTER_FAMILIES), the route, provider, save and limit of an AI
        action (_ai_arguments), whether the classifier adds the Mapillary signs ('on', 'off', or 'pretrain': first
        on them alone, then on ours from those weights) and where it starts ({'start': 'imagenet' or a classifier's
        id}, _classifier_start) and the cutter it learns the false boxes of ({'cutter': a SignNet's id}, else the one
        in use), and the networks and the drive of a replay ({'set', 'route' or 'certified'}, _simulate_arguments): never
        a command.
        The classifier's and the cutter's trainings run on a new run named here (new_run_name)."""
        if action not in ACTIONS:
            raise PanelError(404, 'unknown_action', 'Unknown action.')
        start_env = {}
        start_args = []
        if (options or {}).get('size') is not None:
            # The cutter is trained at the car's band only (CUTTER_BAND): no action takes a size any more.
            raise PanelError(400, 'bad_option', 'The cutter is trained at the comma’s band, ' + CUTTER_BAND.replace('x', ' × ')
                             + ': no size to choose.')
        if (options or {}).get('families') is not None and 'families' not in ACTION_OPTIONS.get(action, ()):
            raise PanelError(400, 'bad_option', 'Only "Train the cutter" takes a choice of signs.')
        # [ai-reviewer] - START
        for key in AI_OPTIONS:
            value = (options or {}).get(key)
            if value is not None and key not in ACTION_OPTIONS.get(action, ()):
                raise PanelError(400, 'bad_option', '"' + ACTIONS[action]['label'] + '" does not take a ' + key + '.')
            if value is not None and not isinstance(value, str):
                raise PanelError(400, 'bad_option', 'The ' + key + ' is a string.')
        if action in AI_ACTIONS:
            start_args += self._ai_arguments(action, options or {})
        # [ai-reviewer] - END
        # Fit box's refiner, checked before any script is looked for: refused on every other action.
        for key in BOX_FIT_OPTIONS:
            value = (options or {}).get(key)
            if value is not None and key not in ACTION_OPTIONS.get(action, ()):
                raise PanelError(400, 'bad_option', '"' + ACTIONS[action]['label'] + '" does not take ' + key + '.')
            if value is not None and not isinstance(value, str):
                raise PanelError(400, 'bad_option', 'The ' + key + ' is a string.')
        for key in ROAD_LABELER_OPTIONS:
            value = (options or {}).get(key)
            if value is not None and key not in ACTION_OPTIONS.get(action, ()):
                raise PanelError(400, 'bad_option', '"' + ACTIONS[action]['label'] + '" does not take ' + key + '.')
            if value is not None and not isinstance(value, str):
                raise PanelError(400, 'bad_option', 'The ' + key + ' is a string.')
        if action in ROAD_LABELER_ACTIONS:
            if not (self.repo / ROAD_LABELER_SCRIPT).is_file():
                raise PanelError(403, 'actions_disabled', ROAD_LABELER_SCRIPT + ' is missing in this installation.')
            start_args += self._road_labeler_arguments(action, options or {})
        if action == 'box_fit_train':
            if not (self.repo / BOX_FIT_SCRIPT).is_file():
                raise PanelError(403, 'actions_disabled', BOX_FIT_SCRIPT + ' is missing in this installation.')
            start_args += self._box_fit_arguments(options or {})
        # The set of networks and the certified drives of a replay: refused on every other action (its drive is the
        # 'route' checked above).
        for key in SIMULATE_OPTIONS:
            value = (options or {}).get(key)
            if value is not None and key not in ACTION_OPTIONS.get(action, ()):
                raise PanelError(400, 'bad_option', '"' + ACTIONS[action]['label'] + '" does not take ' + key + '.')
            if value is not None and not isinstance(value, str):
                raise PanelError(400, 'bad_option', 'The ' + key + ' is a string.')
        if action == 'simulate':
            if not (self.repo / SIMULATE_SCRIPT).is_file():
                raise PanelError(403, 'actions_disabled', SIMULATE_SCRIPT + ' is missing in this installation.')
            start_args += self._simulate_arguments(options or {})
        # [ai-reviewer] - START
        if action == 'fetch':
            # The address set in the comma panel of the top bar; without one the script's ssh host "comma".
            from nnslr_tools.comma_link import configured
            if configured(self.root):
                start_args.append('--host=' + configured(self.root))
        # [ai-reviewer] - END
        if action == 'prepare_again' and (options or {}).get('route') is not None:
            start_args += self._again_routes(options['route'])
        proposer = (options or {}).get('proposer')
        if proposer is None and action in PREPARING:
            # Started without a choice (the Overview's next step, Try again of an old job): the one of the Prepare
            # tab, never the helpers' own default behind it.
            proposer = self.proposers()['chosen']
        if proposer is not None:
            if action not in ('prepare', 'prepare_again', 'measure_proposer'):
                raise PanelError(400, 'bad_option', 'Only preparing routes takes a proposer.')
            if proposer not in PROPOSERS:
                raise PanelError(400, 'bad_option', 'Unknown proposer: ' + ', '.join(PROPOSERS) + '.')
            if proposer == 'ours':
                from nnslr_tools.preannotate import ours_missing
                missing = ours_missing(self.root)
                if missing:
                    raise PanelError(409, 'not_available', missing + '.')
            if proposer == 'qwen':
                from nnslr_tools.preannotate import vlm_missing
                missing = vlm_missing(self.root)
                if missing:
                    raise PanelError(409, 'not_available', missing + '.')
                # Qwen3-VL needs about 17.5 GB of the GPU: refused, not run out of memory, when they are taken
                # (another Qwen job, a training, llama.cpp in another container) on the card with the most free memory,
                # the one it would take.
                taken = _qwen_gpu_reason(_best_card(_cards_of(self.system())))
                if taken:
                    raise PanelError(409, 'gpu_taken', taken + '.')
            start_env['NNSLR_PROPOSER'] = proposer
        if action == 'train_cutter':
            families = (options or {}).get('families') or CUTTER_FAMILY
            if families not in CUTTER_FAMILIES:
                raise PanelError(400, 'bad_option', 'Unknown signs for the cutter: all or speed.')
            start_env['NNSLR_CUTTER_FAMILIES'] = families
        mapillary = (options or {}).get('mapillary')
        if mapillary is not None:
            if action not in MAPILLARY_ENV:
                raise PanelError(400, 'bad_option', 'Only "Train the cutter" and "Train the classifier" take the '
                                 'Mapillary signs.')
            # 'pretrain': two stages, the Mapillary crops (the cutter's windows) alone, then ours from those weights.
            if mapillary not in ('on', 'off', 'pretrain'):
                raise PanelError(400, 'bad_option', 'Mapillary is off, on, or pretrain (Mapillary first, then ours).')
            if mapillary in ('on', 'pretrain'):
                if action == 'train_cutter':
                    from nnslr_tools.mtsd_windows import windows_state as mapillary_state
                else:
                    from nnslr_tools.mapillary import crops_state as mapillary_state
                state = mapillary_state(self.root)
                if not state['available']:
                    raise PanelError(409, 'not_available', state['reason'][0].upper() + state['reason'][1:] + '.')
                start_env[MAPILLARY_ENV[action]] = '1' if mapillary == 'on' else 'pretrain'
        start_env.update(self._mapillary_choice(action, options or {}, mapillary in ('on', 'pretrain')))
        if action == 'train_classifier':
            start_env.update(self._classifier_start((options or {}).get('start')))
        if action in NETWORK_STARTS:
            start_env.update(self._network_start(action, (options or {}).get('start')))
        for key in ('works', 'road', 'types', 'driver', 'pool', 'seeds', 'users', 'routes_from', *BRAIN_SIGN_NETWORKS):
            if (options or {}).get(key) is not None and action != 'train_brain':
                raise PanelError(400, 'bad_option', 'Only "Train the brain" takes ' + key + '.')
        if action == 'train_brain':
            start_env.update(self._brain_choices(options or {}, check_only))
        stage = (options or {}).get('stage')
        if stage is not None and action != 'train_users':
            raise PanelError(400, 'bad_option', 'Only "Train UserNet" takes a stage.')
        if action == 'train_users':
            stage = USERNET_STAGES[0] if stage is None else stage
            if stage not in USERNET_STAGES:
                raise PanelError(400, 'bad_option', 'The stage of UserNet is bdd (BDD100K) or drives (your drives).')
            if stage == 'bdd' and not (Path(os.environ.get('NNSLR_BDD_DIR') or BDD_DIR) / 'samples.json').is_file():
                raise PanelError(409, 'not_available', 'BDD100K is not on this server (its samples.json): train on '
                                 'your drives, or put it back first.')
            start_env['NNSLR_USERNET_STAGE'] = stage
        for key in ('openpilot', 'brains'):
            if (options or {}).get(key) is not None and action != 'regression':
                raise PanelError(400, 'bad_option', 'Only "Run the regression" takes ' + key + '.')
        if action == 'regression':
            start_env.update(self._regression_choices(options or {}))
        cutter_choice = (options or {}).get('cutter')
        if cutter_choice is not None and action != 'train_classifier':
            raise PanelError(400, 'bad_option', 'Only "Train the classifier" takes a cutter.')
        if cutter_choice is not None and not (check_only and cutter_choice == PIPELINE_NETWORK):
            start_env['NNSLR_CUTTER'] = self._classifier_cutter(cutter_choice)
        # The settings of the training changed in Train (nnslr_tools.train_settings): the variables its script reads.
        from nnslr_tools import train_settings
        start_env.update(train_settings.environment(self.root, action))
        if action == 'train_cutter' and mapillary != 'on':
            # The cap of the windows added to ours (Train › Cutter › Settings): scripts/run_signnet.sh refuses it
            # without them, or with Mapillary first.
            start_env.pop('NNSLR_CUTTER_MAPILLARY_RATIO', None)
        package = (options or {}).get('package')
        if package is not None and action != 'networks_publish':
            raise PanelError(400, 'bad_option', 'Only "Publish on Hugging Face" takes a package.')
        if action == 'networks_publish':
            from nnslr_tools.networks_package import packages
            if not isinstance(package, str) or package not in {item['name'] for item in packages(self.root)}:
                raise PanelError(400, 'bad_option', 'Unknown package: prepare it first.')
            from nnslr_tools import hf_account
            if not hf_account.public_state(self.root)['connected']:
                raise PanelError(409, 'not_available', 'Connect Hugging Face first: Settings, Hugging Face.')
            start_args.append('--name=' + package)
            # The token kept in Settings, in the job's environment only (never its command line or log); without
            # it the script uses this server's own Hugging Face login.
            kept = hf_account.token(self.root)
            if kept:
                start_env['HF_TOKEN'] = kept
        # "Build again": the build also when nothing changed. Only 'on' reaches the command line.
        rebuild = (options or {}).get('rebuild')
        if rebuild not in (None, 'on') or (rebuild and action != 'send_to_car'):
            raise PanelError(400, 'bad_option', 'Only "Publish to the car" builds again, with on.')
        signnet = (options or {}).get('signnet')
        if signnet is not None and action not in SIGNNET_ACTIONS:
            raise PanelError(400, 'bad_option', 'Only "Publish to the car" and "Try on the comma now" take a SignNet.')
        if action in SIGNNET_ACTIONS:
            # A finished SignNet of the Models tab, by its run name: nothing else reaches the command line.
            wanted = f'cutter:{signnet}/{SIGNNET_PART}' if isinstance(signnet, str) and RUN_NAME.fullmatch(signnet) else None
            entry = next((entry for entry in self.models_index.artifacts(('cutter',))['cutter']
                          if wanted is not None and entry['id'] == wanted and entry['finished']), None)
            if entry is None:
                raise PanelError(400, 'bad_option', 'Unknown SignNet: choose a finished one on the Models tab.')
            if not entry.get('fits_the_car'):
                # Another band or stride: the car's vision.py cannot run it (car_release.choose would fail later).
                raise PanelError(409, 'not_for_the_car', NOT_SELECTABLE['not_for_the_car'][1])
            script = CAR_SCRIPT if action == 'send_to_car' else TRY_SCRIPT
            if not (self.repo / script).is_file():
                raise PanelError(403, 'actions_disabled', script + ' is missing in this installation.')
            # Only Publish puts the brain in use on the car: a trial copies the sign networks alone.
            missing = ((car_missing if action == 'send_to_car' else trial_missing)(self.repo)
                       or self.models_index.in_use('classifier')[1]
                       or (brain_refused(self.models_index) if action == 'send_to_car' else ''))
            if missing:
                raise PanelError(409, 'not_available', missing[0].upper() + missing[1:] + '.')
            start_args.append('--signnet=' + signnet)
            if rebuild:
                start_args.append('--rebuild')
        if action == 'networks_prepare':
            from nnslr_tools.networks_package import new_name
            from nnslr_tools.preannotate import ours_missing
            missing = ours_missing(self.root)
            if missing:
                raise PanelError(409, 'not_available', missing + '.')
            start_args.append('--name=' + new_name())
        resumed = (options or {}).get('run')
        if action == 'train_resume':
            if not isinstance(resumed, str) or not RUN_NAME.fullmatch(resumed) or not (self.root / 'runs' / resumed).is_dir():
                raise PanelError(400, 'bad_option', 'Unknown run to resume.')
            start_args.append('--resume=' + resumed)
        elif resumed is not None:
            raise PanelError(400, 'bad_option', 'Only "Resume the training" takes a run.')
        start = (options or {}).get('start')
        if start is not None and action != 'train_classifier' and action not in NETWORK_STARTS:
            if action != 'train':
                raise PanelError(400, 'bad_option', 'Only a training has a starting point.')
            choice = next((item for item in self.training_starts() if item['key'] == start), None)
            if choice is None:
                raise PanelError(400, 'bad_option', 'Unknown starting point.')
            if not choice['available']:
                raise PanelError(409, 'not_available', choice['reason'] + '.')
            if start == 'random':
                start_env['NNSLR_TRAIN_INIT'] = 'random'
            elif start == 'stable':
                start_args.append('--continue')
            elif start.startswith('run:'):
                start_env['NNSLR_TRAIN_FROM_RUN'] = start[len('run:'):]
        if not self.actions_enabled or not self.available():
            raise PanelError(403, 'actions_disabled', 'The panel is read-only on this server.')
        group = ACTIONS[action]['group']
        if action == 'update' and not (self.repo / 'helpers/update.sh').is_file():
            raise PanelError(403, 'actions_disabled', 'helpers/update.sh is missing in this installation.')
        fetch_python = None
        if action == 'fetch':
            if not (self.repo / 'helpers/fetch_comma_segments.sh').is_file():
                raise PanelError(403, 'actions_disabled', 'helpers/fetch_comma_segments.sh is missing in this installation.')
            fetch_python = self.fetch_python()
            if fetch_python is None:
                raise PanelError(403, 'actions_disabled', 'The reader of the comma logs is missing (see docs/GUIDE.md).')
        if action == 'train_cutter' and not (self.repo / CUTTER_SCRIPT).is_file():
            raise PanelError(403, 'actions_disabled', CUTTER_SCRIPT + ' is missing in this installation.')
        if action == 'train_decider' and not (self.repo / DECIDER_SCRIPT).is_file():
            raise PanelError(403, 'actions_disabled', DECIDER_SCRIPT + ' is missing in this installation.')
        if action == 'train_brain' and not (self.repo / BRAIN_SCRIPT).is_file():
            raise PanelError(403, 'actions_disabled', BRAIN_SCRIPT + ' is missing in this installation.')
        if action == 'train_users' and not (self.repo / USERNET_SCRIPT).is_file():
            raise PanelError(403, 'actions_disabled', USERNET_SCRIPT + ' is missing in this installation.')
        if action == 'train_brain' and (missing := self._brain_drives_missing()):
            # Not only the button: a POST of its own or a pipeline's Brain step (checked when the pipeline starts) is
            # refused the same way while there is no dashcam drive to learn from.
            raise PanelError(409, 'not_available', missing[0].upper() + missing[1:] + '.')
        if action == 'regression' and not (self.repo / REGRESSION_SCRIPT).is_file():
            raise PanelError(403, 'actions_disabled', REGRESSION_SCRIPT + ' is missing in this installation.')
        if action in ('networks_prepare', 'networks_publish') and not (self.repo / NETWORKS_SCRIPT).is_file():
            raise PanelError(403, 'actions_disabled', NETWORKS_SCRIPT + ' is missing in this installation.')
        if action == 'road_types' and not (self.repo / ROAD_TYPES_SCRIPT).is_file():
            raise PanelError(403, 'actions_disabled', ROAD_TYPES_SCRIPT + ' is missing in this installation.')
        if action == 'mapillary_windows' and not (self.repo / WINDOWS_SCRIPT).is_file():
            raise PanelError(403, 'actions_disabled', WINDOWS_SCRIPT + ' is missing in this installation.')
        if action == 'mapillary_locations':
            missing = self._locations_missing()
            if missing:
                raise PanelError(409 if (self.repo / LOCATIONS_SCRIPT).is_file() else 403,
                                 'not_available' if (self.repo / LOCATIONS_SCRIPT).is_file() else 'actions_disabled',
                                 missing[0].upper() + missing[1:] + '.')
        # [ai-reviewer] - START
        if action in AI_ACTIONS and not (self.repo / ACTIONS[action]['command'][0]).is_file():
            raise PanelError(403, 'actions_disabled', ACTIONS[action]['command'][0] + ' is missing in this installation.')
        # [ai-reviewer] - END
        if action == 'train_classifier':
            if not (self.repo / CLASSIFIER_SCRIPT).is_file():
                raise PanelError(403, 'actions_disabled', CLASSIFIER_SCRIPT + ' is missing in this installation.')
            cutter, why = self.models_index.in_use('cutter')
            if cutter is None and cutter_choice is None:
                raise PanelError(409, 'no_cutter', why[0].upper() + why[1:] + '.')
        if check_only:
            return None
        # The card: a training (EXCLUSIVE_GPU, not a dry run on the CPU) takes a free one of its own, a preparation on
        # the GPU the one with the most free memory. nvidia-smi is read now, outside the lock (it can take a second);
        # the card is chosen and recorded under it, so two starts never take the same one.
        dry = action in DRY_ON_CPU and '--dry-run' in start_args
        takes_card = (action in EXCLUSIVE_GPU and not dry) or action in SHARED_GPU or (
            action in PREPARING and (proposer is None or proposer in GPU_PROPOSERS))
        cards = _cards_of(self.system()) if takes_card else []
        with self._lock:
            activities = self.activities()
            conflict = _conflicting_group(group, self._running_groups(activities))
            if conflict is not None:
                raise PanelError(409, 'busy', self._busy_reason(conflict, activities) + ': wait for it to finish.')
            if group in PAIRED_GROUPS and (clash := self._train_clash(action, activities)):
                raise PanelError(409, 'busy', clash + ': wait for it to finish.')
            gpus = self._take_card(action, proposer, cards, activities) if takes_card else []
            self.logs.mkdir(parents=True, exist_ok=True)
            job_id = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + action
            path = self.logs / (job_id + '.log')
            if path.exists():
                raise PanelError(409, 'busy', 'An action has just started: try again in a second.')
            command = list(ACTIONS[action]['command']) + start_args
            if action == 'prepare':
                command.append('--jobs=' + os.environ.get(PREPARE_JOBS_ENV, '2'))
            if action == 'train_classifier':
                command.append(self.new_run_name('classifier'))
            if action == 'train_decider':
                command.append(self.new_run_name('decider'))
            if action == 'train_brain':
                command.append(self.new_run_name('brain'))
            if action == 'train_users':
                command.append(self.new_run_name('users'))
            if action == 'train_cutter':
                # NAME: one SignNet, at the car's band, on a run of its own.
                command.append(self.new_run_name('cutter'))
            # Temporary files in the data root, not in /tmp: the service has a private /tmp (PrivateTmp)
            # that systemd empties at every restart, which broke a pip download of a running training.
            temporary = self.root / 'tmp'
            temporary.mkdir(parents=True, exist_ok=True)
            environment = dict(os.environ, NNSLR_DATA_ROOT=str(self.root), TMPDIR=str(temporary), **start_env)
            if fetch_python:
                environment[FETCH_PYTHON_ENV] = fetch_python
            # Its own progress file (nnslr_tools.progress): two jobs at once never read each other's.
            environment['NNSLR_PROGRESS_FILE'] = str(self.root / 'logs' / f'progress-{job_id}.json')
            # Its record, where the brain writes the cards it borrows for its readings of the signs (borrow_cards).
            environment['NNSLR_JOB_RECORD'] = str(path.with_suffix('.json'))
            if gpus:
                # Its card, by nvidia-smi's index: the helpers and scripts check that one (NNSLR_GPU_INDEX), CUDA
                # counts the cards in the same order (by PCI bus) and sees only it.
                environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', NNSLR_GPU_INDEX=str(gpus[0]),
                                   CUDA_VISIBLE_DEVICES=str(gpus[0]))
            if action in SHARED_TRAINING_MB and 'NNSLR_NUM_WORKERS' not in environment:
                # The processes that read its data: the cores shared among the trainings that run with it (half the
                # cores each for two, as the scripts' default; a third one a third of them), 4 to 16.
                trainings = 1 + sum(1 for holder in self._gpu_holders(activities)
                                    if holder.get('action') in EXCLUSIVE_GPU)
                environment['NNSLR_NUM_WORKERS'] = str(min(16, max(4, (os.cpu_count() or 8) // max(2, trainings))))
            # The record of the job (<job id>.json): its cards (what holds which, _job_gpus), that it has its own
            # progress file, and who proposes, for the page (the running job's choice) and Try again: the log does
            # not always say. Our networks: the parts in use now, by name (the Models tab can change them while it
            # runs).
            kept = {**(_read_json(path.with_suffix('.json')) or {}), 'gpus': gpus, 'own_progress': True}
            if proposer is not None:
                kept.update({'proposer': proposer, **({'networks': _active_parts(self.root)} if proposer == 'ours' else {})})
            path.with_suffix('.json').write_text(json.dumps(kept) + '\n', encoding='utf-8')
            with path.open('wb') as log:
                log.write(('== ' + ' '.join(command) + '\n').encode())
                log.flush()
                # A new session: the action outlives this server and can be stopped as a group. At a lower CPU
                # priority (nice runs bash in its place, same pid): a job on every core leaves this site answering.
                process = subprocess.Popen(['nice', '-n', str(JOB_NICE), 'bash', '-c', WRAPPER, 'panel', *command],
                                           cwd=self.repo, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           env=environment, start_new_session=True)
            self._processes[job_id] = process
            # The process group of the action (a new session: its pid), so Ferma works also after a
            # restart of this server, which forgets self._processes.
            path.with_suffix('.pid').write_text(str(process.pid) + '\n')
        # Reaps the process as soon as it ends, so it never stays a zombie, and starts the next training of a pipeline.
        threading.Thread(target=self._reap, args=(process,), daemon=True).start()
        self._prune_progress()
        return {'id': job_id, 'action': action, 'label': ACTIONS[action]['label']}

    def _take_card(self, action: str, proposer: str | None, cards: list[dict], activities: list[dict]) -> list[int] | None:
        """The card a job starts on (called under the lock): a preparation the one with the most free memory (Qwen3-VL
        refused there without QWEN_GPU_MB free), shared with what else runs; a replay (SHARED_GPU) the same, refused
        without the memory it needs; any other the free one of lowest index, refused 'busy' with what holds them when
        none is free. Without nvidia-smi the cards are not known: None (no card is named to the job: a replay's helper
        then takes the freest card itself, or stops), refused while a training holds the GPU."""
        holders = self._gpu_holders(activities)
        states = _card_states(cards, holders)
        if action in SHARED_GPU:
            best = _best_card(states)
            if best is None:
                return None
            room = _room_reason(best, SHARED_GPU[action], 'A replay')
            if room:
                raise PanelError(409, 'gpu_taken', room + ': free some memory on a card, then try again.')
            return [best['index']]
        if action in PREPARING:
            best = _best_card(states)
            if best is None:
                return None
            taken = _qwen_gpu_reason(best) if proposer == 'qwen' else ''
            if taken:
                raise PanelError(409, 'gpu_taken', taken + '.')
            return [best['index']]
        reason = _gpu_busy(states, holders)
        if reason and (shared := _shared_card(states, action)) is not None:
            return [shared['index']]
        if reason:
            raise PanelError(409, 'busy', reason[0].upper() + reason[1:] + ': wait for it to finish.')
        free = [card for card in states if card['free']]
        return [free[0]['index']] if free else None

    def _prune_progress(self) -> None:
        """The progress files of the jobs that left the history (the newest HISTORY logs) and no longer run are
        removed: nothing reads them any more."""
        names = sorted((path.stem for path in self.logs.glob('*.log') if JOB_ID.fullmatch(path.stem)), reverse=True)
        kept = set(names[:HISTORY]) | set(self._jobs_now())
        for path in (self.root / 'logs').glob('progress-*.json'):
            job_id = path.stem[len('progress-'):]
            if JOB_ID.fullmatch(job_id) and job_id not in kept:
                try:
                    path.unlink()
                except OSError:
                    pass

    def train_settings(self) -> dict:
        """The settings of the trainings of Train (nnslr_tools.train_settings), and whether this panel can change
        them."""
        from nnslr_tools import train_settings
        return {**train_settings.view(self.root), 'can_edit': self.actions_enabled and self.available()}

    def save_train_settings(self, body, who: str) -> dict:
        """Train › Settings: {"training": action, "values": {setting: number or null}} saved for the next training
        of that kind; a read-only panel refuses it, as it refuses the actions."""
        from nnslr_tools import train_settings
        if not self.actions_enabled or not self.available():
            raise PanelError(403, 'actions_disabled', 'The panel is read-only on this server.')
        try:
            record = train_settings.save(self.root, body, who)
        except train_settings.SettingsError as exc:
            raise PanelError(400, 'invalid_settings', str(exc)) from None
        print(f'Panel: {train_settings.TRAININGS[body["training"]]["label"].lower()} training settings '
              f'{record["values"] or "all default"} ({who!r})', flush=True)
        return self.train_settings()

    # -- Train › Pipeline -----------------------------------------------------------------------
    def start_pipeline(self, body, who: str | None) -> dict:
        """Train › Pipeline: {"steps": [{"action", "options"}]}, some of PIPELINE_ORDER in that order, each with the
        options of its own button; all are checked now, the first starts now, the others one after the other
        (advance_pipeline). Refused while another pipeline runs."""
        if not self.actions_enabled or not self.available():
            raise PanelError(403, 'actions_disabled', 'The panel is read-only on this server.')
        steps = body.get('steps') if isinstance(body, dict) else None
        if not isinstance(steps, list) or not steps:
            raise PanelError(400, 'bad_option', 'Choose at least one training.')
        chosen = []
        for step in steps:
            action = step.get('action') if isinstance(step, dict) else None
            if action not in PIPELINE_ORDER:
                raise PanelError(400, 'bad_option', 'Unknown training in the pipeline.')
            options = step.get('options') or {}
            if not isinstance(options, dict) or not all(isinstance(value, str) for value in options.values()):
                raise PanelError(400, 'bad_option', 'The options of a training are strings.')
            unknown = set(options) - set(ACTION_OPTIONS.get(action, ())) | ({'cutter'} & set(options))
            if unknown:
                raise PanelError(400, 'bad_option', '"' + ACTIONS[action]['label'] + '" in a pipeline does not take '
                                 + ', '.join(sorted(unknown)) + '.')
            chosen.append({'action': action, 'options': options, 'job': None})
        actions = [step['action'] for step in chosen]
        if len(set(actions)) != len(actions) or actions != sorted(actions, key=PIPELINE_ORDER.index):
            raise PanelError(400, 'bad_option', 'Each training at most once, in the order of the pipeline.')
        for step in chosen:
            options = dict(step['options'])
            if step['action'] == 'train_classifier' and 'train_cutter' in actions:
                options['cutter'] = PIPELINE_NETWORK
            if step['action'] == 'train_brain':
                # Its signs read with the networks the pipeline trains before it (advance_pipeline names them then).
                made = {key for key, (role, _) in BRAIN_SIGN_NETWORKS.items()
                        if any(PIPELINE_ROLES.get(action) == role for action in actions)}
                if made & set(options):
                    raise PanelError(400, 'bad_option', 'In a pipeline the brain reads its signs with the networks '
                                     'trained before it: reload the page.')
                options.update(dict.fromkeys(made, PIPELINE_NETWORK))
            # Now, not hours later: a Mapillary not prepared, a start that is gone...
            self.start(step['action'], options, check_only=True)
        with self._lock:
            plan = _read_json(self.logs / PIPELINE_FILE)
            if plan is not None and plan.get('state') == 'running':
                raise PanelError(409, 'busy', 'A pipeline is running: wait for it to finish or stop it.')
            job = self.start(chosen[0]['action'], chosen[0]['options'] or None)
            chosen[0]['job'] = job['id']
            self._write_pipeline({'state': 'running', 'started_at': _now(), 'ended_at': None, 'by': who,
                                  'reason': '', 'steps': chosen})
        print(f'Panel: pipeline started, {", ".join(actions)} ({who!r})', flush=True)
        self._watch_pipeline()
        return self.pipeline_view()

    def stop_pipeline(self) -> dict:
        """Train › Pipeline, Stop after this one: the training at work goes on to its end (its own Stop stops it), the
        next ones do not start."""
        if not self.actions_enabled or not self.available():
            raise PanelError(403, 'actions_disabled', 'The panel is read-only on this server.')
        with self._lock:
            plan = _read_json(self.logs / PIPELINE_FILE)
            if plan is None or plan.get('state') != 'running':
                raise PanelError(409, 'not_running', 'No pipeline is running.')
            plan.update(state='stopped', ended_at=_now(), reason='Stopped by hand: the next trainings did not start.')
            self._write_pipeline(plan)
        return self.pipeline_view()

    def advance_pipeline(self) -> None:
        """Starts the next training of the running pipeline when the one at work ended well; a training that failed
        or was stopped ends the pipeline. While the train group or every GPU is busy with something else it waits (the
        reason says so) and tries again later."""
        with self._lock:
            plan = _read_json(self.logs / PIPELINE_FILE)
            if plan is None or plan.get('state') != 'running':
                return
            steps = plan['steps']
            started = [step for step in steps if step.get('job')]
            if started:
                last = started[-1]
                state = self._job_state(last['job'])
                if state == 'running':
                    return
                if state != 'ok':
                    plan.update(state='failed' if state == 'failed' else 'stopped', ended_at=_now(),
                                reason=ACTIONS[last['action']]['label'] + (' failed.' if state == 'failed' else ' was stopped.'))
                    self._write_pipeline(plan)
                    return
            following = next((step for step in steps if not step.get('job')), None)
            if following is None:
                plan.update(state='done', ended_at=_now(), reason=self._put_in_use(plan))
                self._write_pipeline(plan)
                return
            options = dict(following['options'])
            cutter = next((step for step in steps if step['action'] == 'train_cutter'), None)
            if following['action'] == 'train_classifier' and cutter is not None:
                run = _run_of(self.logs / (cutter['job'] + '.log'), 'train_cutter')
                options['cutter'] = f'cutter:{run}/{SIGNNET_PART}' if run else ''
            if following['action'] == 'train_brain':
                trained = self._pipeline_trained(steps)
                for key, (role, _) in BRAIN_SIGN_NETWORKS.items():
                    if any(PIPELINE_ROLES.get(step['action']) == role for step in steps):
                        # A training of the pipeline whose run cannot be told: refused by start(), never the one in use.
                        options[key] = trained.get(role, '')
                # Kept in the plan: the applicability network put in use at the end is the one it read with.
                following['options'] = options
            try:
                # A busy GPU (no card free) is start()'s 'busy', as a busy train group: the pipeline waits for it.
                job = self.start(following['action'], options or None)
            except PanelError as error:
                if error.reason == 'busy':
                    plan['reason'] = 'Waiting: ' + error.message.rstrip('.').split(': wait')[0] + '.'
                else:
                    plan.update(state='failed', ended_at=_now(),
                                reason=ACTIONS[following['action']]['label'] + ' could not start: ' + error.message)
                self._write_pipeline(plan)
                return
            following['job'] = job['id']
            plan['reason'] = ''
            self._write_pipeline(plan)

    def _pipeline_trained(self, steps: list[dict]) -> dict[str, str]:
        """The networks a pipeline's trainings that ended well have made, by role, as Models names them (#71): the
        cutter's SignNet, the classifier, the applicability network the brain read its signs with, else of the variant in
        use (PIPELINE_DECIDER_VARIANT when none is), the brain of its one seed or, of three, the one with the smallest
        mean speed error on validation."""
        from nnslr_tools.preannotate import networks_in_use
        from nnslr_tools.runs import EXPERIMENT_DIR
        found = {}
        for step in steps:
            role = PIPELINE_ROLES.get(step['action'])
            if role is None or not step.get('job') or self._job_state(step['job']) != 'ok':
                continue
            run = _run_of(self.logs / (step['job'] + '.log'), step['action'])
            if run is None:
                continue
            if role == 'cutter':
                found[role] = f'cutter:{run}/{SIGNNET_PART}'
            elif role == 'classifier':
                found[role] = f'classifier:{run}/classifier'
            elif role == 'decider':
                read = next((item['options'].get('sign_decider') for item in steps
                             if item['action'] == 'train_brain' and item.get('job')), None)
                in_use = networks_in_use(self.root).get('decider', '')
                variant = in_use.rsplit('/', 1)[-1] if in_use.startswith('decider:') else PIPELINE_DECIDER_VARIANT
                found[role] = read if isinstance(read, str) and read.startswith(f'decider:{run}/') \
                    else f'decider:{run}/{EXPERIMENT_DIR}/{variant}'
            else:
                seeds = [entry for entry in self.models_index.artifacts(('brain',))['brain']
                         if entry['finished'] and re.fullmatch(re.escape(run) + r'-s[0-9]', entry['run'])]
                error = lambda entry: (entry.get('validation') or {}).get('mean_error_kph')  # noqa: E731
                best = min(seeds, key=lambda entry: (error(entry) is None, error(entry) or 0.0, entry['run']),
                           default=None)
                if best is not None:
                    found[role] = best['id']
        return found

    def _put_in_use(self, plan: dict) -> str:
        """A pipeline whose trainings all ended well puts every network they made in use (models/active.json, as
        Models › Use this one; not the car), #71. Its last line: which went in use, and which could not and why."""
        done, refused = [], []
        for role, model_id in self._pipeline_trained(plan['steps']).items():
            name = PIPELINE_WORDS[role] + ' ' + model_id.split(':', 1)[1].split('/', 1)[0] \
                + (' (' + model_id.rsplit('/', 1)[-1] + ')' if role == 'decider' else '')
            try:
                self.models_index.set_active(role, model_id, who=plan.get('by'), via='local')
            except ModelChoiceError as error:
                refused.append(name + ', ' + error.message[0].lower() + error.message[1:].rstrip('.'))
                continue
            done.append(name)
        if done:
            print(f'Panel: the pipeline put in use {", ".join(done)}', flush=True)
        return ' '.join((['Put in use: ' + ', '.join(done) + '.'] if done else [])
                        + (['Not put in use: ' + '; '.join(refused) + '.'] if refused else []))

    def pipeline_view(self) -> dict | None:
        """The last pipeline for the page: {'state': running|done|failed|stopped, 'started_at', 'ended_at', 'by',
        'reason', 'steps': [{'action', 'label', 'options', 'job', 'run', 'state': the job's state, or todo (still to
        start), skipped (never started)}]}; None before the first."""
        plan = _read_json(self.logs / PIPELINE_FILE)
        if plan is None or not isinstance(plan.get('steps'), list):
            return None
        rows = []
        for step in plan['steps']:
            action, job = step.get('action'), step.get('job')
            if action not in ACTIONS:
                continue
            state = self._job_state(job) if job else 'todo' if plan.get('state') == 'running' else 'skipped'
            rows.append({'action': action, 'label': ACTIONS[action]['label'], 'options': step.get('options') or {},
                         'job': job, 'run': _run_of(self.logs / (job + '.log'), action) if job else None, 'state': state})
        return {key: plan.get(key) for key in ('state', 'started_at', 'ended_at', 'by', 'reason')} | {'steps': rows}

    def _job_state(self, job_id: str) -> str:
        """running, ok, failed or interrupted, as jobs() says it."""
        with self._lock:
            process = self._processes.get(job_id)
        if process is not None and process.poll() is None:
            return 'running'
        code = _exit_code(self.logs / (job_id + '.log'))
        if code is not None:
            return 'ok' if code == 0 else 'failed'
        if process is None and self._group(job_id) is not None:
            return 'running'
        return 'interrupted'

    def _write_pipeline(self, plan: dict) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        temporary = self.logs / (PIPELINE_FILE + '.tmp')
        temporary.write_text(json.dumps(plan, indent=1) + '\n', encoding='utf-8')
        temporary.replace(self.logs / PIPELINE_FILE)

    def _reap(self, process: subprocess.Popen) -> None:
        process.wait()
        try:
            self.advance_pipeline()
        except Exception as error:  # a thread of its own: said in the journal, the watch tries again
            print(f'Panel: the pipeline could not go on: {error}', flush=True)

    def _watch_pipeline(self) -> None:
        """A thread that calls advance_pipeline every PIPELINE_WATCH_SECONDS while a pipeline runs."""
        with self._lock:
            if self._pipeline_watch is not None and self._pipeline_watch.is_alive():
                return
            self._pipeline_watch = threading.Thread(target=self._pipeline_loop, daemon=True, name='nnslr-pipeline')
            self._pipeline_watch.start()

    def _pipeline_loop(self) -> None:
        while True:
            time.sleep(PIPELINE_WATCH_SECONDS)
            try:
                self.advance_pipeline()
            except Exception as error:
                print(f'Panel: the pipeline could not go on: {error}', flush=True)
            plan = _read_json(self.logs / PIPELINE_FILE)
            if plan is None or plan.get('state') != 'running':
                return

    def new_run_name(self, kind: str) -> str:
        """The run of a new classifier or cutter training: the time it starts and what it is,
        2026-10-24_1730-classifier, 2026-10-24_1730-cutter; -2, -3... when a run of that name is already there, or was
        handed out already (two trainings started in the same minute, before the first made its run)."""
        base = datetime.now().strftime('%Y-%m-%d_%H%M') + '-' + kind
        name, number = base, 1
        with self._lock:
            while (self.root / 'runs' / name).exists() or (self.root / 'runs' / name).is_symlink() or (
                    name in self._reserved_runs):
                number += 1
                name = f'{base}-{number}'
            self._reserved_runs.add(name)
        return name

    def _group(self, job_id: str) -> int | None:
        """The process group of an action still running, also one started before a restart of this
        server: the pid written at start, if that process is still the wrapper of this action."""
        if not JOB_ID.fullmatch(job_id):
            return None
        with self._lock:
            process = self._processes.get(job_id)
        if process is not None:
            return process.pid if process.poll() is None else None
        return _wrapper_pid(self.logs, job_id)

    def stop(self, job_id: str) -> dict:
        group = self._group(job_id)
        if group is None:
            if JOB_ID.fullmatch(job_id) and (self.logs / (job_id + '.log')).is_file():
                raise PanelError(409, 'not_running', 'The action has already finished.')
            raise PanelError(404, 'unknown_job', 'Not an action of the panel (stop it from its terminal).')
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        return {'id': job_id, 'stopping': True}

    def log(self, job_id: str, offset: int) -> dict:
        if not JOB_ID.fullmatch(job_id):
            raise PanelError(404, 'unknown_job', 'This job is not on the server.')
        path = self.logs / (job_id + '.log')
        if not path.is_file():
            raise PanelError(404, 'unknown_job', 'This job is not on the server.')
        size = path.stat().st_size
        if offset < 0 or offset > size:
            offset = 0
        # The first call shows only the end of a long log.
        if offset == 0 and size > LOG_CHUNK_BYTES:
            offset = size - LOG_CHUNK_BYTES
        with path.open('rb') as source:
            source.seek(offset)
            data = source.read(LOG_CHUNK_BYTES)
        return {'id': job_id, 'text': data.decode('utf-8', 'replace'), 'offset': offset + len(data), 'size': size}


def _conflicting_group(group: str, running: set[str]) -> str | None:
    """The running group that keeps an action of `group` from starting, or None.

    One action per group at a time, and never together with the groups of CONFLICTS. The actions of PAIRED_GROUPS run
    side by side by their pair rules instead (_pair_apart, Panel._train_clash), each on a card of its own.
    """
    for other in ((() if group in PAIRED_GROUPS else (group,)) + tuple(CONFLICTS.get(group, ()))):
        if other in running:
            return other
    return None


def _log_of_runs(root: Path, path: Path, runs: set[str]) -> bool:
    """Whether a log is one of these runs': under runs/<run>/, or the logs/train-<run>.log of helpers/3_train.sh."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    return (len(parts) > 2 and parts[0] == 'runs' and parts[1] in runs) or (
        len(parts) == 2 and parts[0] == 'logs' and parts[1].startswith('train-') and parts[1][len('train-'):-len('.log')]
        in runs)


def _line_count(path: Path) -> int:
    count = 0
    try:
        with path.open('rb') as handle:
            for line in handle:
                if line.strip():
                    count = count + 1
    except OSError:
        return 0
    return count


def _exit_code(path: Path) -> int | None:
    try:
        with path.open('rb') as source:
            source.seek(max(0, path.stat().st_size - 200))
            tail = source.read().decode('utf-8', 'replace')
    except OSError:
        return None
    match = FINISHED.search(tail)
    return int(match.group(1)) if match else None


def _active_parts(root: Path) -> dict[str, str]:
    """The cutter, classifier and decider in use (models/active.json) by id: what Our networks propose with."""
    from nnslr_tools.preannotate import networks_in_use
    return networks_in_use(root)


def _networks_of(root: Path, log: Path) -> dict[str, str]:
    """The networks a preparation of Our networks proposes with: kept next to its log when it started, else (a job
    started before they were kept) the ones in use now."""
    kept = (_read_json(log.with_suffix('.json')) or {}).get('networks')
    return kept if isinstance(kept, dict) else _active_parts(root)


def _proposer_of(log: Path) -> str | None:
    """The proposer of a job: the one kept next to its log when it started (<job id>.json, Panel.start), else the
    one its log names near its top (PROPOSER_LINE); None when neither says (the helpers' default)."""
    options = _read_json(log.with_suffix('.json'))
    if options is not None and options.get('proposer') in PROPOSERS:
        return options['proposer']
    try:
        with log.open('rb') as handle:
            head = handle.read(65536).decode('utf-8', 'replace')
    except OSError:
        return None
    match = PROPOSER_LINE.search(head)
    return match.group(1) if match else None


def _best_proposer(reports: list[dict], left_out: set[str]) -> str | None:
    """The proposer whose latest measure on the test frames reads the most speed signs right (then finds the most,
    then is the fastest), among those not left out (one that cannot run here); None when none is measured."""
    def score(report: dict) -> tuple:
        speed = report.get('speed_signs') or {}
        return (speed.get('read_right_share') or 0.0, speed.get('found_share') or 0.0,
                -(report.get('seconds_per_frame') or 0.0))
    measured = [report for report in reports if report.get('partition') == 'test' and report.get('proposer') in PROPOSERS
                and report['proposer'] not in left_out and (report.get('speed_signs') or {}).get('truth')]
    return max(measured, key=score)['proposer'] if measured else None


def _room_reason(gpu: dict | None, needed_mb: int, who: str) -> str:
    """Why a job that shares a card cannot start on it ('' when it can, or when its memory is not known): it needs
    needed_mb free."""
    free = _free_mb(gpu or {})
    if free < 0 or free >= needed_mb:
        return ''
    return (f'{who} needs about {needed_mb / 1024:g} GB of a GPU and the freest card has only {free / 1024:.1f} GB '
            f'free (GPU {(gpu or {}).get("index", 0)})')


def _qwen_gpu_reason(gpu: dict | None) -> str:
    """Why Qwen3-VL cannot load on the GPU now ('' when it can, or when the GPU is not known): it needs QWEN_GPU_MB
    free (another Qwen job, a training, llama.cpp in another container take it)."""
    used, total = (gpu or {}).get('memory_used_mb'), (gpu or {}).get('memory_total_mb')
    if used is None or not total or total - used >= QWEN_GPU_MB:
        return ''
    return (f'Qwen3-VL needs about {QWEN_GPU_MB // 1024} GB of the GPU and only {max(0, total - used) / 1024:.1f} GB '
            'are free: another job or program is using it')


# English names of the sign families of the dataset (sign_family of objects.jsonl).
FAMILY_LABELS = {'maximum_speed': 'Speed limit', 'variable_display': 'Variable speed limit', 'zone': 'Zone',
                 'cancellation': 'End of limit', 'other_sign': 'Other sign', 'not_a_sign': 'Not a sign',
                 'unreadable': 'Unreadable', 'road_marking': 'Road marking'}
SPEED_FAMILIES = ('maximum_speed', 'variable_display', 'zone', 'cancellation', 'road_marking')


def _dataset_summary(root: Path) -> dict:
    families: dict[str, int] = {}
    per_route: dict[str, dict] = {}
    frames: set[str] = set()
    human = ai = objects = 0
    # The sha256 of the file, as a split records it (dataset_sha256): the split is of this dataset when they match.
    digest = hashlib.sha256()
    with (root / 'annotations' / 'objects.jsonl').open('rb') as source:
        for line in source:
            digest.update(line)
            try:
                item = json.loads(line)
            except ValueError:
                continue
            objects += 1
            family = item.get('sign_family') or 'unknown'
            families[family] = families.get(family, 0) + 1
            by_ai = (item.get('provenance') or {}).get('label_basis') == 'ai_review'
            ai += by_ai
            human += not by_ai
            route = per_route.setdefault(item.get('route_id') or '?', {'frames': set(), 'objects': 0,
                                                                        'speed_signs': 0, 'human': 0, 'ai': 0})
            route['frames'].add(item.get('image_path'))
            frames.add(item.get('image_path'))
            route['objects'] += 1
            route['speed_signs'] += family in SPEED_FAMILIES
            route['ai' if by_ai else 'human'] += 1
    # The split in use: data/splits/latest.json points to it (logs/splits.json is rewritten while a
    # training makes a new one).
    pointer = _read_json(root / 'splits' / 'latest.json') or {}
    split_path = root / str(pointer.get('path', '')) if pointer.get('path') else None
    split = _read_json(split_path) if split_path is not None and split_path.resolve().is_relative_to(root.resolve()) else None
    imported = _read_json(root / 'logs' / 'import.json')
    return {
        'frames': len(frames), 'objects': objects, 'routes': len(per_route), 'human': human, 'ai': ai,
        'families': [{'key': key, 'label': FAMILY_LABELS.get(key, key.replace('_', ' ').capitalize()), 'count': count}
                     for key, count in sorted(families.items(), key=lambda pair: -pair[1])],
        'per_route': [{'route': name, 'frames': len(values['frames']), 'objects': values['objects'],
                       'speed_signs': values['speed_signs'], 'human': values['human'], 'ai': values['ai']}
                      for name, values in sorted(per_route.items())],
        'changed_at': _mtime(root / 'annotations' / 'objects.jsonl'),
        'sha256': digest.hexdigest(),
        'split': None if not split or split_path is None else {'counts': split.get('counts') or {}, 'made_at': _mtime(split_path),
                                         'warnings': split.get('warnings') or [],
                                         # The dataset it was made of, and the proposals a person rejected it leaves out.
                                         'dataset_sha256': split.get('dataset_sha256') if isinstance(split.get('dataset_sha256'), str) else None,
                                         'rejected': len(split['excluded_rejected_ids'])
                                         if isinstance(split.get('excluded_rejected_ids'), list) else 0},
        'last_import': None if not imported else {
            'at': _mtime(root / 'logs' / 'import.json'),
            'new_frames': imported.get('imported_frame_count') or 0,
            'already': imported.get('already_imported_frame_count') or 0,
            # Frames reviewed again after their import replace their old rows, and withdrawn ones leave the
            # dataset; only an import made before that (no replaced_frame_count) left changed frames unapplied.
            'changed_after_import': 0 if 'replaced_frame_count' in imported
            else len(imported.get('changed_after_import') or []),
            'replaced_frames': imported.get('replaced_frame_count') or 0,
            'removed_frames': imported.get('removed_withdrawn_frame_count') or 0,
            'warnings': imported.get('warnings') or []},
    }


def current_split(root: Path) -> str | None:
    """The sha256 of the split a training takes now (splits/latest.json, as nnslr train reads it), None without one."""
    pointer = _read_json(root / 'splits' / 'latest.json') or {}
    sha = pointer.get('sha256')
    return sha if isinstance(sha, str) and SPLIT_SHA.fullmatch(sha) else None


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _wrapper_pid(logs: Path, job_id: str) -> int | None:
    """The pid of a job's wrapper (<job id>.pid) while it still runs that job, None when it ended."""
    try:
        pid = int((logs / (job_id + '.pid')).read_text().strip())
        command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
    except (OSError, ValueError):
        return None
    # The pid may have been reused by another program: it must be the wrapper of this action.
    action = job_id.split('-', 2)[2]
    wanted = [part.encode() for part in ACTIONS.get(action, {}).get('command', [])]
    wrappers = [[b'bash', b'-c', wrapper.encode(), b'panel'] for wrapper in (WRAPPER, *OLD_WRAPPERS)]
    if not wanted or command[:4] not in wrappers or command[4:4 + len(wanted)] != wanted:
        return None
    return pid


# How long borrow_cards waits before it reads the records again: a start of the panel reads them and writes its own
# within this, under its lock.
BORROW_RECHECK_SECONDS = 1.0


def _cards_held(logs: Path, skip: str | None) -> set[int] | None:
    """The cards the running jobs of the panel hold, but the job skip: those of their records ('gpus', and 'borrowed'
    of a brain); None when one holds every card (its cards not known)."""
    held: set[int] = set()
    for path in logs.glob('*.pid') if logs.is_dir() else ():
        job_id = path.stem
        if job_id == skip or not JOB_ID.fullmatch(job_id) or job_id.split('-', 2)[2] not in ACTIONS:
            continue
        record = _read_json(logs / (job_id + '.json'))
        if record is None or _wrapper_pid(logs, job_id) is None:
            continue
        gpus = record.get('gpus', [])
        if gpus is None:
            return None
        held.update(i for i in [*gpus, *record.get('borrowed', [])] if isinstance(i, int))
    return held


def _free_of_others(logs: Path, own: int, skip: str | None) -> list[int]:
    """The cards but own that no compute process is on and no other running job of the panel holds."""
    held = _cards_held(logs, skip)
    if held is None:
        return []
    return [card['index'] for card in _gpu_cards() if card['index'] != own and not card['processes']
            and card['index'] not in held]


def _write_borrowed(record: Path, cards: list[int]) -> None:
    kept = _read_json(record) or {}
    if cards:
        kept['borrowed'] = cards
    else:
        kept.pop('borrowed', None)
    temporary = record.with_name(record.name + '.tmp')
    temporary.write_text(json.dumps(kept) + '\n', encoding='utf-8')
    temporary.replace(record)


def borrow_cards(logs: Path, own: int, record: Path | None, cards: list[int] | None = None) -> list[int]:
    """scripts/run_brain.sh, step 3 (#36): the cards its readings of the signs run on. Without cards, own first and
    every other card no compute process is on and no other running job of the panel holds; with cards (NNSLR_SIGN_GPUS),
    those as given. The ones but own are written to the brain's record of the panel (record, NNSLR_JOB_RECORD; None
    from a terminal: nothing written) as 'borrowed', so the panel names them as the brain's and starts nothing else on
    them until release_cards. Free cards are read again a moment after they are written, so a job the panel started
    meanwhile on one of them keeps it."""
    if cards is not None:
        chosen = list(dict.fromkeys(cards))
        if record is not None:
            _write_borrowed(record, [card for card in chosen if card != own])
        return chosen
    skip = record.stem if record is not None else None
    borrowed = _free_of_others(logs, own, skip)
    if record is not None:
        _write_borrowed(record, borrowed)
        if borrowed:
            time.sleep(BORROW_RECHECK_SECONDS)
            still = set(_free_of_others(logs, own, skip))
            if any(card not in still for card in borrowed):
                borrowed = [card for card in borrowed if card in still]
                _write_borrowed(record, borrowed)
    return [own, *borrowed]


def release_cards(record: Path | None) -> None:
    """The cards of borrow_cards given back: the brain holds its own only."""
    if record is not None and record.is_file():
        _write_borrowed(record, [])


def _now() -> str:
    """This moment, as the jobs' times are written (local time with its offset)."""
    return datetime.now().astimezone().isoformat(timespec='seconds')


def _mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec='seconds')
    except OSError:
        return None


def _next_step_text(payload: dict) -> tuple[str, bool]:
    """The next step in words for the page (route_status.next_step names the helpers, for the terminal), and
    whether everything is ready (dataset current, a training can start)."""
    rows = payload['routes']
    states = {row['state'] for row in rows}
    if not rows:
        return 'No drives yet: download drives from the comma.', False
    if states & {'collegare', 'estrarre', 'preannotare'}:
        return 'New videos to turn into frames with proposed signs: prepare the new drives.', False
    if 'revisionare' in states:
        return 'Open the drives to review and confirm or correct the signs.', False
    if payload['reviewed_routes'] < payload['min_routes']:
        return (f'Training needs at least {payload["min_routes"]} reviewed drives (now {payload["reviewed_routes"]}): '
                'download more drives from the comma.'), False
    if 'importare' in states:
        return 'The reviews are ready: import them and train the detector and the reader on the GPU.', False
    return 'The dataset is up to date: you can train a new model whenever you like.', True


# The last /proc/stat sample: the CPU use is the busy share of the time between two status calls.
_cpu_sample = {'times': None}
_cpu_lock = threading.Lock()


def _proc_stat(text: str) -> list[tuple[int, int]]:
    """(busy, total) jiffies of the whole CPU (first) and of every core."""
    rows = []
    for line in text.splitlines():
        if not line.startswith('cpu'):
            continue
        try:
            numbers = [int(value) for value in line.split()[1:]]
        except ValueError:
            continue
        if len(numbers) < 4:
            continue
        # idle + iowait are not busy; guest time is already inside user and nice.
        idle = numbers[3] + (numbers[4] if len(numbers) > 4 else 0)
        total = sum(numbers[:8])
        rows.append((total - idle, total))
    return rows


def _cpu_usage() -> dict | None:
    """{'total': %, 'cores': [%, ...]} since the previous call; None on the first call or without /proc."""
    try:
        current = _proc_stat(Path('/proc/stat').read_text())
    except OSError:
        return None
    with _cpu_lock:
        previous = _cpu_sample['times']
        _cpu_sample['times'] = current
    if previous is None or len(previous) != len(current):
        # The first call after a start: a short second sample, so the page shows a value at once.
        time.sleep(0.25)
        return _cpu_usage()
    percents = []
    for (busy, total), (old_busy, old_total) in zip(current, previous):
        elapsed = total - old_total
        percents.append(round(max(0.0, min(100.0, (busy - old_busy) * 100 / elapsed)), 1) if elapsed > 0 else 0.0)
    return {'total': percents[0], 'cores': percents[1:]}


def _memory() -> dict | None:
    """RAM from /proc/meminfo (inside the LXC it is the container's own memory)."""
    try:
        text = Path('/proc/meminfo').read_text()
    except OSError:
        return None
    values = {}
    for line in text.splitlines():
        name, separator, rest = line.partition(':')
        parts = rest.split()
        if separator and parts and parts[0].isdigit():
            values[name.strip()] = int(parts[0]) * 1024
    if 'MemTotal' not in values or 'MemAvailable' not in values:
        return None
    total = values['MemTotal']
    available = values['MemAvailable']
    return {'total_gb': round(total / 1e9, 1), 'available_gb': round(available / 1e9, 1),
            'used_percent': round((total - available) * 100 / total, 1)}


def _gpu_cards() -> list[dict]:
    """Each GPU on its own, in nvidia-smi's order: its index (the one NNSLR_GPU_INDEX takes, and CUDA's with
    CUDA_DEVICE_ORDER=PCI_BUS_ID), PCI bus ('81:00.0'), name, use, memory, temperature, power and the compute processes
    on it. A process is tied to its card by the bus: nvidia-smi gives the pids of the host, not of this container."""
    rows = _nvidia(['--query-gpu=index,pci.bus_id,name,utilization.gpu,memory.used,memory.total,temperature.gpu,'
                    'power.draw', '--format=csv,noheader,nounits'])
    if not rows:
        return []
    apps = _nvidia(['--query-compute-apps=gpu_bus_id,pid,process_name,used_memory', '--format=csv,noheader,nounits'])
    processes = {}
    for line in (apps or '').splitlines():
        parts = line.split(',')
        if len(parts) < 4:
            continue
        pid = parts[1].strip()
        processes.setdefault(_bus(parts[0]), []).append({'pid': int(pid) if pid.isdigit() else None,
                                                         'name': ','.join(parts[2:-1]).strip(),
                                                         'memory_mb': _number(parts[-1])})
    cards = []
    for line in rows.splitlines():
        fields = [part.strip() for part in line.split(',')]
        if len(fields) != 8 or not fields[0].isdigit():
            continue
        found = processes.get(_bus(fields[1]), [])
        cards.append({'index': int(fields[0]), 'bus_id': _bus(fields[1]), 'name': fields[2],
                      'utilization': _number(fields[3]), 'memory_used_mb': _number(fields[4]),
                      'memory_total_mb': _number(fields[5]), 'temperature': _number(fields[6]),
                      'power_w': _number(fields[7]), 'processes': len(found), 'process_list': found})
    if cards:
        # Kept for _card_indexes: the card of a /dev/nvidia<minor> a process has open, by its bus.
        _CARD_INDEX.clear()
        _CARD_INDEX.update({card['bus_id']: card['index'] for card in cards})
    return cards


# nvidia-smi's index of each card by its PCI bus, from its last answer (_gpu_cards).
_CARD_INDEX: dict[str, int] = {}
# A card's device file, as a process has it open: /dev/nvidia0, /dev/nvidia1 (not nvidiactl, nvidia-uvm...).
NVIDIA_DEVICE = re.compile(r'/dev/nvidia(\d+)')


def _card_indexes() -> dict[str, int]:
    """nvidia-smi's index of each card by its bus: from its last answer, asked now when there was none."""
    if not _CARD_INDEX:
        _gpu_cards()
    return _CARD_INDEX


def _session_of(pid: int) -> int | None:
    """The session of a process (the fourth field after its name in /proc/<pid>/stat), None when it is gone."""
    try:
        text = Path('/proc', str(pid), 'stat').read_text()
        return int(text[text.rindex(')') + 1:].split()[3])
    except (OSError, ValueError, IndexError):
        return None


def _children(pid: int) -> list[int]:
    """The children of a process, of each of its threads (/proc/<pid>/task/<tid>/children)."""
    found = []
    try:
        tasks = list(Path('/proc', str(pid), 'task').iterdir())
    except OSError:
        return found
    for task in tasks:
        try:
            found += [int(value) for value in (task / 'children').read_text().split()]
        except (OSError, ValueError):
            continue
    return found


def _process_tree(pids) -> list[int]:
    """The processes and all their descendants."""
    seen: dict[int, None] = {}
    waiting = list(pids)
    while waiting:
        pid = waiting.pop()
        if pid in seen:
            continue
        seen[pid] = None
        waiting += _children(pid)
    return list(seen)


def _command_name(pid: int) -> str:
    """The name of a process's program (/proc/<pid>/comm), '' when it is gone."""
    try:
        return Path('/proc', str(pid), 'comm').read_text().strip()
    except OSError:
        return ''


def _nvidia_minors(pid: int, proc: str = '/proc') -> set[int]:
    """The minors of the cards a process has open (/dev/nvidia<minor> among its open files, <proc>/<pid>/fd)."""
    found = set()
    try:
        handles = list(Path(proc, str(pid), 'fd').iterdir())
    except OSError:
        return found
    for handle in handles:
        try:
            match = NVIDIA_DEVICE.fullmatch(os.readlink(handle))
        except OSError:
            continue
        if match:
            found.add(int(match.group(1)))
    return found


def _gpu_minor_buses(gpus: str = '/proc/driver/nvidia/gpus') -> dict[int, str]:
    """The bus of each card by the minor of its device file, as the driver writes them
    (/proc/driver/nvidia/gpus/<bus>/information: "Device Minor", "Bus Location")."""
    found = {}
    for info in Path(gpus).glob('*/information'):
        try:
            text = info.read_text()
        except OSError:
            continue
        minor = re.search(r'^Device Minor:\s*(\d+)', text, re.M)
        bus = re.search(r'^Bus Location:\s*(\S+)', text, re.M)
        if minor:
            found[int(minor.group(1))] = _bus(bus.group(1) if bus else info.parent.name)
    return found


def _gpu_lock_holders(path: str | None = None) -> list[int]:
    """The processes that hold openpilot.distill's GPU lock, from /proc/locks by the lock's device and inode: the lock
    is only read, never opened nor taken. A lock taken by bash with the flock program (exec 9>>lock; flock 9) names
    the flock program that has gone, so /proc/locks leaves it out: that holder is not seen here."""
    try:
        info = os.stat(path or GPU_LOCK)
        lines = Path('/proc/locks').read_text().splitlines()
    except OSError:
        return []
    wanted = (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
    found = []
    for line in lines:
        fields = line.split()
        # "5: FLOCK  ADVISORY  WRITE 585 00:4e:1969 0 EOF"; a waiter has "->" after its number.
        if len(fields) < 6 or fields[1] == '->':
            continue
        try:
            major, minor, inode = fields[5].split(':')
            where = (int(major, 16), int(minor, 16), int(inode))
            pid = int(fields[4])
        except ValueError:
            continue
        if where == wanted and pid > 0 and pid not in found:
            found.append(pid)
    return found


def _tree_cards(pids) -> list[int] | None:
    """nvidia-smi's index of each card the processes and their descendants have open, None when none is open or one
    is not known (they hold every card then). A program that watches the cards opens the files of all of them,
    whichever card the job runs on (nvidia-smi, which the scripts run to check theirs; the system monitor of wandb):
    nvidia-smi is left out, and when some processes have only some cards open (CUDA with CUDA_VISIBLE_DEVICES opens
    only its own), those are the job's cards."""
    opened = []
    for pid in _process_tree(pids):
        if _command_name(pid) != 'nvidia-smi':
            minors = _nvidia_minors(pid)
            if minors:
                opened.append(minors)
    if not opened:
        return None
    buses = _gpu_minor_buses()
    some = [minors for minors in opened if not set(buses) <= minors]
    minors = set().union(*(some or opened))
    indexes = _card_indexes()
    cards = [indexes.get(buses.get(minor, '')) for minor in minors]
    if any(card is None for card in cards):
        return None
    return sorted(set(cards))


def _cards_of(system: dict) -> list[dict]:
    """The cards of a system() reading by index: 'gpus', or the one card of 'gpu' (a reading of before two cards)."""
    if 'gpus' in system:
        return sorted(system.get('gpus') or [], key=lambda card: card['index'])
    gpu = system.get('gpu')
    if gpu is None:
        return []
    return [{**gpu, 'index': 0, 'processes': gpu.get('processes') or 0}]


def _free_mb(card: dict) -> float:
    used, total = card.get('memory_used_mb'), card.get('memory_total_mb')
    return total - used if isinstance(used, (int, float)) and isinstance(total, (int, float)) else -1


def _best_card(states: list[dict]) -> dict | None:
    """The card with the most free memory (then the fewest holders, then the lowest index), None without cards."""
    if not states:
        return None
    return min(states, key=lambda card: (-_free_mb(card), len(card.get('holders') or []), card['index']))


def _card_states(cards: list[dict], holders: list[dict]) -> list[dict]:
    """Each card with its holders (those on it, and those of unknown card: on every card) and whether it is free. A card
    with a compute process and no holder known is held by a program the server cannot name."""
    states = []
    for card in cards:
        on = [holder for holder in holders if holder['gpus'] is None or card['index'] in holder['gpus']]
        if not on and card.get('processes'):
            on = [{'job': None, 'name': UNNAMED, 'label': UNNAMED[0].upper() + UNNAMED[1:], 'gpus': [card['index']],
                   'action': None}]
        states.append({**card, 'holders': on, 'free': not on})
    return states


def _shared_card(states: list[dict], action: str) -> dict | None:
    """The card a training of SHARED_TRAINING_MB shares when none is free: of those held only by such trainings and
    replays, the one with the most room, if it has the memory the training needs; None for another action or when
    none has the room. A card's room is its memory less what its holders use, or less what they may take while they
    start (SHARED_TRAINING_MB, SHARED_GPU), whichever is more."""
    need = SHARED_TRAINING_MB.get(action)
    if need is None:
        return None
    rooms = []
    for card in states:
        actions = [holder.get('action') for holder in card['holders']]
        total, used = card.get('memory_total_mb'), card.get('memory_used_mb')
        if not actions or not all(a in SHARED_TRAINING_MB or a in SHARED_GPU for a in actions) or not isinstance(
                total, (int, float)) or not isinstance(used, (int, float)):
            continue
        taken = max(used, sum(SHARED_TRAINING_MB.get(a) or SHARED_GPU[a] for a in actions))
        if total - taken >= need:
            rooms.append((-(total - taken), len(actions), card['index'], card))
    return min(rooms, key=lambda room: room[:3])[3] if rooms else None


def _gpu_busy(states: list[dict], holders: list[dict]) -> str:
    """Why no card can be taken now, in words, '' when one is free: "the GPU is busy with a cutter training" (one card),
    "both GPUs are busy: a cutter training on GPU 0, a program the server cannot name on GPU 1". Without nvidia-smi the
    cards are not known: busy while a job or work from a terminal other than a preparation holds the GPU."""
    if not states:
        names = list(dict.fromkeys(holder['name'] for holder in holders if holder['action'] not in PREPARING))
        return 'the GPU is busy with ' + ' and '.join(names) if names else ''
    if any(card['free'] for card in states):
        return ''

    def named(card: dict) -> str:
        return ' and '.join(dict.fromkeys(holder['name'] for holder in card['holders']))

    if len(states) == 1:
        # Today's words when nothing known holds it.
        return 'the GPU is busy with another process' if named(states[0]) == UNNAMED else (
            'the GPU is busy with ' + named(states[0]))
    if len(states) == 2:
        if named(states[0]) == named(states[1]):
            return 'both GPUs are busy with ' + named(states[0])
        return 'both GPUs are busy: ' + ', '.join(f'{named(card)} on GPU {card["index"]}' for card in states)
    return f'all {len(states)} GPUs are busy: ' + ', '.join(f'{named(card)} on GPU {card["index"]}' for card in states)


def _pair_apart(action: str, other: str) -> bool:
    """Whether two actions of the train group never run together (TRAIN_ALONE, TRAIN_ONCE, TRAIN_APART)."""
    return (action in TRAIN_ALONE or other in TRAIN_ALONE or (action == other and action in TRAIN_ONCE)
            or frozenset((action, other)) in TRAIN_APART)


def _bus(text: str) -> str:
    """A PCI bus as nvidia-smi ('00000000:81:00.0') or the driver ('0000:c1:00.0') write it: '81:00.0', 'c1:00.0'."""
    text = text.strip().lower()
    return text.split(':', 1)[1] if text.count(':') >= 2 else text


def _nvidia(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(['nvidia-smi', *arguments], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _number(text: str):
    try:
        return int(float(text))
    except ValueError:
        return None


def _next_action(next_step: str) -> str | None:
    if '1_prepare' in next_step:
        return 'prepare'
    if '2_review' in next_step:
        return 'review'
    if '3_train' in next_step:
        return 'train'
    if 'fetch_comma_segments' in next_step:
        return 'fetch'
    return None
# [control-panel] - END
