"""The Evaluate tab's data: the training runs of <data root>/runs summarised from their files (nnslr_tools.runs),
and GET /api/panel/runs, /api/panel/runs/<run>.

Every run here is a synthetic folder in a temporary data root, with reports built by the real metric functions of
nnslr_tools.evaluation: nothing is trained or evaluated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import nnslr_tools.runs as runs_module
from nnslr_tools.evaluation import (detector_metrics, human_reviewed_metrics, label_basis_counts, match_detections,
                                    reader_metrics)
from nnslr_tools.panel import Panel, PanelError
from nnslr_tools.runs import RunsIndex
from review_helpers import TOKEN, RunningServer, json_of, make_report, sign_in

DATASET_SHA = 'd' * 64
SPLIT_SHA = 'e' * 64
TEST_ROUTE = '0000012d--6552a45378'
OTHER_ROUTE = '00000123--8fe1b32468'
HUMAN = 'human_review'
AI = 'ai_review'


def write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value, indent=1), encoding='utf-8')
    return path


def detector_report():
    """Four test frames: signs 10, 15, 30 and 50 px wide; the 10 and 30 px ones found, a false box of 8 px on a
    frame without signs and one of 60 px."""
    frames = [
        (TEST_ROUTE, HUMAN, [[100, 100, 110, 110]], [{'box': [100, 100, 110, 110], 'score': 0.9}]),
        (TEST_ROUTE, AI, [[0, 0, 15, 15], [200, 200, 230, 230]],
         [{'box': [200, 200, 230, 230], 'score': 0.8}, {'box': [500, 500, 560, 560], 'score': 0.7}]),
        (OTHER_ROUTE, HUMAN, [[300, 300, 350, 350]], []),
        (OTHER_ROUTE, AI, [], [{'box': [10, 10, 18, 18], 'score': 0.5}]),
    ]
    images = []
    for index, (route, basis, truth, predictions) in enumerate(frames):
        matched = match_detections(predictions, truth, iou_threshold=0.5)
        images.append({'image_path': f'f/{index}.jpg', 'image_sha256': str(index) * 64, 'route_id': route,
                       'segment_index': 0, 'label_basis': basis, 'truth_boxes': truth, 'predictions': predictions,
                       'truth_count': len(truth), 'prediction_count': len(predictions),
                       'true_positives': matched['true_positives'], 'false_positives': matched['false_positives'],
                       'false_negatives': matched['false_negatives'], 'matches': matched['matches'],
                       'false_positive_details': matched['false_positive_details']})
    return {'schema_version': 1, 'task': 'detector', 'partition': 'test', **detector_metrics(images),
            'counts_by_label_basis': label_basis_counts(images),
            'human_reviewed': human_reviewed_metrics(images, detector_metrics),
            'score_threshold': 0.25, 'iou_threshold': 0.5, 'checkpoint': 'x', 'dataset_sha256': DATASET_SHA,
            'split_sha256': SPLIT_SHA, 'device': 'cuda', 'images': images}


def reader_report():
    """Six test crops: 80 read right twice and once as 60, 60 once right and once as 80, an end of limit right."""
    rows = [('maximum_speed:80', 'maximum_speed:80', HUMAN), ('maximum_speed:80', 'maximum_speed:80', AI),
            ('maximum_speed:80', 'maximum_speed:60', AI), ('maximum_speed:60', 'maximum_speed:60', AI),
            ('maximum_speed:60', 'maximum_speed:80', AI), ('cancellation', 'cancellation', HUMAN)]
    samples = [{'annotation_id': str(index), 'image_path': f'f/{index}.jpg', 'truth': truth, 'predicted': predicted,
                'confidence': 0.9, 'correct': truth == predicted, 'label_basis': basis}
               for index, (truth, predicted, basis) in enumerate(rows)]
    classes = ['cancellation', 'maximum_speed:60', 'maximum_speed:80']
    return {'schema_version': 1, 'task': 'reader', 'partition': 'test', 'classes': classes,
            **reader_metrics(samples, classes), 'counts_by_label_basis': label_basis_counts(samples),
            'human_reviewed': human_reviewed_metrics(samples, lambda items: reader_metrics(items, classes)),
            'samples': samples, 'checkpoint': 'x', 'dataset_sha256': DATASET_SHA, 'split_sha256': SPLIT_SHA, 'device': 'cuda'}


TRAIN_LOG = '''== NNSLR baseline pipeline ==
run:          {name}
training mode: {mode}
== 4/10 Train detector ==
{weights}detector epoch 1/2 train_loss 2.0000 val_loss 1.5000 30s
detector epoch 2/2 train_loss 1.0000 val_loss 1.2000 31s
== 5/10 Train reader ==
reader epoch 1/2 train_loss 1.0000 val_loss 0.9000 train_accuracy 0.5000 val_accuracy 0.6000 10s
reader epoch 2/2 train_loss 0.5000 val_loss 0.8000 train_accuracy 0.7000 val_accuracy 0.8000 11s
'''


def make_run(root: Path, name: str, *, upto: str = 'done', mode: str = 'fresh', weights: str = '', log_tail: str = '') -> Path:
    """A run folder as scripts/run_v100_baseline.sh leaves it when it stops at the step `upto` (detector, reader,
    evaluate, export, bundle), or at its end ('done')."""
    run = root / 'runs' / name
    split = write(root / 'splits' / (SPLIT_SHA + '.json'), {
        'policy': 'place-v3', 'seed': 0, 'grouping': ['place'], 'test_routes': [TEST_ROUTE],
        'routes': {'train': [OTHER_ROUTE], 'validation': [OTHER_ROUTE], 'test': [TEST_ROUTE, OTHER_ROUTE], 'buffer': [OTHER_ROUTE]},
        'counts': {'train': 70, 'validation': 14, 'test': 12, 'buffer': 3}, 'human_reviewed_counts': {'test': 5},
        'warnings': ['groups are incomplete'], 'assignments': {str(index): 'train' for index in range(70)}})
    write(root / 'splits' / 'latest.json', {'path': 'splits/' + split.name, 'sha256': SPLIT_SHA})
    common = {'schema_version': 1, 'dataset_kind': 'gold', 'dataset_sha256': DATASET_SHA, 'split_sha256': SPLIT_SHA,
              'split_path': str(split), 'dataset_path': str(root / 'annotations/objects.jsonl')}
    write(run / 'detector/training-plan.json', {**common, 'task': 'detector', 'classes': ['speed_sign'],
                                                 'counts_by_split': {'train': 60, 'validation': 10, 'test': 4},
                                                 'boxes_by_split': {'train': 30, 'validation': 6, 'test': 4},
                                                 'negative_frames_by_split': {'train': 40, 'validation': 5, 'test': 1},
                                                 'frames': [{'image_path': 'f/0.jpg'}] * 50})
    (run / 'evaluation').mkdir(parents=True, exist_ok=True)
    (run / 'export').mkdir(parents=True, exist_ok=True)
    write(root / 'logs' / f'train-{name}.log', TRAIN_LOG.format(name=name, mode=mode, weights=weights) + log_tail)
    steps = ('detector', 'reader', 'evaluate', 'export', 'bundle')
    reached = steps[:steps.index(upto)] if upto in steps else steps
    if 'detector' in reached:
        write(run / 'detector/result.json', '{"task": "detector", "best_validation_loss": 1.2, "initialization": "imagenet1k_v1",'
              ' "config": {"epochs": 2}, "gpu": {"name": "Tesla V100"}, "history": [{"epoch": 1, "train_loss": NaN,'
              ' "validation_loss": 1.5}, {"epoch": 2, "train_loss": 1.0, "validation_loss": 1.2}]}')
        write(run / 'reader/training-plan.json', {**common, 'task': 'reader', 'classes': ['a', 'b', 'c'],
                                                   'counts_by_split': {'train': 20, 'validation': 4, 'test': 6}, 'records': []})
    if 'reader' in reached:
        write(run / 'reader/result.json', {'task': 'reader', 'best_validation_accuracy': 0.8, 'initialization': 'imagenet1k_v1',
                                           'config': {'epochs': 2}, 'history': [
                                               {'epoch': 1, 'train_loss': 1.0, 'validation_loss': 0.9, 'train_accuracy': 0.5, 'validation_accuracy': 0.6},
                                               {'epoch': 2, 'train_loss': 0.5, 'validation_loss': 0.8, 'train_accuracy': 0.7, 'validation_accuracy': 0.8}]})
    if 'evaluate' in reached:
        write(run / 'evaluation/detector-test.json', detector_report())
        write(run / 'evaluation/reader-test.json', reader_report())
    if 'export' in reached:
        write(run / 'export/detector.onnx', 'onnx')
        write(run / 'export/reader.onnx', 'onnx')
    if 'bundle' in reached:
        write(run / 'bundle/sha256sums.json', {})
    return run


@pytest.fixture
def root(tmp_path):
    return tmp_path / 'data'


@pytest.fixture
def index(root):
    value = RunsIndex(root)
    value.running_runs = lambda: {}
    return value


def test_without_runs_the_list_is_empty(index):
    assert index.summaries() == {'runs': [], 'runs_dir_exists': False}


def test_a_finished_run_has_its_headline_numbers(root, index):
    make_run(root, '2026-10-20_0900')
    [run] = index.summaries()['runs']
    assert (run['name'], run['kind'], run['state'], run['stage']) == ('2026-10-20_0900', 'training', 'finished', 'done')
    assert run['started_at'].startswith('2026-10-20T09:00:00') and run['finished_at']
    assert (run['mode'], run['resumed'], run['start']['kind'], run['start']['label']) == ('fresh', False, 'imagenet', 'Pre-trained on ImageNet')
    assert run['reader'] == {'samples': 6, 'accuracy': pytest.approx(4 / 6), 'human_samples': 2, 'human_accuracy': 1.0}
    detector = run['detector']
    assert (detector['found'], detector['missed'], detector['false_boxes'], detector['truth_boxes']) == (2, 2, 2, 4)
    assert detector['recall'] == detector['precision'] == 0.5
    # The human-reviewed frames: the 10 px sign found, the 50 px one missed, no false box.
    assert (detector['human_recall'], detector['human_precision']) == (0.5, 1.0)
    assert run['best'] == {'detector_val_loss': 1.2, 'reader_val_accuracy': 0.8}
    assert run['split_sha256'] == SPLIT_SHA and run['evaluated'] and run['problems'] == []


def test_the_detail_has_labels_sizes_routes_split_and_curves(root, index):
    make_run(root, 'r1')
    detail = index.detail('r1')
    reader = detail['reader_eval']
    # By count, then label; the wrong answer it gives most.
    assert [(row['label'], row['name'], row['count'], row['correct']) for row in reader['labels']] == [
        ('maximum_speed:80', 'Speed limit 80', 3, 2), ('maximum_speed:60', 'Speed limit 60', 2, 1), ('cancellation', 'End of limit', 1, 1)]
    assert reader['labels'][0]['wrong'] == {'label': 'maximum_speed:60', 'name': 'Speed limit 60', 'count': 1}
    assert reader['labels'][2]['wrong'] is None
    assert [(row['label'], row['count']) for row in reader['labels_human']] == [('cancellation', 1), ('maximum_speed:80', 1)]
    sizes = {row['key']: (row['truth'], row['found'], row['false_boxes']) for row in detail['detector_eval']['sizes']}
    assert sizes == {'lt12': (1, 1, 1), '12-20': (1, 0, 0), '20-40': (1, 1, 0), 'ge40': (1, 0, 1)}
    human_sizes = {row['key']: (row['truth'], row['found']) for row in detail['detector_eval']['sizes_human']}
    assert human_sizes == {'lt12': (1, 1), '12-20': (0, 0), '20-40': (0, 0), 'ge40': (1, 0)}
    routes = {row['route']: (row['images'], row['found'], row['missed'], row['false_boxes']) for row in detail['detector_eval']['routes']}
    assert routes == {TEST_ROUTE: (2, 2, 1, 1), OTHER_ROUTE: (2, 0, 1, 1)}
    dataset = detail['dataset']
    assert dataset['split']['policy'] == 'place-v3' and dataset['split']['test_routes'] == [TEST_ROUTE]
    assert dataset['current_split'] and dataset['split_file'] == f'splits/{SPLIT_SHA}.json'
    assert dataset['detector']['boxes'] == {'train': 30, 'validation': 6, 'test': 4} and dataset['reader']['crops']['test'] == 6
    assert 'assignments' not in dataset['split']
    # The curves of the log, with the time of each epoch.
    assert detail['curves']['detector']['points'][1] == {'epoch': 2, 'train_loss': 1.0, 'val_loss': 1.2, 'seconds': 31}
    assert detail['curves']['reader']['epochs'] == 2
    assert detail['training']['detector']['gpu'] == 'Tesla V100' and detail['training']['reader']['best_epoch'] == 2
    assert all(detail['files'].values()) and detail['experiment'] is None


def test_without_a_log_the_curves_come_from_result_json(root, index):
    make_run(root, 'r1')
    (root / 'logs' / 'train-r1.log').unlink()
    detail = index.detail('r1')
    assert detail['curves']['reader']['points'][0] == {'epoch': 1, 'train_loss': 1.0, 'val_loss': 0.9, 'train_accuracy': 0.5,
                                                       'val_accuracy': 0.6}
    # A NaN of the history is left out (the answer is strict JSON).
    assert detail['curves']['detector']['points'][0] == {'epoch': 1, 'val_loss': 1.5}
    assert detail['mode'] is None and detail['start']['kind'] == 'imagenet'
    json.dumps(detail, allow_nan=False)


def test_a_running_run_and_a_stopped_one(root, index):
    make_run(root, '2026-10-24_1000', upto='detector')
    make_run(root, '2026-10-23_1000', upto='evaluate',
             log_tail='Traceback (most recent call last):\nValueError: checkpoint_split_mismatch\nERROR: training failed\n')
    index.running_runs = lambda: {'2026-10-24_1000': {'pipeline'}}
    running, stopped = index.summaries()['runs']
    assert (running['name'], running['state'], running['stage'], running['progress']) == ('2026-10-24_1000', 'running', 'detector', 'reader · epoch 2 of 2')
    assert running['start']['kind'] is None and running['start']['label'] == 'Not recorded yet'
    assert not running['evaluated'] and running['reader'] is None
    assert (stopped['state'], stopped['stage'], stopped['error']) == ('incomplete', 'evaluate', 'ERROR: training failed')
    assert stopped['progress'] is None
    # Not running any more: the same folder is a stopped run.
    index.running_runs = lambda: {}
    assert index.summaries()['runs'][0]['state'] == 'incomplete'


def test_where_the_weights_started_and_a_resume(root, index):
    make_run(root, 'from-run', mode='fresh', weights='continue detector from weights: /srv/data/runs/older_1/detector/detector-best.pt\n',
             log_tail='training mode: resume\n')
    make_run(root, 'from-stable', mode='continue',
             weights='continue detector from weights: cristianku/models@main/training/releases/v1/detector.pt\n')
    starts = {run['name']: run for run in index.summaries()['runs']}
    assert starts['from-run']['start'] == {'kind': 'run', 'label': 'Continued from run older_1', 'run': 'older_1',
                                           'source': '/srv/data/runs/older_1/detector/detector-best.pt'}
    assert starts['from-run']['mode'] == 'fresh' and starts['from-run']['resumed'] is True
    assert starts['from-stable']['start']['kind'] == 'stable' and starts['from-stable']['mode'] == 'continue'


def test_newest_first_and_unreadable_files(root, index):
    make_run(root, '2026-10-20_0900')
    run = make_run(root, '2026-10-22_0900')
    (run / 'evaluation/reader-test.json').write_text('{"task": "reader", "samp')
    first, second = index.summaries()['runs']
    assert (first['name'], second['name']) == ('2026-10-22_0900', '2026-10-20_0900')
    assert first['reader'] is None and first['detector'] is not None
    assert first['problems'] == ['evaluation/reader-test.json is unreadable']


def test_digests_are_kept_until_the_file_changes(root, index, monkeypatch):
    run = make_run(root, 'r1')
    calls = []
    original = runs_module.reader_digest
    monkeypatch.setattr(runs_module, 'reader_digest', lambda report: calls.append(1) or original(report))
    index.summaries()
    index.summaries()
    index.detail('r1')
    assert len(calls) == 1
    path = run / 'evaluation/reader-test.json'
    report = json.loads(path.read_text())
    report['accuracy'] = 0.25
    path.write_text(json.dumps(report))
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))
    assert index.summaries()['runs'][0]['reader']['accuracy'] == 0.25 and len(calls) == 2


def test_strict_names_and_no_way_out_of_the_data_root(root, index, tmp_path):
    make_run(root, 'r1')
    for name in ('..', '../r1', '.hidden', 'r1/..', '', 'a' * 101, 'r 1', 'r1\x00'):
        assert index.detail(name) is None, name
    # A run folder that is a symlink to somewhere else is not a run.
    outside = make_run(tmp_path / 'elsewhere', 'secret')
    (root / 'runs' / 'linked').symlink_to(outside, target_is_directory=True)
    assert [run['name'] for run in index.summaries()['runs']] == ['r1']
    assert index.detail('linked') is None
    # A report inside the run that is a symlink out of it is not read.
    report = root / 'runs' / 'r1' / 'evaluation' / 'reader-test.json'
    report.unlink()
    report.symlink_to(outside / 'evaluation' / 'reader-test.json')
    assert index.detail('r1')['reader_eval'] is None


def experiment_result(variant: str, reading: float, other_recall: float, *, human: bool = True) -> dict:
    """A result.json of nnslr_tools.context_experiment (metrics.<partition>.reading / ownership / human_reviewed)."""
    def report(scale: float) -> dict:
        return {'box_count': 40, 'reading': {'accuracy': reading * scale, 'expected_calibration_error': 0.05, 'sample_count': 40},
                'ownership': {'sample_count': 30, 'accuracy': 0.8, 'macro_f1': 0.6 * scale, 'abstain_rate': 0.1,
                              'decided_accuracy': 0.85, 'expected_calibration_error': 0.07,
                              'other_road': {'precision': 0.5, 'recall': other_recall * scale, 'f1': 0.4},
                              'confusion': {'other_road': {'other_road': 2, 'own_road': 1, 'unresolved': 0},
                                            'own_road': {'other_road': 1, 'own_road': 20, 'unresolved': 2},
                                            'unresolved': {'other_road': 0, 'own_road': 1, 'unresolved': 3}}}}
    test = {'loss': 0.7, **report(1.0), 'counts_by_label_basis': {'human_review': 10, 'ai_review': 30},
            'human_reviewed': {'label_basis': 'human_review', **report(0.9)} if human else None}
    return {'schema_version': 1, 'task': 'experiment_b', 'variant': variant, 'inputs': ['crop', 'box_geometry'],
            'best_epoch': 7, 'config': {'epochs': 20, 'seed': 0}, 'best_validation_selection_score': 0.7,
            'selection_metric': 'mean of validation reading accuracy and validation ownership macro-F1',
            'metrics': {'validation': {'loss': 0.6, **report(1.0), 'human_reviewed': None}, 'test': test},
            'split_sha256': SPLIT_SHA, 'initialization': 'random', 'history': [{'epoch': 1}], 'limit': None}


def test_experiment_b_variants_side_by_side(root, index):
    run = make_run(root, 'r1')
    write(run / 'experiment-b/crop/result.json', experiment_result('crop', 0.9, 0.2))
    write(run / 'experiment-b/full/result.json', experiment_result('full', 0.88, 0.6, human=False))
    assert index.summaries()['runs'][0]['experiment_b'] == ['crop', 'full']
    experiment = index.detail('r1')['experiment']
    assert [(item['key'], item['found'], item['best_epoch']) for item in experiment['variants']] == [
        ('crop', True, 7), ('crop_geom', False, None), ('full', True, 7)]
    assert experiment['variants'][0]['inputs'] == ['crop', 'box position'] and experiment['partitions'] == ['test', 'validation']
    rows = {row['key']: row for row in experiment['metrics']}
    assert rows['reading_accuracy']['values']['crop']['test'] == {'all': 0.9, 'human': pytest.approx(0.81)}
    assert rows['reading_accuracy']['values']['full']['test'] == {'all': 0.88, 'human': None}
    assert rows['reading_accuracy']['values']['crop_geom']['test'] == {'all': None, 'human': None}
    assert rows['other_road_recall']['better'] == 'higher' and rows['reading_calibration']['better'] == 'lower'
    assert rows['loss']['values']['crop']['validation']['all'] == 0.6
    assert list(rows)[0] == 'reading_accuracy' and 'seed' not in rows
    [confusion] = experiment['confusions']
    matrix = confusion['variants']['crop']['test']['all']
    assert matrix['labels'] == ['own_road', 'other_road', 'unresolved'] and matrix['names'] == ['Ours', 'Another', 'Unclear']
    assert matrix['matrix'][0] == [20, 1, 2] and confusion['variants']['full']['test']['human'] is None


def test_experiment_b_as_context_experiment_writes_it(root, index):
    """The reader of result.json against the module that writes it (skipped while that module is not importable)."""
    try:
        from nnslr_tools.context_experiment import partition_report
    except ImportError:
        pytest.skip('nnslr_tools.context_experiment not available')
    samples = [{'truth': 'a', 'predicted': 'a', 'correct': True, 'confidence': 0.9, 'label_basis': HUMAN,
                'ownership_truth': 'own_road', 'ownership_predicted': 'own_road', 'ownership_confidence': 0.8},
               {'truth': 'b', 'predicted': 'a', 'correct': False, 'confidence': 0.6, 'label_basis': AI,
                'ownership_truth': 'other_road', 'ownership_predicted': 'unresolved', 'ownership_confidence': 0.5}]
    run = make_run(root, 'r1')
    result = {'variant': 'crop', 'metrics': {'validation': None, 'test': {'loss': 0.5, **partition_report(samples, ['a', 'b'])}}}
    write(run / 'experiment-b/crop/result.json', json.dumps(result))
    rows = {row['key']: row for row in index.detail('r1')['experiment']['metrics']}
    assert rows['reading_accuracy']['values']['crop']['test'] == {'all': 0.5, 'human': 1.0}
    assert rows['abstain_rate']['values']['crop']['test']['all'] == 0.5
    assert rows['box_count']['values']['crop']['test'] == {'all': 2, 'human': 1}


def test_a_signnet_and_an_applicability_run_that_went_on_from_an_earlier_one(root, index):
    run = make_signnet_run(root)
    result = json.loads((run / 'result.json').read_text())
    write(run / 'result.json', {**result, 'start': {'path': '/srv/data/runs/older_1/signnet-best.pt', 'run': 'older_1', 'epoch': 9}})
    base = root / 'runs' / 'experiment-b-002' / 'experiment-b'
    write(base / 'crop/result.json', experiment_result('crop', 0.9, 0.2))
    write(base / 'full/result.json', {**experiment_result('full', 0.9, 0.2), 'initialization': 'start',
                                      'start': {'path': '/srv/data/runs/older_2/experiment-b/full/full-best.pt', 'run': 'older_2'}})
    runs = {item['name']: item for item in index.summaries()['runs']}
    assert runs['2026-10-30_1200-cutter']['start'] == {'kind': 'run', 'label': 'Continued from run older_1', 'run': 'older_1',
                                                       'source': '/srv/data/runs/older_1/signnet-best.pt'}
    assert (runs['experiment-b-002']['kind'], runs['experiment-b-002']['start']['run']) == ('experiment', 'older_2')
    # A start without a run name of ours is not one.
    write(run / 'result.json', {**result, 'start': {'run': '../x'}})
    assert index.detail('2026-10-30_1200-cutter')['start']['kind'] == 'random'


def test_a_folder_with_experiment_b_only(root, index):
    base = root / 'runs' / 'experiment-b-001' / 'experiment-b'
    write(base / 'crop/result.json', experiment_result('crop', 0.9, 0.2))
    write(base / 'crop_geom.log', 'experiment_b_crop_geom epoch 3/20 train_loss 1.0 val_loss 1.0 4s\n')
    write(base / 'dry-run.json', {'record_count': 3})
    index.running_runs = lambda: {'experiment-b-001': {'experiment'}}
    [run] = index.summaries()['runs']
    assert (run['kind'], run['state'], run['stage']) == ('experiment', 'running', 'experiment')
    assert run['progress'] == 'experiment B · crop_geom · epoch 3 of 20' and run['split_sha256'] == SPLIT_SHA
    assert run['start']['kind'] == 'random' and not run['evaluated']
    # The logs and the dry run next to the variant folders are not variants.
    assert run['experiment_b'] == ['crop'] and run['updated_at']
    assert [item['key'] for item in index.detail('experiment-b-001')['experiment']['variants']] == ['crop', 'crop_geom', 'full']
    for variant in ('crop_geom', 'full'):
        write(base / variant / 'result.json', experiment_result(variant, 0.9, 0.3))
    index.running_runs = lambda: {}
    [run] = index.summaries()['runs']
    assert (run['state'], run['stage'], run['experiment_b']) == ('finished', 'done', ['crop', 'crop_geom', 'full'])


def test_the_panel_answers_404_for_an_unknown_run(root):
    panel = Panel(root, repo=root / 'no-checkout', actions_enabled=False)
    panel.runs_index.running_runs = lambda: {}
    make_run(root, 'r1')
    assert [run['name'] for run in panel.runs()['runs']] == ['r1']
    assert panel.run('r1')['name'] == 'r1'
    with pytest.raises(PanelError) as error:
        panel.run('nope')
    assert error.value.status == 404 and error.value.reason == 'unknown_run'


def test_runs_over_http_with_the_sign_in(tmp_path):
    make_report(tmp_path)
    make_run(tmp_path, 'r1')
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    server.app.panel.runs_index.running_runs = lambda: {}
    try:
        # Signed out: the page is the sign-in page and the API answers 401.
        status, response, data = server.request('GET', '/evaluate')
        assert status in (200, 401) and b'id="app"' in data
        assert server.request('GET', '/api/panel/runs')[0] == 401
        assert server.request('GET', '/api/panel/runs/r1')[0] == 401
        cookie = sign_in(server)
        status, response, data = server.request('GET', '/api/panel/runs', headers=cookie)
        assert status == 200 and [run['name'] for run in json_of(data)['runs']] == ['r1']
        status, response, data = server.request('GET', '/api/panel/runs/r1', headers=cookie)
        detail = json_of(data)
        assert status == 200 and detail['detector_eval']['sizes'][0]['key'] == 'lt12'
        # Counts only: never the per-image or per-crop lists of the reports.
        assert detail['detector_eval']['images'] == 4 and detail['reader_eval']['samples'] == 6 and len(data) < 20000
        status, response, data = server.request('GET', '/api/panel/runs/nope', headers=cookie)
        assert status == 404 and json_of(data)['error'] == 'unknown_run'
        for path in ('/api/panel/runs/..%2F..%2Fsplits', '/api/panel/runs/../splits', '/api/panel/runs/r1/x', '/api/panel/runs/%2e%2e'):
            assert server.request('GET', path, headers=cookie)[0] == 404, path
        status, response, data = server.request('GET', '/evaluate?run=r1', headers=cookie)
        assert status == 200 and 'aria-current="page">Models</a>' in data.decode()
    finally:
        server.close()


# -- a run of scripts/run_classifier.sh: the classifier with "not a sign" ----------------------------------------------
CLASSES_V2 = ['cancellation', 'maximum_speed:50', 'maximum_speed:60', 'maximum_speed:80', 'not_a_sign', 'other_sign',
              'other_sign:give_way', 'other_sign:stop']
# Five sign crops (80 right twice, one of them confirmed by a person; 60 thrown away as "not a sign"; give way right, and
# once read as stop: the family is right) and four "not a sign" crops (a mined box let through as a 50).
CLASSIFIER_CROPS = [
    ('maximum_speed:80', 'maximum_speed:80', HUMAN, 'reviewed'), ('maximum_speed:80', 'maximum_speed:80', AI, 'reviewed'),
    ('maximum_speed:60', 'not_a_sign', AI, 'reviewed'), ('other_sign:give_way', 'other_sign:give_way', AI, 'reviewed'),
    ('other_sign:give_way', 'other_sign:stop', AI, 'reviewed'),
    ('not_a_sign', 'not_a_sign', AI, 'mined'), ('not_a_sign', 'maximum_speed:50', AI, 'mined'),
    ('not_a_sign', 'not_a_sign', AI, 'rejected'), ('not_a_sign', 'not_a_sign', HUMAN, 'reviewed'),
]


def classifier_report(partition: str = 'test') -> dict:
    """evaluation/classifier-<partition>.json as the spec of scripts/run_classifier.sh has it: the reader's report on
    every crop, plus the sign crops apart and the "not a sign" crops."""
    samples = [{'annotation_id': str(index), 'image_path': f'f/{index}.jpg', 'truth': truth, 'predicted': predicted,
                'confidence': 0.9, 'correct': truth == predicted, 'label_basis': basis, 'source': source}
               for index, (truth, predicted, basis, source) in enumerate(CLASSIFIER_CROPS)]
    signs = [item for item in samples if item['truth'] != 'not_a_sign']
    human_signs = [item for item in signs if item['label_basis'] == HUMAN]
    return {'schema_version': 1, 'task': 'reader', 'partition': partition, 'classes': CLASSES_V2, 'class_scheme': 'v2',
            **reader_metrics(samples, CLASSES_V2), 'counts_by_label_basis': label_basis_counts(samples),
            'human_reviewed': human_reviewed_metrics(samples, lambda items: reader_metrics(items, CLASSES_V2)),
            'signs': {**reader_metrics(signs, CLASSES_V2),
                      'human_reviewed': {'label_basis': HUMAN, **reader_metrics(human_signs, CLASSES_V2)},
                      'rejected': 1, 'rejected_rate': 0.2, 'family_accuracy': 0.8},
            'not_a_sign': {'crops': 4, 'rejected': 3, 'rejection_rate': 0.75,
                           'by_source': {'rejected': {'crops': 1, 'rejected': 1, 'rejection_rate': 1.0},
                                         'mined': {'crops': 2, 'rejected': 1, 'rejection_rate': 0.5},
                                         'reviewed': {'crops': 1, 'rejected': 1, 'rejection_rate': 1.0}},
                           'human_reviewed': {'crops': 1, 'rejected': 1, 'rejection_rate': 1.0},
                           'survivors': {'maximum_speed:50': 1}},
            'samples': samples, 'checkpoint': 'x', 'dataset_sha256': DATASET_SHA, 'split_sha256': SPLIT_SHA, 'device': 'cuda'}


def chain_counts(truth: int, found: int, correct: int, rejected: int, *, widths: bool = False, human=None) -> dict:
    counts = {'truth': truth, 'found': found, 'correct': correct, 'rejected': rejected,
              'recall': correct / truth if truth else None, 'recall_found': found / truth if truth else None,
              'read_right_when_found': correct / found if found else None, 'human_reviewed': human}
    if widths:
        counts['by_width'] = {'lt12': {'truth': 2, 'found': 0, 'correct': 0, 'rejected': 0, 'recall': 0.0, 'recall_found': 0.0},
                              '12-20': {'truth': 2, 'found': 2, 'correct': 1, 'rejected': 1, 'recall': 0.5, 'recall_found': 1.0},
                              '20-40': {'truth': 2, 'found': 2, 'correct': 1, 'rejected': 0, 'recall': 0.5, 'recall_found': 1.0},
                              'ge40': {'truth': 2, 'found': 2, 'correct': 2, 'rejected': 0, 'recall': 1.0, 'recall_found': 1.0}}
    return counts


def chain_report(partition: str = 'test', *, rejects: bool = True) -> dict:
    """evaluation/chain-<partition>.json: four frames, eight speed signs (six found by the cutter at 0.1), four false
    boxes of which the classifier rejects three (the old reader, without "not a sign", none)."""
    by_score = {}
    for score, found in ((0.05, 7), (0.1, 6), (0.25, 4)):
        correct = found - 2 if rejects else found - 3
        surviving = 1 if rejects else 4
        human = chain_counts(2, 2, 2 if rejects else 1, 0)
        by_score[f'{score:g}'] = {
            'speed_signs': chain_counts(8, found, correct, 1 if rejects else 0, widths=True, human=human),
            'all_boxes': chain_counts(10, found + 2, correct + 1, 1 if rejects else 0),
            'boxes_per_frame': (found + 4) / 4, 'false_boxes': 4, 'false_per_frame': 1.0,
            'surviving_false_boxes': surviving, 'surviving_false_per_frame': surviving / 4,
            'false_rejected_rate': (4 - surviving) / 4,
            'human': {'images': 2, 'false_per_frame': 0.5, 'surviving_false_per_frame': 0.0 if rejects else 0.5},
            'surviving_false_by_label': {'maximum_speed:50': 1} if rejects else {'maximum_speed:50': 3, 'cancellation': 1},
            'confusion': {'maximum_speed:80': {'maximum_speed:80': 3, 'not_a_sign': 1},
                          'not_a_sign': {'not_a_sign': 3, 'maximum_speed:50': 1}}}
    return {'schema_version': 1, 'task': 'chain', 'partition': partition,
            'cutter': {'checkpoint': '/data/runs/r1/detector/detector-best.pt', 'id': 'cutter:r1/detector', 'input_size': [320, 320],
                       'dataset_sha256': DATASET_SHA, 'split_sha256': SPLIT_SHA, 'same_split': True},
            'classifier': {'checkpoint': 'x', 'class_scheme': 'v2' if rejects else 'v1', 'classes': 8 if rejects else 3,
                           'rejects': rejects, 'dataset_sha256': DATASET_SHA, 'split_sha256': SPLIT_SHA, 'same_split': rejects},
            'dataset_sha256': DATASET_SHA, 'split_sha256': SPLIT_SHA, 'iou_threshold': 0.5,
            'image_count': 4, 'human_image_count': 2, 'speed_sign_count': 8, 'reviewed_box_count': 10,
            'score_thresholds': [0.05, 0.1, 0.25], 'main_score_threshold': 0.1, 'by_score_threshold': by_score,
            'images': [{'image_path': f'f/{index}.jpg', 'boxes': [[0, 0, 10, 10]] * 5} for index in range(4)],
            'device': 'Tesla V100', 'seconds': 12.4, 'limitations': ['A frame with an unboxed sign counts its box as false.']}


NEGATIVES_HEADER = {
    'kind': 'header', 'schema_version': 1, 'task': 'classifier_negatives', 'dataset_kind': 'gold', 'dataset_sha256': DATASET_SHA,
    'split_sha256': SPLIT_SHA, 'split_path': '/x', 'created_at': '2026-10-24T17:31:00+02:00',
    'cutter': {'checkpoint': '/data/runs/r1/detector/detector-best.pt', 'sha256': 'c' * 64, 'id': 'cutter:r1/detector',
               'input_size': [320, 320], 'dataset_sha256': DATASET_SHA, 'split_sha256': SPLIT_SHA, 'same_split': True},
    'settings': {'score_threshold': 0.1, 'max_iou': 0.1, 'per_frame': 3, 'total_factor': 3.0, 'seed': 0,
                 'rule_change_at': '2026-10-24T15:33:17Z'},
    'counts': {split: {'frames': 40, 'frames_with_candidates': 20, 'candidates': 50, 'after_frame_cap': 45, 'positives': 10,
                       'cap': 30, 'selected': 30, 'selected_by_tier': {'human': 10, 'ai_after_rule': 15, 'ai_before_rule': 5, 'other': 0}}
               for split in ('test', 'train', 'validation')},
}

CLASSIFIER_LOG = '''== Classifier (cutter -> classifier -> decider) ==
run:          {name}
== 1/7 Check the dataset, the split and the cutter (CPU-only) ==
== 2/7 Explicit CUDA smoke test ==
== 3/7 Mine the cutter's false boxes ==
== 4/7 Train the classifier ==
classifier epoch 1/3 train_loss 1.2000 val_loss 1.1000 train_accuracy 0.6000 val_accuracy 0.6500 val_sign_accuracy 0.6000 val_rejection 0.8000 40s
classifier epoch 2/3 train_loss 0.8000 val_loss 0.9000 train_accuracy 0.7500 val_accuracy 0.8000 val_sign_accuracy 0.7000 val_rejection 0.9000 41s
'''
CLASSIFIER_STEPS_LOG = ('== 5/7 Evaluate the classifier alone ==\n', '== 6/7 Evaluate the chain ==\n',
                        '== 7/7 Compare with the old reader ==\nClassifier complete: x\n')


def make_classifier_run(root: Path, name: str = '2026-10-24_1730-classifier', *, upto: str = 'done', log: str | None = None) -> Path:
    """A run folder as scripts/run_classifier.sh leaves it when it stops at the step `upto` (checks, mine, classifier,
    evaluate, chain), or at its end ('done')."""
    make_run(root, 'r1')
    run = root / 'runs' / name
    steps = ('checks', 'mine', 'classifier', 'evaluate', 'chain', 'done')
    reached = steps.index(upto)
    text = CLASSIFIER_LOG.format(name=name)
    if upto == 'checks':
        text = text.split('== 3/7')[0]
    elif upto == 'mine':
        text = text.split('== 4/7')[0]
    text += ''.join(CLASSIFIER_STEPS_LOG[:max(0, reached - 2)])
    write(run / 'classifier.log', text if log is None else log)
    (run / 'evaluation').mkdir(parents=True, exist_ok=True)
    if reached >= 2:
        write(run / 'negatives.jsonl', json.dumps(NEGATIVES_HEADER) + '\n' + json.dumps(
            {'kind': 'negative', 'source': 'mined', 'split': 'train', 'image_path': 'f/1.jpg', 'bbox_xyxy': [0, 0, 9, 9],
             'score': 0.4, 'label_basis': AI, 'tier': 'ai_after_rule'}) + '\n')
        write(run / 'classifier/training-plan.json', {
            'schema_version': 1, 'task': 'reader', 'class_scheme': 'v2', 'dataset_kind': 'gold', 'dataset_sha256': DATASET_SHA,
            'split_sha256': SPLIT_SHA, 'split_path': str(root / 'splits' / (SPLIT_SHA + '.json')), 'classes': CLASSES_V2,
            'counts_by_split': {'train': 60, 'validation': 12, 'test': 9},
            'negatives': {'train': {'rejected': 5, 'mined': 30, 'reviewed': 2}, 'validation': {'rejected': 1, 'mined': 6, 'reviewed': 0},
                          'test': {'rejected': 1, 'mined': 2, 'reviewed': 1}},
            'folded_labels': {'other_sign:roundabout': 'other_sign'}, 'records': [{'image_path': 'f/0.jpg'}] * 20})
        write(run / 'classifier/reader-best.pt', 'weights')
        write(run / 'classifier/training-state.pt', 'state')
    if reached >= 3:
        write(run / 'classifier/result.json', {
            'task': 'reader', 'architecture': 'mobilenet_v3_small', 'class_scheme': 'v2', 'classes': CLASSES_V2,
            'best_epoch': 2, 'best_validation_accuracy': 0.8, 'best_validation_sign_accuracy': 0.7,
            'best_validation_rejection': 0.9, 'best_validation_balanced_accuracy': 0.8, 'selection': {'by': 'balanced_accuracy'},
            'loss_weights': {'not_a_sign': 0.5}, 'augmentation': {'brightness': [0.6, 1.2]}, 'stopped_early_after_epoch': 2,
            'negatives': {'file': str(run / 'negatives.jsonl'), 'sha256': 'n' * 64,
                          'by_split': {'train': {'rejected': 5, 'mined': 30, 'reviewed': 2}}},
            'initialization': 'imagenet1k_v1', 'config': {'epochs': 3}, 'gpu': {'name': 'Tesla V100'},
            'history': [{'epoch': 1, 'train_loss': 1.2, 'validation_loss': 1.1, 'train_accuracy': 0.6, 'validation_accuracy': 0.65,
                         'validation_sign_accuracy': 0.6, 'validation_rejection': 0.8, 'validation_balanced_accuracy': 0.7,
                         'seconds': 40.4},
                        {'epoch': 2, 'train_loss': 0.8, 'validation_loss': 0.9, 'train_accuracy': 0.75, 'validation_accuracy': 0.8,
                         'validation_sign_accuracy': 0.7, 'validation_rejection': 0.9, 'validation_balanced_accuracy': 0.8,
                         'seconds': 41.0}]})
    if reached >= 4:
        write(run / 'evaluation/classifier-test.json', classifier_report('test'))
        write(run / 'evaluation/classifier-validation.json', classifier_report('validation'))
    if reached >= 5:
        write(run / 'evaluation/chain-test.json', chain_report('test'))
        write(run / 'evaluation/chain-validation.json', chain_report('validation'))
        write(run / 'evaluation/chain-test-old-reader.json', chain_report('test', rejects=False))
        write(run / 'comparison.txt', 'new classifier vs old reader\nend to end 50.0% vs 37.5%\n')
    return run


def test_a_classifier_that_went_on_from_another_does_not_break_the_list(root, index):
    run = make_classifier_run(root)
    result = json.loads((run / 'classifier/result.json').read_text())
    write(run / 'classifier/result.json', {**result, 'initialization': 'checkpoint', 'start': {
        'path': str(root / 'runs/older_1/classifier/reader-best.pt'), 'epoch': 9, 'mapillary_only': False}})
    runs = {item['name']: item for item in index.summaries()['runs']}
    assert runs['2026-10-24_1730-classifier']['start']['kind'] == 'weights'


def test_names_of_the_labels_of_the_classifier():
    assert runs_module.label_name('other_sign:give_way') == 'Other sign: give way'
    assert runs_module.label_name('other_sign') == 'Other sign'
    assert runs_module.label_name('not_a_sign') == 'Not a sign'
    assert runs_module.label_name('maximum_speed:60') == 'Speed limit 60'
    assert runs_module.label_name('maximum_speed_hours:30') == 'Speed limit 30, with hours'
    assert runs_module.label_name('maximum_speed_hours:30:22-6') == 'Speed limit 30, hours 22-6'
    assert runs_module.label_name('maximum_speed_wet:80') == 'Speed limit 80, when wet'
    assert runs_module.label_name('variable_display:unknown') == 'Variable speed limit, value unknown'


def test_a_classifier_run_is_a_kind_of_its_own_with_its_headline_numbers(root, index):
    make_classifier_run(root)
    runs = {run['name']: run for run in index.summaries()['runs']}
    run = runs['2026-10-24_1730-classifier']
    assert (run['kind'], run['state'], run['stage'], run['stage_label']) == ('classifier', 'finished', 'done', 'finished')
    # The time of the name, with what it is after it.
    assert run['started_at'].startswith('2026-10-24T17:30:00') and run['finished_at']
    assert run['reader'] is None and run['detector'] is None and run['evaluated']
    assert (run['split_sha256'], run['dataset_sha256']) == (SPLIT_SHA, DATASET_SHA)
    assert run['start'] == {'kind': 'imagenet', 'label': 'Pre-trained on ImageNet', 'run': None, 'source': None} and run['mode'] is None
    classifier = run['classifier']
    assert (classifier['samples'], classifier['accuracy'], classifier['human_samples'], classifier['human_accuracy']) == (5, 0.6, 1, 1.0)
    assert (classifier['family_accuracy'], classifier['rejection_rate'], classifier['not_a_sign_crops']) == (0.8, 0.75, 4)
    assert (classifier['signs_rejected'], classifier['signs_rejected_rate'], classifier['class_scheme']) == (1, 0.2, 'v2')
    assert (classifier['best_validation_balanced_accuracy'], classifier['best_epoch']) == (0.8, 2)
    chain = run['chain']
    assert (chain['score_threshold'], chain['at_chain_score'], chain['truth'], chain['found'], chain['correct']) == (0.1, True, 8, 6, 4)
    assert (chain['recall'], chain['recall_found'], chain['human_recall']) == (0.5, 0.75, 1.0)
    assert (chain['false_per_frame'], chain['surviving_false_per_frame'], chain['rejects']) == (1.0, 0.25, True)
    old = run['chain_old_reader']
    assert (old['recall'], old['surviving_false_per_frame'], old['rejects']) == (0.375, 1.0, False)
    # The pipeline run next to it keeps its own numbers, and has no classifier.
    assert runs['r1']['kind'] == 'training' and runs['r1']['classifier'] is None and runs['r1']['chain'] is None
    assert run['problems'] == []


@pytest.mark.parametrize('upto, stage, label', [
    ('checks', 'checks', 'checking the dataset, the split and the cutter'), ('mine', 'mine', 'mining the cutter’s false boxes'),
    ('classifier', 'classifier', 'training the classifier'), ('evaluate', 'evaluate', 'evaluating the classifier alone'),
    ('chain', 'chain', 'evaluating the chain end to end')])
def test_the_steps_of_a_classifier_run(root, index, upto, stage, label):
    make_classifier_run(root, 'c1', upto=upto)
    run = next(run for run in index.summaries()['runs'] if run['name'] == 'c1')
    assert (run['kind'], run['state'], run['stage'], run['stage_label']) == ('classifier', 'incomplete', stage, label)
    index.running_runs = lambda: {'c1': {'classifier'}}
    run = next(run for run in index.summaries()['runs'] if run['name'] == 'c1')
    assert run['state'] == 'running' and run['stage'] == stage
    # A pipeline running in a folder of the same name is not this run's work.
    index.running_runs = lambda: {'c1': {'pipeline'}}
    assert next(run for run in index.summaries()['runs'] if run['name'] == 'c1')['state'] == 'incomplete'


def test_a_running_classifier_run_says_its_epoch_and_a_stopped_one_its_error(root, index):
    make_classifier_run(root, 'c1', upto='classifier')
    index.running_runs = lambda: {'c1': {'classifier'}}
    run = next(run for run in index.summaries()['runs'] if run['name'] == 'c1')
    assert (run['progress'], run['error'], run['evaluated']) == ('classifier · epoch 2 of 3', None, False)
    # Before the first file the folder is a classifier run already, from its log or because its script runs.
    write(root / 'runs/c2/classifier.log', '== 1/7 Check the dataset, the split and the cutter (CPU-only) ==\n'
                                           'error: the cutter in use is missing\n')
    (root / 'runs/c3').mkdir(parents=True)
    index.running_runs = lambda: {'c3': {'classifier'}}
    runs = {run['name']: run for run in index.summaries()['runs']}
    assert (runs['c2']['kind'], runs['c2']['state'], runs['c2']['error']) == ('classifier', 'incomplete', 'error: the cutter in use is missing')
    assert (runs['c3']['kind'], runs['c3']['state'], runs['c3']['stage']) == ('classifier', 'running', 'checks')
    assert runs['c1']['state'] == 'incomplete' and runs['c1']['started_at']


def test_the_detail_of_a_classifier_run(root, index):
    make_classifier_run(root)
    detail = index.detail('2026-10-24_1730-classifier')
    assert detail['reader_eval'] is None and detail['detector_eval'] is None and detail['experiment'] is None
    own = detail['classifier_run']
    assert all(own['files'].values())

    # The classifier alone: every crop, the sign crops apart, the "not a sign" crops by source.
    test = own['evaluations']['test']
    assert (test['partition'], test['class_scheme'], test['samples'], test['accuracy']) == ('test', 'v2', 9, pytest.approx(6 / 9))
    assert next(row for row in test['labels'] if row['label'] == 'not_a_sign')['count'] == 4
    signs = test['signs']
    assert (signs['samples'], signs['correct'], signs['accuracy'], signs['family_accuracy']) == (5, 3, 0.6, 0.8)
    assert (signs['rejected'], signs['rejected_rate'], signs['human']) == (1, 0.2, {'samples': 1, 'accuracy': 1.0, 'correct': 1})
    labels = {row['label']: row for row in signs['labels']}
    assert 'not_a_sign' not in labels and labels['other_sign:give_way']['name'] == 'Other sign: give way'
    assert labels['other_sign:give_way']['wrong'] == {'label': 'other_sign:stop', 'name': 'Other sign: stop', 'count': 1}
    assert labels['maximum_speed:60']['wrong'] == {'label': 'not_a_sign', 'name': 'Not a sign', 'count': 1}
    assert [row['label'] for row in signs['labels_human']] == ['maximum_speed:80']
    rejection = test['not_a_sign']
    assert (rejection['crops'], rejection['rejected'], rejection['rejection_rate']) == (4, 3, 0.75)
    assert [(row['source'], row['label'], row['crops'], row['rejection_rate']) for row in rejection['by_source']] == [
        ('rejected', 'Removed proposals', 1, 1.0), ('mined', 'Cutter’s false boxes', 2, 0.5),
        ('reviewed', 'Reviewed “not a sign”', 1, 1.0)]
    assert rejection['human'] == {'crops': 1, 'rejected': 1, 'rejection_rate': 1.0}
    assert rejection['survivors'] == [{'label': 'maximum_speed:50', 'name': 'Speed limit 50', 'count': 1}]
    assert own['evaluations']['validation']['partition'] == 'validation'

    # The chain: every score, sorted; the labels only at the main score; the frames never.
    chain = own['chains']['test']
    assert [item['score'] for item in chain['thresholds']] == [0.05, 0.1, 0.25]
    main = chain['thresholds'][1]
    assert (main['speed_signs']['recall'], main['speed_signs']['recall_found'], main['surviving_false_per_frame']) == (0.5, 0.75, 0.25)
    assert [(row['key'], row['label'], row['correct'], row['truth']) for row in main['speed_signs']['widths']] == [
        ('lt12', '< 12 px', 0, 2), ('12-20', '12–20 px', 1, 2), ('20-40', '20–40 px', 1, 2), ('ge40', '≥ 40 px', 2, 2)]
    assert main['speed_signs']['human']['recall'] == 1.0 and main['all_boxes']['truth'] == 10
    assert 'widths' not in main['all_boxes']
    assert main['surviving_false_by_label'] == [{'label': 'maximum_speed:50', 'name': 'Speed limit 50', 'count': 1}]
    assert [(row['label'], row['count'], row['correct']) for row in main['labels']] == [('maximum_speed:80', 4, 3), ('not_a_sign', 4, 3)]
    assert chain['thresholds'][0]['labels'] is None and chain['thresholds'][2]['labels'] is None
    assert (chain['cutter']['id'], chain['cutter']['input_size'], chain['classifier']['rejects']) == ('cutter:r1/detector', '320x320', True)
    assert (chain['image_count'], chain['human_image_count'], chain['limitations']) == (4, 2, ['A frame with an unboxed sign counts its box as false.'])
    assert own['chains']['old_reader']['classifier']['rejects'] is False and own['chains']['validation']['partition'] == 'validation'

    # The negatives it learnt from: the header only, the partitions in the order of the training.
    negatives = own['negatives']
    assert [row['split'] for row in negatives['splits']] == ['train', 'validation', 'test']
    assert negatives['splits'][0]['by_tier'] == {'human': 10, 'ai_after_rule': 15, 'ai_before_rule': 5, 'other': 0}
    assert (negatives['cutter']['id'], negatives['cutter']['input_size'], negatives['settings']['per_frame']) == ('cutter:r1/detector', '320x320', 3)
    assert negatives['settings']['rule_change_at'] == '2026-10-24T15:33:17Z'

    training = own['training']
    assert (training['best_epoch'], training['best_validation_balanced_accuracy'], training['selection']) == (2, 0.8, 'balanced_accuracy')
    assert (training['not_a_sign_weight'], training['augmentation'], training['stopped_early'], training['classes']) == (0.5, True, 2, 8)
    assert training['negatives'] == {'file': 'negatives.jsonl', 'sha256': 'n' * 64,
                                     'by_split': {'train': {'rejected': 5, 'mined': 30, 'reviewed': 2}}}
    assert 'history' not in training and own['comparison'].startswith('new classifier vs old reader')

    # The curves of its own log, the balanced accuracy computed when the line has not got it.
    curve = detail['curves']['classifier']
    assert curve['epochs'] == 3 and curve['points'][1] == {
        'epoch': 2, 'train_loss': 0.8, 'val_loss': 0.9, 'train_accuracy': 0.75, 'val_accuracy': 0.8, 'val_sign_accuracy': 0.7,
        'val_rejection': 0.9, 'val_balanced_accuracy': pytest.approx(0.8), 'seconds': 41}
    assert set(detail['curves']) == {'classifier'}

    dataset = detail['dataset']
    assert dataset['split']['policy'] == 'place-v3' and dataset['detector'] is None and dataset['reader'] is None
    assert dataset['classifier']['crops'] == {'train': 60, 'validation': 12, 'test': 9} and dataset['classifier']['classes'] == 8
    assert dataset['classifier']['negatives']['test'] == {'rejected': 1, 'mined': 2, 'reviewed': 1}
    assert dataset['classifier']['folded_labels'] == [{'label': 'other_sign:roundabout', 'name': 'Other sign: roundabout',
                                                       'into': 'other_sign', 'into_name': 'Other sign'}]
    text = json.dumps(detail, allow_nan=False)
    # Counts only: never the crops, the frames or the rows of the plan.
    assert '"images": [' not in text and '"samples": [' not in text and '"records"' not in text and len(text) < 40000
    # A pipeline run has no classifier part.
    assert index.detail('r1')['classifier_run'] is None and 'classifier' not in index.detail('r1')['curves']



def test_a_classifier_in_two_stages_has_the_curves_of_its_pretraining(root, index):
    # NNSLR_CLASSIFIER_MAPILLARY=pretrain: the Mapillary signs alone ("pretrain epoch ..."), then ours from there.
    log = CLASSIFIER_LOG.format(name='c3').replace('== 4/7 Train the classifier ==\n', (
        '== 4/7 Train the classifier ==\n'
        'pretrain epoch 1/2 train_loss 0.4500 val_loss 0.9200 train_accuracy 0.8700 val_accuracy 0.7900 val_sign_accuracy 0.7900 31s\n'
        'pretrain stops early after epoch 1: x\n'))
    run = make_classifier_run(root, 'c3', log=log)
    curves = index.detail(run.name)['curves']
    assert set(curves) == {'pretrain', 'classifier'} and curves['pretrain']['stopped_early'] == 1
    assert curves['pretrain']['points'] == [{'epoch': 1, 'train_loss': 0.45, 'val_loss': 0.92, 'train_accuracy': 0.87,
                                             'val_accuracy': 0.79, 'val_sign_accuracy': 0.79, 'seconds': 31}]
    assert len(curves['classifier']['points']) == 2

def test_without_its_log_the_classifier_curves_come_from_result_json(root, index):
    run = make_classifier_run(root)
    (run / 'classifier.log').unlink()
    curve = index.detail(run.name)['curves']['classifier']
    assert curve == {'epochs': 3, 'stopped_early': 2, 'points': [
        {'epoch': 1, 'train_loss': 1.2, 'val_loss': 1.1, 'train_accuracy': 0.6, 'val_accuracy': 0.65, 'val_sign_accuracy': 0.6,
         'val_rejection': 0.8, 'val_balanced_accuracy': 0.7, 'seconds': 40},
        {'epoch': 2, 'train_loss': 0.8, 'val_loss': 0.9, 'train_accuracy': 0.75, 'val_accuracy': 0.8, 'val_sign_accuracy': 0.7,
         'val_rejection': 0.9, 'val_balanced_accuracy': 0.8, 'seconds': 41}]}
    # Still a classifier run, by its other files.
    assert index.detail(run.name)['kind'] == 'classifier'


def test_unreadable_files_of_a_classifier_run(root, index):
    run = make_classifier_run(root)
    (run / 'negatives.jsonl').write_text('{"kind": "negative"}\n')
    (run / 'evaluation/chain-test.json').write_text('{"task": "chain", "by_sc')
    (run / 'evaluation/classifier-validation.json').write_text('[]')
    summary = next(item for item in index.summaries()['runs'] if item['name'] == run.name)
    assert sorted(summary['problems']) == ['evaluation/chain-test.json is unreadable',
                                          'evaluation/classifier-validation.json is unreadable', 'negatives.jsonl is unreadable']
    assert summary['chain'] is None and summary['classifier']['accuracy'] == 0.6
    detail = index.detail(run.name)
    assert detail['classifier_run']['negatives'] is None and detail['classifier_run']['chains']['test'] is None
    # A report of another task is not a chain.
    (run / 'evaluation/chain-test.json').write_text(json.dumps({'task': 'detector'}))
    assert 'evaluation/chain-test.json is unreadable' in index.summaries()['runs'][0]['problems'] + \
        next(item for item in index.summaries()['runs'] if item['name'] == run.name)['problems']


def test_only_the_first_line_of_the_negatives_is_read(root, index, monkeypatch):
    run = make_classifier_run(root)
    with (run / 'negatives.jsonl').open('a') as handle:
        handle.write('not json at all\n' * 1000)
    assert index.detail(run.name)['classifier_run']['negatives']['splits'][0]['selected'] == 30
    monkeypatch.setattr(runs_module, 'MAX_HEADER_BYTES', 64)
    (run / 'negatives.jsonl').write_text(json.dumps(NEGATIVES_HEADER) + '\n')
    assert index.detail(run.name)['classifier_run']['negatives'] is None


def test_a_classifier_script_running_on_this_data_root_is_seen(root, tmp_path):
    import subprocess
    import sys
    import time as clock
    make_classifier_run(root, 'c1', upto='classifier')
    index = RunsIndex(root)
    # A process whose arguments are those of the script: "scripts/run_classifier.sh c1", on this data root.
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)', 'scripts/run_classifier.sh', 'c1'],
                               env={**os.environ, 'NNSLR_DATA_ROOT': str(root)})
    try:
        deadline = clock.monotonic() + 5
        while 'c1' not in index.running_runs() and clock.monotonic() < deadline:
            clock.sleep(0.05)
        assert 'classifier' in index.running_runs().get('c1', set())
        run = next(run for run in index.summaries()['runs'] if run['name'] == 'c1')
        assert (run['state'], run['progress']) == ('running', 'classifier · epoch 2 of 3')
    finally:
        process.kill()
        process.wait()


# -- a run of scripts/run_detector_sizes.sh alone: "Train the cutter" ------------------------------------------------
CUTTER_SIZE_LOG = '''detector epoch 1/3 train_loss 2.0000 val_loss 1.8000 val_recall 0.3000 80s
detector epoch 2/3 train_loss 1.5000 val_loss 1.6000 val_recall 0.4500 81s
'''


def cutter_report(partition: str = 'test', families: str = 'all', size: str = '960x544') -> dict:
    """detector-<partition>.json of a cutter of every sign: the report of test_panel_models, with its size, what it
    finds and its recall by family at each score."""
    from test_panel_models import sizes_report
    report = sizes_report(partition)
    report.update({'families': families, 'input_size': [int(side) for side in size.split('x')], 'input_size_label': size})
    for metrics in report['by_score_threshold'].values():
        metrics['recall_by_family'] = {'speed': {'truth': 3, 'found': 2, 'recall': 2 / 3}}
        if families == 'all':
            metrics['recall_by_family']['other_sign'] = {'truth': 1, 'found': 0, 'recall': 0.0}
    return report


def make_cutter_run(root: Path, name: str = '2026-10-24_1800-cutter', *, size: str = '960x544', upto: str = 'done',
                    families: str = 'all') -> Path:
    """A run folder as scripts/run_detector_sizes.sh NAME SIZE leaves it at the step `upto` (checks, detector, evaluate)
    or at its end ('done')."""
    make_run(root, 'r1')
    run = root / 'runs' / name
    steps = ('checks', 'detector', 'evaluate', 'done')
    reached = steps.index(upto)
    (run / 'detector-sizes').mkdir(parents=True, exist_ok=True)
    folder = run / 'detector-sizes' / size
    if reached >= 1:
        write(run / 'detector-sizes' / (size + '.log'), CUTTER_SIZE_LOG)
        write(folder / 'training-plan.json', {
            'schema_version': 1, 'task': 'detector', 'dataset_kind': 'gold', 'dataset_sha256': DATASET_SHA, 'split_sha256': SPLIT_SHA,
            'split_path': str(root / 'splits' / (SPLIT_SHA + '.json')), 'classes': ['speed_sign'],
            'counts_by_split': {'train': 60, 'validation': 10, 'test': 4}, 'boxes_by_split': {'train': 40, 'validation': 8, 'test': 4},
            'negative_frames_by_split': {'train': 40, 'validation': 5, 'test': 1}, 'frames': []})
        write(folder / 'detector-best.pt', 'weights')
    if reached >= 2:
        write(folder / 'result.json', {'task': 'detector', 'architecture': 'ssdlite320_mobilenet_v3_large', 'families': families,
                                       'input_size': [int(side) for side in size.split('x')], 'anchor_min_ratio': 0.02,
                                       'augmentation': {'brightness': [0.4, 1.2]}, 'best_epoch': 2, 'best_validation_loss': 1.6,
                                       'initialization': 'imagenet1k_v1', 'config': {'epochs': 3}, 'gpu': {'name': 'Tesla V100'},
                                       'history': [{'epoch': 1, 'train_loss': 2.0, 'validation_loss': 1.8},
                                                   {'epoch': 2, 'train_loss': 1.5, 'validation_loss': 1.6}]})
    if reached >= 3:
        write(folder / 'evaluation/detector-test.json', cutter_report('test', families, size))
        write(folder / 'evaluation/detector-validation.json', cutter_report('validation', families, size))
        write(run / 'detector-sizes' / 'comparison.txt', 'size 960x544: recall 50.0%\n')
    return run


def test_a_cutter_run_is_a_kind_of_its_own(root, index):
    make_cutter_run(root)
    runs = {run['name']: run for run in index.summaries()['runs']}
    run = runs['2026-10-24_1800-cutter']
    assert (run['kind'], run['state'], run['stage'], run['stage_label']) == ('cutter', 'finished', 'done', 'finished')
    assert run['started_at'].startswith('2026-10-24T18:00:00') and run['finished_at'] and run['evaluated']
    assert (run['split_sha256'], run['start']['kind'], run['reader'], run['classifier']) == (SPLIT_SHA, 'imagenet', None, None)
    [line] = run['cutters']
    assert (line['size'], line['families'], line['finished'], line['partition'], line['score_threshold']) == ('960x544', 'all', True, 'test', 0.1)
    assert (line['recall'], line['found'], line['truth_boxes'], line['speed_recall']) == (0.5, 2, 4, pytest.approx(2 / 3))
    assert (line['boxes_per_frame'], line['ms_per_frame']) == (1.0, 12.5)
    assert runs['r1']['kind'] == 'training' and runs['r1']['cutters'] is None


@pytest.mark.parametrize('upto, stage, label', [('checks', 'checks', 'checking the dataset and the GPU'),
                                                ('detector', 'detector', 'training the cutter'),
                                                ('evaluate', 'evaluate', 'evaluating on test and validation')])
def test_the_steps_of_a_cutter_run(root, index, upto, stage, label):
    make_cutter_run(root, 'k1', upto=upto)
    run = next(run for run in index.summaries()['runs'] if run['name'] == 'k1')
    assert (run['kind'], run['state'], run['stage'], run['stage_label']) == ('cutter', 'incomplete', stage, label)
    index.running_runs = lambda: {'k1': {'cutter'}}
    run = next(run for run in index.summaries()['runs'] if run['name'] == 'k1')
    assert run['state'] == 'running'
    if upto != 'checks':
        assert run['progress'] == '960x544 · detector · epoch 2 of 3'


def test_the_detail_of_a_cutter_run(root, index):
    run = make_cutter_run(root)
    detail = index.detail(run.name)
    [size] = detail['cutter_run']['sizes']
    assert (size['size'], size['label'], size['families'], size['finished']) == ('960x544', '960 × 544', 'all', True)
    assert (size['training']['best_epoch'], size['training']['anchor_min_ratio'], size['training']['families']) == (2, 0.02, 'all')
    test = size['test']
    assert (test['score_threshold'], test['at_cutter_score'], test['recall'], test['families']) == (0.1, True, 0.5, 'all')
    assert test['by_family'] == {'speed': {'truth': 3, 'found': 2, 'recall': pytest.approx(2 / 3)},
                                 'other_sign': {'truth': 1, 'found': 0, 'recall': 0.0}}
    assert [row['key'] for row in test['widths']] == ['lt12', '12-20', '20-40', 'ge40'] and size['validation']['partition'] == 'validation'
    assert all(size['files'].values()) and detail['cutter_run']['comparison'] == 'size 960x544: recall 50.0%\n'
    # The curves of its log, with the validation recall that chooses its best epoch.
    assert detail['curves'] == {'detector': {'epochs': 3, 'points': [
        {'epoch': 1, 'train_loss': 2.0, 'val_loss': 1.8, 'val_recall': 0.3, 'seconds': 80},
        {'epoch': 2, 'train_loss': 1.5, 'val_loss': 1.6, 'val_recall': 0.45, 'seconds': 81}]}}
    assert detail['dataset']['detector']['boxes'] == {'train': 40, 'validation': 8, 'test': 4} and detail['dataset']['split']['policy'] == 'place-v3'
    assert detail['classifier_run'] is None
    (run / 'detector-sizes/960x544.log').unlink()
    assert index.detail(run.name)['curves']['detector']['points'][1] == {'epoch': 2, 'train_loss': 1.5, 'val_loss': 1.6}
    json.dumps(index.detail(run.name), allow_nan=False)


def test_the_sizes_compared_in_a_pipeline_run_stay_part_of_it(root, index):
    from test_panel_models import make_size
    run = make_run(root, '2026-10-24_1452')
    make_size(run, '640x360')
    [summary] = index.summaries()['runs']
    assert (summary['kind'], summary['state'], summary['stage']) == ('training', 'finished', 'done')
    assert [line['size'] for line in summary['cutters']] == ['640x360'] and summary['cutters'][0]['families'] == 'speed'
    detail = index.detail(run.name)
    assert [size['size'] for size in detail['cutter_run']['sizes']] == ['640x360'] and set(detail['curves']) == {'detector', 'reader'}
    # The script running on that folder does not make it another kind.
    index.running_runs = lambda: {run.name: {'cutter'}}
    assert index.summaries()['runs'][0]['kind'] == 'training'


def test_an_unreadable_size_of_a_cutter_run(root, index):
    run = make_cutter_run(root)
    (run / 'detector-sizes/960x544/evaluation/detector-test.json').write_text('{"task": ')
    summary = next(item for item in index.summaries()['runs'] if item['name'] == run.name)
    assert summary['problems'] == ['detector-sizes/960x544/evaluation/detector-test.json is unreadable']
    assert summary['cutters'][0]['partition'] == 'validation'


# -- a SignNet run: "Train the cutter" (scripts/run_signnet.sh NAME) --------------------------------------------------
SIGNNET_LOG = '''== scripts/run_signnet.sh {name}
run:          {name}
band:         896x288
settings:     --families all --band-width 896 --stride 8 --fuse-kernel 1 --widths 16,24,32,48 --epochs 3 --batch 16
== 1/3 Check the training population (CPU-only) ==
== 2/3 Explicit CUDA smoke test ==
== 3/3 Train SignNet ==
signnet epoch 1/3 train_loss 4.0846 val_recall 0.064 val_false_per_frame 4.98 80s
signnet epoch 2/3 train_loss 2.9279 val_recall 0.173 val_false_per_frame 4.90 81s
'''
SIGNNET_END = 'signnet measure the best epoch 2 on validation and test\nSignNet complete: {run}\n'


def signnet_measure(recall: float, false_per_frame: float, car_recall: float, *, car: bool = True) -> dict:
    """validation or test of a SignNet's result.json (signnet.measure), with its signs found at 3, 5 and 10 boxes,
    at score 0.3 and, unless car is False (one written before 30 Oct 2026), as the car reads it (car_recall)."""
    budget = lambda boxes: {'recall_iou': recall * boxes / 5, 'false_positives_per_image': false_per_frame * boxes / 5,
                            'true_positives': 2, 'truth_box_count': 4}
    measured = {'by_score_threshold': {'0.3': {'recall_iou': 0.5, 'false_positives_per_image': 0.25, 'true_positives': 2,
                                               'truth_box_count': 4, 'predictions_per_image': 0.75}},
                'by_box_budget': {'3': budget(3), '5': budget(5), '10': budget(10)}, 'ignore_boxes': 0,
                'found_on_ignore_boxes': 0}
    if car:
        measured['car'] = {'recall_iou': car_recall, 'false_positives_per_image': 0.2, 'true_positives': 1,
                           'truth_box_count': 4, 'predictions_per_image': 0.5,
                           'recall_by_family': {'speed': {'truth': 2, 'found': 1, 'recall': 0.5}}}
    return measured


def signnet_result(band=(896, 288), stride: int = 8, car: bool = True) -> dict:
    """result.json of nnslr train-signnet: the car's setup by default."""
    return {'task': 'signnet', 'widths': [16, 24, 32, 48], 'stride': stride, 'fuse_kernel': 1, 'band': list(band),
            'parameters': 62997, 'epochs': 3, 'best_epoch': 2, 'dataset_sha256': DATASET_SHA, 'split_sha256': SPLIT_SHA,
            'families': 'all', 'counts_by_split': {'train': 60, 'validation': 10, 'test': 4},
            'boxes_by_split': {'train': 40, 'validation': 8, 'test': 4},
            'history': [{'epoch': 1, 'loss': 4.08, 'focal': 3.1, 'shape': 0.97, 'validation_recall_at_5': 0.064,
                         'validation_false_positives_per_image_at_5': 4.98, 'seconds': 80.3},
                        {'epoch': 2, 'loss': 2.93, 'focal': 2.0, 'shape': 0.9, 'validation_recall_at_5': 0.173,
                         'validation_false_positives_per_image_at_5': 4.9, 'seconds': 81.0}],
            'validation': signnet_measure(0.6, 4.8, 0.375, car=car), 'test': signnet_measure(0.55, 4.5, 0.25, car=car)}


def make_signnet_run(root: Path, name: str = '2026-10-30_1200-cutter', *, upto: str = 'done', hand: bool = False, **result) -> Path:
    """A run folder as scripts/run_signnet.sh NAME leaves it at the step `upto` (checks, signnet, evaluate) or at its
    end ('done'); hand: as nnslr train-signnet by hand leaves it (train.log, epochs in the old words)."""
    make_run(root, 'r1')
    run = root / 'runs' / name
    reached = ('checks', 'signnet', 'evaluate', 'done').index(upto)
    log = SIGNNET_LOG.format(name=name)
    if reached == 0:
        log = log.split('== 3/3')[0]
    if reached >= 1:
        write(run / 'signnet-best.pt', 'weights')
    if reached >= 2:
        log += SIGNNET_END.format(run=run).splitlines(keepends=True)[0]
    if reached >= 3:
        log += SIGNNET_END.format(run=run).splitlines(keepends=True)[1]
        write(run / 'result.json', signnet_result(**result))
    if hand:
        write(run / 'train.log', 'signnet epoch 1/3 loss 4.0846 validation recall@5 0.064\n'
                                 'signnet epoch 2/3 loss 2.9279 validation recall@5 0.173\n')
    else:
        write(run / 'signnet.log', log)
    return run


def test_a_signnet_run_is_a_cutter_run(root, index):
    make_signnet_run(root)
    runs = {run['name']: run for run in index.summaries()['runs']}
    run = runs['2026-10-30_1200-cutter']
    assert (run['kind'], run['state'], run['stage'], run['stage_label']) == ('cutter', 'finished', 'done', 'finished')
    assert run['started_at'].startswith('2026-10-30T12:00:00') and run['finished_at'] and run['evaluated']
    assert (run['split_sha256'], run['dataset_sha256'], run['start']['kind']) == (SPLIT_SHA, DATASET_SHA, 'random')
    assert run['problems'] == [] and runs['r1']['kind'] == 'training'
    # Its line among the cutters (Compare, the run list): what the car reads, its best 5 boxes of 0.3 or more on test.
    assert run['cutters'] == [{
        'kind': 'signnet', 'size': '896x288', 'families': 'all', 'finished': True, 'evaluated': True,
        'partition': 'test', 'score_threshold': 0.3, 'max_boxes': 5, 'recall': 0.25, 'found': 1, 'truth_boxes': 4,
        'speed_recall': 0.5, 'boxes_per_frame': 0.5, 'false_per_frame': 0.2, 'ms_per_frame': None,
        'fits_the_car': True}]


def test_an_old_signnet_result_is_read_at_the_cars_score(root, index):
    """A result.json written before measure's 'car' entry: the car's numbers are those at score 0.3 (the cap of 5
    boxes almost never bites there)."""
    run = make_signnet_run(root, car=False)
    [line] = next(item for item in index.summaries()['runs'] if item['name'] == run.name)['cutters']
    assert (line['kind'], line['recall'], line['false_per_frame'], line['found'], line['speed_recall']) == (
        'signnet', 0.5, 0.25, 2, None)
    test = index.detail(run.name)['signnet_run']['test']
    assert test['car']['recall'] == 0.5 and test['recall'] == 0.55  # the 5 best boxes at any score stay
    # Without a result.json: no line.
    (run / 'result.json').unlink()
    assert next(item for item in index.summaries()['runs'] if item['name'] == run.name)['cutters'] is None


@pytest.mark.parametrize('upto, stage, label', [('checks', 'checks', 'checking the dataset and the GPU'),
                                                ('signnet', 'signnet', 'training SignNet'),
                                                ('evaluate', 'evaluate', 'evaluating on test and validation')])
def test_the_steps_of_a_signnet_run(root, index, upto, stage, label):
    make_signnet_run(root, 's1', upto=upto)
    run = next(run for run in index.summaries()['runs'] if run['name'] == 's1')
    assert (run['kind'], run['state'], run['stage'], run['stage_label']) == ('cutter', 'incomplete', stage, label)
    assert run['finished_at'] is None and run['started_at']
    index.running_runs = lambda: {'s1': {'cutter'}}
    run = next(run for run in index.summaries()['runs'] if run['name'] == 's1')
    assert run['state'] == 'running'
    if upto == 'signnet':
        assert run['progress'] == 'SignNet · epoch 2 of 3'


def test_the_detail_of_a_signnet_run(root, index):
    run = make_signnet_run(root)
    detail = index.detail(run.name)
    assert detail['signnet_run'] == {
        'band': [896, 288], 'stride': 8, 'fuse_kernel': 1, 'widths': [16, 24, 32, 48], 'parameters': 62997, 'epochs': 3,
        'best_epoch': 2, 'families': 'all', 'split_sha256': SPLIT_SHA, 'fits_the_car': True,
        # The best 5 boxes at any score, and in 'car' as the car reads them (those of score 0.3 or more).
        'validation': {'recall': 0.6, 'false_per_frame': 4.8, 'car': {
            'recall': 0.375, 'false_per_frame': 0.2, 'found': 1, 'truth_boxes': 4, 'speed_recall': 0.5,
            'boxes_per_frame': 0.5, 'score_threshold': 0.3, 'max_boxes': 5}},
        'test': {'recall': 0.55, 'false_per_frame': 4.5, 'car': {
            'recall': 0.25, 'false_per_frame': 0.2, 'found': 1, 'truth_boxes': 4, 'speed_recall': 0.5,
            'boxes_per_frame': 0.5, 'score_threshold': 0.3, 'max_boxes': 5}}}
    # The curve of its log: the loss, the recall and the false boxes in the 5 best boxes of a frame.
    assert detail['curves'] == {'signnet': {'epochs': 3, 'points': [
        {'epoch': 1, 'train_loss': 4.0846, 'val_recall': 0.064, 'val_false_per_frame': 4.98, 'seconds': 80},
        {'epoch': 2, 'train_loss': 2.9279, 'val_recall': 0.173, 'val_false_per_frame': 4.9, 'seconds': 81}]}}
    assert (detail['cutter_run'], detail['classifier_run']) == (None, None)
    # Without its log, the curve of its result.json.
    (run / 'signnet.log').unlink()
    assert index.detail(run.name)['curves']['signnet']['points'][1] == {
        'epoch': 2, 'train_loss': 2.93, 'val_recall': 0.173, 'val_false_per_frame': 4.9, 'seconds': 81.0}
    json.dumps(index.detail(run.name), allow_nan=False)
    # Other runs have none.
    assert index.detail('r1')['signnet_run'] is None


def test_a_signnet_of_another_band_does_not_fit_the_car(root, index):
    run = make_signnet_run(root, 'wide', band=(1344, 416), stride=4)
    signnet = index.detail(run.name)['signnet_run']
    assert (signnet['band'], signnet['stride'], signnet['fits_the_car']) == ([1344, 416], 4, False)


def test_a_signnet_trained_by_hand(root, index):
    """The SignNets of 28 Oct 2026: signnet-best.pt, result.json and train.log, whose epochs carry the loss in words."""
    run = make_signnet_run(root, '2026-10-28_1028-signnet-twothirds-s8-f1', hand=True)
    detail = index.detail(run.name)
    assert (detail['kind'], detail['state'], detail['stage']) == ('cutter', 'finished', 'done')
    assert detail['signnet_run']['fits_the_car'] and detail['curves']['signnet']['points'][0]['train_loss'] == 4.08
    # Stopped before result.json: an incomplete SignNet, still training by its weights.
    (run / 'result.json').unlink()
    detail = index.detail(run.name)
    assert (detail['kind'], detail['state'], detail['stage'], detail['signnet_run']) == ('cutter', 'incomplete', 'signnet', None)
    assert [point['epoch'] for point in detail['curves']['signnet']['points']] == [1, 2]


def test_an_unreadable_signnet_result(root, index):
    run = make_signnet_run(root, 's1')
    (run / 'result.json').write_text('{"task": ')
    summary = next(item for item in index.summaries()['runs'] if item['name'] == 's1')
    assert (summary['kind'], summary['stage'], summary['problems']) == ('cutter', 'evaluate', ['result.json is unreadable'])


def test_a_chain_report_names_its_signnet_by_the_path_of_its_weights(root, index):
    """A chain measured with a SignNet (no id recorded): its id from <run>/signnet-best.pt, judged at its main score 0.3."""
    report = chain_report('test')
    report['cutter'] = {**report['cutter'], 'checkpoint': str(root / 'runs' / 's1' / 'signnet-best.pt'), 'id': None,
                        'input_size': [896, 288]}
    report['by_score_threshold']['0.3'] = report['by_score_threshold'].pop('0.25')
    report.update({'score_thresholds': [0.05, 0.1, 0.3], 'main_score_threshold': 0.3})
    headline = runs_module.chain_headline(runs_module.chain_digest(report))
    assert (headline['cutter_id'], headline['score_threshold'], headline['at_chain_score']) == ('cutter:s1/signnet', 0.3, True)
    assert headline['cutter_size'] == '896x288'


# -- a UserNet run: "Train UserNet" (scripts/run_usernet.sh NAME) ----------------------------------------------------
USERNET_LOG = '''== scripts/run_usernet.sh {name}
== 1/3 Check the training population (CPU-only) ==
== 2/3 Explicit CUDA smoke test ==
== 3/3 Train UserNet ==
usernet epoch 1/2 train_loss 3.1000 val_recall 0.200 val_false_per_frame 2.50 60s
usernet epoch 2/2 train_loss 2.4000 val_recall 0.350 val_false_per_frame 1.80 61s
'''
USERNET_END = 'usernet measure the best epoch 2 on validation and test\nUserNet complete: {run}\n'


def usernet_measure(frames: int, recall: float, precision: float, false_per_frame: float) -> dict:
    """validation or test of a UserNet's result.json (usernet.measure): the road users found at MIN_USER_SCORE."""
    found = {'recall': recall, 'precision': precision, 'false_per_frame': false_per_frame,
             'by_kind': {'person': {'recall': recall}, 'two_wheeler': {'recall': None}, 'vehicle': {'recall': recall}}}
    return {'frames': frames, 'at_min_score': found}


def make_usernet_run(root: Path, name: str = '2026-11-12_1200-users', *, upto: str = 'done') -> Path:
    """A run folder as scripts/run_usernet.sh NAME leaves it at the step `upto` (usernet, evaluate) or at its end."""
    run = root / 'runs' / name
    log = USERNET_LOG.format(name=name)
    if upto in ('evaluate', 'done'):
        log += USERNET_END.format(run=run)
    write(run / 'usernet.log', log)
    write(run / 'usernet-best.pt', 'weights')
    if upto == 'done':
        write(run / 'result.json', {
            'task': 'usernet', 'stage': 'bdd', 'epochs': 2, 'best_epoch': 2, 'uses_bdd': True,
            'validation_source': 'bdd', 'dataset_sha256': DATASET_SHA, 'split_sha256': SPLIT_SHA, 'start': None,
            'history': [{'epoch': 1, 'loss': 3.1, 'validation_recall': 0.2, 'validation_false_per_frame': 2.5, 'seconds': 60.2},
                        {'epoch': 2, 'loss': 2.4, 'validation_recall': 0.35, 'validation_false_per_frame': 1.8, 'seconds': 61.0}],
            'validation': usernet_measure(10, 0.35, 0.5, 1.8), 'test': usernet_measure(12, 0.3, 0.45, 2.0)})
    return run


def test_a_usernet_run_is_a_kind_of_its_own(root, index):
    run = make_usernet_run(root)
    make_signnet_run(root)
    runs = {item['name']: item for item in index.summaries()['runs']}
    summary = runs[run.name]
    assert (summary['kind'], summary['state'], summary['stage'], summary['stage_label']) == (
        'usernet', 'finished', 'done', 'finished')
    assert summary['finished_at'] and summary['evaluated'] and summary['problems'] == []
    assert (summary['split_sha256'], summary['dataset_sha256'], summary['start']['kind']) == (SPLIT_SHA, DATASET_SHA, 'random')
    # Its line: the stage, BDD100K's licence and, on test, the road users found at its score.
    assert summary['usernet'] == {'stage': 'bdd', 'uses_bdd': True, 'validation_source': 'bdd', 'best_epoch': 2,
                                  'partition': 'test', 'frames': 12, 'score_threshold': 0.5, 'recall': 0.3,
                                  'precision': 0.45, 'false_per_frame': 2.0}
    # A SignNet run beside it stays a cutter run without a UserNet line.
    assert (runs['2026-10-30_1200-cutter']['kind'], runs['2026-10-30_1200-cutter']['usernet']) == ('cutter', None)
    detail = index.detail(run.name)
    assert detail['curves']['usernet']['epochs'] == 2
    assert detail['curves']['usernet']['points'][1] == {'epoch': 2, 'train_loss': 2.4, 'val_recall': 0.35,
                                                        'val_false_per_frame': 1.8, 'seconds': 61}
    assert detail['usernet_run']['test']['recall'] == 0.3 and 'history' not in detail['usernet_run']
    # Without its log, the curve of its result.json.
    (run / 'usernet.log').unlink()
    assert index.detail(run.name)['curves']['usernet']['points'][0]['train_loss'] == 3.1
    json.dumps(index.detail(run.name), allow_nan=False)
    assert index.detail('2026-10-30_1200-cutter')['usernet_run'] is None


def test_a_usernet_run_that_went_on_from_another(root, index):
    run = make_usernet_run(root)
    result = json.loads((run / 'result.json').read_text())
    write(run / 'result.json', {**result, 'stage': 'drives', 'start': {'run': 'older_u', 'epoch': 9, 'stage': 'bdd',
                                                                        'uses_bdd': True, 'path': '/x/usernet-best.pt'}})
    summary = next(item for item in index.summaries()['runs'] if item['name'] == run.name)
    assert (summary['start']['kind'], summary['start']['run'], summary['start']['label']) == (
        'run', 'older_u', 'Continued from run older_u')
    assert (summary['usernet']['stage'], summary['usernet']['uses_bdd']) == ('drives', True)


@pytest.mark.parametrize('upto, stage, label', [('usernet', 'usernet', 'training UserNet'),
                                                ('evaluate', 'evaluate', 'measuring on validation and test')])
def test_the_steps_of_a_usernet_run(root, index, upto, stage, label):
    make_usernet_run(root, 'u1', upto=upto)
    run = next(run for run in index.summaries()['runs'] if run['name'] == 'u1')
    assert (run['kind'], run['state'], run['stage'], run['stage_label']) == ('usernet', 'incomplete', stage, label)
    assert run['finished_at'] is None and run['started_at'] and run['usernet'] is None
    index.running_runs = lambda: {'u1': {'usernet'}}
    run = next(run for run in index.summaries()['runs'] if run['name'] == 'u1')
    assert run['state'] == 'running'
    if upto == 'usernet':
        assert run['progress'] == 'UserNet · epoch 2 of 2'


def test_an_unreadable_usernet_result(root, index):
    run = make_usernet_run(root, 'u1')
    (run / 'result.json').write_text('{"task": ')
    summary = next(item for item in index.summaries()['runs'] if item['name'] == 'u1')
    assert (summary['kind'], summary['stage'], summary['problems']) == ('usernet', 'evaluate', ['result.json is unreadable'])
