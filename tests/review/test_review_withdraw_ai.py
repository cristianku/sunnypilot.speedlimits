"""review-withdraw-ai: every frame an AI confirmed goes back to not confirmed, with history, and leaves the dataset."""
from __future__ import annotations

import json

from nnslr_tools import cli
from nnslr_tools.annotations import load_jsonl
from nnslr_tools.review_db import ReviewStore, db_path, import_from_db, save_frame_review
from review_helpers import ROUTE, RUN_ID, make_report, page_body


def save(root, store, row, reviewer):
    return save_frame_review(store, root, route_id=ROUTE, run_id=RUN_ID, report_row=row,
                             body=page_body(row, reviewer=reviewer), client='test')


def run(capsys, root, *args):
    code = cli.main(['review-withdraw-ai', '--data-root', str(root), *args])
    return code, json.loads(capsys.readouterr().out)


def test_withdraw_ai_reviews(tmp_path, capsys):
    rows = make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    first = save(tmp_path, store, rows[0], 'Codex')
    save(tmp_path, store, rows[1], 'Cristian')
    store.reclassify_as_ai('Codex', client='test')
    output = tmp_path / 'objects.jsonl'
    assert import_from_db(tmp_path, output, store=store)['imported_frame_count'] == 2
    code, report = run(capsys, tmp_path, '--dry-run')
    assert code == 0 and report['dry_run'] and report['frames'] == 1 and 'copy' not in report
    assert report['by_reviewer'] == {'ai/Codex': 1}
    assert store.review(ROUTE, RUN_ID, '0/0')['confirmed']
    code, report = run(capsys, tmp_path)
    assert code == 0 and report['frames'] == 1 and report['by_route_run'] == {ROUTE + ' ' + RUN_ID: 1}
    copy = ReviewStore(tmp_path / report['copy'])
    assert copy.review(ROUTE, RUN_ID, '0/0')['confirmed']
    withdrawn = store.review(ROUTE, RUN_ID, '0/0')
    assert not withdrawn['confirmed'] and withdrawn['revision'] == 3 and withdrawn['reviewer'] == 'ai/Codex'
    assert withdrawn['payload']['detections'] == first['payload']['detections']
    history = store.history(ROUTE, RUN_ID, '0/0')
    assert [(h['revision'], h['confirmed'], h['client']) for h in history][-1] == (3, False, 'review-withdraw-ai')
    person = store.review(ROUTE, RUN_ID, '0/1')
    assert person['confirmed'] and person['revision'] == 1
    # Running it again changes nothing.
    code, report = run(capsys, tmp_path)
    assert report['frames'] == 0
    # The import takes the withdrawn frame out of the dataset; the person's stays.
    result = import_from_db(tmp_path, output, store=store)
    assert result['removed_withdrawn_frame_count'] == 1
    assert {record['reviewer'] for record in load_jsonl(output)} == {'Cristian'}
