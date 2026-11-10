"""The pages of the site: one shared look (shared.css, the tokens of distill-web), the tabs, the Account
menu (appearance, settings, sign out), the Stato page, and the scripts built from the Vue and TypeScript of
web/src."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from nnslr_tools.pages import asset
from review_helpers import ROUTE, RunningServer, make_report

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def served(tmp_path):
    make_report(tmp_path)
    server = RunningServer(tmp_path)
    yield server
    server.close()


# Each page and the section its tab marks; the review page of a drive has no tab bar (its controls are at the bottom).
PAGES = [('/', 'Overview'), ('/prepare', 'Data'), ('/dataset', 'Data'), ('/train', 'Training'), ('/evaluate', 'Models'),
         ('/models', 'Models'), ('/drive', 'Models'), ('/publish', 'Models'), ('/speed-review', 'Data'),
         ('/account', None), (f'/route/{ROUTE}/', None), ('/nonexistent', None)]


@pytest.mark.parametrize('path, tab', PAGES)
def test_every_page_has_the_shared_look_the_account_menu_and_its_script(served, path, tab):
    status, response, data = served.request('GET', path)
    page = data.decode()
    nonce = response.getheader('Content-Security-Policy').split("'nonce-")[1].split("'")[0]
    # The theme is applied before the first paint; the page script carries the same nonce.
    assert f'<script nonce="{nonce}">try{{var t=localStorage.getItem(\'nnslr-theme\')' in page
    assert page.count(f'<script nonce="{nonce}">') == 2
    assert asset('shared.css') in page
    # The theme choice is in the Account menu, not a separate switch; the review page draws its own bar
    # (and the menu in it) with Vue.
    if not path.startswith('/route/'):
        assert 'id="account-menu"' in page and 'id="account-btn"' in page
    assert 'theme-toggle' not in page
    assert '__NNSLR_' not in page.replace('"server": false', '')
    tabs = re.search(r'<nav class="tabs".*?</nav>', page, re.S)
    if tab is not None:
        assert tabs and re.findall(r'>([^<]+)</a>', tabs.group(0)) == ['Overview', 'Data', 'Training', 'Models']
        assert f'aria-current="page">{tab}</a>' in tabs.group(0)
        assert '<nav class="tabbar"' in page
    if path.startswith('/route/'):
        assert '<nav class="tabbar"' not in page


@pytest.mark.parametrize('path, title', [('/', 'Overview'), ('/prepare', 'Drives'), ('/speed-review', 'Speed review'),
                                         ('/drive', 'Drive'),
                                         ('/account', 'Settings'), (f'/route/{ROUTE}/', 'Sign review'),
                                         ('/nonexistent', 'Page not found')])
def test_every_page_title_names_the_page_as_the_bar_does(served, path, title):
    page = served.request('GET', path)[2].decode()
    assert f'<title>Speed limits · {title}</title>' in page
    # The connection state is in the bar of every page, empty while the site answers.
    assert '<span id="live-text"></span>' in page


def test_error_pages_speak_plainly_and_lead_back(served):
    status, response, data = served.request('GET', '/nonexistent')
    page = data.decode()
    assert status == 404 and '<h1>Page not found</h1>' in page and 'HTTP code' not in page
    assert '<a class="btn back" href="/">← Overview</a>' in page
    status, response, data = served.request('GET', '/route/foo/')
    page = data.decode()
    assert status == 404 and '<h1>Drive not found</h1>' in page
    assert 'This drive is not on the server, or it is not prepared yet.' in page and 'pre-annotation' not in page
    assert '<a class="btn back" href="/speed-review">← Speed review</a>' in page


def test_the_empty_route_list_says_what_to_do_in_words(tmp_path):
    from nnslr_tools.pages import render_index
    page = render_index([], tmp_path / 'reviews.sqlite3', 'nonce')
    assert 'No drives to review yet' in page and 'nnslr' not in page.split('<main')[1].split('Technical details')[0]


def test_control_page_reads_the_panel_status(served):
    page = served.request('GET', '/')[2].decode()
    script = asset('js/control.js')
    assert script in page and '/api/panel/status' in script and 'cores' in script and 'run-row' in script
    # The processes on the GPU are in the server's panel of the top bar, on every page.
    assert 'tm-gpu' in asset('js/site.js')
    # The classifier with "not a sign": its action on Train, its sections on Evaluate.
    assert 'train_classifier' in script and 'chain-eval' in script and 'classifier-eval' in script
    # The Train tab by role, one at a time: the cutter (its sizes and signs), the classifier, the decider, the pipeline.
    for marker in ('train_cutter', 'train-sections', 'train-decider-note', 'Applicability', 'Pipeline', 'cutter-eval'):
        assert marker in script, marker


def test_built_scripts_are_the_build_of_the_typescript():
    """src/nnslr_tools/web/js is committed: it must be what helpers/build_web.sh makes of web/src (vue-tsc, Vite)."""
    if not (REPO / 'web/node_modules/vite').is_dir() or shutil.which('uvx') is None:
        pytest.skip('web/node_modules missing: run helpers/build_web.sh once')
    result = subprocess.run([str(REPO / 'helpers/build_web.sh'), '--check'], capture_output=True, text=True,
                            env=dict(os.environ), timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_page_script_has_its_source():
    built = {path.stem for path in (REPO / 'src/nnslr_tools/web/js').glob('*.js')}
    sources = {path.stem for path in (REPO / 'web/src/pages').glob('*.ts')}
    assert built == sources
