"""The HTML pages of the review site.

Templates, styles and scripts live in ``nnslr_tools/web``: ``shared.css`` (tokens and frame, the look of
distill-web), ``site.css`` (route list, error pages, sign-in, account), the page templates and
``js/<page>.js``, built from the Vue and TypeScript of ``web/src`` by ``helpers/build_web.sh``; a page with
a script renders its content in ``<div id="app">``. Every page gets
its script inline with the nonce of its Content-Security-Policy. Files are read on every request: a
change to them is live at the next page load, a change to this module after a restart.

Placeholders of the templates:

- ``__NNSLR_THEME__``: the inline script that applies the dark / light choice before the first paint;
- ``__NNSLR_SHARED_CSS__``, ``__NNSLR_SITE_CSS__``: the two style sheets;
- ``__NNSLR_TOPBAR__``: brand, live state, Account menu and tabs;
- ``__NNSLR_BAR_END__``: only the Account menu (the review page has its own bar);
- ``__NNSLR_TITLE__``: the window title, "Speed limits · " and the name of the page as the bar says it;
- ``__NNSLR_SCRIPT__``: the page script; ``__NNSLR_NONCE__``: the nonce, where a template needs it;
- ``__NNSLR_DATA__``: the frames of the review page.
"""
from __future__ import annotations

import html
import json
from importlib.resources import files
from pathlib import Path

from nnslr_tools.gps import country_names
from nnslr_tools.review_access import SIGN_OUT_ADDRESS
from nnslr_tools.review_hints import empty_stats

# The logo: a Swiss 50 sign in the yellow corners of the box the review selects (the model reading a sign), on
# the dark tile .brand-mark of shared.css; the digits are paths, the same on every system. The same drawing as
# web/src/components/BrandMark.vue, for the Vue pages.
BRAND_MARK = ('<span class="brand-mark" aria-hidden="true"><svg viewBox="0 0 32 32" focusable="false">'
              '<path d="M4.2 9V5.6c0-.8.6-1.4 1.4-1.4H9M23 4.2h3.4c.8 0 1.4.6 1.4 1.4V9M27.8 23v3.4c0 .8-.6 1.4-1.4 '
              '1.4H23M9 27.8H5.6c-.8 0-1.4-.6-1.4-1.4V23" fill="none" stroke="#ffd60a" stroke-width="1.6" '
              'stroke-linecap="round"/><circle cx="16" cy="16" r="9" fill="#fff" stroke="#e3001b" stroke-width="2.7"/>'
              '<g fill="none" stroke="#111" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
              'transform="matrix(.88 0 0 .88 1.92 1.92)"><path d="M15.2 12.2H11.4l-.3 3.5c.6-.5 1.3-.7 2-.7 1.4 0 2.3 '
              '1 2.3 2.4 0 1.5-1 2.4-2.4 2.4-1 0-1.8-.5-2.2-1.2"/><rect x="17.2" y="12.2" width="3.9" height="7.6" '
              'rx="1.95"/></g></svg></span>')
# The server at the right of the top bar, on every page with the tabs: a pill with how busy it is, the gauges
# (GPU, its memory and temperature, CPU, RAM, disk) in a panel under it (web/src/components/TopMachine.vue,
# mounted here by the page script). data-feed: the control pages have it from their status, the others ask
# /api/panel/system.
TABS_MACHINE = '<div class="tabs-machine" id="tabs-machine" data-feed="{feed}"></div>'
# The comma before the server: reachable or not, and Download new drives (web/src/components/TopComma.vue).
COMMA_PILL = '<div class="tabs-machine" id="comma-pill"></div>'

# The four sections of the site, in the order of the work: (key, address, label, its pages). A section with pages
# shows them under its title (sub_nav; web/src/control/SectionHead.vue draws the same): Data holds the drives, their
# review, Speed review (the speed a person wants held along a drive, with the map's limits it proposes), Sign distance
# (where the comma's vision says a sign is passed against the frame a person marks) and the training dataset; Models the
# versions to choose, their results and the package to publish.
# Account is the icon at the right. A page is (key of the current page, address, label).
SECTIONS = (
    ('overview', '/', 'Overview', ()),
    ('data', '/prepare', 'Data', (('prepare', '/prepare', 'Drives'), ('review', '/routes', 'Sign review'),
                                  ('drivereview', '/drive-review', 'Drive review'),
                                  ('speedreview', '/speed-review', 'Speed review'),
                                  ('signdistance', '/sign-distance', 'Sign distance'),
                                  ('dataset', '/dataset', 'Training dataset'), ('reviewers', '/reviewers', 'Reviewers'))),
    ('train', '/train', 'Training', ()),
    ('models', '/models', 'Models', (('models', '/models', 'Versions'), ('evaluate', '/evaluate', 'Results'),
                                     ('regression', '/regression', 'Regression'), ('publish', '/publish', 'Publish'))),
)
# Every page of the site: (key, address, label), the page label for a page of a section ("Drives").
SITE_TABS = tuple(page for key, href, label, pages in SECTIONS for page in (pages or ((key, href, label),)))
# The section of each page ("review" → "data").
SECTION_OF = {page[0]: key for key, _, _, pages in SECTIONS for page in (pages or ((key, '', ''),))}
# The tabs that are one Vue app (web/src/control), by address: the app shows the tab of the address. The
# route list (/routes) is a page of its own.
CONTROL_TABS = {'/': 'overview', '/prepare': 'prepare', '/dataset': 'dataset',
                '/drive-review': 'drivereview', '/speed-review': 'speedreview', '/sign-distance': 'signdistance',
                '/reviewers': 'reviewers', '/train': 'train', '/evaluate': 'evaluate', '/models': 'models', '/regression': 'regression', '/publish': 'publish'}
# The icons of the sections in the phone's tab bar (24 × 24, stroked like web/src/components/icons).
SECTION_ICONS = {
    'overview': '<path d="M4 11.5 12 5l8 6.5V19a1 1 0 0 1-1 1h-4.5v-5h-5v5H5a1 1 0 0 1-1-1z"/>',
    'data': '<ellipse cx="12" cy="6" rx="7" ry="2.6"/><path d="M5 6v6c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6V6"/>'
            '<path d="M5 12v6c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6v-6"/>',
    'train': '<path d="M4 19h16"/><path d="M5 16l4.5-5 3.5 3 6-7"/><path d="M15 7h4v4"/>',
    'models': '<path d="M12 3.5 19.5 7.7v8.6L12 20.5l-7.5-4.2V7.7z"/><path d="M4.8 7.9 12 12l7.2-4.1M12 12v8.3"/>',
}
# Earlier addresses of these pages, kept working, and the names of the bar typed as addresses (Settings, Review).
# The home of a reviewer from outside. / cannot be: on the public names it is the page of the project, open to all, and
# what is open carries nobody's name. The administrator has none of their own and lands on the Overview.
GUEST_HOME = '/reviews'
OLD_ADDRESSES = {GUEST_HOME: '/', '/panoramica': '/', '/pannello': '/', '/stato': '/', '/prepara': '/prepare', '/allena': '/train',
                 '/settings': '/account', '/review': '/routes', '/roads': '/speed-review'}
# The name of each page in the window title.
TAB_TITLES = {key: label for key, _, label in SITE_TABS}
# Dark (the default) or the choice kept in this browser, applied before the first paint
# (web/src/common/theme.ts changes it).
THEME_HEAD = ('<script nonce="{nonce}">try{{var t=localStorage.getItem(\'nnslr-theme\');'
              'if(t===\'light\'||t===\'auto\')document.documentElement.dataset.theme=t}}catch(e){{}}</script>')
# The Account menu at the right of every top bar (web/src/components/AccountMenu.vue, mounted here by the
# page script): who is signed in, the appearance (dark, light, automatic), the settings and sign out. The
# icon is already in the HTML, so nothing moves while the script starts.
ACCOUNT_MENU = (
    '<div class="account-menu" id="account-menu" data-current="{current}">'
    '<button class="icon-btn account-btn" type="button" id="account-btn" aria-haspopup="dialog" aria-expanded="false"'
    ' aria-controls="account-pop" aria-label="Account and settings" title="Account">'
    '<svg viewBox="0 0 20 20" aria-hidden="true">'
    '<circle cx="10" cy="7" r="3.4" fill="none" stroke="currentColor" stroke-width="1.8"/>'
    '<path d="M3.6 17.2c.9-3.1 3.4-4.8 6.4-4.8s5.5 1.7 6.4 4.8" fill="none" stroke="currentColor"'
    ' stroke-width="1.8" stroke-linecap="round"/></svg></button>'
    '<div class="account-pop" id="account-pop" role="dialog" aria-label="Account and settings" hidden></div></div>')
LIVE = ('<span class="live" role="status"><span class="dot" id="live-dot"></span>'
        '<span id="live-text"></span></span>')
# Groups of the route list, in this order: the work in progress first, finished routes last.
ROUTE_GROUPS = (('doing', 'In progress'), ('todo', 'To start'), ('done', 'Done'))


def asset(name: str) -> str:
    """A file of nnslr_tools/web, read now (the server picks up a change at the next request)."""
    return files('nnslr_tools').joinpath('web').joinpath(name).read_text(encoding='utf-8')


def page_script(page: str, nonce: str) -> str:
    """The built script of a page (web/src/pages/<page>.ts), inline with the nonce."""
    script = asset('js/' + page + '.js')
    # Inline, a "</script" in a string would end the element early.
    if '</script' in script.lower():
        raise ValueError('web/js/' + page + '.js contains "</script": escape it in the TypeScript source')
    return '<script nonce="' + nonce + '">\n' + script + '</script>'


def bar_end(current: str = '') -> str:
    """The Account menu, at the right of every top bar."""
    return ACCOUNT_MENU.format(current='true' if current == 'account' else 'false')


def section_paths(pages: tuple, href: str) -> str:
    """The addresses of a section, for the busy dot of its tab (web/src/composables/busy.ts)."""
    return ' '.join(page[1] for page in pages) or href


# [guest-reviewers] - START
def guest_bar(current: str = '') -> str:
    """The top bar of a reviewer from outside: the brand, their list and the Account menu, nothing of the server.
    data-guest tells the page scripts (web/src/common/guest.ts) to leave out what only the administrator can use."""
    return ('<header class="topbar" data-guest="true"><div class="topbar-inner site-bar">'
            '<a class="brand" href="' + GUEST_HOME + '" title="My reviews">' + BRAND_MARK
            + '<span class="brand-name">Speed limits</span></a>'
            '<nav class="tabs" aria-label="Sections"><a href="' + GUEST_HOME + '" data-section="overview">My reviews</a></nav>'
            '<div class="bar-end">' + bar_end(current) + '</div></div></header>')
# [guest-reviewers] - END


def top_bar(current: str = '', *, live: bool = False, signed_in: bool = True, tab_bar: bool = True) -> str:
    """Brand, the four sections, then the comma and server pills, the live state and the Account menu, in one row
    (the sections in a second one on a narrow screen). On a phone the sections are a tab bar at the bottom instead
    (tab_bar; not on the review page of a drive, whose controls are at the bottom). current: the key of the page to
    mark (its section is marked), or 'account'. The sign-in page (signed_in=False) shows only the brand."""
    brand = ('<a class="brand" href="/" title="Overview">' + BRAND_MARK
             + '<span class="brand-name">Speed limits</span></a>')
    if not signed_in:
        return '<header class="topbar"><div class="topbar-inner">' + brand + '</div></header>'
    section = SECTION_OF.get(current, '')
    tabs = ''.join('<a href="' + href + '" data-section="' + key + '" data-paths="' + section_paths(pages, href) + '"'
                   + (' aria-current="page"' if key == section else '') + '>' + label + '</a>'
                   for key, href, label, pages in SECTIONS)
    bar = ''
    if tab_bar:
        bar = ('<nav class="tabbar" aria-label="Sections">' + ''.join(
            '<a href="' + href + '" data-section="' + key + '" data-paths="' + section_paths(pages, href) + '"'
            + (' aria-current="page"' if key == section else '') + '><svg viewBox="0 0 24 24" aria-hidden="true">'
            + SECTION_ICONS[key] + '</svg><span>' + label + '</span></a>' for key, href, label, pages in SECTIONS)
            + '</nav>')
    # The connection state on every page (web/src/composables/connection.ts): fed by the status of the control pages
    # (live), by the server pill elsewhere.
    return ('<header class="topbar"><div class="topbar-inner site-bar">' + brand
            + '<nav class="tabs" aria-label="Sections">' + tabs + '</nav>'
            + '<div class="bar-end">' + LIVE + COMMA_PILL
            + TABS_MACHINE.format(feed='status' if live else 'system') + bar_end(current) + '</div></div></header>' + bar)


def sub_nav(current: str) -> str:
    """The pages of the current page's section, under its title, as an iOS segmented control (SectionHead.vue draws the
    same on the control pages); '' for a section without pages."""
    for key, _, label, pages in SECTIONS:
        if pages and any(page[0] == current for page in pages):
            return ('<nav class="subnav" aria-label="' + label + '">' + ''.join(
                '<a href="' + href + '"' + (' aria-current="page"' if page == current else '') + '>' + name + '</a>'
                for page, href, name in pages) + '</nav>')
    return ''


def fill(template: str, nonce: str, page: str, current: str = '', *, live: bool = False,
         signed_in: bool = True, data: str = '', title: str = '', guest: bool = False) -> str:
    """A template with its placeholders replaced (see the module docstring)."""
    values = {
        '__NNSLR_THEME__': THEME_HEAD.format(nonce=nonce),
        '__NNSLR_TITLE__': html.escape('Speed limits · ' + title if title else 'Speed limits'),
        '__NNSLR_SHARED_CSS__': asset('shared.css'),
        '__NNSLR_SITE_CSS__': asset('site.css'),
        '__NNSLR_TOPBAR__': guest_bar(current) if guest else top_bar(current, live=live, signed_in=signed_in,
                                                                     tab_bar=page != 'review'),
        '__NNSLR_BAR_END__': bar_end(current),
        '__NNSLR_SCRIPT__': page_script(page, nonce),
        '__NNSLR_NONCE__': nonce,
        # Last: the frames are data and must never be scanned for placeholders.
        '__NNSLR_DATA__': data,
    }
    for name, value in values.items():
        if name in template:
            template = template.replace(name, value)
    return template


def render_review(rows: list[dict], run_id: str, *, route_id: str = '', server: bool = False,
                  nonce: str = '', hints: dict | None = None, hint_stats: dict | None = None,
                  suspicious: dict | None = None, routes: list[dict] | None = None,
                  proposer: str | None = None, busy: dict | None = None, videos: dict | None = None,
                  guest: bool = False) -> str:
    """The review page with its data embedded. Hints and suspicious flags travel apart from the
    frame rows, so the JSONL export (which copies the rows) never contains them. proposer: who proposed the
    signs of the report (vietnam, ours, qwen; None when not known), for the page's "More". busy: the drive is read
    only, Propose again works on it (nnslr_tools.reproposing.public), else None. videos: the segments with their road
    video on the server (ReviewService.videos), None offline. guest: for a reviewer from outside (guest_bar)."""
    data = {'run_id': run_id, 'route_id': route_id, 'server': server, 'frames': rows, 'proposer': proposer,
            'busy': busy, 'videos': videos,
            'hints': hints if hints is not None else {},
            'hint_stats': hint_stats if hint_stats is not None else empty_stats(),
            'suspicious': suspicious if suspicious is not None else {},
            'routes': routes if routes is not None else [],
            'countries': country_names()}
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False)
    payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    # The same top bar as every page (the Review tab marked); the offline report has no site: only the brand.
    # The window title names the drive ("Herznach → Zürich"), as the route list does; Review without a name.
    drive = next((link.get('title') for link in (routes or []) if link.get('route_id') == route_id), None)
    return fill(asset('review.html'), nonce, 'review', 'review', signed_in=server, data=payload,
                title=drive or 'Sign review', guest=guest)


# [control-panel] - START
def render_control(nonce: str, tab: str) -> str:
    """Overview, Prepare, Train, Evaluate, Models: one page (control.html, web/src/control), the tab marked in the
    bar; data from /api/panel/status (and /api/panel/runs for Evaluate, /api/panel/models for Models)."""
    return fill(asset('control.html'), nonce, 'control', tab, live=True, title=TAB_TITLES.get(tab, ''))
# [control-panel] - END


# [guest-reviewers] - START
def render_guest(nonce: str) -> str:
    """The site of a reviewer from outside: their list and the pages of their drives, one app (web/src/control/GuestApp.vue)
    in the frame of the control pages. It holds no data: all comes from the calls the guest may make."""
    return fill(asset('control.html'), nonce, 'guest', title='My reviews', guest=True)
# [guest-reviewers] - END


# [site-login] - START
def render_login(nonce: str) -> str:
    """The sign-in page (also the first visit, which creates the administrator)."""
    return fill(asset('login.html'), nonce, 'login', signed_in=False, title='Sign in')


def render_account(nonce: str) -> str:
    """The Account & Settings page (web/src/account): who is signed in, sign out, change the password, the
    appearance, the AI reviewer. On 127.0.0.1, where nobody signs in, the app shows the settings alone."""
    return fill(asset('account.html'), nonce, 'account', 'account', title='Settings')
# [site-login] - END


def simple_page(title: str, current: str, nonce: str, main: str, *, guest: bool = False) -> str:
    """A page without a template of its own: the frame, the content, the script of the Account menu; for a guest
    only the brand and no script, as someone refused may be nobody the site knows."""
    return '\n'.join([
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        '<meta name="color-scheme" content="light dark">',
        THEME_HEAD.format(nonce=nonce),
        '<title>Speed limits · ' + html.escape(title) + '</title>',
        '<style>' + asset('shared.css') + asset('site.css') + '</style></head><body>',
        top_bar(current, signed_in=not guest),
        main,
        '' if guest else page_script('site', nonce),
        '</body></html>',
    ])


def review_percent(confirmed: int, total: int) -> int:
    """The share of frames confirmed, as every page shows it: the nearest whole number, 100 only when all are
    (web/src/common/format.ts reviewPercent says the same)."""
    if not total:
        return 0
    return 100 if confirmed >= total else min(99, round(confirmed * 100 / total))


def _plural(count: int, one: str, many: str) -> str:
    if count == 1:
        return '1 ' + one
    return str(count) + ' ' + many


def route_state(confirmed: int, total: int) -> str:
    """doing, todo or done: the group of a route in the list."""
    if total and confirmed >= total:
        return 'done'
    if confirmed == 0:
        return 'todo'
    return 'doing'


def _route_row(summary: dict, *, ai_reviewer: bool = True) -> str:
    total = summary['frame_count']
    confirmed = summary['confirmed_count']
    by_ai = min(summary.get('confirmed_by_ai') or 0, confirmed)
    # The bar: the frames people confirmed, then the AI's share hatched (as on the review page's timeline).
    by_people_percent = round((confirmed - by_ai) * 100 / total) if total else 0
    by_ai_percent = round(by_ai * 100 / total) if total else 0
    state = route_state(confirmed, total)
    # What opening the route means: carry on (the one obvious action), start, or look again at a finished one.
    action_class, action = {'doing': ('btn small primary open', 'Resume'),
                            'todo': ('btn small open', 'Start'),
                            'done': ('btn small open again', 'Open')}[state]
    suspicious = summary.get('suspicious_count', 0)
    # Only what says something: how many frames, while the drive is to review how many have a sign the model
    # proposed (not confirmed signs: the review page's "With proposals"), the suspicious ones when there are.
    stats = ['<span>' + _plural(total, 'frame', 'frames').replace(str(total), f'{total:,}', 1) + '</span>']
    if state != 'done':
        stats.append('<span>' + f"{summary['frames_with_proposals']:,}" + ' with proposals</span>')
    if suspicious:
        stats.append('<span class="sus">' + _plural(suspicious, 'suspicious', 'suspicious') + '</span>')
    waiting = summary.get('waiting_for_person') or 0
    if waiting:
        # Saved by the site's AI reviewer: a person confirms them before they count (review_db.SITE_AI_REVIEWER).
        stats.append('<span class="sus">' + f'{waiting:,}' + ' by AI to confirm</span>')
    from nnslr_tools.preannotate import PROPOSER_NAMES
    # Mid-sentence, "our networks" in lower case ("signs by our networks"); the other names are names.
    proposer = summary.get('proposer') or 'vietnam'
    proposer_name = 'our networks' if proposer == 'ours' else PROPOSER_NAMES.get(proposer, 'Vietnamese YOLO')
    stats.append('<span>signs by ' + html.escape(proposer_name) + '</span>')
    route = html.escape(summary['route_id'])
    # The drive (when, how long, from where to where); the route's code only in the tooltip.
    drive = summary.get('drive') or {}
    when = [part for part in (drive.get('when'), _plural(drive['minutes'], 'min', 'min') if drive.get('minutes') else None) if part]
    name = html.escape(drive.get('title') or summary['route_id'])
    detail = html.escape(' · '.join(when))
    # [ai-reviewer] - START
    # The frames still to confirm can be reviewed by the AI of Settings: the button opens its sheet (site.ts,
    # AiRouteSheet), which names the drive and how many frames are left.
    ai = ''
    if confirmed < total and ai_reviewer:
        drive_name = ' · '.join(part for part in (drive.get('title'), drive.get('when')) if part) or summary['route_id']
        ai = ('<button type="button" class="btn small ai-route" data-ai-route="' + route + '" data-ai-name="'
              + html.escape(drive_name) + '" data-ai-left="' + str(total - confirmed) + '">Review with AI</button>')
    # [ai-reviewer] - END
    return ('<li><a class="card-row ' + state + '" href="' + html.escape(summary['url']) + '">'
            + '<span class="state-icon" aria-hidden="true">' + ('✓' if state == 'done' else '') + '</span>'
            + '<span class="route" title="' + route + '"><span class="drive">' + name + '</span>'
            + '<span class="drive-when">' + detail + '</span></span>'
            + '<span class="count">' + f'{confirmed:,}' + ' of ' + f'{total:,}'
            + ' <span class="unit">confirmed</span>'
            + ('<span class="by-ai">' + f'{by_ai:,}' + ' by AI</span>' if by_ai else '') + '</span>'
            + '<span class="bar" aria-hidden="true"><span style="width:' + str(by_people_percent) + '%"></span>'
            + ('<span class="ai" style="width:' + str(by_ai_percent) + '%"></span>' if by_ai else '') + '</span>'
            + '<span class="stats">' + ''.join(stats) + '</span>'
            + '<span class="' + action_class + '">' + action + '</span></a>'
            # Filled by the page script while the route is proposed again or reviewed by the AI (web/src/pages/site.ts):
            # under the row and its AI button, which stay side by side.
            + '<span class="row-live" data-live="' + route + '" hidden></span>' + ai + '</li>')


def _group_sub(key: str, rows: list[dict], short: bool = False) -> str:
    """What a group of the route list holds, in a few words: the frames left, or confirmed (short: for a tile)."""
    total = sum(row['frame_count'] for row in rows)
    confirmed = sum(min(row['confirmed_count'], row['frame_count']) for row in rows)
    if key == 'done':
        return f'{confirmed:,} frames' + ('' if short else ' confirmed')
    return f'{total - confirmed:,}' + (' left' if short else ' frames left')


def _review_overview(summaries: list[dict], grouped: dict[str, list[dict]]) -> str:
    """The route list at a glance, before the groups: the frames confirmed of every drive (by people, then the AI
    assistants' share hatched), what waits for a person, and a tile per group that opens it."""
    total = sum(row['frame_count'] for row in summaries)
    confirmed = sum(min(row['confirmed_count'], row['frame_count']) for row in summaries)
    by_ai = sum(min(row.get('confirmed_by_ai') or 0, row['confirmed_count']) for row in summaries)
    waiting = sum(row.get('waiting_for_person') or 0 for row in summaries)
    suspicious = sum(row.get('suspicious_count') or 0 for row in summaries)
    people = round((confirmed - by_ai) * 100 / total, 2) if total else 0
    ai = round(by_ai * 100 / total, 2) if total else 0
    notes = [f'{confirmed - by_ai:,} by you', f'{by_ai:,} by AI assistants'] if by_ai else []
    if waiting:
        notes.append('<span class="sus">' + f'{waiting:,}' + ' by AI to confirm</span>')
    if suspicious:
        notes.append('<span class="sus">' + f'{suspicious:,}' + ' suspicious</span>')
    tiles = []
    for key, title in ROUTE_GROUPS:
        rows = grouped[key]
        tiles.append('<button type="button" class="rv-tile ' + key + '" data-open-group="' + key + '" aria-controls="group-'
                     + key + '"' + (' disabled' if not rows else '') + '><span class="rv-tile-label">' + title
                     + '</span><span class="rv-tile-n">' + str(len(rows)) + '</span><span class="rv-tile-sub">'
                     + (_group_sub(key, rows, short=True) if rows else 'none') + '</span></button>')
    return ('<section class="card rv-overview" aria-label="Sign review at a glance">'
            '<p class="rv-big"><strong>' + str(review_percent(confirmed, total)) + '%</strong><span>of the frames confirmed · '
            + f'{confirmed:,}' + ' of ' + f'{total:,}' + ' in ' + _plural(len(summaries), 'drive', 'drives') + '</span></p>'
            '<span class="rv-bar" aria-hidden="true"><span style="width:' + str(people) + '%"></span>'
            + ('<span class="ai" style="width:' + str(ai) + '%"></span>' if by_ai else '') + '</span>'
            + ('<p class="rv-notes">' + ' · '.join(notes) + '</p>' if notes else '')
            + '<div class="rv-tiles">' + ''.join(tiles) + '</div></section>')


def render_index(summaries: list[dict], database: Path, nonce: str = '', *, ai_reviewer: bool = True) -> str:
    """The route list, grouped: in progress, to start, done. ai_reviewer: an AI reviewer is chosen in Settings
    (each drive left to review gets "Review with AI"); without one, a single link to set it up."""
    main = ['<main class="page">',
            '<header class="section-head"><h1 class="head-title">Data</h1>' + sub_nav('review') + '</header>']
    if not ai_reviewer and any(s['confirmed_count'] < s['frame_count'] for s in summaries):
        main.append('<p class="ai-setup">An AI can review the drives for you. '
                    '<a href="/account#ai-reviewer">Set up AI review ›</a></p>')
    if not summaries:
        main.append('<div class="card state-box"><h2>No drives to review yet</h2><p>Download the drives from the '
                    'comma (the comma button at the top), then prepare them in <a href="/prepare">Data › Drives</a>'
                    '. They show up here when they are ready.</p></div>')
    # The latest drive first, in every group: by its start (UTC), then by the route's counter; a drive whose start
    # is unknown goes last.
    summaries = sorted(summaries, key=lambda s: ((s.get('drive') or {}).get('start') or '', s['route_id']), reverse=True)
    grouped: dict[str, list[dict]] = {key: [] for key, _ in ROUTE_GROUPS}
    for summary in summaries:
        grouped[route_state(summary['confirmed_count'], summary['frame_count'])].append(summary)
    if summaries:
        main.append(_review_overview(summaries, grouped))
    # Each group folded under its title: the overview comes first, a group opens from its tile or its title (the
    # page script remembers which are open, web/src/pages/site.ts).
    for key, title in ROUTE_GROUPS:
        if not grouped[key]:
            continue
        main.append('<details class="group" id="group-' + key + '" data-group="' + key + '">'
                    '<summary class="group-title"><span>' + title + '</span> <span class="n">' + str(len(grouped[key]))
                    + '</span><span class="group-sub">' + _group_sub(key, grouped[key]) + '</span>'
                    '<span class="group-chev" aria-hidden="true">›</span></summary><ul class="cards">')
        main.extend(_route_row(summary, ai_reviewer=ai_reviewer) for summary in grouped[key])
        main.append('</ul></details>')
    main.append('<details class="foot"><summary>Technical details</summary>'
                '<p>Review database: <code>' + html.escape(str(database)) + '</code></p></details>')
    main.append('</main>')
    return simple_page('Sign review', 'review', nonce, '\n'.join(main))


def render_problem(status: int, message: str, nonce: str = '', *, drive: bool = False, guest: bool = False,
                   uninvited: bool = False) -> str:
    """An error page (404, 403, ...): a title and one sentence in words, no code, and the way back: Review for a
    drive that is not there (drive), else Overview; My reviews for a guest. Someone signed in with an account that has no
    access (uninvited) gets the public page and the sign-out of the proxy, to come back with another account."""
    title = ({404: 'Drive not found' if drive else 'Page not found', 403: 'Access denied'}.get(status)
             or 'Something went wrong')
    main = ['<main class="page"><div class="card state-box center">',
            '<h1>' + html.escape(title) + '</h1>']
    # "Page not found." under the title "Page not found" would only repeat it.
    if message.rstrip('.') != title:
        main.append('<p>' + html.escape(message) + '</p>')
    # Every 403 page (token missing or wrong, host not allowed) would come back here from the list.
    if uninvited:
        main.append('<a class="btn back" href="/">← Smartspeed</a> <a class="btn back" href="' + SIGN_OUT_ADDRESS + '">Sign out</a>')
    if status != 403:
        main.append('<a class="btn back" href="' + GUEST_HOME + '">← My reviews</a>' if guest
                    else '<a class="btn back" href="/routes">← Sign review</a>' if drive
                    else '<a class="btn back" href="/">← Overview</a>')
    main.append('</div></main>')
    return simple_page(title, 'review' if drive else '', nonce, '\n'.join(main), guest=guest)
