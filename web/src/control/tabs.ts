// Every page of the site but the route list and the review pages is one Vue app: the tab shown follows the address
// ("/", "/prepare", "/dataset", "/speed-review", "/sign-distance", "/train", "/evaluate",
// "/models", "/drive", "/publish", "/reviewers"). The pages are grouped in four sections (pages.SECTIONS): Overview, Data (Drives,
// Speed review, Sign distance, Training dataset, Reviewers), Training, Models (Versions, Drive, Results, Publish); the
// section links of the top bar and the phone's tab bar (rendered by pages.py), the pages under a section's title
// (SectionHead.vue) and the brand switch tab here without a reload. Signs are reviewed on the Signs side of Speed review
// (the route list /routes and the page of Sign review lead there since 9 Nov 2026). A reviewer from outside has their list at "/reviews": "/" of the public names is the page of the project. A drive opened on Speed review, Sign distance or Models › Drive is "/speed-review?drive=<route>"
// ("/sign-distance?drive=<route>&sign=<key>"):
// an address of its own, Back returns to the list.
import {computed, nextTick, onBeforeUnmount, onMounted, ref} from 'vue';
import {isGuest} from '../common/guest';

export type ControlTab = 'overview' | 'prepare' | 'dataset' | 'speedreview' | 'signdistance'
  | 'reviewers' | 'train' | 'evaluate' | 'models' | 'drive' | 'publish';
export type Section = 'overview' | 'data' | 'train' | 'models';

export const TAB_PATH: Record<ControlTab, string> = {
  overview: isGuest() ? '/reviews' : '/', prepare: '/prepare', dataset: '/dataset',
  speedreview: '/speed-review', signdistance: '/sign-distance', reviewers: '/reviewers', train: '/train', evaluate: '/evaluate', models: '/models',
  drive: '/drive', publish: '/publish',
};
/** The name of each page, in the window title and under its section's title. */
const TAB_TITLE: Record<ControlTab, string> = {
  overview: 'Overview', prepare: 'Drives', dataset: 'Training dataset',
  speedreview: 'Speed review', signdistance: 'Sign distance', reviewers: 'Reviewers', train: 'Training', evaluate: 'Results', models: 'Versions',
  drive: 'Drive', publish: 'Publish',
};
/** A page of a section: its address and name. */
export interface SectionPage {
  key: ControlTab;
  href: string;
  label: string;
}
/** The sections, as pages.SECTIONS: their title, and their pages under it. */
export const SECTIONS: Record<Section, {title: string; pages: SectionPage[]}> = {
  overview: {title: 'Overview', pages: []},
  data: {title: 'Data', pages: [{key: 'prepare', href: '/prepare', label: 'Drives'},
    {key: 'speedreview', href: '/speed-review', label: 'Speed review'},
    {key: 'signdistance', href: '/sign-distance', label: 'Sign distance'},
    {key: 'dataset', href: '/dataset', label: 'Training dataset'}, {key: 'reviewers', href: '/reviewers', label: 'Reviewers'}]},
  train: {title: 'Training', pages: []},
  models: {title: 'Models', pages: [{key: 'models', href: '/models', label: 'Versions'}, {key: 'drive', href: '/drive', label: 'Drive'},
    {key: 'evaluate', href: '/evaluate', label: 'Results'}, {key: 'publish', href: '/publish', label: 'Publish'}]},
};
export function sectionOf(tab: ControlTab): Section {
  return (Object.keys(SECTIONS) as Section[]).find((key) => SECTIONS[key].pages.some((page) => page.key === tab))
    ?? (tab === 'train' ? 'train' : 'overview');
}

export function tabOf(path: string): ControlTab | null {
  const clean = path.replace(/\/+$/, '') || '/';
  return (Object.keys(TAB_PATH) as ControlTab[]).find((tab) => TAB_PATH[tab] === clean) ?? null;
}

/** The tab shown. */
export const currentTab = ref<ControlTab>(tabOf(location.pathname) ?? 'overview');

/** The tabs with a drive of their own at "?drive=<route>". */
const DRIVE_TABS: readonly ControlTab[] = ['speedreview', 'signdistance', 'drive'];
/** The tabs whose drive has one thing of its own opened in the address: a sign on Sign distance ("&sign=<key>"). */
const ITEM_KEY: Partial<Record<ControlTab, string>> = {signdistance: 'sign'};
/** The drive of an address on Speed review, Sign distance or Models › Drive ("/speed-review?drive=<route>"), or null (the
 * list, or another tab). */
function driveOf(path: string, search: string): string | null {
  const tab = tabOf(path);
  return tab !== null && DRIVE_TABS.includes(tab) ? new URLSearchParams(search).get('drive') || null : null;
}
/** The sign (Sign distance) of a drive in an address, or null. */
function itemOf(path: string, search: string): string | null {
  const tab = tabOf(path);
  const key = tab === null ? undefined : ITEM_KEY[tab];
  return key === undefined ? null : new URLSearchParams(search).get(key) || null;
}
/** The drive opened on Speed review, Sign distance or Models › Drive, null for the list. */
export const tabDrive = ref<string | null>(driveOf(location.pathname, location.search));
/** The sign opened in the drive shown, null for none. */
const tabItem = ref<string | null>(itemOf(location.pathname, location.search));
/** The sign opened on Sign distance, null for none. */
export const tabSign = computed(() => currentTab.value === 'signdistance' ? tabItem.value : null);

/** The address of a tab, with its drive and the sign of that drive. */
function addressOf(tab: ControlTab, drive: string | null, item: string | null): string {
  const query = new URLSearchParams();
  if (drive !== null) query.set('drive', drive);
  const key = ITEM_KEY[tab];
  if (drive !== null && item !== null && key !== undefined) query.set(key, item);
  const search = query.toString();
  return TAB_PATH[tab] + (search ? '?' + search : '');
}

/** aria-current on the server's section links (top bar, tab bar) of the tab shown, and the page title. */
function markCurrent(tab: ControlTab): void {
  const section = sectionOf(tab);
  for (const link of document.querySelectorAll<HTMLAnchorElement>('nav.tabs a, nav.tabbar a')) {
    if (link.dataset['section'] === section) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  document.title = 'Speed limits · ' + (isGuest() && tab === 'overview' ? 'My reviews' : TAB_TITLE[tab]);
}

/** Shows a tab and puts its address in the history; with a hash ("#dataset"), scrolls to it once the tab is drawn; with
 * a drive, that drive of Speed review, Sign distance or Models › Drive (and a sign of it on Sign distance). */
export function goTo(wanted: ControlTab, hash = '', drive: string | null = null, item: string | null = null): void {
  // A reviewer from outside has one list, their own.
  const tab = isGuest() && drive === null ? 'overview' : wanted;
  const nextItem = drive === null || ITEM_KEY[tab] === undefined ? null : item;
  if (tab === currentTab.value && !hash && drive === tabDrive.value && nextItem === tabItem.value) return;
  // A drive opened from its list: its Back control goes back in the history, to the list as it was.
  const fromList = drive !== null && currentTab.value === tab && tabDrive.value === null;
  currentTab.value = tab;
  tabDrive.value = drive;
  tabItem.value = nextItem;
  history.pushState({tab, fromList}, '', addressOf(tab, drive, nextItem) + hash);
  markCurrent(tab);
  if (!hash) {
    window.scrollTo({top: 0});
    return;
  }
  void nextTick(() => document.getElementById(hash.slice(1))?.scrollIntoView({block: 'start'}));
}

/** Another sign of the drive shown, in place of the one in the address (no new history entry: Back still returns to
 * the list). */
function showItem(tab: ControlTab, item: string | null): void {
  if (currentTab.value !== tab || tabDrive.value === null || item === tabItem.value) return;
  tabItem.value = item;
  history.replaceState(history.state, '', addressOf(tab, tabDrive.value, item));
}
/** Another sign of the drive shown on Sign distance. */
export function showSign(sign: string | null): void {
  showItem('signdistance', sign);
}

/** Clicks on the tab links and the back button, for the life of the app. */
export function useTabRouting(): void {
  function onClick(event: MouseEvent): void {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = (event.target as Element | null)?.closest<HTMLAnchorElement>('a[href]');
    if (!link || link.origin !== location.origin || link.target) return;
    // An anchor on this tab ("#dataset") scrolls as usual.
    if (link.hash && link.pathname === location.pathname) return;
    const tab = tabOf(link.pathname);
    if (tab === null) return;
    event.preventDefault();
    goTo(tab, link.hash, driveOf(link.pathname, link.search), itemOf(link.pathname, link.search));
  }
  function onPop(): void {
    const tab = tabOf(location.pathname);
    if (tab === null) return;
    currentTab.value = tab;
    tabDrive.value = driveOf(location.pathname, location.search);
    tabItem.value = itemOf(location.pathname, location.search);
    markCurrent(tab);
  }
  onMounted(() => {
    markCurrent(currentTab.value);
    document.addEventListener('click', onClick);
    window.addEventListener('popstate', onPop);
  });
  onBeforeUnmount(() => {
    document.removeEventListener('click', onClick);
    window.removeEventListener('popstate', onPop);
  });
}
