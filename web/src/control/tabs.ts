// Every page of the site but the route list and the review pages is one Vue app: the tab shown follows the address
// ("/", "/prepare", "/dataset", "/roads", "/train", "/evaluate", "/models", "/publish"). The pages are
// grouped in four sections (pages.SECTIONS): Overview, Data (Drives, Review, Roads, Training dataset),
// Training, Models (Versions, Results, Publish); the section links of the top bar and the phone's tab bar (rendered by
// pages.py), the pages under a section's title (SectionHead.vue) and the brand switch tab here without a reload; the
// route list (/routes) and the review pages stay normal links. A drive opened on Roads is "/roads?drive=<route>": an
// address of its own, Back returns to the list.
import {nextTick, onBeforeUnmount, onMounted, ref} from 'vue';

export type ControlTab = 'overview' | 'prepare' | 'dataset' | 'roads' | 'train' | 'evaluate' | 'models'
  | 'publish';
export type Section = 'overview' | 'data' | 'train' | 'models';

export const TAB_PATH: Record<ControlTab, string> = {
  overview: '/', prepare: '/prepare', dataset: '/dataset', roads: '/roads', train: '/train',
  evaluate: '/evaluate', models: '/models', publish: '/publish',
};
/** The name of each page, in the window title and under its section's title. */
const TAB_TITLE: Record<ControlTab, string> = {
  overview: 'Overview', prepare: 'Drives', dataset: 'Training dataset', roads: 'Roads',
  train: 'Training', evaluate: 'Results', models: 'Versions', publish: 'Publish',
};
/** A page of a section: its address and name; the route list (/routes) is Data's, outside this app. */
export interface SectionPage {
  key: ControlTab | 'review';
  href: string;
  label: string;
}
/** The sections, as pages.SECTIONS: their title, and their pages under it. */
export const SECTIONS: Record<Section, {title: string; pages: SectionPage[]}> = {
  overview: {title: 'Overview', pages: []},
  data: {title: 'Data', pages: [{key: 'prepare', href: '/prepare', label: 'Drives'}, {key: 'review', href: '/routes', label: 'Sign review'},
    {key: 'roads', href: '/roads', label: 'Roads'},
    {key: 'dataset', href: '/dataset', label: 'Training dataset'}]},
  train: {title: 'Training', pages: []},
  models: {title: 'Models', pages: [{key: 'models', href: '/models', label: 'Versions'}, {key: 'evaluate', href: '/evaluate', label: 'Results'},
    {key: 'publish', href: '/publish', label: 'Publish'}]},
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
const DRIVE_TABS: readonly ControlTab[] = ['roads'];
/** The drive of an address on Roads ("/roads?drive=<route>"), or null (the list, or another tab). */
function driveOf(path: string, search: string): string | null {
  const tab = tabOf(path);
  return tab !== null && DRIVE_TABS.includes(tab) ? new URLSearchParams(search).get('drive') || null : null;
}
/** The drive opened on Roads, null for the list. */
export const tabDrive = ref<string | null>(driveOf(location.pathname, location.search));

/** The address of a tab, with its drive. */
function addressOf(tab: ControlTab, drive: string | null): string {
  const query = new URLSearchParams();
  if (drive !== null) query.set('drive', drive);
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
  document.title = 'Speed limits · ' + TAB_TITLE[tab];
}

/** Shows a tab and puts its address in the history; with a hash ("#dataset"), scrolls to it once the tab is drawn; with
 * a drive, that drive of Roads. */
export function goTo(tab: ControlTab, hash = '', drive: string | null = null): void {
  if (tab === currentTab.value && !hash && drive === tabDrive.value) return;
  // A drive opened from its list: its Back control goes back in the history, to the list as it was.
  const fromList = drive !== null && currentTab.value === tab && tabDrive.value === null;
  currentTab.value = tab;
  tabDrive.value = drive;
  history.pushState({tab, fromList}, '', addressOf(tab, drive) + hash);
  markCurrent(tab);
  if (!hash) {
    window.scrollTo({top: 0});
    return;
  }
  void nextTick(() => document.getElementById(hash.slice(1))?.scrollIntoView({block: 'start'}));
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
    goTo(tab, link.hash, driveOf(link.pathname, link.search));
  }
  function onPop(): void {
    const tab = tabOf(location.pathname);
    if (tab === null) return;
    currentTab.value = tab;
    tabDrive.value = driveOf(location.pathname, location.search);
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
