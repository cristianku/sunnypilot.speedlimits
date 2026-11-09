// Mounting the Vue parts of a page.
import {createApp, type Component} from 'vue';
import AccountMenu from '../components/AccountMenu.vue';
import ConfirmDialog from '../components/ConfirmDialog.vue';
import ToastHost from '../components/ToastHost.vue';
import TopComma from '../components/TopComma.vue';
import TopMachine from '../components/TopMachine.vue';

/** A part of every page in its own element at the end of <body>: the confirm dialog (composables/confirm.ts), the
 * toast (composables/toast.ts). */
function mountHost(id: string, part: Component): void {
  if (document.getElementById(id) !== null) return;
  const host = document.createElement('div');
  host.id = id;
  document.body.append(host);
  createApp(part).mount(host);
}

/**
 * The parts every signed-in page shares: the Account menu in the top bar rendered by pages.py (#account-menu;
 * data-current marks the Account page), the comma (#comma-pill) and the server (#tabs-machine) of the top bar, the
 * confirm dialog and the toast (so the comma and the server can tell how an action went on any page).
 */
export function mountSiteParts(): void {
  mountHost('dialog-host', ConfirmDialog);
  mountHost('toast-host', ToastHost);
  const host = document.getElementById('account-menu');
  if (host !== null) createApp(AccountMenu, {current: host.dataset['current'] === 'true'}).mount(host);
  const machine = document.getElementById('tabs-machine');
  if (machine !== null) createApp(TopMachine, {feed: machine.dataset['feed'] ?? 'system'}).mount(machine);
  const comma = document.getElementById('comma-pill');
  if (comma !== null) createApp(TopComma).mount(comma);
}

/** The page's own app in #app, plus the Account menu and the confirm dialog. */
export function mountPage(root: Component, props: Record<string, unknown> = {}): void {
  mountSiteParts();
  const host = document.getElementById('app');
  if (host === null) throw new Error('Missing element in the page: #app');
  createApp(root, props).mount(host);
}
