// What the page tells the reviewer, as data for ConfirmArea.vue and MobileBar.vue: warnings above the confirm
// button (server, browser storage, frames not saved), the confirm buttons with what they save, the save line
// of the frame, the message of the phone bar.
import {shortName} from './catalog';
import {aiConfirmed, frameStatus, sameAsServer, unsavedKeys} from './frames';
import {report, state} from './store';
import type {FrameEdit, FrameStatus} from './types';

export interface PageNotice {
  text: string;
  error: boolean;
  /** With "Retry saving". */
  retry: boolean;
}

export function pageNotices(): PageNotice[] {
  const out: PageNotice[] = [];
  if (state.serverWarning) out.push({text: state.serverWarning, error: false, retry: false});
  if (state.storageWarning) out.push({text: state.storageWarning, error: false, retry: false});
  // The first send of a fresh confirmation is not an error: it becomes one only if it fails.
  const unsaved = report.apiBase
    ? unsavedKeys().filter((key) => !(state.inFlight.has(key) && !state.edits[key]?.error)) : [];
  if (unsaved.length === 0) return out;
  let list = unsaved.slice(0, 8).join(', ');
  if (unsaved.length > 8) list += ', …';
  const text = unsaved.length === 1
    ? '1 confirmed frame NOT saved on the server (' + list + ').'
    : unsaved.length + ' confirmed frames NOT saved on the server (' + list + ').';
  out.push({text, error: true, retry: true});
  return out;
}

export type LineClass = '' | 'ok' | 'warn' | 'error';

// The save line next to the confirm button: what the server (or this browser) holds for the frame.
function saveText(key: string, s: FrameEdit | undefined): [string, LineClass] {
  const status = frameStatus(key);
  if (!report.apiBase) {
    if (status === 'done') return ['Confirmed in this browser: export the JSONL to keep it.', 'ok'];
    if (status === 'dirty') return ['Changes not confirmed (only in this browser).', 'warn'];
    return ['', ''];
  }
  if (status === 'saving') return ['Saving…', ''];
  if (!s) return ['', ''];
  if (status === 'done' && aiConfirmed(key)) {
    const via = s.channel === 'mcp' ? 'MCP' : s.channel === 'rest' ? 'REST' : '';
    const who = s.reviewer || 'an AI';
    return ['Saved ✓ · revision ' + s.revision + ' · ' + who + ' (AI' + (via ? ' · ' + via : '') + ')', 'ok'];
  }
  if (status === 'done') {
    return ['Saved ✓ · revision ' + s.revision + (s.reviewer ? ' · ' + s.reviewer : ''), 'ok'];
  }
  if (status === 'error' && s.error) return ['NOT saved: ' + s.error, 'error'];
  if (status === 'error') return ['Confirmed but not saved yet: it is tried again by itself.', 'error'];
  if (status === 'dirty' && s.server_confirmed) {
    return ['Changed: press Confirm to save it. The server keeps the earlier version.', 'warn'];
  }
  if (status === 'dirty') return ['Changes to confirm.', 'warn'];
  if (s.base && s.base.confirmed === false) {
    return ['Confirmation withdrawn: it will not be imported until you confirm it again.', ''];
  }
  return ['', ''];
}

export interface ConfirmLabels {
  label: string;
  mobileLabel: string;
  /** Already on the server and unchanged: Confirm only moves on. */
  already: boolean;
}

/** Where Confirm goes on a frame already confirmed: the next frame of the filter in use. */
function nextInFilter(): string {
  switch (state.filter) {
    case 'todo': return 'Next to review';
    case 'suspicious': return 'Next suspicious';
    case 'proposals': return 'Next with proposals';
    case 'all': return 'Next frame';
    default: return 'Next in the filter';
  }
}

/** Confirm on the panel and on the phone bar: they say what they save ("Confirm · 60 · 40"). */
export function confirmLabels(key: string, work: FrameEdit, status: FrameStatus): ConfirmLabels {
  const names = work.detections.map(shortName);
  // On a phone too: "Confirm · 60", so the list under the photo need not be checked to know what is saved.
  let label = 'Confirm · ' + names.join(' · ');
  let mobileLabel = label;
  if (work.detections.length === 0) {
    label = 'Confirm · no sign';
    mobileLabel = 'Confirm no sign';
  }
  let already = false;
  if (status === 'done' && report.apiBase && sameAsServer(state.edits[key])) {
    label = 'Already confirmed · next →';
    // Between ‹ and › (the frames beside this one) the phone's button says where it goes: on in the filter.
    mobileLabel = nextInFilter() + ' →';
    already = true;
  } else if (status === 'done') {
    label = 'Confirmed ✓ · confirm again';
    mobileLabel = 'Confirmed ✓';
  }
  if (status === 'saving') {
    label = 'Saving…';
    mobileLabel = 'Saving…';
  }
  return {label, mobileLabel, already};
}

export interface StatusLine {
  text: string;
  cls: LineClass;
  /** Discard changes, Withdraw confirmation (armed: "Yes, withdraw"). */
  discard: boolean;
  withdraw: boolean;
  armed: boolean;
  /** The message of the phone bar, empty for none. */
  mobile: string;
}

/** The line under the inspector: Discard changes / Withdraw confirmation, the save line, the phone bar message. */
export function statusLine(key: string, status: FrameStatus): StatusLine {
  const s = state.edits[key];
  // One status line of fixed height: the latest message for this frame, else what the server holds.
  const [savedText, savedClass] = saveText(key, s);
  const notice = state.notice && state.notice.key === key ? state.notice : null;
  let text = savedText;
  let cls: LineClass = savedClass;
  if (notice !== null) {
    if (notice.error) {
      text = notice.text;
      cls = 'error';
    } else if (savedClass !== 'error') {
      // A failed save of this frame stays on the line: a note about the frame saved before must not hide it.
      text = notice.text;
      cls = notice.warn ? 'warn' : 'ok';
    }
  }
  let mobile = '';
  const unsaved = report.apiBase ? unsavedKeys().length : 0;
  if (notice && notice.error) mobile = notice.text;
  else if (savedClass === 'error') mobile = savedText;
  else if (state.serverWarning) mobile = state.serverWarning;
  else if (state.storageWarning) mobile = state.storageWarning;
  else if (unsaved > 0) mobile = unsaved + (unsaved === 1 ? ' confirmed frame' : ' confirmed frames') + ' NOT saved: retry from the panel.';
  return {
    text,
    cls,
    discard: status === 'dirty' || status === 'error',
    withdraw: Boolean(report.apiBase && s && s.saved === true && s.server_confirmed === true && status !== 'saving'),
    armed: state.withdrawArmed === key,
    mobile,
  };
}
