// Keyboard shortcuts: ← → frame by frame of the video (video.ts), Shift ← → from picture to picture (a second), Shift ↑ ↓ box (the arrows alone too on a box with a number or none selected), the speed typed
// with one key (2 = 20 … 9 = 90, 1 = 100, "1 2" = 120, 0 = not readable; typed whole, "5 0", it works too), O X U the road (ours, another, unclear), S other
// sign for our road, V on a vehicle (not a sign), L variable speed limit (a lit display) or back, G give way, P stop, R roundabout, E something else, M main road, ← → ↑ ↓ the
// direction (left, right, straight, several), Enter confirm, B or N new box, C fit the box picked to its sign on the server
// (every box of the frame when none was picked; Shift+C the same with the AI), A apply previous, I ask the AI, H second reading, D removes every
// box of the frame, Del removes a box the reviewer selected (on the one the page opened on it says to click it
// first), Esc cancel, ? shows or hides the list of the keys. While a menu or panel of the site's top bar is open, the keys are its own.
import {onBeforeUnmount, onMounted} from 'vue';
import {askAi, fitBox, fitLocal} from './ai';
import {applyPrevious} from './carry';
import {confirmFrame} from './confirm';
import {cancelAll, removeAllBoxes, removeSelected, selectBox, setField, setKind, setOnVehicle, setOtherOurs, setValue, toggleDrawing,
  toggleHints, toggleVariable} from './edits';
import {currentRow, workOf} from './frames';
import {step} from './navigation';
import {stepFrame} from './video';
import {report, state, toggleKeys} from './store';

// A speed is one key (2 = 20 … 9 = 90); after a 1 the page waits this long for a second digit ("1 2" = 120, 1 alone
// = 100), and a 0 that comes this soon after a speed set at once only ends it typed whole ("5 0", "1 2 0").
const TYPING_MS = 900;

// G P R E M: the kind of an other sign.
const KIND_KEYS: Record<string, string> = {g: 'give_way', p: 'stop', r: 'roundabout', e: 'other', m: 'main_road'};

// The arrows with a box selected: its direction (↓ several directions), on an "Other sign" or any box without a number
// (one just drawn, too). On a box with a number or "Not a sign" they keep their moves: a speed limit never becomes a
// direction sign, its number lost, by an arrow pressed to change box.
const DIRECTION_KEYS: Record<string, string> = {ArrowLeft: 'direction_left', ArrowRight: 'direction_right',
  ArrowUp: 'direction_straight', ArrowDown: 'direction_several'};

// O X U: the road of the selected box (A is Apply previous).
const ROAD_KEYS: Record<string, string> = {o: 'own_road', x: 'other_road', u: 'unresolved'};

/** The selected box, if any. */
function selectedBox() {
  const row = currentRow();
  return row ? workOf(row).detections[state.selected] : undefined;
}

/** Whether the arrows set the direction of the selected box (DIRECTION_KEYS). */
function takesDirection(): boolean {
  const d = selectedBox();
  return d !== undefined && (d.family === 'other_sign' || (d.value_kph == null && d.family !== 'not_a_sign'));
}

/** The kind of the selected box when it is an other sign, else null. */
function selectedKind(): string | null {
  const d = selectedBox();
  return d && d.family === 'other_sign' ? d.sign_type ?? null : null;
}

/** ↑ ↓: the box before or after the selected one (from none: the first or the last). */
function stepBox(delta: number): void {
  const row = currentRow();
  if (!row) return;
  const count = workOf(row).detections.length;
  if (count === 0) return;
  const next = state.selected < 0 ? (delta > 0 ? 0 : count - 1) : (state.selected + delta + count) % count;
  selectBox(next);
}

// Enter on these buttons just pressed with the mouse still confirms, as the shortcut list says.
const CONFIRM_ON_ENTER = new Set(['prev', 'next', 'mPrev', 'mNext', 'draw']);

// The menus and panels of the site's top bar: the Account menu (AccountMenu.vue), the server and the comma
// (TopMachine.vue, TopComma.vue), and any dialog.
const POPOVERS = '#account-pop, .tm-pop';
const INSIDE_POPOVER = POPOVERS + ', [role="dialog"], [role="alertdialog"]';

/** A key pressed inside one of them, or while one is open: it is not for the frame behind. */
export function popoverHasKeys(target: Element | null): boolean {
  if (target?.closest(INSIDE_POPOVER)) return true;
  // Closed, the Account menu is display:none and the two panels are not in the page.
  return [...document.querySelectorAll(POPOVERS)].some((element) => element.getClientRects().length > 0);
}

export function useKeyboard(): void {
  // The button last pressed with the mouse or a finger; Tab moves the focus elsewhere and forgets it.
  let pointerButton: Element | null = null;
  const onPointerDown = (event: PointerEvent): void => {
    pointerButton = event.target instanceof Element ? event.target.closest('button') : null;
  };
  const onFocusIn = (event: FocusEvent): void => {
    if (event.target !== pointerButton) pointerButton = null;
  };
  let typingTimer: number | undefined;
  /** A speed was just set at once: a 0 now is the end of it typed whole ("5 0", "1 2 0"), not "not readable". */
  let justSet = false;
  let justSetTimer: number | undefined;
  /** The digits waiting (a 1, or 1 0) become the value of the selected box: 100. */
  const flushTyping = (): void => {
    window.clearTimeout(typingTimer);
    const typed = state.typing;
    state.typing = '';
    if (!typed || state.selected < 0) return;
    if (typed === '1' || typed === '10') {
      setValue(100);
      return;
    }
    if (typed === '0') {
      setValue(null);
      return;
    }
    const value = Number(typed);
    if (value >= 10 && value <= 250) {
      setValue(value);
      return;
    }
    const key = state.currentKey ?? '';
    state.notice = {key, text: 'One key per speed: 2 for 20 … 9 for 90, 1 for 100, 1 2 for 120 (0: not readable).', error: true};
  };
  /** A digit typed on the selected box: 2-9 set 20-90 at once, 0 sets "not readable", 1 waits for a second digit
   * (1 1 = 110, 1 2 = 120; 1 alone or 1 0 0 = 100). */
  const typeDigit = (digit: string): void => {
    window.clearTimeout(typingTimer);
    window.clearTimeout(justSetTimer);
    if (justSet && digit === '0') {
      justSet = false;
      return;
    }
    justSet = false;
    const typed = state.typing + digit;
    if (typed === '1' || typed === '10') {
      state.typing = typed;
      typingTimer = window.setTimeout(flushTyping, TYPING_MS);
      return;
    }
    state.typing = '';
    if (typed === '0') {
      setValue(null);
      return;
    }
    if (typed === '100') {
      setValue(100);
      return;
    }
    // "5" = 50, "1 2" = 120: a 0 right after is the end of the same number typed whole.
    setValue(Number(typed) * 10);
    justSet = true;
    justSetTimer = window.setTimeout(() => { justSet = false; }, TYPING_MS);
  };
  const onKeyDown = (event: KeyboardEvent): void => {
    const target = event.target instanceof HTMLElement ? event.target : null;
    // Never a frame step, a value or a confirmation from a key meant for an open menu or panel.
    if (popoverHasKeys(event.target instanceof Element ? event.target : null)) return;
    const tag = target ? target.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'select' || tag === 'textarea') {
      if (event.key === 'Escape') target?.blur();
      return;
    }
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    // Any key but a digit ends the number being typed first: Enter then confirms the value just typed.
    if (!/^[0-9]$/.test(event.key)) {
      justSet = false;
      if (state.typing) flushTyping();
    }
    // The same buttons reached with Tab keep their own Enter.
    const byPointer = target !== null && target === pointerButton;
    if (event.key === 'Enter' && byPointer && CONFIRM_ON_ENTER.has(target.id)) {
      event.preventDefault();
      void confirmFrame();
      return;
    }
    // Enter on any other focused button or summary belongs to that control.
    if (event.key === 'Enter' && (tag === 'button' || tag === 'summary' || tag === 'a')) return;
    // With a box selected: its speed (one key), and its road (O X U).
    if (state.selected >= 0 && !event.shiftKey && /^[0-9]$/.test(event.key)) {
      event.preventDefault();
      typeDigit(event.key);
      return;
    }
    if (state.selected >= 0 && !event.shiftKey) {
      const road = ROAD_KEYS[event.key.toLowerCase()];
      if (road) {
        event.preventDefault();
        setField('applicability', road);
        return;
      }
      if (event.key === 's' || event.key === 'S') {
        event.preventDefault();
        setOtherOurs();
        return;
      }
      if (event.key === 'v' || event.key === 'V') {
        event.preventDefault();
        setOnVehicle();
        return;
      }
      if (event.key === 'l' || event.key === 'L') {
        event.preventDefault();
        toggleVariable();
        return;
      }
      // G give way, P stop, R roundabout, E something else, M main road, an arrow a direction: pressed again the kind
      // stays, as with its button.
      const kind = KIND_KEYS[event.key.toLowerCase()]
        ?? (takesDirection() ? DIRECTION_KEYS[event.key] : undefined);
      if (kind) {
        event.preventDefault();
        if (selectedKind() !== kind) setKind(kind);
        return;
      }
    }
    switch (event.key) {
      case 'ArrowUp':
        event.preventDefault();
        stepBox(-1);
        break;
      case 'ArrowDown':
        event.preventDefault();
        stepBox(1);
        break;
      case 'ArrowLeft':
        event.preventDefault();
        if (event.shiftKey) step(-1);
        else stepFrame(-1);
        break;
      case 'ArrowRight':
        event.preventDefault();
        if (event.shiftKey) step(1);
        else stepFrame(1);
        break;
      case 'Enter':
        event.preventDefault();
        void confirmFrame();
        break;
      case 'b':
      case 'B':
      case 'n':
      case 'N':
        event.preventDefault();
        toggleDrawing();
        break;
      case 'c':
      case 'C':
        // The box picked (a click, ↑ ↓), else every box of the frame: C on this server, Shift+C with the AI.
        if (report.apiBase) {
          event.preventDefault();
          void (event.shiftKey ? fitBox() : fitLocal());
        }
        break;
      case 'a':
      case 'A':
        if (report.apiBase) {
          event.preventDefault();
          void applyPrevious();
        }
        break;
      case 'i':
      case 'I':
        if (report.apiBase) {
          event.preventDefault();
          void askAi();
        }
        break;
      case 'h':
      case 'H':
        if (report.routeHasHints) {
          event.preventDefault();
          toggleHints();
        }
        break;
      case 'd':
      case 'D':
        event.preventDefault();
        removeAllBoxes();
        break;
      case 'Delete':
        // Only a box the reviewer selected: never the one the page opened on, never with Backspace.
        if (state.selected >= 0 && state.selectedByUser) {
          event.preventDefault();
          removeSelected();
        } else if (state.selected >= 0 && state.currentKey !== null) {
          event.preventDefault();
          state.notice = {key: state.currentKey, error: false, warn: true,
            text: 'Click box ' + (state.selected + 1) + ' first (or pick it with ↑ ↓), then press Del to delete it.'};
        }
        break;
      case 'Escape':
        cancelAll();
        break;
      case '?':
        event.preventDefault();
        toggleKeys();
        break;
    }
  };
  onMounted(() => {
    document.addEventListener('pointerdown', onPointerDown, true);
    document.addEventListener('focusin', onFocusIn);
    document.addEventListener('keydown', onKeyDown);
  });
  onBeforeUnmount(() => {
    document.removeEventListener('pointerdown', onPointerDown, true);
    document.removeEventListener('focusin', onFocusIn);
    document.removeEventListener('keydown', onKeyDown);
  });
}
