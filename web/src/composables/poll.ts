// A request every few seconds, one at a time: the next one leaves the interval after the answer. On a busy machine an
// answer can take longer than the interval, and requests sent on a fixed clock would pile up, each slowing the others.
import {onBeforeUnmount, onMounted} from 'vue';

/** Asks while the component is shown; the returned refresh() asks now, or once more after the request on its way. The
 * interval may be a function, read after each answer (a page that needs the answer less often at times). */
export function usePoll(ask: () => Promise<void>, intervalMs: number | (() => number)): () => Promise<void> {
  let timer: number | undefined;
  let pending: Promise<void> | null = null;
  let again = false;
  let stopped = false;
  function refresh(): Promise<void> {
    if (pending !== null) {
      // Asked after something changed: the answer on its way may be older than the change.
      again = true;
      return pending;
    }
    window.clearTimeout(timer);
    pending = ask().finally(() => {
      pending = null;
      if (stopped) return;
      if (again) {
        again = false;
        void refresh();
      } else {
        timer = window.setTimeout(() => void refresh(), typeof intervalMs === 'function' ? intervalMs() : intervalMs);
      }
    });
    return pending;
  }
  onMounted(() => void refresh());
  onBeforeUnmount(() => {
    stopped = true;
    window.clearTimeout(timer);
  });
  return refresh;
}
