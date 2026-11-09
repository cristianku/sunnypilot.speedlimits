// Scrolling, the one thing the components leave to the DOM: the photo back in view on a phone, the editor of the
// chosen box in view on a computer. Called once Vue has drawn the change (nextTick).
import {narrowLayout} from './dom';
import {report} from './store';

// On a phone the panel is below the photo: bring the photo back into view after a frame change. Not on the Signs side of
// Speed review (report.embedded): the page there is the player's, and the picture stays where it is.
export function revealStage(): void {
  if (!narrowLayout() || report.embedded) return;
  const header = document.querySelector('.topbar');
  const stage = document.getElementById('stage');
  if (header === null || stage === null) return;
  const top = getComputedStyle(header).position === 'sticky' ? header.getBoundingClientRect().bottom : 0;
  const rect = stage.getBoundingClientRect();
  if (rect.top < top || rect.top > window.innerHeight * 0.6) {
    window.scrollTo(0, window.scrollY + rect.top - top - 8);
  }
}

// On a computer the panel scrolls so that values, type and road of the chosen box are in sight. Not on the Signs side of
// Speed review: the whole page would scroll there and take the player's picture up with it (Apply previous, a box chosen).
export function revealEditor(): void {
  if (narrowLayout() || report.embedded) return;
  document.getElementById('roads')?.scrollIntoView({block: 'nearest'});
}
