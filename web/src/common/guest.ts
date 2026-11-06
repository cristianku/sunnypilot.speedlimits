// The reviewer from outside (nnslr_tools.review_access): pages.guest_bar marks their top bar. The server refuses all they
// are not given, whatever this says: here it only leaves out of their pages the controls that could not work.

export const isGuest = (): boolean => document.querySelector('.topbar[data-guest]') !== null;
