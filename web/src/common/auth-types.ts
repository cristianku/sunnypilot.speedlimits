// GET /api/auth/state, and the answers of the sign-in calls.

export interface AuthState {
  configured: boolean;
  signed_in: boolean;
  /** The site on 127.0.0.1 without sign-in. */
  local: boolean;
  user?: string;
  sessions?: number;
  /** A reviewer from outside has no account here: they sign out at this address of the sign-in in front of the site. */
  role?: 'admin' | 'reviewer';
  sign_out?: string;
}

export interface AuthAnswer {
  ok: boolean;
  state: AuthState;
}
