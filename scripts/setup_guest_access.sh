#!/usr/bin/env bash
# Open the portal to the reviewers from outside (Data > Reviewers): write the settings of
# src/nnslr_tools/review_access.py in the file the service reads. Without them nobody comes in from
# outside. Safe to run again: the secret is created once and kept, the other lines are replaced.
#
#   scripts/setup_guest_access.sh --proxy-ips IPS --public-hosts HOSTS --dry-run   change nothing
#   sudo scripts/setup_guest_access.sh --proxy-ips IPS --public-hosts HOSTS        write them
#   sudo scripts/setup_guest_access.sh --new-secret ...                            and a new secret
#   sudo scripts/setup_guest_access.sh --off                                       close it again
#
# IPS: the address of the proxy in front of the site, HOSTS: the public names of the site, both
# separated by commas. The secret is never printed: the proxy's administrator reads it from the file
# (NNSLR_PROXY_SECRET). The service reads the file when it starts, so it needs a restart after this.
# Full procedure: docs/GUIDE.md. NNSLR_REVIEW_ENV_FILE overrides the file (used by the tests).
set -euo pipefail

ENV_FILE="${NNSLR_REVIEW_ENV_FILE:-/etc/speedlimits-web.env}"
PROXY_IPS=""
PUBLIC_HOSTS=""
DRY_RUN=0
NEW_SECRET=0
OFF=0

usage() {
  sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

value_of() {
  sed -n "s/^$1=//p" "$ENV_FILE" | tail -n 1
}

without_settings() {
  grep -v -E '^NNSLR_(PROXY_IPS|PROXY_SECRET|PUBLIC_HOSTS)=' "$ENV_FILE" || true
}

while [ $# -gt 0 ]; do
  case "$1" in
    --proxy-ips) PROXY_IPS="${2:-}"; shift 2 ;;
    --public-hosts) PUBLIC_HOSTS="${2:-}"; shift 2 ;;
    --new-secret) NEW_SECRET=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --off) OFF=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ ! -f "$ENV_FILE" ]; then
  echo "$ENV_FILE is missing: install the service first (scripts/install_review_service.sh)." >&2
  exit 1
fi
if [ "$DRY_RUN" = 0 ] && [ ! -w "$ENV_FILE" ]; then
  echo "$ENV_FILE cannot be written: run this as root." >&2
  exit 1
fi

if [ "$OFF" = 1 ]; then
  if [ "$DRY_RUN" = 1 ]; then
    echo "[dry run] would remove the settings of the reviewers from $ENV_FILE."
    exit 0
  fi
  new_file="$(mktemp "$ENV_FILE.XXXXXX")"
  without_settings > "$new_file"
  chmod 600 "$new_file"
  mv "$new_file" "$ENV_FILE"
  echo "Closed in $ENV_FILE. Restart speedlimits-web for it to count."
  exit 0
fi

if ! [[ "$PROXY_IPS" =~ ^[0-9a-fA-F:.]+(,[0-9a-fA-F:.]+)*$ ]]; then
  echo "--proxy-ips needs the addresses of the proxy, separated by commas." >&2
  exit 2
fi
if ! [[ "$PUBLIC_HOSTS" =~ ^[a-z0-9.-]+(,[a-z0-9.-]+)*$ ]]; then
  echo "--public-hosts needs the public names of the site, separated by commas." >&2
  exit 2
fi

secret="$(value_of NNSLR_PROXY_SECRET)"
kept="kept"
if [ -z "$secret" ] || [ "$NEW_SECRET" = 1 ]; then
  kept="new"
fi
if [ "$DRY_RUN" = 1 ]; then
  echo "[dry run] would write in $ENV_FILE:"
  echo "  NNSLR_PROXY_IPS=$PROXY_IPS"
  echo "  NNSLR_PUBLIC_HOSTS=$PUBLIC_HOSTS"
  echo "  NNSLR_PROXY_SECRET ($kept, not shown)"
  exit 0
fi
if [ "$kept" = "new" ]; then
  secret="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
fi

new_file="$(mktemp "$ENV_FILE.XXXXXX")"
chmod 600 "$new_file"
{
  without_settings
  echo "NNSLR_PROXY_IPS=$PROXY_IPS"
  echo "NNSLR_PROXY_SECRET=$secret"
  echo "NNSLR_PUBLIC_HOSTS=$PUBLIC_HOSTS"
} > "$new_file"
mv "$new_file" "$ENV_FILE"
echo "Written in $ENV_FILE (secret: $kept, not shown). Restart speedlimits-web for it to count."
