#!/usr/bin/env bash
# The timer of this server that copies with the machine of the outside reviewers every 5 minutes (helpers/review_box.sh),
# after nnslr review-box link. As root, on the server of the site.
#
#   scripts/install_review_box_sync.sh [--uninstall]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ENV_FILE=/etc/speedlimits-web.env
TIMER=speedlimits-review-box-sync.timer
SERVICE=speedlimits-review-box-sync.service
[ "$(id -u)" = 0 ] || { echo "Run it as root." >&2; exit 1; }
if [ "${1:-}" = --uninstall ]; then
  systemctl disable --now "$TIMER" || true
  rm -f "/etc/systemd/system/$TIMER" "/etc/systemd/system/$SERVICE"
  systemctl daemon-reload
  exit 0
fi
sed -e "s|__NNSLR_REPO__|$REPO|g" -e "s|__NNSLR_ENV_FILE__|$ENV_FILE|g" "$REPO/deploy/$SERVICE" > "/etc/systemd/system/$SERVICE"
cp "$REPO/deploy/$TIMER" "/etc/systemd/system/$TIMER"
systemctl daemon-reload
systemctl enable --now "$TIMER"
echo "The copy with the review machine runs every 5 minutes."
