#!/usr/bin/env bash
# Installs the site of the outside reviewers on their own machine, from a checkout of this repository on that machine, as
# root. Needs the copy account and its folder already there (made with the machine). Safe to run again: the secrets are
# created only when the file is missing, and the units are rewritten.
#
#   scripts/install_review_box.sh --proxy-ips <address of the proxy> --public-hosts <the public address>
#       [--user reviewsync] [--box /srv/reviews] [--env-file /etc/speedlimits-reviews.env]
#
# On a first install it writes the env file with a random access token and a random proxy secret, never printed: the
# proxy needs that secret in its own configuration, copied from file to file. The machine needs python3-venv, rsync and
# ffmpeg, and PyPI for the two packages of the pages (the extra review-box).
set -euo pipefail

USER_NAME=reviewsync
BOX=/srv/reviews
ENV_FILE=/etc/speedlimits-reviews.env
PROXY_IPS=""
PUBLIC_HOSTS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --user) USER_NAME="${2:-}"; shift 2 ;;
    --box) BOX="${2:-}"; shift 2 ;;
    --env-file) ENV_FILE="${2:-}"; shift 2 ;;
    --proxy-ips) PROXY_IPS="${2:-}"; shift 2 ;;
    --public-hosts) PUBLIC_HOSTS="${2:-}"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[ "$(id -u)" = 0 ] || { echo "Run it as root." >&2; exit 1; }
id "$USER_NAME" >/dev/null || { echo "The account $USER_NAME is missing." >&2; exit 1; }
[ -d "$BOX" ] || { echo "The folder $BOX is missing." >&2; exit 1; }
if [ ! -f "$ENV_FILE" ] && { [ -z "$PROXY_IPS" ] || [ -z "$PUBLIC_HOSTS" ]; }; then
  echo "A first install needs --proxy-ips and --public-hosts." >&2
  exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install --quiet -e "$REPO[review-box]"

install -d -o "$USER_NAME" -g "$USER_NAME" -m 700 "$BOX/data" "$BOX/videos" "$BOX/inbox" "$BOX/outbox"

if [ ! -f "$ENV_FILE" ]; then
  umask 077
  secret() { "$REPO/.venv/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))'; }
  cat > "$ENV_FILE" <<ENV
NNSLR_DATA_ROOT=$BOX/data
COMMA_VIDEOS_DIR=$BOX/videos
NNSLR_REVIEW_BOX=1
NNSLR_REVIEW_TOKEN=$(secret)
NNSLR_PROXY_IPS=$PROXY_IPS
NNSLR_PROXY_SECRET=$(secret)
NNSLR_PUBLIC_HOSTS=$PUBLIC_HOSTS
ENV
  chown "root:$USER_NAME" "$ENV_FILE"
  chmod 640 "$ENV_FILE"
fi

for unit in speedlimits-reviews.service speedlimits-reviews-tick.service speedlimits-reviews-tick.timer; do
  sed -e "s|__NNSLR_REPO__|$REPO|g" -e "s|__NNSLR_ENV_FILE__|$ENV_FILE|g" -e "s|__NNSLR_USER__|$USER_NAME|g" \
      -e "s|__NNSLR_BOX__|$BOX|g" "$REPO/deploy/$unit" > "/etc/systemd/system/$unit"
done
systemctl daemon-reload
systemctl enable --now speedlimits-reviews-tick.timer
systemctl enable speedlimits-reviews.service
systemctl restart speedlimits-reviews.service
echo "The site of the reviewers is up on port 8765. The proxy secret is NNSLR_PROXY_SECRET in $ENV_FILE."
