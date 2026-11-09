#!/usr/bin/env bash
# Keep data/derived (frames, videos, sign reads: rebuilt by Prepare, about 160 GB) on the TrueNAS
# share instead of the LXC's disk: a systemd bind mount of /data/sunnypilot/SPEEDLIMITS/derived
# on <clone>/data/derived, started after the share's own mount, and a drop-in that makes
# speedlimits-web wait for it. A bind mount and not a symlink: the paths stay under the data root,
# which the site checks (path.resolve() inside root.resolve()). The databases (data/annotations)
# stay on the local disk: SQLite does not belong on NFS, and the LXC's backup on PBS keeps them.
#
#   scripts/install_derived_mount.sh             install / update and mount
#   scripts/install_derived_mount.sh --dry-run   show what would be done, change nothing
#   scripts/install_derived_mount.sh --uninstall unmount and remove (the copy on TrueNAS stays)
#
# The mount point must be an empty folder: move the local copy first (docs/INSTALL.md,
# "data/derived on TrueNAS"). NNSLR_DERIVED_SOURCE overrides the folder on the share.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SOURCE="${NNSLR_DERIVED_SOURCE:-/data/sunnypilot/SPEEDLIMITS/derived}"
TARGET="$REPO/data/derived"
UNIT="$(systemd-escape -p --suffix=mount "$TARGET")"
UNIT_FILE="/etc/systemd/system/$UNIT"
DROPIN="/etc/systemd/system/speedlimits-web.service.d/derived-mount.conf"
DRY=0
UNINSTALL=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --uninstall) UNINSTALL=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else "$@"; fi; }

if [ "$UNINSTALL" = 1 ]; then
  run systemctl disable --now "$UNIT" || true
  run rm -f "$UNIT_FILE" "$DROPIN"
  run systemctl daemon-reload
  echo "removed $UNIT; $TARGET is the local folder again (empty until Prepare fills it)"
  exit 0
fi

# The share itself: <mount of SOURCE> must be up, the folder there.
SHARE="$(findmnt -n -o TARGET -T "$SOURCE" || true)"
if [ ! -d "$SOURCE" ] || [ -z "$SHARE" ] || [ "$SHARE" = / ]; then
  echo "$SOURCE is not on a mounted share" >&2
  exit 1
fi
if ! mountpoint -q "$TARGET" && [ -d "$TARGET" ] && [ -n "$(ls -A "$TARGET")" ]; then
  echo "$TARGET is not empty: copy it to $SOURCE and move it aside first (docs/INSTALL.md)" >&2
  exit 1
fi

unit="[Unit]
Description=data/derived of speedlimits on the TrueNAS share
Documentation=file://$REPO/docs/INSTALL.md
RequiresMountsFor=$SOURCE

[Mount]
What=$SOURCE
Where=$TARGET
Type=none
Options=bind

[Install]
WantedBy=multi-user.target
"
dropin="[Unit]
# data/derived lives on the TrueNAS share (scripts/install_derived_mount.sh).
RequiresMountsFor=$TARGET
"

if [ "$DRY" = 1 ]; then
  printf '%s:\n%s\n%s:\n%s\n' "$UNIT_FILE" "$unit" "$DROPIN" "$dropin"
else
  mkdir -p "$TARGET" "$(dirname "$DROPIN")"
  printf '%s' "$unit" > "$UNIT_FILE"
  printf '%s' "$dropin" > "$DROPIN"
fi
run systemctl daemon-reload
run systemctl enable --now "$UNIT"
[ "$DRY" = 1 ] || findmnt "$TARGET"
