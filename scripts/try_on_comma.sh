#!/usr/bin/env bash
# Try a SignNet on the comma now: the button "Try on the comma now" of a SignNet version on the Models tab runs it.
# The SignNet of <run> and the classifier in use are copied straight onto the comma over ssh, into the models folder
# of the build it runs: no publication, no build. They last until the comma's next software update; --restore (the
# link "Back to the build's networks") puts the build's networks back now.
#
# Usage:
#   scripts/try_on_comma.sh --signnet=<run> [--classifier=<id>] [--dry-run]
#   scripts/try_on_comma.sh --restore
#
#   1/5 Check    the comma answers over ssh (root's key, Host comma), is parked (IsOnroad), its build runs the
#                speed-limits networks (runner.py with MANIFEST_FORMAT = 2, warmup.py, vision.py) and prepares them
#                (Show Vision Speed Limit on, or Speed Limit Source Map Data + Vision or Vision Only). Its vision.py is
#                copied here.
#   2/5 Export   the SignNet and the classifier to ONNX on the CPU, as Send to the car does.
#   3/5 Check    them with the comma's code before copying anything: the contracts against its vision.py (SignNet's
#                band 1x3x288x896, grid 8 px, one output of 5 channels, heat first; the classifier's 1x3x160x160 input,
#                its normalisation, its classes as the comma reads them), then its read_signs with both ONNX under
#                onnxruntime on 5 reviewed frames with a confirmed speed sign: the speed read right on 3 at least.
#   4/5 Copy     each file to <name>.tmp, its sha256 checked on the comma, then moved into place, speed_limits.json
#                last (version <day>-try). Recorded in models/on-comma-trial.json of the data root.
#   5/5 Wait     every 15 s, up to 6 minutes, for the comma to compile both (speed_limits_warmup, offroad). If it does
#                not, the build's networks are put back and the job fails with the warmup's lines of the comma's log.
#
#   --dry-run   steps 1 to 3 only: the comma is only read (its state and its vision.py), nothing copied or recorded.
#   --restore   1/2 Check the comma (answers, parked), 2/2 git checkout -- . in its models folder: the build's
#               networks again; models/on-comma-trial.json cleared.
#
# Nothing else on the comma is touched: no reboot, no process restarted.
# Env: NNSLR_DATA_ROOT (default <repo>/data), NNSLR_GPU_VENV (default <repo>/.venv-gpu: torch, onnx, onnxruntime),
#      NNSLR_COMMA_SSH (the ssh host, default comma), NNSLR_COMMA_OPENPILOT (default /data/openpilot),
#      NNSLR_COMMA_PARAMS (default /data/params/d), NNSLR_COMMA_LOG (default /data/log).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT/data}"
VENV="${NNSLR_GPU_VENV:-$ROOT/.venv-gpu}"

die() { echo "error: $*" >&2; exit 1; }

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

signnet=""
classifier=""
restore=0
dry_run=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage ;;
    --signnet=*) signnet="${1#--signnet=}" ;;
    --signnet) [ "$#" -ge 2 ] || die "--signnet needs a run"; signnet="$2"; shift ;;
    --classifier=*) classifier="${1#--classifier=}" ;;
    --classifier) [ "$#" -ge 2 ] || die "--classifier needs an id"; classifier="$2"; shift ;;
    --restore) restore=1 ;;
    --dry-run) dry_run=1 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done
if [ "$restore" -eq 1 ]; then
  [ -z "$signnet$classifier" ] && [ "$dry_run" -eq 0 ] || die "--restore takes no SignNet, classifier or --dry-run"
else
  [ -n "$signnet" ] || die "which SignNet? --signnet=<run> (a run of the Models tab), or --restore"
  [[ "$signnet" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$ ]] || die "invalid run name: $signnet"
  if [ -n "$classifier" ] && [[ ! "$classifier" =~ ^classifier:[A-Za-z0-9][A-Za-z0-9._-]{0,99}/[A-Za-z0-9/._-]{1,200}$ ]]; then
    die "invalid classifier id: $classifier"
  fi
fi
[ -x "$VENV/bin/nnslr" ] || die "GPU environment missing: $VENV/bin/nnslr (helpers/3_train.sh installs it)"
command -v ssh >/dev/null && command -v scp >/dev/null || die "ssh and scp are needed on this server"
export NNSLR_DATA_ROOT="$DATA_ROOT"

cli=(try-on-comma --data-root "$DATA_ROOT")
if [ "$restore" -eq 1 ]; then
  echo "== Back to the build's networks on the comma"
  cli+=(--restore)
else
  echo "== Try on the comma: SignNet $signnet${classifier:+, classifier $classifier}"
  cli+=(--signnet "$signnet")
  [ -z "$classifier" ] || cli+=(--classifier "$classifier")
  [ "$dry_run" -eq 0 ] || cli+=(--dry-run)
fi
# On the CPU: the GPU is left to the trainings.
CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 exec "$VENV/bin/nnslr" "${cli[@]}"
