#!/usr/bin/env bash
# Publish to the car: the button Publish of the Publish tab runs it, for the SignNet in use. The SignNet of <run>, the
# classifier in use and the road type in use replace the networks the comma downloads, and the testing branch is
# built again.
#
# Usage:
#   scripts/send_to_car.sh --signnet=<run> [--classifier=<id>] [--rebuild] [--dry-run]
#
#   1/5 Export   the SignNet of <run> and the classifier in use (models/active.json; --classifier: another id of the
#                Models tab) to ONNX on the CPU with the GPU venv, each checked against PyTorch with onnxruntime;
#                the road type in use (with memory), when one is chosen, is the ONNX step its training exported.
#   2/5 Publish  them in the clone of cristianku/speed-limits-models: the ONNX, their contracts and speed_limits.json
#                (format 2, version = the day, .2 .3 ... for another one the same day), one commit pushed to main.
#                Nothing when every ONNX is the published one already (the export is the same byte for byte).
#                Without a road type in use the published one is removed (the comma reads the signs alone).
#                The brain in use, when one is chosen, is exported here too, and with it no road type goes.
#   3/5 Record   models/on-car.json of the data root (what `nnslr on-car` writes): Models, "On the car".
#   4/5 Move     the submodule openpilot/sunnypilot/speed_limits/models of the sunnypilot clone (branch speed-limits)
#                to that commit, one commit of that path alone pushed to speed-limits; nothing when it is there.
#                Refuses a clone on another branch, a submodule with changes of its own, a branch with commits not
#                pushed or diverged (fast-forwarded first when it is only behind): checked before step 1, so a
#                refusal publishes nothing.
#   5/5 Build    cristianku/openpilot psa-torque-sunny-testing again (openpilot_scripts' setup_psa_torque.sh sunny,
#                pushed) when step 2 or 4 changed something or a build is still owed (models/send-to-car.pending),
#                or with --rebuild. Then, on the comma: Settings, Software, Check for updates, then install.
#                A trial of "Try on the comma now" (models/on-comma-trial.json) is cleared once the build is pushed.
#
#   --dry-run   export and compare only: nothing copied, committed, pushed, recorded or built.
#   --rebuild   step 5 also when nothing changed.
#
# Env: NNSLR_DATA_ROOT (default <repo>/data), NNSLR_GPU_VENV (default <repo>/.venv-gpu: torch, onnx, onnxruntime),
#      NNSLR_MODELS_CLONE (default /root/speed-limits-models), NNSLR_SUNNYPILOT_CLONE (default /root/sunnypilot),
#      NNSLR_OPENPILOT_SCRIPTS (default /root/openpilot_scripts), NNSLR_CAR_WORKSPACE (WORKSPACE_ROOT of the build,
#      default /root/work/op-setup-ws), NNSLR_TESTING_REPO (where the last line reads psa-torque-sunny-testing, default
#      https://github.com/cristianku/openpilot.git), NO_PUSH=true (the build commits psa-torque-sunny-testing
#      without pushing it).
# The pushes use this server's ssh key for github.com: no token is passed, printed or written.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT/data}"
VENV="${NNSLR_GPU_VENV:-$ROOT/.venv-gpu}"
MODELS="${NNSLR_MODELS_CLONE:-/root/speed-limits-models}"
SUNNYPILOT="${NNSLR_SUNNYPILOT_CLONE:-/root/sunnypilot}"
OP_SCRIPTS="${NNSLR_OPENPILOT_SCRIPTS:-/root/openpilot_scripts}"
WORKSPACE="${NNSLR_CAR_WORKSPACE:-/root/work/op-setup-ws}"
BUILD="$OP_SCRIPTS/OP-setup-psa-torque-testing/scripts/setup_psa_torque.sh"
BRANCH="speed-limits"
SUBMODULE="openpilot/sunnypilot/speed_limits/models"
TESTING_REPO="${NNSLR_TESTING_REPO:-https://github.com/cristianku/openpilot.git}"
TESTING_BRANCH="psa-torque-sunny-testing"
PENDING="$DATA_ROOT/models/send-to-car.pending"
TRIAL="$DATA_ROOT/models/on-comma-trial.json"

die() { echo "error: $*" >&2; exit 1; }

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

signnet=""
classifier=""
rebuild=0
dry_run=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage ;;
    --signnet=*) signnet="${1#--signnet=}" ;;
    --signnet) [ "$#" -ge 2 ] || die "--signnet needs a run"; signnet="$2"; shift ;;
    --classifier=*) classifier="${1#--classifier=}" ;;
    --classifier) [ "$#" -ge 2 ] || die "--classifier needs an id"; classifier="$2"; shift ;;
    --rebuild) rebuild=1 ;;
    --dry-run) dry_run=1 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done
[ -n "$signnet" ] || die "which SignNet? --signnet=<run> (a run of the Models tab)"
[[ "$signnet" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$ ]] || die "invalid run name: $signnet"
if [ -n "$classifier" ] && [[ ! "$classifier" =~ ^classifier:[A-Za-z0-9][A-Za-z0-9._-]{0,99}/[A-Za-z0-9/._-]{1,200}$ ]]; then
  die "invalid classifier id: $classifier"
fi
[ -x "$VENV/bin/nnslr" ] || die "GPU environment missing: $VENV/bin/nnslr (helpers/3_train.sh installs it)"
[ -e "$MODELS/.git" ] || die "no clone of the published models in $MODELS (NNSLR_MODELS_CLONE)"
[ -e "$SUNNYPILOT/.git" ] || die "no sunnypilot clone in $SUNNYPILOT (NNSLR_SUNNYPILOT_CLONE)"
[ "$dry_run" -eq 1 ] || [ -f "$BUILD" ] || die "build script missing: $BUILD (NNSLR_OPENPILOT_SCRIPTS)"
export NNSLR_DATA_ROOT="$DATA_ROOT"

cli=(send-to-car --signnet "$signnet" --models "$MODELS" --data-root "$DATA_ROOT" --steps-of 5)
[ -z "$classifier" ] || cli+=(--classifier "$classifier")
mode=""
if [ "$dry_run" -eq 1 ]; then
  cli+=(--dry-run)
  mode=" (dry run)"
fi
echo "== Send to the car: SignNet $signnet${classifier:+, classifier $classifier}$mode"
# The sunnypilot clone is checked before anything is published: models pushed with a clone step 4 then refuses would
# leave the comma's download ahead of the branch that names it.
head="$(git -C "$SUNNYPILOT" symbolic-ref --quiet --short HEAD || true)"
[ "$head" = "$BRANCH" ] || die "$SUNNYPILOT is on ${head:-a detached HEAD}, not $BRANCH: switch it back first"
[ -z "$(git -C "$SUNNYPILOT" status --porcelain -- "$SUBMODULE")" ] \
  || die "$SUBMODULE in $SUNNYPILOT has changes of its own: commit or drop them first"
git -C "$SUNNYPILOT" fetch --quiet origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
counts="$(git -C "$SUNNYPILOT" rev-list --left-right --count "$BRANCH...refs/remotes/origin/$BRANCH")"
read -r ahead behind <<< "$counts"
[ "$ahead" -eq 0 ] || [ "$behind" -eq 0 ] || die "$BRANCH of $SUNNYPILOT and origin's have diverged: reconcile them first"
[ "$ahead" -eq 0 ] || die "$BRANCH of $SUNNYPILOT has $ahead commit(s) not pushed: push or drop them first"
models_before="$(git -C "$MODELS" rev-parse HEAD)"
output="$(mktemp)"
trap 'rm -f "$output"' EXIT
# Steps 1 to 3 (nnslr_tools.car_release). On the CPU: the GPU is left to the trainings.
CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 "$VENV/bin/nnslr" "${cli[@]}" 2>&1 | tee "$output"
models_commit="$(git -C "$MODELS" rev-parse HEAD)"
published=0
[ "$models_commit" = "$models_before" ] || published=1
would_publish=0
! grep -q '^dry run: would publish' "$output" || would_publish=1

echo "== 4/5 Move the models of the sunnypilot $BRANCH branch to ${models_commit:0:7}"
if [ "$behind" -gt 0 ]; then
  if [ "$dry_run" -eq 1 ]; then
    echo "dry run: $BRANCH is $behind commit(s) behind origin, would fast-forward"
  else
    git -C "$SUNNYPILOT" merge --quiet --ff-only "refs/remotes/origin/$BRANCH" \
      || die "$BRANCH of $SUNNYPILOT cannot fast-forward to origin's (changes in the way)"
    echo "$BRANCH fast-forwarded to origin's ($behind commit(s))"
  fi
fi
pointer="$(git -C "$SUNNYPILOT" rev-parse "HEAD:$SUBMODULE")"
moved=0
if [ "$would_publish" -eq 1 ]; then
  echo "dry run: would move $SUBMODULE from ${pointer:0:7} to the new models commit and push $BRANCH"
elif [ "$pointer" = "$models_commit" ]; then
  echo "unchanged: $BRANCH already has the models at ${models_commit:0:7}"
elif [ "$dry_run" -eq 1 ]; then
  echo "dry run: would move $SUBMODULE from ${pointer:0:7} to ${models_commit:0:7} and push $BRANCH"
else
  git -C "$SUNNYPILOT/$SUBMODULE" fetch --quiet origin main
  git -C "$SUNNYPILOT/$SUBMODULE" checkout --quiet --detach "$models_commit"
  # Owed from here: a push that fails after the remote took it must still lead to a build.
  mkdir -p "$(dirname "$PENDING")"
  printf '%s\n' "$models_commit" > "$PENDING"
  old_head="$(git -C "$SUNNYPILOT" rev-parse HEAD)"
  # The submodule path alone: whatever else is staged in that clone is somebody else's and stays as it is.
  git -C "$SUNNYPILOT" commit --quiet -m "speed limits: models ${models_commit:0:7} (Send to the car, SignNet $signnet)" \
    -- "$SUBMODULE"
  if ! git -C "$SUNNYPILOT" push --quiet origin "$BRANCH:refs/heads/$BRANCH"; then
    # Back where it was: the branch, the path's index entry and the submodule (nothing else was touched).
    git -C "$SUNNYPILOT" update-ref "refs/heads/$BRANCH" "$old_head"
    git -C "$SUNNYPILOT" reset --quiet "$old_head" -- "$SUBMODULE"
    git -C "$SUNNYPILOT/$SUBMODULE" checkout --quiet --detach "$pointer"
    die "the push of $BRANCH failed: nothing changed in $SUNNYPILOT, run it again"
  fi
  git -C "$SUNNYPILOT" fetch --quiet origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
  echo "moved: $SUBMODULE ${pointer:0:7} -> ${models_commit:0:7}, $BRANCH $(git -C "$SUNNYPILOT" rev-parse --short HEAD) pushed"
  moved=1
fi

echo "== 5/5 Build $TESTING_BRANCH"
if [ "$published" -eq 1 ] || [ "$moved" -eq 1 ]; then
  mkdir -p "$(dirname "$PENDING")"
  printf '%s\n' "$models_commit" > "$PENDING"
fi
testing_head() { git ls-remote "$TESTING_REPO" "refs/heads/$TESTING_BRANCH" | cut -f1; }
if [ "$rebuild" -eq 0 ] && [ "$would_publish" -eq 0 ] && [ ! -f "$PENDING" ]; then
  echo "unchanged: nothing to build ($TESTING_BRANCH stays at $(testing_head | cut -c1-9); --rebuild builds it again)"
  echo "Nothing to update on the comma."
  exit 0
fi
if [ "$dry_run" -eq 1 ]; then
  echo "dry run: would build $TESTING_BRANCH with $BUILD sunny"
  exit 0
fi
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.git@github.com:cristianku/.pushInsteadOf \
  GIT_CONFIG_VALUE_0=https://github.com/cristianku/ WORKSPACE_ROOT="$WORKSPACE" "$BUILD" sunny </dev/null
if [ "${NO_PUSH:-false}" = "true" ]; then
  echo "NO_PUSH=true: $TESTING_BRANCH built in $WORKSPACE and committed there, not pushed"
  echo "$TESTING_BRANCH on GitHub is still $(testing_head)"
  exit 0
fi
rm -f "$PENDING"
# A trial on the comma ("Try on the comma now") ends with the update this build brings: the Models tab stops marking it.
if [ -f "$TRIAL" ]; then
  rm -f "$TRIAL"
  echo "the networks tried on the comma give way to this build at its update (models/on-comma-trial.json cleared)"
fi
echo "$TESTING_BRANCH is now $(testing_head)"
echo "On the comma: Settings, Software, Check for updates, then install."
