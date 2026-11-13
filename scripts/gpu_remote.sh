#!/usr/bin/env bash
set -uo pipefail

# The GPU runner (#77, 13 Nov 2026): runs one GPU job of the panel on CT 131 (serverone-ai, the two V100s) while the site
# lives on CT 1310 (serverone2, no GPU). The panel puts it in front of the job's own command when this machine has no
# card (panel.REMOTE_GPU_SCRIPT); a job never falls back to the CPU here: when CT 131 cannot be reached it stops
# with the reason.
#
# It never powers anything on or off (Cristian, 13 Nov 2026): serverone-ai and CT 131 are started and stopped only by
# Cristian or cluster-admin. The runner only connects to CT 131 when it is already up.
#
#   1. checks that CT 131 answers by ssh; when it does not, the job does not start and says to ask Cristian or
#      cluster-admin to start serverone-ai and CT 131;
#   2. copies the code as it is here (git stash create: the commit plus the changes not committed yet, never the
#      untracked files) into /root/gpu-runner/jobs/JOB/code on CT 131, with the .venv and .venv-gpu of
#      /root/sunnypilot.speedlimits there (PYTHONPATH puts this code's src before their editable install);
#   3. sends the data root (rsync, never --delete): everything but logs/, tmp/ and the live databases, which go as a
#      consistent copy (sqlite backup) into annotations/ there;
#   4. takes a card nothing computes on there (or the one the panel named), runs the job under setsid with the job's
#      environment, its paths turned into CT 131's, and streams its log here; the progress file comes back every
#      5 s; Stop in the panel (TERM) stops the job there too;
#   5. brings the results back (rsync --update, never --delete): everything but annotations/, logs/ and tmp/.
#
# usage: scripts/gpu_remote.sh COMMAND [ARGUMENTS...]     the job, as the panel would run it here
#        scripts/gpu_remote.sh --status                    whether CT 131 answers (reads only)
#
# Exit code: the job's own; 75 when CT 131 or a free card cannot be had.
#
# Environment: NNSLR_DATA_ROOT (data/), NNSLR_PROGRESS_FILE, NNSLR_JOB_RECORD, NNSLR_GPU_INDEX (the panel's), and:
# NNSLR_GPU_CT (root@10.10.10.213, CT 131), NNSLR_GPU_REMOTE_REPO (/root/sunnypilot.speedlimits on CT 131: its
# .venv-gpu, .venv and data/), NNSLR_GPU_RUNNER_DIR (/root/gpu-runner on CT 131).

REPO=$(cd "$(dirname "$0")/.." && pwd)
CT=${NNSLR_GPU_CT:-root@10.10.10.213}
REMOTE_REPO=${NNSLR_GPU_REMOTE_REPO:-/root/sunnypilot.speedlimits}
REMOTE_DATA=$REMOTE_REPO/data
RUNNER=${NNSLR_GPU_RUNNER_DIR:-/root/gpu-runner}
DATA=${NNSLR_DATA_ROOT:-$REPO/data}
UNAVAILABLE=75

log() { echo "[gpu runner $(date +%H:%M:%S)] $*"; }

SSH_BASE=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

OFF_MESSAGE="CT 131 (serverone-ai, the V100s) is off: ask Cristian or cluster-admin to start serverone-ai and CT 131, then start the job again"

case "${1:-}" in
    --status)
        if "${SSH_BASE[@]}" "$CT" true 2>/dev/null; then echo "CT 131 answers"; exit 0; fi
        echo "$OFF_MESSAGE"; exit 1 ;;
    '') echo "usage: $0 COMMAND [ARGUMENTS...] | --status" >&2; exit 2 ;;
esac

# The job: its id from its progress file (progress-<job id>.json), else the time.
JOB=$(basename "${NNSLR_PROGRESS_FILE:-}" .json); JOB=${JOB#progress-}
[ -n "$JOB" ] || JOB=$(date +%Y%m%d-%H%M%S)-remote
JOBDIR=$RUNNER/jobs/$JOB
WORK=$DATA/tmp/gpu-runner-$JOB
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
CONTROL=$WORK/ssh.sock
SSH=("${SSH_BASE[@]}" -o ControlMaster=auto -o "ControlPath=$CONTROL" -o ControlPersist=120)
on_ct() { "${SSH[@]}" "$CT" "$@"; }

# 1. CT 131 must already be up: the runner never starts it.
on_ct true 2>/dev/null || { log "$OFF_MESSAGE"; exit $UNAVAILABLE; }
log "CT 131 answers"

# 2. The code as it is here, with the venvs of CT 131.
revision=$(git -C "$REPO" stash create 2>/dev/null); revision=${revision:-HEAD}
log "code: $(git -C "$REPO" rev-parse --short HEAD)$( [ "$revision" != HEAD ] && echo ' + changes not committed') -> $JOBDIR/code"
on_ct "rm -rf '$JOBDIR' && mkdir -p '$JOBDIR/code'" || { log "cannot write $JOBDIR on CT 131"; exit $UNAVAILABLE; }
git -C "$REPO" archive --format=tar "$revision" | on_ct "tar -x -C '$JOBDIR/code' && ln -s '$REMOTE_REPO/.venv-gpu' '$REMOTE_REPO/.venv' '$JOBDIR/code/'" \
    || { log "the code could not be copied"; exit $UNAVAILABLE; }

# 3. The data. The databases the site writes go as a consistent copy.
log "sending the data root (rsync, only what changed)"
mkdir -p "$WORK/annotations"
for db in "$DATA"/annotations/*.sqlite; do
    [ -f "$db" ] || continue
    python3 -c 'import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True); dst = sqlite3.connect(sys.argv[2])
src.backup(dst); dst.close(); src.close()' "$db" "$WORK/annotations/$(basename "$db")" || { log "cannot copy $db"; exit 1; }
done
rsync -a --rsh="${SSH[*]}" --exclude=/logs/ --exclude=/tmp/ --exclude='/annotations/*.sqlite*' "$DATA/" "$CT:$REMOTE_DATA/" \
    && rsync -a --rsh="${SSH[*]}" "$WORK/annotations/" "$CT:$REMOTE_DATA/annotations/" \
    || { log "the data could not be sent"; exit 1; }

# 4. The job's environment, its paths those of CT 131.
remote_path() { local value=${1//$DATA/$REMOTE_DATA}; echo "${value//$REPO/$JOBDIR/code}"; }
{
    echo "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    while IFS='=' read -r name _; do
        case "$name" in
            NNSLR_GPU_CT|NNSLR_GPU_REMOTE_REPO) ;;
            NNSLR_*|CUDA_*|HF_*|TMPDIR|PYTHONUNBUFFERED) printf 'export %s=%q\n' "$name" "$(remote_path "${!name}")" ;;
        esac
    done < <(env)
    echo "export NNSLR_DATA_ROOT=$REMOTE_DATA TMPDIR=$REMOTE_DATA/tmp PYTHONPATH=$JOBDIR/code/src"
    printf 'cmd=('; for argument in "$@"; do printf '%q ' "$(remote_path "$argument")"; done; echo ')'
} > "$WORK/env"
on_ct "cat > '$JOBDIR/env'" < "$WORK/env" || { log "cannot write the job's environment"; exit $UNAVAILABLE; }

# The card: the panel's when it named one, else one that nothing computes on and no other job of ours holds, under a
# lock so two jobs never take the same one.
card=$(on_ct "mkdir -p '$RUNNER' && exec 9>'$RUNNER/cards.lock' && flock 9 && bash -s" <<EOF
want='${NNSLR_GPU_INDEX:-}'
held=\$(for f in $RUNNER/jobs/*/card; do [ -f "\$f" ] && kill -0 -- "-\$(cat "\${f%card}pgid" 2>/dev/null || echo 0)" 2>/dev/null && cat "\$f"; done)
busy=\$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader 2>/dev/null)
for line in \$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader,nounits | tr -d ' '); do
    index=\${line%%,*}; bus=\${line#*,}
    [ -n "\$want" ] && [ "\$index" != "\$want" ] && continue
    echo "\$held" | grep -qx "\$index" && continue
    echo "\$busy" | grep -qi "\$bus" && [ '${NNSLR_ALLOW_BUSY_GPU:-}' != 1 ] && continue
    echo "\$index" > '$JOBDIR/card'; echo "\$index"; exit 0
done
exit 1
EOF
) || { log "no V100 is free on CT 131: wait for the job that holds it"; on_ct "rm -rf '$JOBDIR'"; exit $UNAVAILABLE; }
log "card $card on CT 131 (CUDA_DEVICE_ORDER=PCI_BUS_ID)"

# The job, under setsid with its own process group, its log and its exit code in its folder: it outlives a dropped ssh.
on_ct "cat > '$JOBDIR/run.sh'" <<EOF
#!/usr/bin/env bash
. '$JOBDIR/env'
export CUDA_DEVICE_ORDER=PCI_BUS_ID NNSLR_GPU_INDEX=$card CUDA_VISIBLE_DEVICES=$card
cd '$JOBDIR/code'
echo "== on CT 131: \$(hostname), card $card, nnslr_tools from \$(.venv-gpu/bin/python -c 'import nnslr_tools, os; print(os.path.dirname(nnslr_tools.__file__))' 2>&1)"
nice -n 10 "\${cmd[@]}"
EOF
# The wrapper keeps a TERM of the whole group for itself (a handler, not an ignore: the job still gets it) to write the
# code.
on_ct "cd '$JOBDIR' && setsid bash -c 'trap : TERM INT HUP; echo \$\$ > pgid; bash run.sh > job.log 2>&1 < /dev/null; echo \$? > exit.code' > /dev/null 2>&1 < /dev/null &" \
    || { log "the job could not be started on CT 131"; exit $UNAVAILABLE; }
for _ in $(seq 1 20); do on_ct "test -s '$JOBDIR/pgid'" && break; sleep 0.5; done

stopping=0
stop_remote() {
    stopping=1
    log "stopped in the panel: stopping the job on CT 131"
    on_ct "kill -TERM -- -\$(cat '$JOBDIR/pgid')" 2>/dev/null || true
}
trap stop_remote TERM INT HUP

# The progress file, back every 5 s.
if [ -n "${NNSLR_PROGRESS_FILE:-}" ]; then
    remote_progress=$(remote_path "$NNSLR_PROGRESS_FILE")
    (
        trap - TERM INT HUP
        while :; do
            sleep 5
            on_ct "cat '$remote_progress'" > "$NNSLR_PROGRESS_FILE.part" 2>/dev/null \
                && [ -s "$NNSLR_PROGRESS_FILE.part" ] && mv -f "$NNSLR_PROGRESS_FILE.part" "$NNSLR_PROGRESS_FILE"
        done
    ) &
    progress_pid=$!
fi

# The log: streamed here; after a dropped ssh it goes on from the byte it had reached.
copy=$WORK/job.log
: > "$copy"
# A group gone without its code (the wrapper killed by hand): 143.
until on_ct "test -f '$JOBDIR/exit.code' || { kill -0 -- -\$(cat '$JOBDIR/pgid') 2>/dev/null || echo 143 > '$JOBDIR/exit.code'; false; }" 2>/dev/null; do
    offset=$(($(stat -c %s "$copy") + 1))
    on_ct "tail -c +$offset -F --pid=\$(cat '$JOBDIR/pgid') '$JOBDIR/job.log' 2>/dev/null" | tee -a "$copy" &
    wait $! 2>/dev/null
    # wait returns early when TERM arrives: the job goes on being followed until it stops.
    [ "$stopping" = 1 ] && sleep 2
    on_ct true 2>/dev/null || sleep 10
done
# What the job wrote after the last tail.
offset=$(($(stat -c %s "$copy") + 1))
on_ct "tail -c +$offset '$JOBDIR/job.log'" | tee -a "$copy"
code=$(on_ct "cat '$JOBDIR/exit.code'" 2>/dev/null); code=${code:-1}
[ -n "${progress_pid:-}" ] && kill "$progress_pid" 2>/dev/null
[ -n "${NNSLR_PROGRESS_FILE:-}" ] && on_ct "cat '$remote_progress'" > "$NNSLR_PROGRESS_FILE.part" 2>/dev/null \
    && [ -s "$NNSLR_PROGRESS_FILE.part" ] && mv -f "$NNSLR_PROGRESS_FILE.part" "$NNSLR_PROGRESS_FILE"
rm -f "${NNSLR_PROGRESS_FILE:-/nonexistent}.part"

# 5. The results back.
log "job ended with code $code on CT 131: bringing the results back"
rsync -a --update --rsh="${SSH[*]}" --exclude=/logs/ --exclude=/tmp/ --exclude=/annotations/ "$CT:$REMOTE_DATA/" "$DATA/" \
    || { log "the results could not be brought back: they stay in $REMOTE_DATA on CT 131"; [ "$code" = 0 ] && code=1; }
on_ct "rm -rf '$JOBDIR/code'" 2>/dev/null || true
"${SSH[@]}" -O exit "$CT" 2>/dev/null || true

exit "$code"
