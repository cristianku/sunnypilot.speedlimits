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
#      /root/sunnypilot.speedlimits there (PYTHONPATH puts this code's src before their editable install), and the
#      car's speed_limits package of the sunnypilot clone here into JOB/sunnypilot (NNSLR_SUNNYPILOT_CLONE);
#   3. sends the data root (rsync, never --delete): everything but logs/, tmp/ and the live databases, which go as a
#      consistent copy (sqlite backup) into annotations/ there;
#   4. takes a card nothing computes on there (or the one the panel named), runs the job under setsid with the job's
#      environment, its paths turned into CT 131's, and streams its log here; the progress file comes back every
#      5 s; Stop in the panel (TERM) stops the job there too; Prepare new drives (helpers/1_prepare.sh) also brings
#      back every 10 s what the panel shows its progress from, and each drive as soon as it is ready to review;
#   5. brings the results back (rsync --update, never --delete): everything but annotations/, logs/ and tmp/ (of a
#      Prepare, its logs in logs/prepare/ too).
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

# Prepare new drives (14 Nov 2026): the panel shows how far it is from files of the data root here
# (helpers/route_status.py, Panel.preparation): logs/prepare/batch.json, the links of each drive's videos, the
# frames.jsonl of each segment taken out, the report being written (preannotations/<drive>/.pending-<run>/
# preannotations.jsonl) and the drives' logs. Before this, they came back only when the whole job ended, logs/ never:
# for its whole run the panel read the batch of an earlier run and stood at 0 %. Now they come back every 10 s (the
# batch once this job wrote it), and a drive the job says is "ready to review" comes back whole at once, so it can be
# reviewed while the others are being prepared.
PREPARE=0
[ "$(basename "$1")" = 1_prepare.sh ] && PREPARE=1
if [ "$PREPARE" = 1 ]; then
    # CT 131's clock: the runs of the report are named by it. Taken before anything is written there.
    JOB_STAMP=$(on_ct "date -u +%Y%m%dT%H%M%SZ") && [ -n "$JOB_STAMP" ] || { log "$OFF_MESSAGE"; exit $UNAVAILABLE; }
    # The batch of an earlier run would show its drives until this job writes its own (1_prepare.sh replaces it too).
    rm -f "$DATA/logs/prepare/batch.json"
fi

# 2. The code as it is here, with the venvs of CT 131.
revision=$(git -C "$REPO" stash create 2>/dev/null); revision=${revision:-HEAD}
log "code: $(git -C "$REPO" rev-parse --short HEAD)$( [ "$revision" != HEAD ] && echo ' + changes not committed') -> $JOBDIR/code"
on_ct "rm -rf '$JOBDIR' && mkdir -p '$JOBDIR/code'" || { log "cannot write $JOBDIR on CT 131"; exit $UNAVAILABLE; }
git -C "$REPO" archive --format=tar "$revision" | on_ct "tar -x -C '$JOBDIR/code' && ln -s '$REMOTE_REPO/.venv-gpu' '$REMOTE_REPO/.venv' '$JOBDIR/code/'" \
    || { log "the code could not be copied"; exit $UNAVAILABLE; }
# The car's speed_limits package as it is here (brain_inputs.py, vision.py, models/...): the server loads it by its path
# (NNSLR_SUNNYPILOT_CLONE), so a job on CT 131 must not read that machine's own, older clone.
CAR_CLONE=${NNSLR_SUNNYPILOT_CLONE:-/root/sunnypilot}
CAR_PACKAGE=openpilot/sunnypilot/speed_limits
[ -d "$CAR_CLONE/$CAR_PACKAGE" ] || { log "no car package in $CAR_CLONE/$CAR_PACKAGE"; exit 1; }
log "car package: $(git -C "$CAR_CLONE" rev-parse --short HEAD 2>/dev/null) of $CAR_CLONE -> $JOBDIR/sunnypilot"
tar -C "$CAR_CLONE" --exclude=__pycache__ -cf - "$CAR_PACKAGE" | on_ct "mkdir -p '$JOBDIR/sunnypilot' && tar -x -C '$JOBDIR/sunnypilot'" \
    || { log "the car package could not be copied"; exit $UNAVAILABLE; }

# 3. The data. The databases the site writes go as a consistent copy.
log "sending the data root (rsync, only what changed)"
mkdir -p "$WORK/annotations"
for db in "$DATA"/annotations/*.sqlite; do
    [ -f "$db" ] || continue
    python3 -c 'import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True); dst = sqlite3.connect(sys.argv[2])
src.backup(dst); dst.close(); src.close()' "$db" "$WORK/annotations/$(basename "$db")" || { log "cannot copy $db"; exit 1; }
done
# Never a report being written (.pending-<run>): one copied here by a Prepare at work there is older than the live one,
# which this send would replace under the proposer (it then finds its report short and fails the drive).
rsync -a --rsh="${SSH[*]}" --exclude=/logs/ --exclude=/tmp/ --exclude='/annotations/*.sqlite*' \
    --exclude='/derived/preannotations/*/.pending-*' "$DATA/" "$CT:$REMOTE_DATA/" \
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
    echo "export NNSLR_DATA_ROOT=$REMOTE_DATA TMPDIR=$REMOTE_DATA/tmp PYTHONPATH=$JOBDIR/code/src NNSLR_SUNNYPILOT_CLONE=$JOBDIR/sunnypilot"
    # The comma's videos (Prepare): the same path on CT 131 (/root/comma-videos); left unset, the helper would look for
    # them next to the job's code.
    printf 'export COMMA_VIDEOS_DIR=%q\n' "${COMMA_VIDEOS_DIR:-$(dirname "$REPO")/comma-videos}"
    printf 'cmd=('; for argument in "$@"; do printf '%q ' "$(remote_path "$argument")"; done; echo ')'
} > "$WORK/env"
on_ct "cat > '$JOBDIR/env'" < "$WORK/env" || { log "cannot write the job's environment"; exit $UNAVAILABLE; }

# The card: the panel's when it named one, else one that nothing computes on and no other job of ours holds, under a
# lock so two jobs never take the same one. A job holds its card until its exit.code is written: while its process
# group's leader (the wrapper below, in the job's folder) lives, or, before the job has started (no pgid yet), for
# 2 minutes. A card left by a runner that died before starting its job, or by a job that ended or a CT 131 restarted
# under it (its pgid now another process's), is held by nobody.
card=$(on_ct "mkdir -p '$RUNNER' && exec 9>'$RUNNER/cards.lock' && flock 9 && bash -s" <<EOF
want='${NNSLR_GPU_INDEX:-}'
held=\$(for f in $RUNNER/jobs/*/card; do
    [ -f "\$f" ] || continue; d=\${f%/card}
    [ -f "\$d/exit.code" ] && continue
    p=\$(cat "\$d/pgid" 2>/dev/null)
    if [ -n "\$p" ]; then [ "\$(readlink "/proc/\$p/cwd" 2>/dev/null)" = "\$d" ] || continue
    elif [ -z "\$(find "\$f" -mmin -2 2>/dev/null)" ]; then continue; fi
    cat "\$f"
done)
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

# Prepare (above): what comes back while the job runs.
RSYNC_BACK=(rsync -a --update --rsh="${SSH[*]}")

# The drives of this job's batch, one per line, once the job wrote it (newer than its environment); none before.
batch_routes() {
    on_ct "f='$REMOTE_DATA/logs/prepare/batch.json'; [ \"\$f\" -nt '$JOBDIR/env' ] && cat \"\$f\"" 2>/dev/null \
        | python3 -c 'import json, re, sys
for route in json.load(sys.stdin).get("routes", []):
    if isinstance(route, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", route):
        print(route)' 2>/dev/null
}

# logs/prepare/: the batch, the linking's output and the drives' logs (Details of a drive in the panel).
prepare_logs() {
    local route rules=(--include=/batch.json --include=/link-videos.json --include=/link-videos.err)
    for route in "$@"; do rules+=("--include=/$route.log"); done
    mkdir -p "$DATA/logs/prepare"
    "${RSYNC_BACK[@]}" "${rules[@]}" --exclude='*' "$CT:$REMOTE_DATA/logs/prepare/" "$DATA/logs/prepare/" 2>/dev/null
}

# The links of the drives' videos (raw/routes/<drive>/<segment>/, made by the job's linking): without them the panel
# counts a drive as "to link" until the end. Links only: a video file itself never comes back here (--max-size).
prepare_links() {
    local route rules=()
    for route in "$@"; do rules+=("--include=/$route/" "--include=/$route/*/" "--include=/$route/*/*"); done
    mkdir -p "$DATA/raw/routes"
    "${RSYNC_BACK[@]}" --max-size=1M "${rules[@]}" --exclude='*' "$CT:$REMOTE_DATA/raw/routes/" "$DATA/raw/routes/" 2>/dev/null
}

# A drive at work: only what its progress is counted from, not its pictures.
prepare_progress() {
    local route rules=(--include=/frames/ --include=/preannotations/)
    [ "$#" -gt 0 ] || return 0
    for route in "$@"; do
        rules+=("--include=/frames/$route/" "--include=/frames/$route/*/" "--include=/frames/$route/*/frames.jsonl"
                "--include=/preannotations/$route/" "--include=/preannotations/$route/.pending-*/"
                "--include=/preannotations/$route/.pending-*/preannotations.jsonl")
    done
    "${RSYNC_BACK[@]}" "${rules[@]}" --exclude='*' "$CT:$REMOTE_DATA/derived/" "$DATA/derived/" 2>/dev/null
}

# A .pending-<run> of this job copied while it was written, gone from CT 131 (it became <run>, the finished report):
# it goes here too. One CT 131 still has (a drive stopped half way) stays, as there.
drop_moved_pending() {
    local route=$1 pending answer
    for pending in "$DATA/derived/preannotations/$route"/.pending-*; do
        [ -d "$pending" ] || continue
        [[ "${pending##*/.pending-}" < "$JOB_STAMP" ]] && continue
        answer=$(on_ct "test -e '$REMOTE_DATA/derived/preannotations/$route/${pending##*/}' && echo kept || echo gone" 2>/dev/null)
        [ "$answer" = gone ] && rm -rf -- "$pending"
    done
    return 0
}

# A drive ready to review: all it has in derived/ (pictures, report, positions, video), its latest.json last: with it
# the site takes the drive for review.
prepare_ready() {
    local route=$1
    "${RSYNC_BACK[@]}" --exclude="/preannotations/$route/latest.json" --exclude="/preannotations/$route/.pending-*" \
        --include='/*/' --include="/*/$route/***" --include="/*/$route.*" --exclude='*' \
        "$CT:$REMOTE_DATA/derived/" "$DATA/derived/" 2>/dev/null || return 1
    "${RSYNC_BACK[@]}" "$CT:$REMOTE_DATA/derived/preannotations/$route/latest.json" "$DATA/derived/preannotations/$route/" \
        2>/dev/null || return 1
    drop_moved_pending "$route"
}

# One pass: the batch and the logs, each drive at work, each drive newly ready (named once in $WORK/ready).
prepare_mirror() {
    local routes route working=()
    routes=$(batch_routes)
    [ -n "$routes" ] || return 0
    # shellcheck disable=SC2086
    prepare_logs $routes
    # shellcheck disable=SC2086
    prepare_links $routes
    for route in $routes; do
        grep -qxF "$route" "$WORK/ready" 2>/dev/null && continue
        if grep -qxF "$route: ready to review" "$copy" 2>/dev/null; then
            prepare_ready "$route" && echo "$route" >> "$WORK/ready"
        else
            working+=("$route")
        fi
    done
    prepare_progress "${working[@]}"
    return 0
}

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
# code. "cd ... || exit 1; setsid ... &" and never "cd ... && setsid ... &": the shell runs an "A && B &" list in a
# subshell that keeps ssh's pipes open while B runs, so this ssh would return only when the job ends, and until then
# no log, no progress and no Stop would reach the panel (the brain of 13 Nov 2026, 23:09).
on_ct "cd '$JOBDIR' || exit 1; setsid bash -c 'trap : TERM INT HUP; echo \$\$ > pgid; bash run.sh > job.log 2>&1 < /dev/null; echo \$? > exit.code' > /dev/null 2>&1 < /dev/null &" \
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

# Prepare: its progress and its ready drives, back every 10 s (above).
if [ "$PREPARE" = 1 ]; then
    : > "$WORK/ready"
    (
        trap - TERM INT HUP
        until [ -e "$WORK/mirror.stop" ]; do
            sleep 10
            [ -e "$WORK/mirror.stop" ] || prepare_mirror
        done
    ) &
    mirror_pid=$!
fi
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
# The pass of the mirror at work ends before the results come back; one last pass takes the drives the job made ready
# since (each with its latest.json last, so none is taken for review before its pictures are here).
if [ -n "${mirror_pid:-}" ]; then
    touch "$WORK/mirror.stop"; wait "$mirror_pid" 2>/dev/null
    prepare_mirror
fi
[ -n "${NNSLR_PROGRESS_FILE:-}" ] && on_ct "cat '$remote_progress'" > "$NNSLR_PROGRESS_FILE.part" 2>/dev/null \
    && [ -s "$NNSLR_PROGRESS_FILE.part" ] && mv -f "$NNSLR_PROGRESS_FILE.part" "$NNSLR_PROGRESS_FILE"
rm -f "${NNSLR_PROGRESS_FILE:-/nonexistent}.part"

# 5. The results back.
log "job ended with code $code on CT 131: bringing the results back"
rsync -a --update --rsh="${SSH[*]}" --exclude=/logs/ --exclude=/tmp/ --exclude=/annotations/ "$CT:$REMOTE_DATA/" "$DATA/" \
    || { log "the results could not be brought back: they stay in $REMOTE_DATA on CT 131"; [ "$code" = 0 ] && code=1; }
# Prepare: the drives' logs (logs/ does not come back above) and the .pending-<run> copied that became reports.
if [ "$PREPARE" = 1 ]; then
    # Only when this job wrote its batch: CT 131's batch of an earlier run would come back as this job's.
    routes=$(batch_routes)
    if [ -n "$routes" ]; then
        # shellcheck disable=SC2086
        prepare_logs $routes
        for route in $routes; do drop_moved_pending "$route"; done
    fi
fi
on_ct "rm -rf '$JOBDIR/code' '$JOBDIR/sunnypilot'" 2>/dev/null || true
"${SSH[@]}" -O exit "$CT" 2>/dev/null || true

exit "$code"
