#!/usr/bin/env bash
set -euo pipefail

# The one thing the GPU runner (scripts/gpu_remote.sh, #77) may do on the Proxmox hosts: wake serverone-ai and start
# CT 131 on it, say whether it is up, and turn serverone-ai off. Installed on serverone2 by cluster-admin as the forced
# command of CT 1310's key in root's authorized_keys:
#
#   command="/usr/local/sbin/gpu_host.sh",no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-pty ssh-ed25519 ...
#
# The verb comes in SSH_ORIGINAL_COMMAND (or $1 when run by hand):
#   status  prints "host up|down" and "ct running|stopped|unknown";
#   wake    Wake-on-LAN of serverone-ai, waits for its ssh (8 min at most), then pct start 131; prints "ct running";
#   off     shuts serverone-ai down (CT 131 with it), unless one of our jobs still runs on CT 131.
# Nothing else is accepted.
#
# Environment (for a test): GPU_HOST (root@10.10.10.110), GPU_HOST_MAC (18:c0:4d:79:15:2e), GPU_CT_ID (131),
# GPU_CT (root@10.10.10.213).

HOST=${GPU_HOST:-root@10.10.10.110}
MAC=${GPU_HOST_MAC:-18:c0:4d:79:15:2e}
CT_ID=${GPU_CT_ID:-131}
CT=${GPU_CT:-root@10.10.10.213}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=5)

verb=${SSH_ORIGINAL_COMMAND:-${1:-}}

host_up() { "${SSH[@]}" "$HOST" true 2>/dev/null; }
ct_state() { "${SSH[@]}" "$HOST" "pct status $CT_ID" 2>/dev/null | awk '{print $2}' || true; }

magic() {
    # The magic packet: 6 x 0xff, then the MAC 16 times, to the broadcast of every interface with an IPv4.
    python3 - "$MAC" <<'EOF'
import socket, sys
mac = bytes.fromhex(sys.argv[1].replace(':', ''))
packet = b'\xff' * 6 + mac * 16
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    for port in (9, 7):
        s.sendto(packet, ('255.255.255.255', port))
EOF
}

case "$verb" in
    status)
        if host_up; then
            echo "host up"
            state=$(ct_state)
            echo "ct ${state:-unknown}"
        else
            echo "host down"
            echo "ct stopped"
        fi
        ;;
    wake)
        if ! host_up; then
            echo "waking serverone-ai ($MAC)"
            for _ in $(seq 1 48); do
                magic
                sleep 10
                host_up && break
            done
            host_up || { echo "serverone-ai did not answer within 8 minutes"; exit 1; }
        fi
        if [ "$(ct_state)" != running ]; then
            echo "starting CT $CT_ID"
            "${SSH[@]}" "$HOST" "pct start $CT_ID"
        fi
        [ "$(ct_state)" = running ] || { echo "CT $CT_ID did not start"; exit 1; }
        echo "ct running"
        ;;
    off)
        host_up || { echo "host down"; exit 0; }
        # Our jobs leave a pid file in /root/gpu-runner/jobs while they run (scripts/gpu_remote.sh).
        if "${SSH[@]}" "$CT" 'for f in /root/gpu-runner/jobs/*/pgid; do [ -f "$f" ] && kill -0 -- "-$(cat "$f")" 2>/dev/null && exit 0; done; exit 1' 2>/dev/null; then
            echo "a job still runs on CT $CT_ID: not turning off"
            exit 0
        fi
        echo "turning serverone-ai off"
        "${SSH[@]}" "$HOST" 'shutdown -h now' || true
        ;;
    *)
        echo "gpu_host.sh: unknown verb '$verb' (status, wake, off)" >&2
        exit 2
        ;;
esac
