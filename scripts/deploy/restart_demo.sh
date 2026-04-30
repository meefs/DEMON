#!/bin/bash
# Kill the demo by exact-PID rather than pkill -f, which matches the
# enclosing SSH shell when the pattern includes the demo module name.
set -uo pipefail

PIDS=$(pgrep -f 'python3 -u -m demos.realtime_motion_graph_web' || true)
echo "killing demo PIDs: ${PIDS:-(none)}"
if [ -n "${PIDS}" ]; then
    kill ${PIDS} 2>/dev/null || true
    sleep 3
    # SIGKILL stragglers
    PIDS=$(pgrep -f 'python3 -u -m demos.realtime_motion_graph_web' || true)
    if [ -n "${PIDS}" ]; then
        kill -9 ${PIDS} 2>/dev/null || true
        sleep 1
    fi
fi

# Also kill the wrapping `uv run python ...`, which holds /var/log/acestep/demo.log open
WRAP=$(pgrep -f 'uv run python -u -m demos.realtime_motion_graph_web' || true)
if [ -n "${WRAP}" ]; then
    echo "killing wrapper: ${WRAP}"
    kill ${WRAP} 2>/dev/null || true
    sleep 2
fi

echo "spawning new demo"
nohup setsid bash /root/run_demo.sh </dev/null >/var/log/acestep/demo.log 2>&1 &
disown
sleep 8
echo "now-running:"
pgrep -af 'python3 -u -m demos.realtime_motion_graph_web' || echo "  (none)"
echo "log tail:"
tail -10 /var/log/acestep/demo.log
