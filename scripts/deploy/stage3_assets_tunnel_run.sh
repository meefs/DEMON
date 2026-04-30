#!/bin/bash
# Run AFTER stage 2 succeeds.
# 1. Install cloudflared
# 2. Start cloudflared tunnel
# 3. Start the realtime_motion_graph_web demo server
set -euo pipefail
export PATH="/root/.local/bin:$PATH"

CF_TOKEN="${CLOUDFLARE_TOKEN:-}"
if [ -z "$CF_TOKEN" ]; then
    echo "CLOUDFLARE_TOKEN not set" >&2
    exit 1
fi

# ---------- install cloudflared ----------
if ! command -v cloudflared >/dev/null 2>&1; then
    echo "Installing cloudflared..."
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
fi
cloudflared --version

# ---------- start cloudflared tunnel ----------
# Token-managed: dashboard controls public-hostname → service mapping.
mkdir -p /var/log/cloudflared
pkill -f "cloudflared tunnel" 2>/dev/null || true
nohup cloudflared tunnel --no-autoupdate run --token "$CF_TOKEN" \
    > /var/log/cloudflared/cloudflared.log 2>&1 &
CF_PID=$!
echo "cloudflared PID=$CF_PID"

# ---------- start demo server ----------
cd /root/acestep
mkdir -p /var/log/acestep
pkill -f "demos.realtime_motion_graph_web" 2>/dev/null || true
nohup uv run python -u -m demos.realtime_motion_graph_web --host 0.0.0.0 --port 8765 \
    > /var/log/acestep/demo.log 2>&1 &
DEMO_PID=$!
echo "demo PID=$DEMO_PID"

# ---------- wait for both to come up ----------
for i in $(seq 1 60); do
    if curl -fsS http://localhost:8765/ -o /dev/null 2>/dev/null; then
        echo "[stage3] demo http OK"
        break
    fi
    sleep 2
done

for i in $(seq 1 30); do
    if grep -q "Registered tunnel connection" /var/log/cloudflared/cloudflared.log 2>/dev/null; then
        echo "[stage3] cloudflared connected"
        break
    fi
    sleep 2
done

echo "=== STAGE 3 STARTED ==="
echo "Demo log:        /var/log/acestep/demo.log"
echo "Cloudflared log: /var/log/cloudflared/cloudflared.log"
echo "PIDs: cloudflared=$CF_PID demo=$DEMO_PID"
tail -n 5 /var/log/cloudflared/cloudflared.log || true
echo "---"
tail -n 5 /var/log/acestep/demo.log || true
