#!/bin/bash
set -euo pipefail
export PATH=/root/.local/bin:$PATH
cd /root/acestep
echo "branch: $(git branch --show-current)"
echo "head:   $(git log -1 --oneline)"
uv sync 2>&1 | tail -20
echo "=== installing extra runtime deps that aren't in lock ==="
uv pip install librosa sounddevice 2>&1 | tail -5
echo "=== done ==="
