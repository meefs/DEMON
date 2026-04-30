#!/bin/bash
export PATH=/root/.local/bin:$PATH
cd /root/acestep
exec uv run python -u -m demos.realtime_motion_graph_web --host 0.0.0.0 --port 8765
