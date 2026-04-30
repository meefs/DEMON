#!/bin/bash
# Rebuild VAE decode engine with decode_min_frames=25 (1s) so the demo's
# vae_window=3.0 default works.  The 240s engine built earlier had
# min=125 (5s) which is too large.
set -euo pipefail
export PATH=/root/.local/bin:$PATH
cd /root/acestep

# Move the too-large-min engine aside so the build script rebuilds.
mkdir -p /root/.claude-trash
mv /root/.daydream-scope/models/demon/trt_engines/vae_decode_fp16_240s/vae_decode_fp16_240s.engine \
   /root/.claude-trash/vae_decode_fp16_240s.engine.min125.$(date -u +%Y%m%dT%H%M%S)

uv run python - <<'PY'
# Monkeypatch decode_min_frames to 25, then rebuild just the VAE decode engine.
from acestep.engine.trt.vae_export import VAETRTBuildConfig, build_vae_decode_engine
from acestep.paths import trt_engines_dir
from pathlib import Path

cfg = VAETRTBuildConfig(
    workspace_gb=8.0,
    decode_min_frames=25,        # 1s min window
    decode_opt_frames=1500,      # 60s opt
    decode_max_frames=6000,      # 240s max
)

eng_dir = trt_engines_dir()
onnx = eng_dir / "_onnx_vae" / "vae_decode" / "vae_decode.onnx"
out = eng_dir / "vae_decode_fp16_240s" / "vae_decode_fp16_240s.engine"
print(f"rebuilding: {out}")
print(f"min={cfg.decode_min_frames} opt={cfg.decode_opt_frames} max={cfg.decode_max_frames}")
build_vae_decode_engine(onnx, out, config=cfg)
print(f"OK: {out.stat().st_size / 1e6:.1f} MB")
PY
