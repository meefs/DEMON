#!/bin/bash
# Run AFTER bootstrap_h100.sh succeeds.
# 1. Force-download ACE-Step main model (idempotent)
# 2. Build TRT engines (240s decoder + VAE)
set -euo pipefail
export PATH="/root/.local/bin:$PATH"
export HF_TOKEN="${HF_TOKEN:-}"

cd /root/acestep

echo "=== STAGE 2: model download ==="
uv run python -c "
from acestep.model_downloader import ensure_main_model, check_main_model_exists
from acestep.paths import checkpoints_dir
cp = checkpoints_dir()
cp.mkdir(parents=True, exist_ok=True)
print('checkpoints_dir:', cp)
if check_main_model_exists(cp):
    print('Main model already present.')
else:
    ok, msg = ensure_main_model(cp, prefer_source='huggingface')
    print(msg)
    assert ok, 'download failed'
print('OK')
"

echo "=== STAGE 2: ONNX export + TRT engine build (decoder 240s + VAE) ==="
# decoder_mixed_refit_b8_240s + vae_decode_fp16_240s + vae_encode_fp16_240s
uv run python -m acestep.engine.trt.build --all --duration 240 --decoder-mixed --decoder-refit

echo "=== STAGE 2: verify engines exist ==="
ls -lh /root/.daydream-scope/models/demon/trt_engines/decoder_mixed_refit_b8_240s/decoder_mixed_refit_b8_240s.engine \
       /root/.daydream-scope/models/demon/trt_engines/vae_decode_fp16_240s/vae_decode_fp16_240s.engine \
       /root/.daydream-scope/models/demon/trt_engines/vae_encode_fp16_240s/vae_encode_fp16_240s.engine

echo "=== STAGE 2 DONE ==="
