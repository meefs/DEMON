"""Startup self-warmup that drives one synthetic session through the
WebSocket handler at boot to pay one-time engine costs before real
traffic arrives.

Measured 2026-05-18: a cold first session on a fresh engine takes ~40s to
``ready`` (TRT decoder-engine load ~3s + LoRA-refit manager ~7s + Session /
ModelContext / conditioning ~10s + first-tick pipeline build), while the
*second* session on the same warm engine is ~5-6s. ~30s of the cold path is
one-time-after-engine-start state that persists in the process. Driving one
synthetic default-fixture session through the WS handler at boot, before
the pod accepts real traffic, pays that once so every real "begin" gets the
warm path. Behaviour-neutral for real clients; the warmup session is fully
torn down by the WS handler's own finally.

``handle_client`` is injected as a positional parameter so this module
doesn't import the demo.
"""

import json
import struct
import time
from typing import Callable

import numpy as np
from websockets.exceptions import ConnectionClosed

from acestep.engine.obs import logger
from acestep.fixtures import audio_fixture


WARMUP_STATE: dict = {"done": False, "error": None, "seconds": None}

_WARMUP_FIXTURE = "low_fi_Gm_loop_60s_gnm.wav"  # PREFERRED_DEFAULT_FIXTURE
_WARMUP_PROMPT = "ambient electronic, warm pads"


class _WarmupWS:
    """In-process synthetic WebSocket that drives one default-fixture
    session through the handler to warm one-time engine state.

    Scripts the init handshake (config JSON, then the audio frame),
    lets the streaming loop spin long enough to build the pipeline +
    run the first generation tick, then raises ConnectionClosed so the
    handler's teardown path frees all per-session GPU state.
    """

    def __init__(self, config_json: str, audio_frame: bytes, budget_s: float = 35.0):
        self._queue = [config_json, audio_frame]
        self._t0 = time.monotonic()
        self._budget_s = budget_s
        self._initial_seen_at = None
        self.closed = False

    def recv(self, timeout=None):
        if self._queue:
            return self._queue.pop(0)
        # Spin until warm budget elapsed, then end the session.
        # Mimic "no client message" via TimeoutError when the caller
        # passed a timeout (the streaming loop polls that way); raise
        # ConnectionClosed once warmed so every recv site unwinds into
        # the handler's finally (which closes the session).
        now = time.monotonic()
        warm_enough = (
            now - self._t0 > self._budget_s
            or (self._initial_seen_at is not None
                and now - self._initial_seen_at > 10.0)
        )
        if warm_enough:
            raise ConnectionClosed(None, None)
        if timeout is not None:
            time.sleep(min(timeout, 0.05))
            raise TimeoutError
        time.sleep(0.1)
        raise TimeoutError

    def send(self, msg):
        # The initial buffer is a large binary frame (the echoed source);
        # spotting it tells us the init handshake finished so we can
        # bound how long the streaming loop spins after.
        if isinstance(msg, (bytes, bytearray)) and len(msg) > 1_000_000:
            if self._initial_seen_at is None:
                self._initial_seen_at = time.monotonic()

    def close(self, *args, **kwargs):
        self.closed = True


def _load_warmup_audio_frame() -> bytes:
    """Build the wire frame (<II channels,samples> + interleaved f32)
    for the default fixture, matching the browser's upload format."""
    path = str(audio_fixture(_WARMUP_FIXTURE))
    try:
        import soundfile as sf
        data, _sr = sf.read(path, dtype="float32", always_2d=True)  # (n, ch)
    except Exception:
        import librosa
        mono, _sr = librosa.load(path, sr=None, mono=True)
        data = np.stack([mono, mono], axis=1).astype(np.float32)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    ch = int(data.shape[1])
    samples = int(data.shape[0])
    interleaved = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
    return struct.pack("<II", ch, samples) + interleaved.tobytes()


def run_startup_warmup(
    handle_client: Callable,
    *,
    decoder_backend: str,
    vae_backend: str,
    checkpoint: str,
    offload_text_encoder: bool,
) -> None:
    """Drive one synthetic default-fixture session at boot. Never raises:
    a failed warmup must not stop the server from serving.

    ``handle_client`` is injected (rather than imported from
    ``demos/realtime_motion_graph_web/backend.py``) so this module stays
    free of demo imports.
    """
    t0 = time.monotonic()
    warm_log = logger.bind(component="warmup")
    warm_log.info("warmup_start fixture={}", _WARMUP_FIXTURE)
    try:
        cfg = {
            "fixture_name": _WARMUP_FIXTURE,
            "prompt": _WARMUP_PROMPT,
            "steps": 8,
            "depth": 4,
            "lora": False,
            "enabled_loras": [],
        }
        frame = _load_warmup_audio_frame()
        ws = _WarmupWS(json.dumps(cfg), frame)
        handle_client(
            ws,
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
            checkpoint=checkpoint,
            offload_text_encoder=offload_text_encoder,
        )
        WARMUP_STATE["done"] = True
        WARMUP_STATE["seconds"] = round(time.monotonic() - t0, 1)
        warm_log.info(
            "warmup_done elapsed_s={}", WARMUP_STATE["seconds"],
        )
    except Exception as exc:
        WARMUP_STATE["error"] = repr(exc)
        WARMUP_STATE["seconds"] = round(time.monotonic() - t0, 1)
        warm_log.opt(exception=True).error(
            "warmup_failed elapsed_s={} error={}",
            WARMUP_STATE["seconds"], exc,
        )
