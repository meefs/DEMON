"""Thin client for the realtime motion-to-music demo.

Torch-free. Imports only numpy, opencv, pygame, sounddevice, soundfile,
mido, websockets, and zstandard. Talks to a remote GPU server via the
binary protocol in :mod:`protocol`.
"""

from .audio_engine import AudioEngine
from .knobs import (
    CHANNEL_GROUPS,
    KEYSTONE_CHANNELS,
    KnobBank,
    KnobDef,
    build_banks,
)
from .protocol import (
    CROSSFADE_SECONDS,
    RemoteBackend,
    SAMPLE_RATE,
    SLICE_FLAG_DELTA,
    SLICE_FLAG_RAW,
    SLICE_HDR_FMT,
    SLICE_HDR_SIZE,
    T,
)

__all__ = [
    "AudioEngine",
    "CHANNEL_GROUPS",
    "CROSSFADE_SECONDS",
    "KEYSTONE_CHANNELS",
    "KnobBank",
    "KnobDef",
    "RemoteBackend",
    "SAMPLE_RATE",
    "SLICE_FLAG_DELTA",
    "SLICE_FLAG_RAW",
    "SLICE_HDR_FMT",
    "SLICE_HDR_SIZE",
    "T",
    "build_banks",
]
