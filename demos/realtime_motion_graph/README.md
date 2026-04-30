# Realtime Motion-to-Music

Drive an ACE-Step music stream in real time from MIDI knobs or webcam
motion. Source audio plays back continuously; parameter changes reshape
the generation on the fly.

Two components:

- **Server** runs the GPU pipeline. Needs the full project install
  (torch, acestep, TensorRT engines).
- **Client** is a thin display and input app. Needs only a handful of
  Python packages (numpy, opencv, pygame, websockets, sounddevice,
  soundfile, mido). No torch, no CUDA, no acestep.

Run the server on a GPU machine, run the client anywhere, and point
the client at the server's WebSocket URL. For local dev on a single
machine, `python -m demos.realtime_motion_graph` starts both at once.

## Requirements

- **Server:** CUDA GPU, Python 3.11, full `uv sync` of this repo,
  pre-built TensorRT engines in the expected locations.
- **Client:** Python 3.11 on any OS. Audio output device. Webcam (for
  motion mode) or a MIDI controller (for knob mode).

## Server

Install the full project and start the server:

```bash
uv sync
uv run python -u -m demos.realtime_motion_graph.server --host 0.0.0.0 --port 8765
```

Flags:

- `--host <ip>` bind address (default `0.0.0.0`)
- `--port <n>` bind port (default `8765`)

All per-session options (decoder depth, VAE window, LoRA, fast VAE) are
set by the client at connect time.

## Client

Install only the client dependency group:

```bash
uv sync --only-group client --no-install-project
```

Run it:

```bash
uv run python -m demos.realtime_motion_graph.client \
    --remote ws://<server-host>:8765 \
    --audio path/to/source.wav
```

The server expects 48 kHz stereo; the client resamples automatically
(via soxr) if the source is a different rate.

### Client flags

- `--remote ws://host:port` required, server URL
- `--audio <file.wav>` required, 48 kHz stereo source
- `--midi` MIDI knob mode (default is webcam motion)
- `--sde` SDE denoise curves (requires `--midi`)
- `--lora` ask the server to load its configured LoRA
- `--fast-vae` use the fast VAE decoder engine on the server
- `--vae-window <seconds>` windowed decode length
- `--crop <seconds>` crop playback buffer
- `--depth <n>` pipeline depth (default 8)
- `--display <n>` pygame monitor index
- `--window-pos x,y` initial window position
- `--prompt <text>` initial prompt (editable live with RETURN)

### Controls

- **ESC** quit
- **RETURN** edit prompt; RETURN again to apply
- **Webcam mode:** visible motion drives the denoise amount
- **MIDI mode:** knobs on CC 70-77 across three banks
  - Bank 0 Core: denoise/sde_amp, seed, feedback, shift, etc.
  - Bank 1 Groups: eight 8-channel guidance groups
  - Bank 2 Keystones: six individual channels
  - Pad 3 (note 38) cycles banks
  - Pad 4 (note 39) resets all params

## Local dev: run both with one command

For local development on a single machine, launch the server and client
together:

```bash
uv run python -m demos.realtime_motion_graph --midi --audio source.wav
```

This spawns `server.py` as a subprocess bound to `127.0.0.1:8765`, waits
for it to finish loading the model, then runs `client.app` in the
foreground connected to it. Ctrl-C in the client shuts both down.

All client flags (`--midi`, `--sde`, `--lora`, `--audio`, `--prompt`,
etc.) pass through unchanged. Use `--host` and `--port` to override the
server bind.

## Layout

```
demos/realtime_motion_graph/
├── README.md
├── __main__.py               # local-dev launcher (server + client)
├── server.py                 # GPU server (full install only)
├── pipeline.py               # PipelineRunner (used by server)
└── client/                   # thin client package
    ├── app.py                # entrypoint
    ├── protocol.py           # WebSocket protocol + binary slice format
    ├── audio_engine.py       # sounddevice playback buffer
    ├── knobs.py              # MIDI knob banks
    ├── input_sources.py      # webcam and MIDI readers
    └── hud.py                # pygame/opencv overlay
```

`protocol.py`, `audio_engine.py`, and `knobs.py` are torch-free and
shared between the server and the thin client, so the wire format
stays consistent.

## Protocol

The client and server speak a mixed text/binary WebSocket protocol
defined in `client/protocol.py`.

### Init: client to server

1. JSON config:
   ```json
   {
     "sde": false, "lora": false, "depth": 8,
     "vae_window": 3.0, "crop": 0.0, "steps": 8,
     "prompt": "...",
     "lora_path": null, "fast_vae": false
   }
   ```
2. Binary audio upload: `<uint32 channels><uint32 samples>` header
   followed by `float32` samples, shape `(samples, channels)`.

### Init: server to client

1. JSON ready: `{"type":"ready","duration":...,"sample_rate":48000,"channels":2}`
2. Binary initial buffer: `float16` samples, shape `(samples, channels)`.

### Streaming: client to server

- JSON params: `{"type":"params","raw":{...},"playback_pos":<seconds>}`
- JSON prompt (optional): `{"type":"prompt","tags":"..."}`

### Streaming: server to client

- Binary slice (little-endian header):
  ```
  uint8  flags          // 0 = raw float16, 1 = zstd-compressed float16 delta
  uint32 start_sample
  uint32 num_samples
  uint16 channels
  float32 tick_ms
  float32 dec_ms
  uint32 num_gens
  ```
  Followed by raw `float16` samples, or a zstd-compressed `float16`
  delta that is added in place to the client's current audio.
- JSON params update: `{"type":"params_update","params":{...}}`
- JSON prompt applied: `{"type":"prompt_applied","tags":"..."}`

## Troubleshooting

- **No MIDI input devices found:** client auto-opens the first MIDI
  port. Plug in the controller before launching, or use webcam mode.
- **Cannot open webcam:** default is camera index 0. Edit
  `client/input_sources.py` to change.
- **Audio file wrong sample rate:** the client resamples to 48 kHz
  automatically using soxr. If you see `ImportError: soxr`, run
  `uv sync --group client` to install it.
- **GPU OOM on server:** lower `--depth` or `--vae-window`, or use
  `--crop` to shorten the buffer.
- **WebSocket disconnects mid-stream:** `max_size` is 50 MB on both
  ends. If the initial upload is too big for the network, use `--crop`.
