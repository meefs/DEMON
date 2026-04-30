"""Thin client for the realtime motion-to-music demo.

Connects to a remote GPU server over WebSocket. Streams MIDI knob or
webcam-motion params to the server and receives decoded audio slices
back in near-real-time.

Usage:
    uv run python -m demos.realtime_motion_graph.client \\
        --remote ws://server-host:8765 --audio path/to/file.wav

    # MIDI mode
    uv run python -m demos.realtime_motion_graph.client \\
        --remote ws://server-host:8765 --midi --audio path/to/file.wav

    # MIDI + SDE curves
    uv run python -m demos.realtime_motion_graph.client \\
        --remote ws://server-host:8765 --midi --sde --audio path/to/file.wav

Controls:
    ESC = quit
    RETURN = edit prompt
    Pad 3 (note 38) = cycle MIDI knob bank forward
    Pad 4 (note 39) = reset all params to defaults
"""

import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pygame
import soundfile as sf

from .audio_engine import AudioEngine
from .hud import compute_waveform_image, draw_hud
from .input_sources import MidiKnobs, MotionTracker
from .knobs import build_banks
from .protocol import RemoteBackend, SAMPLE_RATE, SLICE_FLAG_DELTA


def _load_audio_48k(audio_path: Path) -> np.ndarray:
    """Load audio file and return a (channels, samples) float32 array at 48kHz.

    The server expects 48 kHz input. Non-48 kHz files are resampled with
    soxr; if soxr is missing, raise a clear error with install guidance.
    """
    data, sr = sf.read(str(audio_path), dtype="float32")
    if data.ndim == 1:
        data = data[:, None]
    # data is (samples, channels), the layout soxr expects.
    if sr != SAMPLE_RATE:
        try:
            import soxr
        except ImportError as exc:
            raise SystemExit(
                f"[Client] {audio_path.name} is {sr} Hz but the server "
                f"expects {SAMPLE_RATE} Hz, and the soxr resampler is not "
                f"installed.\n"
                f"  Install: uv sync --group client\n"
                f"  Or convert upstream: "
                f"ffmpeg -i {audio_path.name} -ar {SAMPLE_RATE} -ac 2 out.wav"
            ) from exc
        data = soxr.resample(data, sr, SAMPLE_RATE).astype(np.float32, copy=False)
    waveform = data.T.astype(np.float32, copy=False)  # (channels, samples)
    waveform = waveform[:2, :int(60.0 * SAMPLE_RATE)]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]
    return waveform


def main():
    audio_path = None
    use_midi = False
    use_sde = False
    use_lora = False
    use_fast_vae = False
    vae_window = 0.0
    crop_seconds = 0.0
    depth = 8
    display_index = 0
    window_pos = None
    remote_url = None
    initial_prompt = "deathstep, heavy bass, dark atmosphere"

    args = list(sys.argv[1:])

    def _pop_flag(name):
        if name in args:
            args.remove(name)
            return True
        return False

    def _pop_value(name):
        if name in args:
            i = args.index(name)
            val = args[i + 1]
            del args[i:i + 2]
            return val
        return None

    use_midi = _pop_flag("--midi")
    use_sde = _pop_flag("--sde")
    use_lora = _pop_flag("--lora")
    use_fast_vae = _pop_flag("--fast-vae")

    v = _pop_value("--vae-window")
    if v is not None:
        vae_window = float(v)
    v = _pop_value("--crop")
    if v is not None:
        crop_seconds = float(v)
    v = _pop_value("--depth")
    if v is not None:
        depth = int(v)
    v = _pop_value("--display")
    if v is not None:
        display_index = int(v)
    v = _pop_value("--window-pos")
    if v is not None:
        xs, _, ys = v.partition(",")
        window_pos = (int(xs), int(ys))
    v = _pop_value("--remote")
    if v is not None:
        remote_url = v
    v = _pop_value("--prompt")
    if v is not None:
        initial_prompt = v
    v = _pop_value("--audio")
    if v is not None:
        audio_path = Path(v)
    elif args:
        audio_path = Path(args[0])

    if remote_url is None:
        raise SystemExit(
            "[Client] --remote ws://host:port is required.\n"
            "Example: --remote ws://192.168.1.10:8765 --audio song.wav"
        )
    if audio_path is None:
        raise SystemExit(
            "[Client] --audio <file.wav> is required (48 kHz stereo)."
        )

    k1_name = "sde_amp" if use_sde else "denoise"

    print("=" * 60)
    print("Real-Time Motion-to-Music (THIN CLIENT)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load source audio (numpy-only; no torch on the client)
    # ------------------------------------------------------------------
    print(f"[Setup] Loading source audio from {audio_path}")
    waveform = _load_audio_48k(audio_path)

    # ------------------------------------------------------------------
    # Connect to remote server
    # ------------------------------------------------------------------
    remote = RemoteBackend(remote_url, waveform, {
        "sde": use_sde, "lora": use_lora, "depth": depth,
        "vae_window": vae_window if vae_window > 0 else 3.0,
        "crop": crop_seconds, "steps": 8,
        "prompt": initial_prompt,
        "lora_path": None,  # server-side path; not supported from thin client
        "fast_vae": use_fast_vae,
    })
    src_np = remote.initial_buffer

    # Rebuild banks with server's actual LoRA count
    lora_count = remote.lora_count if use_lora else 0
    banks = build_banks(use_sde, lora=lora_count)
    if crop_seconds > 0:
        src_np = src_np[:int(crop_seconds * SAMPLE_RATE)]

    # ------------------------------------------------------------------
    # Audio + input
    # ------------------------------------------------------------------
    audio_eng = AudioEngine(src_np, SAMPLE_RATE)
    audio_eng.start()
    print(f"[Audio] Playing ({audio_eng.duration:.1f}s, {SAMPLE_RATE}Hz)")

    tracker = None
    midi_knobs = None
    if use_midi:
        midi_knobs = MidiKnobs(banks)
        disp_w, disp_h = 960, 720
    else:
        tracker = MotionTracker()
        test_ok, test_frame = tracker.cap.read()
        if not test_ok:
            print("Cannot open webcam")
            return
        disp_h, disp_w = test_frame.shape[:2]
        disp_w, disp_h = int(disp_w * 1.5), int(disp_h * 1.5)

    if window_pos is not None:
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{window_pos[0]},{window_pos[1]}"
    pygame.init()
    pygame.key.set_repeat(400, 35)
    try:
        sizes = pygame.display.get_desktop_sizes()
        if display_index < 0 or display_index >= len(sizes):
            print(
                f"[Display] --display {display_index} out of range "
                f"(0..{len(sizes) - 1}); using 0."
            )
            display_index = 0
    except Exception:
        pass
    mode_str = "MIDI" + (" SDE" if use_sde else "") if use_midi else "Webcam"
    screen = pygame.display.set_mode((disp_w, disp_h), display=display_index)
    pygame.display.set_caption(f"Real-Time {mode_str} (Thin Client)")

    print(f"\n  Mode: {mode_str}  |  Remote: {remote_url}")
    if use_midi:
        for bi, bank in enumerate(banks):
            print(f"  Bank {bi} ({bank.name}): {', '.join(f'K{i+1}={name}' for i, name in enumerate(bank.knobs))}")
        print(f"  Pad 3 = cycle bank")
    print(f"  {'Move knobs' if use_midi else 'Move'} to change the music. ESC to quit.\n")

    # ------------------------------------------------------------------
    # Shared state
    # ------------------------------------------------------------------
    motion_val = [0.0]
    motion_lock = threading.Lock()
    running = [True]
    SEED = 1528

    params = {"num_gens": 0, "tick_ms": 0.0, "dec_ms": 0.0}
    params[k1_name] = 0.0
    params["seed"] = SEED
    params["feedback"] = 0.0
    params["shift"] = 3.0
    if use_lora:
        for i in range(1, lora_count + 1):
            params[f"lora_str_{i}"] = 0.0
    if use_sde:
        params["periodicity"] = 0.0

    prompt_text = [initial_prompt]
    prompt_input_active = [False]
    prompt_input_buffer = [""]
    encoding_in_progress = [False]

    def encode_and_apply(text):
        encoding_in_progress[0] = True
        try:
            remote.send_prompt(text)
            prompt_text[0] = text
        finally:
            encoding_in_progress[0] = False

    sde_curve_display = [None]

    # ------------------------------------------------------------------
    # Remote send/recv threads
    # ------------------------------------------------------------------
    def remote_send_loop():
        while running[0]:
            if use_midi:
                raw = midi_knobs.get_all_values()
            else:
                with motion_lock:
                    m = motion_val[0]
                raw = {k1_name: m, "seed": 0.0, "feedback": 0.0, "shift": 0.5}
                if use_sde:
                    raw["periodicity"] = 0.0
            remote.send_raw(raw, audio_eng.position / SAMPLE_RATE)
            time.sleep(0.008)

    def remote_recv_loop():
        while running[0]:
            got_any = False
            while True:
                result = remote.recv(timeout=0.005)
                if result is None:
                    break
                got_any = True
                kind, data = result
                if kind == "audio":
                    s = data["start_sample"]
                    n = data["num_samples"]
                    if data.get("flags") == SLICE_FLAG_DELTA:
                        end = min(s + n, len(audio_eng.current))
                        with audio_eng._lock:
                            audio_eng.current[s:end] += data["audio"][:end - s]
                    else:
                        audio_eng.patch(data["audio"], s)
                    params["num_gens"] = data["num_gens"]
                    params["tick_ms"] = data["tick_ms"]
                    params["dec_ms"] = data["dec_ms"]
                elif kind == "json":
                    msg = data
                    if msg.get("type") == "prompt_applied":
                        prompt_text[0] = msg.get("tags", prompt_text[0])
                        encoding_in_progress[0] = False
                    if "params" in msg:
                        for k, v in msg["params"].items():
                            params[k] = v
            if not got_any:
                time.sleep(0.001)

    send_thread = threading.Thread(target=remote_send_loop, daemon=True)
    recv_thread = threading.Thread(target=remote_recv_loop, daemon=True)
    send_thread.start()
    recv_thread.start()

    # ------------------------------------------------------------------
    # Display loop
    # ------------------------------------------------------------------
    all_knob_defs = {}
    for bank in banks:
        all_knob_defs.update(bank.knobs)
    if use_midi:
        histories = {name: [] for name in all_knob_defs}
    else:
        histories = {"motion": []}
    max_history = 600
    knob_maxes = {name: k.max_val for name, k in all_knob_defs.items()}

    bg_gw, bg_gh = disp_w - 40, disp_h - 90
    bg_current = compute_waveform_image(audio_eng.current, bg_gw, bg_gh).astype(np.float32)
    bg_target = bg_current.copy()
    bg_version = audio_eng.swap_count

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if prompt_input_active[0]:
                        if event.key == pygame.K_RETURN:
                            text = prompt_input_buffer[0].strip()
                            if text and not encoding_in_progress[0]:
                                threading.Thread(
                                    target=encode_and_apply,
                                    args=(text,),
                                    daemon=True,
                                ).start()
                            prompt_input_active[0] = False
                        elif event.key == pygame.K_ESCAPE:
                            prompt_input_active[0] = False
                            prompt_input_buffer[0] = prompt_text[0]
                        elif event.key == pygame.K_BACKSPACE:
                            prompt_input_buffer[0] = prompt_input_buffer[0][:-1]
                        elif event.unicode and event.unicode.isprintable():
                            prompt_input_buffer[0] += event.unicode
                    else:
                        if event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt
                        elif event.key == pygame.K_RETURN:
                            prompt_input_active[0] = True
                            prompt_input_buffer[0] = prompt_text[0]

            if tracker is not None:
                frame, motion = tracker.read_latest()
                if frame is None:
                    break
                frame = cv2.resize(frame, (disp_w, disp_h))
            else:
                motion = midi_knobs.get(k1_name)
                frame = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)

            with motion_lock:
                motion_val[0] = motion

            if use_midi:
                raw_hist = midi_knobs.get_all_values()
                for name in histories:
                    histories[name].append(raw_hist.get(name, 0.0))
            else:
                histories["motion"].append(motion)
            for hist in histories.values():
                if len(hist) > max_history:
                    del hist[:-max_history]

            if use_midi:
                active_bank = midi_knobs.active_bank
                knob_order = list(active_bank.knobs.keys())
                bank_name = active_bank.name
            else:
                knob_order = ["motion"]
                bank_name = None

            if audio_eng.swap_count != bg_version:
                bg_version = audio_eng.swap_count
                bg_target = compute_waveform_image(audio_eng.current, bg_gw, bg_gh).astype(np.float32)
            cv2.addWeighted(bg_current, 0.85, bg_target, 0.15, 0, dst=bg_current)

            draw_hud(frame, audio_eng, params, histories,
                     motion=motion, sde_curve_np=sde_curve_display[0],
                     knob_maxes=knob_maxes, knob_order=knob_order,
                     spec_img=bg_current.astype(np.uint8),
                     bank_name=bank_name)

            if prompt_input_active[0]:
                box_y = disp_h - 65
                cv2.rectangle(frame, (10, box_y), (disp_w - 10, box_y + 36), (30, 30, 30), -1)
                cv2.rectangle(frame, (10, box_y), (disp_w - 10, box_y + 36), (0, 200, 255), 1)
                cv2.putText(frame, prompt_input_buffer[0] + "|", (20, box_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            elif encoding_in_progress[0]:
                cv2.putText(frame, "encoding prompt...", (10, disp_h - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            time.sleep(0.016)

    except KeyboardInterrupt:
        pass
    finally:
        running[0] = False
        send_thread.join(timeout=2)
        recv_thread.join(timeout=2)
        audio_eng.stop()
        if tracker is not None:
            tracker.release()
        if midi_knobs is not None:
            midi_knobs.release()
        remote.close()
        pygame.quit()
        print(f"\n{params.get('num_gens', 0)} generations completed.")


if __name__ == "__main__":
    main()
