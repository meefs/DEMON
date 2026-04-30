"""
Real-time input-to-music using the Session graph API.

Full demo: runs the local GPU pipeline by default, or connects to a remote
GPU server with ``--remote ws://host:port``. Requires the full project
install (torch, acestep, TRT). The product team should use the thin client
under :mod:`demos.realtime_motion_graph.client` instead.

Usage:
    uv run python -m demos.realtime_motion_graph.full_demo --audio path/to/file.wav   # webcam mode
    uv run python -m demos.realtime_motion_graph.full_demo --midi --audio file.wav    # MIDI knobs
    uv run python -m demos.realtime_motion_graph.full_demo --midi --sde               # MIDI + SDE curves
    uv run python -m demos.realtime_motion_graph.full_demo --vae-window 15            # windowed decode
    uv run python -m demos.realtime_motion_graph.full_demo --lora --midi              # with LoRA (K5=strength)
    uv run python -m demos.realtime_motion_graph.full_demo --midi --display 1         # pygame window on 2nd monitor
    uv run python -m demos.realtime_motion_graph.full_demo --remote ws://host:8765    # use remote GPU

Controls:
    ESC = quit
    Pad 3 (note 38) = cycle knob bank forward
    Pad 4 (note 39) = reset all params to defaults
"""

import os
import sys
import threading
import time
from pathlib import Path

import torch
torch.set_grad_enabled(False)
torch._dynamo.config.disable = True

import cv2
import numpy as np
import pygame
import soundfile as sf

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes.types import Audio
from acestep.paths import project_root, trt_engine_path, checkpoints_dir, select_trt_engines

from .client.audio_engine import AudioEngine
from .client.hud import compute_waveform_image, draw_hud
from .client.input_sources import MidiKnobs, MotionTracker
from .client.knobs import build_banks
from .client.protocol import (
    RemoteBackend,
    SAMPLE_RATE,
    SLICE_FLAG_DELTA,
)
from .pipeline import PipelineRunner

PROJECT_ROOT = project_root()
DEFAULT_AUDIO = PROJECT_ROOT / "tests/fixtures" / "new_order_confusion_60seconds.wav"
LORA_PATHS = []


def main():
    audio_path = DEFAULT_AUDIO
    use_midi = False
    use_sde = False
    use_lora = False
    vae_window = 0.0
    args = list(sys.argv[1:])
    if "--midi" in args:
        use_midi = True
        args.remove("--midi")
    if "--sde" in args:
        use_sde = True
        args.remove("--sde")
    if "--lora" in args:
        use_lora = True
        args.remove("--lora")
    use_fast_vae = False
    if "--fast-vae" in args:
        use_fast_vae = True
        args.remove("--fast-vae")
    if "--vae-window" in args:
        idx = args.index("--vae-window")
        vae_window = float(args[idx + 1])
        del args[idx:idx + 2]
    crop_seconds = 0.0
    if "--crop" in args:
        idx = args.index("--crop")
        crop_seconds = float(args[idx + 1])
        del args[idx:idx + 2]

    depth = 8
    if "--depth" in args:
        idx = args.index("--depth")
        depth = int(args[idx + 1])
        del args[idx:idx + 2]

    display_index = 0
    if "--display" in args:
        idx = args.index("--display")
        display_index = int(args[idx + 1])
        del args[idx : idx + 2]

    window_pos = None
    if "--window-pos" in args:
        idx = args.index("--window-pos")
        xs, _, ys = args[idx + 1].partition(",")
        window_pos = (int(xs), int(ys))
        del args[idx : idx + 2]

    use_remote = None
    if "--remote" in args:
        idx = args.index("--remote")
        use_remote = args[idx + 1]
        del args[idx : idx + 2]

    initial_prompt = "deathstep, heavy bass, dark atmosphere"
    if "--prompt" in args:
        idx = args.index("--prompt")
        initial_prompt = args[idx + 1]
        del args[idx : idx + 2]

    if "--audio" in args:
        idx = args.index("--audio")
        audio_path = Path(args[idx + 1])
        del args[idx : idx + 2]
    elif args:
        audio_path = Path(args[0])

    k1_name = "sde_amp" if use_sde else "denoise"

    print("=" * 60)
    print("Real-Time Motion-to-Music (FULL DEMO)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load source audio (shared)
    # ------------------------------------------------------------------
    print("[Setup] Loading source audio...")
    data, sr = sf.read(str(audio_path), dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform[:2, :int(60.0 * SAMPLE_RATE)]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]

    # ------------------------------------------------------------------
    # Setup: remote OR local
    # ------------------------------------------------------------------
    remote = None
    session = None
    stream = None
    source = None
    lora_ids = []
    engine_obj = None
    detected_bpm = 120

    if use_remote:
        remote = RemoteBackend(use_remote, waveform, {
            "sde": use_sde, "lora": use_lora, "depth": depth,
            "vae_window": vae_window if vae_window > 0 else 3.0,
            "crop": crop_seconds, "steps": 8,
            "prompt": initial_prompt,
            "lora_paths": [str(p) for p in LORA_PATHS] if use_lora else None,
            "fast_vae": use_fast_vae,
        })
        src_np = remote.initial_buffer
        if crop_seconds > 0:
            src_np = src_np[:int(crop_seconds * SAMPLE_RATE)]
    else:
        # Audio is clamped to 60s above (waveform[:2, :int(60.0 * SAMPLE_RATE)]),
        # so 60s engines are the right default. Mirrors the modern server.py path.
        audio_duration_s = waveform.shape[1] / SAMPLE_RATE
        trt_engines = select_trt_engines(duration_s=audio_duration_s)
        if use_fast_vae:
            fast_name = "dreamvae_decode_fp16_60s" if audio_duration_s <= 60.0 else "dreamvae_decode_fp16_240s"
            if Path(str(trt_engine_path(fast_name))).exists():
                trt_engines["vae_decode"] = str(trt_engine_path(fast_name))
            else:
                print(f"[Setup] WARNING: {fast_name} engine missing, using {Path(trt_engines['vae_decode']).stem}")
                use_fast_vae = False
        print("[Setup] Loading model...")
        t0 = time.time()
        session = Session(
            project_root=str(checkpoints_dir()),
            decoder_backend="tensorrt",
            vae_backend="tensorrt",
            trt_engines=trt_engines,
            vae_window=vae_window,
        )
        print(f"  Model loaded in {time.time()-t0:.1f}s")

        if use_lora:
            engine_obj = session.handler._diffusion_engine
            if engine_obj is not None and engine_obj.trt_lora_available:
                t0 = time.time()
                for lp in LORA_PATHS:
                    if not Path(lp).exists():
                        print(f"[Setup] WARNING: LoRA path missing: {lp}")
                        continue
                    print(f"[Setup] Applying LoRA: {Path(lp).name}")
                    lid = engine_obj.apply_trt_lora(lp, strength=0.0)
                    lora_ids.append(lid)
                print(f"  {len(lora_ids)} LoRA(s) applied in {time.time()-t0:.1f}s")
                if not lora_ids:
                    use_lora = False
            else:
                print("[Setup] WARNING: --lora requested but TRT LoRA refit not available")
                use_lora = False

        audio_in = Audio(waveform=waveform, sample_rate=SAMPLE_RATE)
        print("[Setup] Detecting BPM...")
        import librosa
        mono_np = waveform.mean(dim=0).numpy()
        detected_bpm, _ = librosa.beat.beat_track(y=mono_np, sr=SAMPLE_RATE)
        detected_bpm = int(round(float(detected_bpm)))
        print(f"  BPM: {detected_bpm}")

        print("[Setup] Preparing source...")
        source = session.prepare_source(audio_in)

        print("[Setup] Text encode...")
        t0 = time.time()
        conditioning = session.encode_text(
            tags=initial_prompt,
            instruction=TASK_INSTRUCTIONS["cover"],
            refer_latent=source.latent,
            bpm=detected_bpm, duration=60.0, key="G# minor",
        )
        print(f"  Prompt: \"{initial_prompt}\" ({time.time()-t0:.1f}s)")

        print("[Setup] Creating stream...")
        stream = session.stream(
            source=source, conditioning=conditioning,
            steps=8, shift=3.0, pipeline_depth=depth,
        )
        print("[Setup] Stream handle ready (pipeline built on first tick)")

        src_np = waveform.numpy().T
        if crop_seconds > 0:
            crop_samples = int(crop_seconds * SAMPLE_RATE)
            src_np = src_np[:crop_samples]
            print(f"[Audio] Cropping playback to {crop_seconds}s (discarding tail)")

    # ------------------------------------------------------------------
    # Knob banks (built after setup so lora_ids count is known)
    # ------------------------------------------------------------------
    lora_count = remote.lora_count if use_remote and use_lora else len(lora_ids)
    banks = build_banks(use_sde, lora=lora_count)

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
    pygame.display.set_caption(f"Real-Time {mode_str} (Graph)")

    print(f"\n  Mode: {mode_str}")
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
    skip_threshold = 1e-3

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
            if use_remote:
                remote.send_prompt(text)
                prompt_text[0] = text
            else:
                cond = session.encode_text(
                    tags=text,
                    instruction=TASK_INSTRUCTIONS["cover"],
                    refer_latent=source.latent,
                    bpm=detected_bpm, duration=60.0, key="G# minor",
                )
                stream.conditioning = cond
                prompt_text[0] = text
                print(f"  Prompt applied: \"{text}\"")
        finally:
            encoding_in_progress[0] = False

    sde_curve_display = [None]

    # ------------------------------------------------------------------
    # Pipeline / remote threads
    # ------------------------------------------------------------------
    recv_thread = None
    if use_remote:
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

        pipe_thread = threading.Thread(target=remote_send_loop, daemon=True)
        recv_thread = threading.Thread(target=remote_recv_loop, daemon=True)
        recv_thread.start()
    else:
        runner = PipelineRunner(
            session, stream, audio_eng,
            use_midi=use_midi, use_sde=use_sde, use_lora=use_lora,
            midi_knobs=midi_knobs, lora_ids=lora_ids,
            engine_obj=engine_obj,
            vae_window=vae_window, crop_seconds=crop_seconds,
            k1_name=k1_name, seed=SEED, skip_threshold=skip_threshold,
            sde_curve_display=sde_curve_display, params=params,
            prompt_text=prompt_text, running=running,
            motion_val=motion_val, motion_lock=motion_lock,
        )
        pipe_thread = threading.Thread(target=runner.run, daemon=True)
    pipe_thread.start()

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
        pipe_thread.join(timeout=2)
        if recv_thread is not None:
            recv_thread.join(timeout=2)
        audio_eng.stop()
        if tracker is not None:
            tracker.release()
        if midi_knobs is not None:
            midi_knobs.release()
        if remote is not None:
            remote.close()
        pygame.quit()
        print(f"\n{params.get('num_gens', 0)} generations completed.")


if __name__ == "__main__":
    main()
