"""HUD rendering: waveform background, knob trails, playhead, stats."""

import cv2
import numpy as np

# Colors for graph lines (auto-looked-up by parameter name)
GRAPH_COLORS = {
    "denoise":        (0, 255, 0),
    "sde_amp":        (0, 255, 0),
    "seed":           (255, 180, 0),
    "feedback":       (0, 200, 255),
    "shift":          (180, 0, 255),
    "periodicity":    (60, 40, 60),
    "lora_str_1":     (255, 50, 200),
    "lora_str_2":     (200, 50, 255),
    "hint_strength":  (255, 200, 50),
    "noise_share":    (120, 120, 255),
    "ode_noise":      (200, 200, 200),
    "prompt_blend":   (255, 140, 0),
    "x0_target":      (0, 255, 180),
    "motion":         (0, 255, 0),
    # Channel groups
    "ch_g0":          (255, 80, 80),
    "ch_g1":          (255, 160, 60),
    "ch_g2":          (255, 220, 40),
    "ch_g3":          (180, 255, 60),
    "ch_g4":          (60, 255, 140),
    "ch_g5":          (40, 220, 255),
    "ch_g6":          (100, 140, 255),
    "ch_g7":          (200, 120, 255),
    # Keystone channels
    "ch13":           (255, 100, 100),
    "ch14":           (255, 180, 80),
    "ch19":           (220, 255, 80),
    "ch23":           (80, 255, 180),
    "ch29":           (80, 180, 255),
    "ch56":           (180, 80, 255),
}


def compute_waveform_image(audio_data, width, height):
    """Render a filled waveform envelope with a center-to-edge color gradient.

    Teal at the center line blending to deep purple at the waveform edges.
    Returns a BGR uint8 image of shape (height, width, 3).
    """
    mono = audio_data.mean(axis=1) if audio_data.ndim > 1 else audio_data
    n_samples = len(mono)
    chunk = max(1, n_samples // width)
    n_cols = min(width, n_samples // chunk)
    if n_cols < 2:
        return np.zeros((height, width, 3), dtype=np.uint8)

    peaks = np.abs(mono[:n_cols * chunk].reshape(n_cols, chunk)).max(axis=1)
    pmax = peaks.max()
    if pmax > 0:
        peaks /= pmax

    cy = height // 2
    amps = (peaks * cy * 0.9).astype(np.int32)

    xs = np.linspace(0, width - 1, n_cols).astype(np.int32)
    top = np.column_stack([xs, cy - amps])
    bot = np.column_stack([xs[::-1], cy + amps[::-1]])
    poly = np.vstack([top, bot]).astype(np.int32)

    # Sharp mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)

    # Vertical color gradient: teal at center -> purple at edges
    center_color = np.array([110, 85, 22], dtype=np.float32)   # BGR teal
    edge_color = np.array([35, 10, 45], dtype=np.float32)      # BGR purple
    dist = np.abs(np.arange(height, dtype=np.float32) - cy) / max(cy, 1)
    t = dist.reshape(-1, 1, 1)
    gradient = center_color * (1.0 - t) + edge_color * t       # (H, 1, 3)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    inside = mask > 0
    img[inside] = gradient.astype(np.uint8)[np.where(inside)[0], 0, :]

    # Light anti-alias only (sigma ~1px)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def draw_hud(frame, audio_eng, params, histories, motion=0.0, sde_curve_np=None,
             knob_maxes=None, knob_order=None, spec_img=None, bank_name=None):
    """Draw heads-up display. All active parameters render automatically."""
    h, w = frame.shape[:2]

    gx, gy = 10, 65
    gw, gh = w - 40, h - 90
    frac = audio_eng.playback_position / audio_eng.duration if audio_eng.duration else 0
    cx = gx + int(frac * gw)

    # Waveform background
    if spec_img is not None:
        frame[gy:gy + gh, gx:gx + gw] = spec_img

    # --- Glow layer: collect all glowing elements, blur once, blend ---
    glow = np.zeros((h, w, 3), dtype=np.uint8)

    # Audio-reactive playhead: brightness tracks current RMS
    pos = audio_eng.position
    buf = audio_eng.current
    n_buf = len(buf)
    rms_win = 4096
    s = max(0, pos - rms_win // 2)
    e = min(n_buf, s + rms_win)
    rms = np.sqrt(np.mean(buf[s:e].astype(np.float32) ** 2)) if e > s else 0.0
    brightness = min(1.0, rms * 5)
    ph_color = (int(150 * brightness), int(180 * brightness), int(220 * brightness))
    cv2.line(glow, (cx, gy), (cx, gy + gh), ph_color, 5)

    # Collect graph line segments (history trails + SDE curve)
    all_segments = []

    if histories:
        if knob_maxes is None:
            knob_maxes = {}
        draw_order = sorted(histories.keys(), key=lambda n: 0 if n == "periodicity" else 1)
        trail_w = max(1, cx - gx)
        for name in draw_order:
            hist = histories[name]
            if len(hist) < 2:
                continue
            color = GRAPH_COLORS.get(name, (255, 255, 255))
            max_val = knob_maxes.get(name, 1.0)
            n = min(len(hist), trail_w)
            pts = []
            for i in range(n):
                x = cx - n + 1 + i
                v = hist[-(n - i)] / max_val if max_val > 0 else 0.0
                y = gy + gh - int(min(v, 1.0) * gh)
                pts.append((x, y))
            all_segments.append((pts, color))

    if sde_curve_np is not None and len(sde_curve_np) > 1:
        sde_color = (100, 255, 100)
        n_pts = min(len(sde_curve_np), gw)
        step = max(1, len(sde_curve_np) // n_pts)
        pts = []
        for i in range(n_pts):
            x = gx + int(i * gw / n_pts)
            y = gy + gh - int(float(sde_curve_np[i * step]) * gh)
            pts.append((x, y))
        all_segments.append((pts, sde_color))

    # Draw thick lines on glow layer
    for pts, color in all_segments:
        for a, b in zip(pts, pts[1:]):
            cv2.line(glow, a, b, color, 3)

    # Single blur + additive blend
    cv2.GaussianBlur(glow, (0, 0), sigmaX=6, sigmaY=6, dst=glow)
    cv2.add(frame, glow, dst=frame)

    # Sharp lines on top of glow
    cv2.line(frame, (cx, gy), (cx, gy + gh), (255, 255, 255), 1)
    for pts, color in all_segments:
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, color, 1)

    # Playback bar (bottom)
    cv2.rectangle(frame, (0, h - 6), (int(w * frac), h), (0, 200, 255), -1)
    cv2.putText(frame, f"{audio_eng.playback_position:.1f}s / {audio_eng.duration:.1f}s",
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Motion bar (right side)
    bar_h = int(motion * 200)
    cv2.rectangle(frame, (w - 20, 80), (w - 5, 280), (40, 40, 40), -1)
    cv2.rectangle(frame, (w - 20, 280 - bar_h), (w - 5, 280), (0, 255, 0), -1)

    # Stats line (top)
    bank_str = f"  [{bank_name}]" if bank_name else ""
    cv2.putText(frame,
                f"gen #{params.get('num_gens', 0)}  tick={params.get('tick_ms', 0):.0f}ms  dec={params.get('dec_ms', 0):.0f}ms{bank_str}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Active prompt (top right)
    prompt = params.get("_prompt")
    if prompt:
        (tw, _), _ = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, prompt, (w - tw - 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    # Parameter lines in knob order: two rows of 4
    timing_keys = {"num_gens", "tick_ms", "dec_ms", "_prompt"}
    param_keys = knob_order if knob_order else [k for k in params if k not in timing_keys]
    parts = []
    for k in param_keys:
        v = params.get(k)
        if v is None:
            continue
        if isinstance(v, int):
            parts.append(f"{k}={v}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
    row1 = "  ".join(parts[:4])
    row2 = "  ".join(parts[4:])
    if row1:
        cv2.putText(frame, row1, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    if row2:
        cv2.putText(frame, row2, (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
