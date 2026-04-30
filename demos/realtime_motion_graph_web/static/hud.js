// Scrolling waveform HUD for the PWA.
// Playhead is fixed at horizontal center; waveform scrolls past it.

export class HUD {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this._peaks = null;    // Float32Array of normalised peak values per column
    this._nFrames = 0;
    this._channels = 2;
    this._resizeObserver = new ResizeObserver(() => this._resize());
    this._resizeObserver.observe(canvas);
    this._resize();
  }

  _resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width || window.innerWidth || 800;
    const h = rect.height || window.innerHeight || 600;
    this.canvas.width = Math.floor(w * dpr);
    this.canvas.height = Math.floor(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w;
    this.h = h;
    this._peaksDirty = true;
  }

  resize() { this._resize(); }

  // Recompute peak envelope from the full interleaved buffer.
  updateWaveform(interleaved, channels) {
    if (!interleaved || !interleaved.length) return;
    this._channels = channels;
    this._nFrames = (interleaved.length / channels) | 0;

    // Build one peak per ~200 frames (~4ms at 48kHz) — enough resolution
    // for smooth scrolling at any canvas width.
    const chunk = 200;
    const nPeaks = Math.floor(this._nFrames / chunk);
    if (nPeaks < 2) return;

    const peaks = new Float32Array(nPeaks);
    let pmax = 0;
    for (let col = 0; col < nPeaks; col++) {
      let m = 0;
      const base = col * chunk * channels;
      for (let i = 0; i < chunk; i++) {
        let sum = 0;
        for (let c = 0; c < channels; c++) sum += interleaved[base + i * channels + c];
        const v = Math.abs(sum / channels);
        if (v > m) m = v;
      }
      peaks[col] = m;
      if (m > pmax) pmax = m;
    }
    if (pmax > 0) for (let i = 0; i < nPeaks; i++) peaks[i] /= pmax;
    this._peaks = peaks;
  }

  // Draw one frame. playbackFrac is 0..1 through the track.
  draw(playbackFrac, { transparentBg = false } = {}) {
    const ctx = this.ctx;
    const w = this.w, h = this.h;
    if (transparentBg) {
      ctx.clearRect(0, 0, w, h);
    } else {
      ctx.fillStyle = "#06060c";
      ctx.fillRect(0, 0, w, h);
    }

    if (!this._peaks || this._peaks.length < 2) return;

    const peaks = this._peaks;
    const nPeaks = peaks.length;
    const centerX = Math.floor(w / 2);
    const cy = Math.floor(h / 2);
    const maxAmp = cy * 0.75;

    // How many peaks fit on screen (1 peak per 2px for density).
    const pxPerPeak = 2;
    const peaksOnScreen = Math.floor(w / pxPerPeak);

    // Current peak index corresponding to playback position.
    const currentPeak = playbackFrac * (nPeaks - 1);

    // Range of peaks visible: center maps to currentPeak.
    const halfScreen = peaksOnScreen / 2;
    const startPeak = currentPeak - halfScreen;

    // Draw waveform bars.
    for (let i = 0; i < peaksOnScreen; i++) {
      const peakIdx = Math.floor(startPeak + i);
      if (peakIdx < 0 || peakIdx >= nPeaks) continue;

      const x = i * pxPerPeak;
      const amp = peaks[peakIdx] * maxAmp;
      const isPlayed = x < centerX;

      // Gradient: played = dimmer teal, upcoming = brighter teal.
      if (isPlayed) {
        const fade = 0.25 + 0.35 * (x / centerX);
        ctx.fillStyle = `rgba(22, 85, 110, ${fade})`;
      } else {
        const fade = 0.6 + 0.3 * (1 - (x - centerX) / (w - centerX));
        ctx.fillStyle = `rgba(40, 140, 180, ${fade})`;
      }

      ctx.fillRect(x, cy - amp, pxPerPeak - 1, amp * 2);
    }

    // Playhead line at center.
    ctx.save();
    ctx.shadowBlur = 16;
    ctx.shadowColor = "rgba(90, 184, 138, 0.8)";
    ctx.strokeStyle = "rgba(90, 184, 138, 0.9)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, h);
    ctx.stroke();
    ctx.restore();

    // Glow around playhead.
    const glow = ctx.createLinearGradient(centerX - 40, 0, centerX + 40, 0);
    glow.addColorStop(0, "rgba(90, 184, 138, 0)");
    glow.addColorStop(0.5, "rgba(90, 184, 138, 0.06)");
    glow.addColorStop(1, "rgba(90, 184, 138, 0)");
    ctx.fillStyle = glow;
    ctx.fillRect(centerX - 40, 0, 80, h);
  }
}
