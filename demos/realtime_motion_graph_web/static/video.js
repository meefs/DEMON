// Video layer: crossfade player with beat-grid sync and marker detection.
// Ported from the glinside installation prototype.
//
// Beat markers: if a video encodes a bright-green pixel in its top-left
// corner on downbeat frames, the marker detector will lock the video
// playback to the audio beat grid automatically.  Without markers the
// layer falls back to periodic drift correction.

export class VideoLayer {
  constructor({ videoA, videoB, bpm = 134, crossfadeDuration = 1.5, useMarkers = false }) {
    this.videoA = videoA;
    this.videoB = videoB;
    this.activeVideo = videoA;
    this.inactiveVideo = videoB;
    this.hasVideo = false;
    this.videos = [];
    this.currentIndex = 0;

    this.bpm = bpm;
    this.beatDuration = 60 / bpm;
    this.crossfadeDuration = crossfadeDuration;

    // Beat marker detection (opt-in: only for videos with encoded green
    // pixels in the top-left corner on downbeat frames)
    this.useMarkers = useMarkers;
    this._markerCanvas = document.createElement("canvas");
    this._markerCanvas.width = 4;
    this._markerCanvas.height = 4;
    this._markerCtx = this._markerCanvas.getContext("2d", { willReadFrequently: true });
    this._lastMarkerDetected = false;
    this._beatOffset = null;
    this._markerCount = 0;

    // Audio position source (set via setAudioSource)
    this._getAudioPos = () => 0;
    this._hasAudio = false;

    // Start marker detection loop only if enabled
    this._markerRaf = null;
    if (this.useMarkers) {
      this._markerRaf = requestAnimationFrame(() => this._markerLoop());
    }

    // Drift correction (1 Hz)
    this._driftInterval = setInterval(() => this._correctDrift(), 1000);
  }

  setBpm(bpm) {
    this.bpm = bpm;
    this.beatDuration = 60 / bpm;
  }

  setAudioSource(getPos, hasAudio) {
    this._getAudioPos = getPos;
    this._hasAudio = hasAudio;
  }

  setVideos(videos) {
    this.videos = videos;
    this.currentIndex = 0;
  }

  play(filename, transition = "crossfade") {
    this._resetMarkerState();

    if (transition === "crossfade" && this.hasVideo) {
      const inactive = this.inactiveVideo;
      const active = this.activeVideo;
      inactive.src = `videos/${filename}`;
      inactive.loop = true;
      inactive.muted = true;
      const onReady = () => {
        inactive.removeEventListener("canplay", onReady);
        this._syncToBeat(inactive);
        inactive.play().catch(() => {});
        inactive.style.opacity = "1";
        active.style.opacity = "0";
        setTimeout(() => {
          active.pause();
          active.removeAttribute("src");
          active.load();
          const tmp = this.activeVideo;
          this.activeVideo = this.inactiveVideo;
          this.inactiveVideo = tmp;
        }, this.crossfadeDuration * 1000 + 100);
      };
      inactive.addEventListener("canplay", onReady);
      inactive.load();
    } else {
      const el = this.activeVideo;
      el.src = `videos/${filename}`;
      el.loop = true;
      el.muted = true;
      el.style.opacity = "1";
      const onReady = () => {
        el.removeEventListener("canplay", onReady);
        this.hasVideo = true;
        this._syncToBeat(el);
        el.play().catch(() => {});
      };
      el.addEventListener("canplay", onReady);
      el.load();
    }
  }

  next() {
    if (this.videos.length === 0) return;
    this.currentIndex = (this.currentIndex + 1) % this.videos.length;
    this.play(this.videos[this.currentIndex]);
  }

  previous() {
    if (this.videos.length === 0) return;
    this.currentIndex = (this.currentIndex - 1 + this.videos.length) % this.videos.length;
    this.play(this.videos[this.currentIndex]);
  }

  // -- internal --------------------------------------------------------

  _resetMarkerState() {
    this._lastMarkerDetected = false;
    this._beatOffset = null;
    this._markerCount = 0;
  }

  _markerLoop() {
    if (this.hasVideo && this.activeVideo && !this.activeVideo.paused) {
      this._sampleMarker(this.activeVideo);
    }
    this._markerRaf = requestAnimationFrame(() => this._markerLoop());
  }

  _sampleMarker(el) {
    if (!el || el.paused || !el.videoWidth) return;
    this._markerCtx.drawImage(el, 0, 0, 8, 8, 0, 0, 4, 4);
    const px = this._markerCtx.getImageData(1, 1, 1, 1).data;
    const detected = px[1] > 50 && px[1] > px[0] * 3 && px[1] > px[2] * 3;
    if (detected && !this._lastMarkerDetected) this._onVideoDownbeat(el);
    this._lastMarkerDetected = detected;
  }

  _onVideoDownbeat(el) {
    if (!this._hasAudio) return;
    const audioPos = this._getAudioPos();
    const phase = (audioPos % this.beatDuration) / this.beatDuration;
    const near = phase < 0.2 || phase > 0.8;
    this._markerCount++;
    if (near) {
      this._beatOffset = el.currentTime -
        (Math.round(audioPos / this.beatDuration) * this.beatDuration % (el.duration || 1));
    } else {
      const nearest = Math.round(audioPos / this.beatDuration) * this.beatDuration;
      const vd = el.duration;
      const target = this._beatOffset !== null
        ? (nearest % vd) + this._beatOffset
        : nearest % vd;
      el.currentTime = ((target % vd) + vd) % vd;
    }
  }

  _syncToBeat(el) {
    if (!el.duration || el.duration === Infinity) return;
    if (this._hasAudio) {
      const ap = this._getAudioPos();
      el.currentTime = Math.floor(ap / this.beatDuration) * this.beatDuration % el.duration;
    } else {
      el.currentTime = 0;
    }
  }

  _correctDrift() {
    if (!this._hasAudio || !this.hasVideo) return;
    if (!this.activeVideo.duration || this.activeVideo.paused) return;
    if (this._markerCount > 0) return;
    const ap = this._getAudioPos();
    const vd = this.activeVideo.duration;
    let drift = (ap % vd) - this.activeVideo.currentTime;
    if (drift > vd / 2) drift -= vd;
    if (drift < -vd / 2) drift += vd;
    if (Math.abs(drift) > this.beatDuration * 0.4) {
      this.activeVideo.currentTime =
        Math.floor(ap / this.beatDuration) * this.beatDuration % vd;
    }
  }

  destroy() {
    cancelAnimationFrame(this._markerRaf);
    clearInterval(this._driftInterval);
    this.activeVideo.pause();
    this.inactiveVideo.pause();
  }
}
