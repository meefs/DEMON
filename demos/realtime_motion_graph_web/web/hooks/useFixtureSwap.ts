"use client";

import { useEffect, useRef } from "react";

import { loadFixtureAudio } from "@/engine/audio/loadFixture";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// In-place fixture swap. Mirrors swapToFixture() in DEMON's app.js: when the
// user picks a different fixture mid-session, the server keeps the model
// loaded and re-encodes the new source; the worklet crossfades the new
// buffer in over 50 ms. We surface "Decoding ..." then "Swapping to ..."
// in the status bar so the user knows the swap is in flight.
//
// Falls back to a full session restart on swap_failed (e.g. server in a
// state where it can't accept a new source). The full restart is delegated
// back to useStartSession via the same fixture name.

export function useFixtureSwap() {
  // Skip the very first fixture write (which fires when the catalog populates
  // and writes the default name into the store).
  const lastSwappedTo = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const run = async (name: string) => {
      if (cancelled) return;
      const session = useSessionStore.getState();
      if (session.status !== "ready" || !session.remote || !session.player) {
        return; // No live session yet; the next Play will pick the new fixture.
      }
      if (lastSwappedTo.current === name) return;

      const { setStatus } = useSessionStore.getState();
      setStatus("ready", `Loading ${name}…`);

      let interleaved: Float32Array;
      let channels: number;
      try {
        const decoded = await loadFixtureAudio(name);
        interleaved = decoded.interleaved;
        channels = decoded.channels;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setStatus("ready", `Load failed: ${msg}`);
        return;
      }
      if (cancelled) return;

      setStatus("ready", `Swapping to ${name}…`);

      const remote = session.remote;
      const ok = await new Promise<boolean>((resolve) => {
        const onReady = (e: Event) => {
          remote.removeEventListener("swap_ready", onReady);
          remote.removeEventListener("swap_failed", onFail);
          const detail = (e as CustomEvent<{
            interleaved: Float32Array;
            channels: number;
            key?: string;
          }>).detail;
          session.player?.swap(detail.interleaved, detail.channels);
          // Always update detectedKey so the advanced strip's
          // "Detected: …" readout reflects what the CNN actually
          // inferred — even when the user overrode it below.
          const perf = usePerformanceStore.getState();
          if (detail.key) {
            perf.setDetected(perf.detectedBpm, detail.key);
          }
          // One-shot override (set by AlmostReadyDialog's "Set
          // manually" mode) wins over the server's detection and is
          // cleared after use. Tell the server too so the model hint
          // matches what the UI shows.
          if (perf.pendingKeyOverride) {
            const override = perf.pendingKeyOverride;
            perf.setKey(override);
            perf.setPendingKeyOverride(null);
            remote.sendPrompt(perf.promptA, override);
          } else if (detail.key) {
            perf.setKey(detail.key);
          }
          resolve(true);
        };
        const onFail = (e: Event) => {
          remote.removeEventListener("swap_ready", onReady);
          remote.removeEventListener("swap_failed", onFail);
          console.warn("[fixture-swap] server swap_failed:", (e as CustomEvent).detail);
          resolve(false);
        };
        remote.addEventListener("swap_ready", onReady);
        remote.addEventListener("swap_failed", onFail);

        const perf = usePerformanceStore.getState();
        const sent = remote.sendSwapSource(
          interleaved,
          channels,
          perf.promptA,
          perf.activeKey,
          name,
        );
        if (!sent) {
          remote.removeEventListener("swap_ready", onReady);
          remote.removeEventListener("swap_failed", onFail);
          resolve(false);
        }
      });

      if (cancelled) return;
      if (!ok) {
        setStatus("ready", "Swap failed — please try again");
        return;
      }
      lastSwappedTo.current = name;
      // Each new track re-enters the "hear source first" gate: snap the
      // engine value to 0 instantly (user hears the source from frame
      // 1) and play a visual-only glide on the ribbon from its prior
      // position down to 0 as a "this is a slider" hint. Clear
      // remixStarted so the top-edge ribbon shows "drag to start"
      // again. Side-rail hints stay suppressed until the user drags
      // the top ribbon up.
      const perfState = usePerformanceStore.getState();
      const prevDenoise = perfState.sliderTargets["denoise"] ?? 0;
      perfState.setSliderDirect("denoise", 0);
      perfState.animateSliderDisplayFrom("denoise", prevDenoise, 700);
      perfState.setRemixStarted(false);
      setStatus("ready", "Playing");
    };

    const unsub = usePerformanceStore.subscribe((s, prev) => {
      if (s.fixture !== prev.fixture && s.fixture) {
        void run(s.fixture);
      }
    });

    // Seed lastSwappedTo with the current fixture so the initial population
    // (catalog → default fixture write) doesn't trigger a no-op swap.
    lastSwappedTo.current = usePerformanceStore.getState().fixture;

    return () => {
      cancelled = true;
      unsub();
    };
  }, []);
}
