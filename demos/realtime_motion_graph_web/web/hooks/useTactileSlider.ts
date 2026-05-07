"use client";

import { useEffect, useRef } from "react";

import { usePerformanceStore } from "@/store/usePerformanceStore";

// Lightweight per-slider tactile augmentation: feature-detected haptics on
// landmark crossings (0, 0.5, 1.0) and a long-press reset gesture.
//
// Touch the rail/slider for >= LONG_PRESS_MS without moving more than
// LONG_PRESS_MOVE_PX and we treat it as a "snap back" gesture: the param
// is reset to its default. iOS Safari ignores `navigator.vibrate` so the
// haptic is a no-op there but the long-press still works.
//
// We do NOT take over the pointer pipeline — callers keep their own drag
// handlers. We just attach a few extra listeners to the same element and
// react to the same events.

const LONG_PRESS_MS = 500;
const LONG_PRESS_MOVE_PX = 8;
const HAPTIC_CROSSINGS = [0, 0.5, 1.0] as const;
const HAPTIC_TOL = 0.04;

function vibrate(ms: number): void {
  if (typeof navigator === "undefined") return;
  const v = (navigator as Navigator & { vibrate?: (n: number) => boolean })
    .vibrate;
  if (typeof v === "function") {
    try {
      v.call(navigator, ms);
    } catch {}
  }
}

interface Options {
  /** sliderTargets key (e.g. "denoise", "lora_blend"). */
  param: string;
  /** Slider max — used to normalize value into 0..1 for crossing detection. */
  max: number;
  /** Element to attach long-press detection on. */
  ref: React.MutableRefObject<HTMLElement | null>;
}

export function useTactileSlider({ param, max, ref }: Options): void {
  // Track previous fraction so we only fire haptic at the moment a crossing
  // happens, not on every redraw at that position.
  const prevFrac = useRef<number | null>(null);

  // Subscribe directly to the store: cheaper than re-rendering the parent
  // for every value change, and lets us fire vibrate() outside React's
  // commit phase.
  useEffect(() => {
    const fire = () => {
      const v = usePerformanceStore.getState().sliderTargets[param] ?? 0;
      const frac = max > 0 ? v / max : 0;
      const prev = prevFrac.current;
      prevFrac.current = frac;
      if (prev === null) return;
      for (const cross of HAPTIC_CROSSINGS) {
        const wasNear = Math.abs(prev - cross) <= HAPTIC_TOL;
        const isNear = Math.abs(frac - cross) <= HAPTIC_TOL;
        if (!wasNear && isNear) {
          vibrate(8);
          break;
        }
      }
    };
    return usePerformanceStore.subscribe(fire);
  }, [param, max]);

  // Long-press to reset.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let startX = 0;
    let startY = 0;

    const cancel = () => {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
    };

    const onDown = (e: PointerEvent) => {
      startX = e.clientX;
      startY = e.clientY;
      cancel();
      timer = setTimeout(() => {
        usePerformanceStore.getState().resetSlider(param);
        // Stronger feedback for the destructive-feeling reset.
        vibrate(20);
      }, LONG_PRESS_MS);
    };
    const onMove = (e: PointerEvent) => {
      if (!timer) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      if (dx * dx + dy * dy > LONG_PRESS_MOVE_PX * LONG_PRESS_MOVE_PX) {
        cancel();
      }
    };
    const onUp = () => {
      cancel();
    };

    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onUp);
    return () => {
      cancel();
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onUp);
    };
  }, [param, ref]);
}
