"use client";

import { useEffect, useState } from "react";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import {
  DCW_MODES,
  DCW_WAVELETS,
  type DcwMode,
  type DcwWavelet,
} from "@/types/engine";

// Operator-editable startup config. Mirrors the static app's
// static/config.json (now lost in the React port). Lives at
// web/public/config.json so an installer can edit and refresh without a
// rebuild — the Next.js dev/prod server serves it as-is, no caching.
//
// Boot order:
//   1. RTMGBoot (module load) calls loadConfig() → applyConfig().
//   2. applyConfig() pushes values into the relevant zustand stores
//      (perf, lora) and notifies non-store subscribers (effects renderer,
//      PerformanceShell's useConfig()).
//   3. Stores already initialize with hardcoded defaults that match
//      DEFAULT_CONFIG below, so first paint is correct even if the fetch
//      is still in flight.

export interface RtmgConfigEngine {
  sde: boolean;
  lora: boolean;
  depth: number;
  vae_window: number;
  crop: number;
  steps: number;
  fast_vae: boolean;
  key: string;
  /** Filename stems to auto-enable on first catalog load. Empty falls
   * back to the count-rule in useLoraStore (first two from the sorted
   * catalog, with name-fallback per slot). */
  enabled_loras: string[];
}

export interface RtmgConfigPrompts {
  a: string;
  b: string;
  blend: number;
}

export interface RtmgConfigEffects {
  parallax_strength: number;
  bloom_on_kick: number;
  bloom_threshold: number;
  warp_strength: number;
}

/** controls.* — initial slider values plus the DCW companion controls
 * (enabled / mode / wavelet) and lora_default_strength. Numeric entries
 * seed sliderValues + sliderTargets; the named DCW entries drive the
 * non-numeric DCW state. Unknown keys are ignored. */
export type RtmgConfigControls = Record<string, number | boolean | string>;

export interface RtmgConfig {
  engine: RtmgConfigEngine;
  prompts: RtmgConfigPrompts;
  controls: RtmgConfigControls;
  seed: number;
  effects: RtmgConfigEffects;
  reset_seconds: number;
}

export const DEFAULT_CONFIG: RtmgConfig = {
  engine: {
    sde: false,
    lora: true,
    depth: 4,
    vae_window: 6,
    crop: 0,
    steps: 8,
    fast_vae: false,
    key: "G# minor",
    enabled_loras: [],
  },
  prompts: {
    a: "heavy dubstep, deathstep, afxdump, growl heavy bass distortion",
    b: "daft punk style, beautiful, four to the floor, angelic",
    blend: 0.4,
  },
  controls: {
    denoise: 0.7,
    hint_strength: 1.4,
    feedback: 0.0,
    shift: 0.5,
    noise_share: 0.0,
    ode_noise: 0.0,
    ch_g0: 1.0,
    ch_g1: 1.0,
    ch_g2: 1.0,
    ch_g3: 1.0,
    ch_g4: 1.0,
    ch_g5: 1.0,
    ch_g6: 1.0,
    ch_g7: 1.0,
    ch13: 1.0,
    ch14: 1.0,
    ch19: 1.0,
    ch23: 1.0,
    ch29: 1.0,
    ch56: 1.0,
    dcw_scaler: 0.05,
    dcw_high_scaler: 0.02,
    dcw_enabled: true,
    dcw_mode: "double",
    dcw_wavelet: "haar",
    lora_default_strength: 1.4,
  },
  seed: 0,
  effects: {
    parallax_strength: 0.4,
    bloom_on_kick: 0.3,
    bloom_threshold: 0.15,
    warp_strength: 0.4,
  },
  reset_seconds: 0,
};

let _activeConfig: RtmgConfig = DEFAULT_CONFIG;
const listeners = new Set<(c: RtmgConfig) => void>();

/** Snapshot of the active config. Read at point-of-use by code paths
 * that don't need re-render reactivity (e.g. useStartSession.buildConfig
 * runs once per Play click). For reactive reads, use useConfig(). */
export function getConfig(): RtmgConfig {
  return _activeConfig;
}

/** Subscribe to applyConfig() calls. Returns an unsubscribe. Used by
 * non-store consumers (the effects renderer in useRenderLoop) that need
 * to re-apply settings when the config arrives async after their mount. */
export function subscribeConfig(fn: (c: RtmgConfig) => void): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

/** React hook variant — subscribes the calling component to config
 * changes so it re-renders when applyConfig() fires. */
export function useConfig(): RtmgConfig {
  const [c, setC] = useState(_activeConfig);
  useEffect(() => subscribeConfig(setC), []);
  return c;
}

function mergeConfig(
  base: RtmgConfig,
  override: Partial<RtmgConfig> | null | undefined,
): RtmgConfig {
  if (!override) return base;
  return {
    engine: { ...base.engine, ...(override.engine ?? {}) },
    prompts: { ...base.prompts, ...(override.prompts ?? {}) },
    controls: { ...base.controls, ...(override.controls ?? {}) },
    seed: typeof override.seed === "number" ? override.seed : base.seed,
    effects: { ...base.effects, ...(override.effects ?? {}) },
    reset_seconds:
      typeof override.reset_seconds === "number"
        ? override.reset_seconds
        : base.reset_seconds,
  };
}

/** Fetch /config.json (no cache). Missing file or parse error → defaults
 * silently — the bundled defaults already match the React port's current
 * behavior, so a deploy without a config.json works unchanged. */
export async function loadConfig(): Promise<RtmgConfig> {
  try {
    const res = await fetch(`/config.json?t=${Date.now()}`, {
      cache: "no-store",
    });
    if (!res.ok) return DEFAULT_CONFIG;
    const json = (await res.json()) as Partial<RtmgConfig>;
    return mergeConfig(DEFAULT_CONFIG, json);
  } catch {
    return DEFAULT_CONFIG;
  }
}

function isDcwMode(v: unknown): v is DcwMode {
  return typeof v === "string" && (DCW_MODES as readonly string[]).includes(v);
}
function isDcwWavelet(v: unknown): v is DcwWavelet {
  return typeof v === "string" && (DCW_WAVELETS as readonly string[]).includes(v);
}

/** Push the supplied config into stores + non-store subscribers. Idempotent;
 * safe to call multiple times. The only mid-session callers today are the
 * boot path; future "Reload config" affordances would call this too. */
export function applyConfig(c: RtmgConfig): void {
  _activeConfig = c;

  // Numeric controls land on sliderValues + sliderTargets so the slider
  // UI and the param-sync tick agree.
  const sliderUpdates: Record<string, number> = {};
  for (const [k, v] of Object.entries(c.controls)) {
    if (typeof v === "number") sliderUpdates[k] = v;
  }

  usePerformanceStore.setState((s) => ({
    sliderValues: { ...s.sliderValues, ...sliderUpdates },
    sliderTargets: { ...s.sliderTargets, ...sliderUpdates },
    promptA: c.prompts.a,
    promptB: c.prompts.b,
    blend: c.prompts.blend,
    activeKey: c.engine.key,
    seed: c.seed,
    dcwEnabled:
      typeof c.controls.dcw_enabled === "boolean"
        ? c.controls.dcw_enabled
        : s.dcwEnabled,
    dcwMode: isDcwMode(c.controls.dcw_mode) ? c.controls.dcw_mode : s.dcwMode,
    dcwWavelet: isDcwWavelet(c.controls.dcw_wavelet)
      ? c.controls.dcw_wavelet
      : s.dcwWavelet,
  }));

  for (const fn of listeners) fn(c);
}
