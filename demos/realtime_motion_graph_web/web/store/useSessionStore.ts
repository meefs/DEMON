"use client";

import { create } from "zustand";

import type { AudioPlayer } from "@/engine/audio/AudioPlayer";
import type { NetworkMonitor } from "@/engine/networkMonitor";
import type { RemoteBackend } from "@/engine/protocol";

// Live-session lifecycle state. The non-serializable RemoteBackend +
// AudioPlayer instances live here so React components and hooks can react
// to state changes (status, errors) without owning the lifecycle directly.

export type SessionStatus =
  | "idle"
  | "loading-fixture"
  | "connecting"
  | "ready"
  | "error"
  | "closed";

interface SessionState {
  status: SessionStatus;
  message: string;
  remote: RemoteBackend | null;
  player: AudioPlayer | null;
  /** Network-quality monitor — owns the slice listener + 500ms evaluator
   *  interval. Lifecycle == session lifecycle so reset() always tears it
   *  down. Mirrors how `remote` and `player` are owned here. */
  monitor: NetworkMonitor | null;
  /** Server-issued WS URL (from /api/queue/join). Null when no queue is in
   *  use — useStartSession falls back to defaultWsUrl(). */
  wsUrl: string | null;
  /** Active checkpoint's model-scale label ("2B" | "5B" | null). Set
   *  from the WS ready message and from /api/loras. Null when unknown.
   *  The LoRA library uses this to hide LoRAs whose trained
   *  ``base_model_scale`` doesn't match. */
  checkpointScale: string | null;
  /** Current StreamPipeline ring-buffer depth (concurrent denoising
   *  slots). Mirrors the server's view; updated from ``ready`` and
   *  ``depth_applied`` messages. Null until a session is live. */
  pipelineDepth: number | null;
  /** Server-imposed ceiling on ``pipelineDepth`` — TRT engine batch_max
   *  for TRT decoders, 4 for eager / compile. Null until ready. */
  maxPipelineDepth: number | null;

  setStatus: (status: SessionStatus, message?: string) => void;
  setSession: (remote: RemoteBackend | null, player: AudioPlayer | null) => void;
  setMonitor: (monitor: NetworkMonitor | null) => void;
  setWsUrl: (wsUrl: string | null) => void;
  setCheckpointScale: (scale: string | null) => void;
  setPipelineDepth: (depth: number | null) => void;
  setMaxPipelineDepth: (max: number | null) => void;
  reset: () => void;
}

export const useSessionStore = create<SessionState>((set, get) => ({
  status: "idle",
  message: "",
  remote: null,
  player: null,
  monitor: null,
  wsUrl: null,
  checkpointScale: null,
  pipelineDepth: null,
  maxPipelineDepth: null,

  setStatus: (status, message = "") => set({ status, message }),
  setSession: (remote, player) => set({ remote, player }),
  setMonitor: (monitor) => set({ monitor }),
  setWsUrl: (wsUrl) => set({ wsUrl }),
  setCheckpointScale: (scale) => set({ checkpointScale: scale }),
  setPipelineDepth: (depth) => set({ pipelineDepth: depth }),
  setMaxPipelineDepth: (max) => set({ maxPipelineDepth: max }),
  reset: () => {
    try {
      get().monitor?.stop();
    } catch {}
    set({
      status: "idle",
      message: "",
      remote: null,
      player: null,
      monitor: null,
      pipelineDepth: null,
      maxPipelineDepth: null,
      // checkpointScale survives reset on purpose: the server's
      // checkpoint doesn't change across sessions, and pre-fetching
      // it from /api/loras lets the library filter render correctly
      // even before the first session starts.
    });
  },
}));
