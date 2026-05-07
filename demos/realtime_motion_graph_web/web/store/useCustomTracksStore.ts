"use client";

import { create } from "zustand";

import type { DecodedFixture } from "@/engine/audio/loadFixture";

// In-memory cache for user-uploaded tracks. The decoded PCM lives in a
// non-reactive Map (Float32Array doesn't survive JSON / localStorage), and
// the names are mirrored into a reactive list so the fixture dropdown
// re-renders when an upload completes. Cleared on page reload — uploads
// are session-scoped, matching how the pod treats fixtures (it only ever
// sees the decoded PCM, never the file).
//
// `originalFiles` keeps the original encoded File alongside the decoded
// buffer so downstream consumers (e.g. demon-public-demo's saved-sessions
// feature, which uploads the original to a bucket on session save) can
// recover the bytes without having to re-prompt the user. It's a Map of
// name → File and lives next to `decoded` so adds stay atomic. Like
// `decoded`, it's non-reactive and read via getState().

interface CustomTracksState {
  /** Names in upload order. Reactive — components subscribe to this. */
  names: string[];
  /** Decoded buffers keyed by name. Read directly via getState() from
   *  non-React code (loadFixtureAudio); updates don't re-render. */
  decoded: Map<string, DecodedFixture>;
  /** Original encoded File keyed by name. Populated when add() is called
   *  with a File argument (the AudioSourceCrate upload path). May be
   *  empty for tracks added via other paths. */
  originalFiles: Map<string, File>;

  add: (name: string, decoded: DecodedFixture, file?: File) => void;
  has: (name: string) => boolean;
}

export const useCustomTracksStore = create<CustomTracksState>((set, get) => ({
  names: [],
  decoded: new Map(),
  originalFiles: new Map(),

  add: (name, decoded, file) =>
    set((s) => {
      const nextDecoded = new Map(s.decoded);
      nextDecoded.set(name, decoded);
      const nextOriginalFiles = new Map(s.originalFiles);
      if (file) nextOriginalFiles.set(name, file);
      const nextNames = s.names.includes(name) ? s.names : [...s.names, name];
      return {
        names: nextNames,
        decoded: nextDecoded,
        originalFiles: nextOriginalFiles,
      };
    }),

  has: (name) => get().decoded.has(name),
}));
