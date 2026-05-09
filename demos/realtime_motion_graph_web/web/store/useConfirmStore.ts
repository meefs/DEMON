"use client";

import type { ReactNode } from "react";
import { create } from "zustand";

export interface ConfirmOptions {
  title?: string;
  message: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: "default" | "danger";
}

interface ConfirmState {
  options: ConfirmOptions | null;
  resolve: ((value: boolean) => void) | null;
  request: (opts: ConfirmOptions) => Promise<boolean>;
  resolveDialog: (value: boolean) => void;
}

export const useConfirmStore = create<ConfirmState>((set, get) => ({
  options: null,
  resolve: null,
  request: (opts) =>
    new Promise<boolean>((resolve) => {
      // If a previous dialog is still pending, treat it as cancelled so
      // its caller never hangs. Last-call-wins.
      const prev = get().resolve;
      if (prev) prev(false);
      set({ options: opts, resolve });
    }),
  resolveDialog: (value) => {
    const { resolve } = get();
    if (resolve) resolve(value);
    set({ options: null, resolve: null });
  },
}));

// Module-level imperative API so non-component code (event handlers,
// store actions) can prompt without threading a hook through.
export function confirm(opts: ConfirmOptions): Promise<boolean> {
  return useConfirmStore.getState().request(opts);
}
