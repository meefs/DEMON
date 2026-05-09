"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { useConfirmStore } from "@/store/useConfirmStore";

// Singleton dialog driven by useConfirmStore. Mount once near the app
// root; any caller (component or store action) can prompt via
// `await confirm({ message, ... })` from useConfirmStore.
//
// Behavior parity with window.confirm(): blocks the operator's flow
// behind a backdrop, returns boolean, Esc/cancel resolves false,
// Enter/confirm resolves true. Differs in being async — call sites
// must await the result instead of branching synchronously.

export function ConfirmDialog() {
  const options = useConfirmStore((s) => s.options);
  const resolveDialog = useConfirmStore((s) => s.resolveDialog);
  const [mounted, setMounted] = useState(false);
  const confirmRef = useRef<HTMLButtonElement | null>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  useEffect(() => setMounted(true), []);

  const open = options != null;

  // Keep refs to the latest handler so the keydown listener doesn't
  // need to re-bind on every render.
  const resolveRef = useRef(resolveDialog);
  resolveRef.current = resolveDialog;

  useEffect(() => {
    if (!open) return;
    previouslyFocusedRef.current = document.activeElement as HTMLElement | null;
    confirmRef.current?.focus();
    return () => {
      previouslyFocusedRef.current?.focus?.();
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        e.preventDefault();
        resolveRef.current(false);
      } else if (e.key === "Enter") {
        const tag = (e.target as HTMLElement | null)?.tagName;
        if (tag === "SELECT" || tag === "TEXTAREA") return;
        e.preventDefault();
        resolveRef.current(true);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  if (!mounted || !options) return null;

  const {
    title = "Confirm",
    message,
    confirmLabel = "OK",
    cancelLabel = "Cancel",
    variant = "default",
  } = options;

  // Preserve \n in plain-string messages so existing confirm() copy
  // (which relied on \n\n paragraph breaks) renders the same way.
  const body =
    typeof message === "string" ? (
      <p className="confirm-dialog-message">{message}</p>
    ) : (
      message
    );

  return createPortal(
    <div
      className="confirm-dialog-backdrop"
      onClick={() => resolveDialog(false)}
      role="presentation"
    >
      <div
        className="confirm-dialog-modal"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="config-modal-accent" aria-hidden="true" />

        <div className="confirm-dialog-header">
          <h2 id="confirm-dialog-title" className="confirm-dialog-title">
            {title}
          </h2>
          <button
            type="button"
            className="config-modal-close"
            onClick={() => resolveDialog(false)}
            aria-label={cancelLabel}
          >
            ×
          </button>
        </div>

        <div className="confirm-dialog-body">{body}</div>

        <div className="confirm-dialog-footer">
          <button
            type="button"
            className="confirm-dialog-btn confirm-dialog-btn--secondary"
            onClick={() => resolveDialog(false)}
          >
            {cancelLabel}
          </button>
          <button
            ref={confirmRef}
            type="button"
            className={
              variant === "danger"
                ? "confirm-dialog-btn confirm-dialog-btn--danger"
                : "confirm-dialog-btn confirm-dialog-btn--primary"
            }
            onClick={() => resolveDialog(true)}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
