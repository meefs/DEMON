"use client";

import { useEffect } from "react";

// First-run coachmark that points at the .install-drawer-handle to teach
// users that more controls live behind it. Mounts only on desktop (mobile
// has the LiteControls strip + "All controls" link, so the gap is
// desktop-shaped), only after the session is ready, and only the first
// time per user (dismissal persisted in localStorage).
//
// Dismissed by: any pointerdown anywhere, Esc key, the drawer opening,
// or an 8-second auto-hide. Whichever fires first writes the localStorage
// flag so the coachmark never reappears.
//
// Parent (AdvancedDrawer) owns the visibility gate and the dismiss
// handler; this component just renders the callout and wires the
// document-level dismiss listeners.

export const advancedCoachmarkStorageKey = "dd:advanced-coachmark-dismissed";
const AUTO_HIDE_MS = 8000;

interface Props {
  visible: boolean;
  onDismiss: () => void;
}

export function AdvancedCoachmark({ visible, onDismiss }: Props) {
  // Auto-hide after AUTO_HIDE_MS even without explicit dismissal so we
  // don't nag once the user has had a chance to notice it.
  useEffect(() => {
    if (!visible) return;
    const t = window.setTimeout(onDismiss, AUTO_HIDE_MS);
    return () => window.clearTimeout(t);
  }, [visible, onDismiss]);

  // Pointer / Esc dismissal. We listen at the document level so any
  // interaction at all counts as acknowledgement — clicks on the
  // coachmark itself fall through (pointer-events: none in CSS) and
  // hit whatever's underneath, so the user can click the handle to
  // both dismiss + open in one motion.
  useEffect(() => {
    if (!visible) return;
    const onPointer = () => onDismiss();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onDismiss();
    };
    document.addEventListener("pointerdown", onPointer, { once: true });
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [visible, onDismiss]);

  if (!visible) return null;

  return (
    <div className="advanced-coachmark" role="status" aria-live="polite">
      <span className="advanced-coachmark-text">
        Drag up for more controls — Press <kbd>O</kbd> to toggle
      </span>
      <span className="advanced-coachmark-arrow" aria-hidden="true">
        ▾
      </span>
    </div>
  );
}
