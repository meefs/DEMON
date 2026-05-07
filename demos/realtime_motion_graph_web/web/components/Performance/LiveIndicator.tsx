"use client";

import { useSessionStore } from "@/store/useSessionStore";

// Persistent "the session is live" indicator. Renders a pulsing red dot
// next to a "LIVE" word in the top-right safe-area corner whenever the
// session is connected and ready. Helps the operator confirm at a glance
// that audio is still being processed even when their attention is on
// the spectrum / ribbons.
//
// Pointer-events disabled so it never blocks taps on the canvas.
export function LiveIndicator() {
  const status = useSessionStore((s) => s.status);
  if (status !== "ready") return null;
  return (
    <div className="live-indicator" aria-hidden="true">
      <span className="live-indicator-dot" />
      <span>LIVE</span>
    </div>
  );
}
