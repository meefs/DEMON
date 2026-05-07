"use client";

// Phone-only "rotate to landscape" gate. Shows a full-bleed overlay
// whenever a phone-sized viewport is in portrait orientation. The
// rest of the UI keeps mounting underneath — the overlay just covers
// it — so dismissing the overlay (rotating) reveals a fully-warm app
// without re-mount cost.
//
// Active when:
//   - max-width: 768px AND orientation: portrait  (covers small portraits)
//   - max-height: 500px AND orientation: portrait (no — height excludes here)
// Practically: phones in portrait. iPad portrait does NOT trigger because
// 820 > 768. Pure CSS — `display: none` flips to `display: flex` based on
// the media query.
export function PortraitLockOverlay() {
  return (
    <div
      className="portrait-lock"
      role="dialog"
      aria-modal="true"
      aria-label="Rotate to landscape"
    >
      <div className="portrait-lock-card">
        <div className="portrait-lock-icon" aria-hidden="true">
          <svg
            viewBox="0 0 64 64"
            width="64"
            height="64"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <rect x="14" y="6" width="28" height="48" rx="4" />
            <line x1="22" y1="50" x2="34" y2="50" />
            <path d="M44 22 L52 30 L44 38" />
            <path d="M52 30 L26 30" strokeDasharray="3 3" />
          </svg>
        </div>
        <div className="portrait-lock-title">Rotate to landscape</div>
        <div className="portrait-lock-body">
          Daydream is a landscape-only experience on phones. Turn your device
          sideways to begin.
        </div>
      </div>
    </div>
  );
}
