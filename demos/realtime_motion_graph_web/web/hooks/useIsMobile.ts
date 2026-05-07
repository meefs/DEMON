"use client";

import { useEffect, useState } from "react";

// Catches:
//   • portrait phones (max-width 768) — iPhone, small Android.
//   • landscape phones (max-height 500 + landscape) — iPhone in landscape is
//     ~852×393, wider than the 768px width breakpoint, so we need a height-
//     based fallback. iPad landscape (e.g. 1180×820) doesn't match.
const QUERY =
  "(max-width: 768px), (max-height: 500px) and (orientation: landscape)";

// Returns true while the viewport matches the mobile breakpoint. SSR-safe:
// initial value is false on the server, then synchronized after mount.
export function useIsMobile(): boolean {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const mql = window.matchMedia(QUERY);
    setIsMobile(mql.matches);
    const onChange = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);

  return isMobile;
}
