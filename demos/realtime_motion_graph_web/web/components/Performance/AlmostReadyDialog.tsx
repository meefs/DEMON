"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { VALID_KEYSCALES } from "@/types/engine";

// Sits between file-pick and fixture-swap so users can confirm the
// upload before playback transitions. Two jobs:
//   1. If the source was longer than 240 s, the parent has already
//      auto-trimmed it; we surface that fact and offer "Pick another
//      song" as a one-click escape.
//   2. Let the user choose a key for the model: default is "Auto-detect"
//      (server CNN runs as part of the swap and its result populates the
//      activeKey). "Set manually" sets a one-shot override that wins
//      over the CNN's result for this swap (see useFixtureSwap.ts).

type KeyMode = "auto" | "manual";

export interface AlmostReadyDialogProps {
  fileName: string;
  wasTrimmed: boolean;
  /** Default value for the manual-mode dropdown. Usually the user's
   *  current activeKey so they don't have to re-pick if they had a
   *  preference. */
  defaultKey: string;
  onContinue: (opts: { keyOverride: string | null }) => void;
  /** Only invoked when wasTrimmed is true; parent re-opens the file
   *  picker so the user can swap to a shorter source instead of
   *  accepting the trim. */
  onPickAnother: () => void;
  onClose: () => void;
}

export function AlmostReadyDialog({
  fileName,
  wasTrimmed,
  defaultKey,
  onContinue,
  onPickAnother,
  onClose,
}: AlmostReadyDialogProps) {
  const [mounted, setMounted] = useState(false);
  const [mode, setMode] = useState<KeyMode>("auto");
  const [manualKey, setManualKey] = useState(
    VALID_KEYSCALES.includes(defaultKey) ? defaultKey : "C major",
  );
  const continueRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => setMounted(true), []);

  // Esc closes; preventDefault so an open AdvancedDrawer underneath
  // doesn't also toggle.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        e.preventDefault();
        onClose();
      } else if (e.key === "Enter") {
        // Enter = primary action, but only when focus isn't already
        // on a form control whose Enter has its own meaning.
        const tag = (e.target as HTMLElement | null)?.tagName;
        if (tag === "SELECT" || tag === "TEXTAREA") return;
        e.preventDefault();
        onContinue({ keyOverride: mode === "manual" ? manualKey : null });
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [mode, manualKey, onClose, onContinue]);

  // Move keyboard focus to the primary button when the dialog mounts so
  // Enter / Space immediately fires Continue.
  useEffect(() => {
    if (!mounted) return;
    continueRef.current?.focus();
  }, [mounted]);

  if (!mounted) return null;

  return createPortal(
    <div
      className="almost-ready-backdrop"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="almost-ready-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="almost-ready-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="config-modal-accent" aria-hidden="true" />

        <div className="almost-ready-header">
          <h2 id="almost-ready-title" className="almost-ready-title">
            Almost Ready
          </h2>
          <button
            type="button"
            className="config-modal-close"
            onClick={onClose}
            aria-label="Cancel upload"
          >
            ×
          </button>
        </div>

        <div className="almost-ready-body">
          <p className="almost-ready-filename" title={fileName}>
            {fileName}
          </p>

          {wasTrimmed && (
            <p className="almost-ready-trim-msg">
              Uploads are limited to 240 seconds. We&apos;ve trimmed your
              upload to fit within this limit.
            </p>
          )}

          <fieldset className="almost-ready-key-section">
            <legend className="almost-ready-key-legend">Key</legend>

            <label className="almost-ready-key-mode">
              <input
                type="radio"
                name="key-mode"
                value="auto"
                checked={mode === "auto"}
                onChange={() => setMode("auto")}
              />
              <span>
                <strong>Auto-detect</strong>
                <span className="almost-ready-key-mode-hint">
                  We&apos;ll detect the song&apos;s key automatically.
                </span>
              </span>
            </label>

            <label className="almost-ready-key-mode">
              <input
                type="radio"
                name="key-mode"
                value="manual"
                checked={mode === "manual"}
                onChange={() => setMode("manual")}
              />
              <span>
                <strong>Set manually</strong>
                <span className="almost-ready-key-mode-hint">
                  Tells the model the song&apos;s key. Does not change
                  the song&apos;s pitch.
                </span>
              </span>
            </label>

            {mode === "manual" && (
              <select
                className="almost-ready-key-select fixture-select"
                value={manualKey}
                onChange={(e) => setManualKey(e.target.value)}
                aria-label="Pick a key"
              >
                {VALID_KEYSCALES.map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            )}
          </fieldset>
        </div>

        <div className="almost-ready-footer">
          {wasTrimmed && (
            <button
              type="button"
              className="almost-ready-btn almost-ready-btn--secondary"
              onClick={onPickAnother}
            >
              Pick another song
            </button>
          )}
          <button
            ref={continueRef}
            type="button"
            className="almost-ready-btn almost-ready-btn--primary"
            onClick={() =>
              onContinue({
                keyOverride: mode === "manual" ? manualKey : null,
              })
            }
          >
            Continue
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
