"""Webcam motion tracker and MIDI knob reader.

Torch-free. mido and rtmidi are imported lazily so the module can be
loaded without a MIDI stack installed.
"""

import threading
import time

import cv2
import numpy as np

from .knobs import KnobBank, KnobDef


class MotionTracker:
    def __init__(self, camera=0):
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.prev_gray = None
        self.smoothed = 0.0
        self.alpha = 0.3

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame, 0.0
        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        raw = np.mean(diff) / 255.0
        scaled = min(raw * 40.0, 1.0)
        self.smoothed = self.alpha * scaled + (1 - self.alpha) * self.smoothed
        return frame, self.smoothed

    def read_latest(self):
        return self.read()

    def release(self):
        self.cap.release()


class MidiKnobs:
    """Reads MIDI CC values from endless encoders and pad presses.

    Supports multiple knob banks. All 8 CCs (70-77) are remapped when
    the bank changes. Values persist across bank switches.

    Pad 3 (note 38) cycles to the next knob bank.
    Pad 4 (note 39) resets all params to defaults.
    """

    def __init__(self, banks: list, port_name=None):
        import mido
        names = mido.get_input_names()
        if not names:
            raise RuntimeError("No MIDI input devices found")
        if port_name is None:
            port_name = names[0]
        print(f"  MIDI: available ports: {names}")
        print(f"  MIDI: opening '{port_name}'...")
        try:
            self._port = mido.open_input(port_name)
        except Exception as e:
            print(f"  MIDI: failed to open '{port_name}': {e}")
            print(f"  MIDI: trying virtual port...")
            import rtmidi
            mi = rtmidi.MidiIn()
            ports = mi.get_ports()
            print(f"  MIDI: rtmidi ports: {ports}")
            for i, p in enumerate(ports):
                if port_name.split(" ")[0].lower() in p.lower():
                    self._port = mido.open_input(p)
                    print(f"  MIDI: opened '{p}' via fallback")
                    break
            else:
                raise

        self._banks = banks
        self._active_bank = 0

        # Flat value store keyed by param name (persists across bank switches)
        self._all_knobs: dict = {}
        self._values: dict = {}
        for bank in banks:
            for name, k in bank.knobs.items():
                if name not in self._values:
                    self._values[name] = k.default
                self._all_knobs[name] = k

        self._rebuild_cc_map()
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _rebuild_cc_map(self):
        """Remap CCs to the active bank's params."""
        bank = self._banks[self._active_bank]
        self._cc_map = {k.cc: name for name, k in bank.knobs.items()}
        self._active_knobs = bank.knobs

    @property
    def active_bank_index(self) -> int:
        with self._lock:
            return self._active_bank

    @property
    def active_bank(self) -> KnobBank:
        with self._lock:
            return self._banks[self._active_bank]

    def _poll(self):
        while self._running:
            for msg in self._port.iter_pending():
                if msg.type == "control_change" and msg.control in self._cc_map:
                    name = self._cc_map[msg.control]
                    knob = self._active_knobs[name]
                    delta = msg.value if msg.value < 64 else msg.value - 128
                    with self._lock:
                        v = self._values[name] + delta * knob.sensitivity / 127.0
                        self._values[name] = max(0.0, min(knob.max_val, v))
                elif msg.type == "note_on" and msg.velocity > 0:
                    note = msg.note
                    with self._lock:
                        if note == 38:                      # Pad 3: next bank
                            self._active_bank = (self._active_bank + 1) % len(self._banks)
                            self._rebuild_cc_map()
                            print(f"  MIDI: Bank -> {self._banks[self._active_bank].name}")
                        elif note == 39:                    # Pad 4: reset all
                            for n, k in self._all_knobs.items():
                                self._values[n] = k.default
                            print("  MIDI: Reset all params to defaults")
            time.sleep(0.001)

    def get(self, name: str) -> float:
        with self._lock:
            return self._values.get(name, 0.0)

    def get_all(self) -> dict:
        """Return all values for the active bank's params."""
        with self._lock:
            bank = self._banks[self._active_bank]
            return {name: self._values[name] for name in bank.knobs}

    def get_all_values(self) -> dict:
        """Return all values across all banks."""
        with self._lock:
            return dict(self._values)

    def get_param(self, name: str) -> float:
        """Read a param by name, returning 0.0 if not wired to any bank."""
        with self._lock:
            return self._values.get(name, 0.0)

    def all_knob_defs(self) -> dict:
        """Return all knob definitions across all banks."""
        return dict(self._all_knobs)

    def release(self):
        self._running = False
        self._thread.join(timeout=1)
        self._port.close()
