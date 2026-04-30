"""Differential Correction in Wavelet domain (DCW) for flow-matching sampling.

Sampler-side correction from Yu et al. "Elucidating the SNR-t Bias of
Diffusion Probabilistic Models", CVPR 2026 (arXiv:2604.16044). Ported
verbatim from ace-step/ACE-Step-1.5 v0.1.7 (PR #1120 + math fix
5d52875a), with the original three-file split collapsed into one local
module.

After each sampler step, decompose ``x_next`` and the predicted clean
sample ``denoised = x - v * t`` with a single-level 1-D DWT along the
temporal axis, then push ``x_next``'s frequency band(s) away from the
denoised estimate::

    xL, xH = DWT(x_next);   yL, yH = DWT(denoised)
    xL    += s_low  * (xL - yL)
    xH    += s_high * (xH - yH)
    x_next = IDWT(xL, xH)

ACE-Step latents are 1-D temporal tensors ``[B, T, C]`` at 25 Hz, so we
transpose to ``[B, C, T]`` before the DWT and back after the IDWT.
"""

from __future__ import annotations

from typing import Tuple

import torch
from loguru import logger


__all__ = [
    "VALID_DCW_MODES",
    "DCWCorrector",
    "dcw_low",
    "dcw_high",
    "dcw_double",
    "dcw_pix",
]

VALID_DCW_MODES = ("low", "high", "double", "pix")


# ---------------------------------------------------------------------------
# Lazy ``pytorch_wavelets`` loader
# ---------------------------------------------------------------------------


class _LazyWavelet:
    """Lazy loader + cache for ``pytorch_wavelets`` DWT1D modules.

    One ``DWT1DForward`` / ``DWT1DInverse`` pair per
    ``(device, dtype, wavelet)`` triple so repeated sampler steps don't
    rebuild filter banks.
    """

    def __init__(self) -> None:
        self._cache: dict = {}

    def get(
        self,
        device: torch.device,
        dtype: torch.dtype,
        wavelet: str,
    ) -> Tuple["torch.nn.Module", "torch.nn.Module"]:
        from pytorch_wavelets import DWT1DForward, DWT1DInverse
        key = (str(device), str(dtype), wavelet)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        # DCW is numerically sensitive; always run the DWT in fp32 on the
        # latent's device and cast back to the caller's dtype after IDWT.
        dwt = DWT1DForward(J=1, mode="zero", wave=wavelet).to(
            device=device, dtype=torch.float32
        )
        iwt = DWT1DInverse(mode="zero", wave=wavelet).to(
            device=device, dtype=torch.float32
        )
        self._cache[key] = (dwt, iwt)
        try:
            h0 = getattr(dwt, "h0", None)
            ntap = int(h0.shape[-1]) if h0 is not None else -1
        except Exception:
            ntap = -1
        logger.info(
            "[DCW] Built DWT1D for wavelet={!r} (low-pass filter taps={}, "
            "device={}, dtype={}).",
            wavelet, ntap, str(device), str(dtype),
        )
        return dwt, iwt


WAVELET_CACHE = _LazyWavelet()


# ---------------------------------------------------------------------------
# Layout helpers + primitives
# ---------------------------------------------------------------------------


def _btc_to_bct(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def _bct_to_btc(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def dcw_pix(x: torch.Tensor, y: torch.Tensor, scaler: float) -> torch.Tensor:
    """Pixel/latent-space differential correction (no wavelet transform).

    Matches the ``dcw_pix`` baseline in the DCW reference. Used as an
    ablation.
    """
    if scaler == 0.0:
        return x
    return x + scaler * (x - y)


def _dwt_pair(x: torch.Tensor, y: torch.Tensor, wavelet: str):
    """Run DWT on both latents.

    ``out_T`` is the original time length: ``pytorch_wavelets``
    zero-pads odd-T inputs to even, and the IDWT output is one sample
    longer than the input, so callers must trim back to ``out_T``.
    """
    dwt, iwt = WAVELET_CACHE.get(x.device, x.dtype, wavelet)
    x_bct = _btc_to_bct(x.to(torch.float32))
    y_bct = _btc_to_bct(y.to(torch.float32))
    xl, xh = dwt(x_bct)
    yl, yh = dwt(y_bct)
    return xl, xh, yl, yh, iwt, x.shape[1]


def dcw_low(
    x: torch.Tensor, y: torch.Tensor, scaler: float, wavelet: str = "haar"
) -> torch.Tensor:
    """Low-band-only correction (paper Eq. 18 / 20)."""
    if scaler == 0.0:
        return x
    xl, xh, yl, _yh, iwt, out_T = _dwt_pair(x, y, wavelet)
    xl = xl + scaler * (xl - yl)
    x_new = iwt((xl, xh))
    return _bct_to_btc(x_new[:, :, :out_T]).to(dtype=x.dtype)


def dcw_high(
    x: torch.Tensor, y: torch.Tensor, scaler: float, wavelet: str = "haar"
) -> torch.Tensor:
    """High-band-only correction."""
    if scaler == 0.0:
        return x
    xl, xh, _yl, yh, iwt, out_T = _dwt_pair(x, y, wavelet)
    xh_new = [xhi + scaler * (xhi - yhi) for xhi, yhi in zip(xh, yh)]
    x_new = iwt((xl, xh_new))
    return _bct_to_btc(x_new[:, :, :out_T]).to(dtype=x.dtype)


def dcw_double(
    x: torch.Tensor,
    y: torch.Tensor,
    low_scaler: float,
    high_scaler: float,
    wavelet: str = "haar",
) -> torch.Tensor:
    """Both bands corrected with independent scalers."""
    if low_scaler == 0.0 and high_scaler == 0.0:
        return x
    xl, xh, yl, yh, iwt, out_T = _dwt_pair(x, y, wavelet)
    if low_scaler != 0.0:
        xl = xl + low_scaler * (xl - yl)
    if high_scaler != 0.0:
        xh = [xhi + high_scaler * (xhi - yhi) for xhi, yhi in zip(xh, yh)]
    x_new = iwt((xl, xh))
    return _bct_to_btc(x_new[:, :, :out_T]).to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Sampler-facing wrapper
# ---------------------------------------------------------------------------


class DCWCorrector:
    """Stateful wrapper that applies DCW per sampler step.

    Per-step coefficients (paper Eq. 20 / 21, post-fix-5d52875a):

    * ``low``    : ``λ = t * scaler``         (strongest at high noise)
    * ``high``   : ``λ = (1 - t) * scaler``   (complementary, late steps)
    * ``double`` : low ``t * scaler``, high ``(1 - t) * high_scaler``
    * ``pix``    : raw ``scaler`` (no t modulation)
    """

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "double",
        scaler: float = 0.05,
        high_scaler: float = 0.02,
        wavelet: str = "haar",
    ) -> None:
        if mode not in VALID_DCW_MODES:
            raise ValueError(
                f"Invalid dcw_mode='{mode}'. Expected one of {VALID_DCW_MODES}."
            )
        self.enabled = bool(enabled)
        self.mode = mode
        self.scaler = float(scaler)
        self.high_scaler = float(high_scaler)
        self.wavelet = wavelet

    @property
    def is_active(self) -> bool:
        if not self.enabled:
            return False
        if self.mode == "double":
            return self.scaler != 0.0 or self.high_scaler != 0.0
        return self.scaler != 0.0

    def apply(
        self, x_next: torch.Tensor, denoised: torch.Tensor, t_curr: float,
    ) -> torch.Tensor:
        if not self.is_active:
            return x_next
        t = float(t_curr)
        low_s = t * self.scaler
        high_s = (1.0 - t) * self.scaler
        double_high_s = (1.0 - t) * self.high_scaler
        if self.mode == "low":
            return dcw_low(x_next, denoised, low_s, self.wavelet)
        if self.mode == "high":
            return dcw_high(x_next, denoised, high_s, self.wavelet)
        if self.mode == "double":
            return dcw_double(
                x_next, denoised, low_s, double_high_s, self.wavelet,
            )
        if self.mode == "pix":
            return dcw_pix(x_next, denoised, self.scaler)
        raise RuntimeError(f"unreachable dcw_mode={self.mode}")
