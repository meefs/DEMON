"""Shared runtime initialization helpers (backend validation + TRT wiring).

Both ``Session`` and the node-level ``LoadModel`` need to configure
tensorrt / compile / eager backends on a ``ModelContext``. This module
hosts the shared logic so the two entry points can't drift.
"""

from __future__ import annotations

from typing import Optional

_VALID_BACKENDS = ("eager", "compile", "tensorrt")


def validate_backends(
    *,
    decoder_backend: str,
    vae_backend: str,
    trt_engines: Optional[dict[str, str]],
) -> dict[str, str]:
    """Validate backend/trt_engines combinations. Returns a non-None dict."""
    if decoder_backend not in _VALID_BACKENDS:
        raise ValueError(
            f"decoder_backend must be one of {_VALID_BACKENDS}, got {decoder_backend!r}"
        )
    if vae_backend not in _VALID_BACKENDS:
        raise ValueError(
            f"vae_backend must be one of {_VALID_BACKENDS}, got {vae_backend!r}"
        )

    trt_engines = trt_engines or {}
    if decoder_backend == "tensorrt" and "decoder" not in trt_engines:
        raise ValueError("decoder_backend='tensorrt' requires trt_engines['decoder']")
    if vae_backend == "tensorrt" and (
        "vae_encode" not in trt_engines or "vae_decode" not in trt_engines
    ):
        raise ValueError(
            "vae_backend='tensorrt' requires both "
            "trt_engines['vae_encode'] and trt_engines['vae_decode']"
        )
    if "decoder" in trt_engines and decoder_backend != "tensorrt":
        raise ValueError(
            "trt_engines['decoder'] provided but decoder_backend != 'tensorrt'"
        )
    if (
        ("vae_encode" in trt_engines or "vae_decode" in trt_engines)
        and vae_backend != "tensorrt"
    ):
        raise ValueError(
            "trt_engines['vae_encode'/'vae_decode'] provided but "
            "vae_backend != 'tensorrt'"
        )

    return trt_engines


def apply_trt_backends(
    ctx,
    *,
    decoder_backend: str,
    vae_backend: str,
    trt_engines: dict[str, str],
    device: str,
) -> None:
    """Wire TRT engines into a freshly constructed ModelContext.

    Assumes ``ctx`` was built with ``skip_decoder``/``skip_vae`` set to match
    the tensorrt flags — PyTorch weights for those components are absent and
    TRT engines take over.
    """
    import torch

    if decoder_backend == "tensorrt":
        from acestep.engine.diffusion import DiffusionEngine

        ctx._diffusion_engine = DiffusionEngine(
            ctx.model,
            trt_engine_path=trt_engines["decoder"],
            compile_loops=False,  # TRT decoder, no need to compile loops
        )

    if vae_backend == "tensorrt":
        from acestep.nodes.vae_nodes import _get_trt_vae

        dev = torch.device(device)
        _get_trt_vae(trt_engines["vae_encode"], dev)
        _get_trt_vae(trt_engines["vae_decode"], dev)


def backends_to_model_context_flags(
    *, decoder_backend: str, vae_backend: str
) -> dict[str, bool]:
    """Map high-level backend strings to ModelContext low-level flags."""
    return {
        "skip_decoder": decoder_backend == "tensorrt",
        "skip_vae": vae_backend == "tensorrt",
        "compile_decoder": decoder_backend == "compile",
        "compile_vae": vae_backend == "compile",
    }
