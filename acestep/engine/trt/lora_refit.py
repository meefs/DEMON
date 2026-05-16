"""TRT subclass of :class:`LoRAManagerBase`: writeback via IRefitter.

The base class owns catalog, lifecycle, prewarm, and delta math. This
file is just the TRT-specific writeback path: pre-allocated CPU
buffers per refittable weight, named in the engine's ``decoder.``
prefix space.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Set

from loguru import logger
import numpy as np
import torch

from acestep.engine.lora import (
    LoRAManagerBase,
    LoRAState,           # re-exported for back-compat
    LoRADescriptor,      # re-exported for back-compat
    _LoRAEntry,          # re-exported for tests
)

# numpy dtype -> torch dtype
_NP_TO_TORCH = {
    np.float32: torch.float32,
    np.float16: torch.float16,
}


# Backward-compat alias for tests / external code that imported the old
# name.  The old type held only (lora_id, path, strength, deltas); the
# new one is a superset, so kw-only construction with the old fields
# still works.
_ActiveLoRA = _LoRAEntry


class TRTLoRAManager(LoRAManagerBase):
    """LoRA writeback into a TRT engine via IRefitter."""

    def __init__(
        self,
        engine,
        decoder: torch.nn.Module,
        device: torch.device = torch.device("cuda"),
        trt_weight_prefix: str = "decoder.",
        checkpoint_path: Optional[str] = None,
        engine_path: Optional[str] = None,
    ):
        import tensorrt as trt

        self._engine = engine
        self._device = device
        self._trt_prefix = trt_weight_prefix
        self._trt = trt
        self._trt_logger = trt.Logger(trt.Logger.WARNING)

        # Per-param transpose flag: True if the engine slot stores the
        # weight in ONNX MatMul's ``[in_dim, out_dim]`` orientation (dynamo
        # output) instead of torch nn.Linear's ``[out_dim, in_dim]``.
        # Populated from a sidecar refit manifest emitted by
        # ``rename_val_initializers_to_fqn`` next to the ONNX, then copied
        # next to the engine by the build pipeline. Absent manifest means
        # all weights use torch orientation (legacy torchscript path).
        self._transpose_for_engine: Dict[str, bool] = {}
        if engine_path is not None:
            manifest_path = Path(str(engine_path) + ".refit_manifest.json")
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    transposed = manifest.get("weights_transposed", [])
                    prefix = trt_weight_prefix
                    for fqn in transposed:
                        # Manifest entries are engine-namespace ("decoder.X");
                        # strip the prefix to match the param-name keys used
                        # by the manager elsewhere.
                        if fqn.startswith(prefix):
                            self._transpose_for_engine[fqn[len(prefix):]] = True
                    logger.info(
                        "Refit manifest loaded: {} transposed-layout weights",
                        len(self._transpose_for_engine),
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to read refit manifest at {}: {}. LoRA refit "
                        "will assume torch [out, in] orientation; for dynamo-"
                        "built engines this produces wrong outputs under LoRA.",
                        manifest_path, exc,
                    )

        # TRT dtype -> numpy dtype
        _trt_to_np = {trt.float32: np.float32, trt.float16: np.float16}
        if hasattr(trt, "bfloat16"):
            _trt_to_np[trt.bfloat16] = np.float32

        refitter = trt.Refitter(engine, self._trt_logger)
        if not hasattr(refitter, "get_all_weights"):
            raise RuntimeError("TRT engine refitting requires TensorRT 10.0+")

        all_trt_names = list(refitter.get_all_weights())
        if not all_trt_names:
            raise RuntimeError(
                "Engine has no refittable weights. Rebuild with refit=True."
            )

        self._refitter = refitter

        decoder_params = dict(decoder.named_parameters()) if decoder is not None else {}
        checkpoint_file = None
        if not decoder_params and checkpoint_path:
            from safetensors import safe_open
            logger.info("Loading base weights from checkpoint: {}", checkpoint_path)
            checkpoint_file = safe_open(checkpoint_path, framework="pt")

        has_prototype = hasattr(refitter, "get_weights_prototype")

        # Mapping + per-weight buffers/dtypes. _base_weights and
        # _param_dtype are what LoRAManagerBase looks at.
        self._param_to_trt: Dict[str, str] = {}
        self._base_weights: Dict[str, torch.Tensor] = {}   # native dtype, CPU
        self._refit_bufs: Dict[str, torch.Tensor] = {}     # pre-alloc output
        self._np_dtype: Dict[str, np.dtype] = {}
        self._param_dtype: Dict[str, torch.dtype] = {}

        matched = 0
        for trt_name in all_trt_names:
            if not trt_name.startswith(trt_weight_prefix):
                continue
            param_name = trt_name[len(trt_weight_prefix):]

            np_dt = np.float32
            if has_prototype:
                try:
                    proto = refitter.get_weights_prototype(trt_name)
                    np_dt = _trt_to_np.get(proto.dtype, np.float32)
                except Exception:
                    pass
            torch_dt = _NP_TO_TORCH.get(np_dt, torch.float32)

            raw_w = None
            if param_name in decoder_params:
                raw_w = decoder_params[param_name].data
            elif checkpoint_file is not None:
                try:
                    raw_w = checkpoint_file.get_tensor(trt_name)
                except Exception:
                    pass

            if raw_w is None:
                continue

            base = raw_w.to(dtype=torch_dt).cpu().contiguous()
            # If the engine stored this weight transposed (dynamo MatMul
            # layout), keep our base + refit buffer in the same
            # orientation so the bytes we hand to ``set_named_weights``
            # match the slot's memory layout. Deltas arrive in torch
            # [out, in] from ``_compute_deltas`` and are transposed
            # on the fly inside ``_apply_to_engine``.
            if self._transpose_for_engine.get(param_name, False) and base.dim() == 2:
                base = base.transpose(0, 1).contiguous()
            self._param_to_trt[param_name] = trt_name
            self._base_weights[param_name] = base
            self._refit_bufs[param_name] = torch.empty_like(base)
            self._np_dtype[param_name] = np_dt
            self._param_dtype[param_name] = torch_dt
            matched += 1

        logger.info(
            "TRT LoRA manager ready: {}/{} engine weights mapped (prefix='{}')",
            matched, len(all_trt_names), trt_weight_prefix,
        )
        if matched == 0:
            logger.warning(
                "No engine weights matched! TRT names sample: {}",
                all_trt_names[:5],
            )

        super().__init__()

    def _delta_compute_device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------
    # Engine writeback (IRefitter)
    # ------------------------------------------------------------------

    def _apply_to_engine(self, param_names: Set[str]) -> None:
        """Refit engine weights using pre-allocated buffers + in-place ops.

        All math runs in the engine's native dtype (typically fp16) so
        the numpy view fed to ``set_named_weights`` is zero-copy. A
        strength-0 ENABLED entry is skipped explicitly: the add_ would
        be a math-no-op but still walks the full weight, wasting cycles
        for slider-driven UIs that leave placeholders at 0.
        """
        refitter = self._refitter
        count = 0

        for param_name in param_names:
            trt_name = self._param_to_trt.get(param_name)
            if trt_name is None:
                continue

            buf = self._refit_bufs[param_name]
            buf.copy_(self._base_weights[param_name])

            transpose_delta = self._transpose_for_engine.get(param_name, False)
            for entry in self._loras.values():
                if entry.state != LoRAState.ENABLED:
                    continue
                if entry.strength == 0.0:
                    continue
                if entry.deltas and param_name in entry.deltas:
                    delta = entry.deltas[param_name]
                    # Deltas live in torch nn.Linear's [out, in] orientation.
                    # If the engine slot stores [in, out] (dynamo MatMul),
                    # transpose before accumulating into the engine-layout
                    # buffer. add_ requires shape match with buf.
                    if transpose_delta and delta.dim() == 2:
                        delta = delta.transpose(0, 1).contiguous()
                    buf.add_(delta, alpha=entry.strength)

            arr = buf.numpy()
            ok = refitter.set_named_weights(trt_name, arr)
            if not ok:
                proto_desc = "unknown"
                if hasattr(refitter, "get_weights_prototype"):
                    try:
                        proto = refitter.get_weights_prototype(trt_name)
                        proto_desc = f"dtype={proto.dtype}, size={proto.size}"
                    except Exception:
                        pass
                raise RuntimeError(
                    "TRT rejected refit weights for "
                    f"{trt_name}: array dtype={arr.dtype}, shape={arr.shape}; "
                    f"engine prototype {proto_desc}"
                )
            count += 1

        if count > 0:
            ok = refitter.refit_cuda_engine()
            if not ok:
                missing = refitter.get_missing_weights()
                raise RuntimeError(
                    f"TRT refit failed. Missing weights: {missing}"
                )


# Re-export the lifecycle public symbols so existing imports
# (`from acestep.engine.trt.lora_refit import TRTLoRAManager, LoRAState, ...`)
# keep working without churn.
__all__ = [
    "TRTLoRAManager",
    "LoRAState",
    "LoRADescriptor",
    "_LoRAEntry",
    "_ActiveLoRA",
]
