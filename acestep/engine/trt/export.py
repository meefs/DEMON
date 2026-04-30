"""ONNX export and TensorRT engine build for the ACE-Step decoder.

Export flow:
  1. Wrap decoder in DecoderForExport (fixes Lambda, forces SDPA, no cache)
  2. Export to ONNX with dynamic B / T / L_enc axes
  3. Build TRT engine with FP16 and optimization profiles

Precision strategy:
  - Export weights in fp32 (preserves full precision in ONNX graph)
  - TRT builder converts to fp16 internally with its own kernel selection
  - This avoids the bf16-to-fp16 silent truncation that causes wrong output
    when exporting directly in half precision
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

from loguru import logger
import torch
import torch.nn as nn


# ------------------------------------------------------------------
# Traceable replacement for the Lambda(transpose) modules
# ------------------------------------------------------------------

class _Transpose12(nn.Module):
    """Transpose dims 1 and 2.  Drop-in for Lambda(lambda x: x.transpose(1, 2))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)


class _Fp32CastWrapper(nn.Module):
    """Run an inner module in fp32, casting around it.

    Used for the bf16_mixed recipe where TRT has no bf16 kernel for some
    op (e.g., the proj_out ConvTranspose1d). The inner module's weights
    are cast to fp32 in-place; this wrapper bridges the dtype boundary at
    the call site so PyTorch nn.Conv*/nn.Linear strict dtype checks pass.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        inner.float()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        return self.inner(x.float()).to(out_dtype)


def _cast_recursive(obj, dtype: torch.dtype):
    """Recursively cast floating-point tensors in nested structures."""
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return obj.to(dtype)
        return obj
    if isinstance(obj, tuple):
        return tuple(_cast_recursive(x, dtype) for x in obj)
    if isinstance(obj, list):
        return [_cast_recursive(x, dtype) for x in obj]
    if isinstance(obj, dict):
        return {k: _cast_recursive(v, dtype) for k, v in obj.items()}
    return obj


class _AttnFp32Wrapper(nn.Module):
    """Wrap a self-attention module to run entirely in fp32.

    Used for the fp16_attn_safe recipe to handle XL turbo's outlier
    attention layers (0 and 30) where q_norm/k_norm weights reach ~31
    and amplify Q/K so far that the Q@K^T matmul output overflows fp16's
    65 504 ceiling. Casting the attention into fp32 keeps the rest of
    the model in fp16 for full tensor-core speed while making the few
    overflowing layers numerically safe.

    The wrapper accepts the same call signature as AceStepAttention.forward
    (hidden_states + arbitrary kwargs) and handles the dtype crossing at
    the call boundary.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        inner.float()
        self.inner = inner

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        out_dtype = hidden_states.dtype
        hidden_states = _cast_recursive(hidden_states, torch.float32)
        args = tuple(_cast_recursive(a, torch.float32) for a in args)
        kwargs = {k: _cast_recursive(v, torch.float32) for k, v in kwargs.items()}
        out = self.inner(hidden_states, *args, **kwargs)
        return _cast_recursive(out, out_dtype)


class _MlpFp16Wrapper(nn.Module):
    """Wrap a Qwen3-style MLP (gate_proj/up_proj/down_proj + SiLU) to run
    its body in fp16 with bf16 I/O.

    Used for the bf16_mlp_fp16 recipe. The MLP is the biggest single
    contributor to per-layer compute (gate+up+down matmuls plus the
    activation), so wrapping it as a unit gives the biggest single
    speedup per cast pair. Cast surface is 1 layer = 2 casts (in + out)
    instead of 11 casts (one per Linear), so TRT's tactic search stays
    tractable.

    The MLP's input is the AdaLN-modulated post-RMSNorm output of
    hidden_states, which is bounded; the output is fed into a gated
    residual addition. Both fit in fp16.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        inner.half()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        return self.inner(x.to(torch.float16)).to(out_dtype)


class _LinearFp16Wrapper(nn.Module):
    """Wrap a single nn.Linear (or any single-tensor-in / single-tensor-out
    module) so it computes in fp16 with bf16 I/O.

    Used for the bf16_matmul_fp16 recipe: instead of wrapping whole DiT
    layers (which exposes a wide bf16<->fp16 cast boundary that TRT
    fuses across, breaking the precision intent), we wrap the smallest
    possible unit -- the individual matmul -- with a tight cast pair.
    The trace records:
        Cast(bf16 -> fp16) -> matmul (fp16 weights, fp16 inputs) -> Cast(fp16 -> bf16)
    The cast surface is per-Linear, so TRT's fusion radius is limited
    to the cast itself plus the matmul. Profiling shows the XL turbo
    decoder spends ~77 % of its time in linear/matmul, so swapping
    those to fp16 kernels recovers most of the bf16 throughput penalty
    on Blackwell while leaving the residual stream in bf16 (which is
    necessary because the residual peaks at ~190 000 in the middle
    layers, well above fp16's 65 504 ceiling).

    The wrapper assumes the inner module's input is bounded enough to
    fit in fp16. For AceStepDiTLayer's Linears (q/k/v/o_proj and
    mlp.gate/up/down_proj), the input is always post-RMSNorm output
    multiplied by AdaLN scale/shift, which RMSNorm bounds to unit
    magnitude times moderate scale weights -- well within fp16 range.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        inner.half()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        return self.inner(x.to(torch.float16)).to(out_dtype)


class _LayerFp16Wrapper(nn.Module):
    """Wrap a full DiT layer so its body runs in fp16 with bf16 I/O.

    The XL turbo decoder's residual stream grows past fp16's 65 504 ceiling
    in the middle layers (peaks ~190k around layer 18 in bf16), so the
    cumulative residual cannot be stored in fp16. But the *individual*
    matmuls inside each layer (Q/K/V projections, attention output,
    MLP gate/up/down) operate on per-layer hidden states that are bounded
    by RMSNorm's unit-RMS output, so the matmuls themselves can run in
    fp16 as long as the residual addition happens in bf16.

    This wrapper casts the layer's weights to fp16 in-place and converts
    every floating-point input to fp16 at the call boundary, then casts
    the layer outputs back to bf16 so the upstream caller (the DiT
    forward loop) can do the residual addition in bf16. The trace records:
        Cast(bf16 -> fp16) -> layer body (fp16 matmuls) -> Cast(fp16 -> bf16)
    With strongly_typed=True the engine respects these casts and uses
    fp16 tensor cores for the layer body while preserving bf16 dynamic
    range across the residual stream.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        inner.half()
        self.inner = inner

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        out_dtype = hidden_states.dtype
        hidden_states = _cast_recursive(hidden_states, torch.float16)
        args = tuple(_cast_recursive(a, torch.float16) for a in args)
        kwargs = {k: _cast_recursive(v, torch.float16) for k, v in kwargs.items()}
        out = self.inner(hidden_states, *args, **kwargs)
        return _cast_recursive(out, out_dtype)


# ------------------------------------------------------------------
# Export wrapper
# ------------------------------------------------------------------

class DecoderForExport(nn.Module):
    """Thin wrapper that makes AceStepDiTModel safe for ONNX tracing.

    Changes vs. the raw decoder forward():
      - Lambda modules replaced with _Transpose12 for traceability
      - Attention implementation forced to SDPA (no flash_attn CUDA kernels)
      - KV cache disabled (use_cache=False, past_key_values=None)
      - output_attentions=False (no extra tuple elements)
      - timestep_r set equal to timestep (inference convention)
      - Returns the velocity tensor directly, not a tuple

    The input attention_mask and encoder_attention_mask parameters are
    intentionally set to None because the decoder's forward() shadows
    them immediately with local None assignments (lines 1378-1382 in
    the modeling file) and constructs full bidirectional masks from
    scratch via create_4d_mask().  Passing None here is therefore
    identical to passing torch.ones and avoids two unnecessary dynamic
    inputs in the TRT engine.
    """

    def __init__(
        self,
        decoder: nn.Module,
        mixed_precision: bool = False,
        precision: str = "fp32",
    ):
        """
        Args:
            decoder: AceStepDiTModel instance to wrap.
            mixed_precision: Legacy fp16+fp32 islands recipe (2B turbo).
                When True, ``precision`` is ignored. Convert decoder to fp16
                bulk + fp32 for timestep / AdaLN / RMSNorm. Designed for the
                2B turbo where activations stay inside fp16 dynamic range.
            precision: Used when ``mixed_precision`` is False. One of:
                - "fp32": leave dtypes as-is for tracing in fp32 (default)
                - "bf16": pure bf16 throughout. Use only for non-strongly-
                  typed builds (TRT picks bf16 vs fp32 freely). The dynamo
                  exporter is required for tracing.
                - "bf16_mixed": bf16 bulk + fp32 islands for AdaLN, norms,
                  timestep, AND proj_in/proj_out. The XL turbo decoder
                  needs this because (a) q_norm/k_norm scales reaching ~31
                  in layers 0/30 break fp16, so bf16 is the only safe
                  bulk dtype, and (b) TRT lacks a bf16 ConvTranspose
                  kernel for proj_out's [2560,64,2,1] shape, so the patch
                  embedding has to stay fp32. Strongly-typed builds.
        """
        super().__init__()
        self.decoder = decoder
        if mixed_precision:
            self.precision = "fp16_mixed"
        else:
            self.precision = precision

        # Replace Lambda with traceable transpose
        self._replace_lambdas()

        # Force SDPA so the graph contains only standard ops
        self.decoder.config._attn_implementation = "sdpa"

        # Patch the decoder forward to be ONNX-trace-safe
        self._patch_decoder_for_trace()

        if mixed_precision:
            self._setup_mixed_precision()
        elif precision == "bf16":
            # Decoder weights already loaded as bf16 by from_pretrained;
            # just ensure that's the case (no-op cast for safety).
            self.decoder.to(torch.bfloat16)
        elif precision == "bf16_mixed":
            self._setup_bf16_mixed()
        elif precision == "bf16_layer_fp16":
            self._setup_bf16_layer_fp16()
        elif precision == "bf16_matmul_fp16":
            self._setup_bf16_matmul_fp16()
        elif precision == "bf16_mlp_fp16":
            self._setup_bf16_mlp_fp16()
        elif precision == "fp16_attn_safe":
            self._setup_fp16_attn_safe()

    # ---- internal helpers ----

    def _replace_lambdas(self) -> None:
        for seq in (self.decoder.proj_in, self.decoder.proj_out):
            for i, mod in enumerate(seq):
                if type(mod).__name__ == "Lambda":
                    seq[i] = _Transpose12()

    def _setup_mixed_precision(self) -> None:
        """Convert bulk of model to fp16, keep precision-critical ops in fp32.

        The AdaLN pattern (scale_shift_table + temb -> scale/shift/gate)
        and RMSNorm are numerically sensitive. In pure fp16, the gate
        values get slightly wrong, and over 24 layers the error compounds
        multiplicatively (0.92^24 ~ 7x dampening). Keeping these ops in
        fp32 while running attention/MLP in fp16 gives near-full accuracy
        with most of the fp16 speedup.
        """
        decoder = self.decoder

        # Convert everything to fp16 first
        decoder.half()

        # Force fp32 for precision-critical paths:

        # 1. Timestep embedding (sinusoidal encoding + projection)
        decoder.time_embed.float()
        decoder.time_embed_r.float()

        # 2. Output AdaLN: scale_shift_table, norm_out
        decoder.scale_shift_table = nn.Parameter(
            decoder.scale_shift_table.data.float()
        )
        decoder.norm_out.float()

        # 3. Per-layer AdaLN: scale_shift_table + RMSNorm (all 3 norms)
        # Note: condition_embedder stays fp16 so encoder_hidden_states
        # match Q dtype in cross-attention (SDPA requires same dtype).
        for layer in decoder.layers:
            layer.scale_shift_table = nn.Parameter(
                layer.scale_shift_table.data.float()
            )
            layer.self_attn_norm.float()
            layer.mlp_norm.float()
            if hasattr(layer, "cross_attn_norm"):
                layer.cross_attn_norm.float()

    def _setup_fp16_attn_safe(self) -> None:
        """fp16 mixed precision (2B recipe) + ALL self_attn modules in fp32.

        XL turbo's hidden states reach ~30 000 in absolute magnitude (vs
        2B's ~5 000), so Q@K^T intermediates inside self-attention can
        exceed fp16's 65 504 ceiling in any layer, not just the q_norm
        outliers. PyTorch's fused SDPA hides the overflow with fp32
        accumulation, but TRT's unrolled SDPA writes fp16 intermediates
        and NaNs.

        Solution: wrap every self_attn in _AttnFp32Wrapper. The rest of
        the model (MLP, cross-attn, AdaLN, norms, projections) stays on
        the proven 2B fp16 + fp32 islands recipe so we keep TRT's mature
        fp16 kernel path for everything except self-attention.

        Cross-attention is left in fp16 because its q_norm/k_norm scales
        are well-behaved across all XL layers (max ~2) and its inputs
        are encoder hidden states which don't grow through the layer
        cycle.
        """
        # First apply the standard fp16 mixed-precision recipe (decoder.half(),
        # fp32 islands for timestep / AdaLN / RMSNorms).
        self._setup_mixed_precision()

        # Then wrap every self_attn in fp32.
        decoder = self.decoder
        for i, layer in enumerate(decoder.layers):
            layer.self_attn = _AttnFp32Wrapper(layer.self_attn)

        logger.info(
            "fp16_attn_safe: wrapped all %d self_attn modules in fp32 "
            "(cross_attn stays in fp16)",
            len(decoder.layers),
        )

    def _setup_bf16_mixed(self) -> None:
        """bf16 bulk + minimal fp32 island for the XL turbo decoder.

        Keep everything in bf16 (the model's training dtype, full
        precision/range for this model) and only add an fp32 island
        around the proj_out ConvTranspose1d. TRT has no bf16 kernel for
        that specific deconv shape ([2560, 64, 2, 1]), so without the
        island the strongly-typed bf16 build dies with "No matching
        rules found for input operand types".

        Unlike the 2B fp16_mixed recipe, no AdaLN/norm fp32 islands are
        needed: bf16 has the same exponent range as fp32 and the q_norm
        overflow that bites fp16 doesn't bite bf16.
        """
        decoder = self.decoder

        # Make sure everything is bf16 to start (no-op if the model was
        # loaded with dtype="bfloat16", but defensive).
        decoder.to(torch.bfloat16)

        # Patch unembedding: wrap the inner ConvTranspose1d so trace
        # records cast(bf16->fp32) -> deconv(fp32) -> cast(fp32->bf16).
        # proj_out is nn.Sequential(Lambda, ConvTranspose1d, Lambda); the
        # Lambda -> _Transpose12 replacement already happened earlier in
        # _replace_lambdas, so we look for the ConvTranspose1d by type.
        deconv_idx = None
        for i, mod in enumerate(decoder.proj_out):
            if isinstance(mod, nn.ConvTranspose1d):
                deconv_idx = i
                break
        if deconv_idx is None:
            raise RuntimeError(
                "bf16_mixed: could not find ConvTranspose1d inside proj_out"
            )
        decoder.proj_out[deconv_idx] = _Fp32CastWrapper(decoder.proj_out[deconv_idx])

    # XL turbo: layer indices that can run their bodies in fp16 without
    # input or output overflowing.
    #
    # Determined empirically by running the bf16 forward on real cover
    # inputs and recording per-layer absmax of the running residual.
    # The pattern looks like:
    #
    #     layer 0..13:    residual ranges 4 288 -> 33 856 (fp16-safe)
    #     layer 14..28:   residual peaks at 190 464 (FP16 OVERFLOW)
    #     layer 29..31:   residual settles back to 9 856 -> 16 256
    #
    # A layer can be wrapped in fp16 only if BOTH its input residual
    # AND its output residual fit in fp16 (the wrapper does an input
    # cast to fp16). Layers 29..31 produce small outputs but RECEIVE
    # the ~85 000 bf16 residual from layer 28, which overflows their
    # input cast and cascades NaN through the rest of the forward.
    # So the safe set is layers 0..13 only (14 of 32).
    XL_FP16_SAFE_LAYERS = tuple(range(0, 14))

    # XL turbo: list of (parent_module_attr, linear_attr_name) pairs that
    # are safe to wrap in fp16 inside each AceStepDiTLayer. Inputs to
    # these Linears are bounded by post-RMSNorm output * AdaLN scale,
    # which fits comfortably in fp16. Outputs feed into residual adds
    # that stay in bf16 outside the wrapper.
    XL_LINEAR_WRAP_TARGETS = (
        ("self_attn", "q_proj"),
        ("self_attn", "k_proj"),
        ("self_attn", "v_proj"),
        ("self_attn", "o_proj"),
        ("cross_attn", "q_proj"),
        ("cross_attn", "k_proj"),
        ("cross_attn", "v_proj"),
        ("cross_attn", "o_proj"),
        ("mlp", "gate_proj"),
        ("mlp", "up_proj"),
        ("mlp", "down_proj"),
    )

    def _setup_bf16_mlp_fp16(self) -> None:
        """Hybrid: bf16 residual stream + fp16 MLP body in every DiT layer.

        Builds on _setup_bf16_mixed (bf16 bulk + fp32 proj_out island),
        then wraps every layer's `mlp` submodule with _MlpFp16Wrapper.
        Cast surface is 32 in + 32 out = 64 casts total, well within
        TRT's compilation tractability (compared to 704 casts for the
        per-Linear wrap which segfaults the builder).

        MLP is roughly 60-65 % of per-layer compute for these decoder
        shapes (intermediate=9728 vs hidden=2560, three big GEMMs vs
        attention's 2-3 smaller ones), so swapping just MLPs to fp16
        recovers most of the bf16 throughput penalty without disturbing
        attention or the residual stream.
        """
        self._setup_bf16_mixed()

        decoder = self.decoder
        wrapped = 0
        for layer in decoder.layers:
            if hasattr(layer, "mlp"):
                layer.mlp = _MlpFp16Wrapper(layer.mlp)
                wrapped += 1
        logger.info(
            "bf16_mlp_fp16: wrapped %d/%d layer MLPs in fp16",
            wrapped, len(decoder.layers),
        )

    def _setup_bf16_matmul_fp16(self) -> None:
        """Hybrid recipe: bf16 residual stream + per-matmul fp16 weights/compute.

        Builds on _setup_bf16_mixed (bf16 bulk + fp32 proj_out island),
        then wraps every nn.Linear inside every DiT layer with
        _LinearFp16Wrapper. The wrapper casts the Linear's input to fp16,
        runs the matmul with fp16 weights, then casts the output back
        to bf16 before it flows into the residual addition.

        This is the matmul-grain version of bf16_layer_fp16. The
        layer-grain wrapper failed because TRT's optimizer fused across
        the bf16<->fp16 cast boundary in ways that corrupted the
        precision intent. The matmul-grain wrapper limits TRT's fusion
        radius to a single matmul + its two casts, which empirically
        survives compilation.

        Profiling shows linear/matmul accounts for 77 % of the bf16
        engine's runtime, with bf16 GEMM kernels running ~40 % slower
        than the equivalent fp16 GEMMs on Blackwell. Swapping these to
        fp16 should recover most of that gap without changing the
        residual stream's dtype, which has to stay bf16 to handle XL
        turbo's ~190 000 peak residual magnitude.
        """
        self._setup_bf16_mixed()

        decoder = self.decoder
        wrapped = 0
        for layer in decoder.layers:
            for parent_attr, lin_attr in self.XL_LINEAR_WRAP_TARGETS:
                parent = getattr(layer, parent_attr, None)
                if parent is None:
                    continue
                lin = getattr(parent, lin_attr, None)
                if lin is None:
                    continue
                setattr(parent, lin_attr, _LinearFp16Wrapper(lin))
                wrapped += 1
        logger.info(
            "bf16_matmul_fp16: wrapped %d Linears across %d layers",
            wrapped, len(decoder.layers),
        )

    def _setup_bf16_layer_fp16(self) -> None:
        """Hybrid recipe: bf16 residual stream + per-layer fp16 matmuls.

        Builds on _setup_bf16_mixed (bf16 bulk + fp32 proj_out island),
        then additionally wraps the safe XL turbo layers (those whose
        residual stays under fp16's 65 504 ceiling) with _LayerFp16Wrapper
        so their internal matmuls run in fp16 while the residual stream
        between layers stays in bf16.

        The cast-in / cast-out pattern produces a trace where ~53 % of
        the DiT's layer compute uses fp16 tensor cores (faster on
        Blackwell), while the remaining 47 % stays in bf16 to handle
        the residual magnitude. End-to-end speedup vs pure bf16_mixed
        is roughly proportional to the fraction of fp16 layers times
        the bf16/fp16 throughput gap.
        """
        # First do everything bf16_mixed does (bf16 bulk + proj_out fp32 island).
        self._setup_bf16_mixed()

        decoder = self.decoder
        n_layers = len(decoder.layers)
        safe = [i for i in self.XL_FP16_SAFE_LAYERS if i < n_layers]
        for i in safe:
            decoder.layers[i] = _LayerFp16Wrapper(decoder.layers[i])

        logger.info(
            "bf16_layer_fp16: wrapped %d/%d layers in fp16 (indices %s); "
            "remaining %d layers stay bf16",
            len(safe), n_layers, safe, n_layers - len(safe),
        )

    def _patch_decoder_for_trace(self) -> None:
        """Monkey-patch the decoder forward to be ONNX-trace-safe.

        Fixes three trace-hostile patterns in the stock forward():

          1. GQA in SDPA: transformers passes ``enable_gqa=True`` to
             ``F.scaled_dot_product_attention`` when num_key_value_groups > 1
             and attention_mask is None.  The ONNX exporter cannot convert
             this.  We monkey-patch ``use_gqa_in_sdpa`` to return False so
             the SDPA path falls back to ``repeat_kv`` (head expansion via
             ``repeat_interleave``), which is fully traceable.

          2. Shape-dependent Python branches: the original forward captures
             ``original_seq_len = shape[1]`` as a Python int (baked constant
             in ONNX) and uses ``if pad_length > 0`` (baked branch).  We
             remove padding/cropping entirely; the caller must ensure
             seq_len is a multiple of patch_size (=2, i.e. even).

          3. ``create_4d_mask()`` builds shape-dependent masks that bake
             traced dimensions.  Replaced with inline tensor ops for the
             sliding window mask (bidirectional, ``|i-j| <= window``).
             Full attention layers get ``None`` (is_causal=False on the
             module means SDPA treats None as bidirectional).
        """
        import types

        # --- Fix GQA: disable enable_gqa in SDPA for ONNX traceability ---
        # When use_gqa_in_sdpa returns False, the transformers SDPA function
        # manually expands K/V heads via repeat_kv (repeat_interleave) instead
        # of passing enable_gqa=True.  repeat_interleave traces cleanly.
        import transformers.integrations.sdpa_attention as _sdpa_mod
        _sdpa_mod.use_gqa_in_sdpa = lambda *args, **kwargs: False

        decoder = self.decoder
        sliding_window = decoder.config.sliding_window  # 128
        layer_types = decoder.config.layer_types  # list of "full_attention"/"sliding_attention"

        def _export_forward(
            self_dec,
            hidden_states,
            timestep,
            timestep_r,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            context_latents,
            use_cache=None,
            past_key_values=None,
            cache_position=None,
            position_ids=None,
            output_attentions=False,
            return_hidden_states=None,
            custom_layers_config=None,
            enable_early_exit=False,
            **flash_attn_kwargs,
        ):
            # Timestep embeddings
            temb_t, timestep_proj_t = self_dec.time_embed(timestep)
            temb_r, timestep_proj_r = self_dec.time_embed_r(timestep - timestep_r)
            temb = temb_t + temb_r
            timestep_proj = timestep_proj_t + timestep_proj_r

            # Concatenate context
            hidden_states = torch.cat([context_latents, hidden_states], dim=-1)

            # No padding or cropping.  seq_len must be a multiple of
            # patch_size (=2).  This avoids shape-dependent Python branches
            # that bake constants into the ONNX graph.

            # proj_in (patch embedding: Conv1d stride=2 halves seq_len)
            hidden_states = self_dec.proj_in(hidden_states)
            encoder_hidden_states = self_dec.condition_embedder(encoder_hidden_states)

            # Position IDs / embeddings
            seq_len_pat = hidden_states.shape[1]
            cache_position = torch.arange(seq_len_pat, device=hidden_states.device)
            position_ids = cache_position.unsqueeze(0)
            position_embeddings = self_dec.rotary_emb(hidden_states, position_ids)

            # Sliding window mask: bidirectional, |i-j| <= window.
            # Uses tensor ops (arange, abs, where) so ONNX can trace them.
            # Full attention layers get None (is_causal=False on the module
            # means SDPA treats None as fully bidirectional).
            indices = cache_position  # [seq_len_pat]
            diff = indices.unsqueeze(0) - indices.unsqueeze(1)  # [S, S]
            sw_mask = torch.where(
                torch.abs(diff) <= sliding_window,
                torch.zeros(1, device=hidden_states.device, dtype=hidden_states.dtype),
                torch.full((1,), torch.finfo(hidden_states.dtype).min, device=hidden_states.device, dtype=hidden_states.dtype),
            )
            sw_mask = sw_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

            # Layer loop: static branching on layer_types (config, not runtime)
            for i, layer_module in enumerate(self_dec.layers):
                attn_mask = sw_mask if layer_types[i] == "sliding_attention" else None
                layer_outputs = layer_module(
                    hidden_states,
                    position_embeddings,
                    timestep_proj,
                    attn_mask,
                    position_ids,
                    None,   # past_key_values
                    False,  # output_attentions
                    False,  # use_cache
                    cache_position,
                    encoder_hidden_states,
                    None,   # encoder_attention_mask
                )
                hidden_states = layer_outputs[0]

            # Output AdaLN + proj_out (ConvTranspose1d stride=2 doubles seq_len)
            shift, scale = (self_dec.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
            hidden_states = (self_dec.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)
            hidden_states = self_dec.proj_out(hidden_states)

            return (hidden_states, None)

        decoder.forward = types.MethodType(_export_forward, decoder)

    # ---- forward ----

    def forward(
        self,
        hidden_states: torch.Tensor,       # [B, T, 64]
        timestep: torch.Tensor,            # [B]
        encoder_hidden_states: torch.Tensor,  # [B, L_enc, 2048]
        context_latents: torch.Tensor,     # [B, T, 128]
    ) -> torch.Tensor:
        outputs = self.decoder(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            context_latents=context_latents,
            use_cache=False,
            past_key_values=None,
            output_attentions=False,
        )
        return outputs[0]  # velocity [B, T, 64]


# ------------------------------------------------------------------
# ONNX export
# ------------------------------------------------------------------

@dataclass
class OnnxExportConfig:
    """Configuration for ONNX export."""

    # Trace input sizes (should be "typical" values)
    batch_size: int = 1
    seq_len: int = 750       # 30s at 25 Hz, must be even
    enc_len: int = 200       # typical encoder seq len

    opset_version: int = 17
    do_constant_folding: bool = True

    # Mixed precision: export with fp16 bulk + fp32 for AdaLN/timestep/norm.
    # Use with TRTBuildConfig.strongly_typed=True for best FP16 accuracy.
    # Recommended for 2B turbo. Ignored when ``precision`` is set.
    mixed_precision: bool = False

    # Trace dtype for the wrapper. Used when ``mixed_precision`` is False.
    # One of: "fp32" (default), "bf16". For XL turbo, use "bf16" to
    # match the training dtype and avoid fp16 overflow in attention
    # intermediates from layers with large q_norm/k_norm scales.
    precision: str = "fp32"

    # When True, disables ONNX constant folding to preserve PyTorch
    # parameter names as ONNX initializer names.  Required for TRT
    # REFIT so the refitter can address weights by their original names.
    # Without this, nn.Linear weights get auto-generated names like
    # "onnx__MatMul_12882" that can't be mapped back to LoRA targets.
    for_refit: bool = False


def export_decoder_onnx(
    model,
    onnx_path: Union[str, Path],
    device: str = "cuda",
    config: Optional[OnnxExportConfig] = None,
) -> Path:
    """Export the decoder to ONNX with dynamic shapes.

    Args:
        model: AceStepConditionGenerationModel (the full model, we extract .decoder).
        onnx_path: Where to write the .onnx file.
        device: Device for tracing ("cuda" or "cpu").
        config: Export configuration.  Defaults are fine for most cases.

    Returns:
        Path to the written ONNX file.
    """
    if config is None:
        config = OnnxExportConfig()

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    decoder = model.decoder
    wrapper = DecoderForExport(
        decoder,
        mixed_precision=config.mixed_precision,
        precision=config.precision,
    ).eval()

    if config.mixed_precision:
        # Mixed precision: model already has fp16/fp32 regions set up.
        # Move to device without changing dtypes.
        wrapper = wrapper.to(device)
        trace_dtype = torch.float16
        ts_dtype = torch.float32  # time_embed is fp32 in mixed mode
        logger.info("Exporting with mixed precision (fp16 bulk + fp32 critical ops)")
    elif config.precision == "fp16_attn_safe":
        # 2B-style fp16 mixed + extra fp32 islands around outlier
        # attention layers (XL turbo's q_norm/k_norm overflow sites).
        wrapper = wrapper.to(device)
        trace_dtype = torch.float16
        ts_dtype = torch.float32
        logger.info("Exporting fp16_attn_safe (fp16 mixed + per-layer attn fp32 islands)")
    elif config.precision == "bf16":
        # bf16 throughout: weights are already bf16, just move to device.
        wrapper = wrapper.to(device)
        trace_dtype = torch.bfloat16
        ts_dtype = torch.bfloat16  # time_embed runs in bf16
        logger.info("Exporting in bf16 (matches training dtype)")
    elif config.precision == "bf16_mixed":
        # bf16 bulk + minimal fp32 island around proj_out's deconv.
        # Trace inputs are bf16; timestep also bf16 (no fp32 islands
        # outside the deconv wrapper).
        wrapper = wrapper.to(device)
        trace_dtype = torch.bfloat16
        ts_dtype = torch.bfloat16
        logger.info("Exporting bf16 mixed (bf16 bulk + fp32 deconv island)")
    elif config.precision == "bf16_layer_fp16":
        # bf16 residual stream + per-layer fp16 matmul wrappers (XL turbo
        # hybrid). Inputs to the engine are bf16 because that's what the
        # cross-layer residual stream uses.
        wrapper = wrapper.to(device)
        trace_dtype = torch.bfloat16
        ts_dtype = torch.bfloat16
        logger.info("Exporting bf16_layer_fp16 (bf16 residual + per-layer fp16 matmuls)")
    elif config.precision == "bf16_matmul_fp16":
        # bf16 residual stream + per-Linear fp16 cast wrappers around
        # every matmul inside the DiT layers. Tighter cast surface than
        # bf16_layer_fp16; survives TRT compilation reliably.
        wrapper = wrapper.to(device)
        trace_dtype = torch.bfloat16
        ts_dtype = torch.bfloat16
        logger.info("Exporting bf16_matmul_fp16 (bf16 residual + per-Linear fp16 matmuls)")
    elif config.precision == "bf16_mlp_fp16":
        # bf16 residual stream + per-MLP fp16 wrappers (32 cast pairs total)
        wrapper = wrapper.to(device)
        trace_dtype = torch.bfloat16
        ts_dtype = torch.bfloat16
        logger.info("Exporting bf16_mlp_fp16 (bf16 residual + per-MLP fp16 wrappers)")
    else:
        # Full fp32 export
        wrapper = wrapper.float().to(device)
        trace_dtype = torch.float32
        ts_dtype = torch.float32

    B = config.batch_size
    T = config.seq_len
    L = config.enc_len

    example_inputs = (
        torch.randn(B, T, 64, device=device, dtype=trace_dtype),
        torch.full((B,), 0.5, device=device, dtype=ts_dtype),
        torch.randn(B, L, 2048, device=device, dtype=trace_dtype),
        torch.randn(B, T, 128, device=device, dtype=trace_dtype),
    )

    input_names = [
        "hidden_states",
        "timestep",
        "encoder_hidden_states",
        "context_latents",
    ]
    output_names = ["velocity"]

    dynamic_axes = {
        "hidden_states":          {0: "batch", 1: "seq_len"},
        "timestep":               {0: "batch"},
        "encoder_hidden_states":  {0: "batch", 1: "enc_len"},
        "context_latents":        {0: "batch", 1: "seq_len"},
        "velocity":               {0: "batch", 1: "seq_len"},
    }

    # For refit-enabled builds, disable constant folding to preserve
    # weight names as ONNX initializer names.  TRT does its own constant
    # folding internally, so this has no effect on engine quality.
    do_constant_folding = config.do_constant_folding
    if config.for_refit:
        do_constant_folding = False
        logger.info("REFIT mode: constant folding disabled to preserve weight names")

    # The legacy torchscript-based ONNX exporter (dynamo=False) has a bug
    # in its shape-type inference pass when tracing bf16 graphs: it produces
    # complex tensors during constant folding, then fails with
    # "ScalarType ComplexDouble is an unexpected tensor scalar type". The
    # new dynamo-based exporter (torch.export) doesn't have this bug.
    # Use dynamo for any bf16-containing trace; keep legacy for fp16 mixed
    # and fp32.
    use_dynamo = (
        not config.mixed_precision
        and config.precision in ("bf16", "bf16_mixed", "bf16_layer_fp16", "bf16_matmul_fp16", "bf16_mlp_fp16")
    )

    logger.info(
        "Tracing decoder for ONNX export (T=%d, L=%d, exporter=%s) ...",
        T, L, "dynamo" if use_dynamo else "torchscript",
    )

    with torch.no_grad():
        if use_dynamo:
            # Dynamo path: pass tensors as positional args, use dynamic_shapes
            # API instead of dynamic_axes. Dynamo requires opset >= 18; the
            # downconversion to 17 fails for some ops, so don't force a
            # version here (let dynamo pick its native default).
            from torch.export import Dim
            batch = Dim("batch", min=1, max=8)
            seq = Dim("seq", min=126, max=1500)
            enc = Dim("enc", min=32, max=512)
            dynamic_shapes = {
                "hidden_states":         {0: batch, 1: seq},
                "timestep":              {0: batch},
                "encoder_hidden_states": {0: batch, 1: enc},
                "context_latents":       {0: batch, 1: seq},
            }
            torch.onnx.export(
                wrapper,
                example_inputs,
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_shapes=dynamic_shapes,
                dynamo=True,
            )
        else:
            torch.onnx.export(
                wrapper,
                example_inputs,
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=config.opset_version,
                do_constant_folding=do_constant_folding,
                dynamo=False,
            )

    # The ONNX file may exceed the 2GB protobuf limit since the decoder
    # is ~6GB.  This is fine:
    #   - OnnxRuntime uses its own parser (not protobuf) and handles it
    #   - TRT's OnnxParser.parse_from_file also handles large inline ONNX
    # The onnx Python library's load() cannot read >2GB files, which is
    # why the previous external_data conversion produced 0-byte files.
    # We skip it and rely on the native parsers.

    size_mb = onnx_path.stat().st_size / (1 << 20)
    logger.info("ONNX saved to %s (%.1f MB)", onnx_path, size_mb)
    return onnx_path


# ------------------------------------------------------------------
# TensorRT engine build
# ------------------------------------------------------------------

@dataclass
class TRTBuildConfig:
    """Configuration for TensorRT engine build."""

    fp16: bool = True
    bf16: bool = False          # TRT 9.0+ on Ampere/Hopper
    tf32: bool = True           # TF32 for fp32 accumulation kernels

    workspace_gb: float = 4.0

    # Dynamic shape profiles: (min, optimal, max) per axis
    batch_min: int = 1
    batch_opt: int = 1
    batch_max: int = 4

    seq_min: int = 126          # ~5s, even
    seq_opt: int = 750          # 30s
    seq_max: int = 1500         # 60s

    enc_min: int = 32
    enc_opt: int = 200
    enc_max: int = 512

    # Builder optimization level (0-5, higher = slower build, faster engine)
    builder_optimization_level: int = 3

    # When True, TRT respects the dtypes in the ONNX graph exactly.
    # Use with mixed-precision ONNX export to ensure fp32 regions
    # (timestep embedding, AdaLN, norms) stay in fp32 while
    # attention/MLP run in fp16.
    strongly_typed: bool = False

    # Enable weight refitting.  Allows updating engine weights at runtime
    # via trt.Refitter without rebuilding.  Required for dynamic LoRA.
    # Slight engine size increase; negligible performance impact.
    refit: bool = False

    # DiT variant name, included in engine filename when not "turbo"
    # so engines from different checkpoints coexist in the same directory.
    variant: str = "turbo"

    @property
    def max_duration_s(self) -> int:
        """Max duration in seconds, derived from seq_max at 25Hz."""
        return self.seq_max // 25

    def engine_filename(self) -> str:
        """Generate a standardized engine filename from build config.

        Format: decoder_{variant}_{precision}[_refit]_b{batch_max}_{duration}s.engine
        The variant tag is omitted for "turbo" (backward compat).
        Uses seconds so naming is stable across frame rates.
        """
        if self.strongly_typed:
            prec = "mixed"
        elif self.bf16:
            prec = "bf16"
        elif self.fp16:
            prec = "fp16"
        else:
            prec = "fp32"
        refit_tag = "_refit" if self.refit else ""
        dur = self.max_duration_s
        # Include variant in name for non-turbo models
        variant_tag = f"_{self.variant}" if self.variant != "turbo" else ""
        return f"decoder{variant_tag}_{prec}{refit_tag}_b{self.batch_max}_{dur}s.engine"


def build_trt_engine(
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path],
    config: Optional[TRTBuildConfig] = None,
) -> Path:
    """Parse ONNX and build a TensorRT engine with dynamic shapes.

    Args:
        onnx_path: Path to the ONNX model.
        engine_path: Where to write the serialized TRT engine.
        config: Build configuration.

    Returns:
        Path to the written engine file.
    """
    import tensorrt as trt

    if config is None:
        config = TRTBuildConfig()

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)

    # Network creation flags
    net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if config.strongly_typed and hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        net_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        logger.info("Using STRONGLY_TYPED network (precision from ONNX graph)")

    network = builder.create_network(net_flags)
    parser = trt.OnnxParser(network, trt_logger)

    logger.info("Parsing ONNX from %s ...", onnx_path)
    # Use parse_from_file so TRT resolves external data relative to the ONNX path
    onnx_abs = str(onnx_path.resolve())
    if not parser.parse_from_file(onnx_abs):
        for i in range(parser.num_errors):
            logger.error("ONNX parse error: %s", parser.get_error(i))
        raise RuntimeError("ONNX parsing failed")

    logger.info(
        "Network: %d inputs, %d outputs, %d layers",
        network.num_inputs, network.num_outputs, network.num_layers,
    )

    # Builder config
    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(config.workspace_gb * (1 << 30)),
    )

    # Precision flags. STRONGLY_TYPED mode forbids FP16/BF16 flags
    # (TRT enforces this with an API error: kBF16 must not be set when
    # strongly_typed). The dtypes are baked into the ONNX graph instead.
    # TF32 is still allowed under strongly_typed.
    if not config.strongly_typed:
        if config.fp16:
            build_config.set_flag(trt.BuilderFlag.FP16)
        if config.bf16 and hasattr(trt.BuilderFlag, "BF16"):
            build_config.set_flag(trt.BuilderFlag.BF16)

    if config.tf32:
        build_config.set_flag(trt.BuilderFlag.TF32)

    if config.refit:
        build_config.set_flag(trt.BuilderFlag.REFIT)
        logger.info("REFIT enabled: engine weights can be updated at runtime")

    if hasattr(build_config, "builder_optimization_level"):
        build_config.builder_optimization_level = config.builder_optimization_level

    # Optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()

    Bmin, Bopt, Bmax = config.batch_min, config.batch_opt, config.batch_max
    Smin, Sopt, Smax = config.seq_min, config.seq_opt, config.seq_max
    Emin, Eopt, Emax = config.enc_min, config.enc_opt, config.enc_max

    profile.set_shape(
        "hidden_states",
        min=(Bmin, Smin, 64), opt=(Bopt, Sopt, 64), max=(Bmax, Smax, 64),
    )
    profile.set_shape(
        "timestep",
        min=(Bmin,), opt=(Bopt,), max=(Bmax,),
    )
    profile.set_shape(
        "encoder_hidden_states",
        min=(Bmin, Emin, 2048), opt=(Bopt, Eopt, 2048), max=(Bmax, Emax, 2048),
    )
    profile.set_shape(
        "context_latents",
        min=(Bmin, Smin, 128), opt=(Bopt, Sopt, 128), max=(Bmax, Smax, 128),
    )

    build_config.add_optimization_profile(profile)

    logger.info(
        "Building TRT engine (fp16=%s, bf16=%s, opt_level=%d) ...",
        config.fp16, config.bf16, config.builder_optimization_level,
    )
    logger.info(
        "  Profiles: B=[%d,%d,%d]  T=[%d,%d,%d]  L_enc=[%d,%d,%d]",
        Bmin, Bopt, Bmax, Smin, Sopt, Smax, Emin, Eopt, Emax,
    )

    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = engine_path.stat().st_size / (1 << 20)
    logger.info("Engine saved to %s (%.1f MB)", engine_path, size_mb)
    return engine_path


# ------------------------------------------------------------------
# Validation helper
# ------------------------------------------------------------------

@torch.no_grad()
def validate_trt_vs_pytorch(
    model,
    engine_path: Union[str, Path],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seq_len: int = 750,
    enc_len: int = 200,
    seed: int = 42,
) -> dict:
    """Compare TRT decoder output against PyTorch decoder output.

    Returns a dict with per-element statistics so you can gauge accuracy.
    """
    from .runtime import TRTDecoder

    torch.manual_seed(seed)
    B = 1

    hidden_states = torch.randn(B, seq_len, 64, device=device, dtype=dtype)
    timestep = torch.tensor([0.75], device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(B, enc_len, 2048, device=device, dtype=dtype)
    context_latents = torch.randn(B, seq_len, 128, device=device, dtype=dtype)

    # PyTorch reference
    model.decoder.eval()
    with torch.no_grad():
        pt_out = model.decoder(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            context_latents=context_latents,
            use_cache=False,
        )[0]

    # TRT
    trt_decoder = TRTDecoder(engine_path)
    trt_out = trt_decoder(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        context_latents=context_latents,
    )

    # Compare
    diff = (pt_out.float() - trt_out.float()).abs()
    rel_diff = diff / (pt_out.float().abs() + 1e-8)

    results = {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "pt_mean": pt_out.float().mean().item(),
        "trt_mean": trt_out.float().mean().item(),
        "pt_std": pt_out.float().std().item(),
        "trt_std": trt_out.float().std().item(),
        "cosine_sim": torch.nn.functional.cosine_similarity(
            pt_out.float().flatten().unsqueeze(0),
            trt_out.float().flatten().unsqueeze(0),
        ).item(),
    }

    logger.info("Validation results:")
    for k, v in results.items():
        logger.info("  %-20s: %.6f", k, v)

    return results
