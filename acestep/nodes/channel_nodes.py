"""Latent channel manipulation nodes.

Provides two complementary tools for channel-level control:

  - LatentChannelScale: post-sampling scaling (between Generate and VAEDecode).
    Scales channels in the finished latent before decoding.  Zero compute cost,
    no loop modification.

  - ChannelGuidance: input scaling during denoising.  Attaches channel gain
    configs to the model handle so the diffusion engine scales specific
    channels of xt before each forward pass.  The model sees the modified
    input and responds to it (enabling emergent behaviors for some channel
    groups).  Stackable: chain multiple nodes to steer multiple ranges.

  - RemoveChannelGuidance: clears all channel guidance configs from the model.

Background: ACE-Step 1.5's VAE compresses audio into 64 latent channels.
Empirical characterization (1,600+ experiments) identified 8 functional
groups of 8 contiguous channels and 6 high-impact keystone channels.
These nodes expose that structure for creative control without baking in
perceptual interpretations.
"""

from __future__ import annotations

from typing import Any, ClassVar

from loguru import logger
import torch

from .base import BaseNode, NodeDefinition, NodeParam, NodePort, NodeRegistry

_CHANNEL_RANGE_PARAMS = (
    NodeParam(
        name="channel_start", type="integer", default=0,
        description="First channel (0-63)",
        min=0, max=63, step=1,
    ),
    NodeParam(
        name="channel_end", type="integer", default=63,
        description="Last channel (0-63, inclusive)",
        min=0, max=63, step=1,
    ),
    NodeParam(
        name="scale", type="number", default=1.0,
        description="Channel gain",
        min=0.0, max=4.0, step=0.01,
    ),
)
from .types import ChannelGuidanceEntry, Latent, ModelHandle


@NodeRegistry.register
class LatentChannelScale(BaseNode):
    """Scale specific latent channels after generation, before VAE decode.

    Multiplies channels [channel_start .. channel_end] by the given scale
    factor.  Operates on the finished latent tensor; does not affect the
    diffusion loop.

    Node parameters:
        channel_start: First channel index (0-63).
        channel_end: Last channel index (0-63, inclusive).
        scale: Multiplicative gain factor.
    """

    node_type_id: ClassVar[str] = "acestep.LatentChannelScale"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Latent Channel Scale",
            category="vae",
            description="Scale specific latent channels (post-sampling, before VAE decode).",
            inputs=(
                NodePort(name="latent", type="LATENT"),
            ),
            outputs=(
                NodePort(name="latent", type="LATENT"),
            ),
            params=_CHANNEL_RANGE_PARAMS,
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        latent: Latent = kwargs["latent"]
        ch_start = int(kwargs.get("channel_start", 0))
        ch_end = int(kwargs.get("channel_end", 63))
        scale = float(kwargs.get("scale", 1.0))

        ch_start = max(0, min(ch_start, 63))
        ch_end = max(ch_start, min(ch_end, 63))

        if scale == 1.0:
            return {"latent": latent}

        # Latent tensor is [B, T, D] where D=64
        out = latent.tensor.clone()
        out[:, :, ch_start:ch_end + 1] *= scale

        return {"latent": Latent(tensor=out, mask=latent.mask)}


@NodeRegistry.register
class ChannelGuidance(BaseNode):
    """Attach channel gain config to the model for input scaling during denoising.

    During each denoising step, the specified channels of xt are scaled
    before the forward pass.  The model sees the modified input and its
    velocity prediction reflects the perturbation.

    Stackable: chain multiple ChannelGuidance nodes to steer multiple
    channel ranges independently.  All configs are composed into a single
    gain tensor (one forward pass, zero extra cost).

    Node parameters:
        channel_start: First channel index (0-63).
        channel_end: Last channel index (0-63, inclusive).
        scale: Multiplicative gain for these channels during denoising.
    """

    node_type_id: ClassVar[str] = "acestep.ChannelGuidance"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Channel Guidance",
            category="model",
            description="Steer generation by scaling latent channels during denoising.",
            inputs=(
                NodePort(name="model", type="MODEL"),
            ),
            outputs=(
                NodePort(name="model", type="MODEL"),
            ),
            params=_CHANNEL_RANGE_PARAMS,
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        model_handle: ModelHandle = kwargs["model"]
        ch_start = int(kwargs.get("channel_start", 0))
        ch_end = int(kwargs.get("channel_end", 63))
        scale = float(kwargs.get("scale", 1.0))

        ch_start = max(0, min(ch_start, 63))
        ch_end = max(ch_start, min(ch_end, 63))

        entry = ChannelGuidanceEntry(
            channel_start=ch_start,
            channel_end=ch_end,
            scale=scale,
        )

        handler = model_handle.handler
        if not hasattr(handler, '_channel_guidance'):
            handler._channel_guidance = []
        handler._channel_guidance.append(entry)

        logger.info(
            "Channel guidance: ch %d-%d scale=%.3f (%d total configs)",
            ch_start, ch_end, scale, len(handler._channel_guidance),
        )

        return {"model": model_handle}


@NodeRegistry.register
class RemoveChannelGuidance(BaseNode):
    """Remove all channel guidance configs from the model."""

    node_type_id: ClassVar[str] = "acestep.RemoveChannelGuidance"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Remove Channel Guidance",
            category="model",
            description="Clear all channel guidance configs from the model.",
            inputs=(
                NodePort(name="model", type="MODEL"),
            ),
            outputs=(
                NodePort(name="model", type="MODEL"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        model_handle: ModelHandle = kwargs["model"]
        handler = model_handle.handler
        count = len(getattr(handler, '_channel_guidance', []))
        handler._channel_guidance = []
        if count:
            logger.info("Removed %d channel guidance configs", count)
        return {"model": model_handle}


def build_channel_gain(
    configs: list[ChannelGuidanceEntry],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Build a [1, 1, 64] gain tensor from channel guidance configs.

    Returns None if no configs or all gains are 1.0 (no-op).
    """
    if not configs:
        return None

    gain = torch.ones(1, 1, 64, device=device, dtype=dtype)
    any_non_unity = False
    for cfg in configs:
        if cfg.scale != 1.0:
            gain[0, 0, cfg.channel_start:cfg.channel_end + 1] = cfg.scale
            any_non_unity = True

    return gain if any_non_unity else None
