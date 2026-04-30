"""Semantic hint extraction node.

Semantic hints share the LATENT wire type: they are structurally
identical to VAE latents (both ``[B, T, D]``) and flow into the same
downstream ``context_latent`` slots. Blending hints is handled by
``LatentBlend``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .base import BaseNode, NodeDefinition, NodePort, NodeRegistry
from .types import Latent, ModelHandle


@NodeRegistry.register
class SemanticExtract(BaseNode):
    """Extract semantic hints from source audio latents.

    Pre-computes the tokenizer/detokenizer representation that provides
    stable structural guidance to the decoder. This is an alternative
    to letting the model recompute hints from noisy latents at each
    diffusion step. The output is a LATENT carrying the detokenized
    hint tensor, usable anywhere a latent is accepted.
    """

    node_type_id: ClassVar[str] = "acestep.SemanticExtract"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Semantic Extract",
            category="semantic",
            description="Extract semantic structural hints from source audio latents.",
            inputs=(
                NodePort(name="model", type="MODEL"),
                NodePort(name="latent", type="LATENT"),
            ),
            outputs=(
                NodePort(name="latent", type="LATENT"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        from acestep.engine.ops import extract_semantic_hints

        model_handle: ModelHandle = kwargs["model"]
        latent: Latent = kwargs["latent"]
        handler = model_handle.handler

        with handler._load_model_context("model"):
            hints = extract_semantic_hints(handler.model, latent.tensor)

        return {"latent": Latent(tensor=hints)}
