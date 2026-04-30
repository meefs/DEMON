#!/usr/bin/env python3
"""Unit tests for block-causal attention mask."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import pytest

from experiments.self_forcing.block_causal import (
    create_block_causal_4d_mask,
    build_block_causal_mask_mapping,
)


class TestBlockCausalMask:
    def test_basic_shape(self):
        mask = create_block_causal_4d_mask(
            seq_len=100, block_size=25,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        assert mask.shape == (1, 1, 100, 100)

    def test_within_block_bidirectional(self):
        """Tokens within the same block should see each other."""
        mask = create_block_causal_4d_mask(
            seq_len=20, block_size=5,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        # Token 0 and token 4 are in block 0 -- should be visible
        assert mask[0, 0, 0, 4] == 0.0  # 0 sees 4
        assert mask[0, 0, 4, 0] == 0.0  # 4 sees 0

        # Token 2 and token 3 in block 0
        assert mask[0, 0, 2, 3] == 0.0
        assert mask[0, 0, 3, 2] == 0.0

    def test_across_blocks_causal(self):
        """Block N can see block N-1 but not block N+1."""
        mask = create_block_causal_4d_mask(
            seq_len=20, block_size=5,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        # Token 5 (block 1) should see token 0 (block 0)
        assert mask[0, 0, 5, 0] == 0.0

        # Token 0 (block 0) should NOT see token 5 (block 1)
        assert mask[0, 0, 0, 5] != 0.0  # masked (-inf)

    def test_block_boundary(self):
        """Last token of block N sees first token of block N, not block N+1."""
        mask = create_block_causal_4d_mask(
            seq_len=20, block_size=5,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        # Token 4 (last in block 0) can see token 0 (first in block 0)
        assert mask[0, 0, 4, 0] == 0.0
        # Token 4 (block 0) cannot see token 5 (block 1)
        assert mask[0, 0, 4, 5] != 0.0

    def test_later_blocks_see_all_previous(self):
        """Block 3 should see blocks 0, 1, 2, and 3."""
        mask = create_block_causal_4d_mask(
            seq_len=20, block_size=5,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        # Token 15 (block 3) should see token 0 (block 0)
        assert mask[0, 0, 15, 0] == 0.0
        # Token 15 (block 3) should see token 7 (block 1)
        assert mask[0, 0, 15, 7] == 0.0
        # Token 15 (block 3) should see token 14 (block 2)
        assert mask[0, 0, 15, 14] == 0.0

    def test_sliding_window_intersection(self):
        """Block-causal + sliding window: respects both constraints."""
        mask = create_block_causal_4d_mask(
            seq_len=100, block_size=10,
            dtype=torch.float32, device=torch.device("cpu"),
            sliding_window=15,
            is_sliding_window=True,
        )
        # Token 50 (block 5) trying to see token 0 (block 0):
        # Block-causal allows it (block 0 <= block 5), but
        # sliding window blocks it (|50-0| = 50 > 15)
        assert mask[0, 0, 50, 0] != 0.0  # masked by sliding window

        # Token 50 seeing token 40 (block 4):
        # Block-causal: allowed (block 4 <= block 5)
        # Sliding window: |50-40| = 10 <= 15: allowed
        assert mask[0, 0, 50, 40] == 0.0

    def test_padding_mask(self):
        """Padding tokens should be masked out."""
        attn_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
        mask = create_block_causal_4d_mask(
            seq_len=10, block_size=5,
            dtype=torch.float32, device=torch.device("cpu"),
            attention_mask=attn_mask,
        )
        # Token 2 seeing token 7 (padding): should be masked
        assert mask[0, 0, 2, 7] != 0.0

    def test_mask_is_additive(self):
        """Valid positions should be 0.0, masked positions should be -inf."""
        mask = create_block_causal_4d_mask(
            seq_len=10, block_size=5,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        # All values should be either 0.0 or -inf
        unique = mask.unique()
        assert 0.0 in unique
        assert len(unique) == 2  # exactly two values

    def test_build_mapping(self):
        """build_block_causal_mask_mapping returns all required keys."""
        mapping = build_block_causal_mask_mapping(
            seq_len=100, block_size=25,
            dtype=torch.float32, device=torch.device("cpu"),
            sliding_window=128,
            encoder_seq_len=50,
        )
        assert "full_attention" in mapping
        assert "sliding_attention" in mapping
        assert "encoder_attention_mask" in mapping
        assert mapping["full_attention"].shape == (1, 1, 100, 100)
        assert mapping["sliding_attention"].shape == (1, 1, 100, 100)
        assert mapping["encoder_attention_mask"].shape[2] == 100  # query dim
        assert mapping["encoder_attention_mask"].shape[3] == 50   # key dim

    def test_full_vs_sliding_differ(self):
        """Full and sliding masks should differ (sliding is more restrictive)."""
        mapping = build_block_causal_mask_mapping(
            seq_len=200, block_size=25,
            dtype=torch.float32, device=torch.device("cpu"),
            sliding_window=30,
        )
        full = mapping["full_attention"]
        sliding = mapping["sliding_attention"]
        # Sliding should mask more positions than full
        full_attend = (full == 0.0).sum()
        sliding_attend = (sliding == 0.0).sum()
        assert sliding_attend < full_attend


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
