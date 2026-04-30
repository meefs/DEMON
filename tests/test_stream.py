"""Tests for the Session.stream() graph handle (Phase 3 streaming API).

Validates the streaming contract after the ``SessionStream`` class was
dissolved: ``Session.stream()`` returns a ``StreamHandle`` that wraps a
persistent ``StreamDenoise`` node. Every test drives the handle via
``handle.tick(drain=True, ...)`` which runs one submit+drain cycle on
the underlying ring-buffer pipeline. Each test gets a fresh handle to
avoid cross-test state leakage.

Run:  uv run pytest tests/test_stream.py -v
"""

import pytest
import torch

from acestep.engine.session import StreamHandle
from acestep.nodes.types import Latent

SAMPLE_RATE = 48000
FRAMES_PER_SEC = 25


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stream(session, prepared_source, conditioning):
    """Fresh stream handle per test."""
    return session.stream(
        source=prepared_source,
        conditioning=conditioning,
        steps=8,
        shift=3.0,
    )


def _drain(stream, **kwargs) -> Latent:
    """Run one submit+drain cycle and return the finished latent."""
    result = stream.tick(drain=True, **kwargs)
    assert result is not None, "drain returned no latent"
    return result


# ---------------------------------------------------------------------------
# Creation and properties
# ---------------------------------------------------------------------------

class TestStreamCreation:

    def test_returns_stream_handle(self, stream):
        assert isinstance(stream, StreamHandle)

    def test_stats_has_backend(self, stream):
        # Stats are empty until the first tick builds the pipeline.
        stream.tick(denoise=0.5, seed=1, drain=True)
        s = stream.stats()
        assert isinstance(s, dict)
        assert "backend" in s

    def test_source_latent_shape(self, stream, prepared_source):
        sl = stream.source.latent.tensor
        assert isinstance(sl, torch.Tensor)
        T = prepared_source.latent.tensor.shape[1]
        assert sl.shape[1] == T
        assert sl.shape[2] == 64

    def test_source_latent_on_device(self, stream):
        assert stream.source.latent.tensor.is_cuda


# ---------------------------------------------------------------------------
# Submit + tick lifecycle
# ---------------------------------------------------------------------------

class TestStreamGeneration:

    def test_submit_and_tick_produces_latent(self, stream):
        result = _drain(stream, denoise=0.5, seed=42)
        assert isinstance(result, Latent)
        assert result.tensor.ndim == 3

    def test_output_shape_matches_source(self, stream, prepared_source):
        result = _drain(stream, denoise=0.5, seed=42)
        assert result.tensor.shape == prepared_source.latent.tensor.shape

    def test_deterministic_seed(self, session, prepared_source, conditioning):
        outputs = []
        for _ in range(2):
            s = session.stream(
                source=prepared_source,
                conditioning=conditioning,
                steps=8, shift=3.0,
            )
            out = _drain(s, denoise=0.5, seed=42)
            outputs.append(out.tensor.clone())
        assert torch.allclose(outputs[0], outputs[1], atol=1e-4)

    def test_different_seeds_differ(self, stream):
        a = _drain(stream, denoise=0.5, seed=1)
        b = _drain(stream, denoise=0.5, seed=2)
        assert not torch.allclose(a.tensor, b.tensor)

    def test_multiple_submissions_all_complete(self, stream):
        results = [
            _drain(stream, denoise=0.5, seed=100 + i)
            for i in range(4)
        ]
        assert len(results) == 4


# ---------------------------------------------------------------------------
# Denoise control
# ---------------------------------------------------------------------------

class TestStreamDenoise:

    def test_low_denoise_close_to_source(self, stream, prepared_source):
        out = _drain(stream, denoise=0.1, seed=42)
        src = prepared_source.latent.tensor.float()
        mse = (out.tensor.float() - src.to(out.tensor.device)).pow(2).mean().item()
        assert mse < 0.5

    def test_full_denoise_far_from_source(self, stream, prepared_source):
        out = _drain(stream, denoise=1.0, seed=42)
        src = prepared_source.latent.tensor.float()
        mse = (out.tensor.float() - src.to(out.tensor.device)).pow(2).mean().item()
        assert mse > 0.01


# ---------------------------------------------------------------------------
# SDE denoise curves
# ---------------------------------------------------------------------------

class TestStreamSDE:

    def test_flat_curve_produces_result(self, stream):
        T = stream.source.latent.tensor.shape[1]
        curve = torch.full((1, T, 1), 0.5, dtype=torch.float32)
        out = _drain(stream, denoise=1.0, seed=42, sde_denoise_curve=curve)
        assert out is not None

    def test_sine_curve_produces_result(self, stream):
        T = stream.source.latent.tensor.shape[1]
        t = torch.linspace(0, 1, T).unsqueeze(0).unsqueeze(-1)
        curve = 0.5 * (0.5 + 0.5 * torch.sin(2 * 3.14159 * 4 * t))
        out = _drain(stream, denoise=1.0, seed=42, sde_denoise_curve=curve)
        assert out is not None

    def test_different_curves_produce_different_output(
        self, session, prepared_source, conditioning
    ):
        T = prepared_source.latent.tensor.shape[1]
        outputs = []
        for amp in [0.3, 0.9]:
            s = session.stream(
                source=prepared_source,
                conditioning=conditioning,
                steps=8, shift=3.0,
            )
            curve = torch.full((1, T, 1), amp, dtype=torch.float32)
            out = _drain(s, denoise=1.0, seed=42, sde_denoise_curve=curve)
            outputs.append(out.tensor.clone())
        assert not torch.allclose(outputs[0], outputs[1])


# ---------------------------------------------------------------------------
# Source latent override (feedback pattern)
# ---------------------------------------------------------------------------

class TestStreamFeedback:

    def test_custom_source_latents_accepted(self, stream):
        sl = stream.source.latent.tensor
        noisy = sl + torch.randn_like(sl) * 0.1
        out = _drain(
            stream, denoise=0.5, seed=42,
            source_latent=Latent(tensor=noisy),
        )
        assert out is not None

    def test_feedback_changes_output(self, session, prepared_source, conditioning):
        outputs = []
        for feedback_val in [0.0, 0.5]:
            s = session.stream(
                source=prepared_source,
                conditioning=conditioning,
                steps=8, shift=3.0,
            )
            first = _drain(s, denoise=0.5, seed=42).tensor.clone()
            if feedback_val > 0:
                src = s.source.latent.tensor
                blended = (1.0 - feedback_val) * src + feedback_val * first
                second = _drain(
                    s, denoise=0.5, seed=42,
                    source_latent=Latent(tensor=blended),
                )
            else:
                second = _drain(s, denoise=0.5, seed=42)
            outputs.append(second.tensor.clone())
        assert not torch.allclose(outputs[0], outputs[1])


# ---------------------------------------------------------------------------
# Shift control
# ---------------------------------------------------------------------------

class TestStreamShift:

    def test_shift_flows_to_pipeline(self, stream):
        # Shift is a hot-updatable widget param: pass it on a tick and
        # verify it lands on the underlying pipeline's config.
        stream.tick(denoise=0.5, seed=1, shift=5.0, drain=True)
        assert abs(stream.pipeline.config.shift - 5.0) < 1e-6

    def test_different_shift_changes_output(
        self, session, prepared_source, conditioning
    ):
        outputs = []
        for shift in [1.5, 5.0]:
            s = session.stream(
                source=prepared_source,
                conditioning=conditioning,
                steps=8, shift=shift,
            )
            out = _drain(s, denoise=0.5, seed=42)
            outputs.append(out.tensor.clone())
        assert not torch.allclose(outputs[0], outputs[1])


# ---------------------------------------------------------------------------
# Decode integration
# ---------------------------------------------------------------------------

class TestStreamDecode:

    def test_stream_output_decodable(self, session, stream):
        out = _drain(stream, denoise=0.5, seed=42)
        audio = session.decode(out)
        assert audio.sample_rate == SAMPLE_RATE
        rms = audio.waveform.float().pow(2).mean().sqrt().item()
        assert rms > 1e-4
