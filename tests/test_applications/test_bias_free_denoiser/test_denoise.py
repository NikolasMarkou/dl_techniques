"""Fast CPU unit tests for the single-pass DENOISING task (no checkpoint, no GPU).

Covers the two GUI-free primitives added for the denoise task in ``main.py``:

* :func:`run_problem` on ``problem="denoise"`` — the 5-key result-dict contract, and
  the invariant that it NEVER routes through ``build_operator`` / the solver (H1).
* :func:`denoise_frame` — the per-frame primitive the streamlit-webrtc ``recv()`` loop
  reuses; shape/range/finiteness on an arbitrary (non-÷8, non-square) frame.

A tiny identity stub prior (``D(y) = y``) stands in for the real 22 MB checkpoint so
the whole module runs fast on CPU. The static ``ingest`` / ``denorm`` domain helpers
are used directly from :class:`DenoiserPrior`.
"""

import argparse

import numpy as np
import pytest

import applications.bias_free_denoiser.main as main_mod
from applications.bias_free_denoiser.main import run_problem, denoise_frame


class _IdentityPrior:
    """Minimal stub exposing ``.denoise`` as the identity ``D(y) = y``.

    Enough to exercise the denoise wiring without a Keras model or a checkpoint.
    """

    def denoise(self, y):
        return np.asarray(y)


@pytest.fixture
def stub_prior() -> _IdentityPrior:
    return _IdentityPrior()


def _denoise_args(**overrides) -> argparse.Namespace:
    """A minimal argparse namespace with the fields the denoise branch reads."""
    base = dict(noise_sigma=0.1, seed=0)
    base.update(overrides)
    return argparse.Namespace(**base)


def _target(h: int = 16, w: int = 16) -> np.ndarray:
    """A clean in-domain ``[1, h, w, 3]`` target in ``[0, 1]``."""
    return np.random.default_rng(1).uniform(0.0, 1.0, size=(1, h, w, 3)).astype(np.float32)


def test_run_problem_denoise_contract(stub_prior, monkeypatch):
    """5-key dict contract + noise added + solver path (build_operator) NEVER invoked."""
    # H1: if the denoise branch ever fell through to build_operator, this raises.
    def _boom(*_a, **_k):
        raise AssertionError("build_operator must NOT be called for denoise (H1)")

    monkeypatch.setattr(main_mod, "build_operator", _boom)

    target = _target()
    result = run_problem("denoise", stub_prior, target, _denoise_args(noise_sigma=0.1, seed=0))

    assert set(result.keys()) == {"title", "target", "degraded", "recon", "info"}
    assert result["title"] == "Denoising"
    assert result["info"] == {}

    recon = result["recon"]
    assert np.isfinite(recon).all()
    assert float(recon.min()) >= 0.0 and float(recon.max()) <= 1.0

    # noise_sigma > 0 -> the degraded (noisy) panel must differ from the clean target.
    assert not np.allclose(result["degraded"], result["target"])


def test_run_problem_denoise_as_is(stub_prior):
    """noise_sigma == 0 -> denoise-as-is: degraded == denorm(target), no noise added."""
    target = _target()
    result = run_problem("denoise", stub_prior, target, _denoise_args(noise_sigma=0.0, seed=0))
    np.testing.assert_allclose(result["degraded"], result["target"], atol=1e-6)


def test_denoise_frame_shape_range(stub_prior):
    """Arbitrary (480, 640, 3) uint8 frame -> same-HxW uint8 RGB, finite, in [0, 255]."""
    frame = np.random.default_rng(2).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    out = denoise_frame(stub_prior, frame, size=256)

    assert out.shape == (480, 640, 3)
    assert out.dtype == np.uint8
    assert np.isfinite(out).all()
    assert int(out.min()) >= 0 and int(out.max()) <= 255


@pytest.mark.parametrize("noise_sigma", [0.0, 0.1])
def test_denoiser_processor_recv_track_stable(stub_prior, noise_sigma):
    """The webrtc DenoiserProcessor.recv keeps the track resolution stable + never raises.

    Exercises the H4 (resize-back) + H5 (try/except passthrough) + optional-noise recv
    logic on a real ``av.VideoFrame`` round-trip, using the identity stub prior (no
    checkpoint). Imports ``streamlit_app`` (streamlit + streamlit_webrtc + av) — this
    module is the ONLY package file allowed those imports (H3).
    """
    import av

    from applications.bias_free_denoiser.streamlit_app import DenoiserProcessor

    proc = DenoiserProcessor(stub_prior, size=64, noise_sigma=noise_sigma)
    frame_bgr = np.random.default_rng(3).integers(0, 256, size=(48, 80, 3), dtype=np.uint8)
    in_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")

    out_frame = proc.recv(in_frame)
    out_bgr = out_frame.to_ndarray(format="bgr24")

    assert out_bgr.shape == frame_bgr.shape  # H4: track resolution unchanged.
    assert out_bgr.dtype == np.uint8
    assert np.isfinite(out_bgr).all()


def test_denoiser_processor_recv_passthrough_on_error():
    """H5: a prior whose denoise raises must NOT kill the track -> original frame back."""
    import av

    from applications.bias_free_denoiser.streamlit_app import DenoiserProcessor

    class _BoomPrior:
        def denoise(self, y):
            raise RuntimeError("simulated per-frame failure")

    proc = DenoiserProcessor(_BoomPrior(), size=64, noise_sigma=0.0)
    frame_bgr = np.random.default_rng(4).integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    in_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")

    out_bgr = proc.recv(in_frame).to_ndarray(format="bgr24")
    # The original frame is passed through unmodified (recv swallowed the exception).
    np.testing.assert_array_equal(out_bgr, frame_bgr)
