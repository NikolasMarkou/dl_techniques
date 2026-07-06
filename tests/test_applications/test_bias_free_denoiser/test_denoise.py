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
    """A clean in-domain ``[1, h, w, 3]`` target in ``[-0.5, +0.5]``."""
    return np.random.default_rng(1).uniform(-0.5, 0.5, size=(1, h, w, 3)).astype(np.float32)


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
