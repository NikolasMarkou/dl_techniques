"""Pins the TWO divergent AdaLN modulation broadcast contracts in this package.

Two functions spelled ``modulate`` / ``_modulate`` survive in the tree and they
are **not** the same function:

* ``layers/transformers/sd3_adaln.py::modulate`` — module level, PUBLIC. Takes
  ``h`` of shape ``(B, N, D)`` and ``shift``/``scale`` of shape ``(B, D)``, and
  performs the ``(B, 1, D)`` ``expand_dims`` ITSELF. The helper owns the
  broadcast. It was named ``_modulate`` until
  ``plan-2026-08-31-a4e0c303/iter-1/step-2.1`` promoted it: it has a
  cross-package consumer, so a leading underscore was a false privacy marker.
* ``models/vision_language/sd3_mmdit/blocks.py::_modulate`` — deleted by
  ``plan-2026-08-31-a4e0c303/iter-1/step-2``; ``blocks.py`` now imports the
  ``sd3_adaln`` one. This file is the guard that the surviving implementation
  still has the contract ``blocks.py`` was written against.
* ``layers/transformers/adaln_zero.py::AdaLNZeroConditionalBlock._modulate`` —
  a ``@staticmethod`` with NO ``expand_dims``: the CALLER owns the broadcast
  (its two call sites pass ``(B, T, D)``-shaped chunks). It is deliberately NOT
  merged with the other two, and this file pins that divergence so a future
  reader does not "finish the job" and silently change broadcasting.

Both oracles below are computed in numpy from first principles; neither calls
the subject to derive its own expected value.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.transformers import sd3_adaln
from dl_techniques.layers.transformers.adaln_zero import (
    AdaLNZeroConditionalBlock,
)

# ---------------------------------------------------------------------
# ``B == N`` on purpose: under plain numpy broadcasting a ``(B, D)`` operand
# against a ``(B, N, D)`` tensor aligns on the TRAILING axes, i.e. it is read as
# ``(1, B, D)`` and lands on the SEQUENCE axis. With ``B != N`` that raises;
# with ``B == N`` it silently computes the wrong tensor. The square case is the
# discriminating one, so it is the one the oracle uses.
# ---------------------------------------------------------------------
B = 4
N = 4
D = 3


@pytest.fixture
def operands():
    rng = np.random.default_rng(20260831)
    h = rng.normal(size=(B, N, D)).astype("float32")
    shift = rng.normal(size=(B, D)).astype("float32")
    scale = rng.normal(size=(B, D)).astype("float32")
    return h, shift, scale


def _oracle_helper_broadcasts(h, shift, scale):
    """``(B, D)`` conditioning applied per-SAMPLE, constant along ``N``."""
    out = np.empty_like(h)
    for b in range(h.shape[0]):
        for n in range(h.shape[1]):
            for d in range(h.shape[2]):
                out[b, n, d] = h[b, n, d] * (1.0 + scale[b, d]) + shift[b, d]
    return out


def _oracle_caller_broadcasts(h, shift, scale):
    """No ``expand_dims``: ``(B, D)`` aligns on the trailing axes, i.e. on N."""
    out = np.empty_like(h)
    for b in range(h.shape[0]):
        for n in range(h.shape[1]):
            for d in range(h.shape[2]):
                out[b, n, d] = h[b, n, d] * (1.0 + scale[n, d]) + shift[n, d]
    return out


class TestSd3AdaLNModulateOwnsTheBroadcast:
    """``sd3_adaln.modulate`` must expand ``(B, D)`` onto the SAMPLE axis."""

    def test_matches_the_per_sample_numpy_oracle(self, operands):
        h, shift, scale = operands
        got = keras.ops.convert_to_numpy(
            sd3_adaln.modulate(
                keras.ops.convert_to_tensor(h),
                keras.ops.convert_to_tensor(shift),
                keras.ops.convert_to_tensor(scale),
            )
        )
        np.testing.assert_allclose(
            got, _oracle_helper_broadcasts(h, shift, scale), atol=1e-6
        )

    def test_is_not_the_caller_broadcast_form(self, operands):
        """Anti-vacuity: the two oracles must actually disagree here."""
        h, shift, scale = operands
        per_sample = _oracle_helper_broadcasts(h, shift, scale)
        per_step = _oracle_caller_broadcasts(h, shift, scale)
        assert np.max(np.abs(per_sample - per_step)) > 1e-2, (
            "the two broadcast contracts coincide on this fixture, so the "
            "test above cannot discriminate between them"
        )
        got = keras.ops.convert_to_numpy(
            sd3_adaln.modulate(
                keras.ops.convert_to_tensor(h),
                keras.ops.convert_to_tensor(shift),
                keras.ops.convert_to_tensor(scale),
            )
        )
        assert np.max(np.abs(got - per_step)) > 1e-2

    def test_non_square_conditioning_is_accepted(self):
        """``B != N`` must work — it is exactly what the caller passes."""
        rng = np.random.default_rng(7)
        h = rng.normal(size=(2, 5, D)).astype("float32")
        shift = rng.normal(size=(2, D)).astype("float32")
        scale = rng.normal(size=(2, D)).astype("float32")
        got = keras.ops.convert_to_numpy(
            sd3_adaln.modulate(
                keras.ops.convert_to_tensor(h),
                keras.ops.convert_to_tensor(shift),
                keras.ops.convert_to_tensor(scale),
            )
        )
        assert got.shape == (2, 5, D)
        np.testing.assert_allclose(
            got, _oracle_helper_broadcasts(h, shift, scale), atol=1e-6
        )


class TestSd3MMDiTBlocksUsesTheSharedHelper:
    """``blocks.py`` must not re-define ``modulate``; it imports the owner."""

    def test_blocks_modulate_is_the_sd3_adaln_object(self):
        from dl_techniques.models.vision_language.sd3_mmdit import blocks

        assert blocks.modulate is sd3_adaln.modulate


class TestAdaLNZeroStaticMethodDoesNotBroadcast:
    """The third copy has a DIFFERENT contract and is deliberately kept."""

    def test_it_is_the_caller_broadcast_form(self, operands):
        h, shift, scale = operands
        got = keras.ops.convert_to_numpy(
            AdaLNZeroConditionalBlock._modulate(
                keras.ops.convert_to_tensor(h),
                keras.ops.convert_to_tensor(shift),
                keras.ops.convert_to_tensor(scale),
            )
        )
        np.testing.assert_allclose(
            got, _oracle_caller_broadcasts(h, shift, scale), atol=1e-6
        )

    def test_it_does_not_agree_with_the_sd3_helper(self, operands):
        h, shift, scale = operands
        args = (
            keras.ops.convert_to_tensor(h),
            keras.ops.convert_to_tensor(shift),
            keras.ops.convert_to_tensor(scale),
        )
        static = keras.ops.convert_to_numpy(
            AdaLNZeroConditionalBlock._modulate(*args)
        )
        shared = keras.ops.convert_to_numpy(sd3_adaln.modulate(*args))
        assert np.max(np.abs(static - shared)) > 1e-2, (
            "the staticmethod and the sd3_adaln helper agree — either one was "
            "merged onto the other's contract, or this fixture cannot tell "
            "them apart"
        )

    def test_it_refuses_non_square_conditioning(self):
        """Proof the caller really does own the broadcast: ``(2, D)`` cannot
        be applied to a ``(2, 5, D)`` tensor without an explicit expand.

        The oracle is the MESSAGE, not the type. Measured on the shipped
        TensorFlow backend the multiply raises
        ``tensorflow.python.framework.errors_impl.InvalidArgumentError:
        required broadcastable shapes [Op:Mul]`` -- a type that inherits from
        ``OpError``, not from ``ValueError``, and that other Keras 3 backends do
        not raise at all (numpy/jax report the same condition as a plain
        ``ValueError``). So both plausible types are accepted and the substring
        ``broadcast`` carries the discrimination. Do NOT widen this back to a
        bare ``pytest.raises(Exception)``: that also accepts an ``ImportError``,
        an ``AttributeError`` from a renamed staticmethod or a ``TypeError``
        from a changed signature, none of which is the refusal this arm claims
        to prove.
        """
        rng = np.random.default_rng(11)
        h = keras.ops.convert_to_tensor(
            rng.normal(size=(2, 5, D)).astype("float32")
        )
        shift = keras.ops.convert_to_tensor(
            rng.normal(size=(2, D)).astype("float32")
        )
        scale = keras.ops.convert_to_tensor(
            rng.normal(size=(2, D)).astype("float32")
        )
        with pytest.raises(
            (tf.errors.InvalidArgumentError, ValueError), match="broadcast"
        ):
            keras.ops.convert_to_numpy(
                AdaLNZeroConditionalBlock._modulate(h, shift, scale)
            )
