"""FNet's central claim: parameter-free Fourier TOKEN mixing.

Why this file exists
--------------------
FNet replaces self-attention with `y = Real(FFT2D(x))`. Because that mixer holds
no parameters, every structural test in `test_fnet/test_model.py` -- shapes,
weight counts, config round trips, gradient flow -- is satisfied by a model in
which the transform is replaced by the identity. Nothing measured whether the
token axis is mixed at all.

The instrument is `test_fftnet/test_model.py::test_mixer_actually_mixes_tokens`,
copied here for the same reason it was written there: a token mixer must NOT be
permutation-equivariant over the token axis. Its docstring's warnings are load
bearing and are respected below:

* "the output differs from the input" is vacuous -- true of any layer;
* a REVERSAL or a CYCLIC SHIFT is a false guard -- a DFT is equivariant to both
  (they become conjugation and a unit-modulus phase), so such a probe fails
  identically with and without a working mixer. The permutation used here is
  checked to be neither.

MEASURED 2026-08-18 on `FNetFourierTransform` at (2, 8, 16): permuting the
tokens and undoing the permutation moves the output by max|delta| = 6.618. The
identity injection scores exactly 0.0 -- that is the RED proof, and it is run in
the same test rather than asserted.

At MODEL level the permutation probe does not work at all (positional
embeddings already break equivariance; measured 4.163 live vs 3.055 with every
transform dead), so the second test uses cross-token dependence instead. Its
docstring carries that measurement.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.fnet_fourier_transform import FNetFourierTransform
from dl_techniques.models.fnet.model import FNet

from ..test_sam.dead_component_oracle import layer_returns_its_input


#: Not the identity, not a reversal, not any cyclic shift -- see the module
#: docstring for why each of those three would be a false guard.
PERM = np.array([3, 1, 7, 0, 5, 2, 6, 4])


def _check_permutation_is_admissible() -> None:
    n = PERM.size
    assert not np.array_equal(PERM, np.arange(n))
    assert not np.array_equal(PERM, np.arange(n)[::-1])
    assert all(
        not np.array_equal(PERM, np.roll(np.arange(n), shift)) for shift in range(n)
    ), "permutation must not be a cyclic shift"


def _equivariance_delta(call_fn, x: np.ndarray) -> float:
    """max|f(x) - unpermute(f(permute(x)))| -- zero iff f is permutation-equivariant."""
    inverse = np.argsort(PERM)
    direct = np.asarray(keras.ops.convert_to_numpy(call_fn(x)))
    permuted = np.asarray(
        keras.ops.convert_to_numpy(call_fn(x[:, PERM, :].copy()))
    )[:, inverse, :]
    return float(np.max(np.abs(direct - permuted)))


class TestFourierTransformMixesTokens:
    def test_the_transform_is_not_permutation_equivariant(self):
        _check_permutation_is_admissible()
        x = np.random.default_rng(0).random((2, 8, 16)).astype("float32")

        layer = FNetFourierTransform()
        layer.build((None, 8, 16))

        live = _equivariance_delta(lambda v: layer(v, training=False), x)
        assert live > 1e-3, (
            f"FNetFourierTransform is permutation-equivariant over tokens "
            f"(max|delta| = {live:.3e}): it performs NO token mixing -- the FFT "
            f"is being applied to the feature axis, or not at all."
        )

        # RED proof, same assertion, same test: the identity scores exactly 0.
        with layer_returns_its_input(layer, name="FNetFourierTransform"):
            dead = _equivariance_delta(lambda v: layer(v, training=False), x)
        assert dead == 0.0, (
            f"the identity injection is not equivariant either ({dead:.3e}); "
            "the probe is measuring something other than token mixing"
        )
        assert live > 1000 * max(dead, 1e-12)


class TestTokenMixingReachesTheModel:
    """The model must inherit the property, not just the layer."""

    @staticmethod
    def _model() -> FNet:
        keras.utils.set_random_seed(4)
        model = FNet(
            vocab_size=64,
            hidden_size=32,
            num_layers=2,
            intermediate_size=64,
            max_position_embeddings=16,
        )
        model(keras.ops.zeros((1, 8), dtype="int32"), training=False)
        return model

    def test_changing_one_token_moves_every_other_position(self):
        """The model-level form of the claim: cross-token dependence.

        A permutation probe is the WRONG instrument at model level: the
        positional embeddings already make FNet non-equivariant, so a
        permutation moves the output by a similar amount with the mixer live
        (max|delta| 4.163) and with every Fourier transform replaced by the
        identity (3.055) -- MEASURED 2026-08-18. That probe cannot see the
        mixer.

        What CAN: with the mixer dead every position is processed
        independently, so perturbing the token at position 0 leaves every other
        position BIT-IDENTICAL. That control is exactly 0.0, not merely small.
        """
        model = self._model()
        rng = np.random.default_rng(1)
        ids = rng.integers(0, 64, size=(2, 8)).astype("int32")
        bumped = ids.copy()
        bumped[:, 0] = (bumped[:, 0] + 7) % 64
        assert not np.array_equal(ids[:, 0], bumped[:, 0])

        def hidden(token_ids):
            out = model(keras.ops.convert_to_tensor(token_ids), training=False)
            return np.asarray(keras.ops.convert_to_numpy(out["last_hidden_state"]))

        def elsewhere_delta():
            return float(
                np.max(np.abs(hidden(ids)[:, 1:, :] - hidden(bumped)[:, 1:, :]))
            )

        live = elsewhere_delta()

        transforms = [block.fourier_transform for block in model.encoder_layers]
        assert transforms, "no Fourier transform found in the encoder"

        import contextlib

        with contextlib.ExitStack() as stack:
            for transform in transforms:
                stack.enter_context(
                    layer_returns_its_input(transform, name="fourier_transform")
                )
            dead = elsewhere_delta()

        assert dead == 0.0, (
            f"with every Fourier transform replaced by the identity, changing "
            f"token 0 still moved the other positions by {dead:.3e}; something "
            f"other than the mixer is carrying information across the token "
            f"axis, so the measurement below is not attributable to it"
        )
        assert live > 1e-3, (
            f"changing the token at position 0 moved every OTHER position by "
            f"only {live:.3e}: FNet is not mixing tokens at all (the "
            f"identity-injected control scores exactly {dead})"
        )
