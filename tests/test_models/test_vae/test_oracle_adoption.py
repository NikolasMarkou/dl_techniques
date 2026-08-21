"""
Oracle adoption for ``models/vae`` -- Phase 5 batch C.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored and no ``src/`` file is added.

THE FORWARD IS NON-DETERMINISTIC AT ``training=False``, AND EVERY COMPARISON
HERE IS BUILT AROUND THAT
-----------------------------------------------------------------------------
A VAE samples ``z ~ N(z_mean, exp(z_log_var/2))`` on every call, including at
``training=False``. Iteration-2 step 19 measured the consequence on this exact
package: **the same model called twice differs from ITSELF by 1.118e-01, while a
full ``.keras`` save/load round trip differs by 9.678e-02** -- the self-vs-self
noise is LARGER than the thing a round-trip test is trying to detect. Any
output-comparison instrument pointed at ``reconstruction`` or ``z`` on this
model is therefore measuring the draw.

So every comparison in this file extracts ``z_mean``, which is a deterministic
function of the input, or decodes ``z_mean`` explicitly. That is asserted, not
assumed: :class:`TestTheSampledForwardIsADraw` measures both the self-vs-self
delta on ``reconstruction`` and the EXACT self-vs-self equality on ``z_mean``,
so a reader who sees a large delta here has a test naming the reason instead of
a defect report to file, and a future change that made the forward deterministic
would fail loudly rather than silently making this file's care pointless.

Measured 2026-08-21, one Adam step, ramp loss, at
``latent_dim=8 / input_shape=(32, 32, 3) / depths=2 / filters=[8, 16]``:

===============================  ==========  ======
arm                              weights     dead
===============================  ==========  ======
VAE (dict output)                60          0
===============================  ==========  ======

Step 19.2 gave this package's ``train_step`` its ``scale_loss`` (measured ratio
1.0020) and repaired a gradient clip that was clipping in the SCALED domain.
Nothing here re-litigates either; a red in this file is not either of those.

``dropout_rate`` is pinned to ``0.0`` and every build is seeded. That is on top
of the sampling non-determinism, not instead of it -- seeding fixes the WEIGHTS,
not the per-call ``keras.random.normal`` draw inside ``Sampling``.
"""

from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vae.model import VAE, create_vae

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
)
from ..precision_arm_oracle import _asymmetric_loss, flatten_tensors
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

LATENT_DIM = 8
SHAPE = (32, 32, 3)
BUILD_SEED = 0

#: Measured 2026-08-21, one Adam step, ramp loss.
GF_WEIGHTS = 60

#: Measured 2026-08-21: the decoder owns exactly this many TRAINABLE weights,
#: and they are exactly the ones a loss built from ``z_mean``/``z_log_var``
#: alone cannot reach.
DECODER_TRAINABLE = 28

#: The ONLY output key that is a deterministic function of the input.
#: Every knob comparison in this file extracts it -- see the module docstring.
DETERMINISTIC_KEY = "z_mean"


def ramp_loss(outputs: Any) -> Any:
    """IMPORTED from ``precision_arm_oracle``, never re-typed (D-059)."""
    return sum(_asymmetric_loss(t) for t in flatten_tensors(outputs))


def _images(batch: int = 2, shape=SHAPE, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((batch,) + shape).astype("float32")


def _vae(**o) -> VAE:
    kwargs: Dict[str, Any] = dict(
        latent_dim=LATENT_DIM, input_shape=SHAPE, depths=2, steps_per_depth=1,
        filters=[8, 16], dropout_rate=0.0,
    )
    kwargs.update(o)
    return VAE(**kwargs)


def _built(build_fn=_vae, shape=SHAPE, seed: int = BUILD_SEED) -> VAE:
    keras.utils.set_random_seed(seed)
    model = build_fn()
    model(_images(1, shape), training=False)
    return model


def _one_adam_step(model: keras.Model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = ramp_loss(model(inputs, training=True))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, variables) if g is not None]
    )


class TestTheSampledForwardIsADraw:
    """The trap, pinned first, because every comparison below depends on it."""

    def test_the_reconstruction_differs_from_itself_across_two_calls(self):
        """The false CRITICAL. This is NOT a broken model.

        Step 19 measured 1.118e-01 self-vs-self here against a 9.678e-02
        round-trip delta: the noise is bigger than the signal an output-equality
        test would be looking for.
        """
        model = _built()
        x = _images()
        a = keras.ops.convert_to_numpy(model(x, training=False)["reconstruction"])
        b = keras.ops.convert_to_numpy(model(x, training=False)["reconstruction"])
        delta = float(np.max(np.abs(a - b)))
        assert delta > 1e-4, (
            f"the sampled reconstruction is REPRODUCIBLE across two calls "
            f"(max|delta| = {delta:.3e}). Either the sampling layer stopped "
            f"drawing, or this file's whole justification for extracting "
            f"z_mean everywhere is now obsolete -- check which before "
            f"deleting anything."
        )

    def test_z_mean_is_bit_identical_across_two_calls(self):
        """The discriminating half, and the reason ``z_mean`` is safe to
        compare: it is the ENCODER's output, upstream of the draw."""
        model = _built()
        x = _images()
        a = keras.ops.convert_to_numpy(model(x, training=False)[DETERMINISTIC_KEY])
        b = keras.ops.convert_to_numpy(model(x, training=False)[DETERMINISTIC_KEY])
        np.testing.assert_array_equal(a, b)

    def test_decoding_z_mean_is_also_bit_identical(self):
        """``decode(z_mean)`` is the deterministic reconstruction path, and the
        one any A/B on this package must use."""
        model = _built()
        x = _images()
        z_mean, _ = model.encode(x)
        a = keras.ops.convert_to_numpy(model.decode(z_mean))
        b = keras.ops.convert_to_numpy(model.decode(z_mean))
        np.testing.assert_array_equal(a, b)


class TestVAEGradientFlow:

    def test_no_layer_is_stochastic(self):
        """Dropout only. The SAMPLING is deliberate and is pinned above."""
        model = _built()
        stochastic = [
            (layer.name, attr, getattr(layer, attr))
            for layer in model._flatten_layers(include_self=False)
            for attr in ("rate", "drop_path_rate", "dropout_rate")
            if isinstance(getattr(layer, attr, None), float)
            and getattr(layer, attr) > 0.0
        ]
        assert stochastic == [], f"a non-zero dropout rate is live: {stochastic}"

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _built()
        x = _images()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=ramp_loss)

        assert len(report) == GF_WEIGHTS == len(model.trainable_weights)

    def test_the_encoder_is_live_through_z_mean_alone(self):
        """The reparameterisation trick's whole point, asserted.

        If ``z`` were produced by a draw that did NOT carry ``z_mean`` and
        ``z_log_var`` on the backward graph, the decoder would still train, the
        reconstruction would still improve, and the ENCODER would be frozen --
        with every shape and finiteness check green. A loss made only of
        ``z_mean`` and ``z_log_var`` reaches the encoder or nothing does.
        """
        model = _built()
        x = _images()
        _one_adam_step(model, x)

        def encoder_only(outputs):
            return (_asymmetric_loss(outputs["z_mean"])
                    + _asymmetric_loss(outputs["z_log_var"]))

        report = assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=encoder_only,
            # The decoder is downstream of z and cannot be reached from a loss
            # built only out of the encoder's own outputs. Named EXACTLY, as a
            # set, rather than waived by a broad pattern.
            expect_zero=tuple(sorted(_decoder_trainable_paths(model))),
        )
        assert len(report) == GF_WEIGHTS
        assert len(_decoder_trainable_paths(model)) == DECODER_TRAINABLE, (
            f"the decoder's trainable-weight count moved from "
            f"{DECODER_TRAINABLE}; re-derive the waiver rather than widening it"
        )

    def test_the_gradient_assertion_can_fail(self):
        model = _built()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _images(), loss_fn=ramp_loss)


def _decoder_trainable_paths(model: VAE):
    """Every TRAINABLE weight path owned by the decoder sub-model.

    DERIVED from ``model.decoder``, never typed out as a literal list. Two
    reasons, both learned the hard way in this batch:

    * a hand-written list rots the moment the decoder gains a layer, and
      ``expect_zero``'s unmatched-pattern check would then fail with a stale
      name rather than the real story;
    * this package's weight paths are only TWO segments deep
      (``decoder_projection/kernel``) because the encoder and decoder are
      Functional sub-models with explicit layer names -- so the "take everything
      after the first slash" suffix rule the rest of this batch uses would
      reduce every path to ``kernel`` / ``bias`` / ``gamma`` / ``beta`` and
      waive the ENTIRE model. That is exactly what a first draft of this test
      did, and the oracle's ``live_but_waived`` clause is what caught it.

    Non-trainable weights (BatchNorm moving statistics) are excluded: they never
    appear in a gradient report, and an ``expect_zero`` pattern matching nothing
    is itself an error.
    """
    decoder_weights = {id(w) for w in model.decoder.weights}
    return {
        w.path for w in model.trainable_weights if id(w) in decoder_weights
    }


class TestVAEKnobSensitivity:

    def test_latent_dim_changes_the_parameterisation(self):
        builders = {
            d: (lambda d=d: _built(lambda: _vae(latent_dim=d)))
            for d in (4, 8, 16)
        }
        assert_structural_knob_changes_weights(builders, knob="latent_dim")

    def test_filters_change_the_parameterisation(self):
        builders = {
            f: (lambda f=f: _built(lambda: _vae(filters=list(f))))
            for f in ((8, 16), (16, 32), (32, 64))
        }
        assert_structural_knob_changes_weights(builders, knob="filters")

    def test_variant_changes_the_parameterisation(self):
        builders = {
            v: (lambda v=v: _built(
                lambda: create_vae(input_shape=SHAPE, latent_dim=LATENT_DIM,
                                   variant=v)))
            for v in ("micro", "small", "medium")
        }
        assert_structural_knob_changes_weights(builders, knob="variant")

    def test_activation_reaches_the_forward_pass(self):
        """A VALUE knob, compared on ``z_mean`` ONLY.

        Compared on ``reconstruction`` this assertion would pass on a build
        that dropped the kwarg entirely, because the sampling draw alone moves
        the output by ~1.1e-01 -- three orders of magnitude above the
        instrument's floor. That is the trap this whole file is arranged
        around, and this is the test where it would have bitten.
        """
        x = _images()
        builders = {
            a: (lambda a=a: _vae(activation=a)) for a in ("leaky_relu", "relu")
        }
        deltas = assert_value_knob_changes_output(
            builders, x, knob="activation",
            extract=lambda o: o[DETERMINISTIC_KEY])
        assert all(d > 1e-4 for d in deltas.values()), deltas

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _built()), "b": (lambda: _built())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="latent_dim")

    def test_the_value_knob_assertion_can_fail_on_the_deterministic_key(self):
        """And the SAME inert pair passed through ``reconstruction`` does NOT
        fail -- which is the measurement that justifies ``extract`` above."""
        x = _images()
        builders = {k: (lambda: _vae(activation="leaky_relu"))
                    for k in ("a", "b")}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_value_knob_changes_output(
                builders, x, knob="activation",
                extract=lambda o: o[DETERMINISTIC_KEY])

        # Same builders, same seed, same knob value -- but read through the
        # SAMPLED output, where the draw alone satisfies the assertion.
        deltas = assert_value_knob_changes_output(
            builders, x, knob="activation",
            extract=lambda o: o["reconstruction"])
        assert all(d > 1e-4 for d in deltas.values()), (
            f"the sampled output no longer masks an inert knob: {deltas}. "
            f"If the forward became deterministic, `extract` can be relaxed -- "
            f"but check TestTheSampledForwardIsADraw first."
        )


class TestVAESmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _built()
        x = _images()

        def contract(out):
            assert isinstance(out, dict), (
                f"VAE returns a dict, got {type(out)}")
            assert set(out) == {"z", "z_mean", "z_log_var", "reconstruction"}, (
                f"unexpected key set {sorted(out)}")
            assert tuple(out["z_mean"].shape) == (x.shape[0], LATENT_DIM), (
                f"z_mean is (batch, latent_dim); got {tuple(out['z_mean'].shape)}")
            assert tuple(out["reconstruction"].shape) == (x.shape[0],) + SHAPE, (
                f"the reconstruction must match the input shape; got "
                f"{tuple(out['reconstruction'].shape)}")
            for key in out:
                assert_finite(out[key])

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }

    def test_the_reconstruction_lands_in_the_sigmoid_range(self):
        """``final_activation`` defaults to ``sigmoid``, so the reconstruction
        is a [0, 1] image. A build that dropped the activation would produce
        unbounded values with an identical shape."""
        model = _built()
        recon = np.asarray(keras.ops.convert_to_numpy(
            model(_images(), training=False)["reconstruction"]))
        assert recon.min() >= 0.0 and recon.max() <= 1.0, (
            f"reconstruction range is {recon.min()} .. {recon.max()}, not [0, 1]")

    def test_a_filters_list_disagreeing_with_depths_is_refused(self):
        with pytest.raises(ValueError):
            _vae(depths=3, filters=[8, 16])
