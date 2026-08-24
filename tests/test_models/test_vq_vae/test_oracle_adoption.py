"""
Oracle adoption for ``models/vq_vae`` -- Phase 5 batch C.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored and no ``src/`` file is added.

THE CODEBOOK IS *NOT* ON THE RECONSTRUCTION'S BACKWARD GRAPH, BY DESIGN
------------------------------------------------------------------------
FOUND BY THIS ADOPTION, and it is the finding that shaped the file. Pointing
the gradient oracle at ``VQVAEModel``'s forward output alone reports
``vector_quantizer/embeddings`` as receiving **NO gradient at all** -- a
``None``, not a small number -- while the other 8 trainable weights are live.

That is correct and expected: VQ-VAE's straight-through estimator routes the
reconstruction gradient AROUND the quantizer to the encoder (that is what makes
an ``argmin`` trainable at all), and the codebook is moved by the codebook and
commitment terms, which the quantizer publishes through ``add_loss`` rather
than through its return value. A reader who measures ``None`` here and files a
"dead codebook" report would be reporting the architecture.

So every gradient assertion in this file uses **the loss the model actually
trains with** -- the output term PLUS ``model.losses`` -- which is what
``train_step`` builds. Under it, ``0 of 9`` are dead. Both halves are asserted
(:class:`TestTheCodebookNeedsTheAddLossTerms`), so a change that put the
codebook on the reconstruction path, or that dropped the ``add_loss`` calls,
fails a test instead of quietly changing what this file means. This is the same
shape batch B used for ``memory_bank``.

Measured 2026-08-21, one Adam step, on a 3-conv encoder / 3-conv decoder at
``(2, 8, 8, 3)``, ``num_embeddings=16``, ``embedding_dim=8``:

===================================  =========  =====================
loss                                 weights    dead
===================================  =========  =====================
ramp(output) only                    9          1 (``embeddings``, ``None``)
ramp(output) + ``model.losses``      9          0
===================================  =========  =====================

THE EMA VARIABLES CARRY ``autocast=False`` AS WELL AS ``dtype="float32"``
--------------------------------------------------------------------------
Both, and the pairing is load-bearing: Keras AUTOCASTS a float variable on READ
inside ``call``, so ``dtype="float32"`` alone leaves an EMA accumulator being
read as float16 under mixed precision. That is asserted here on all three
statistics, because the failure is invisible to every shape and finiteness
check -- the same autocast-on-read mechanism step 18.1 fixed in ``som``.

NOT A REGRESSION, AND NOT MEASURED HERE
-----------------------------------------
``test_dead_code_changes_codebook`` is PRE-EXISTING FLAKY at roughly 4.5%
(9/200 on the working tree and 9/200 at HEAD -- identical), caused by an
unseeded ``keras.random.uniform``. It lives in the LAYER suite, not this
directory. A single failure of it is not a regression of this file.
"""

from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vq_vae.model import VQVAEModel, create_vq_vae

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    gradient_report,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..precision_arm_oracle import _asymmetric_loss, flatten_tensors
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

SHAPE = (8, 8, 3)
NUM_EMBEDDINGS = 16
EMBEDDING_DIM = 8
BUILD_SEED = 0

#: Measured 2026-08-21, one Adam step.
GF_WEIGHTS = 9

#: The codebook, as a path SUFFIX -- never an absolute ``Variable.path``: Keras
#: uniquifies a model name per process, so the second ``VQVAEModel`` built in
#: one session is ``vqvae_model_1/...``. An absolute pin is green alone and red
#: behind any other test that builds the same class; it bit batch B twice.
CODEBOOK = "vector_quantizer/embeddings"


def _images(batch: int = 2, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((batch,) + SHAPE).astype("float32")


def _encoder(embedding_dim: int = EMBEDDING_DIM) -> keras.Model:
    return keras.Sequential([
        keras.layers.Input(SHAPE),
        keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        keras.layers.Conv2D(embedding_dim, 1, padding="same"),
    ], name="enc")


def _decoder(embedding_dim: int = EMBEDDING_DIM) -> keras.Model:
    return keras.Sequential([
        keras.layers.Input(SHAPE[:2] + (embedding_dim,)),
        keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        keras.layers.Conv2D(SHAPE[-1], 1, padding="same"),
    ], name="dec")


def _vq_vae(**o) -> VQVAEModel:
    embedding_dim = o.get("embedding_dim", EMBEDDING_DIM)
    kwargs: Dict[str, Any] = dict(
        encoder=_encoder(embedding_dim), decoder=_decoder(embedding_dim),
        num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM,
    )
    kwargs.update(o)
    return VQVAEModel(**kwargs)


def _built(build_fn=_vq_vae, seed: int = BUILD_SEED) -> VQVAEModel:
    keras.utils.set_random_seed(seed)
    model = build_fn()
    model(_images(1), training=False)
    return model


def ramp_loss(outputs: Any) -> Any:
    """IMPORTED from ``precision_arm_oracle``, never re-typed (D-059)."""
    return sum(_asymmetric_loss(t) for t in flatten_tensors(outputs))


def _training_loss(model: keras.Model):
    """The loss ``VQVAEModel.train_step`` actually optimises.

    ``model.losses`` holds the quantizer's codebook and commitment terms, which
    the forward RETURN VALUE does not carry. A closure over ``model`` is needed
    because ``gradient_report`` hands the loss function only the outputs -- and
    ``model.losses`` must be read INSIDE the tape, after the forward, or it is
    the previous call's list.
    """

    def loss_fn(outputs: Any) -> Any:
        extra = model.losses
        return ramp_loss(outputs) + (sum(extra) if extra else 0.0)

    return loss_fn


def _one_adam_step(model: keras.Model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = _training_loss(model)(outputs)
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, variables) if g is not None]
    )


class TestTheCodebookNeedsTheAddLossTerms:
    """The finding, pinned two-sided, before anything depends on it."""

    def test_the_reconstruction_alone_leaves_the_codebook_disconnected(self):
        """The false CRITICAL: a ``None`` gradient on a healthy codebook.

        The straight-through estimator routes the reconstruction gradient
        AROUND the quantizer on purpose. Asserted as an EXACT one-element set,
        not as "at least the codebook", so a genuinely dead encoder or decoder
        weight cannot hide behind this explanation.
        """
        model = _built()
        report = gradient_report(model, _images(), loss_fn=ramp_loss)
        disconnected = {p for p, v in report.items() if v is None}
        assert {p.split("/", 1)[1] for p in disconnected} == {CODEBOOK}, (
            f"expected exactly the codebook to be off the reconstruction's "
            f"backward graph, got {sorted(disconnected)}"
        )

    def test_the_quantizer_publishes_exactly_two_add_loss_terms(self):
        """The premise: the codebook and commitment terms. If these stop being
        published, the model trains a frozen codebook and every shape and
        finiteness check stays green."""
        model = _built()
        model(_images(), training=True)
        assert len(model.losses) == 2, (
            f"expected the codebook + commitment terms, got "
            f"{len(model.losses)} add_loss term(s)")

    def test_under_the_training_loss_the_codebook_is_live(self):
        """The discriminating half."""
        model = _built()
        x = _images()
        _one_adam_step(model, x)
        report = gradient_report(model, x, loss_fn=_training_loss(model))
        path = next(p for p in report if p.endswith(CODEBOOK))
        assert report[path] is not None and report[path] > 0.0, (
            f"the codebook is dead even under the training loss "
            f"(max|grad|={report[path]}) -- the add_loss explanation is then "
            f"wrong and this IS a disconnected codebook"
        )


class TestVQVAEGradientFlow:

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _built()
        x = _images()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=_training_loss(model))

        assert len(report) == GF_WEIGHTS == len(model.trainable_weights)

    def test_the_gradient_assertion_can_fail(self):
        model = _built()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _images(), loss_fn=_training_loss(model))


class TestVQVAEEmaStatisticsAreNotAutocast:
    """``dtype="float32"`` alone is NOT enough, and both halves are asserted.

    Keras autocasts a float variable on READ inside ``call``. An EMA
    accumulator declared float32 but left autocasting is read as float16 under
    ``mixed_float16`` -- and nothing about the shape, the finiteness or the
    reconstruction changes. Step 18.1 fixed exactly this mechanism in ``som``.
    """

    @staticmethod
    def _ema_variables(model: VQVAEModel):
        return [
            w for w in model.quantizer.weights
            if w.path.rsplit("/", 1)[-1] != "embeddings"
        ]

    def test_ema_adds_statistics_and_only_under_ema(self):
        plain = _built()
        ema = _built(lambda: _vq_vae(use_ema=True))
        assert self._ema_variables(plain) == []
        assert len(self._ema_variables(ema)) > 0, (
            "use_ema=True created no EMA statistic; the kwarg is not reaching "
            "the quantizer")

    def test_every_ema_statistic_is_float32_and_not_autocast(self):
        model = _built(lambda: _vq_vae(use_ema=True))
        variables = self._ema_variables(model)
        assert variables, "no EMA statistic to inspect"
        # Keras 3 exposes the flag privately (`Variable._autocast`); there is no
        # public accessor. The premise is asserted first so this test dies
        # loudly if the attribute is ever renamed, instead of silently reading
        # `None` and waving every variable through -- which is exactly what a
        # first draft of it did.
        assert all(hasattr(w, "_autocast") for w in variables), (
            "keras.Variable no longer exposes `_autocast`; this test is now "
            "reading nothing and must be rewritten, not deleted")
        offenders = [
            (w.path, str(w.dtype), w._autocast)
            for w in variables
            if str(w.dtype) != "float32" or w._autocast
        ]
        assert offenders == [], (
            f"an EMA statistic is not float32-and-not-autocast: {offenders}. "
            f"dtype alone is insufficient -- Keras autocasts a float variable "
            f"on READ inside call()."
        )

    def test_the_ema_arm_still_trains_every_gradient_weight(self):
        """Under EMA the codebook is updated by the EMA rule rather than by a
        gradient, so the trainable set SHRINKS. Asserted rather than assumed."""
        model = _built(lambda: _vq_vae(use_ema=True))
        x = _images()
        _one_adam_step(model, x)
        assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=_training_loss(model))


class TestVQVAEKnobSensitivity:

    def test_num_embeddings_changes_the_parameterisation(self):
        builders = {
            n: (lambda n=n: _built(lambda: _vq_vae(num_embeddings=n)))
            for n in (8, 16, 32)
        }
        assert_structural_knob_changes_weights(builders, knob="num_embeddings")

    def test_embedding_dim_changes_the_parameterisation(self):
        builders = {
            d: (lambda d=d: _built(lambda: _vq_vae(embedding_dim=d)))
            for d in (4, 8, 16)
        }
        assert_structural_knob_changes_weights(builders, knob="embedding_dim")

    def test_use_ema_changes_the_parameterisation(self):
        builders = {
            e: (lambda e=e: _built(lambda: _vq_vae(use_ema=e)))
            for e in (False, True)
        }
        assert_structural_knob_changes_weights(builders, knob="use_ema")

    def test_commitment_cost_reaches_the_published_loss(self):
        """A VALUE knob, and one no output comparison can see.

        ``commitment_cost`` scales the commitment term ONLY. The forward return
        value -- the reconstruction -- is bit-identical between two settings, so
        ``assert_value_knob_changes_output`` on the model's output would report
        it INERT and be right about the output and wrong about the model. The
        claim is therefore made where the knob acts: on ``model.losses``.
        """
        published = {}
        recon = {}
        for cost in (0.25, 1.0):
            model = _built(lambda cost=cost: _vq_vae(commitment_cost=cost))
            out = model(_images(), training=True)
            recon[cost] = np.asarray(keras.ops.convert_to_numpy(out))
            published[cost] = [
                float(keras.ops.convert_to_numpy(t)) for t in model.losses]

        np.testing.assert_array_equal(recon[0.25], recon[1.0])  # the output IS inert
        assert published[0.25] != published[1.0], (
            f"commitment_cost is a no-op: both settings publish {published}")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _built()), "b": (lambda: _built())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(
                builders, knob="num_embeddings")


class TestVQVAESmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _built()
        x = _images()

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"VQVAEModel.call returns ONE reconstruction tensor, got "
                f"{type(out)}")
            assert tuple(out.shape) == tuple(x.shape), (
                f"the reconstruction must match the input shape; got "
                f"{tuple(out.shape)} for {tuple(x.shape)}")
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }

    def test_the_quantized_latent_is_drawn_from_the_codebook(self):
        """The claim that makes this a VQ-VAE rather than an autoencoder.

        Every quantized vector must be EXACTLY a codebook row. A quantizer that
        passed its input through unchanged would leave the reconstruction shape,
        the finiteness and the loss all perfectly healthy.
        """
        model = _built()
        x = _images()
        z_e = model.encoder(x, training=False)
        z_q = np.asarray(keras.ops.convert_to_numpy(
            model.quantizer(z_e, training=False)))
        codebook = np.asarray(keras.ops.convert_to_numpy(
            model.quantizer.embeddings))
        # The codebook is stored (D, K) or (K, D) depending on the layer; accept
        # either by matching against both orientations.
        rows = codebook if codebook.shape[-1] == EMBEDDING_DIM else codebook.T
        flat = z_q.reshape(-1, EMBEDDING_DIM)
        distances = np.abs(flat[:, None, :] - rows[None, :, :]).max(axis=-1)
        nearest = distances.min(axis=1)
        # BOUND DERIVED FROM THE DEFECT SIGNAL, not from taste. The quantizer
        # returns the straight-through form ``z_e + stop_gradient(z_q - z_e)``,
        # which is z_q only up to float32 rounding: measured worst case
        # **1.374e-05 on GPU 1** (and below 1e-5 on CPU -- a single green CPU
        # run at atol=1e-5 shipped a test that failed 2 runs in 2 on GPU). The
        # PASSTHROUGH signal this test exists to catch is the distance to the
        # nearest codebook row for an unquantised vector, measured here at
        # ~7.9e-02. 1e-3 sits 73x above the noise and 79x below the signal.
        assert float(nearest.max()) < 1e-3, (
            f"a quantized vector is {float(nearest.max()):.3e} from every "
            f"codebook row; the quantizer is passing its input through")
        # The control: without quantization the same statistic is orders of
        # magnitude larger, so the bound above is not satisfied by any vector.
        raw = np.asarray(keras.ops.convert_to_numpy(z_e)).reshape(
            -1, EMBEDDING_DIM)
        raw_nearest = np.abs(
            raw[:, None, :] - rows[None, :, :]).max(axis=-1).min(axis=1)
        assert float(raw_nearest.max()) > 1e-2, (
            f"even the UNQUANTISED encoder output is within "
            f"{float(raw_nearest.max()):.3e} of a codebook row; this test "
            f"cannot distinguish a real quantizer from a passthrough")

    def test_an_encoder_whose_width_disagrees_with_embedding_dim_is_refused(self):
        with pytest.raises(Exception):
            model = VQVAEModel(
                encoder=_encoder(EMBEDDING_DIM + 3), decoder=_decoder(),
                num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM)
            model(_images(), training=False)

    def test_the_factory_requires_an_encoder_and_a_decoder(self):
        """``create_vq_vae`` deliberately has no default backbone -- inventing
        one would silently pick an architecture the paper does not specify."""
        with pytest.raises(TypeError):
            create_vq_vae()
