"""
Oracle adoption for ``models/beit`` -- Phase 5 batch A.

Zero adoption of ``gradient_flow_oracle`` / ``knob_sensitivity_oracle`` /
``smoke_contract_oracle`` before this file. All three are adopted; no new oracle
is authored.

Adoption baseline, re-measured 2026-08-24 before the step-9 edit
---------------------------------------------------------------
Adopted in this package at that point: ``gradient_flow_oracle``,
``knob_sensitivity_oracle``, ``smoke_contract_oracle`` -- the three above, all
added by Phase 5 batch A. NOT adopted: ``lazy_build_contract_oracle`` (zero
occurrences of ``lazy_build`` anywhere under ``tests/test_models/test_beit/``),
which ``TestBeitLazyBuildContract`` at the foot of this file now adopts.

Two shared instruments are deliberately still not adopted here, because ``beit``
already reaches them CENTRALLY and a per-package copy would duplicate an
instrument rather than extend coverage: it is a registered subject in
``tests/test_models/precision_arm_subjects.py:145-152``, which feeds both the
family mixed-precision / XLA arm and ``test_roundtrip_instrument_family.py``.
That is why this file grows a lazy-build class and no fp16, XLA or round-trip
class.

The one dead weight, and why it is not a defect
-----------------------------------------------
Measured 2026-08-21 on ``create_beit_classifier('tiny', (32,32,3), 16,
num_classes=7)`` after one real optimizer step: **224** trainable weights,
**exactly one** with an identically-zero gradient --
``beit_classifier/beit_backbone/mask_token/mask_token``.

That is documented, deliberate behaviour, not a finding. ``BeitModel.build``
carries the comment "ALWAYS built -- even in the classifier, which never calls
it" (``models/beit/model.py``), and ``BeitModel.call`` applies the mask token
only when the caller passes a mask. Building it unconditionally is the fix for a
different defect entirely: a lazily-built sub-layer drops its weights on a
``.keras`` round trip.

So the waiver here is TWO-SIDED, and that is what makes it a claim rather than an
allowance. ``expect_zero`` pins the weight as dead **in the classifier**, and
``test_the_mim_head_makes_the_mask_token_live`` proves the SAME weight receives a
live gradient in the masked-image-modelling head. If someone wires masking into
the classifier, the first test fails ("the waiver is obsolete"); if someone
breaks masking in the MIM head, the second fails.
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.models.beit import (
    create_beit_classifier,
    create_beit_mim,
)

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    gradient_report,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..lazy_build_contract_oracle import assert_lazy_build_costs_nothing
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

IMG = (32, 32, 3)
PATCH = 16
NUM_PATCHES = (IMG[0] // PATCH) * (IMG[1] // PATCH)
NUM_CLASSES = 7
VOCAB = 64

#: Measured 2026-08-21, `tiny` variant at IMG/PATCH above.
GF_N_WEIGHTS_CLASSIFIER = 224

#: The single documented dead weight in the CLASSIFIER head. See the module
#: docstring: it is built unconditionally so the `.keras` round trip keeps it,
#: and it is applied only when a mask is passed -- which the classifier never
#: does. The companion test below proves it is live in the MIM head.
# DECISION plan-2026-08-19T163559-499b6f0e/D-094
# Do NOT let this waiver stand alone, and do NOT widen it. `expect_zero` on its
# own is a one-sided allowance: it says "this weight may be dead" and a second
# dead weight appearing next to it would be one line away from being absorbed.
# The two companion tests are what make it a claim --
# `test_the_mim_head_makes_the_mask_token_live` proves the SAME weight is live
# in the head that masks, and `test_the_waiver_is_not_a_blanket` proves the
# dead set is EXACTLY this one path. Deleting either leaves 224 weights guarded
# by an allowance instead of a measurement.
# See D-094 in plans/plan-2026-08-19T163559-499b6f0e/decisions.md.
CLASSIFIER_EXPECT_ZERO = ("mask_token/mask_token",)


def _images(batch: int = 2) -> np.ndarray:
    return np.random.default_rng(0).random((batch,) + IMG).astype("float32")


def _mask(batch: int = 2) -> np.ndarray:
    """A MIXED mask: patch 0 masked, the rest not.

    An all-False mask leaves `mask_token` dead even in the MIM head (`ops.where`
    selects nothing), so it would prove the opposite of what the companion test
    claims. An all-True mask would work too, but a mixed mask additionally keeps
    the unmasked path alive.
    """
    mask = np.zeros((batch, NUM_PATCHES), dtype=bool)
    mask[:, 0] = True
    return mask


def _classifier(variant: str = "tiny"):
    # `drop_path_rate=0.0` is LOAD-BEARING, not tidiness. At the shipped default
    # the backbone's stochastic depth can drop a whole encoder block on the
    # single draw the tape sees, and the oracle then reports that block's 7
    # weights dead -- a property of the DRAW, not of the model. MEASURED
    # 2026-08-21 on the MIM arm before this argument was added: 217/224 live in
    # roughly one run of four, 224/224 in the others.
    model = create_beit_classifier(variant, IMG, PATCH, num_classes=NUM_CLASSES,
                                   drop_path_rate=0.0)
    model(_images(1), training=False)
    return model


def _one_adam_step(model, inputs, loss_fn=None) -> None:
    """One REAL optimizer step, so the gradient reading is not an init reading."""
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        out = model(inputs, training=True)
        loss = default_loss(out) if loss_fn is None else loss_fn(out)
    optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))


class TestBeitGradientFlow:

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _classifier()
        x = _images()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(
            model, x, expect_zero=CLASSIFIER_EXPECT_ZERO,
        )

        assert len(report) == GF_N_WEIGHTS_CLASSIFIER == len(model.trainable_weights)

    def test_the_mim_head_makes_the_mask_token_live(self):
        """The other half of the `expect_zero` claim above.

        The classifier's dead `mask_token` is a property of the HEAD, not of the
        weight. In the MIM head, fed a mask that actually masks something, the
        same weight must receive a live gradient -- with NO waiver at all.
        """
        model = create_beit_mim("tiny", IMG, PATCH, vocab_size=VOCAB,
                                drop_path_rate=0.0)  # see `_classifier`
        inputs = (_images(), _mask())
        model(inputs, training=False)
        _one_adam_step(model, inputs)

        report = assert_gradients_reach_every_trainable_weight(model, inputs)

        mask_paths = [p for p in report if "mask_token" in p]
        assert len(mask_paths) == 1, mask_paths
        assert report[mask_paths[0]] > 0.0

    def test_the_gradient_assertion_can_fail(self):
        """RED proof: detach the forward and all 224 weights must be convicted."""
        model = _classifier()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _images(), expect_zero=CLASSIFIER_EXPECT_ZERO,
                )

    def test_the_waiver_is_not_a_blanket(self):
        """RED proof for `expect_zero` itself: without it, the suite is RED.

        A one-sided skip list is how an oracle rots. This pins that the waiver
        covers EXACTLY one weight -- drop it and the oracle convicts that weight
        by name, so the waiver cannot be silently widened to hide a second one.
        """
        model = _classifier()
        x = _images()
        _one_adam_step(model, x)
        with pytest.raises(AssertionError, match="mask_token"):
            assert_gradients_reach_every_trainable_weight(model, x)

        report = gradient_report(model, x)
        dead = [p for p, v in report.items() if v is None or v == 0.0]
        assert dead == ["beit_classifier/beit_backbone/mask_token/mask_token"]


class TestBeitKnobSensitivity:

    def test_variant_changes_the_parameterisation(self):
        """`variant` is a STRUCTURAL knob -- it re-shapes the weight set."""
        builders = {
            v: (lambda v=v: _classifier(v)) for v in ("tiny", "small", "base")
        }
        signatures = assert_structural_knob_changes_weights(builders, knob="variant")
        sizes = [
            sum(int(np.prod(s)) for s in signatures[v])
            for v in ("tiny", "small", "base")
        ]
        assert sizes == sorted(sizes), (
            f"variant is not monotone in parameter count: {sizes}"
        )

    def test_the_knob_assertion_can_fail(self):
        """RED proof: two arms at the SAME variant must be convicted as a no-op."""
        builders = {"a": (lambda: _classifier("tiny")), "b": (lambda: _classifier("tiny"))}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="variant")


class TestBeitSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _classifier()
        x = _images()

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"the classifier returns one tensor, got {type(out)}"
            )
            assert tuple(out.shape) == (x.shape[0], NUM_CLASSES), (
                f"expected {(x.shape[0], NUM_CLASSES)}, got {tuple(out.shape)}"
            )
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }


class TestBeitLazyBuildContract:
    """``lazy_build_contract_oracle`` -- does BEiT's lazy build cost anything?

    The oracle does NOT assert the contract ("is it built after ``build()``?").
    It asserts the CONSEQUENCE: perturb every float weight, prove the
    perturbation MOVED the output, then save/load and require an EXACT match at
    ``atol=0.0``. Both real lazy-build defects this tree has ever found (D-029
    ``SHGCNLinkPredictor``, D-049 ``BERT``) were found by that value comparison
    and by nothing else.

    Both heads are probed, because they materialize differently and because the
    difference is the one this package already documents. ``mask_token`` is
    built UNCONDITIONALLY -- the comment in ``BeitModel.build`` says "ALWAYS
    built -- even in the classifier, which never calls it" -- precisely so a
    lazy build cannot drop it on a ``.keras`` round trip. That is a claim about
    save/load, and until this class nothing in this package tested it: the
    ``expect_zero`` waiver above proves the weight is DEAD in the classifier,
    which is exactly the condition under which a partial materialization would
    lose it unnoticed. The MIM arm additionally puts a TWO-input model
    ``(images, mask)`` through the oracle, so the mask branch of ``build`` is
    materialized too.

    MEASURED 2026-08-24, GPU 1 (RTX 4070), ``tiny`` at ``IMG``/``PATCH`` above,
    ``drop_path_rate=0.0``:

    ============================  ==================  ==================
    ..                            classifier, 1 out   MIM head, 1 out
    ============================  ==================  ==================
    weights after one call        224                 224
    weights after ``.build()``    224                 224
    materialization ratio         1.0                 1.0
    ``count_params()`` after      5,490,871           5,501,872
    weights perturbed             224                 224
    perturbation liveness         1.638e-01           2.488e-01
    round-trip ``max|delta|``     0.000e+00           0.000e+00
    ============================  ==================  ==================

    The two weight counts being EQUAL at 224 is not a copy-paste slip: the heads
    differ by one dense layer's kernel+bias against the classifier's, so the
    parameter TOTALS differ (the third row) while the tensor count does not.
    """

    #: Measured 2026-08-24 on GPU 1; see the class docstring. The same 224 that
    #: `GF_N_WEIGHTS_CLASSIFIER` pins for the gradient arm, reached by a
    #: different route -- `len(model.weights)`, not the trainable set.
    N_WEIGHTS = 224

    @staticmethod
    def _unbuilt_classifier():
        """The file's standard subject WITHOUT ``_classifier()``'s forward pass.

        ``_classifier()`` calls the model before returning it, which would hand
        the oracle an already-materialized instance and delete the one thing its
        materialization arm measures.
        """
        return create_beit_classifier(
            "tiny", IMG, PATCH, num_classes=NUM_CLASSES, drop_path_rate=0.0,
        )

    @staticmethod
    def _unbuilt_mim():
        return create_beit_mim(
            "tiny", IMG, PATCH, vocab_size=VOCAB, drop_path_rate=0.0,
        )

    def test_the_classifier_lazy_build_costs_nothing(self):
        report = assert_lazy_build_costs_nothing(
            build=self._unbuilt_classifier,
            make_inputs=lambda: _images(1),
            input_shape=(None,) + IMG,
        )
        assert report["n_weights"] == self.N_WEIGHTS
        assert report["roundtrip_max_delta"] == 0.0
        assert report["perturb_liveness"] > 1e-3
        assert report["materialization"]["ratio"] == 1.0
        assert report["materialization"]["n_weights_after_build"] == self.N_WEIGHTS, (
            "BeitModel.build() no longer materializes every weight; a PARTIAL "
            "materialization is the D-049 shape and a `> 0` assertion would "
            "pass against exactly that"
        )

    def test_the_masked_image_modelling_lazy_build_costs_nothing(self):
        """The two-input arm -- and the one that carries a LIVE ``mask_token``.

        In the classifier the mask token is dead weight (see
        ``CLASSIFIER_EXPECT_ZERO``), so a round trip that lost it would still
        produce an identical output. Here it is applied, so its restoration is
        inside the ``atol=0.0`` comparison rather than beside it.
        """
        report = assert_lazy_build_costs_nothing(
            build=self._unbuilt_mim,
            make_inputs=lambda: (_images(1), _mask(1)),
            input_shape=[(None,) + IMG, (None, NUM_PATCHES)],
        )
        assert report["n_weights"] == self.N_WEIGHTS
        assert report["roundtrip_max_delta"] == 0.0
        assert report["perturb_liveness"] > 1e-3
        assert report["materialization"]["ratio"] == 1.0

    def test_the_lazy_build_assertion_can_fail(self):
        """RED proof: a model whose forward ignores its weights must be caught.

        The liveness arm is the whole instrument -- without it, a round trip
        comparing an unperturbed model against itself passes over total weight
        loss, which is how ``ScoreBasedNanoVLM``'s own round-trip test passed
        3/3 while 464 of 1,305 tensors were never written. This injects exactly
        that shape: a subject whose output is a CONSTANT of its inputs, so
        perturbing all 224 weights moves nothing.
        """

        @keras.saving.register_keras_serializable(package="beit_test_red_proof")
        class _WeightBlindClassifier(keras.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.inner = create_beit_classifier(
                    "tiny", IMG, PATCH, num_classes=NUM_CLASSES,
                    drop_path_rate=0.0,
                )

            def call(self, inputs, training=None):
                self.inner(inputs, training=training)
                return keras.ops.zeros(
                    (keras.ops.shape(inputs)[0], NUM_CLASSES), dtype="float32",
                )

        with pytest.raises(AssertionError, match="did not move the output"):
            assert_lazy_build_costs_nothing(
                build=_WeightBlindClassifier,
                make_inputs=lambda: _images(1),
            )
