"""
Oracle adoption for ``models/SAM/SAM3`` -- Phase 5 batch B.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

WHAT IS MEASURED, AND WHY IT IS THE WRAPPER
--------------------------------------------
``Sam3TrainingModel`` is the object that trains: it returns ONE packed
supervision tensor ready for a single ``Sam3DetectionLoss`` under stock
``fit()``, and adds no parameters of its own. Every measurement here runs on it
at the ``tiny`` development geometry (``img_size=32``), batch 2.

``include_masks`` IS THE MEASUREMENT
-------------------------------------
Measured 2026-08-21 (GPU 1), one real Adam step, ``default_loss``, **217**
trainable weights:

=========================  ====  ==============================================
wrapper                    dead  what the dead set is
=========================  ====  ==============================================
``include_masks=False``    44    the ENTIRE segmentation head plus the neck
(the default)                    levels that feed only it
``include_masks=True``      6    4 scalped neck weights + 2 semantic-head
=========================  ====  ==============================================

The 44 is the default's own three-way contract doing exactly what it says --
masks are not in the packed tensor, so nothing downstream of the mask branch is
supervised -- and it is pinned rather than avoided, because "44 dead weights"
read cold is indistinguishable from a broken model.

The residual 6 are two named, independent facts:

1. **``sam3_3_conv_1x1`` / ``sam3_3_conv_3x3`` (4).** ``scalp=1`` at every
   shipped variant, and ``scalp`` DISCARDS the coarsest pyramid levels before
   the segmentation pyramid is formed (``sam3_image.py`` ``_scalp``). Level 3 is
   the coarsest of four, so its neck convolutions are built and never called.
2. **``semantic_seg_head`` (2).** The segmentation head emits both
   ``instance_seg`` and ``semantic_seg``; the wrapper's packed tensor carries
   the instance masks only, so the semantic head is an auxiliary output no
   supervision reaches on this path.

``expect_zero`` is two-sided, so the main assertion proves every one of those 6
IS dead AND that all 211 others are live.

STOCHASTIC RATES: the ``tiny`` variant is 0.0 on all three, and that is
DELIBERATE, not incidental -- the repository's shared ``StochasticDepth``
short-circuits on ``training is False`` only, so ``training=None`` (what a plain
``model(inputs)`` passes) DROPS PATHS. ``test_the_tiny_variant_pins_every_
stochastic_rate`` asserts it, because a gradient report taken under a live rate
reports the DRAW, not the model.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.SAM.SAM3.sam3_image import Sam3Image
from dl_techniques.models.SAM.SAM3.training_model import Sam3TrainingModel

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    gradient_report,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

VARIANT = "tiny"
TABLE = Sam3Image.MODEL_VARIANTS[VARIANT]
IMG = TABLE["img_size"]
CTX = TABLE["context_length"]
VOCAB = TABLE["vocab_size"]
QUERIES = TABLE["num_queries"]
BATCH = 2

#: Measured 2026-08-21 at the ``tiny`` geometry.
GF_N_WEIGHTS = 217
GF_N_OFF_GRAPH_WITH_MASKS = 6
GF_N_OFF_GRAPH_WITHOUT_MASKS = 44

#: The six weights off the graph even at ``include_masks=True``, matched by path
#: SUFFIX -- see the module docstring for the two causes. Suffixes rather than
#: absolute ``Variable.path`` strings because Keras uniquifies a model's name
#: per process, so an absolute pin is green alone and red behind any other test
#: that builds this class in the same session.
EXPECTED_OFF_GRAPH = tuple(sorted(
    # 1: the scalped (coarsest) pyramid level
    [f"sam3_3_conv_{k}/{w}"
     for k in ("1x1", "3x3") for w in ("kernel", "bias")]
    # 2: the auxiliary semantic head, absent from the packed supervision tensor
    + [f"semantic_seg_head/{w}" for w in ("kernel", "bias")]
))


def _inputs(batch: int = BATCH, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "image": rng.normal(size=(batch, IMG, IMG, 3)).astype("float32"),
        "token_ids": rng.integers(
            0, VOCAB, size=(batch, CTX)).astype("int32"),
    }


def _wrapper(include_masks: bool = True, seed: int = 1234,
             **sam3_overrides) -> Sam3TrainingModel:
    keras.utils.set_random_seed(seed)
    model = Sam3TrainingModel(
        Sam3Image.from_variant(VARIANT, **sam3_overrides),
        include_masks=include_masks,
    )
    model.build(None)
    return model


def _one_adam_step(model: keras.Model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, variables) if g is not None]
    )


def _dead(model: keras.Model, inputs) -> list:
    report = gradient_report(model, inputs)
    return [p for p, v in report.items() if v is None or v == 0.0]


class TestSAM3GradientFlow:

    def test_the_tiny_variant_pins_every_stochastic_rate(self):
        for key in ("drop_path_rate", "dropout_rate", "prompt_mlp_dropout_rate"):
            assert TABLE[key] == 0.0, (
                f"{key} is {TABLE[key]} at the {VARIANT!r} variant; a gradient "
                f"report taken under it reports the DRAW, not the model"
            )

    def test_the_wrapper_adds_no_weights_of_its_own(self):
        """The premise every count below rests on."""
        keras.utils.set_random_seed(1234)
        sam3 = Sam3Image.from_variant(VARIANT)
        wrapper = Sam3TrainingModel(sam3, include_masks=True)
        wrapper.build(None)
        assert len(wrapper.trainable_weights) == len(sam3.trainable_weights)

    def test_the_expected_off_graph_set_is_exactly_the_measured_one(self):
        model = _wrapper(include_masks=True)
        x = _inputs()
        _one_adam_step(model, x)
        dead = _dead(model, x)

        assert len(dead) == GF_N_OFF_GRAPH_WITH_MASKS, (
            f"expected {GF_N_OFF_GRAPH_WITH_MASKS} off-graph weights, got "
            f"{len(dead)}: {sorted(dead)}"
        )
        unexplained = [p for p in dead
                       if not any(p.endswith(s) for s in EXPECTED_OFF_GRAPH)]
        assert not unexplained, (
            f"off-graph weights with NO attributed cause: {sorted(unexplained)}"
        )

    def test_gradients_reach_every_other_trainable_weight_after_one_step(self):
        model = _wrapper(include_masks=True)
        x = _inputs()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(
            model, x, expect_zero=EXPECTED_OFF_GRAPH)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)
        live = [p for p, v in report.items() if v is not None and v > 0.0]
        assert len(live) == GF_N_WEIGHTS - GF_N_OFF_GRAPH_WITH_MASKS

    def test_include_masks_false_leaves_the_whole_segmentation_head_unsupervised(
            self):
        """The default's own contract, pinned so 44 is not read as a defect.

        This is also the discriminating half of the claim above: the
        segmentation head IS reachable, and ``include_masks`` is what reaches
        it.
        """
        model = _wrapper(include_masks=False)
        x = _inputs()
        _one_adam_step(model, x)
        dead = _dead(model, x)

        assert len(dead) == GF_N_OFF_GRAPH_WITHOUT_MASKS, (
            f"expected {GF_N_OFF_GRAPH_WITHOUT_MASKS} off-graph weights at "
            f"include_masks=False, got {len(dead)}: {sorted(dead)}"
        )
        seg_head = [p for p in dead if "segmentation_head" in p]
        assert len(seg_head) > 2, (
            f"the segmentation head is barely in the off-graph set "
            f"({sorted(seg_head)}); this test then proves nothing about it"
        )

        # The discriminating half: all but the two auxiliary semantic-head
        # weights come BACK when the masks are packed.
        with_masks = _wrapper(include_masks=True)
        _one_adam_step(with_masks, x)
        still_dead = [p for p in _dead(with_masks, x)
                      if "segmentation_head" in p]
        assert len(still_dead) == 2 and all(
            "semantic_seg_head" in p for p in still_dead), (
            f"exactly the two semantic-head weights must remain dead at "
            f"include_masks=True; got {sorted(still_dead)}"
        )

    def test_the_gradient_assertion_can_fail(self):
        """RED proof: detach the forward and every weight must be convicted."""
        model = _wrapper(include_masks=True)
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _inputs())


class TestSAM3KnobSensitivity:

    def test_num_queries_changes_the_parameterisation(self):
        builders = {
            q: (lambda q=q: _wrapper(num_queries=q)) for q in (4, 8, 16)
        }
        assert_structural_knob_changes_weights(builders, knob="num_queries")

    def test_decoder_layers_changes_the_parameterisation(self):
        builders = {
            n: (lambda n=n: _wrapper(decoder_layers=n)) for n in (1, 2, 3)
        }
        assert_structural_knob_changes_weights(builders, knob="decoder_layers")

    def test_include_masks_is_NOT_a_parameterisation_knob(self):
        """The negative control, stated rather than left implicit.

        ``include_masks`` changes what the wrapper PACKS, not what the model
        holds -- the segmentation head is built either way (which is why its
        weights survive a ``.keras`` round trip at the default). A weight-shape
        difference here would mean the wrapper had started building model
        parameters, which its own interface contract forbids. The oracle's
        ``is a no-op`` message is the PASSING condition.
        """
        builders = {
            flag: (lambda flag=flag: _wrapper(include_masks=flag))
            for flag in (False, True)
        }
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(
                builders, knob="include_masks")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _wrapper()), "b": (lambda: _wrapper())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="num_queries")


class TestSAM3SmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _wrapper(include_masks=True)
        x = _inputs()

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"Sam3TrainingModel returns ONE packed tensor, got {type(out)}"
            )
            assert len(out.shape) == 3, (
                f"the packed tensor is (B, (Q + 1) * blocks, C); got rank "
                f"{len(out.shape)}"
            )
            assert tuple(out.shape)[:2] == (BATCH, QUERIES + 1), (
                f"expected {(BATCH, QUERIES + 1)} on the leading two axes "
                f"(Q + 1 = the queries plus the presence slot), got "
                f"{tuple(out.shape)[:2]}"
            )
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
