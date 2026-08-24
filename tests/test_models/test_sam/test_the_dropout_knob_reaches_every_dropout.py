"""The SAM 1 attention-dropout rate is reachable from every entry point.

Before this guard existed the rate was unreachable: ``TwoWayTransformer``
declared ``attention_dropout_rate: float = 0.0`` and ``SAM.from_variant``
constructed it without passing the keyword, so the mask decoder's ``Dropout``
layers could not be reached from ``from_variant`` at all. The knob added by
plan-2026-08-23T091307-9a110062 D-601 is threaded as
``SAM.MODEL_VARIANTS[...]["dropout_rate"]`` ->
``TwoWayTransformer(attention_dropout_rate=)``.

WHY THIS REPLACES A "CLOSED, NOT A DEFECT" RULING
-------------------------------------------------
D-091 (plan-2026-08-22T035419-a11304c8) closed this gap on the ground that the
unreachable rate was ``0.0`` and therefore inert, citing "MEASURED on vit_b: 0
live ``keras.layers.Dropout``". That measurement is REPRODUCIBLE AND MISLEADING,
and ``test_an_unbuilt_model_reports_zero_dropouts`` below pins the reason: SAM 1's
``Dropout`` layers are created inside ``MultiHeadAttention.build()``, not in
``__init__``, so ``_flatten_layers()`` on a freshly constructed model really does
report **0** — and reports **7** the moment the mask decoder is built. The rate
was inert, but the layers were there all along.

WHY THE DEFAULT ARM ALONE WOULD BE VACUOUS
------------------------------------------
Asserting that a stock model's rates are ``{0.0}`` passes just as well with the
threading DELETED — the layer default is also ``0.0``. That is even more true
here than it was for SAM 2, whose default at least differed from zero. So the
load-bearing assertion is the NON-DEFAULT one: every rate must equal a value that
exists nowhere as a default.

The count (7) is MEASURED, not derived, and it is the same for all three variants
because SAM 1's prompt-encoder and mask-decoder geometry does not vary by variant
— only the image encoder does, and the image encoder owns no ``Dropout``.
"""

from typing import Any, Set

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.sam.sam1.model import SAM
from dl_techniques.models.vision_language.sam.sam1.transformer import (
    DEFAULT_ATTENTION_DROPOUT_RATE,
    TwoWayTransformer,
)

#: MEASURED 2026-08-23 on CPU, AFTER the mask decoder is built: live
#: ``keras.layers.Dropout`` instances reachable from ``SAM._flatten_layers()``.
#: Every one lives inside ``mask_decoder.transformer``; the image encoder, the
#: prompt encoder and the mask-decoder heads own none.
MEASURED_DROPOUT_COUNT = 7

#: A rate that is the default of nothing, so an arm asserting it cannot be
#: satisfied by a dead knob falling back to a default.
NON_DEFAULT_RATE = 0.37

#: The mask decoder's fixed geometry: 64x64 image-embedding grid at 256 channels.
GRID = 64
CHANNELS = 256


def _build_mask_decoder(model: SAM) -> None:
    """Force the mask decoder to build. See this module's docstring: SAM 1's
    ``Dropout`` layers do not exist until their attention layers are built, so
    every count below is taken AFTER this call, never before."""
    image = np.zeros((1, GRID, GRID, CHANNELS), "float32")
    sparse = np.zeros((1, 3, CHANNELS), "float32")
    model.mask_decoder(
        image, image, sparse, image, multimask_output=True, training=False)


def dropout_rates(model: keras.Model) -> Set[float]:
    """Set of rates over every live ``Dropout`` in ``model``."""
    return {
        float(layer.rate)
        for layer in model._flatten_layers()
        if isinstance(layer, keras.layers.Dropout)
    }


def dropout_count(model: keras.Model) -> int:
    """Number of live ``Dropout`` layers in ``model``."""
    return sum(
        1 for layer in model._flatten_layers()
        if isinstance(layer, keras.layers.Dropout)
    )


class TestTheInstrument:
    """The measurement that produced the ruling this guard overturns."""

    def test_an_unbuilt_model_reports_zero_dropouts(self) -> None:
        """D-091's "0 live Dropout" reproduces — on an UNBUILT model.

        This is pinned rather than deleted because it is the trap: a count taken
        straight after ``from_variant`` is a count of layers that have not been
        created yet, and it reads a live population as an absent one.
        """
        model = SAM.from_variant("vit_b")
        assert dropout_count(model) == 0
        _build_mask_decoder(model)
        assert dropout_count(model) == MEASURED_DROPOUT_COUNT


class TestTheShippedDefault:
    """The default arm: it pins today's behaviour, and nothing more."""

    @pytest.mark.parametrize("variant", ["vit_b", "vit_l", "vit_h"])
    def test_the_table_defers_to_the_single_home_of_the_number(
            self, variant: str) -> None:
        """The rate is read from ``DEFAULT_ATTENTION_DROPOUT_RATE``, not restated."""
        assert SAM.MODEL_VARIANTS[variant]["dropout_rate"] == (
            DEFAULT_ATTENTION_DROPOUT_RATE)
        assert DEFAULT_ATTENTION_DROPOUT_RATE == 0.0

    def test_both_class_signatures_read_the_same_constant(self) -> None:
        """A number with one home. Both classes in ``transformer.py`` default to it."""
        import inspect

        from dl_techniques.models.vision_language.sam.sam1.transformer import TwoWayAttentionBlock

        for cls in (TwoWayAttentionBlock, TwoWayTransformer):
            default = inspect.signature(
                cls.__init__).parameters["attention_dropout_rate"].default
            assert default == DEFAULT_ATTENTION_DROPOUT_RATE, cls.__name__

    def test_the_stock_model_still_carries_the_measured_rate(self) -> None:
        """Adding the knob changed no shipped number."""
        model = SAM.from_variant("vit_b")
        _build_mask_decoder(model)
        assert dropout_count(model) == MEASURED_DROPOUT_COUNT
        assert dropout_rates(model) == {0.0}
        assert model.dropout_rate == pytest.approx(0.0)


class TestTheKnobIsLive:
    """The load-bearing arm: a NON-default value must reach every sublayer."""

    @pytest.mark.parametrize("variant", ["vit_b", "vit_l", "vit_h"])
    def test_a_non_default_rate_reaches_every_dropout(
            self, variant: str) -> None:
        """Every rate equals the requested value -- not just one of them.

        A set comparison, so a knob that reached the self-attention dropout but
        not the two cross-attentions would report ``{0.0, 0.37}`` and fail.
        """
        model = SAM.from_variant(variant, dropout_rate=NON_DEFAULT_RATE)
        _build_mask_decoder(model)
        assert dropout_count(model) == MEASURED_DROPOUT_COUNT
        assert dropout_rates(model) == {NON_DEFAULT_RATE}
        assert model.dropout_rate == pytest.approx(NON_DEFAULT_RATE)

    def test_zero_is_reachable_and_is_not_treated_as_absent(self) -> None:
        """``0.0`` is falsy; the sentinel is ``None``, so it must survive."""
        model = SAM.from_variant("vit_b", dropout_rate=0.0)
        _build_mask_decoder(model)
        assert dropout_rates(model) == {0.0}

    @pytest.mark.parametrize("bad", [1.0, 1.5, -0.1])
    def test_an_out_of_range_rate_is_refused(self, bad: float) -> None:
        """A rate of 1.0 zeroes the stack; it is refused, not silently taken."""
        with pytest.raises(ValueError, match="dropout_rate"):
            SAM.from_variant("vit_b", dropout_rate=bad)

    def test_the_rate_is_behaviourally_live_on_the_transformer(self) -> None:
        """A behavioural companion to the structural assertion above.

        Probed on a small ``TwoWayTransformer`` rather than on a ``vit_b`` SAM
        for cost only: the layer the knob configures is the same one, and the
        full model's image encoder is 90M parameters of irrelevance here.

        Stochastic by construction, so the draw is SEEDED. This is an addition
        to the rate-set assertion, never a replacement: a rate that reached one
        sublayer out of seven would also move this number.
        """
        keras.utils.set_random_seed(5)
        stack = TwoWayTransformer(
            depth=2, embedding_dim=32, num_heads=4, mlp_dim=64,
            attention_dropout_rate=NON_DEFAULT_RATE,
        )
        rng = np.random.default_rng(3)
        image = rng.standard_normal((2, 4, 4, 32)).astype("float32")
        pe = rng.standard_normal((2, 4, 4, 32)).astype("float32")
        points = rng.standard_normal((2, 5, 32)).astype("float32")
        stack(image, pe, points, training=False)

        keras.utils.set_random_seed(4)
        wet = np.asarray(stack(image, pe, points, training=True)[0])
        for layer in stack._flatten_layers():
            if isinstance(layer, keras.layers.Dropout):
                layer.rate = 0.0
        keras.utils.set_random_seed(4)
        dry = np.asarray(stack(image, pe, points, training=True)[0])
        assert float(np.max(np.abs(wet - dry))) > 0.0

    def test_the_default_is_behaviourally_inert(self) -> None:
        """The other half of the same measurement, and the reason the default is
        a behaviour-preserving choice: at ``0.0`` the SAME probe measures exactly
        ``0.0``, so shipping this knob changed nothing for existing callers."""
        keras.utils.set_random_seed(5)
        stack = TwoWayTransformer(
            depth=2, embedding_dim=32, num_heads=4, mlp_dim=64)
        rng = np.random.default_rng(3)
        image = rng.standard_normal((2, 4, 4, 32)).astype("float32")
        pe = rng.standard_normal((2, 4, 4, 32)).astype("float32")
        points = rng.standard_normal((2, 5, 32)).astype("float32")
        stack(image, pe, points, training=False)

        keras.utils.set_random_seed(4)
        a = np.asarray(stack(image, pe, points, training=True)[0])
        keras.utils.set_random_seed(4)
        b = np.asarray(stack(image, pe, points, training=False)[0])
        assert float(np.max(np.abs(a - b))) == 0.0


class TestSerialization:
    """The rate round-trips, and no pre-existing config gains a required key."""

    def test_the_outer_config_deliberately_has_no_dropout_key(self) -> None:
        """D-601: the rate has ONE home, the nested transformer config.

        If this ever goes red because someone added ``"dropout_rate"`` to
        ``SAM.get_config()``, read D-601 before "fixing" the test: a stored outer
        copy can silently disagree with the sub-layer actually built, because
        ``SAM(...)`` accepts an already-constructed ``mask_decoder``.
        """
        config = SAM.from_variant("vit_b", dropout_rate=NON_DEFAULT_RATE).get_config()
        assert "dropout_rate" not in config
        nested = config["mask_decoder"]["config"]["transformer"]["config"]
        assert nested["attention_dropout_rate"] == NON_DEFAULT_RATE

    def test_a_config_without_the_key_still_reconstructs(self) -> None:
        """The pre-knob config shape IS the current config shape."""
        model = SAM.from_variant("vit_b", dropout_rate=NON_DEFAULT_RATE)
        clone = SAM.from_config(model.get_config())
        _build_mask_decoder(clone)
        assert dropout_rates(clone) == {NON_DEFAULT_RATE}
        assert clone.dropout_rate == pytest.approx(NON_DEFAULT_RATE)

    def test_a_disagreeing_outer_copy_is_impossible(self, tmp_path: Any) -> None:
        """Why the property is derived rather than stored: a caller may hand in
        an already-built mask decoder, and ``SAM.dropout_rate`` must report what
        that decoder actually carries rather than what anyone was told."""
        donor = SAM.from_variant("vit_b", dropout_rate=NON_DEFAULT_RATE)
        host = SAM.from_variant("vit_b")
        rebuilt = SAM(
            image_encoder=host.image_encoder,
            prompt_encoder=host.prompt_encoder,
            mask_decoder=donor.mask_decoder,
        )
        assert rebuilt.dropout_rate == pytest.approx(NON_DEFAULT_RATE)
