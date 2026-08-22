"""The SAM 2 memory-attention dropout rate is reachable from every entry point.

Before this guard existed the rate was hard-wired: ``SAM2MemoryAttention``
declared ``dropout: float = 0.1`` and ``SAM2.from_variant`` constructed it
without passing the keyword, so the 12 (``tiny``) / 24 (``hiera_l``) live
``Dropout`` layers could not be reached from ``from_variant`` or ``create_sam2``
at all. The knob added by plan-2026-08-22T035419-a11304c8 D-090 is threaded as
``SAM2.MODEL_VARIANTS[...]["dropout_rate"]`` -> ``SAM2MemoryAttention(dropout=)``.

WHY THE DEFAULT ARM ALONE WOULD BE VACUOUS
------------------------------------------
Asserting that a stock model's rates are ``{0.1}`` passes just as well with the
threading DELETED -- the layer default is also ``0.1``. So the load-bearing
assertion here is the NON-DEFAULT one: every rate must equal a value that
exists nowhere as a default. RED-proved by deleting the ``dropout=dropout_rate``
keyword at ``model.py``'s ``SAM2MemoryAttention(...)`` construction, which turns
``test_a_non_default_rate_reaches_every_dropout`` red at the rate-set assertion
(observed ``{0.1} != {0.37}``) while every default-arm assertion stays green.

The counts (12 / 24) are MEASURED, not derived: each ``SAM2MemoryAttentionLayer``
owns 6 ``Dropout`` layers (2 attention + ffn + 3 residual) and the variants have
2 and 4 layers respectively.
"""

from typing import Any, Set

import keras
import numpy as np
import pytest

from dl_techniques.models.SAM.SAM2.memory_attention import DEFAULT_DROPOUT_RATE
from dl_techniques.models.SAM.SAM2.model import SAM2, create_sam2

#: MEASURED 2026-08-22 on CPU: live ``keras.layers.Dropout`` instances reachable
#: from ``SAM2._flatten_layers()``, per variant. Every one of them lives inside
#: ``memory_attention``; no other SAM 2 component owns a ``Dropout``.
MEASURED_DROPOUT_COUNT = {"tiny": 12, "hiera_l": 24}

#: A rate that is the default of nothing, so an arm asserting it cannot be
#: satisfied by a dead knob falling back to a default.
NON_DEFAULT_RATE = 0.37


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


class TestTheShippedDefault:
    """The default arm: it pins today's behaviour, and nothing more."""

    @pytest.mark.parametrize("variant", ["tiny", "hiera_l"])
    def test_the_table_defers_to_the_single_home_of_the_number(
            self, variant: str) -> None:
        """The rate is read from ``DEFAULT_DROPOUT_RATE``, not restated."""
        assert SAM2.MODEL_VARIANTS[variant]["dropout_rate"] == (
            DEFAULT_DROPOUT_RATE)
        assert DEFAULT_DROPOUT_RATE == 0.1

    @pytest.mark.parametrize("variant", ["tiny", "hiera_l"])
    def test_the_stock_model_still_carries_the_measured_rates(
            self, variant: str) -> None:
        """Adding the knob changed no shipped number."""
        model = SAM2.from_variant(variant)
        assert dropout_count(model) == MEASURED_DROPOUT_COUNT[variant]
        assert dropout_rates(model) == {0.1}
        assert model.dropout_rate == pytest.approx(0.1)


class TestTheKnobIsLive:
    """The load-bearing arm: a NON-default value must reach every sublayer."""

    @pytest.mark.parametrize("variant", ["tiny", "hiera_l"])
    def test_a_non_default_rate_reaches_every_dropout(
            self, variant: str) -> None:
        """Every rate equals the requested value -- not just one of them.

        A set comparison, so a knob that reached the attention dropouts but not
        the residual/FFN ones would report ``{0.1, 0.37}`` and fail.
        """
        model = SAM2.from_variant(variant, dropout_rate=NON_DEFAULT_RATE)
        assert dropout_count(model) == MEASURED_DROPOUT_COUNT[variant]
        assert dropout_rates(model) == {NON_DEFAULT_RATE}
        assert model.dropout_rate == pytest.approx(NON_DEFAULT_RATE)

    def test_zero_is_reachable_and_is_not_treated_as_absent(self) -> None:
        """``0.0`` is falsy; the S-3 sentinel is ``None``, so it must survive."""
        model = SAM2.from_variant("tiny", dropout_rate=0.0)
        assert dropout_rates(model) == {0.0}

    def test_create_sam2_threads_the_knob(self) -> None:
        """The factory is an entry point too, and was blind to the rate."""
        model = create_sam2("tiny", dropout_rate=NON_DEFAULT_RATE)
        assert dropout_rates(model) == {NON_DEFAULT_RATE}

    def test_create_sam2_without_the_knob_still_ships_the_default(self) -> None:
        """``None`` defers to the table (S-3)."""
        assert dropout_rates(create_sam2("tiny")) == {0.1}

    @pytest.mark.parametrize("bad", [1.0, 1.5, -0.1])
    def test_an_out_of_range_rate_is_refused(self, bad: float) -> None:
        """A rate of 1.0 zeroes the stack; it is refused, not silently taken."""
        with pytest.raises(ValueError, match="dropout_rate"):
            SAM2.from_variant("tiny", dropout_rate=bad)

    def test_the_rate_is_behaviourally_live_on_the_memory_attention(
            self) -> None:
        """A behavioural companion to the structural assertion above.

        MEASURED 2026-08-22, and the reason this probe is NOT at the model
        level: ``SAM2.call`` hard-wires ``training=False`` at its one
        ``self.memory_attention(...)`` call (``model.py:1170``), so no dropout
        can ever fire through ``SAM2({'image': x}, training=True)`` -- a
        model-level wet-vs-dry arm measures exactly ``0.0`` and would be a
        guard that cannot fail. ``SAM2TrainingModel`` is the consumer that
        threads ``training`` through (``training_model.py:340-346``), so the
        rate is live exactly where a rate is supposed to be live.

        Stochastic by construction, so the draw is SEEDED. This is an addition
        to the rate-set assertion, never a replacement: a rate that reached one
        sublayer out of twelve would also move this number.
        """
        model = SAM2.from_variant("tiny")
        stack = model.memory_attention
        grid = stack.feat_sizes[0] * stack.feat_sizes[1]
        rng = np.random.default_rng(3)
        feats = rng.standard_normal((1, grid, stack.d_model)).astype("float32")
        mem = rng.standard_normal((1, grid, stack.kv_in_dim)).astype("float32")
        stack.build(feats.shape, mem.shape)

        keras.utils.set_random_seed(4)
        wet = np.asarray(stack(feats, mem, training=True))
        for layer in stack._flatten_layers():
            if isinstance(layer, keras.layers.Dropout):
                layer.rate = 0.0
        keras.utils.set_random_seed(4)
        dry = np.asarray(stack(feats, mem, training=True))
        assert float(np.max(np.abs(wet - dry))) > 0.0


class TestSerialization:
    """The rate round-trips, and no pre-existing config gains a required key."""

    def test_the_outer_config_deliberately_has_no_dropout_key(self) -> None:
        """D-090: the rate has ONE home, the nested memory-attention config.

        If this ever goes red because someone added ``"dropout_rate"`` to
        ``SAM2.get_config()``, read D-090 before "fixing" the test: a stored
        outer copy can silently disagree with the sub-layer actually built,
        because ``SAM2(...)`` accepts an already-constructed ``memory_attention``.
        """
        config = SAM2.from_variant("tiny").get_config()
        assert "dropout_rate" not in config
        assert config["memory_attention"]["config"]["dropout"] == 0.1

    def test_a_config_without_the_key_still_reconstructs(self) -> None:
        """The pre-knob config shape IS the current config shape."""
        model = SAM2.from_variant("tiny", dropout_rate=NON_DEFAULT_RATE)
        clone = SAM2.from_config(model.get_config())
        assert dropout_rates(clone) == {NON_DEFAULT_RATE}

    def test_a_keras_round_trip_preserves_the_rate(self, tmp_path: Any) -> None:
        """Save/load a BUILT model and re-read every rate from the file."""
        model = SAM2.from_variant("tiny", dropout_rate=NON_DEFAULT_RATE)
        model.build(None)
        path = tmp_path / "sam2_tiny_dropout.keras"
        model.save(path)

        restored = keras.models.load_model(path)
        assert restored.dropout_rate == pytest.approx(NON_DEFAULT_RATE)
        assert dropout_rates(restored) == {NON_DEFAULT_RATE}
        assert dropout_count(restored) == MEASURED_DROPOUT_COUNT["tiny"]
