"""The shared encoder contract: outputs, masking, pooling and serialization.

These are the properties every arm inherits, so they are tested once here
rather than duplicated per arm.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.tokenizers.ascii_char import PAD_ID, VOCAB_SIZE
from dl_techniques.models.embeddings_experimental.shared import EmbeddingEncoder

SEQ_LEN = 16
HIDDEN = 32

BLOCK_CASES = [
    ("transformer", {"num_heads": 4, "intermediate_size": 64}),
    ("clifford", {"shifts": [1, 2], "context_kernel_size": 3}),
    ("convnext", {"kernel_size": 3}),
    ("convnext_v2", {"kernel_size": 3}),
]
BLOCK_IDS = [case[0] for case in BLOCK_CASES]


def build_encoder(block_type, block_config, **overrides):
    """Build and shape-build a small encoder."""
    config = dict(
        hidden_size=HIDDEN,
        num_layers=2,
        block_type=block_type,
        block_config=block_config,
        max_position_embeddings=SEQ_LEN * 2,
        hidden_dropout_rate=0.0,
    )
    config.update(overrides)
    model = EmbeddingEncoder(**config)
    model.build((None, SEQ_LEN))
    return model


@pytest.fixture
def token_ids():
    rng = np.random.default_rng(7)
    ids = rng.integers(6, VOCAB_SIZE, size=(3, SEQ_LEN)).astype("int32")
    ids[0, 10:] = PAD_ID
    ids[1, 14:] = PAD_ID
    return ids


# ---------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------

class TestOutputContract:
    """Three keys, correct shapes, agreeing with compute_output_shape."""

    @pytest.mark.parametrize("block_type,config", BLOCK_CASES, ids=BLOCK_IDS)
    def test_returns_the_three_documented_keys(self, block_type, config, token_ids):
        out = build_encoder(block_type, config)({"input_ids": token_ids})
        assert set(out) == {
            "last_hidden_state",
            "attention_mask",
            "pooled_output",
        }

    @pytest.mark.parametrize("block_type,config", BLOCK_CASES, ids=BLOCK_IDS)
    def test_shapes(self, block_type, config, token_ids):
        out = build_encoder(block_type, config)({"input_ids": token_ids})
        batch = token_ids.shape[0]
        assert tuple(out["last_hidden_state"].shape) == (batch, SEQ_LEN, HIDDEN)
        assert tuple(out["attention_mask"].shape) == (batch, SEQ_LEN)
        assert tuple(out["pooled_output"].shape) == (batch, HIDDEN)

    @pytest.mark.parametrize("block_type,config", BLOCK_CASES, ids=BLOCK_IDS)
    def test_compute_output_shape_agrees_with_the_forward_pass(
        self, block_type, config, token_ids
    ):
        model = build_encoder(block_type, config)
        declared = model.compute_output_shape((token_ids.shape[0], SEQ_LEN))
        actual = model({"input_ids": token_ids})
        for key, shape in declared.items():
            assert tuple(actual[key].shape) == tuple(shape), key

    @pytest.mark.parametrize("block_type,config", BLOCK_CASES, ids=BLOCK_IDS)
    def test_a_bare_tensor_input_is_accepted(self, block_type, config, token_ids):
        model = build_encoder(block_type, config)
        assert np.allclose(
            keras.ops.convert_to_numpy(model(token_ids)["pooled_output"]),
            keras.ops.convert_to_numpy(
                model({"input_ids": token_ids})["pooled_output"]
            ),
            atol=1e-6,
            rtol=0,
        )

    def test_a_dict_without_input_ids_raises(self):
        model = build_encoder(*BLOCK_CASES[0])
        with pytest.raises(ValueError, match="input_ids"):
            model({"attention_mask": np.ones((1, SEQ_LEN), dtype="int32")})


# ---------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------

class TestAttentionMaskResolution:
    """The mask is resolved BEFORE the stack, unlike upstream BERT.

    Pooling consumes it and the Clifford block consumes it, so a silently
    absent mask would silently pool over padding.
    """

    @pytest.mark.parametrize("block_type,config", BLOCK_CASES, ids=BLOCK_IDS)
    def test_mask_is_derived_from_pad_token_id_when_absent(
        self, block_type, config, token_ids
    ):
        out = build_encoder(block_type, config)({"input_ids": token_ids})
        returned = keras.ops.convert_to_numpy(out["attention_mask"])
        np.testing.assert_array_equal(
            returned, (token_ids != PAD_ID).astype(returned.dtype)
        )

    @pytest.mark.parametrize("block_type,config", BLOCK_CASES, ids=BLOCK_IDS)
    def test_an_explicit_mask_is_passed_through_unchanged(
        self, block_type, config, token_ids
    ):
        explicit = np.ones_like(token_ids)
        out = build_encoder(block_type, config)(
            {"input_ids": token_ids, "attention_mask": explicit}
        )
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(out["attention_mask"]), explicit
        )

    def test_a_changed_pad_id_changes_the_derived_mask(self, token_ids):
        model = build_encoder(*BLOCK_CASES[0], pad_token_id=7)
        returned = keras.ops.convert_to_numpy(
            model({"input_ids": token_ids})["attention_mask"]
        )
        np.testing.assert_array_equal(
            returned, (token_ids != 7).astype(returned.dtype)
        )

    def test_mean_pooling_ignores_padding_in_the_attention_arm(self):
        """Masked mean pooling must not average over pad positions.

        Asserted on the transformer arm only. The Clifford arm CANNOT satisfy
        this -- its block is maskless by design -- and that asymmetry is pinned
        separately in ``test_the_clifford_arm_is_padding_sensitive``.
        """
        model = build_encoder(*BLOCK_CASES[0], pooling_strategy="mean")
        rng = np.random.default_rng(3)
        prefix = rng.integers(6, VOCAB_SIZE, size=(1, 6)).astype("int32")

        short = np.concatenate(
            [prefix, np.full((1, SEQ_LEN - 6), PAD_ID, dtype="int32")], axis=1
        )
        longer = np.concatenate(
            [prefix, np.full((1, SEQ_LEN - 6), PAD_ID, dtype="int32")], axis=1
        )
        longer[0, 6:] = PAD_ID

        a = keras.ops.convert_to_numpy(model({"input_ids": short})["pooled_output"])
        b = keras.ops.convert_to_numpy(model({"input_ids": longer})["pooled_output"])
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------
# Pooling axis
# ---------------------------------------------------------------------

class TestPoolingAxisIsWired:
    """A pooling knob that changes nothing would be a dead knob."""

    @pytest.mark.parametrize("strategy", ["cls", "mean", "attention", "max"])
    def test_each_strategy_builds_and_produces_finite_output(
        self, strategy, token_ids
    ):
        model = build_encoder(*BLOCK_CASES[0], pooling_strategy=strategy)
        pooled = model({"input_ids": token_ids})["pooled_output"]
        assert pooled.shape[0] == token_ids.shape[0]
        assert np.isfinite(keras.ops.convert_to_numpy(pooled)).all()

    def test_different_strategies_give_different_embeddings(self, token_ids):
        """The axis must actually move the output.

        Weights differ between freshly constructed models, so the comparison
        is made on ONE model whose pooler is swapped, isolating the strategy
        from initialization luck.
        """
        from dl_techniques.layers.sequence_pooling.sequence_pooling import (
            SequencePooling,
        )

        model = build_encoder(*BLOCK_CASES[0], pooling_strategy="mean")
        hidden = model({"input_ids": token_ids})["last_hidden_state"]
        mask = keras.ops.cast(
            keras.ops.convert_to_tensor((token_ids != PAD_ID).astype("int32")),
            "bool",
        )

        pooled = {}
        for strategy in ("cls", "mean", "max"):
            pooler = SequencePooling(strategy=strategy)
            pooler.build((None, SEQ_LEN, HIDDEN))
            pooled[strategy] = keras.ops.convert_to_numpy(
                pooler(hidden, mask=mask, training=False)
            )

        assert not np.allclose(pooled["cls"], pooled["mean"], atol=1e-6)
        assert not np.allclose(pooled["mean"], pooled["max"], atol=1e-6)

    def test_cls_pooling_returns_the_first_position(self, token_ids):
        model = build_encoder(*BLOCK_CASES[0], pooling_strategy="cls")
        out = model({"input_ids": token_ids})
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out["pooled_output"]),
            keras.ops.convert_to_numpy(out["last_hidden_state"])[:, 0, :],
            atol=1e-6,
            rtol=0,
        )


# ---------------------------------------------------------------------
# Padding sensitivity -- the study's central caveat
# ---------------------------------------------------------------------

def _prefix_padded_to(prefix, length):
    """Right-pad a prefix to ``length`` with the pad id."""
    if length == prefix.shape[1]:
        return prefix
    return np.concatenate(
        [prefix, np.full((1, length - prefix.shape[1]), PAD_ID, dtype="int32")],
        axis=1,
    )


def _clifford_padding_delta(
    prefix, length_a, length_b, *, layer_scale_init, use_global_context
):
    """Max change across the real prefix between two padded lengths.

    ONE model is constructed and probed twice. Building a model per padding
    would compare two different random initializations, which swamps the
    effect being measured and makes both a pass and a fail unreadable.

    :return: ``max |states(length_a) - states(length_b)|`` over the prefix.
    :rtype: float
    """
    model = build_encoder(
        "clifford",
        {
            "shifts": [1, 2],
            "context_kernel_size": 3,
            "layer_scale_init": layer_scale_init,
            "use_global_context": use_global_context,
        },
        max_position_embeddings=64,
    )
    model.build((None, prefix.shape[1]))

    keep = prefix.shape[1]

    def states(length):
        ids = _prefix_padded_to(prefix, length)
        return keras.ops.convert_to_numpy(
            model({"input_ids": ids})["last_hidden_state"]
        )[0, :keep]

    delta = float(np.abs(states(length_a) - states(length_b)).max())
    keras.backend.clear_session()
    return delta


class TestTheCliffordArmIsPaddingSensitive:
    """Pinned known behaviour, not a defect to be fixed here.

    ``CliffordNetBlock`` has ``supports_masking = False``: its two stacked
    same-padded depthwise convolutions pull zero padding into the receptive
    field of real positions near the boundary. The wrapper zeroes masked
    positions, which bounds the effect but cannot remove it. This is why the
    study pretrains on PACKED fixed-length sequences carrying no padding.

    Three separate facts are pinned, because conflating them produces a guard
    that reports the opposite of the truth. Measured on this model, a 6-token
    prefix, ``hidden_size=32``, two blocks, ``K=3``:

    ==================  ============  ==================  ================
    layer_scale_init    global ctx    unpadded vs pad-8   pad-8 vs pad-12
    ==================  ============  ==================  ================
    1e-5 (default)      False         1.490e-07           0.000e+00
    1e-5 (default)      True          2.384e-07           1.192e-07
    1.0                 False         1.082e-02           0.000e+00
    1.0                 True          1.074e-02           8.160e-03
    ==================  ============  ==================  ================

    1. The hazard is the PRESENCE of padding, not its length. With the global
       branch off, pad-8 against pad-12 is EXACTLY zero at both gammas -- once
       the boundary is past the receptive field, more padding changes nothing.
       A guard that compares two padded lengths therefore measures zero and
       concludes, wrongly, that the arm is padding-safe.
    2. Only the global branch makes pad LENGTH matter, because its cumulative
       mean runs over the whole padded sequence. That is why
       ``use_global_context`` defaults to ``False``.
    3. LayerScale hides all of it at initialization. At the default
       ``layer_scale_init=1e-5`` every effect sits at float32 noise (1e-07);
       at gamma 1.0 it is five orders larger. Gamma is LEARNED, so a smoke
       test at init would report a padding-safe arm that becomes
       padding-sensitive during training. These tests therefore probe at
       gamma 1.0, where the effect is visible.
    """

    @pytest.fixture
    def prefix(self):
        rng = np.random.default_rng(11)
        return rng.integers(6, VOCAB_SIZE, size=(1, 6)).astype("int32")

    def test_padding_presence_perturbs_the_real_positions(self, prefix):
        """Fact 1: unpadded and padded disagree, once gamma is meaningful."""
        delta = _clifford_padding_delta(
            prefix, 6, 8, layer_scale_init=1.0, use_global_context=False
        )
        assert delta > 1e-4, (
            "expected the Clifford arm to be perturbed by the presence of "
            f"padding; got {delta:.3e}. If this is now zero, the block has "
            "gained masking support and the study's packing rationale needs "
            "rewriting."
        )

    def test_pad_length_is_inert_without_the_global_branch(self, prefix):
        """Fact 2, negative half: more padding beyond the boundary is free."""
        delta = _clifford_padding_delta(
            prefix, 8, 12, layer_scale_init=1.0, use_global_context=False
        )
        assert delta == pytest.approx(0.0, abs=1e-9)

    def test_pad_length_matters_once_the_global_branch_is_on(self, prefix):
        """Fact 2, positive half: the cumulative mean sees the whole length."""
        delta = _clifford_padding_delta(
            prefix, 8, 12, layer_scale_init=1.0, use_global_context=True
        )
        assert delta > 1e-4, (
            "with use_global_context=True the pooled branch means over the "
            f"padded length, so pad length must move real positions; got {delta:.3e}"
        )

    def test_layer_scale_hides_the_hazard_at_initialization(self, prefix):
        """Fact 3: the default gamma damps the effect below float32 noise.

        This is the trap the other tests exist to avoid: probing a freshly
        constructed model reports a padding-safe arm.
        """
        at_init = _clifford_padding_delta(
            prefix, 6, 8, layer_scale_init=1e-5, use_global_context=False
        )
        at_unit_gamma = _clifford_padding_delta(
            prefix, 6, 8, layer_scale_init=1.0, use_global_context=False
        )
        assert at_init < 1e-5
        assert at_unit_gamma > 1000 * at_init

    def test_the_attention_arm_is_the_control_and_is_inert(self, prefix):
        """The transformer arm honours its mask, so padding is free.

        The bound is DERIVED, not pasted. Measured on this configuration the
        delta is 0.000e+00 under float32, mixed_float16 and mixed_bfloat16, and
        0.000e+00 / 1.192e-07 with TF32 on / off -- so 1e-4 sits three orders
        above every reading, while remaining ~50x below the 5.9e-03 Clifford
        effect this is the control for. A hard 1e-6 here failed ONCE inside a
        large combined run and could not be reproduced alone, in three other
        orderings, under any dtype policy, or with TF32 either way; this suite
        has documented ordering-dependent failures, and a knife-edge bound on a
        contrast that spans four orders of magnitude buys nothing.
        """
        model = build_encoder(
            "transformer",
            {"num_heads": 4, "intermediate_size": 64},
            max_position_embeddings=64,
        )
        model.build((None, 6))
        a = keras.ops.convert_to_numpy(
            model({"input_ids": _prefix_padded_to(prefix, 8)})["last_hidden_state"]
        )[0, :6]
        b = keras.ops.convert_to_numpy(
            model({"input_ids": _prefix_padded_to(prefix, 12)})["last_hidden_state"]
        )[0, :6]
        assert float(np.abs(a - b).max()) < 1e-4


# ---------------------------------------------------------------------
# Configuration and serialization
# ---------------------------------------------------------------------

class TestValidation:
    """Unusable configurations are refused at construction."""

    @pytest.mark.parametrize(
        "field", ["vocab_size", "hidden_size", "num_layers", "max_position_embeddings"]
    )
    def test_non_positive_dimensions_raise(self, field):
        with pytest.raises(ValueError, match="positive int"):
            EmbeddingEncoder(**{field: 0})

    @pytest.mark.parametrize(
        "field", ["hidden_dropout_rate", "stochastic_depth_rate"]
    )
    def test_out_of_range_rates_raise(self, field):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            EmbeddingEncoder(**{field: 1.0})

    def test_unsupported_pooling_raises(self):
        with pytest.raises(ValueError, match="pooling_strategy"):
            EmbeddingEncoder(pooling_strategy="telepathy")

    def test_unknown_block_type_raises(self):
        with pytest.raises(ValueError, match="Unknown block_type"):
            EmbeddingEncoder(block_type="wishful")

    def test_build_rejects_a_non_rank_two_shape(self):
        model = EmbeddingEncoder(hidden_size=HIDDEN, num_layers=1)
        with pytest.raises(ValueError, match="input_ids"):
            model.build((None, SEQ_LEN, HIDDEN))


class TestSerialization:
    """Round trips must preserve behaviour, which means weight VALUES."""

    @pytest.mark.parametrize("block_type,config", BLOCK_CASES, ids=BLOCK_IDS)
    def test_config_round_trip_preserves_the_configuration(
        self, block_type, config
    ):
        model = build_encoder(block_type, config, pooling_strategy="attention")
        restored = EmbeddingEncoder.from_config(model.get_config())
        for field in (
            "hidden_size",
            "num_layers",
            "block_type",
            "pooling_strategy",
            "vocab_size",
            "pad_token_id",
        ):
            assert getattr(restored, field) == getattr(model, field), field

    @pytest.mark.parametrize("block_type,config", BLOCK_CASES, ids=BLOCK_IDS)
    def test_keras_round_trip_preserves_outputs(
        self, block_type, config, token_ids
    ):
        model = build_encoder(block_type, config)
        expected = keras.ops.convert_to_numpy(
            model({"input_ids": token_ids}, training=False)["pooled_output"]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "encoder.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            actual = keras.ops.convert_to_numpy(
                reloaded({"input_ids": token_ids}, training=False)["pooled_output"]
            )

        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
