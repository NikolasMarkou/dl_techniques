"""Tests for the CliffordNetLMUNet causal U-Net language model.

Mirrors test_cliffordnet_lm.py 1:1 (initialization, forward, causality,
serialization, variants, gradient flow) and adds:

- TestCausalUpsampleHelper — unit-level micro-tests for the
  ``_causal_upsample`` helper (D-007). Isolating the helper keeps a
  future regression in the upsample path from surfacing only as a
  confusing end-to-end failure (the "failed defense" of D-006).
- TestForwardPass — non-multiple seq_len exercising right-pad + crop
  (D-002).
- TestCausality — perturb-last and perturb-middle, with an extra case
  for non-multiple seq_len.
- TestSerialization — adds a save/load .keras round-trip with logit
  comparison.
- TestVariants — adds channels_per_stage assertion (D-005).
"""

import os
import tempfile

import pytest
import numpy as np
import keras
import tensorflow as tf

from dl_techniques.models.cliffordnet.lmunet import (
    CliffordNetLMUNet,
    _causal_upsample,
)
from dl_techniques.layers.geometric.clifford_block import (
    CausalCliffordNetBlock,
    CausalCliffordNetBlockDSv2,
)


def _random_ids(shape, vocab_size):
    return np.random.randint(0, vocab_size, shape).astype(np.int32)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def tiny_config():
    """Minimal CliffordNetLMUNet config for fast testing.

    nano-like topology (stride [2, 2], total_stride=4, num_levels=3) with
    very small channels and a single block per stage.
    """
    return {
        "vocab_size": 64,
        "max_seq_length": 32,
        "base_channels": 16,
        "stride_per_stage": [2, 2],
        "blocks_per_stage": [1, 1, 1],
        "bottleneck_blocks": 1,
        "shifts": [1, 2],
        "dropout_rate": 0.0,
        "stochastic_depth_rate": 0.0,
    }


# ---------------------------------------------------------------------
# Causal-upsample helper (D-007) — unit tests
# ---------------------------------------------------------------------


class TestCausalUpsampleHelper:
    """Isolated tests of _causal_upsample (the D-007 fix).

    A future regression here should surface as a focused failure rather
    than a confusing end-to-end causality error (addresses the failed
    defense identified in D-006).
    """

    def test_stride_one_is_noop(self):
        x = np.random.default_rng(0).standard_normal((1, 1, 4, 2)).astype(np.float32)
        y = keras.ops.convert_to_numpy(_causal_upsample(x, 1))
        np.testing.assert_array_equal(y, x)

    def test_preserves_shape(self):
        x = np.random.default_rng(0).standard_normal((2, 1, 8, 3)).astype(np.float32)
        y = keras.ops.convert_to_numpy(_causal_upsample(x, 2))
        assert y.shape == x.shape

    def test_stride_two_zero_pads_first_position(self):
        # After right-shift by 1, the first column must be zero.
        x = np.ones((1, 1, 4, 2), dtype=np.float32)
        y = keras.ops.convert_to_numpy(_causal_upsample(x, 2))
        np.testing.assert_array_equal(y[:, :, 0, :], 0.0)
        # All other columns must equal the original (shifted right).
        np.testing.assert_array_equal(y[:, :, 1:, :], x[:, :, :-1, :])

    def test_perturb_pooled_cell_only_affects_later_positions(self):
        """Perturbing pooled cell j must only affect outputs k where the
        post-shift mapping (k - (s-1)) // s >= j, i.e. k >= j*s + (s-1).
        """
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 1, 4, 2)).astype(np.float32)
        s = 2
        up = keras.layers.UpSampling2D(size=(1, s), interpolation="nearest")
        y0 = keras.ops.convert_to_numpy(_causal_upsample(up(x), s))

        # Perturb pooled cell j=2 (input position 2).
        x_p = x.copy()
        x_p[0, 0, 2, :] += 5.0
        y1 = keras.ops.convert_to_numpy(_causal_upsample(up(x_p), s))

        diff = np.max(np.abs(y0 - y1), axis=-1).reshape(-1)
        # Output positions < j*s + (s-1) = 5 must be byte-identical.
        np.testing.assert_array_equal(diff[:5], 0.0)
        # Output position 5 onward must reflect the perturbation.
        assert diff[5] > 0.0

    def test_perturb_last_pooled_cell(self):
        """Perturbing the last pooled cell must leave outputs < W-1 zero."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal((1, 1, 4, 2)).astype(np.float32)
        s = 2
        up = keras.layers.UpSampling2D(size=(1, s), interpolation="nearest")
        y0 = keras.ops.convert_to_numpy(_causal_upsample(up(x), s))

        x_p = x.copy()
        x_p[0, 0, 3, :] += 5.0  # last pooled cell
        y1 = keras.ops.convert_to_numpy(_causal_upsample(up(x_p), s))

        diff = np.max(np.abs(y0 - y1), axis=-1).reshape(-1)
        # Only the very last upsampled position (k = 7 = 3*2 + 1) must differ.
        np.testing.assert_array_equal(diff[:7], 0.0)
        assert diff[7] > 0.0


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------


class TestInitialization:

    def test_basic_initialization(self, tiny_config):
        model = CliffordNetLMUNet(**tiny_config)
        assert model.vocab_size == 64
        assert model.base_channels == 16
        assert model.stride_per_stage == [2, 2]
        assert model.blocks_per_stage == [1, 1, 1]
        assert model.bottleneck_blocks == 1
        assert model.shifts == [1, 2]
        assert model.num_levels == 3
        assert model.total_stride == 4
        assert model.channels_per_stage == [16, 32, 64]

    def test_default_initialization(self):
        model = CliffordNetLMUNet(vocab_size=256, base_channels=16)
        # Defaults: stride_per_stage=[2,2], blocks_per_stage=[2,2,2]
        assert model.stride_per_stage == [2, 2]
        assert model.blocks_per_stage == [2, 2, 2]
        assert model.num_levels == 3
        assert model.channels_per_stage == [16, 32, 64]
        assert model.tie_word_embeddings is True

    def test_custom_initialization(self):
        model = CliffordNetLMUNet(
            vocab_size=128,
            max_seq_length=64,
            base_channels=24,
            stride_per_stage=[2, 2, 2],
            blocks_per_stage=[1, 1, 1, 1],
            bottleneck_blocks=2,
            shifts=[1, 2, 4],
            cli_mode="wedge",
            ctx_mode="abs",
            use_global_context=True,
            layer_scale_init=1e-3,
        )
        assert model.cli_mode == "wedge"
        assert model.ctx_mode == "abs"
        assert model.use_global_context is True
        assert model.shifts == [1, 2, 4]
        assert model.num_levels == 4
        assert model.total_stride == 8
        assert model.channels_per_stage == [24, 48, 96, 192]

    def test_invalid_blocks_per_stage_length(self, tiny_config):
        bad = {**tiny_config, "blocks_per_stage": [1, 1]}  # should be 3
        with pytest.raises(ValueError, match="blocks_per_stage"):
            CliffordNetLMUNet(**bad)

    def test_invalid_shifts_too_large(self, tiny_config):
        bad = {**tiny_config, "shifts": [1, 16]}  # 16 >= base_channels=16
        with pytest.raises(ValueError, match="max\\(shifts\\)"):
            CliffordNetLMUNet(**bad)


# ---------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------


class TestForwardPass:

    def test_forward_pass_basic(self, tiny_config):
        model = CliffordNetLMUNet(**tiny_config)
        input_ids = _random_ids((2, 16), tiny_config["vocab_size"])
        outputs = model(input_ids, training=False)
        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 16, tiny_config["vocab_size"])

    def test_forward_pass_batch_size_one(self, tiny_config):
        model = CliffordNetLMUNet(**tiny_config)
        input_ids = _random_ids((1, 8), tiny_config["vocab_size"])
        outputs = model(input_ids, training=False)
        assert outputs["logits"].shape == (1, 8, tiny_config["vocab_size"])

    def test_forward_pass_full_sequence(self, tiny_config):
        model = CliffordNetLMUNet(**tiny_config)
        seq_len = tiny_config["max_seq_length"]
        input_ids = _random_ids((1, seq_len), tiny_config["vocab_size"])
        outputs = model(input_ids, training=False)
        assert outputs["logits"].shape == (1, seq_len, tiny_config["vocab_size"])

    def test_forward_pass_with_global_context(self, tiny_config):
        cfg = {**tiny_config, "use_global_context": True}
        model = CliffordNetLMUNet(**cfg)
        input_ids = _random_ids((2, 16), cfg["vocab_size"])
        outputs = model(input_ids, training=False)
        assert outputs["logits"].shape == (2, 16, cfg["vocab_size"])

    def test_forward_pass_non_multiple_seq_len(self, tiny_config):
        """seq_len NOT divisible by total_stride exercises right-pad + crop."""
        model = CliffordNetLMUNet(**tiny_config)
        # total_stride=4; pick 7 which is not divisible.
        input_ids = _random_ids((1, 7), tiny_config["vocab_size"])
        outputs = model(input_ids, training=False)
        assert outputs["logits"].shape == (1, 7, tiny_config["vocab_size"])

    def test_untied_lm_head(self, tiny_config):
        cfg = {**tiny_config, "tie_word_embeddings": False}
        model = CliffordNetLMUNet(**cfg)
        input_ids = _random_ids((1, 8), cfg["vocab_size"])
        outputs = model(input_ids, training=False)
        assert outputs["logits"].shape == (1, 8, cfg["vocab_size"])


# ---------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------


class TestCausality:

    def test_causality_perturb_last_position(self, tiny_config):
        cfg = {**tiny_config, "layer_scale_init": 1.0}
        model = CliffordNetLMUNet(**cfg)
        seq1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
        seq2 = np.array([[1, 2, 3, 4, 5, 6, 7, 63]], dtype=np.int32)

        out1 = model(seq1, training=False)
        out2 = model(seq2, training=False)
        l1 = keras.ops.convert_to_numpy(out1["logits"])
        l2 = keras.ops.convert_to_numpy(out2["logits"])
        for pos in range(7):
            np.testing.assert_allclose(
                l1[0, pos], l2[0, pos], atol=1e-5,
                err_msg=f"Position {pos} changed when only position 7 changed",
            )
        assert not np.allclose(l1[0, 7], l2[0, 7], atol=1e-3)

    def test_causality_perturb_middle_position(self, tiny_config):
        cfg = {**tiny_config, "layer_scale_init": 1.0}
        model = CliffordNetLMUNet(**cfg)
        seq1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
        seq2 = np.array([[1, 2, 3, 4, 33, 6, 7, 8]], dtype=np.int32)

        out1 = model(seq1, training=False)
        out2 = model(seq2, training=False)
        l1 = keras.ops.convert_to_numpy(out1["logits"])
        l2 = keras.ops.convert_to_numpy(out2["logits"])
        for pos in range(4):
            np.testing.assert_allclose(
                l1[0, pos], l2[0, pos], atol=1e-5,
                err_msg=f"Position {pos} affected by change at position 4",
            )
        assert not np.allclose(l1[0, 4], l2[0, 4], atol=1e-3)

    def test_causality_non_multiple_seq_len(self, tiny_config):
        """Causality holds when seq_len % total_stride != 0 (pad+crop path)."""
        cfg = {**tiny_config, "layer_scale_init": 1.0}
        model = CliffordNetLMUNet(**cfg)
        # total_stride=4; use seq_len=7.
        seq1 = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        seq2 = np.array([[1, 2, 3, 4, 5, 6, 33]], dtype=np.int32)

        out1 = model(seq1, training=False)
        out2 = model(seq2, training=False)
        l1 = keras.ops.convert_to_numpy(out1["logits"])
        l2 = keras.ops.convert_to_numpy(out2["logits"])
        for pos in range(6):
            np.testing.assert_allclose(
                l1[0, pos], l2[0, pos], atol=1e-5,
                err_msg=f"non-mult: position {pos} changed when only position 6 changed",
            )

    def test_causality_with_global_context(self, tiny_config):
        cfg = {**tiny_config, "layer_scale_init": 1.0, "use_global_context": True}
        model = CliffordNetLMUNet(**cfg)
        seq1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
        seq2 = np.array([[1, 2, 3, 4, 5, 6, 7, 63]], dtype=np.int32)
        out1 = model(seq1, training=False)
        out2 = model(seq2, training=False)
        l1 = keras.ops.convert_to_numpy(out1["logits"])
        l2 = keras.ops.convert_to_numpy(out2["logits"])
        for pos in range(7):
            np.testing.assert_allclose(
                l1[0, pos], l2[0, pos], atol=1e-5,
                err_msg=f"global-ctx: position {pos} leaks future info",
            )


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


class TestSerialization:

    def test_get_config(self, tiny_config):
        model = CliffordNetLMUNet(**tiny_config)
        cfg = model.get_config()
        assert cfg["vocab_size"] == tiny_config["vocab_size"]
        assert cfg["base_channels"] == tiny_config["base_channels"]
        assert cfg["stride_per_stage"] == tiny_config["stride_per_stage"]
        assert cfg["blocks_per_stage"] == tiny_config["blocks_per_stage"]
        assert cfg["bottleneck_blocks"] == tiny_config["bottleneck_blocks"]
        assert cfg["shifts"] == tiny_config["shifts"]

    def test_from_config_roundtrip(self, tiny_config):
        m1 = CliffordNetLMUNet(**tiny_config)
        cfg = m1.get_config()
        m2 = CliffordNetLMUNet.from_config(cfg)
        assert m2.vocab_size == m1.vocab_size
        assert m2.base_channels == m1.base_channels
        assert m2.stride_per_stage == m1.stride_per_stage
        assert m2.blocks_per_stage == m1.blocks_per_stage
        assert m2.bottleneck_blocks == m1.bottleneck_blocks
        assert m2.shifts == m1.shifts
        assert m2.channels_per_stage == m1.channels_per_stage

    def test_compute_output_shape(self, tiny_config):
        model = CliffordNetLMUNet(**tiny_config)
        shapes = model.compute_output_shape((None, 16))
        assert shapes["logits"] == (None, 16, tiny_config["vocab_size"])

    def test_save_load_keras(self, tiny_config):
        model = CliffordNetLMUNet(**tiny_config)
        input_ids = _random_ids((1, 8), tiny_config["vocab_size"])
        out_orig = keras.ops.convert_to_numpy(model(input_ids, training=False)["logits"])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lmunet.keras")
            model.save(path)
            loaded = keras.models.load_model(
                path,
                custom_objects={
                    "CliffordNetLMUNet": CliffordNetLMUNet,
                    "CausalCliffordNetBlock": CausalCliffordNetBlock,
                    "CausalCliffordNetBlockDSv2": CausalCliffordNetBlockDSv2,
                },
            )
        out_loaded = keras.ops.convert_to_numpy(loaded(input_ids, training=False)["logits"])
        # atol=1e-4: fp32 reduction-order noise across save/load on GPU (XLA)
        # can exceed 1e-5 even though all weights are bit-identical. Verified
        # by inspection that the diff is uniform float-noise (max ~5e-5 on
        # logits of magnitude ~0.3), not a serialization logic error.
        np.testing.assert_allclose(out_orig, out_loaded, atol=1e-4)


# ---------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------


class TestVariants:

    def test_from_variant_nano_channel_ladder(self):
        model = CliffordNetLMUNet.from_variant("nano", vocab_size=128)
        assert model.base_channels == 64
        assert model.channels_per_stage == [64, 128, 256]
        # Confirm encoder block channels match the ladder.
        for i, level_blocks in enumerate(model.encoder_blocks):
            for blk in level_blocks:
                assert blk.channels == model.channels_per_stage[i], (
                    f"encoder level {i} block channels mismatch"
                )

    def test_from_variant_mini(self):
        model = CliffordNetLMUNet.from_variant("mini", vocab_size=128)
        assert model.base_channels == 96
        assert model.channels_per_stage == [96, 192, 384]

    def test_from_variant_with_overrides(self):
        model = CliffordNetLMUNet.from_variant("nano", vocab_size=128, dropout_rate=0.2)
        assert model.dropout_rate == 0.2
        assert model.base_channels == 64

    def test_from_variant_unknown(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            CliffordNetLMUNet.from_variant("nonexistent", vocab_size=128)

    def test_all_variants_have_required_keys(self):
        required = {
            "base_channels", "stride_per_stage", "blocks_per_stage",
            "bottleneck_blocks", "shifts",
        }
        for name, cfg in CliffordNetLMUNet.MODEL_VARIANTS.items():
            for k in required:
                assert k in cfg, f"Variant '{name}' missing '{k}'"


# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------


class TestGradientFlow:

    def test_gradient_flow(self, tiny_config):
        model = CliffordNetLMUNet(**tiny_config)
        input_ids = _random_ids((2, 8), tiny_config["vocab_size"])
        labels = _random_ids((2, 8), tiny_config["vocab_size"])

        with tf.GradientTape() as tape:
            outputs = model(input_ids, training=True)
            loss = keras.losses.sparse_categorical_crossentropy(
                labels, outputs["logits"], from_logits=True,
            )
            loss = keras.ops.mean(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        for var, grad in zip(model.trainable_variables, grads):
            assert grad is not None, f"No gradient for {var.name}"
