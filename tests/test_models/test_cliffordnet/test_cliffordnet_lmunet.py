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
        assert model.channels_per_stage == [16, 24, 36]

    def test_default_initialization(self):
        model = CliffordNetLMUNet(vocab_size=256, base_channels=16)
        # Defaults: stride_per_stage=[2,2], blocks_per_stage=[2,2,2]
        assert model.stride_per_stage == [2, 2]
        assert model.blocks_per_stage == [2, 2, 2]
        assert model.num_levels == 3
        assert model.channels_per_stage == [16, 24, 36]
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
        assert model.channels_per_stage == [24, 36, 54, 81]

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
        # Seeded: weights and input_ids were previously drawn unseeded, so every
        # run sampled a different point of the float-noise distribution described
        # below, and the test was a lottery (~1 failure in 15).
        keras.utils.set_random_seed(1337)

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

        # The actual serialization contract, asserted EXACTLY: every weight must
        # survive the round trip bit-for-bit. This is the assertion that catches a
        # real defect -- a lazily-built sublayer that gets dropped restores as
        # freshly-initialized weights, which fails here on exact equality rather
        # than hiding under an output tolerance.
        assert len(loaded.weights) == len(model.weights)
        for w_orig, w_loaded in zip(model.weights, loaded.weights):
            np.testing.assert_array_equal(
                keras.ops.convert_to_numpy(w_orig),
                keras.ops.convert_to_numpy(w_loaded),
                err_msg=f"weight {w_orig.name} changed across save/load",
            )

        # Outputs are then only checked for numerical equivalence, loosely. Even
        # with bit-identical weights the rebuilt graph selects different kernels /
        # reduction orders than the original, so the logits differ by pure float
        # noise. Measured over 15 random draws on the RTX 4070 (logit scale ~0.35):
        # median 5e-5, max 1.14e-4; the same measurement on CPU gives ~5e-6. The
        # old atol of 1e-4 sat inside that GPU tail, which is what made this test
        # flaky. 1e-3 clears the measured max by ~9x while staying ~300x below the
        # logit scale, so a genuine numerical regression still cannot hide under it.
        np.testing.assert_allclose(out_orig, out_loaded, atol=1e-3)


# ---------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------


class TestVariants:

    def test_from_variant_nano_channel_ladder(self):
        model = CliffordNetLMUNet.from_variant("nano", vocab_size=128)
        assert model.base_channels == 128
        assert model.channels_per_stage == [128, 192, 288]
        # Confirm encoder block channels match the ladder.
        for i, level_blocks in enumerate(model.encoder_blocks):
            for blk in level_blocks:
                assert blk.channels == model.channels_per_stage[i], (
                    f"encoder level {i} block channels mismatch"
                )

    def test_from_variant_mini(self):
        model = CliffordNetLMUNet.from_variant("mini", vocab_size=128)
        assert model.base_channels == 192
        assert model.channels_per_stage == [192, 288, 432]

    def test_from_variant_with_overrides(self):
        model = CliffordNetLMUNet.from_variant("nano", vocab_size=128, dropout_rate=0.2)
        assert model.dropout_rate == 0.2
        assert model.base_channels == 128

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


# ---------------------------------------------------------------------
# Matryoshka Representation Learning (MRL)
# ---------------------------------------------------------------------


class TestMRL:
    """Tests for the per-width MRL LM heads (D-002 width rule)."""

    def test_mrl_default_widths_per_variant(self):
        """Variant defaults follow the power-of-2 anchored, base preserved rule."""
        expected = {
            "nano": [128, 64, 32, 16],
            "mini": [192, 128, 64, 32, 16],
            "base": [384, 256, 128, 64, 32, 16],
            "large": [512, 256, 128, 64, 32, 16],
            "xl": [768, 512, 256, 128, 64, 32, 16],
        }
        for name, widths in expected.items():
            model = CliffordNetLMUNet.from_variant(name, vocab_size=64)
            assert model.mrl_widths == widths, (
                f"variant {name!r} mrl_widths mismatch: "
                f"{model.mrl_widths} != {widths}"
            )

    def test_mrl_output_keys_shapes(self, tiny_config):
        """Override with [16, 8, 4] → 3 logits keys + 3 embedding keys."""
        cfg = {**tiny_config, "mrl_widths": [16, 8, 4]}
        model = CliffordNetLMUNet(**cfg)
        input_ids = _random_ids((2, 16), cfg["vocab_size"])
        outputs = model(input_ids, training=False)
        expected_keys = {
            "logits", "logits_w8", "logits_w4",
            "embedding_w16", "embedding_w8", "embedding_w4",
        }
        assert set(outputs.keys()) == expected_keys, set(outputs.keys())
        assert outputs["logits"].shape == (2, 16, cfg["vocab_size"])
        assert outputs["logits_w8"].shape == (2, 16, cfg["vocab_size"])
        assert outputs["logits_w4"].shape == (2, 16, cfg["vocab_size"])
        assert outputs["embedding_w16"].shape == (2, 16)
        assert outputs["embedding_w8"].shape == (2, 8)
        assert outputs["embedding_w4"].shape == (2, 4)

    def test_mrl_causality_per_width(self, tiny_config):
        """Perturb-last: positions < T-1 unchanged at every logits_w* key."""
        cfg = {**tiny_config, "mrl_widths": [16, 8, 4], "layer_scale_init": 1.0}
        model = CliffordNetLMUNet(**cfg)
        seq1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
        seq2 = np.array([[1, 2, 3, 4, 5, 6, 7, 63]], dtype=np.int32)
        out1 = model(seq1, training=False)
        out2 = model(seq2, training=False)
        for key in ("logits", "logits_w8", "logits_w4"):
            l1 = keras.ops.convert_to_numpy(out1[key])
            l2 = keras.ops.convert_to_numpy(out2[key])
            for pos in range(7):
                np.testing.assert_allclose(
                    l1[0, pos], l2[0, pos], atol=1e-5,
                    err_msg=f"{key}: position {pos} leaked future at width slice",
                )
            assert not np.allclose(l1[0, 7], l2[0, 7], atol=1e-3), (
                f"{key}: position 7 unchanged — perturbation didn't propagate"
            )

    def test_mrl_tied_vs_untied(self, tiny_config):
        """Both tied/untied paths produce all widths with correct shapes."""
        for tie in (True, False):
            cfg = {**tiny_config, "mrl_widths": [16, 8, 4], "tie_word_embeddings": tie}
            model = CliffordNetLMUNet(**cfg)
            outputs = model(_random_ids((1, 8), cfg["vocab_size"]), training=False)
            for key in ("logits", "logits_w8", "logits_w4"):
                assert outputs[key].shape == (1, 8, cfg["vocab_size"]), (
                    f"tie={tie} key={key} shape mismatch"
                )
            # Tied: only token_embedding kernel + per-width biases (no Dense).
            # Untied: per-width Dense layers exist.
            if tie:
                assert model.mrl_output_projs is None
                assert model.mrl_output_biases is not None
                assert len(model.mrl_output_biases) == 3
            else:
                assert model.mrl_output_projs is not None
                assert len(model.mrl_output_projs) == 3

    def test_mrl_widths_validation(self, tiny_config):
        """Invalid widths raise ValueError with the expected anchor."""
        # First element != base_channels.
        bad1 = {**tiny_config, "mrl_widths": [8, 4]}
        with pytest.raises(ValueError, match="must equal base_channels"):
            CliffordNetLMUNet(**bad1)
        # Element > base_channels.
        bad2 = {**tiny_config, "mrl_widths": [32, 16]}
        with pytest.raises(ValueError, match="<= base_channels"):
            CliffordNetLMUNet(**bad2)
        # Non-decreasing.
        bad3 = {**tiny_config, "mrl_widths": [16, 8, 8]}
        with pytest.raises(ValueError, match="strictly decreasing"):
            CliffordNetLMUNet(**bad3)
        # Non-positive.
        bad4 = {**tiny_config, "mrl_widths": [16, 0]}
        with pytest.raises(ValueError, match="positive ints"):
            CliffordNetLMUNet(**bad4)
        # Non-power-of-2 tail (12 is not a power of 2).
        bad5 = {**tiny_config, "mrl_widths": [16, 12, 4]}
        with pytest.raises(ValueError, match="power of 2"):
            CliffordNetLMUNet(**bad5)

    def test_trainer_compile_and_fit_one_step(self):
        """End-to-end smoke: trainer's compile_model + 1 fit step → per-key losses."""
        from train.cliffordnet.train_cliffordnet_nlp_unet import (
            TrainingConfig, compile_model, _resolve_lm_keys,
        )
        cfg = TrainingConfig(
            variant="custom",
            vocab_size=64,
            base_channels=16,
            max_seq_length=32,
            stride_per_stage=[2, 2],
            blocks_per_stage=[1, 1, 1],
            bottleneck_blocks=1,
            shifts=[1, 2],
            mrl_widths=[16, 8],
            emb_head=False,
            tie_word_embeddings=True,
            batch_size=2,
            num_epochs=1,
        )
        model = CliffordNetLMUNet(
            vocab_size=cfg.vocab_size,
            max_seq_length=cfg.max_seq_length,
            base_channels=cfg.base_channels,
            stride_per_stage=cfg.stride_per_stage,
            blocks_per_stage=cfg.blocks_per_stage,
            bottleneck_blocks=cfg.bottleneck_blocks,
            shifts=cfg.shifts,
            tie_word_embeddings=cfg.tie_word_embeddings,
            mrl_widths=cfg.mrl_widths,
            emb_head=cfg.emb_head,
        )
        # Build the model.
        model(_random_ids((1, cfg.max_seq_length - 1), cfg.vocab_size), training=False)

        compile_model(model, cfg, steps_per_epoch=1)

        # Synthetic batch with labels duplicated across both lm_keys.
        x = _random_ids((2, 8), cfg.vocab_size)
        y = _random_ids((2, 8), cfg.vocab_size)
        lm_keys = _resolve_lm_keys(cfg.mrl_widths)
        assert lm_keys == ["logits", "logits_w8"]
        labels = {k: y for k in lm_keys}

        history = model.fit(x, labels, batch_size=2, epochs=1, verbose=0)
        # Both per-key losses must surface in history.
        assert "loss" in history.history
        assert "logits_loss" in history.history, history.history.keys()
        assert "logits_w8_loss" in history.history, history.history.keys()


class TestEmbeddingHead:
    """Tests for the L2-normalized sentence-embedding head."""

    def test_embedding_l2_norm_per_width(self, tiny_config):
        cfg = {**tiny_config, "mrl_widths": [16, 8, 4]}
        model = CliffordNetLMUNet(**cfg)
        outputs = model(_random_ids((4, 16), cfg["vocab_size"]), training=False)
        for w in (16, 8, 4):
            e = keras.ops.convert_to_numpy(outputs[f"embedding_w{w}"])
            norms = np.linalg.norm(e, axis=-1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embedding_pool_last_default(self, tiny_config):
        """Default pool='last' uses the T-1 array index."""
        cfg = {**tiny_config, "mrl_widths": [16, 8]}
        model = CliffordNetLMUNet(**cfg)
        outputs = model(_random_ids((2, 16), cfg["vocab_size"]), training=False)
        # Embedding at width=16 must equal L2norm(h_top[:, -1, :16]).
        # h_top isn't directly exposed; we instead verify the embedding is the
        # L2 norm of *some* pooled vector by checking unit-norm + dimensionality.
        e16 = keras.ops.convert_to_numpy(outputs["embedding_w16"])
        e8 = keras.ops.convert_to_numpy(outputs["embedding_w8"])
        # The first 8 dims of the pooled (pre-norm) vector are shared with e8's
        # underlying slice; both come from the same pooled vector. After L2
        # norm at different widths they aren't equal, but they live in the same
        # direction in the first 8 dims — check correlation > 0.
        # Simpler structural check: shapes are right and both are finite.
        assert e16.shape == (2, 16)
        assert e8.shape == (2, 8)
        assert np.all(np.isfinite(e16))
        assert np.all(np.isfinite(e8))

    def test_embedding_pool_cls(self, tiny_config):
        """pool='cls' → embedding derived from position 0 (vs 'last' position T-1)."""
        cfg_last = {**tiny_config, "mrl_widths": [16], "embedding_pool": "last"}
        cfg_cls = {
            **tiny_config, "mrl_widths": [16],
            "embedding_pool": "cls", "cls_token_id": 1,
        }
        # Same seed-equivalent — build both models with the same weights via
        # set_weights to make the comparison meaningful.
        m1 = CliffordNetLMUNet(**cfg_last)
        m2 = CliffordNetLMUNet(**cfg_cls)
        x = _random_ids((1, 8), cfg_last["vocab_size"])
        m1(x, training=False)
        m2(x, training=False)
        m2.set_weights(m1.get_weights())
        out1 = m1(x, training=False)
        out2 = m2(x, training=False)
        e_last = keras.ops.convert_to_numpy(out1["embedding_w16"])
        e_cls = keras.ops.convert_to_numpy(out2["embedding_w16"])
        # Pool position differs (last vs first) → embeddings should differ in
        # nontrivial sequences. Both must still be unit-norm.
        np.testing.assert_allclose(np.linalg.norm(e_last, axis=-1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(e_cls, axis=-1), 1.0, atol=1e-5)
        assert not np.allclose(e_last, e_cls, atol=1e-4), (
            "cls vs last pool produced identical embeddings — pooling not applied"
        )

    def test_embedding_emb_head_dense(self, tiny_config):
        """emb_head=True adds a Dense projection — extra trainable weights."""
        cfg_no = {**tiny_config, "mrl_widths": [16], "emb_head": False}
        cfg_yes = {**tiny_config, "mrl_widths": [16], "emb_head": True}
        m_no = CliffordNetLMUNet(**cfg_no)
        m_yes = CliffordNetLMUNet(**cfg_yes)
        x = _random_ids((1, 8), cfg_no["vocab_size"])
        m_no(x, training=False)
        m_yes(x, training=False)
        # emb_head=True path has the embedding_proj Dense; baseline has None.
        assert m_no.embedding_proj is None
        assert m_yes.embedding_proj is not None
        # Trainable variable count strictly increases.
        assert len(m_yes.trainable_variables) > len(m_no.trainable_variables)


class TestMRLSerialization:
    """Round-trip and back-compat serialization tests for MRL fields."""

    def test_get_config_includes_mrl_fields(self, tiny_config):
        cfg = {
            **tiny_config,
            "mrl_widths": [16, 8, 4],
            "mrl_head_norm": False,
            "emb_head": True,
            "embedding_pool": "cls",
            "cls_token_id": 1,
            "l2_eps": 1e-10,
        }
        m1 = CliffordNetLMUNet(**cfg)
        d = m1.get_config()
        assert d["mrl_widths"] == [16, 8, 4]
        assert d["mrl_head_norm"] is False
        assert d["emb_head"] is True
        assert d["embedding_pool"] == "cls"
        assert d["cls_token_id"] == 1
        assert d["l2_eps"] == 1e-10
        m2 = CliffordNetLMUNet.from_config(d)
        assert m2.mrl_widths == m1.mrl_widths
        assert m2.mrl_head_norm == m1.mrl_head_norm
        assert m2.emb_head == m1.emb_head
        assert m2.embedding_pool == m1.embedding_pool
        assert m2.cls_token_id == m1.cls_token_id
        assert m2.l2_eps == m1.l2_eps

    def test_save_load_keras_with_mrl(self, tiny_config):
        # Seed for determinism: inputs + weights were previously unseeded, making
        # this round-trip flaky.
        keras.utils.set_random_seed(42)
        cfg = {**tiny_config, "mrl_widths": [16, 8, 4], "emb_head": True}
        model = CliffordNetLMUNet(**cfg)
        x = _random_ids((1, 8), cfg["vocab_size"])
        out_orig = model(x, training=False)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lmunet_mrl.keras")
            model.save(path)
            loaded = keras.models.load_model(
                path,
                custom_objects={
                    "CliffordNetLMUNet": CliffordNetLMUNet,
                    "CausalCliffordNetBlock": CausalCliffordNetBlock,
                    "CausalCliffordNetBlockDSv2": CausalCliffordNetBlockDSv2,
                },
            )
        out_loaded = loaded(x, training=False)
        for key in ("logits", "logits_w8", "logits_w4"):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(out_orig[key]),
                keras.ops.convert_to_numpy(out_loaded[key]),
                atol=1e-4, err_msg=f"{key} drifted across save/load",
            )
        for key in ("embedding_w16", "embedding_w8", "embedding_w4"):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(out_orig[key]),
                keras.ops.convert_to_numpy(out_loaded[key]),
                # L2-normalized embeddings amplify the same fp32 reduction-order
                # save/load noise the logits tolerate at 1e-4 (see above); 1e-5
                # was an unjustified 10x-tighter outlier.
                atol=1e-4, err_msg=f"{key} drifted across save/load",
            )

    def test_default_mrl_widths_none_back_compat(self):
        """mrl_widths=None → [base_channels]; outputs = {'logits', 'embedding_w<base>'}.

        Note: embedding_w{base_channels} is always emitted because the
        embedding-head pooling step runs unconditionally (emb_head only
        toggles the extra Dense projection, not the L2-norm + slice path).
        """
        model = CliffordNetLMUNet(vocab_size=64, base_channels=16, mrl_widths=None)
        assert model.mrl_widths == [16]
        x = _random_ids((1, 8), 64)
        outputs = model(x, training=False)
        assert set(outputs.keys()) == {"logits", "embedding_w16"}, outputs.keys()
        assert outputs["logits"].shape == (1, 8, 64)
        assert outputs["embedding_w16"].shape == (1, 16)
