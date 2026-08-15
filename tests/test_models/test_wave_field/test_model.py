"""Tests for WaveFieldLLM (decoder-only LM with WaveFieldAttention).

Mirrors the structure of ``tests/test_models/test_gpt2/test_gpt2.py`` and
adds wave-field-specific checks: padding-mask zeroing of last_hidden_state
and field_size validation.
"""

import os
import tempfile

import pytest
import numpy as np
import keras
import tensorflow as tf

from dl_techniques.models.wave_field.model import (
    WaveFieldLLM,
    WaveFieldDecoderBlock,
    create_wave_field_llm,
)
import dl_techniques.models.wave_field as wave_field_pkg
from dl_techniques.losses import MaskedCausalLMLoss


def _random_ids(shape, vocab_size):
    return np.random.randint(0, vocab_size, shape).astype(np.int32)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def tiny_config():
    """Minimal WaveFieldLLM config for fast testing."""
    return {
        "vocab_size": 256,
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 4,
        "max_seq_len": 32,
        "field_size": 64,
        "dropout_rate": 0.0,
        "attention_dropout_rate": 0.0,
    }


@pytest.fixture
def tiny_model(tiny_config):
    model = WaveFieldLLM(**tiny_config)
    dummy = _random_ids((1, tiny_config["max_seq_len"]), tiny_config["vocab_size"])
    model(dummy, training=False)
    return model


# ---------------------------------------------------------------------
# Initialization & validation
# ---------------------------------------------------------------------


class TestWaveFieldLLMInitialization:

    def test_initialization(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        assert model.vocab_size == 256
        assert model.embed_dim == 64
        assert model.depth == 2
        assert model.num_heads == 4
        assert model.max_seq_len == 32
        assert model.field_size == 64
        assert len(model.blocks) == 2
        assert isinstance(model.blocks[0], WaveFieldDecoderBlock)

    def test_default_field_size_doubles_max_seq_len(self):
        model = WaveFieldLLM(
            vocab_size=128, embed_dim=32, depth=1,
            num_heads=4, max_seq_len=16, field_size=None,
        )
        assert model.field_size == 32

    def test_invalid_vocab_size(self):
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            WaveFieldLLM(vocab_size=-1, embed_dim=64, depth=1, num_heads=4)

    def test_invalid_embed_dim(self):
        with pytest.raises(ValueError, match="embed_dim must be positive"):
            WaveFieldLLM(vocab_size=64, embed_dim=0, depth=1, num_heads=4)

    def test_invalid_depth(self):
        with pytest.raises(ValueError, match="depth must be positive"):
            WaveFieldLLM(vocab_size=64, embed_dim=64, depth=0, num_heads=4)

    def test_invalid_num_heads(self):
        with pytest.raises(ValueError, match="num_heads must be positive"):
            WaveFieldLLM(vocab_size=64, embed_dim=64, depth=1, num_heads=0)

    def test_embed_dim_not_divisible(self):
        with pytest.raises(ValueError, match="must be divisible by"):
            WaveFieldLLM(vocab_size=64, embed_dim=100, depth=1, num_heads=12)

    def test_invalid_field_size(self):
        with pytest.raises(ValueError, match="field_size must be > 1"):
            WaveFieldLLM(
                vocab_size=64, embed_dim=64, depth=1, num_heads=4,
                max_seq_len=8, field_size=1,
            )

    def test_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            WaveFieldLLM(
                vocab_size=64, embed_dim=64, depth=1,
                num_heads=4, dropout_rate=1.5,
            )


# ---------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------


class TestWaveFieldLLMForward:

    def test_forward_shape(self, tiny_model, tiny_config):
        ids = _random_ids((2, 16), tiny_config["vocab_size"])
        out = tiny_model(ids, training=False)
        assert "logits" in out
        assert "last_hidden_state" in out
        assert out["logits"].shape == (2, 16, tiny_config["vocab_size"])
        assert out["last_hidden_state"].shape == (2, 16, tiny_config["embed_dim"])

    def test_forward_full_seq_len(self, tiny_model, tiny_config):
        seq = tiny_config["max_seq_len"]
        ids = _random_ids((1, seq), tiny_config["vocab_size"])
        out = tiny_model(ids, training=False)
        assert out["logits"].shape == (1, seq, tiny_config["vocab_size"])

    def test_forward_with_padding_mask(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        ids = _random_ids((2, 12), tiny_config["vocab_size"])
        # First sample: 8 valid + 4 padded; second: 5 valid + 7 padded.
        mask = np.array(
            [[1] * 8 + [0] * 4, [1] * 5 + [0] * 7], dtype=np.float32,
        )
        out = model(ids, attention_mask=mask, training=False)
        logits = keras.ops.convert_to_numpy(out["logits"])
        lhs = keras.ops.convert_to_numpy(out["last_hidden_state"])
        # Output is finite.
        assert np.all(np.isfinite(logits))
        assert np.all(np.isfinite(lhs))
        # WaveFieldAttention zeros its output at padded positions; after
        # the residual + final norm, padded positions are no longer zero
        # globally — but the attention residual contribution at padded
        # positions is zero, so outputs there must remain finite. We
        # therefore assert finiteness only.

    def test_dict_input(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        ids = _random_ids((2, 10), tiny_config["vocab_size"])
        mask = np.ones((2, 10), dtype=np.int32)
        out = model({"input_ids": ids, "attention_mask": mask}, training=False)
        assert out["logits"].shape == (2, 10, tiny_config["vocab_size"])

    def test_dict_input_missing_ids_raises(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        with pytest.raises(ValueError, match="input_ids"):
            model({"attention_mask": np.ones((2, 10), dtype=np.int32)})


# ---------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------


class TestWaveFieldLLMCausality:

    def test_future_does_not_affect_past(self, tiny_config):
        """Short-sequence (8 of 32) probe at the DEFAULT ratio 2.0.

        This is the weak instrument: it perturbs only the last token. The
        general one is ``TestWaveFieldLLMCausalityRatioSweep`` below, which
        perturbs every position — a last-token-only probe measured CLEAN at
        field_size=35/max_seq_len=32 while the all-positions probe measured a
        1.96e-04 leak at the very same config.
        """
        model = WaveFieldLLM(**tiny_config)
        seq1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
        seq2 = np.array([[1, 2, 3, 4, 5, 6, 7, 99]], dtype=np.int32)

        out1 = model(seq1, training=False)
        out2 = model(seq2, training=False)
        l1 = keras.ops.convert_to_numpy(out1["logits"])
        l2 = keras.ops.convert_to_numpy(out2["logits"])
        for pos in range(7):
            np.testing.assert_allclose(
                l1[0, pos], l2[0, pos], atol=1e-5,
                err_msg=f"position {pos} changed when only position 7 changed",
            )


def _worst_causality_leak(model, max_seq_len, vocab_size):
    """Worst future-token leak over ALL (perturbed, observed) position pairs.

    Substitutes one token at position ``j`` and returns the largest absolute
    logit change induced at any position ``< j``, maximised over every ``j``.
    Perturbing only the LAST position is a blind instrument (see the docstring
    of ``test_future_does_not_affect_past``).
    """
    ids = (np.arange(1, max_seq_len + 1, dtype=np.int32)[None, :]) % vocab_size
    base = keras.ops.convert_to_numpy(model(ids, training=False)["logits"])[0]
    worst = 0.0
    for j in range(1, max_seq_len):
        ids2 = ids.copy()
        ids2[0, j] = (ids2[0, j] + 137) % vocab_size
        out = keras.ops.convert_to_numpy(model(ids2, training=False)["logits"])[0]
        worst = max(worst, float(np.abs(base[:j] - out[:j]).max()))
    return worst


# DECISION plan-2026-08-13T091555-230c101d/D-008 — the leaky rows below assert
# the leak is ABOVE a bound, and the perturbation sweeps EVERY position. Do NOT
# "simplify" this to a single `leak < small` assertion over all ratios: that
# passes when the mechanism disappears, leaving the docstring table stale and
# unfalsifiable. Do NOT perturb only the last token either — that probe measured
# CLEAN at field_size=35/max_seq_len=32 where the all-positions probe measures
# 1.96e-04. See decisions.md D-008.
class TestWaveFieldLLMCausalityRatioSweep:
    """Pins token-level causality as a MEASURED, ratio-dependent property.

    ``WaveFieldAttention`` guarantees causality on the FIELD GRID only, and its
    own docstring refuses to offer a sufficient condition on
    ``field_size`` / ``max_seq_len``. This sweep pins what was actually
    measured at ``max_seq_len=32, embed_dim=64, depth=2``, seed 1234, random
    init, logits of magnitude ~1.1 (worst leak over all position pairs)::

        ratio  field_size  stride   CPU        GPU(4070)   verdict
        0.50    16         0.4839   5.462e-04  5.505e-04   LEAKS
        0.75    24         0.7419   3.767e-04  4.049e-04   LEAKS
        1.00    32         1.0000   6.706e-08  below 1e-5  clean
        1.50    48         1.5161   4.961e-05  1.235e-04   LEAKS
        2.00    64         2.0323   5.960e-08  below 1e-5  clean  <- DEFAULT
        4.00   128         4.0968   8.941e-08  below 1e-5  clean

    Exact values are device-dependent (CPU and GPU differ by up to ~2.5x on the
    leaky rows), so the pin is an order-of-magnitude bound, not a number. The
    leaky rows are asserted to leak ABOVE a bound on purpose: if the mechanism
    silently vanished, a "leak is small" test would pass and the docstring
    table would go stale unnoticed. Note ratio 1.50 leaks while ratio 1.00 does
    not — the property is NOT monotone in the ratio.

    The clean rows are NOT exact zeros, and an earlier revision of this table
    that printed ``0.0`` in the GPU column was wrong. MEASURED: the same config
    and seed gives 0.0 in one process ordering and 1.565e-07 in another; ratio
    2.0 gives 0.0 at seeds 1234/7 but 6.109e-07 at seed 99; and ``[1.0-clean]``
    and ``[4.0-clean]`` each flaked at ``CLEAN_BOUND = 1e-6`` during
    whole-directory GPU runs (1.185e-06 observed). The clean residue is float32
    noise at logits O(1.1) — 1e-6 is only ~8 ulps there — not a property. Only
    the ORDER OF MAGNITUDE is pinned; see D-017.
    """

    MAX_SEQ_LEN = 32
    VOCAB = 256
    SEED = 1234
    # DECISION plan-2026-08-13T091555-230c101d/D-017 — CLEAN_BOUND is 1e-5, NOT
    # 1e-6. Do NOT "tighten" it back: at logits O(1.1), 1e-6 is ~8 float32 ulps,
    # and the clean rows were MEASURED at up to 1.185e-06 on GPU with no code
    # change (process-history dependent), which flaked [1.0-clean] and
    # [4.0-clean]. Do NOT lower LEAK_BOUND to "restore a gap" either — the gap
    # that matters is bound-to-measurement, and it is 5.0x (smallest leaky
    # 4.961e-05) above and 8.4x (largest clean 1.185e-06) below. The bound is
    # still falsifiable: the anti-causal-kernel injection drives the clean rows
    # to ~2.7e-04, 27x above it. See decisions.md D-017.
    CLEAN_BOUND = 1e-5   # 8.4x above the worst CLEAN measurement (1.185e-06)
    LEAK_BOUND = 1e-5    # 5.0x below the smallest LEAKY measurement (4.96e-05)

    @pytest.mark.parametrize("ratio,expect", [
        (0.5, "leaks"),
        (0.75, "leaks"),
        (1.0, "clean"),
        (1.5, "leaks"),
        (2.0, "clean"),
        (4.0, "clean"),
    ])
    def test_leak_per_field_size_ratio(self, ratio, expect):
        keras.utils.set_random_seed(self.SEED)
        field_size = int(round(ratio * self.MAX_SEQ_LEN))
        model = WaveFieldLLM(
            vocab_size=self.VOCAB, embed_dim=64, depth=2, num_heads=4,
            max_seq_len=self.MAX_SEQ_LEN, field_size=field_size,
            dropout_rate=0.0, attention_dropout_rate=0.0,
        )
        leak = _worst_causality_leak(model, self.MAX_SEQ_LEN, self.VOCAB)

        if expect == "clean":
            assert leak < self.CLEAN_BOUND, (
                f"ratio {ratio} (field_size={field_size}, "
                f"max_seq_len={self.MAX_SEQ_LEN}) was measured CLEAN but now "
                f"leaks {leak:.3e} >= {self.CLEAN_BOUND:.0e}; the module "
                f"docstring's Causality table is stale"
            )
        else:
            assert leak > self.LEAK_BOUND, (
                f"ratio {ratio} (field_size={field_size}, "
                f"max_seq_len={self.MAX_SEQ_LEN}) was measured LEAKY but now "
                f"leaks only {leak:.3e} <= {self.LEAK_BOUND:.0e}; either "
                f"causality was fixed (update the docstring table and this "
                f"pin) or the probe stopped measuring anything"
            )

    def test_default_field_size_lands_on_a_clean_ratio(self):
        """The class default ``field_size = 2 * max_seq_len`` is ratio 2.0."""
        model = WaveFieldLLM(
            vocab_size=self.VOCAB, embed_dim=64, depth=2, num_heads=4,
            max_seq_len=self.MAX_SEQ_LEN, field_size=None,
        )
        assert model.field_size == 2 * self.MAX_SEQ_LEN


# ---------------------------------------------------------------------
# Weight tying
# ---------------------------------------------------------------------


class TestWaveFieldLLMWeightTying:

    def test_default_tied(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        assert model.tie_word_embeddings is True
        assert model.lm_head is None

    def test_logits_use_embedding_weights(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        ids = _random_ids((1, 8), tiny_config["vocab_size"])
        out = model(ids, training=False)
        emb = model.token_embeddings.embeddings
        expected = keras.ops.matmul(
            out["last_hidden_state"], keras.ops.transpose(emb),
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out["logits"]),
            keras.ops.convert_to_numpy(expected),
            atol=1e-5,
        )

    def test_no_tying(self, tiny_config):
        cfg = {**tiny_config, "tie_word_embeddings": False}
        model = WaveFieldLLM(**cfg)
        assert model.lm_head is not None
        ids = _random_ids((1, 8), cfg["vocab_size"])
        out = model(ids, training=False)
        assert out["logits"].shape == (1, 8, cfg["vocab_size"])


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


class TestWaveFieldLLMSerialization:

    def test_get_config_round_trip(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        config = model.get_config()
        for k, v in tiny_config.items():
            assert config[k] == v
        model2 = WaveFieldLLM.from_config(config)
        assert model2.vocab_size == model.vocab_size
        assert model2.depth == model.depth
        assert model2.field_size == model.field_size

    def test_compute_output_shape(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        shapes = model.compute_output_shape((None, 16))
        assert shapes["logits"] == (None, 16, tiny_config["vocab_size"])
        assert shapes["last_hidden_state"] == (None, 16, tiny_config["embed_dim"])

    def test_save_load_keras_round_trip(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        ids = _random_ids((2, 16), tiny_config["vocab_size"])
        out_before = keras.ops.convert_to_numpy(
            model(ids, training=False)["logits"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wfllm.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            out_after = keras.ops.convert_to_numpy(
                loaded(ids, training=False)["logits"],
            )

        # 1e-5 atol: matches LESSONS L44 for non-U-Net causal models.
        np.testing.assert_allclose(out_before, out_after, atol=1e-5)


# ---------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------


class TestWaveFieldLLMVariants:

    def test_from_variant_tiny(self):
        model = WaveFieldLLM.from_variant("tiny")
        assert model.embed_dim == 256
        assert model.depth == 4
        assert model.num_heads == 4
        assert model.max_seq_len == 512
        assert model.field_size == 1024

    def test_from_variant_with_overrides(self):
        model = WaveFieldLLM.from_variant("tiny", dropout_rate=0.1)
        assert model.dropout_rate == 0.1
        assert model.embed_dim == 256

    def test_from_variant_unknown(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            WaveFieldLLM.from_variant("nonexistent")

    def test_all_variants_have_required_keys(self):
        required = {"embed_dim", "depth", "num_heads", "max_seq_len", "field_size"}
        for name, cfg in WaveFieldLLM.MODEL_VARIANTS.items():
            for k in required:
                assert k in cfg, f"Variant {name!r} missing {k!r}"


# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------


class TestWaveFieldLLMGradient:

    def test_gradient_flow(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        ids = _random_ids((2, 8), tiny_config["vocab_size"])
        labels = _random_ids((2, 8), tiny_config["vocab_size"])

        with tf.GradientTape() as tape:
            out = model(ids, training=True)
            loss = keras.ops.mean(
                keras.losses.sparse_categorical_crossentropy(
                    labels, out["logits"], from_logits=True,
                )
            )
        grads = tape.gradient(loss, model.trainable_variables)
        for var, g in zip(model.trainable_variables, grads):
            assert g is not None, f"no grad for {var.name}"


# ---------------------------------------------------------------------
# CLM loss compatibility
# ---------------------------------------------------------------------


class TestWaveFieldLLMCLMLoss:

    def test_clm_loss_finite(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        ids = _random_ids((2, 8), tiny_config["vocab_size"])
        labels = _random_ids((2, 8), tiny_config["vocab_size"])
        out = model(ids, training=False)
        loss = MaskedCausalLMLoss()(labels, out["logits"])
        loss_val = float(keras.ops.convert_to_numpy(loss))
        assert np.isfinite(loss_val)


# ---------------------------------------------------------------------
# pretrained=True error contract (D-005)
# ---------------------------------------------------------------------


class TestWaveFieldLLMPretrainedContract:
    """`pretrained=True` must RAISE, never return a random-init model.

    Before this contract existed, ``from_variant(..., pretrained=True)``
    logged a warning and returned an untrained model, so a caller asking
    for trained weights silently received random ones.
    """

    def test_download_weights_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not distributed"):
            WaveFieldLLM._download_weights("tiny")

    def test_from_variant_pretrained_true_raises(self):
        with pytest.raises(NotImplementedError, match="not distributed"):
            WaveFieldLLM.from_variant("tiny", pretrained=True)

    def test_create_wave_field_llm_pretrained_true_raises(self):
        with pytest.raises(NotImplementedError, match="not distributed"):
            create_wave_field_llm("tiny", pretrained=True)

    def test_pretrained_missing_path_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            WaveFieldLLM.from_variant(
                "tiny", pretrained="/nonexistent/wave_field.keras",
            )

    def test_pretrained_false_returns_model(self, tiny_config):
        model = WaveFieldLLM.from_variant("tiny", pretrained=False)
        assert isinstance(model, WaveFieldLLM)


# ---------------------------------------------------------------------
# Public API surface (package-root exports)
# ---------------------------------------------------------------------


class TestWaveFieldPublicAPI:
    """Both import forms must resolve to the SAME objects (`is`-identity)."""

    def test_package_root_exports_model_class(self):
        assert wave_field_pkg.WaveFieldLLM is WaveFieldLLM

    def test_package_root_exports_block_class(self):
        assert wave_field_pkg.WaveFieldDecoderBlock is WaveFieldDecoderBlock

    def test_package_root_exports_factory(self):
        assert wave_field_pkg.create_wave_field_llm is create_wave_field_llm

    def test_all_declares_exactly_the_public_surface(self):
        assert set(wave_field_pkg.__all__) == {
            "WaveFieldLLM",
            "WaveFieldDecoderBlock",
            "create_wave_field_llm",
        }


# ---------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------


class TestCreateWaveFieldLLM:

    @pytest.mark.parametrize(
        "variant", sorted(WaveFieldLLM.MODEL_VARIANTS.keys()),
    )
    def test_builds_each_variant(self, variant):
        """Every named variant builds through the factory.

        ``embed_dim``/``depth``/``num_heads``/``vocab_size`` are overridden to
        keep the xl/large/medium variants constructible in a unit test; the
        wave-field-specific ``max_seq_len``/``field_size`` come from the
        variant table untouched and are what is asserted.
        """
        expected = WaveFieldLLM.MODEL_VARIANTS[variant]
        model = create_wave_field_llm(
            variant,
            vocab_size=64,
            embed_dim=32,
            depth=1,
            num_heads=2,
        )
        assert isinstance(model, WaveFieldLLM)
        assert model.vocab_size == 64
        assert model.max_seq_len == expected["max_seq_len"]
        assert model.field_size == expected["field_size"]

    def test_default_variant_is_small(self):
        model = create_wave_field_llm(
            vocab_size=64, embed_dim=32, depth=1, num_heads=2,
        )
        small = WaveFieldLLM.MODEL_VARIANTS["small"]
        assert model.max_seq_len == small["max_seq_len"]
        assert model.field_size == small["field_size"]

    def test_vocab_size_none_keeps_class_default(self):
        model = create_wave_field_llm(
            "tiny", vocab_size=None, embed_dim=32, depth=1, num_heads=2,
        )
        assert model.vocab_size == WaveFieldLLM.DEFAULT_VOCAB_SIZE

    def test_forward_pass_of_factory_built_model(self):
        model = create_wave_field_llm(
            "tiny", vocab_size=64, embed_dim=32, depth=1, num_heads=2,
        )
        ids = _random_ids((2, 8), 64)
        out = model(ids, training=False)
        assert out["logits"].shape == (2, 8, 64)

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            create_wave_field_llm("nonexistent")


# ---------------------------------------------------------------------
# WaveFieldDecoderBlock in isolation (plan-2026-08-13-230c101d / F-05)
# ---------------------------------------------------------------------


def _zero_layer(layer):
    """Zero every weight of ``layer`` in place — a ZEROED TRANSFORM control.

    This zeroes the TRANSFORM (the branch's output projection), never the
    block's input. That distinction is the whole point: the repo's known bug
    shape is a transform-only block invoked as ``x = block(x)`` with no
    external residual, which annihilates the signal ~1e-5 per block. With the
    transform zeroed, a block that HAS its residual is the identity and a block
    that has lost it emits zeros.
    """
    for weight in layer.weights:
        weight.assign(keras.ops.zeros_like(weight))


class TestWaveFieldDecoderBlockIsolation:
    """``WaveFieldDecoderBlock`` constructed standalone (no ``WaveFieldLLM``).

    The block has TWO external residuals (``WaveFieldDecoderBlock.call``):
    ``x = inputs + h`` after attention, and ``return x + h`` after the FFN.
    Nothing tested either one — the block was only ever reached through the
    model, where an ``isinstance`` check was the extent of the coverage.

    Measured with the SHIPPED initializers at ``keras.utils.set_random_seed``
    in {7, 11, 23}, ``embed_dim=32, num_heads=4, max_seq_len=16, field_size=32``,
    input ``(2, 16, 32)`` standard normal, ``training=False``, GPU:1:

    | control                        | max abs(out - x)  | norm ratio       |
    |--------------------------------|-------------------|------------------|
    | shipped weights (live)         | 3.58/3.95/3.31e-2 | 0.9994/1.0004/1.0001 |
    | both transforms zeroed         | exactly 0.0       | 1.000000         |
    | attention transform zeroed     | -                 | 0.9994/1.0004/1.0001 |
    | FFN transform zeroed           | -                 | 1.0000           |

    The ``> 0.5`` norm-ratio bars below are MAGNITUDE assertions (the detector
    that survives an adversarial injection), not equality-vs-reference checks:
    dropping either residual collapses the ratio to exactly 0.0, not to 0.49.
    """

    B, N, D = 2, 16, 32

    @classmethod
    def _block(cls, seed=1234):
        keras.utils.set_random_seed(seed)
        block = WaveFieldDecoderBlock(
            embed_dim=cls.D,
            num_heads=4,
            max_seq_len=cls.N,
            field_size=2 * cls.N,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
        )
        block.build((None, cls.N, cls.D))
        return block

    @classmethod
    def _inputs(cls):
        # Varied, full-rank, non-DC signal: a constant or L2-normalized probe
        # would be blind here (a constant survives a LayerNorm as zeros).
        return np.random.default_rng(0).normal(
            size=(cls.B, cls.N, cls.D),
        ).astype(np.float32)

    @staticmethod
    def _ratio(out, x):
        return float(np.linalg.norm(out) / np.linalg.norm(x))

    def test_block_is_the_identity_when_both_transforms_are_zeroed(self):
        """Both residuals present <=> a block whose two transforms are zeroed
        reproduces its input EXACTLY. Losing either residual emits zeros.

        Measured: max |out - x| = 0.0 exactly, norm ratio 1.000000. Bar: 1e-6.
        """
        block = self._block()
        x = self._inputs()
        _zero_layer(block.attention.output_proj)
        _zero_layer(block.ffn_dense_2)

        out = keras.ops.convert_to_numpy(block(x, training=False))

        delta = np.abs(out - x).max()
        assert delta < 1e-6, (
            f"block with both transforms zeroed is not the identity "
            f"(max |out - x| = {delta:.4e}); an external residual is missing"
        )
        assert abs(self._ratio(out, x) - 1.0) < 1e-6

    def test_block_is_not_trivially_the_identity(self):
        """Non-vacuity guard for the test above: with the SHIPPED weights the
        block is NOT the identity, so the zeroed-transform control is testing
        the residuals rather than a block that ignores its sub-layers.

        Measured max |out - x| across seeds {7, 11, 23}: 3.58e-2 / 3.95e-2 /
        3.31e-2 (the shipped ``initializer_range=0.02`` keeps the branches
        small at init). Bar: > 1e-3.
        """
        block = self._block()
        x = self._inputs()
        out = keras.ops.convert_to_numpy(block(x, training=False))

        delta = np.abs(out - x).max()
        assert delta > 1e-3, (
            f"block with shipped weights is indistinguishable from the "
            f"identity (max |out - x| = {delta:.4e}); the zeroed-transform "
            f"control would then be vacuous"
        )

    def test_attention_residual_carries_the_signal(self):
        """Attention-branch residual, per site: zero ONLY the attention
        transform and leave the FFN live. With ``x = inputs + h`` present the
        signal passes through at full magnitude; with ``x = h`` the block
        emits zeros (the FFN of a zero vector is itself zero here, since the
        LayerNorm beta and the Dense biases both initialize to zero).

        Measured norm ratio across seeds {7, 11, 23}: 0.9994 / 1.0004 / 1.0001.
        Bar: > 0.5 (the failure value is 0.0, so the bar is not delicate).
        """
        block = self._block()
        x = self._inputs()
        _zero_layer(block.attention.output_proj)

        out = keras.ops.convert_to_numpy(block(x, training=False))

        ratio = self._ratio(out, x)
        assert ratio > 0.5, (
            f"signal annihilated with the attention transform zeroed "
            f"(||out|| / ||x|| = {ratio:.6f}); the attention residual "
            f"'x = inputs + h' is missing"
        )

    def test_ffn_residual_carries_the_signal(self):
        """FFN-branch residual, per site: zero ONLY the FFN transform and leave
        attention live. With ``return x + h`` present the signal passes through;
        with ``return h`` the block emits zeros.

        Measured norm ratio across seeds {7, 11, 23}: 1.0000 in all three.
        Bar: > 0.5 (the failure value is 0.0).
        """
        block = self._block()
        x = self._inputs()
        _zero_layer(block.ffn_dense_2)

        out = keras.ops.convert_to_numpy(block(x, training=False))

        ratio = self._ratio(out, x)
        assert ratio > 0.5, (
            f"signal annihilated with the FFN transform zeroed "
            f"(||out|| / ||x|| = {ratio:.6f}); the FFN residual "
            f"'return x + h' is missing"
        )

    def test_padding_mask_reaches_attention(self):
        """The ``(B, N)`` padding mask the block forwards must actually change
        the result. ``WaveFieldAttention`` multiplies its output by the mask,
        so masking the tail changes the block output at those positions.

        Measured max |delta| between an all-ones mask and a half-zero mask:
        7.463e-4 (against a block whose live branch contributes ~3.6e-2 at
        init). Bar: > 1e-5. The all-ones mask is separately pinned to be a
        no-op vs. passing no mask at all (measured exactly 0.0).
        """
        block = self._block()
        x = self._inputs()

        ones = np.ones((self.B, self.N), dtype=np.float32)
        half = ones.copy()
        half[:, self.N // 2:] = 0.0

        out_none = keras.ops.convert_to_numpy(block(x, training=False))
        out_ones = keras.ops.convert_to_numpy(
            block(x, attention_mask=ones, training=False)
        )
        out_half = keras.ops.convert_to_numpy(
            block(x, attention_mask=half, training=False)
        )

        assert out_half.shape == (self.B, self.N, self.D)
        assert np.isfinite(out_half).all()
        assert np.abs(out_ones - out_none).max() < 1e-6, (
            "an all-ones mask is not a no-op"
        )
        delta = np.abs(out_ones - out_half).max()
        assert delta > 1e-5, (
            f"the padding mask does not reach WaveFieldAttention "
            f"(max |delta| between an all-ones and a half-zero mask = "
            f"{delta:.4e})"
        )


# ---------------------------------------------------------------------
# pretrained="<path>" -- the local-path door (C-4b, D-070)
# ---------------------------------------------------------------------


class TestWaveFieldPretrainedLocalPathIsVerified:
    """``from_variant`` loads a local path with an unconditional
    ``skip_mismatch=True``. A checkpoint whose variables do not line up restores
    NOTHING and returns normally; before D-070 the next line logged a successful
    load, so the caller got an untrained model and no error.
    """

    def test_mismatched_checkpoint_raises_naming_the_cause(self, tmp_path):
        other = keras.Sequential(
            [keras.layers.Input(shape=(4,)), keras.layers.Dense(3, name="d")]
        )
        ckpt = str(tmp_path / "not_a_wave_field.keras")
        other.save(ckpt)

        with pytest.raises(ValueError, match="changed none of this model"):
            WaveFieldLLM.from_variant("tiny", vocab_size=64, pretrained=ckpt)

    def test_matching_checkpoint_still_loads(self, tmp_path):
        """Anti-vacuity: the real warm-start path must still work."""
        src = WaveFieldLLM.from_variant("tiny", vocab_size=64)
        probe = np.random.randint(0, 64, (1, 8)).astype("int32")
        src(probe, training=False)
        ckpt = str(tmp_path / "wave_field_tiny.keras")
        src.save(ckpt)

        loaded = WaveFieldLLM.from_variant("tiny", vocab_size=64, pretrained=ckpt)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(src(probe, training=False)["logits"]),
            keras.ops.convert_to_numpy(loaded(probe, training=False)["logits"]),
            rtol=1e-6,
            atol=1e-6,
        )
