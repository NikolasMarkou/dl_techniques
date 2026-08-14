"""
Test suite for the Byte Latent Transformer (BLT).

Covers construction (including ValueError validation paths), the from_variant /
create_blt_model factory, a forward pass, and the M2 full .keras save -> load ->
identical-output round-trip.

BLT `call()` accepts an int32 (B, T) byte-token tensor and returns logits
(B, T, vocab_size).
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.byte_latent_transformer.model import (
    ByteLatentTransformer,
    create_blt_model,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def small_model() -> ByteLatentTransformer:
    """A small but real BLT (micro variant, short context) for fast tests."""
    return create_blt_model(
        variant="micro",
        vocab_size=256,
        max_sequence_length=64,
    )


@pytest.fixture
def sample_tokens() -> np.ndarray:
    return np.random.randint(0, 256, (2, 16)).astype("int32")


# ---------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------

class TestConstruction:

    def test_from_variant_micro(self) -> None:
        model = ByteLatentTransformer.from_variant("micro", vocab_size=256)
        assert model.local_dim == 256
        assert model.global_dim == 384
        assert model.num_heads_local == 4

    def test_create_blt_model_factory(self, small_model) -> None:
        assert isinstance(small_model, ByteLatentTransformer)
        assert small_model.vocab_size == 256

    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(ValueError):
            ByteLatentTransformer.from_variant("nonexistent")

    @pytest.mark.parametrize("bad", [
        {"vocab_size": 0},
        {"local_dim": -1},
        {"global_dim": 0},
        {"num_local_layers": 0},
        {"num_heads_local": 0},
        {"max_patches": 0},
    ])
    def test_nonpositive_args_raise(self, bad) -> None:
        with pytest.raises(ValueError):
            ByteLatentTransformer(**bad)

    def test_indivisible_heads_raise(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            ByteLatentTransformer(local_dim=256, num_heads_local=7)

    def test_invalid_pooling_raises(self) -> None:
        with pytest.raises(ValueError, match="patch_pooling_method"):
            ByteLatentTransformer(patch_pooling_method="bogus")

    def test_invalid_dropout_raises(self) -> None:
        with pytest.raises(ValueError, match="dropout_rate"):
            ByteLatentTransformer(dropout_rate=1.5)


# ---------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------

class TestForward:

    def test_forward_shape(self, small_model, sample_tokens) -> None:
        out = small_model(sample_tokens, training=False)
        b, t = sample_tokens.shape
        assert out.shape == (b, t, small_model.vocab_size)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(out)))

    def test_config_round_trip(self, small_model) -> None:
        config = small_model.get_config()
        rebuilt = ByteLatentTransformer.from_config(config)
        assert rebuilt.vocab_size == small_model.vocab_size
        assert rebuilt.local_dim == small_model.local_dim
        assert rebuilt.patch_pooling_method == small_model.patch_pooling_method


# ---------------------------------------------------------------------
# M2: full .keras round-trip
# ---------------------------------------------------------------------

class TestKerasRoundTrip:

    def test_save_load_identical(self, tmp_path, small_model, sample_tokens) -> None:
        y_before = small_model(sample_tokens, training=False)

        save_path = os.path.join(str(tmp_path), "blt.keras")
        small_model.save(save_path)
        loaded = keras.models.load_model(save_path)

        y_after = loaded(sample_tokens, training=False)

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y_before),
            keras.ops.convert_to_numpy(y_after),
            atol=1e-4,
            err_msg="Outputs differ after .keras round-trip",
        )


# ---------------------------------------------------------------------
# Causality: the future-leak probe
# ---------------------------------------------------------------------

class TestCausality:
    """BLT is trained under a next-byte objective, so no output at position
    ``i`` may depend on any byte at a position ``>= i + 1``.

    The instrument is a future-leak probe: change ONE byte at position ``t``
    and require the logits at every position ``< t`` to be bit-identical. On
    the unmasked implementation (no causal mask in any of the four stacks and
    a cross-attention gathering the byte's OWN patch, whose pooled
    representation contains the target byte) this measured
    ``max|delta| = 5.636312e-01`` over positions ``< t``.
    """

    SEQ_LEN = 12
    PERTURB_AT = 6

    @staticmethod
    def _tiny_model() -> ByteLatentTransformer:
        return create_blt_model(
            variant="micro",
            vocab_size=32,
            max_sequence_length=TestCausality.SEQ_LEN,
            max_patches=4,
            local_dim=32,
            global_dim=48,
            num_local_layers=2,
            num_global_layers=2,
            num_heads_local=4,
            num_heads_global=4,
            dropout_rate=0.0,
        )

    def _perturbed_pair(self):
        rng = np.random.default_rng(0)
        x = rng.integers(1, 32, size=(2, self.SEQ_LEN)).astype("int32")
        x2 = x.copy()
        x2[:, self.PERTURB_AT] = (x2[:, self.PERTURB_AT] + 7) % 31 + 1
        return x, x2

    def test_future_byte_does_not_change_the_past(self) -> None:
        model = self._tiny_model()
        x, x2 = self._perturbed_pair()

        a = keras.ops.convert_to_numpy(model(x, training=False))
        b = keras.ops.convert_to_numpy(model(x2, training=False))

        past = np.abs(a[:, :self.PERTURB_AT] - b[:, :self.PERTURB_AT]).max()
        assert past == 0.0, (
            f"future leak: perturbing byte {self.PERTURB_AT} moved the logits "
            f"at earlier positions by {past:.6e} (must be exactly 0.0)"
        )

    def test_the_model_still_responds_at_and_after_the_perturbation(self) -> None:
        """A mask that froze the whole model would also pass the test above."""
        model = self._tiny_model()
        x, x2 = self._perturbed_pair()

        a = keras.ops.convert_to_numpy(model(x, training=False))
        b = keras.ops.convert_to_numpy(model(x2, training=False))

        future = np.abs(a[:, self.PERTURB_AT:] - b[:, self.PERTURB_AT:]).max()
        assert future > 1e-3, (
            f"the model is inert: positions >= {self.PERTURB_AT} moved by only "
            f"{future:.6e} when the byte there changed"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
