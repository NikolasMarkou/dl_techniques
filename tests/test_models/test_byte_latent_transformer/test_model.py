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

    def test_construction_is_silent_about_the_entropy_threshold(self, caplog) -> None:
        """Construction must NOT warn about ``entropy_threshold``.

        The predecessor check compared the threshold to ``0.5 * ln(vocab_size)``
        -- vocabulary arithmetic that never looked at the entropy -- so at the
        shipped ``vocab_size=260`` (floor 2.78 nats) it fired on the default
        1.5, on ``train_blt.py``'s 1.3, and on every construction in this test
        suite: 100% of shipped configurations, including the one its own
        reasoning called probably right. A diagnostic that always fires carries
        no information. The degeneracy is now measured where it is observable,
        by ``DynamicPatcher.warn_if_segmentation_is_degenerate(entropy)``.
        """
        import logging

        with caplog.at_level(logging.WARNING):
            ByteLatentTransformer(vocab_size=260, max_sequence_length=32,
                                  num_local_layers=1, num_global_layers=1)

        offenders = [
            r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING and "entropy_threshold" in r.getMessage()
        ]
        assert not offenders, (
            "construction warns about entropy_threshold again; the alarm fires "
            f"on every shipped configuration and is noise: {offenders}"
        )

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


# ---------------------------------------------------------------------
# Gradient flow (plan-2026-08-19-a616f581 step 10)
# ---------------------------------------------------------------------

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    gradient_report,
)

#: Every weight of the entropy sub-model. 54 of BLT's 254 trainable tensors.
ENTROPY_SUBTREE = "entropy_model/"


class TestBLTGradientFlow:
    """BLT's entropy model is trained by nothing -- pinned, not waived.

    GF-02 in the plan's ``findings/gradient-flow-adoption-findings.md``.
    ``BLT.call`` DOES execute the entropy model, but its only consumer is the
    patcher, which discretizes to integer patch lengths/ids -- so the backward
    graph is severed there and 54 of 254 trainable weights (21% of the model)
    receive no gradient at all. There is no ``stop_gradient`` anywhere in the
    package; the cut is structural.

    The design intent is defensible on its own terms -- the class docstring calls
    the constructor argument "Optional **pre-trained** entropy model", and in the
    BLT paper it is trained separately. But NOTHING in the code expresses that:
    the 54 variables stay ``trainable=True``, so BLT's own ``fit()`` hands the
    optimizer 54 ``None`` gradients and emits ``UserWarning: Gradients do not
    exist for variables [...]`` (measured) while training the other 200. A user
    therefore ships a model whose patch boundaries are chosen by a permanently
    RANDOM sub-network, and the only signal is a warning that scrolls past.

    That is why this is an ``xfail(strict=True)`` and not an ``expect_zero``
    entry. ``expect_zero`` is for a zero the code ENFORCES; here nothing does.
    Either fix -- marking the default entropy model non-trainable, or adding its
    own LM loss -- turns this test RED, which is exactly the notification wanted.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "GF-02: 54 of 254 trainable weights (the whole entropy_model "
            "subtree) are off the backward graph -- the patcher discretizes the "
            "entropy to integer patch ids, and nothing marks those variables "
            "non-trainable, so BLT's own fit() warns and trains 200/254. "
            "strict=True: goes RED when the model is repaired."
        ),
    )
    def test_gradients_reach_every_trainable_weight(
        self, small_model, sample_tokens
    ):
        small_model(sample_tokens, training=False)
        assert_gradients_reach_every_trainable_weight(small_model, sample_tokens)

    def test_the_rest_of_the_model_does_receive_gradients(
        self, small_model, sample_tokens
    ):
        """Anti-vacuity for the xfail above.

        An xfail is satisfied by ANY failure, including a totally dead model --
        so on its own it would also "pass" if BLT trained nothing at all. This
        pins the complement: every weight OUTSIDE the entropy subtree is live,
        and the dead set is exactly the entropy subtree, by name.
        """
        small_model(sample_tokens, training=False)
        report = gradient_report(small_model, sample_tokens)

        dead = {
            path for path, value in report.items()
            if value is None or value == 0.0
        }
        live = set(report) - dead

        assert dead, "expected the entropy subtree to be dead (GF-02)"
        assert live, "the whole model is dead -- this is a different defect"
        assert all(ENTROPY_SUBTREE in path for path in dead), (
            "a weight OUTSIDE the entropy model is also dead, which GF-02 does "
            f"not cover: {sorted(p for p in dead if ENTROPY_SUBTREE not in p)}"
        )
        assert all(ENTROPY_SUBTREE not in path for path in live), (
            "part of the entropy model now DOES train -- GF-02 may be fixed; "
            "update the finding and remove the xfail above"
        )
        # The measured split, pinned so a silent change in either direction shows.
        assert len(dead) == 54, len(dead)
        assert len(live) == 200, len(live)
