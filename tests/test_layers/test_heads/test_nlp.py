"""Tests for the merged ``heads.nlp`` sub-package.

Covers:

* **SC4 — pooling equivalence (the critical test).** The old inline
  ``BaseNLPHead._pool_sequence`` body for cls/mean/max was replaced by a
  delegation to the shared ``SequencePooling`` facade (D-002). The old code is
  gone, so this test asserts the *analytical* references for cls/mean/max
  (mask-weighted) plus shape/finiteness for the kept-inline ``attention`` path,
  on a fixed ``(B=2, S=4, D=8)`` input with the last token of row 0 masked.
  This locks that the SequencePooling delegation preserved semantics.
* factory smoke + a ``keras.Model`` save/load round-trip for one NLP head.
"""

import os
import tempfile

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.layers.heads.nlp import (
    NLPTaskType,
    NLPTaskConfig,
    create_nlp_head,
    TextClassificationHead,
    QuestionAnsweringHead,
)

# ---------------------------------------------------------------------
# Fixtures: a fixed input + a known attention mask (last token of row 0 masked)
# ---------------------------------------------------------------------

B, S, D = 2, 4, 8


@pytest.fixture
def fixed_input() -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.standard_normal((B, S, D)).astype("float32")


@pytest.fixture
def attention_mask() -> np.ndarray:
    # Row 0: last token masked out (valid = positions 0..2).
    # Row 1: all four tokens valid.
    mask = np.ones((B, S), dtype="float32")
    mask[0, S - 1] = 0.0
    return mask


def _make_head(pooling_type: str) -> TextClassificationHead:
    """Build the smallest head exposing ``pooling_type`` with pooling-only path.

    ``use_intermediate=False`` / ``use_ffn=False`` keeps the head a thin wrapper
    around the pooling stage so we can probe ``_pool_sequence`` directly.
    """
    cfg = NLPTaskConfig(
        name="cls",
        task_type=NLPTaskType.TEXT_CLASSIFICATION,
        num_classes=3,
    )
    head = TextClassificationHead(
        task_config=cfg,
        input_dim=D,
        pooling_type=pooling_type,
        use_pooling=True,
        use_intermediate=False,
        use_ffn=False,
        use_task_attention=False,
    )
    # Build sub-layers (pooler is created in __init__, built in build()).
    head.build((B, S, D))
    return head


# ---------------------------------------------------------------------
# SC4 — pooling-equivalence (analytical references)
# ---------------------------------------------------------------------

class TestPoolingEquivalence:
    """SC4: cls/mean/max are analytically correct; all 4 give (B, D)."""

    @pytest.mark.parametrize("pooling_type", ["cls", "mean", "max", "attention"])
    def test_pool_shape_is_B_D(self, pooling_type, fixed_input, attention_mask) -> None:
        head = _make_head(pooling_type)
        x = ops.convert_to_tensor(fixed_input)
        m = ops.convert_to_tensor(attention_mask)
        pooled = head._pool_sequence(x, m)
        assert tuple(pooled.shape) == (B, D)

    def test_cls_pool_matches_first_token(self, fixed_input, attention_mask) -> None:
        head = _make_head("cls")
        pooled = head._pool_sequence(
            ops.convert_to_tensor(fixed_input),
            ops.convert_to_tensor(attention_mask),
        )
        np.testing.assert_allclose(
            ops.convert_to_numpy(pooled), fixed_input[:, 0, :], atol=1e-6
        )

    def test_mean_pool_matches_masked_mean(self, fixed_input, attention_mask) -> None:
        head = _make_head("mean")
        pooled = head._pool_sequence(
            ops.convert_to_tensor(fixed_input),
            ops.convert_to_tensor(attention_mask),
        )
        # Analytical mask-weighted mean over valid positions.
        m = attention_mask[:, :, None]  # (B, S, 1)
        lengths = np.maximum(m.sum(axis=1), 1.0)  # (B, 1)
        ref = (fixed_input * m).sum(axis=1) / lengths
        np.testing.assert_allclose(ops.convert_to_numpy(pooled), ref, atol=1e-6)

    def test_max_pool_matches_masked_max(self, fixed_input, attention_mask) -> None:
        head = _make_head("max")
        pooled = head._pool_sequence(
            ops.convert_to_tensor(fixed_input),
            ops.convert_to_tensor(attention_mask),
        )
        # Analytical masked max: masked positions pushed to -inf-equivalent.
        m = attention_mask[:, :, None]
        masked = fixed_input + (1.0 - m) * (-1e9)
        ref = masked.max(axis=1)
        np.testing.assert_allclose(ops.convert_to_numpy(pooled), ref, atol=1e-6)

    def test_attention_pool_finite_and_BD(self, fixed_input, attention_mask) -> None:
        """attention path is the kept inline Dense(1, tanh) direct-score pooling
        (D-002). Assert (B, D) + finiteness, then that the masked position is
        ISOLATED.

        The isolation arm replaces an oracle that re-implemented the subject:
        it recomputed ``scores * m + (1 - m) * -1e9`` in the test body and
        asserted the resulting softmax weight was small. That expected value
        was DERIVED FROM THE IMPLEMENTATION, so it was green both before and
        after the additive form was replaced -- and it would have stayed green
        had the sentinel been dropped from the test's copy and the subject's
        copy together. This repo has shipped that defect class five times in
        one plan.

        What replaces it needs no knowledge of the sentinel, of its magnitude,
        or of whether the masking is additive or a selection: PERTURBING A
        MASKED POSITION MUST NOT MOVE THE POOLED OUTPUT AT ALL. A live control
        perturbing a KEPT position must move it by a wide margin, so a probe
        that passes because nothing moves is impossible.
        """
        head = _make_head("attention")
        x = ops.convert_to_tensor(fixed_input)
        m = ops.convert_to_tensor(attention_mask)
        pooled_np = ops.convert_to_numpy(head._pool_sequence(x, m))
        assert pooled_np.shape == (B, D)
        assert np.all(np.isfinite(pooled_np))

        rng = np.random.default_rng(99)

        def _perturbed(position: int) -> np.ndarray:
            out = np.array(fixed_input, copy=True)
            out[0, position, :] += (5.0 * rng.normal(size=(D,))).astype(out.dtype)
            return out

        # LIVE CONTROL FIRST. Position 0 of row 0 is kept, so it must move.
        live = ops.convert_to_numpy(
            head._pool_sequence(ops.convert_to_tensor(_perturbed(0)), m)
        )
        live_delta = float(np.max(np.abs(live - pooled_np)))
        assert live_delta > 1e-2, (
            f"Vacuous probe: perturbing the KEPT position 0 moved the pooled "
            f"output by only {live_delta:.6e}, so the isolation assertion "
            f"below proves nothing."
        )

        # Row 0's last token is masked. Its content must not reach the output.
        leaked = ops.convert_to_numpy(
            head._pool_sequence(ops.convert_to_tensor(_perturbed(S - 1)), m)
        )
        np.testing.assert_allclose(
            leaked, pooled_np, rtol=0, atol=0,
            err_msg=(
                f"a MASKED position leaked into the attention-pooled output by "
                f"{float(np.max(np.abs(leaked - pooled_np))):.6e}; required 0.0."
            ),
        )

    def test_attention_pool_survives_an_all_masked_row(
        self, fixed_input, attention_mask
    ) -> None:
        """A row that keeps nothing must not produce NaN.

        Independent of any sentinel value: softmax over a vector of one
        repeated finite value is uniform and finite, whatever that value is.
        Only a non-finite sentinel (or a ``0 * -inf``) can break this.
        """
        head = _make_head("attention")
        mask = np.array(attention_mask, copy=True)
        mask[0, :] = 0.0
        pooled = ops.convert_to_numpy(
            head._pool_sequence(
                ops.convert_to_tensor(fixed_input),
                ops.convert_to_tensor(mask),
            )
        )
        assert np.all(np.isfinite(pooled)), (
            f"an all-masked row produced non-finite attention pooling: "
            f"{pooled[0]}"
        )

    def test_mean_unmasked_is_plain_mean(self, fixed_input) -> None:
        """Without a mask, mean pooling == plain mean over the sequence axis."""
        head = _make_head("mean")
        pooled = head._pool_sequence(ops.convert_to_tensor(fixed_input), None)
        np.testing.assert_allclose(
            ops.convert_to_numpy(pooled), fixed_input.mean(axis=1), atol=1e-6
        )


# ---------------------------------------------------------------------
# Factory smoke + save/load round-trip
# ---------------------------------------------------------------------

class TestNLPFactoryAndRoundtrip:

    def test_factory_returns_classification_head(self) -> None:
        head = create_nlp_head(
            task_config=NLPTaskConfig(
                name="cls",
                task_type=NLPTaskType.TEXT_CLASSIFICATION,
                num_classes=5,
            ),
            input_dim=D,
        )
        assert isinstance(head, TextClassificationHead)

    def test_factory_from_dict_config(self) -> None:
        head = create_nlp_head(
            task_config={
                "name": "sent",
                "task_type": NLPTaskType.SENTIMENT_ANALYSIS,
                "num_classes": 2,
            },
            input_dim=D,
        )
        assert isinstance(head, TextClassificationHead)

    def test_functional_api_list_input_shape_build(self) -> None:
        """SC2 / Bug-3 regression lock. The Keras functional API hands ``build``
        a LIST/TensorShape ``input_shape`` (not a tuple/dict). The inverted
        ternary in ``TextClassificationHead.build`` used to index it wrong and
        raise; this asserts the fixed path builds and emits ``(None, num_classes)``
        logits."""
        num_classes = 5
        head = create_nlp_head(
            task_config=NLPTaskConfig(
                name="cls",
                task_type=NLPTaskType.TEXT_CLASSIFICATION,
                num_classes=num_classes,
            ),
            input_dim=D,
            pooling_type="mean",
        )
        inp = keras.Input(shape=(S, D))  # functional API -> list input_shape
        out = head(inp)  # must not raise AttributeError / index error
        assert tuple(out["logits"].shape) == (None, num_classes)

    def test_model_save_load_roundtrip(self, fixed_input) -> None:
        # NOTE: the head's own ``build`` only handles tuple/dict input_shape
        # (a pre-existing fragility in the merged code: the Keras functional API
        # hands ``build`` a *list*, which TextClassificationHead.build does not
        # accept). We therefore wrap the head in a tiny subclassed Model that
        # invokes the head on real tensors (so ``build`` receives a tuple) — this
        # still exercises full ``.keras`` save/load of the head, which is the SC5
        # contract.
        cfg = NLPTaskConfig(
            name="cls",
            task_type=NLPTaskType.TEXT_CLASSIFICATION,
            num_classes=3,
        )

        @keras.saving.register_keras_serializable()
        class _NLPWrapper(keras.Model):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.head = create_nlp_head(
                    task_config=cfg, input_dim=D, pooling_type="mean"
                )

            def call(self, inputs, training=None):
                return self.head(inputs, training=training)

        model = _NLPWrapper()
        y0 = model(fixed_input)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nlp_head.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(fixed_input)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0["logits"]),
            ops.convert_to_numpy(y1["logits"]),
            atol=1e-4,
        )


# ---------------------------------------------------------------------
# The fp16 regression guard: the additive mask bias NaN'd the KEPT positions
# ---------------------------------------------------------------------

class TestTheMaskedHeadsSurviveMixedFloat16:
    """``mixed_float16`` is the regime the additive mask bias destroyed.

    Both subjects here used to compute the mask as arithmetic::

        x = x * m + (1 - m) * -1e9

    ``float16`` tops out at ``65504``, so the sentinel materializes as
    ``-inf`` in the compute dtype. The multiplication then evaluates
    ``0.0 * -inf`` at every position the mask KEEPS, which is ``NaN`` -- so the
    corruption lands on the positions the mask exists to preserve, on an
    ordinary right-padding mask, on a path a green float32 suite never runs.

    The replacement is ``keras.ops.where(keep, x, sentinel)`` with the sentinel
    from ``dl_techniques.utils.dtype_policy.mask_sentinel``, which is ``-1e4``
    in float16: nothing is multiplied, and nothing overflows.

    RED PROOF (recorded in the plan's verification notes): re-injecting the
    additive expression into ``layers/heads/nlp/factory.py`` itself -- not into
    a scratch copy, because ``pyproject.toml``'s ``pythonpath = ["src"]``
    overrides ``PYTHONPATH`` and a scratch tree yields a FALSE GREEN -- turns
    every test in this class RED with all-``NaN`` output, and restoring the
    file turns them green again.

    The policy fixture is ``tests/test_layers/conftest.py``'s
    ``mixed_float16_policy``, which restores the previous global policy in a
    ``finally``.
    """

    def test_attention_pooling_is_finite_under_mixed_float16(
        self, mixed_float16_policy, fixed_input, attention_mask
    ) -> None:
        head = _make_head("attention")
        assert head.compute_dtype == "float16", (
            "the head was not built under the half-precision policy, so this "
            "guard would run in float32 and could not fail"
        )
        # Called through the PUBLIC surface, not `_pool_sequence` directly:
        # Keras autocasts the caller's float32 input to the layer's compute
        # dtype only on the public path, and it is the compute dtype that
        # materializes the sentinel.
        out = head({
            "hidden_states": ops.convert_to_tensor(fixed_input),
            "attention_mask": ops.convert_to_tensor(attention_mask),
        })
        logits = ops.convert_to_numpy(out["logits"])
        assert np.all(np.isfinite(logits)), (
            f"attention pooling produced non-finite output under "
            f"mixed_float16: {logits}. `0.0 * float16(-1e9)` is `0.0 * -inf` "
            f"= NaN, and it lands on the KEPT positions."
        )

    def test_question_answering_span_logits_are_finite_under_mixed_float16(
        self, mixed_float16_policy, fixed_input, attention_mask
    ) -> None:
        head = QuestionAnsweringHead(
            task_config=NLPTaskConfig(
                name="qa",
                task_type=NLPTaskType.QUESTION_ANSWERING,
            ),
            input_dim=D,
            use_intermediate=False,
            use_ffn=False,
            use_task_attention=False,
        )
        assert head.compute_dtype == "float16"

        out = head({
            "hidden_states": ops.convert_to_tensor(fixed_input),
            "attention_mask": ops.convert_to_tensor(attention_mask),
        })
        for key in ("start_logits", "end_logits"):
            values = ops.convert_to_numpy(out[key])
            assert np.all(np.isfinite(values)), (
                f"{key} is non-finite under mixed_float16: {values}"
            )

    def test_a_masked_span_position_cannot_win_the_argmax(
        self, fixed_input, attention_mask
    ) -> None:
        """The QA sentinel's ONLY job: lose to every real logit.

        The oracle knows nothing about the sentinel -- not its magnitude, not
        whether it is added or selected. It is self-calibrating and cannot be
        vacuous: the position it masks is CHOSEN as the position that wins the
        unmasked argmax, so "the argmax moved" is exactly the claim, and a
        no-op mask would fail it by construction.
        """
        head = QuestionAnsweringHead(
            task_config=NLPTaskConfig(
                name="qa",
                task_type=NLPTaskType.QUESTION_ANSWERING,
            ),
            input_dim=D,
            use_intermediate=False,
            use_ffn=False,
            use_task_attention=False,
        )
        head.build((B, S, D))

        x = ops.convert_to_tensor(fixed_input)
        unmasked = head(x)

        for key in ("start_logits", "end_logits"):
            winner = int(np.argmax(ops.convert_to_numpy(unmasked[key])[0]))

            # Mask exactly the unmasked winner, in row 0 only.
            mask = np.ones((B, S), dtype="float32")
            mask[0, winner] = 0.0

            logits = ops.convert_to_numpy(
                head({
                    "hidden_states": x,
                    "attention_mask": ops.convert_to_tensor(mask),
                })[key]
            )[0]

            assert int(np.argmax(logits)) != winner, (
                f"{key}: position {winner} won the argmax BEFORE masking and "
                f"still wins it after being masked; the sentinel failed to "
                f"lose. Logits: {logits}"
            )
            kept = np.delete(logits, winner)
            assert logits[winner] < kept.min(), (
                f"{key}: the masked logit {logits[winner]} is not strictly "
                f"below every kept logit (min {kept.min()})."
            )
            assert np.all(np.isfinite(logits)), f"{key} is non-finite: {logits}"
