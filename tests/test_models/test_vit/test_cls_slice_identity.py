"""F-29 RED proof: pin ViT's CLS slice and its pooling exclusion by IDENTITY.

Every ``cls`` reference in ``test_vit/`` and ``test_vit_hmlp/`` asserted a
stored config value (``model.pooling == "cls"``), the existence of
``model.pool`` / ``model.cls_token``, or an output *shape*. MEASURED against
``116c7db21``: changing ``vit/model.py``'s ``cls_token = x_norm[:, 0, :]`` to
``x_norm[:, -1, :]`` — the classifier then reading the last patch token instead
of the CLS summary — left every test in both suites green, and so did deleting
``exclude_positions=[0]`` from the ``SequencePooling`` construction that the
``plan-2026-07-15T144225-5b25d9f1/D-001`` anchor explicitly forbids deleting.
A shape assertion structurally cannot see either: both mutations preserve every
shape in the model.

The instrument here is the model's OWN forward pass used twice against itself.
``ViT.call`` branches on ``self.include_top`` and ``self.pool`` at call time, so
flipping those attributes on an already-built model yields the full normalized
sequence and the pooled/classified output from the SAME weights, and the two can
be compared bit-exactly. No second model, no monkeypatching, no reimplementation
of the model's arithmetic as an oracle.

Every assertion below carries its own anti-vacuity control: the rejected
alternative (the last token, or the CLS-included mean) is asserted to be
measurably DIFFERENT, so none of these can pass on a degenerate input where all
positions happen to agree.

RED, measured at ``38c7493c6`` (4 tests in this file, 119 pre-existing tests in
``test_vit/`` + ``test_vit_hmlp/``):

| injection | this file | pre-existing |
|---|---|---|
| ``x_norm[:, 0, :]`` -> ``x_norm[:, -1, :]`` | 1 failed | **119 passed** |
| CLS slice DELETED (``ops.mean(x_norm, axis=1)``, dead component) | 2 failed | - |
| ``exclude_positions=[0]`` deleted | 2 failed | **119 passed** |
| ``SequencePooling`` DELETED (dead component) | 2 failed | - |

The two "119 passed" rows are F-29 itself: the finding's own mutations, green
against the whole pre-existing surface.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.vit.model import ViT

INPUT_SHAPE = (32, 32, 3)
PATCH = 8
BATCH = 2


def _x(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(
        size=(BATCH,) + INPUT_SHAPE
    ).astype("float32")


def _np(t) -> np.ndarray:
    return keras.ops.convert_to_numpy(t)


class TestViTClsSliceIdentity:
    """``include_top=True`` must classify position 0, and only position 0."""

    @staticmethod
    def _built_model():
        model = ViT(
            input_shape=INPUT_SHAPE,
            patch_size=PATCH,
            scale="tiny",
            num_classes=5,
            include_top=True,
            pooling=None,
        )
        model.build((None,) + INPUT_SHAPE)
        return model

    def test_the_head_is_fed_position_zero_of_the_normalized_sequence(self):
        model = self._built_model()
        x = _x(1)

        logits = _np(model(x, training=False))

        # Same weights, sequence output: ``call`` branches on include_top.
        model.include_top = False
        sequence = _np(model(x, training=False))
        model.include_top = True

        assert sequence.shape[1] > 1, "need >1 position for this to discriminate"

        from_cls = _np(model.head(ops.convert_to_tensor(sequence[:, 0, :])))
        np.testing.assert_allclose(
            logits, from_cls, rtol=0, atol=0,
            err_msg="the classification head is NOT fed x_norm[:, 0, :]",
        )

        # Anti-vacuity: the rejected slice must give a different answer, else
        # the assertion above would hold for any position.
        from_last = _np(model.head(ops.convert_to_tensor(sequence[:, -1, :])))
        assert not np.allclose(logits, from_last, atol=1e-6), (
            "control is vacuous: the last token feeds the head identically to "
            "the CLS token, so this test could not detect the slice moving"
        )

    def test_the_cls_position_is_not_a_pooled_summary(self):
        """A mean/max over the sequence must NOT reproduce the head's input."""
        model = self._built_model()
        x = _x(2)

        logits = _np(model(x, training=False))
        model.include_top = False
        sequence = _np(model(x, training=False))
        model.include_top = True

        for name, pooled in (
            ("mean", sequence.mean(axis=1)),
            ("max", sequence.max(axis=1)),
        ):
            alt = _np(model.head(ops.convert_to_tensor(pooled)))
            assert not np.allclose(logits, alt, atol=1e-6), (
                f"the head's input is indistinguishable from a {name}-pool of "
                "the sequence; the CLS slice is not pinned"
            )


class TestViTPoolingExcludesCls:
    """``vit`` pools mean/max over positions 1: — CLS is deliberately dropped.

    See the ``plan-2026-07-15T144225-5b25d9f1/D-001`` anchor in
    ``vit/model.py``: "Do NOT drop exclude_positions (would start averaging the
    CLS token)."
    """

    @pytest.mark.parametrize("strategy", ["mean", "max"])
    def test_pooling_skips_position_zero(self, strategy):
        model = ViT(
            input_shape=INPUT_SHAPE,
            patch_size=PATCH,
            scale="tiny",
            include_top=False,
            pooling=strategy,
        )
        model.build((None,) + INPUT_SHAPE)
        x = _x(3)

        pooled = _np(model(x, training=False))

        # Same weights, raw sequence: ``call`` returns x_norm when pool is None.
        pool, model.pool = model.pool, None
        sequence = _np(model(x, training=False))
        model.pool = pool

        reduce = np.mean if strategy == "mean" else np.max
        excluding_cls = reduce(sequence[:, 1:, :], axis=1)
        including_cls = reduce(sequence, axis=1)

        # Anti-vacuity first: the two candidate answers must differ.
        assert not np.allclose(excluding_cls, including_cls, atol=1e-6), (
            "control is vacuous: including and excluding the CLS token give "
            "the same pooled vector on this input"
        )
        np.testing.assert_allclose(
            pooled, excluding_cls, rtol=0, atol=1e-6,
            err_msg=f"vit {strategy}-pooling no longer excludes the CLS token",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
