"""F-29 RED proof: pin ViTHMLP's CLS slice and its pooling inclusion by IDENTITY.

The sibling file is ``tests/test_models/test_vit/test_cls_slice_identity.py``;
its module docstring carries the measurement and the method. The reason this
package needs its own copy is that ``vit_hmlp`` INTENTIONALLY differs from
``vit`` on the pooling half — it pools the CLS token IN, where ``vit`` excludes
it — and that divergence is load-bearing, deliberate and documented at both
sites. Pinning it in both packages is what makes "harmonising" the two
impossible to do silently.

RED, measured at ``38c7493c6`` (3 tests in this file):

| injection | result |
|---|---|
| ``x[:, 0, :]`` -> ``x[:, -1, :]`` | 1 failed |
| ``exclude_positions=[0]`` ADDED (the forbidden harmonisation) | 2 failed |
| ``SequencePooling`` DELETED (dead component) | 2 failed |
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.vit_hmlp.model import ViTHMLP

INPUT_SHAPE = (32, 32, 3)
PATCH = 8
BATCH = 2


def _x(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(
        size=(BATCH,) + INPUT_SHAPE
    ).astype("float32")


def _np(t) -> np.ndarray:
    return keras.ops.convert_to_numpy(t)


class TestViTHmlpPoolingIncludesCls:
    """``vit_hmlp`` INTENTIONALLY differs from ``vit``: CLS is pooled in.

    This is not an oversight — the D-001 anchor in ``vit_hmlp/model.py`` says
    "Do NOT add exclude_positions=[0] here (would drop the CLS token and change
    outputs)". The divergence is pinned in both packages so that "harmonising"
    them silently is impossible.
    """

    @pytest.mark.parametrize("strategy", ["mean", "max"])
    def test_pooling_includes_position_zero(self, strategy):
        model = ViTHMLP(
            input_shape=INPUT_SHAPE,
            patch_size=PATCH,
            scale="tiny",
            include_top=False,
            pooling=strategy,
        )
        model.build((None,) + INPUT_SHAPE)
        x = _x(4)

        pooled = _np(model(x, training=False))

        pool, model.pool = model.pool, None
        sequence = _np(model(x, training=False))
        model.pool = pool

        reduce = np.mean if strategy == "mean" else np.max
        including_cls = reduce(sequence, axis=1)
        excluding_cls = reduce(sequence[:, 1:, :], axis=1)

        assert not np.allclose(including_cls, excluding_cls, atol=1e-6), (
            "control is vacuous on this input"
        )
        np.testing.assert_allclose(
            pooled, including_cls, rtol=0, atol=1e-6,
            err_msg="vit_hmlp pooling started excluding the CLS token; the "
                    "deliberate divergence from vit was erased",
        )


class TestViTHmlpClsSliceIdentity:
    """``vit_hmlp``'s classification head reads position 0 as well."""

    def test_the_head_is_fed_position_zero(self):
        model = ViTHMLP(
            input_shape=INPUT_SHAPE,
            patch_size=PATCH,
            scale="tiny",
            num_classes=5,
            include_top=True,
            pooling=None,
        )
        model.build((None,) + INPUT_SHAPE)
        x = _x(5)

        logits = _np(model(x, training=False))
        model.include_top = False
        sequence = _np(model(x, training=False))
        model.include_top = True

        from_cls = _np(model.head(ops.convert_to_tensor(sequence[:, 0, :])))
        np.testing.assert_allclose(
            logits, from_cls, rtol=0, atol=0,
            err_msg="vit_hmlp's head is NOT fed x[:, 0, :]",
        )
        from_last = _np(model.head(ops.convert_to_tensor(sequence[:, -1, :])))
        assert not np.allclose(logits, from_last, atol=1e-6), (
            "control is vacuous on this input"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
