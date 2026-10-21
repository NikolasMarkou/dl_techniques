"""F-29 RED proof: pin ViTHMLP's CLS slice and its pooling exclusion by IDENTITY.

The sibling file is ``tests/test_models/test_vit/test_cls_slice_identity.py``;
its module docstring carries the measurement and the method. Until
plan-2026-10-14T062616-81938e81 D-205 this file pinned the opposite: that
``vit_hmlp`` pooled the CLS token IN, unlike ``vit``. That divergence had no
recorded reason beyond keeping the numbers unchanged (its anchor's plan
directory is gone), and the references exclude it: deit's hMLP reference reads
``x[:, 0]`` only and timm's ``global_pool_nlc`` reduces
``x[:, num_prefix_tokens:]``. The pooling class below is re-derived from that
reading (mean/max over ``sequence[:, 1:]``), not from the current code.

RED, measured at ``38c7493c6`` (3 tests in this file):

| injection | result |
|---|---|
| ``x[:, 0, :]`` -> ``x[:, -1, :]`` | 1 failed |
| ``exclude_positions=[0]`` ADDED (then forbidden; now the rule) | 2 failed |
| ``SequencePooling`` DELETED (dead component) | 2 failed |
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.vit_hmlp.model import ViTHMLP

INPUT_SHAPE = (32, 32, 3)
PATCH = 8
BATCH = 2


def _x(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(
        size=(BATCH,) + INPUT_SHAPE
    ).astype("float32")


def _np(t) -> np.ndarray:
    """`convert_to_numpy` of one tensor: the model returns its logits bare."""
    assert not isinstance(t, dict), "the model returns one logits tensor"
    return keras.ops.convert_to_numpy(t)


class TestViTHmlpPoolingExcludesCls:
    """``vit_hmlp`` pools the patch tokens only, as ``vit`` and timm do (D-205)."""

    @pytest.mark.parametrize("strategy", ["mean", "max"])
    def test_pooling_excludes_position_zero(self, strategy):
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
            pooled, excluding_cls, rtol=0, atol=1e-6,
            err_msg="vit_hmlp pooling reads the CLS token again; the ViT "
                    "references reduce the patch tokens only (D-205)",
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

        # Pass the full sequence to the head; it will extract the CLS token (position 0)
        seq_tensor = keras.ops.convert_to_tensor(sequence)
        from_cls = _np(model.classification_head(seq_tensor)['logits'])
        np.testing.assert_allclose(
            logits, from_cls, rtol=0, atol=0,
            err_msg="vit_hmlp's head is NOT fed x[:, 0, :]",
        )
        # Anti-vacuity: the rejected slice must give a different answer
        swapped_seq = sequence.copy()
        swapped_seq[:, [0, -1], :] = swapped_seq[:, [-1, 0], :]
        from_swapped = _np(model.classification_head(keras.ops.convert_to_tensor(swapped_seq))['logits'])
        assert not np.allclose(logits, from_swapped, atol=1e-6), (
            "vit_hmlp's head produces the same output for swapped CLS token; "
            "it is not pinned to position 0"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
