"""``model.summary()`` runs on a built BEiT classifier, for either pooling mode.

Guard for plan-2026-10-14T062616-81938e81 step 13 (D-099). Measured at 561e02165:
``python -m train.beit.train_classification`` died before its first step at
``model.summary(print_fn=logger.info)`` with ``KeyError: 0`` from Keras 3.8's
``summary_utils.format_layer_shape``. The default classifier
(``use_mean_pooling=True``) built ``_MeanPoolingHead``, whose output was the
one-key dict ``{'logits': ...}``; for a manually built layer Keras maps the
output shape structure and, when it has one entry, indexes it with ``[0]``,
which a dict does not have. The CLS-token path (the shared
``ClassificationHead``, two keys) was unaffected.

The mean-pooling head now returns the shared head's two keys: the linear
logits and their softmax.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.beit.model import create_beit_classifier


IMG = (32, 32, 3)


def _classifier(use_mean_pooling: bool):
    keras.utils.set_random_seed(0)
    model = create_beit_classifier(
        variant="tiny", input_shape=IMG, patch_size=8, num_classes=5,
        use_mean_pooling=use_mean_pooling,
    )
    model.build((None,) + IMG)
    return model


@pytest.mark.parametrize("use_mean_pooling", [True, False])
def test_the_summary_prints(use_mean_pooling):
    lines = []
    _classifier(use_mean_pooling).summary(print_fn=lines.append)
    assert "".join(lines).strip()


def test_the_mean_pooling_head_keeps_the_two_key_contract():
    model = _classifier(use_mean_pooling=True)
    tokens = model.backbone(np.zeros((2,) + IMG, dtype="float32"), training=False)
    out = model.classification_head(tokens, training=False)
    assert set(out) == {"logits", "probabilities"}
    logits = np.asarray(out["logits"])
    np.testing.assert_allclose(
        np.asarray(out["probabilities"]),
        np.asarray(keras.ops.softmax(logits, axis=-1)),
        rtol=0, atol=0,
    )
    # The model still returns the logits, not the probabilities.
    np.testing.assert_array_equal(
        np.asarray(model(np.zeros((2,) + IMG, dtype="float32"), training=False)),
        logits,
    )
