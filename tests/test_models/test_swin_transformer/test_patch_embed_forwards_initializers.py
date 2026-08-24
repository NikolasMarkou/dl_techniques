"""RED proof for F-75 -- Swin's patch embedding drops the initializer set.

``SwinTransformer.__init__`` documents ``kernel_initializer`` /
``bias_initializer`` / ``kernel_regularizer`` / ``bias_regularizer`` as the
model-wide "weight initialization strategy", stores all four, serialises all
four, and forwards all four to every stage block and to the classification head.
``_create_patch_embedding``'s ``create_embedding_layer("patch_2d", ...)`` call
forwarded only ``use_bias``, so the stem -- the projection every later
activation scale is inherited from -- trained at ``patch_2d``'s registry default
``glorot_normal`` regardless.

The instrument is the SCOPED weight probe rather than an output diff: the stage
blocks DO honour ``kernel_initializer`` at HEAD, so two arms already produce
different whole-model outputs and an output-level assertion would pass on the
broken tree. See ``tests/test_models/knob_sensitivity_oracle.py``.
"""

import keras
import numpy as np

from dl_techniques.models.swin_transformer.model import SwinTransformer

from ..knob_sensitivity_oracle import (
    assert_scoped_value_knob_changes_weights,
    build_seeded,
)

SWIN_CONFIG = dict(
    num_classes=4,
    embed_dim=16,
    depths=[1, 1, 1, 1],
    num_heads=[2, 2, 2, 2],
    window_size=4,
    patch_size=4,
    input_shape=(32, 32, 3),
    drop_path_rate=0.0,
)

X = np.random.default_rng(2).random((1, 32, 32, 3)).astype("float32")

PATCH_SCOPE = "patch_embed/"


class TestSwinPatchEmbedHonoursInitializers:
    def test_kernel_initializer_reaches_the_patch_projection(self):
        builders = {
            "he_normal": lambda: SwinTransformer(
                kernel_initializer="he_normal", **SWIN_CONFIG
            ),
            "wide_normal": lambda: SwinTransformer(
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.5),
                **SWIN_CONFIG,
            ),
        }
        assert_scoped_value_knob_changes_weights(
            builders, X, knob="kernel_initializer", scope=PATCH_SCOPE
        )

    def test_bias_initializer_reaches_the_patch_projection(self):
        """Separate arm: both defaults are ``"zeros"``, so the kernel probe alone
        would not distinguish a fix that forwarded only the kernel."""
        builders = {
            "zeros": lambda: SwinTransformer(bias_initializer="zeros", **SWIN_CONFIG),
            "ones": lambda: SwinTransformer(bias_initializer="ones", **SWIN_CONFIG),
        }
        assert_scoped_value_knob_changes_weights(
            builders, X, knob="bias_initializer", scope=PATCH_SCOPE
        )

    def test_kernel_regularizer_reaches_the_patch_projection(self):
        """A regularizer changes no weight value, so it is asserted on the layer."""
        model = build_seeded(
            lambda: SwinTransformer(
                kernel_regularizer=keras.regularizers.L2(1.0), **SWIN_CONFIG
            )
        )
        model(X, training=False)
        assert model.patch_embed.kernel_regularizer is not None, (
            "the patch projection has no kernel_regularizer; SwinTransformer's "
            "kernel_regularizer is not reaching create_embedding_layer('patch_2d')"
        )
