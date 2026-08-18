"""RED proof for N-04 -- ``ViT``'s patch embedding drops the initializer set.

``ViT.__init__`` documents ``kernel_initializer`` / ``bias_initializer`` /
``kernel_regularizer`` / ``bias_regularizer`` as applying to "all layers", stores
all four, serialises all four, and forwards all four to every
``TransformerLayer`` and to the classification head. The
``create_embedding_layer('patch_2d', ...)`` call received none of them, so the
model's very first projection -- the one that decides the scale of every
downstream activation -- trained at ``patch_2d``'s registry default
``glorot_normal`` no matter what was asked for. Same defect as F-75 in
``swin_transformer``; found by the step-5 guard's calibration run.

``activation`` is deliberately NOT part of this fix. ``patch_2d`` declares an
``activation`` parameter, and ``ViT`` stores ``self.activation``, but ViT's is
documented (``model.py``, "activation: ... activation function for FFN") and used
as the transformer FFN's activation. Forwarding the default ``"gelu"`` into the
patch projection would make the stem nonlinear, which no ViT is; that is a name
collision, not a dropped knob, and it is recorded as such in
``tests/test_models/test_package_api_contract.py``'s ``_NAME_COLLISIONS``.

The instrument is the SCOPED weight probe: the transformer blocks DO honour
``kernel_initializer`` at HEAD, so two arms already produce different whole-model
outputs and an output-level assertion would pass on the broken tree.
"""

import keras
import numpy as np

from dl_techniques.models.vit.model import ViT

from ..knob_sensitivity_oracle import (
    assert_scoped_value_knob_changes_weights,
    build_seeded,
)

VIT_CONFIG = dict(
    input_shape=(32, 32, 3),
    scale="pico",
    patch_size=8,
    include_top=False,
    pooling="mean",
)

X = np.random.default_rng(1).random((1, 32, 32, 3)).astype("float32")

PATCH_SCOPE = "patch_embed/"


class TestViTPatchEmbedHonoursInitializers:
    def test_kernel_initializer_reaches_the_patch_projection(self):
        builders = {
            "he_normal": lambda: ViT(kernel_initializer="he_normal", **VIT_CONFIG),
            "wide_normal": lambda: ViT(
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.5),
                **VIT_CONFIG,
            ),
        }
        assert_scoped_value_knob_changes_weights(
            builders, X, knob="kernel_initializer", scope=PATCH_SCOPE
        )

    def test_bias_initializer_reaches_the_patch_projection(self):
        """Asserted separately from the kernel because it has its own default.

        ``bias_initializer`` defaults to ``"zeros"`` in both ``ViT`` and the
        ``patch_2d`` registry entry, so a fix that forwarded only the kernel
        would leave this indistinguishable without its own arm.
        """
        builders = {
            "zeros": lambda: ViT(bias_initializer="zeros", **VIT_CONFIG),
            "ones": lambda: ViT(bias_initializer="ones", **VIT_CONFIG),
        }
        assert_scoped_value_knob_changes_weights(
            builders, X, knob="bias_initializer", scope=PATCH_SCOPE
        )

    def test_kernel_regularizer_reaches_the_patch_projection(self):
        """A regularizer changes no weight and no forward output -- only losses.

        So this one is asserted on the built layer's own attribute rather than
        numerically: with an L2 penalty requested, the patch projection must
        contribute a non-zero entry to ``model.losses``.
        """
        model = build_seeded(
            lambda: ViT(kernel_regularizer=keras.regularizers.L2(1.0), **VIT_CONFIG)
        )
        model(X, training=False)
        assert model.patch_embed.kernel_regularizer is not None, (
            "the patch projection has no kernel_regularizer; ViT's "
            "kernel_regularizer is not reaching create_embedding_layer('patch_2d')"
        )
