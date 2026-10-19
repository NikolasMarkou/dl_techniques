"""
The ACC-UNet reference's ``BatchNormalization`` statistics, in one place.

Every batch norm in the reference (kiharalab/ACC-UNet, ``ACC_UNet/ACC_UNet.py``)
is a default ``torch.nn.BatchNorm2d``: running-statistics momentum 0.1 and
epsilon 1e-5. PyTorch's momentum is the weight of the NEW batch, so the Keras
equivalent is ``momentum=0.9``. Keras' own defaults (0.99, 1e-3) are not the
reference.

Used by :class:`~dl_techniques.layers.acc_unet.hanc_layer.HANCLayer`,
:class:`~dl_techniques.layers.acc_unet.hanc_block.HANCBlock`,
:class:`~dl_techniques.layers.acc_unet.multi_level_feature_compilation.MLFCLayer`
and :class:`~dl_techniques.layers.conv_blocks.res_path.ResPath`.
"""

import keras

# ---------------------------------------------------------------------

# Keras momentum = 1 - PyTorch momentum (torch.nn.BatchNorm2d default 0.1).
REFERENCE_BN_MOMENTUM = 0.9

# torch.nn.BatchNorm2d default eps.
REFERENCE_BN_EPSILON = 1e-5

# ---------------------------------------------------------------------


# DECISION plan-2026-10-14T062616-81938e81/D-156
# Every AccUNet batch norm is built here, at the reference's momentum 0.9 and
# epsilon 1e-5. Do NOT go back to bare `keras.layers.BatchNormalization()`:
# with Keras' momentum 0.99 the moving variance kept 0.99**176 = 17% of its
# initial 1.0 after a 16-epoch trainer smoke (176 steps), up to 352x the
# population variance in one layer, and the trained model predicted background
# for EVERY pixel in inference mode (val accuracy 0.894, the majority class,
# while the same weights with population statistics scored 0.997). Momentum
# 0.9 on the same run gives 0.9966.
def reference_batch_norm(name: str) -> keras.layers.BatchNormalization:
    """Build a ``BatchNormalization`` with the ACC-UNet reference's statistics.

    :param name: Layer name.
    :type name: str
    :return: A new, unbuilt layer with ``momentum=REFERENCE_BN_MOMENTUM`` and
        ``epsilon=REFERENCE_BN_EPSILON``; every other argument is the Keras
        default (channels-last axis, centred and scaled).
    :rtype: keras.layers.BatchNormalization

    Failure mode: none; the layer is built by its caller.
    """
    return keras.layers.BatchNormalization(
        momentum=REFERENCE_BN_MOMENTUM,
        epsilon=REFERENCE_BN_EPSILON,
        name=name,
    )

# ---------------------------------------------------------------------
